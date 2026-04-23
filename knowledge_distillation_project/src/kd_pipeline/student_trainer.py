"""
student_trainer.py
==================
LoRA fine-tunes a student model on the JSONL dataset produced by
``dataset_generator.py``, using Unsloth + TRL's SFTTrainer.

WHY Unsloth?
------------
Unsloth provides patched implementations of LLaMA / Qwen / Mistral
attention + MLP kernels that are 2-5x faster than vanilla HF transformers
at training time, and use ~50% less VRAM thanks to fused kernels and
4-bit base weights. For a small dataset like ours (≈100 samples), this
means a full 3-epoch run completes in minutes on a 32GB Blackwell GPU.

WHY LoRA (not full fine-tune)?
------------------------------
Full fine-tuning a 3B model means storing 3B × 4 bytes (~12 GB) of
optimiser state. LoRA freezes the base weights and trains only ~0.5% of
parameters as low-rank adapters — same downstream quality, a fraction of
the memory, and the result is a tiny ~50 MB adapter you can hot-swap.

WHY a separate trainer module (not bolted onto dataset_generator)?
------------------------------------------------------------------
Generation and training have *different* dependency footprints
(`llama-cpp-python` vs `unsloth`/`torch`/`trl`) and very different
runtimes. Splitting them means a user who only wants the dataset never
has to install Unsloth, and vice-versa.
"""

# `json` to read JSONL train/eval files line-by-line.
import json
# Type hints.
from typing import List, Dict, Optional
# `Path` for OS-agnostic path handling.
from pathlib import Path

from . import config
from .logger import get_logger
from .exceptions import TrainingError, ConfigurationError

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict]:
    """
    Load a JSONL file into a list of dicts.

    Parameters
    ----------
    path : Path
        Absolute path to a `.jsonl` file (one JSON object per line).

    Returns
    -------
    List[Dict]
        Parsed records, in file order.

    Raises
    ------
    ConfigurationError
        If the file does not exist or is empty (the trainer cannot run
        without data, so this is a hard failure rather than a warning).
    """
    if not path.exists():
        raise ConfigurationError(
            f"Training data not found at {path}. "
            f"Run `python -m kd_pipeline` first to generate the dataset."
        )

    # `with open(...)` ensures the file handle is closed even on error.
    # `encoding='utf-8'` is explicit so we don't get surprised by the
    # platform default (which is cp1252 on Windows).
    with open(path, "r", encoding="utf-8") as f:
        # List comp + `.strip()` filter skips any blank lines a user
        # may have introduced when hand-editing the dataset.
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        raise ConfigurationError(f"Dataset {path} is empty")

    log.info("Loaded %d rows from %s", len(rows), path)
    return rows


def _validate_rows(rows: List[Dict], dataset_name: str) -> None:
    """
    Validate the JSONL shape before we hand it to the tokenizer/trainer.

    A little up-front validation produces much clearer errors than
    whatever transformers/TRL would emit later if the dataset is malformed.
    """
    for idx, row in enumerate(rows, start=1):
        if "messages" not in row or not isinstance(row["messages"], list):
            raise ConfigurationError(
                f"{dataset_name} row {idx} is missing a valid `messages` list"
            )

        messages = row["messages"]
        if len(messages) < 2:
            raise ConfigurationError(
                f"{dataset_name} row {idx} must contain at least 2 messages"
            )

        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if not roles or roles[-1] != "assistant":
            raise ConfigurationError(
                f"{dataset_name} row {idx} must end with an assistant message"
            )

        for message_idx, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                raise ConfigurationError(
                    f"{dataset_name} row {idx} message {message_idx} must be an object"
                )
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ConfigurationError(
                    f"{dataset_name} row {idx} message {message_idx} has invalid role "
                    f"{message.get('role')!r}"
                )
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ConfigurationError(
                    f"{dataset_name} row {idx} message {message_idx} must have non-empty text"
                )


def _format_with_chat_template(rows: List[Dict], tokenizer) -> List[Dict]:
    """
    Convert each `{"messages": [...]}` row into a `{"text": "..."}` row
    by running the messages through the model's chat template.

    Why apply the template here instead of inside the trainer?
    ----------------------------------------------------------
    SFTTrainer can format on-the-fly, but doing it once up-front gives
    us:
        1. A single place to inspect/debug the actual training strings.
        2. Faster training (no per-batch tokenizer formatting).
        3. Tokenizer-agnostic call sites — every model gets its own
           native chat format (Qwen's <|im_start|>, Llama's <|begin_of_text|>,
           etc.) without us hard-coding any of them.

    Parameters
    ----------
    rows : List[Dict]
        Dataset rows in OpenAI chat format.
    tokenizer : PreTrainedTokenizerBase
        The student model's tokenizer (carries the chat_template).

    Returns
    -------
    List[Dict]
        Rows shaped as ``[{"text": "<formatted prompt>"}, ...]`` —
        the format SFTTrainer expects when ``dataset_text_field="text"``.
    """
    formatted: List[Dict] = []
    for row in rows:
        # `apply_chat_template` injects the model-specific role markers.
        # `tokenize=False` returns a string (we want SFTTrainer to do
        # the tokenisation in its own batch pipeline).
        # `add_generation_prompt=False` because the assistant turn is
        # *already in* the messages — we want it to be a training target,
        # not a free-running prompt.
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted.append({"text": text})
    return formatted


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_student(
    model_name: Optional[str] = None,
    train_path: Optional[Path] = None,
    eval_path: Optional[Path] = None,
    adapter_dir: Optional[Path] = None,
) -> Path:
    """
    Fine-tune the student model with LoRA on the generated dataset.

    Parameters
    ----------
    model_name : str, optional
        HF / Unsloth model id. Defaults to ``config.STUDENT_MODEL_NAME``.
    train_path, eval_path : Path, optional
        Override the JSONL file locations. Default to config.
    adapter_dir : Path, optional
        Where to write the trained LoRA adapter. Default to config.

    Returns
    -------
    Path
        The directory the adapter was saved into.

    Raises
    ------
    TrainingError
        For any failure inside Unsloth / TRL / torch (wrapped so callers
        don't need to import those packages just to handle errors).
    ConfigurationError
        If the dataset is missing or empty.
    """
    # Resolve defaults from config. Done here (not in defaults) so tests
    # can pass explicit values without triggering env-var lookups.
    model_name  = model_name  or config.STUDENT_MODEL_NAME
    train_path  = train_path  or config.TRAIN_JSONL
    eval_path   = eval_path   or config.EVAL_JSONL
    adapter_dir = adapter_dir or config.ADAPTER_DIR

    log.info("=== STUDENT TRAINING START ===")
    log.info("Model:       %s", model_name)
    log.info("Train file:  %s", train_path)
    log.info("Eval file:   %s", eval_path)
    log.info("Adapter out: %s", adapter_dir)

    # ── Lazy imports ─────────────────────────────────────────────────
    # Heavy ML imports live inside the function so importing this
    # *module* in tests (or for type-checking) doesn't pull in 2 GB of
    # CUDA libraries. The user only pays the import cost when they
    # actually run training.
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
        import torch
    except ImportError as exc:
        log.exception("Required training dependency is missing")
        raise TrainingError(
            "Missing training dependencies. Install with: "
            "pip install unsloth trl datasets"
        ) from exc

    # CUDA sanity check — Unsloth refuses to run without a GPU and the
    # error from deep inside the library is cryptic, so we surface it
    # ourselves with actionable advice.
    if not torch.cuda.is_available():
        raise TrainingError(
            "CUDA is not available. Unsloth requires a GPU. "
            "Verify with `python -c 'import torch; print(torch.cuda.is_available())'`."
        )
    log.info("CUDA OK — device: %s", torch.cuda.get_device_name(0))

    # ── 1. Load the base model + tokenizer ───────────────────────────
    # Unsloth's `from_pretrained` returns the patched model AND a
    # tokenizer correctly configured for SFT (right padding, EOS as
    # pad token if needed, etc.).
    try:
        log.info("[1/5] Loading base model (this can take a minute)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name      = model_name,
            max_seq_length  = config.STUDENT_MAX_SEQ_LEN,
            # `dtype=None` lets Unsloth pick bf16 on Ampere+/Blackwell.
            dtype           = None,
            load_in_4bit    = config.STUDENT_LOAD_IN_4BIT,
        )
    except Exception as exc:
        log.exception("Failed to load base model %s", model_name)
        raise TrainingError(
            f"Failed to load base model {model_name}: {exc}"
        ) from exc

    # ── 2. Attach LoRA adapters ──────────────────────────────────────
    # `get_peft_model` wraps the base model in a PEFT/LoRA layer that
    # only trains a small number of low-rank matrices on top of the
    # frozen base weights.
    try:
        log.info("[2/5] Attaching LoRA adapters (r=%d, alpha=%d)...",
                 config.LORA_R, config.LORA_ALPHA)
        model = FastLanguageModel.get_peft_model(
            model,
            r              = config.LORA_R,
            lora_alpha     = config.LORA_ALPHA,
            lora_dropout   = config.LORA_DROPOUT,
            # `target_modules` = which linear layers to patch with LoRA.
            # All 7 of these are the standard "all linear" set used in
            # the QLoRA paper — gives the best quality on small datasets.
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            # `bias="none"` keeps the bias terms frozen — standard LoRA.
            bias           = "none",
            # Unsloth's gradient checkpointing implementation that
            # halves memory at the cost of ~20% extra forward time.
            use_gradient_checkpointing = "unsloth",
            random_state   = config.RANDOM_SEED,
        )
    except Exception as exc:
        log.exception("Failed to attach LoRA adapters")
        raise TrainingError(f"LoRA attach failed: {exc}") from exc

    # ── 3. Build train + eval HF Datasets ────────────────────────────
    log.info("[3/5] Loading and formatting datasets...")
    train_rows = _load_jsonl(Path(train_path))
    _validate_rows(train_rows, "train dataset")

    eval_rows: List[Dict] = []
    if Path(eval_path).exists():
        eval_rows = _load_jsonl(Path(eval_path))
        _validate_rows(eval_rows, "eval dataset")
    else:
        log.warning("Eval dataset %s not found; training without eval", eval_path)

    # `from_list` is the cheapest way to wrap a list-of-dicts as a
    # `datasets.Dataset` (no Arrow file on disk, no conversion).
    train_ds = Dataset.from_list(_format_with_chat_template(train_rows, tokenizer))
    eval_ds = None
    if eval_rows:
        eval_ds = Dataset.from_list(_format_with_chat_template(eval_rows, tokenizer))

    log.info(
        "Train dataset: %d rows | Eval dataset: %d rows",
        len(train_ds),
        len(eval_ds) if eval_ds is not None else 0,
    )

    # ── 4. Configure + run SFTTrainer ────────────────────────────────
    # SFTConfig is TRL's wrapper around HF TrainingArguments, with extra
    # SFT-specific knobs (packing, dataset_text_field, etc.).
    log.info("[4/5] Starting training: %d epochs, lr=%g, batch=%d×%d",
             config.NUM_TRAIN_EPOCHS, config.LEARNING_RATE,
             config.TRAIN_BATCH_SIZE, config.GRAD_ACCUM_STEPS)

    sft_config = SFTConfig(
        output_dir                  = str(adapter_dir / "checkpoints"),
        num_train_epochs            = config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size = config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size  = config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = config.GRAD_ACCUM_STEPS,
        learning_rate               = config.LEARNING_RATE,
        warmup_ratio                = config.WARMUP_RATIO,
        weight_decay                = config.WEIGHT_DECAY,
        # Linear scheduler is the QLoRA-paper default and works well
        # for short SFT runs. Cosine is also fine but we keep it linear
        # for fewer surprises.
        lr_scheduler_type           = "linear",
        # `bf16` if supported (Ampere+, Blackwell), else `fp16`.
        bf16                        = is_bfloat16_supported(),
        fp16                        = not is_bfloat16_supported(),
        # Run an eval pass at the end of every epoch.
        eval_strategy               = "epoch" if eval_ds is not None else "no",
        # Save a checkpoint at the end of every epoch (so you can pick
        # the best one if eval loss starts climbing).
        save_strategy               = "epoch",
        # Keep only the 2 most recent checkpoints — disk space saver.
        save_total_limit            = 2,
        logging_steps               = 5,
        # Reproducibility: same seed → same shuffle → same loss curve.
        seed                        = config.RANDOM_SEED,
        # Tell SFTTrainer which column carries the formatted prompt.
        dataset_text_field          = "text",
        max_length                  = config.STUDENT_MAX_SEQ_LEN,
        # `packing=False` keeps each sample as one training example.
        # Packing concatenates samples to fill seq_len — faster but
        # mixes unrelated samples, which hurts on small SFT datasets.
        packing                     = False,
        # Disable WandB / TensorBoard reporting unless the user wants it.
        report_to                   = "none",
    )

    try:
        trainer = SFTTrainer(
            model           = model,
            tokenizer       = tokenizer,
            train_dataset   = train_ds,
            eval_dataset    = eval_ds,
            args            = sft_config,
        )
    except Exception as exc:
        log.exception("Failed to construct SFTTrainer")
        raise TrainingError(f"SFTTrainer construction failed: {exc}") from exc

    # `train()` is the actual GPU-burning step. Wrap it so any CUDA OOM
    # / kernel error becomes a typed exception with the original cause.
    try:
        train_result = trainer.train()
    except Exception as exc:
        log.exception("trainer.train() failed")
        raise TrainingError(f"Training run failed: {exc}") from exc

    # Log the headline metrics so the user has something useful in the
    # console even if they don't open the checkpoint dir.
    metrics = train_result.metrics
    log.info(
        "Training complete: train_loss=%.4f, runtime=%.1fs",
        metrics.get("train_loss", float("nan")),
        metrics.get("train_runtime", float("nan")),
    )

    # ── 5. Save the LoRA adapter ─────────────────────────────────────
    # `save_pretrained` writes the adapter weights + adapter_config.json.
    # We deliberately save the *adapter only* (not merged into the base)
    # — keeps the artifact small (~50 MB) and lets you hot-swap adapters
    # against the same base at inference time.
    log.info("[5/5] Saving LoRA adapter to %s ...", adapter_dir)
    try:
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
    except Exception as exc:
        log.exception("Failed to save LoRA adapter")
        raise TrainingError(f"Failed to save adapter: {exc}") from exc

    # Run a final eval pass and log the eval loss — gives you a single
    # number to compare runs against.
    if eval_ds is not None:
        try:
            eval_metrics = trainer.evaluate()
            log.info("Final eval_loss=%.4f", eval_metrics.get("eval_loss", float("nan")))
        except Exception as exc:
            # Eval is non-fatal — we already saved the adapter. Just warn.
            log.warning("Final eval pass failed (non-fatal): %s", exc)
    else:
        log.info("Skipped final eval because no eval dataset was available")

    log.info("=== STUDENT TRAINING COMPLETE ===")
    return adapter_dir
