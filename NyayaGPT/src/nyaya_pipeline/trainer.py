"""
trainer.py — QLoRA fine-tuning for NyayaGPT with MLflow experiment tracking.

Reuses the student_trainer.py pattern from the KD project.
Key addition: MLflow logging of params, per-epoch metrics, and adapter artifact.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

from . import config
from .exceptions import TrainingError, ConfigurationError
from .logger import get_logger

log = get_logger(__name__)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise ConfigurationError(f"Dataset not found: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ConfigurationError(f"{path}:{i} — invalid JSON: {exc}") from exc
    return rows


def _validate_rows(rows: List[Dict], label: str) -> None:
    for i, row in enumerate(rows):
        if "messages" not in row:
            raise ConfigurationError(f"{label}[{i}]: missing 'messages' key")
        msgs = row["messages"]
        if not isinstance(msgs, list) or len(msgs) < 2:
            raise ConfigurationError(f"{label}[{i}]: 'messages' must have ≥2 entries")
        for m in msgs:
            if m.get("role") not in {"system", "user", "assistant"}:
                raise ConfigurationError(f"{label}[{i}]: invalid role '{m.get('role')}'")
            if not m.get("content", "").strip():
                raise ConfigurationError(f"{label}[{i}]: empty content for role '{m['role']}'")


def _format_with_chat_template(rows: List[Dict], tokenizer) -> List[Dict]:
    """Apply tokenizer's chat template once per sample."""
    formatted = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted.append({"text": text})
    return formatted


# ── Main training function ────────────────────────────────────────────────────

def train(
    model_name: Optional[str] = None,
    train_path: Optional[Path] = None,
    eval_path:  Optional[Path] = None,
    adapter_dir: Optional[Path] = None,
    mlflow_run_name: str = "mistral-7b-qlora-v1",
) -> Path:
    """
    Fine-tune Mistral 7B with QLoRA and log to MLflow.

    Returns:
        Path to the saved adapter directory.
    """
    model_name  = model_name  or config.STUDENT_MODEL_NAME
    train_path  = train_path  or config.TRAIN_JSONL
    eval_path   = eval_path   or config.EVAL_JSONL
    adapter_dir = adapter_dir or config.ADAPTER_DIR

    # ── Lazy imports ──────────────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
        import torch
        import mlflow
    except ImportError as exc:
        raise TrainingError(
            f"Missing training dependencies: {exc}\n"
            "pip install unsloth trl datasets mlflow"
        ) from exc

    if not torch.cuda.is_available():
        raise TrainingError("CUDA not available. GPU required for training.")

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading datasets …")
    train_rows = _load_jsonl(train_path)
    eval_rows  = _load_jsonl(eval_path)
    _validate_rows(train_rows, "train")
    _validate_rows(eval_rows,  "eval")
    log.info("  train: %d samples | eval: %d samples", len(train_rows), len(eval_rows))

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading base model: %s", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.STUDENT_MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=config.STUDENT_LOAD_IN_4BIT,
    )

    # ── Attach LoRA ───────────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.RANDOM_SEED,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable params: %s (%.2f%% of total)",
             f"{n_params:,}",
             100 * n_params / sum(p.numel() for p in model.parameters()))

    # ── Format datasets ───────────────────────────────────────────────────────
    train_ds = Dataset.from_list(_format_with_chat_template(train_rows, tokenizer))
    eval_ds  = Dataset.from_list(_format_with_chat_template(eval_rows,  tokenizer))

    # ── Training config ───────────────────────────────────────────────────────
    use_bf16 = is_bfloat16_supported()
    adapter_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir                  = str(adapter_dir / "checkpoints"),
        num_train_epochs            = config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size = config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size  = config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = config.GRAD_ACCUM_STEPS,
        learning_rate               = config.LEARNING_RATE,
        warmup_ratio                = config.WARMUP_RATIO,
        weight_decay                = config.WEIGHT_DECAY,
        lr_scheduler_type           = "linear",
        bf16                        = use_bf16,
        fp16                        = not use_bf16,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        save_total_limit            = 2,
        logging_steps               = 10,
        seed                        = config.RANDOM_SEED,
        dataset_text_field          = "text",
        max_length                  = config.STUDENT_MAX_SEQ_LEN,
        packing                     = False,
        report_to                   = "none",  # MLflow logged manually below
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        args          = sft_config,
    )

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_TRAIN)

    train_params = {
        "base_model":         model_name,
        "lora_r":             config.LORA_R,
        "lora_alpha":         config.LORA_ALPHA,
        "num_epochs":         config.NUM_TRAIN_EPOCHS,
        "learning_rate":      config.LEARNING_RATE,
        "batch_size":         config.TRAIN_BATCH_SIZE,
        "grad_accum_steps":   config.GRAD_ACCUM_STEPS,
        "max_seq_len":        config.STUDENT_MAX_SEQ_LEN,
        "train_samples":      len(train_rows),
        "eval_samples":       len(eval_rows),
        "precision":          "bf16" if use_bf16 else "fp16",
        "trainable_params":   n_params,
    }

    with mlflow.start_run(run_name=mlflow_run_name) as run:
        mlflow.log_params(train_params)
        log.info("MLflow run started: %s", run.info.run_id)

        t0 = time.time()
        train_result = trainer.train()
        elapsed = time.time() - t0

        # Log training metrics
        mlflow.log_metrics({
            "train_loss":         train_result.training_loss,
            "train_samples":      len(train_rows),
            "training_time_secs": elapsed,
            "tokens_per_second":  (len(train_rows) * config.STUDENT_MAX_SEQ_LEN) / elapsed,
        })

        # Log eval loss
        eval_result = trainer.evaluate()
        mlflow.log_metrics({
            "eval_loss":       eval_result.get("eval_loss", -1),
            "eval_perplexity": 2 ** eval_result.get("eval_loss", 0),
        })

        # Save adapter
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        mlflow.log_artifact(str(adapter_dir), artifact_path="adapter")

        log.info("Training complete in %.1f s | train_loss=%.4f | eval_loss=%.4f",
                 elapsed,
                 train_result.training_loss,
                 eval_result.get("eval_loss", -1))

    return adapter_dir
