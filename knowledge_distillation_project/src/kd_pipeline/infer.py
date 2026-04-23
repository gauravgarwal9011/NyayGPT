"""
infer.py
========
CLI entry point for running local inference with the distilled student
model + LoRA adapter.

Run with:

    python -m kd_pipeline.infer --prompt "What services does Ignatiuz offer?"

Or start an interactive shell:

    python -m kd_pipeline.infer --interactive
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional

from . import config
from .logger import get_logger
from .exceptions import KnowledgeDistillationError, TrainingError, ConfigurationError
from .response_cleaner import sanitize_assistant_response

log = get_logger(__name__)


def _build_messages(question: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Build a chat-style prompt for the student model.

    The student was trained on `system -> user -> assistant` samples,
    so we mirror that shape at inference time for best consistency.
    """
    return [
        {"role": "system", "content": system_prompt or config.SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def load_student_for_inference(
    model_name: Optional[str] = None,
    adapter_dir: Optional[Path] = None,
):
    """
    Load the base student model and apply the saved LoRA adapter.
    """
    model_name = model_name or config.STUDENT_MODEL_NAME
    adapter_dir = adapter_dir or config.ADAPTER_DIR

    if not Path(adapter_dir).exists():
        raise ConfigurationError(
            f"Adapter directory not found at {adapter_dir}. "
            "Train the student first with `python3 -m kd_pipeline.train`."
        )

    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError as exc:
        raise TrainingError(
            "Missing inference dependencies. Install the same stack used for training: "
            "pip install unsloth trl datasets"
        ) from exc

    if not torch.cuda.is_available():
        raise TrainingError(
            "CUDA is not available. This inference path expects a GPU-backed setup."
        )

    log.info("Loading student base model for inference: %s", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.STUDENT_MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=config.STUDENT_LOAD_IN_4BIT,
    )

    log.info("Loading LoRA adapter from %s", adapter_dir)
    model.load_adapter(str(adapter_dir))
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response(
    question: str,
    model_name: Optional[str] = None,
    adapter_dir: Optional[Path] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Generate one response from the distilled student model.
    """
    model, tokenizer = load_student_for_inference(
        model_name=model_name,
        adapter_dir=adapter_dir,
    )

    messages = _build_messages(question, system_prompt=system_prompt)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    outputs = model.generate(**inputs, **generation_kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    completion_ids = outputs[0][prompt_length:]
    response = sanitize_assistant_response(
        tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    )
    return response


def _run_interactive(
    model_name: str,
    adapter_dir: Path,
    max_new_tokens: int,
    temperature: float,
) -> int:
    """
    Start a simple REPL for chatting with the student adapter.
    """
    model, tokenizer = load_student_for_inference(
        model_name=model_name,
        adapter_dir=adapter_dir,
    )

    print("Student model ready. Type a question, or `exit` to quit.")
    while True:
        try:
            question = input("\nYou> ").strip()
        except EOFError:
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        messages = _build_messages(question)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        outputs = model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = outputs[0][prompt_length:]
        response = sanitize_assistant_response(
            tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        )
        print(f"Student> {response}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kd_pipeline.infer",
        description="Run local inference with the distilled student model and LoRA adapter.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.STUDENT_MODEL_NAME,
        help="HF/Unsloth base model id (default: %(default)s)",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=str(config.ADAPTER_DIR),
        help="Path to the trained LoRA adapter directory (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to run. If omitted, use --interactive.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive chat loop with the student model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature. Set 0 for greedy decoding (default: %(default)s)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if not args.prompt and not args.interactive:
        raise SystemExit("Provide --prompt for one-shot inference or --interactive for a chat shell.")

    try:
        if args.interactive:
            return _run_interactive(
                model_name=args.model,
                adapter_dir=Path(args.adapter_dir),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

        response = generate_response(
            question=args.prompt,
            model_name=args.model,
            adapter_dir=Path(args.adapter_dir),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(response)
        return 0

    except KnowledgeDistillationError as exc:
        log.error("Inference failed: %s", exc)
        return 1
    except Exception:
        log.exception("Unexpected error during student inference")
        return 2


if __name__ == "__main__":
    sys.exit(main())
