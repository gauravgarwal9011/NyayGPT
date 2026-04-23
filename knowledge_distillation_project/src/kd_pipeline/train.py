"""
train.py
========
CLI entry point for student fine-tuning. Run with:

    python -m kd_pipeline.train

WHY a separate entry point (not folded into ``__main__.py``)?
-------------------------------------------------------------
Generation and training have very different dependency footprints and
runtimes. A user who only wants to *generate* a dataset shouldn't be
forced to install Unsloth + torch + trl just to invoke the package, and
vice-versa. Splitting the entry points keeps each command lean.
"""

# `argparse` for a small zero-dependency CLI.
import argparse
# `sys` so we can return a non-zero exit code on failure.
import sys

from . import config
from .logger import get_logger
from .exceptions import KnowledgeDistillationError
from .student_trainer import train_student

log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """
    Construct the CLI argument parser.

    Kept in its own function so it can be unit-tested without actually
    running training (just call ``_build_parser().parse_args([...])``).
    """
    parser = argparse.ArgumentParser(
        prog="kd_pipeline.train",
        description=(
            "LoRA-fine-tune a student model on the JSONL dataset produced "
            "by `python -m kd_pipeline`. Uses Unsloth + TRL's SFTTrainer."
        ),
    )

    # Override the student model id without editing config.py.
    parser.add_argument(
        "--model",
        type=str,
        default=config.STUDENT_MODEL_NAME,
        help="HF/Unsloth model id of the student (default: %(default)s)",
    )

    # Point at a different train.jsonl (e.g. an augmented version).
    parser.add_argument(
        "--train",
        type=str,
        default=str(config.TRAIN_JSONL),
        help="Path to train.jsonl (default: %(default)s)",
    )

    # Point at a different eval.jsonl.
    parser.add_argument(
        "--eval",
        type=str,
        default=str(config.EVAL_JSONL),
        help="Path to eval.jsonl (default: %(default)s)",
    )

    # Override where the LoRA adapter is written.
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=str(config.ADAPTER_DIR),
        help="Where to save the trained LoRA adapter (default: %(default)s)",
    )

    return parser


def main() -> int:
    """
    Training pipeline entry point.

    Returns
    -------
    int
        Process exit code:
            0 = success
            1 = controlled training error (caught by KnowledgeDistillationError)
            2 = unexpected exception (full traceback in the log)
    """
    args = _build_parser().parse_args()

    # Make sure output / log / adapter directories exist before any
    # stage tries to write to them. Cheap to call repeatedly.
    config.ensure_directories()

    log.info("=== KD TRAINING START ===")
    log.info("Model:       %s", args.model)
    log.info("Train file:  %s", args.train)
    log.info("Eval file:   %s", args.eval)
    log.info("Adapter dir: %s", args.adapter_dir)

    # Wrap the training run so our typed exceptions become a clean
    # `exit 1` and only true bugs become `exit 2` with a traceback.
    try:
        from pathlib import Path
        train_student(
            model_name  = args.model,
            train_path  = Path(args.train),
            eval_path   = Path(args.eval),
            adapter_dir = Path(args.adapter_dir),
        )
    except KnowledgeDistillationError as exc:
        # Expected failure mode — log a one-liner, no traceback noise.
        log.error("Training failed: %s", exc)
        return 1
    except Exception:
        # Unknown / unexpected — log with full traceback for debugging.
        log.exception("Unexpected error during training")
        return 2

    log.info("=== KD TRAINING COMPLETE ===")
    return 0


# Standard guard so importing this module (e.g. for tests) does not
# trigger a training run.
if __name__ == "__main__":
    sys.exit(main())
