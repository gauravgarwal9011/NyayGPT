"""
__main__.py
===========
Package entry point. Lets you run the whole pipeline with:

    python -m kd_pipeline

WHY ``__main__.py`` and not a top-level ``main.py``?
----------------------------------------------------
Putting the entry inside the package itself means:
    1. The package is self-contained — no extra files outside `src/`.
    2. ``python -m kd_pipeline`` works from anywhere on PYTHONPATH,
       so the user doesn't have to ``cd`` into the project first.
    3. All relative imports (``from .dataset_generator import …``)
       resolve correctly because the package is being run as a module.

This file is intentionally tiny: it does NOT contain pipeline logic.
Its only job is to wire stages together and exit with the right code.
"""

# `argparse` provides a small, dependency-free CLI parser. We use it
# rather than reading sys.argv manually so users get --help for free.
import argparse
# `sys` so we can return a non-zero exit code on failure (CI/CD friendly).
import sys

from . import config
from .logger import get_logger
from .exceptions import KnowledgeDistillationError

log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """
    Construct the CLI argument parser.

    Kept in its own function so it can be unit-tested without running
    the pipeline (just call ``_build_parser().parse_args([...])``).
    """
    parser = argparse.ArgumentParser(
        prog="kd_pipeline",
        description=(
            "Knowledge-distillation dataset generator. Reads a PDF + a "
            "verified knowledge base, queries a teacher GGUF model, "
            "filters bad samples, and writes train.jsonl + eval.jsonl."
        ),
    )

    # `--pdf` lets the user point at any PDF without editing config.py.
    # Defaults to whatever config resolved (usually env-var or fallback).
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(config.PDF_PATH),
        help="Path to the source PDF (default: %(default)s)",
    )

    # `--audit` toggles the post-generation interactive review step.
    # `action="store_true"` means the flag itself is the value; no = needed.
    parser.add_argument(
        "--audit",
        action="store_true",
        help="After generation, interactively review N random samples.",
    )

    # How many samples to show in the audit.
    parser.add_argument(
        "--audit-samples",
        type=int,
        default=10,
        help="Samples to show in --audit mode (default: %(default)s)",
    )

    # `--no-generate` lets you skip generation and go straight to audit
    # — useful for re-reviewing a dataset you already produced.
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip dataset generation; only run the auditor.",
    )

    parser.add_argument(
        "--train-student",
        action="store_true",
        help="After dataset generation (or with --no-generate), train the student model.",
    )

    parser.add_argument(
        "--student-model",
        type=str,
        default=config.STUDENT_MODEL_NAME,
        help="HF/Unsloth model id for student training (default: %(default)s)",
    )

    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=str(config.ADAPTER_DIR),
        help="Where to save the trained student adapter (default: %(default)s)",
    )

    return parser


def main() -> int:
    """
    Pipeline entry point.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 on a controlled pipeline error,
        2 on an unexpected exception. The two-tier code lets shell scripts
        distinguish "expected failure" (e.g. missing PDF) from "bug".
    """
    args = _build_parser().parse_args()

    # Make sure output/log directories exist before any stage tries to
    # write to them. Cheap to call repeatedly — it's idempotent.
    config.ensure_directories()

    log.info("=== KD PIPELINE START ===")
    log.info("PDF: %s", args.pdf)
    log.info("Output dir: %s", config.OUTPUT_DIR)

    # Wrap the whole pipeline so any of our typed exceptions becomes a
    # graceful "exit 1" with a clean log line, while truly unexpected
    # exceptions become "exit 2" with a full traceback for debugging.
    try:
        if not args.no_generate:
            from .dataset_generator import generate_dataset
            generate_dataset(args.pdf)

        if args.audit:
            from .dataset_auditor import audit_dataset
            # Always audit the train split — the eval split is sacred and
            # shouldn't bias your impression of training data quality.
            train_path = config.OUTPUT_DIR / "train.jsonl"
            audit_dataset(train_path, n_samples=args.audit_samples)

        if args.train_student:
            from pathlib import Path
            from .student_trainer import train_student

            train_student(
                model_name=args.student_model,
                train_path=config.TRAIN_JSONL,
                eval_path=config.EVAL_JSONL,
                adapter_dir=Path(args.adapter_dir),
            )

    except KnowledgeDistillationError as exc:
        # Our own exception types — these are *expected* failure modes
        # (missing files, bad config, etc.). Log a one-line summary, no
        # traceback noise.
        log.error("Pipeline failed: %s", exc)
        return 1
    except Exception:
        # Anything else is a bug. Log with full traceback so we can fix it.
        log.exception("Unexpected error during pipeline run")
        return 2

    log.info("=== KD PIPELINE COMPLETE ===")
    return 0


# `if __name__ == "__main__"` guards the call so importing this module
# (e.g. for testing `main()`) does NOT trigger a pipeline run.
if __name__ == "__main__":
    sys.exit(main())
