"""
clean_dataset.py
================
CLI utility to sanitize existing train/eval JSONL files by removing
teacher reasoning traces and channel markers from assistant messages.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict

from . import config
from .logger import get_logger
from .exceptions import KnowledgeDistillationError, DatasetSaveError
from .response_cleaner import clean_dataset_rows

log = get_logger(__name__)


def _load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_jsonl(rows: List[Dict], path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise DatasetSaveError(f"Failed to write {path}: {exc}") from exc


def clean_jsonl_file(path: Path, create_backup: bool = True) -> int:
    rows = _load_jsonl(path)
    cleaned_rows = clean_dataset_rows(rows)

    changed = 0
    retained_rows: List[Dict] = []
    dropped_rows = 0
    for old_row, new_row in zip(rows, cleaned_rows):
        old_messages = old_row.get("messages", [])
        new_messages = new_row.get("messages", [])

        assistant_text = ""
        for old_message, new_message in zip(old_messages, new_messages):
            if old_message.get("role") == "assistant" and old_message.get("content") != new_message.get("content"):
                changed += 1
            if new_message.get("role") == "assistant":
                assistant_text = new_message.get("content", "")

        if not isinstance(assistant_text, str) or not assistant_text.strip():
            dropped_rows += 1
            continue

        retained_rows.append(new_row)

    if create_backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            log.info("Backup written to %s", backup_path)
        else:
            log.info("Backup already exists at %s", backup_path)

    _save_jsonl(retained_rows, path)
    log.info(
        "Cleaned %d assistant messages in %s and dropped %d empty rows",
        changed,
        path,
        dropped_rows,
    )
    return changed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kd_pipeline.clean_dataset",
        description="Clean existing train/eval JSONL files before retraining.",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=str(config.TRAIN_JSONL),
        help="Path to train.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=str(config.EVAL_JSONL),
        help="Path to eval.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak copies before rewriting the files.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        train_changed = clean_jsonl_file(Path(args.train), create_backup=not args.no_backup)
        eval_changed = clean_jsonl_file(Path(args.eval), create_backup=not args.no_backup)
        log.info("Dataset cleanup complete: train=%d changed, eval=%d changed", train_changed, eval_changed)
        return 0
    except KnowledgeDistillationError as exc:
        log.error("Dataset cleanup failed: %s", exc)
        return 1
    except Exception:
        log.exception("Unexpected error during dataset cleanup")
        return 2


if __name__ == "__main__":
    sys.exit(main())
