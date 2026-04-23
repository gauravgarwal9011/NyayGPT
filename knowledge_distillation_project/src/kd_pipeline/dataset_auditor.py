"""
dataset_auditor.py
==================
Interactive audit tool for inspecting generated training samples.

WHY this matters
----------------
Generating a dataset is fast; *checking* it is the step most people skip —
and it's the step that decides whether your fine-tune succeeds or wastes
GPU hours. A 10-minute manual audit catches:
    • Hallucinated numbers that the keyword filter missed
    • CoT traces that "reason" without citing the doc
    • Topic extraction misfires producing nonsense questions
    • Systematic teacher biases (always hedging, always listing 3 items, …)

The auditor is intentionally interactive (one sample at a time, press Enter
to advance) because skim-reading 100 samples in a wall of text is useless —
you need to *focus* on each one.
"""

# `json` to deserialise each JSONL line back to a dict.
import json
# `random` to pick a fair sample from the dataset.
import random
# `Path` for OS-agnostic path handling.
from pathlib import Path
# Type hints.
from typing import Union

from .logger import get_logger
from .exceptions import DatasetSaveError

log = get_logger(__name__)


def audit_dataset(
    dataset_path: Union[str, Path],
    n_samples: int = 10,
    interactive: bool = True,
) -> None:
    """
    Print `n_samples` random samples from a JSONL dataset for human review.

    Parameters
    ----------
    dataset_path : str | Path
        Path to a JSONL dataset (e.g., ``output/train.jsonl``).
    n_samples : int
        How many samples to display. Capped to dataset size.
    interactive : bool
        If True, pause between samples and wait for Enter. If False, print
        all samples back-to-back (useful for piping into a file or for tests).

    Raises
    ------
    DatasetSaveError
        If the file cannot be opened or parsed (re-uses the dataset error
        type since auditing is part of the dataset lifecycle).

    Notes
    -----
    Use this BEFORE fine-tuning. The cost of one bad audit is ten minutes;
    the cost of training on bad data is hours of GPU time + a confused
    student model that you then have to debug.
    """
    # Normalise to Path so the rest of the function can use Path methods.
    dataset_path = Path(dataset_path)

    log.info("Auditing dataset: %s (n_samples=%d)", dataset_path, n_samples)

    # Wrap the file read so missing/malformed files raise our typed error
    # instead of a raw OSError or JSONDecodeError surfacing to the caller.
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            # List comprehension parses every line into a dict in one pass.
            # JSONL = one JSON object per line, so we split on newlines.
            samples = [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        log.exception("Failed to load dataset for audit")
        raise DatasetSaveError(
            f"Failed to load dataset {dataset_path}: {exc}"
        ) from exc

    if not samples:
        # An empty file is technically valid JSONL but useless to audit —
        # warn loudly so the user notices their pipeline produced nothing.
        log.warning("Dataset %s is empty — nothing to audit", dataset_path)
        return

    # `random.sample` picks WITHOUT replacement, so the same sample never
    # shows up twice in one audit run. `min(...)` clamps the count so we
    # don't ask for more samples than exist.
    sampled = random.sample(samples, min(n_samples, len(samples)))

    # Header — pure cosmetic banner so the user can easily find the start
    # of the audit in a long terminal scrollback.
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"DATASET AUDIT — {len(sampled)} random samples from {dataset_path}")
    print(f"{bar}")

    # `enumerate(..., start=1)` numbers samples from 1 instead of 0 for
    # human-friendly output ("Sample 1", not "Sample 0").
    for i, sample in enumerate(sampled, start=1):
        print(f"\n--- Sample {i}/{len(sampled)} ---")

        # Each sample has a `messages` list in OpenAI chat format.
        # We walk it role-by-role and pretty-print with truncation rules
        # tailored to each role.
        for msg in sample["messages"]:
            role = msg["role"].upper()
            content = msg["content"]

            if role == "SYSTEM":
                # The system prompt is always (almost) the same — show
                # only the first 120 chars so it doesn't dominate the screen.
                print(f"[SYSTEM]:    {content[:120]}...")
            elif role == "USER":
                # User questions are short — print them in full so we can
                # judge whether the topic substitution worked.
                print(f"[USER]:      {content}")
            elif role == "ASSISTANT":
                # Show the first 500 chars of the response. For CoT this
                # is enough to see the beginning of the reasoning trace
                # (where most failures show up — bad CoT usually fails
                # in the first sentence).
                print(f"[ASSISTANT]: {content[:500]}...")
            else:
                # Defensive: shouldn't happen, but if it does we still
                # want to *see* the unexpected role rather than skip it.
                print(f"[{role}]: {content[:200]}...")

        print()  # blank line between sample body and the prompt

        # Interactive pause. Wrapped in try/except so Ctrl+C ends the audit
        # cleanly without a traceback (KeyboardInterrupt is normal here).
        if interactive and i < len(sampled):
            try:
                input("Press Enter for next sample (Ctrl+C to stop)...")
            except (KeyboardInterrupt, EOFError):
                # EOFError handles the case where stdin is closed (e.g.
                # running under nohup) — we just stop the audit gracefully.
                print("\nAudit interrupted by user.")
                log.info("Audit interrupted at sample %d/%d", i, len(sampled))
                return

    log.info("Audit complete: %d samples reviewed", len(sampled))
