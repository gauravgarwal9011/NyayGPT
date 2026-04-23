"""
dataset_generator.py
====================
Top-level orchestrator: PDF + knowledge base → grounded chunks → teacher
Q&A → quality filter → train/eval JSONL on disk.

WHY this is one file (not split further)
----------------------------------------
The pipeline is fundamentally a sequence of stages where each stage's
output feeds the next. Splitting the orchestration logic into more files
would just make the data flow harder to follow. Each helper function
*it calls* lives in its own focused module, so this file remains short
and focused on coordination.
"""

# `json` for serialising chunks and dataset to disk.
import json
# `random` for sampling templates and shuffling the final dataset.
import random
# Type hints.
from typing import List, Dict
# `Path` for OS-agnostic path joins.
from pathlib import Path

# Internal imports — every stage of the pipeline.
from . import config
from .logger import get_logger
from .exceptions import DatasetSaveError
from .knowledge_base import DOCUMENT_KNOWLEDGE_BASE
from .pdf_extractor import extract_pages_with_fitz, extract_tables_with_pdfplumber
from .chunker import chunk_pages, chunk_knowledge_base
from .teacher_model import TeacherModel
from .prompt_templates import (
    DIRECT_QUESTION_TEMPLATES,
    COT_SCENARIO_TEMPLATES,
    extract_topic_from_chunk,
    build_generation_prompt,
)
from .quality_filter import is_quality_response
from .response_cleaner import sanitize_assistant_response

log = get_logger(__name__)


def _save_json(obj, path: Path) -> None:
    """
    Helper: write `obj` to `path` as pretty-printed JSON.

    A small private helper (single-underscore prefix = "internal") so
    every JSON-write goes through one error-handling code path.
    """
    try:
        # `with open(...)` is the idiomatic way to open files in Python:
        # the file is closed automatically even if an exception fires.
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        log.exception("Failed to write %s", path)
        raise DatasetSaveError(f"Failed to write {path}: {exc}") from exc


def _save_jsonl(rows: List[Dict], path: Path) -> None:
    """
    Helper: write `rows` to `path` as JSON Lines (one object per line).

    JSONL is the standard format for LLM training data because it can be
    streamed line-by-line without loading the whole file into memory.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                # `json.dumps` (with an "s") returns a string instead of
                # writing to a file — we then add the newline ourselves.
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.exception("Failed to write %s", path)
        raise DatasetSaveError(f"Failed to write {path}: {exc}") from exc


def _build_all_chunks(pdf_path: str) -> List[Dict]:
    """
    Build the combined chunk list from BOTH the verified knowledge base
    (primary) and the fitz PDF extraction (secondary).

    The fitz step is wrapped in try/except so a missing or unreadable PDF
    falls back to the knowledge base alone — the dataset is still valid,
    just smaller.
    """
    log.info("[1/5] Building verified chunks from DOCUMENT_KNOWLEDGE_BASE...")
    kb_chunks = chunk_knowledge_base(DOCUMENT_KNOWLEDGE_BASE)

    log.info("[2/5] Extracting supplementary content from PDF via fitz...")
    fitz_chunks: List[Dict] = []
    try:
        # Extract page text and detected tables.
        pages  = extract_pages_with_fitz(pdf_path)
        tables = extract_tables_with_pdfplumber(pdf_path)

        # Merge tables onto their owning pages so the chunker sees both
        # narrative and tabular content from the same page.
        # `setdefault(k, [])` returns the existing list at key k or
        # creates and returns a fresh list if k is missing — concise way
        # to build a dict-of-lists in one pass.
        page_table_map: Dict[int, List[str]] = {}
        for t in tables:
            page_table_map.setdefault(t["page_num"], []).append(t["table_text"])
        for page in pages:
            if page["page_num"] in page_table_map:
                page["text"] += (
                    "\n\n[TABLE DATA]\n"
                    + "\n\n".join(page_table_map[page["page_num"]])
                )

        # Persist the raw pages for human inspection / debugging.
        _save_json(pages, config.OUTPUT_DIR / "extracted_pages.json")
        log.info("Saved extracted_pages.json (%d pages)", len(pages))

        # Now chunk the merged text.
        fitz_chunks = chunk_pages(pages)

        # Tag each fitz chunk so audits can distinguish them from KB chunks.
        for c in fitz_chunks:
            c["source"]  = "fitz_extraction"
            c["section"] = "pdf_page"

        _save_json(fitz_chunks, config.OUTPUT_DIR / "fitz_chunks.json")
        log.info("Saved fitz_chunks.json (%d fitz chunks)", len(fitz_chunks))

    except Exception as exc:
        # We *don't* re-raise — fitz failure is recoverable. We log a
        # warning so the user notices, then carry on with KB-only.
        log.warning(
            "fitz extraction failed (%s) — proceeding with knowledge base only",
            exc,
        )

    # KB chunks first → highest-quality samples land first in the dataset.
    all_chunks = kb_chunks + fitz_chunks
    _save_json(all_chunks, config.OUTPUT_DIR / "all_chunks.json")
    log.info(
        "Total chunks: %d (%d verified + %d fitz)",
        len(all_chunks), len(kb_chunks), len(fitz_chunks),
    )
    return all_chunks


def _generate_samples(
    teacher: TeacherModel,
    all_chunks: List[Dict],
) -> List[Dict]:
    """
    Iterate every chunk and generate direct + CoT Q&A pairs from each.

    Returns a list of sample dicts with the form:
        {"messages": [...], "_meta": {chunk_id, type, topic}}
    """
    log.info("[4/5] Generating dataset from chunks...")
    dataset:  List[Dict] = []
    rejected: int = 0

    # `enumerate(..., start=1)` numbers chunks from 1 in the logs.
    for chunk_idx, chunk in enumerate(all_chunks, start=1):
        source_label = chunk.get("source", "unknown")
        log.info(
            "Chunk %d/%d [%s] (%d chars)",
            chunk_idx, len(all_chunks), source_label, len(chunk["text"]),
        )

        topic = extract_topic_from_chunk(chunk["text"])

        # ── Direct Q&A samples ────────────────────────────────────
        # Pick QA_PER_CHUNK templates without replacement so we never
        # ask the same question twice for one chunk.
        selected_templates = random.sample(
            DIRECT_QUESTION_TEMPLATES,
            min(config.QA_PER_CHUNK, len(DIRECT_QUESTION_TEMPLATES)),
        )

        for template in selected_templates:
            question = template.format(topic=topic)
            messages = build_generation_prompt(chunk["text"], question, use_cot=False)
            response = sanitize_assistant_response(
                teacher.generate(messages, use_cot=False)
            )

            is_good, reason = is_quality_response(response, chunk["text"])
            if is_good:
                dataset.append({
                    "messages": [
                        # The student's training sample uses the BARE
                        # system prompt (no chunk injected) — we want the
                        # student to answer from memory, not RAG context.
                        {"role": "system",    "content": config.SYSTEM_PROMPT},
                        {"role": "user",      "content": question},
                        {"role": "assistant", "content": response},
                    ],
                    "_meta": {
                        "chunk_id": chunk["chunk_id"],
                        "type":     "direct",
                        "topic":    topic,
                    },
                })
                log.info("  ✓ Direct: %s", question[:60])
            else:
                rejected += 1
                log.info("  ✗ Direct rejected (%s): %s", reason, question[:60])

        # ── CoT samples (every other chunk → ~30% of dataset) ────
        # Generating CoT for *every* chunk would over-represent CoT and
        # take ~3-4x longer. Alternating naturally yields a ~30% ratio.
        if chunk_idx % 2 == 0:
            cot_template = random.choice(COT_SCENARIO_TEMPLATES)
            cot_question = cot_template.format(topic=topic)
            messages_cot = build_generation_prompt(chunk["text"], cot_question, use_cot=True)
            response_cot = sanitize_assistant_response(
                teacher.generate(messages_cot, use_cot=True)
            )

            is_good, reason = is_quality_response(response_cot, chunk["text"])
            if is_good:
                dataset.append({
                    "messages": [
                        {"role": "system",    "content": config.SYSTEM_PROMPT + "\nReasoning: high"},
                        {"role": "user",      "content": cot_question},
                        {"role": "assistant", "content": response_cot},
                    ],
                    "_meta": {
                        "chunk_id": chunk["chunk_id"],
                        "type":     "cot",
                        "topic":    topic,
                    },
                })
                log.info("  ✓ CoT: %s", cot_question[:60])
            else:
                rejected += 1
                log.info("  ✗ CoT rejected (%s)", reason)

    log.info(
        "Generation complete: %d samples kept, %d rejected",
        len(dataset), rejected,
    )
    return dataset


def _save_dataset(dataset: List[Dict]) -> List[Dict]:
    """
    Strip _meta, shuffle, train/eval split, and write JSONL files.
    Returns the cleaned (post-_meta) list of samples.
    """
    log.info("[5/5] Saving dataset...")

    # Strip the _meta debug fields — they're useful for audits but
    # would just confuse the trainer.
    clean_dataset = [{"messages": s["messages"]} for s in dataset]

    # Reproducible shuffle: seed the RNG so the same input gives the
    # same train/eval split every run. Critical for science.
    random.seed(config.RANDOM_SEED)
    random.shuffle(clean_dataset)

    # 90/10 split (or whatever TRAIN_SPLIT is set to).
    # For tiny datasets, make sure we still leave at least one sample
    # for eval when possible; an empty eval split makes training-time
    # validation noisy or impossible.
    split = int(len(clean_dataset) * config.TRAIN_SPLIT)
    if len(clean_dataset) >= 2:
        split = max(1, min(split, len(clean_dataset) - 1))
    elif len(clean_dataset) == 1:
        split = 1

    train_data = clean_dataset[:split]
    eval_data  = clean_dataset[split:]

    _save_jsonl(train_data, config.OUTPUT_DIR / "train.jsonl")
    _save_jsonl(eval_data,  config.OUTPUT_DIR / "eval.jsonl")

    # Persist the un-shuffled, _meta-bearing version for audits.
    _save_json(dataset, config.OUTPUT_DIR / "dataset_with_meta.json")

    log.info(
        "Dataset saved to %s — train: %d, eval: %d",
        config.OUTPUT_DIR, len(train_data), len(eval_data),
    )
    return clean_dataset


def generate_dataset(pdf_path: str = config.PDF_PATH) -> List[Dict]:
    """
    Run the full pipeline end-to-end.

    Parameters
    ----------
    pdf_path : str
        Path to the source PDF. Defaults to ``config.PDF_PATH``.

    Returns
    -------
    List[Dict]
        The cleaned (post-_meta) list of samples that was written to disk.

    Notes
    -----
    Each stage logs `[N/5]` so you can see progress in long-running runs.
    """
    log.info("=== STARTING DATASET GENERATION ===")
    config.ensure_directories()

    # ── Stage 1+2: Build the chunk list ────────────────────────────
    all_chunks = _build_all_chunks(pdf_path)

    # ── Stage 3: Load teacher ──────────────────────────────────────
    log.info("[3/5] Loading teacher model (gpt-oss-20b F16 GGUF)...")
    teacher = TeacherModel.load()

    # ── Stage 4: Generate ──────────────────────────────────────────
    dataset = _generate_samples(teacher, all_chunks)

    # ── Stage 5: Save ──────────────────────────────────────────────
    clean_dataset = _save_dataset(dataset)

    log.info("=== DATASET GENERATION COMPLETE ===")
    return clean_dataset
