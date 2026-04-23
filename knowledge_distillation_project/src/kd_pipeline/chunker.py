"""
chunker.py
==========
Splits cleaned text into overlapping windows ("chunks") suitable for use as
RAG-style context inside the teacher's prompt.

WHY chunk?
----------
Feeding the entire document into the teacher would
  • blow past the model's context window, and
  • cause the teacher to draw on whichever part of the doc is closest to
    its current generation, instead of the part the user actually asks about.
Chunking lets the teacher focus on one passage at a time → better grounding,
more samples, lower per-prompt cost.

WHY *overlapping* chunks?
-------------------------
A non-overlapping split sometimes cuts a key sentence in half. With an
overlap of ~100 chars, every sentence appears in *some* chunk in full —
no fact gets lost at a boundary.

WHY merge pages first, then chunk?
----------------------------------
Slide decks often spread one topic across consecutive slides:
    Slide 12: "AP Automation — Challenge"
    Slide 13: "AP Automation — Solution"
    Slide 14: "AP Automation — Results: 90% reduction"
Per-page chunking would isolate "90% reduction" from its context.
Merging first lets a single overlapping chunk span all three slides.
"""

# `typing.List` and `typing.Dict` give us readable type hints.
from typing import List, Dict

# Internal imports.
from . import config
from .logger import get_logger
from .text_cleaner import clean_text
from .exceptions import ChunkingError

log = get_logger(__name__)


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Merge pages into one document, then split into overlapping chunks.

    Parameters
    ----------
    pages : List[Dict]
        Pages emitted by `extract_pages_with_fitz()`.
    chunk_size : int
        Target characters per chunk. Default comes from config.
    overlap : int
        Characters of overlap between consecutive chunks.

    Returns
    -------
    List[Dict]
        Each chunk is ``{"chunk_id": int, "text": str, "char_start": int,
        "char_end": int}``.

    Raises
    ------
    ChunkingError
        If `pages` is empty or all pages collapse to nothing after cleaning.
    """
    # Defensive guard — log AND raise so the caller has structured info.
    if not pages:
        log.error("chunk_pages received an empty `pages` list")
        raise ChunkingError("Cannot chunk: pages list is empty")

    log.info("Chunking %d pages (chunk_size=%d, overlap=%d)",
             len(pages), chunk_size, overlap)

    # ── Step 1: Merge all pages into one big string with page markers. ──
    # We accumulate into a list and `"".join` at the end. (Repeatedly
    # using `+=` on a string is O(n²) — the list trick is the standard
    # idiom in Python for concatenating many strings.)
    parts: List[str] = []
    for page in pages:
        cleaned = clean_text(page["text"])
        if cleaned:
            # Each page becomes its own block, prefixed by a marker like
            # `[Page 12]`. The marker is preserved in the chunks so the
            # teacher knows which page each fact came from — and so we
            # can audit chunks back to the source.
            parts.append(f"\n\n[Page {page['page_num']}]\n{cleaned}")
    full_text = "".join(parts)

    # If cleaning stripped everything, fail loudly with a typed error.
    if not full_text.strip():
        log.error("All pages were empty after cleaning")
        raise ChunkingError("All pages were empty after cleaning")

    # ── Step 2: Sliding window over `full_text`. ──────────────────────
    chunks: List[Dict] = []
    start = 0                     # current window start, in characters
    total_len = len(full_text)    # cached length so we don't recompute

    # `while True` + `break` is the cleanest way to express "do this
    # until we run off the end OR the chunk is too small to keep".
    while start < total_len:
        end = start + chunk_size

        # Slice the window. Python slicing handles `end > total_len`
        # gracefully — it just returns up to the end.
        chunk_text = full_text[start:end]

        # If we're past a useful chunk (e.g. final 50-char tail), stop
        # rather than feed the teacher a near-empty fragment.
        if len(chunk_text.strip()) < 100:
            break

        # ── Try to find a clean sentence/paragraph boundary near the end
        # of the chunk so we don't cut mid-sentence. ──
        # `rfind(needle, lo)` searches backwards starting from index `lo`.
        # We look in the last 150 chars for a period-space, period-newline,
        # or double-newline.
        lookback = max(0, len(chunk_text) - 150)
        last_sentence_end = max(
            chunk_text.rfind('. ',  lookback),
            chunk_text.rfind('.\n', lookback),
            chunk_text.rfind('\n\n', lookback),
        )

        # Only use the boundary if it's at least halfway through the
        # chunk — otherwise we'd discard most of the window for nothing.
        if last_sentence_end > len(chunk_text) // 2:
            chunk_text = chunk_text[: last_sentence_end + 1]

        # Append the cleaned-up chunk dict.
        chunks.append({
            "chunk_id":   len(chunks),
            "text":       chunk_text.strip(),
            "char_start": start,
            "char_end":   start + len(chunk_text),
        })

        # Advance start by (chunk length - overlap). The subtraction is
        # what creates the sliding window effect. `max(1, …)` guarantees
        # forward progress even when the sentence-boundary truncation
        # shrinks chunk_text down to ≤ overlap (which would otherwise
        # leave `start` unchanged and loop forever → OOM kill).
        start += max(1, len(chunk_text) - overlap)

        # Sanity break if we've reached/exceeded the end of the doc.
        if start >= total_len:
            break

    log.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks


def chunk_knowledge_base(
    knowledge_base: Dict[str, str],
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Split each section of the verified knowledge base into chunks.

    Parameters
    ----------
    knowledge_base : Dict[str, str]
        Mapping section_name → section_text. Comes from
        `kd_pipeline.knowledge_base.DOCUMENT_KNOWLEDGE_BASE`.

    Returns
    -------
    List[Dict]
        Each chunk dict has the same fields produced by `chunk_pages`,
        plus ``"source": "knowledge_base"`` and ``"section": <key>``,
        and a string ``chunk_id`` like ``"kb_<section>_<part>"``.

    Notes
    -----
    The knowledge base is the *primary* (verified) source — chunks
    produced here are emitted before the fitz chunks so the highest-
    quality samples land first in the dataset.
    """
    if not knowledge_base:
        log.error("chunk_knowledge_base received an empty knowledge_base")
        raise ChunkingError("Cannot chunk: knowledge_base is empty")

    log.info("Chunking knowledge base (%d sections)", len(knowledge_base))

    kb_chunks: List[Dict] = []
    # `.items()` yields (key, value) pairs in insertion order (Python 3.7+).
    for section_name, section_text in knowledge_base.items():
        cleaned = section_text.strip()
        if not cleaned:
            # Skip empty sections silently — they're a misconfiguration
            # but not a hard error.
            log.warning("Knowledge base section %r is empty", section_name)
            continue

        # Two indexes: `start` walks the source string, `part` numbers
        # the resulting chunks within this section.
        start = 0
        part = 0
        while start < len(cleaned):
            end = start + chunk_size
            chunk_text = cleaned[start:end]

            kb_chunks.append({
                # `kb_` prefix distinguishes from numeric fitz chunk IDs.
                "chunk_id":   f"kb_{section_name}_{part}",
                "text":       chunk_text,
                # Tag the source so audits/logs can tell verified chunks
                # from secondary fitz chunks.
                "source":     "knowledge_base",
                "section":    section_name,
                "char_start": start,
                "char_end":   start + len(chunk_text),
            })

            # Same overlap arithmetic as chunk_pages above. We must
            # advance by at least 1 char or we'd loop forever on any
            # section shorter than `overlap` (chunk_text - overlap ≤ 0).
            advance = max(1, len(chunk_text) - overlap)
            start += advance
            part += 1

            # If the slice already covered the rest of the section, stop —
            # otherwise we'd emit a duplicate tail chunk on the next pass.
            if end >= len(cleaned):
                break

    log.info("Built %d verified chunks from %d sections",
             len(kb_chunks), len(knowledge_base))
    return kb_chunks
