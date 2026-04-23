"""
quality_filter.py
=================
Quality gate for teacher-generated responses.

WHY this is the most important file in the pipeline
----------------------------------------------------
A student model trained on bad samples *permanently* learns the bad
behaviour — there's no recovering it later. Quality filtering is the only
chance to drop hallucinated, refused, or off-topic samples before they
poison the dataset.

What we filter:
    1. Empty / too-short responses (likely truncated outputs)
    2. Refusals ("I cannot answer", "I don't have information")
    3. Responses with no overlap with the source chunk (hallucinations)
    4. Suspect numeric patterns not present in the source
We KEEP:
    • Valid fallbacks ("This information is not in the provided section")
      because they teach the student where the document's boundaries are.
"""

# `re` for keyword extraction and hallucination patterns.
import re
# Type hints — `Tuple` for the (bool, reason) return.
from typing import Tuple

from . import config
from .logger import get_logger
from .exceptions import QualityFilterError

log = get_logger(__name__)


def is_quality_response(response: str, chunk_text: str) -> Tuple[bool, str]:
    """
    Decide whether a teacher response is good enough for the dataset.

    Parameters
    ----------
    response : str
        The teacher's generated answer.
    chunk_text : str
        The source chunk that was used to ground the answer.

    Returns
    -------
    Tuple[bool, str]
        ``(is_good, reason)``. ``is_good=True`` means keep the sample;
        ``reason`` is a short human-readable explanation either way.

    Raises
    ------
    QualityFilterError
        Only on programmer errors (e.g., `None` passed in). Logical
        rejections are returned as `(False, ...)`, NOT raised.
    """
    # Programmer-error guard: a wrong type means somebody upstream is
    # broken. Raise instead of silently returning False.
    if response is None or chunk_text is None:
        log.error("is_quality_response received None inputs")
        raise QualityFilterError("response and chunk_text must not be None")

    # ── Check 1: minimum length ─────────────────────────────────────
    # `.strip()` so a response that's just whitespace counts as empty.
    if len(response.strip()) < config.MIN_RESPONSE_LENGTH:
        return (
            False,
            f"Response too short (< {config.MIN_RESPONSE_LENGTH} chars)"
            " — likely a refusal or empty output",
        )

    # ── Check 2: explicit refusal patterns ──────────────────────────
    # Lowercase the response once so we don't repeat the work.
    lower = response.lower()
    if "i cannot" in lower or "i don't have" in lower:
        return False, "Teacher refused to answer — rephrase the question"

    # ── Check 3: valid fallback (KEEP these) ────────────────────────
    # This is *not* a rejection — the teacher correctly recognised that
    # the chunk doesn't contain the answer. Such samples teach the
    # student to refuse hallucination on out-of-doc topics.
    if "this information is not in the provided section" in lower:
        return True, "Valid fallback — teaches the student document boundaries"

    # ── Check 4: keyword overlap (anti-hallucination) ───────────────
    # Significant words = 5+ chars (skips articles, prepositions, etc.).
    # Using `set` makes the intersection a fast O(min(len)) operation.
    chunk_keywords = {
        word.lower() for word in re.findall(r'\b[A-Za-z]{5,}\b', chunk_text)
    }
    response_words = {
        word.lower() for word in re.findall(r'\b[A-Za-z]{5,}\b', response)
    }
    overlap = chunk_keywords & response_words

    if len(overlap) < config.MIN_KEYWORD_OVERLAP:
        return (
            False,
            f"Response doesn't reference chunk content "
            f"(only {len(overlap)} shared words)",
        )

    # ── Check 5: hallucinated numeric patterns ──────────────────────
    # Pattern: a 4+ digit number followed by a "people" noun, not in a
    # legit savings context. The negative lookahead `(?!...)` exempts
    # phrasing like "1000 staff-hours saved" which IS in the document.
    hallucination_patterns = [
        r'\d{4,}(?:\s*(?:hours|staff|FTE|employees|workers))(?!\s+(?:per|saved|reduced))',
    ]
    for pattern in hallucination_patterns:
        match = re.search(pattern, response)
        if match:
            matched_text = match.group(0)
            # Verify the match actually appears in the source chunk —
            # if not, the model probably invented it.
            if matched_text not in chunk_text:
                return (
                    False,
                    f"Potential hallucination detected: "
                    f"'{matched_text}' not in source chunk",
                )

    # If we made it here, the response passed every check.
    return True, "Passes quality checks"
