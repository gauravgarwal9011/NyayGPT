"""
quality_filter.py — Legal-domain quality gate for synthetic Q&A pairs.

Adapted from the KD project's quality_filter.py with:
  - Higher min_response_length (100 chars vs 80) — legal answers are verbose
  - Higher min_keyword_overlap (6 vs 5) — legal text uses precise terminology
  - Legal-specific hallucination patterns (fake case numbers, IPC sections)
"""
import re
from typing import Tuple

from . import config
from .logger import get_logger

log = get_logger(__name__)

# Patterns that suggest hallucinated legal citations
_HALLUCINATION_PATTERNS = [
    re.compile(r"\bAIR\s+\d{4}\s+SC\s+\d+\b"),       # Fake AIR citations like "AIR 2021 SC 99999"
    re.compile(r"\bCrl\.?\s*A\.?\s*No\.\s*\d{5,}\b"), # Unrealistically long case numbers
    re.compile(r"\bIPC\s+[Ss]ection\s+(?:0|999|1000)\b"),  # Non-existent IPC sections
]

_REFUSAL_PHRASES = [
    "i cannot", "i don't have", "i'm unable to", "as an ai",
    "i do not have access", "i cannot provide legal advice",
    "please consult a lawyer",  # generic deflection (specific ok, boilerplate not)
]


def is_quality_response(response: str, chunk_text: str) -> Tuple[bool, str]:
    """
    Return (is_good: bool, reason: str).

    Args:
        response:   The generated assistant answer.
        chunk_text: The source document chunk used to generate the question.
    """
    stripped = response.strip()

    # ── 1. Minimum length ────────────────────────────────────────────────────
    if len(stripped) < config.MIN_RESPONSE_LENGTH:
        return False, f"too_short ({len(stripped)} < {config.MIN_RESPONSE_LENGTH})"

    lower = stripped.lower()

    # ── 2. Explicit refusals ─────────────────────────────────────────────────
    for phrase in _REFUSAL_PHRASES:
        if phrase in lower:
            return False, f"refusal_phrase: '{phrase}'"

    # ── 3. Valid fallback — keep "not in excerpt" responses ──────────────────
    # These teach the model its knowledge boundaries, which is valuable.
    valid_fallbacks = [
        "this information is not in",
        "the excerpt does not contain",
        "not mentioned in the provided",
        "based on the available excerpt, this",
    ]
    if any(fb in lower for fb in valid_fallbacks):
        return True, "valid_fallback"

    # ── 4. Keyword overlap (anti-hallucination) ───────────────────────────────
    chunk_words  = {w for w in re.findall(r"\b\w{5,}\b", chunk_text.lower())}
    resp_words   = {w for w in re.findall(r"\b\w{5,}\b", lower)}
    overlap      = len(chunk_words & resp_words)
    if overlap < config.MIN_KEYWORD_OVERLAP:
        return False, f"low_overlap ({overlap} < {config.MIN_KEYWORD_OVERLAP})"

    # ── 5. Legal hallucination patterns ──────────────────────────────────────
    for pat in _HALLUCINATION_PATTERNS:
        match = pat.search(stripped)
        if match:
            # Only reject if the citation doesn't appear in the source chunk
            citation = match.group(0)
            if citation not in chunk_text:
                return False, f"hallucinated_citation: '{citation}'"

    return True, "ok"
