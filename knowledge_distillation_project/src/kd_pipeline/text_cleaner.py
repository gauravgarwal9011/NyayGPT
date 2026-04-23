"""
text_cleaner.py
===============
Removes noise from PDF-extracted text before chunking.

WHY clean before chunking?
--------------------------
PDF extractors emit a lot of artefacts that don't help (and actively hurt)
LLM training:

* Multi-blank-line gaps that waste tokens.
* Multi-space alignment from multi-column layouts.
* Standalone page numbers that get embedded mid-paragraph.
* Footers / watermarks that repeat on every slide.

Cleaning is purely textual and idempotent — running it twice on the same
input gives the same output. That makes it safe to call from anywhere.
"""

# `re` is Python's regular-expression module. Every transformation here
# is a small regex substitution.
import re

from .logger import get_logger

# Module-level logger.
log = get_logger(__name__)


def clean_text(text: str) -> str:
    """
    Apply a series of regex passes to remove PDF noise from `text`.

    Parameters
    ----------
    text : str
        Raw text from `extract_pages_with_fitz()` (or any other source).

    Returns
    -------
    str
        Cleaned text. The function never raises — if `text` is empty,
        an empty string is returned.

    Notes
    -----
    The transformations applied (in order):
        1. Collapse 3+ newlines → 2 newlines (preserve paragraph breaks).
        2. Collapse 2+ spaces/tabs → single space.
        3. Drop standalone-number lines (page numbers).
        4. Drop common footer / watermark text.
        5. Collapse multiple `|` characters (table-extraction artefacts).
    """
    # Defensive: handle the empty/None edge case before regex calls so we
    # don't blow up on a TypeError.
    if not text:
        return ""

    log.debug("Cleaning text (%d chars before)", len(text))

    # ── 1. Collapse runs of 3+ newlines into exactly 2 newlines ──
    # `\n{3,}` matches 3 or more consecutive newline characters.
    # Replacing with `\n\n` keeps paragraph separation but kills the
    # giant gaps slide decks insert between sections.
    text = re.sub(r'\n{3,}', '\n\n', text)

    # ── 2. Collapse runs of 2+ spaces or tabs into a single space ──
    # `[ \t]{2,}` is a character class matching 2+ of either space or tab.
    # Multi-column PDFs sometimes use spaces to align text visually; that
    # alignment is meaningless in plain text and wastes token budget.
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # ── 3. Strip standalone page-number lines ──
    # Pattern breakdown:
    #   ^         — start of line (works because of MULTILINE flag)
    #   \s*       — any whitespace
    #   \d+       — one or more digits
    #   \s*       — any whitespace
    #   $         — end of line
    # `re.MULTILINE` makes `^` and `$` match at every newline rather than
    # only at the very start/end of the whole string.
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # ── 4. Strip company footer/watermark variants ──
    # The `(...)` groups three alternatives joined by `|` (regex OR).
    # `re.IGNORECASE` matches "ignatiuz.com" and "Ignatiuz.com" alike.
    text = re.sub(
        r'(Ignatiuz\.com|www\.ignatiuz\.com|@ignatiuz)',
        '',
        text,
        flags=re.IGNORECASE,
    )

    # ── 5. Collapse runs of `||...` from broken table extraction ──
    # `\|{2,}` matches 2+ pipe characters. We need to escape `|` because
    # it's a regex meta-character (alternation).
    text = re.sub(r'\|{2,}', '|', text)

    # Final trim of leading/trailing whitespace from the whole string.
    text = text.strip()

    log.debug("Cleaned text (%d chars after)", len(text))
    return text
