"""
pdf_extractor.py
================
PDF text + table extraction using PyMuPDF (fitz) and pdfplumber.

WHY TWO LIBRARIES?
------------------
* `fitz` (PyMuPDF) is fast and great for plain text in reading order.
  Perfect for slide decks where the goal is "give me the text on this page".
* `pdfplumber` is slow but excellent at *table* detection — it reconstructs
  row/column structure that fitz flattens into a string of cell values.

We use both: fitz for general body text, pdfplumber for tables only. The
two extractions are merged so each page text contains both narrative and
table content.
"""

# `os` is used for `os.path.exists` checks before we touch the file.
import os
# `typing.List` / `typing.Dict` give us readable type hints.
from typing import List, Dict

# Third-party libraries — both must be installed in the venv.
# `import fitz` is the canonical way to import PyMuPDF (the package is
# called PyMuPDF on PyPI but exposes a module called `fitz` because the
# original library was named "MuPDF/fitz").
import fitz
import pdfplumber

# Internal imports — relative (`.`) so the package stays portable.
from .logger import get_logger
from .exceptions import PDFExtractionError

# One module-level logger keyed by `__name__`. Convention.
log = get_logger(__name__)


def extract_pages_with_fitz(pdf_path: str) -> List[Dict]:
    """
    Extract text from every page of the PDF using PyMuPDF (fitz).

    Parameters
    ----------
    pdf_path : str
        Absolute path to the PDF file.

    Returns
    -------
    List[Dict]
        One dict per content page, of the form
        ``{"page_num": int, "text": str, "word_count": int}``.

    Raises
    ------
    PDFExtractionError
        If the file is missing, unreadable, or PyMuPDF refuses to open it.

    Notes
    -----
    Why fitz over pypdf or pdftotext?
        • fitz preserves reading order on slide decks (title → body).
        • fitz handles multi-column text correctly.
        • fitz is 3-5x faster than pdfplumber for plain text.
        • fitz handles UTF-8 characters used in business documents.
    """
    # Defensive check up-front: a missing-file error from fitz is cryptic
    # ("cannot open document"), so we surface a clear message ourselves.
    if not os.path.exists(pdf_path):
        log.error("PDF not found at %s", pdf_path)
        raise PDFExtractionError(f"PDF not found at: {pdf_path}")

    log.info("Opening PDF with PyMuPDF: %s", pdf_path)

    # `try`/`except` here lets us wrap *any* low-level fitz error in our
    # own typed exception, so callers don't need to import fitz to handle
    # failure modes.
    try:
        # `fitz.open(...)` reads the entire PDF into memory and returns a
        # `Document` object. For a 47-page slide deck this is fast (<1 s).
        doc = fitz.open(pdf_path)
    except Exception as exc:
        # We `raise ... from exc` so the original traceback is preserved
        # in the chained exception (good for debugging).
        log.exception("PyMuPDF failed to open the PDF")
        raise PDFExtractionError(f"PyMuPDF failed to open {pdf_path}: {exc}") from exc

    # Accumulator list — one dict per content page.
    pages: List[Dict] = []

    # `len(doc)` returns the page count. Iterating by index lets us also
    # report 1-based page numbers (humans count from 1).
    for page_num in range(len(doc)):
        # `doc[i]` returns a `fitz.Page` object — has methods for text,
        # images, links, annotations, and more.
        page = doc[page_num]

        # `get_text("text")` extracts plain text in reading order
        # (top-to-bottom, left-to-right). Other modes ("blocks", "html",
        # "dict") are richer but more verbose.
        text = page.get_text("text")

        # Strip leading/trailing whitespace; slide decks usually have a
        # trailing newline at the end of each page's text block.
        text = text.strip()

        # Skip near-empty pages: pure-image slides, divider pages, etc.
        # 50 chars is roughly "less than a sentence" — useless as training
        # context.
        if len(text) < 50:
            log.debug("Skipping page %d (only %d chars)", page_num + 1, len(text))
            continue

        # Quick proxy for content density.
        word_count = len(text.split())

        # Append a dict with the data we want downstream. We use 1-based
        # `page_num` so it matches what humans see in the PDF viewer.
        pages.append({
            "page_num":   page_num + 1,
            "text":       text,
            "word_count": word_count,
        })

    # Explicitly close the document. Python's GC would eventually do this,
    # but explicit close avoids file-lock issues on Windows.
    doc.close()

    log.info("Extracted %d content pages from %s", len(pages), pdf_path)
    return pages


def extract_tables_with_pdfplumber(pdf_path: str) -> List[Dict]:
    """
    Extract tables from the PDF using pdfplumber.

    Parameters
    ----------
    pdf_path : str
        Absolute path to the PDF file.

    Returns
    -------
    List[Dict]
        One dict per detected table, of the form
        ``{"page_num": int, "table_text": str}`` where ``table_text`` is a
        readable plain-text rendering with cells separated by ``" | "``.

    Raises
    ------
    PDFExtractionError
        If the PDF cannot be opened by pdfplumber.

    Notes
    -----
    pdfplumber detects tables by analysing the spatial positions of words
    and grouping them into rows/columns. This is *much* better than fitz
    for comparison tables but ~10x slower, so we run it as a separate step
    and only on PDFs known to contain tables.
    """
    if not os.path.exists(pdf_path):
        log.error("PDF not found at %s", pdf_path)
        raise PDFExtractionError(f"PDF not found at: {pdf_path}")

    log.info("Opening PDF with pdfplumber for table extraction: %s", pdf_path)

    tables_data: List[Dict] = []

    # Outer try wraps the open call so we can convert pdfplumber's errors
    # into our typed exception.
    try:
        # `with pdfplumber.open(...) as pdf:` is a context manager — it
        # guarantees the underlying file handle is closed even if an
        # exception fires inside the block.
        with pdfplumber.open(pdf_path) as pdf:

            # `enumerate(..., start=1)` numbers pages from 1, matching the
            # human-visible page numbers in the PDF viewer.
            for page_num, page in enumerate(pdf.pages, start=1):

                # `extract_tables()` returns a list of tables found on
                # this page. Each table is a list of rows; each row is a
                # list of cell strings. Empty cells are `None`.
                tables = page.extract_tables()

                for table in tables:
                    # Sanity-check: skip empty or single-row "tables"
                    # (often false positives — pdfplumber occasionally
                    # mis-labels paragraph headers as tables).
                    if not table or len(table) < 2:
                        continue

                    # Convert the 2-D list into a readable text string.
                    table_lines = []
                    for row in table:
                        # Each `cell` may be `None` (empty cell). We turn
                        # it into "" so we can strip and join uniformly.
                        cleaned_cells = [
                            str(cell).strip() if cell else "" for cell in row
                        ]

                        # `any(cleaned_cells)` is True iff at least one
                        # cell is non-empty — drops fully blank rows.
                        if any(cleaned_cells):
                            # Join cells with " | " — readable as plain
                            # text and unambiguous as a separator.
                            table_lines.append(" | ".join(cleaned_cells))

                    # Only emit non-empty tables.
                    if table_lines:
                        tables_data.append({
                            "page_num":   page_num,
                            # Each row on its own line, cells separated
                            # by " | ".
                            "table_text": "\n".join(table_lines),
                        })

    except Exception as exc:
        log.exception("pdfplumber failed to extract tables")
        # Wrap and re-raise so the caller sees a typed exception.
        raise PDFExtractionError(
            f"pdfplumber failed to extract tables from {pdf_path}: {exc}"
        ) from exc

    log.info("Extracted %d tables from %s", len(tables_data), pdf_path)
    return tables_data
