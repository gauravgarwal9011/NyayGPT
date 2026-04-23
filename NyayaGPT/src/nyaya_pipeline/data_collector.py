"""
data_collector.py — IndianKanoon scraper for legal case text.

Uses the public IndianKanoon search endpoint to retrieve judgment excerpts.
No authentication required for basic search; respects robots.txt and rate-limits.
"""
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

from . import config
from .exceptions import ScrapingError
from .logger import get_logger

log = get_logger(__name__)

INDIANKANOON_SEARCH_URL = "https://indiankanoon.org/search/"
INDIANKANOON_DOC_BASE   = "https://indiankanoon.org"

# Legal topic queries — broad enough to capture diverse case law
DEFAULT_QUERIES = [
    "IPC section 302 murder",
    "IPC section 420 cheating fraud",
    "criminal appeal High Court",
    "constitutional rights fundamental article 21",
    "PIL public interest litigation Supreme Court",
    "property dispute civil suit",
    "cyber crime Information Technology Act",
    "contract law breach damages",
    "domestic violence Protection of Women Act",
    "land acquisition compensation",
    "copyright infringement intellectual property India",
    "motor accident compensation MACT",
    "cheque dishonour NI Act section 138",
    "bail application Sessions Court",
    "habeas corpus writ petition",
]


def _clean_text(raw: str) -> str:
    """Remove HTML entities, extra whitespace, and citation noise."""
    text = re.sub(r"\s+", " ", raw)
    text = re.sub(r"\[\d+\]", "", text)        # Remove footnote refs [1], [2]
    text = re.sub(r"_{3,}", "", text)           # Remove underscores used as dividers
    return text.strip()


def _fetch_search_results(query: str, page: int = 1, timeout: int = 15) -> List[Dict]:
    """Fetch one page of search results. Returns list of {title, url, court, snippet} dicts."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ScrapingError("requests/bs4 not installed. pip install requests beautifulsoup4 lxml") from exc

    params = {"formInput": query, "pagenum": page}
    headers = {"User-Agent": "NyayaGPT-Research-Bot/1.0 (educational; non-commercial)"}

    try:
        resp = requests.get(INDIANKANOON_SEARCH_URL, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ScrapingError(f"Search request failed for query '{query}': {exc}") from exc

    soup = BeautifulSoup(resp.text, "lxml")
    results = []

    # Actual HTML (2025): <article class="result"> inside <div class="results-list">
    for article in soup.select("article.result"):
        title_tag   = article.select_one("h4.result_title a")
        snippet_div = article.select_one("div.headline")
        # Full-document link: <a class="cite_tag" href="/doc/...">Full Document</a>
        doc_link    = article.select_one('a.cite_tag[href^="/doc/"]')
        court_span  = article.select_one("span.docsource")

        if not title_tag or not doc_link:
            continue

        results.append({
            "title":   title_tag.get_text(strip=True),
            "url":     INDIANKANOON_DOC_BASE + doc_link.get("href", ""),
            "court":   court_span.get_text(strip=True) if court_span else "",
            "snippet": _clean_text(snippet_div.get_text()) if snippet_div else "",
        })

    return results


def _fetch_judgment_text(url: str, timeout: int = 20) -> Optional[str]:
    """
    Fetch the full text of a judgment from its IndianKanoon URL.
    Returns cleaned plain text, or None on failure.
    """
    import requests
    from bs4 import BeautifulSoup
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Failed to fetch judgment %s: %s", url, exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    # IndianKanoon doc page: judgment text is in <div id="judgement">
    body = (
        soup.find("div", id="judgement")
        or soup.find("div", class_="judgments")   # IndianKanoon 2025 live selector
        or soup.find("div", class_="judgement")
        or soup.find("div", class_="doc_content")
    )
    if not body:
        return None

    return _clean_text(body.get_text(separator=" "))


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Sliding-window chunker (same algorithm as KD project)."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap    = overlap    or config.CHUNK_OVERLAP

    if len(text) <= chunk_size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return [c for c in chunks if len(c) > 100]  # drop tiny tail chunks


def collect_judgments(
    queries: Optional[List[str]] = None,
    pages_per_query: int = 2,
    max_docs: int = 600,
    output_path: Optional[Path] = None,
    rate_limit_secs: float = 1.5,
) -> List[Dict]:
    """
    Scrape IndianKanoon for judgment texts.

    Returns list of {title, url, court, year, text, chunks} dicts.
    Saves raw JSON to output_path if provided.
    """
    queries = queries or DEFAULT_QUERIES
    output_path = output_path or (config.OUTPUT_DIR / "raw_judgments.json")

    collected: List[Dict] = []
    seen_urls: set = set()

    for query in queries:
        if len(collected) >= max_docs:
            break
        log.info("Scraping query: '%s'", query)

        for page in range(1, pages_per_query + 1):
            if len(collected) >= max_docs:
                break

            try:
                results = _fetch_search_results(query, page=page)
            except ScrapingError as exc:
                log.warning("Skipping query '%s' page %d: %s", query, page, exc)
                break

            for r in results:
                if r["url"] in seen_urls or len(collected) >= max_docs:
                    continue

                time.sleep(rate_limit_secs)   # be polite
                judgment_text = _fetch_judgment_text(r["url"])
                if not judgment_text or len(judgment_text) < 300:
                    continue

                seen_urls.add(r["url"])
                chunks = chunk_text(judgment_text)

                # Extract year from URL or title if possible
                year_match = re.search(r"\b(19|20)\d{2}\b", r["url"] + r["title"])
                year = year_match.group(0) if year_match else "unknown"

                collected.append({
                    "title":  r["title"],
                    "url":    r["url"],
                    "year":   year,
                    "text":   judgment_text[:5000],   # cap stored raw text
                    "chunks": chunks,
                    "query":  query,
                })
                log.debug("Collected: %s (%d chunks)", r["title"][:60], len(chunks))

            time.sleep(rate_limit_secs)

    log.info("Collected %d judgments with %d total chunks",
             len(collected), sum(len(d["chunks"]) for d in collected))

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(collected, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved raw judgments → %s", output_path)

    return collected
