"""
Live SEC EDGAR search — no pre-processing pipeline required.
Fetches filings directly from EDGAR and answers with GPT-4o mini.
"""
import re
import time
import threading
import logging
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from sec_intelligence.config import get_config

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "SEC Intelligence Platform research@secplatform.com",
    "Accept-Encoding": "gzip, deflate",
}


# ── EDGAR rate limiter (token bucket, 8 req/sec — limit is 10) ───────────────

class _RateLimiter:
    """Token-bucket rate limiter enforcing EDGAR's 10 req/sec policy."""

    def __init__(self, rate: float = 8.0):
        self._rate = rate
        self._tokens = rate
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            deficit = 1.0 - self._tokens
            wait_s = deficit / self._rate
            self._tokens = 0.0
        time.sleep(wait_s)


_edgar_limiter = _RateLimiter(8.0)

# ── In-process caches ────────────────────────────────────────────────────────

_filing_text_cache: dict[str, str] = {}            # accession:max_chars → text
_edgar_index_cache: dict[str, list[dict]] = {}     # "YYYY/QTRN" → all quarter entries


# ── Retry wrapper (rate-limits + handles 429/5xx automatically) ───────────────

def _retry_get(url: str, timeout: int = 30, max_retries: int = 3) -> requests.Response:
    """GET with automatic retry on 429/5xx using exponential backoff."""
    for attempt in range(max_retries):
        _edgar_limiter.wait()
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 2.0 ** (attempt + 1)))
                logger.warning(f"EDGAR 429 — waiting {wait:.1f}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504) and attempt < max_retries - 1:
                wait = 2.0 ** attempt
                logger.warning(f"EDGAR {r.status_code} — retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2.0 ** attempt)
                continue
            raise
    raise RuntimeError(f"All {max_retries} attempts failed for {url}")


# ── Ticker / CIK lookup (with in-process cache) ───────────────────────────────

_ticker_data: list[dict] | None = None   # [{cik_str, ticker, title}, ...]


def _load_ticker_data() -> list[dict]:
    global _ticker_data
    if _ticker_data is None:
        r = _retry_get("https://www.sec.gov/files/company_tickers.json", timeout=10)
        _ticker_data = list(r.json().values())
    return _ticker_data


def ticker_to_cik(ticker: str) -> str | None:
    """Return zero-padded 10-digit CIK for a ticker, or None if not found."""
    try:
        for item in _load_ticker_data():
            if item.get("ticker", "").upper() == ticker.upper():
                return str(item["cik_str"]).zfill(10)
    except Exception as e:
        logger.error(f"ticker_to_cik failed: {e}")
    return None


def cik_to_ticker(cik: str | int) -> str | None:
    """Return ticker for a CIK, or None if not in SEC's company list."""
    try:
        cik_int = int(cik)
        for item in _load_ticker_data():
            if item.get("cik_str") == cik_int:
                return item.get("ticker")
    except Exception as e:
        logger.error(f"cik_to_ticker failed: {e}")
    return None


# ── Quarterly index (cached for process lifetime) ─────────────────────────────

def _load_quarter_index(year: int, quarter: int) -> list[dict]:
    """
    Download EDGAR's quarterly company.idx once and cache it in memory.
    ~10 MB download; subsequent calls for the same quarter are instant.
    """
    cache_key = f"{year}/QTR{quarter}"
    if cache_key in _edgar_index_cache:
        return _edgar_index_cache[cache_key]

    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/company.idx"
    try:
        r = _retry_get(url, timeout=45)
    except Exception as e:
        logger.error(f"_load_quarter_index: failed for {cache_key}: {e}")
        return []

    entries: list[dict] = []
    for line in r.text.splitlines():
        date_m = re.search(r'(\d{4}-\d{2}-\d{2})', line)
        if not date_m:
            continue
        try:
            company       = line[:62].strip()
            form_type_val = line[62:74].strip()
            cik_raw       = line[74:86].strip()
            filepath      = line[date_m.start() + 10:].strip()

            acc_m = re.search(r'(\d{10}-\d{2}-\d{6})', filepath)
            if not acc_m or not form_type_val:
                continue

            cik_int = int(cik_raw) if cik_raw.isdigit() else None
            entries.append({
                "company":      company,
                "form_type":    form_type_val,
                "cik":          str(cik_int).zfill(10) if cik_int else cik_raw,
                "filing_date":  date_m.group(1),
                "accession":    acc_m.group(1),
                "ticker":       "",
            })
        except Exception:
            continue

    # Resolve tickers in a single pass against the cached ticker list
    try:
        ticker_map = {item["cik_str"]: item["ticker"] for item in _load_ticker_data()}
        for f in entries:
            try:
                f["ticker"] = ticker_map.get(int(f["cik"]), "")
            except Exception:
                pass
    except Exception:
        pass

    _edgar_index_cache[cache_key] = entries
    logger.info(f"_load_quarter_index: cached {len(entries)} entries for {cache_key}")
    return entries


# ── All filings on a specific date ────────────────────────────────────────────

def get_filings_for_date(
    date_str: str,
    form_types: list[str] | None = None,
) -> list[dict]:
    """
    Return every SEC filing submitted on date_str (YYYY-MM-DD).
    The quarterly index is fetched once and cached; subsequent calls are instant.
    """
    from datetime import datetime as _dt
    dt = _dt.strptime(date_str, "%Y-%m-%d")
    all_entries = _load_quarter_index(dt.year, (dt.month - 1) // 3 + 1)
    filings = [e for e in all_entries if e["filing_date"] == date_str]
    if form_types:
        wanted = {f.upper() for f in form_types}
        filings = [f for f in filings if f["form_type"].upper() in wanted]
    return filings


# ── Paginated filing retrieval (used by fetch_sec_filing tool) ────────────────

def get_filings_paginated(
    date_str: str,
    form_types: list[str] | None = None,
    company_filter: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """
    Paginated wrapper over get_filings_for_date.
    Returns {filings, total, returned, offset, limit, has_more, next_offset}.
    The quarterly index is fetched once; all subsequent pages are instant.
    """
    filings = get_filings_for_date(date_str, form_types)
    if company_filter:
        cf = company_filter.upper()
        filings = [f for f in filings if cf in f.get("company", "").upper()]
    total = len(filings)
    page  = filings[offset: offset + limit]
    return {
        "filings":     page,
        "total":       total,
        "returned":    len(page),
        "offset":      offset,
        "limit":       limit,
        "has_more":    (offset + limit) < total,
        "next_offset": (offset + limit) if (offset + limit) < total else None,
    }


# ── Latest filing lookup ──────────────────────────────────────────────────────

def get_latest_filing(cik: str, form_type: str = "10-K") -> dict | None:
    """Return info dict for the most recent filing of form_type, or None."""
    try:
        r = _retry_get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=10)
        r.raise_for_status()
        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])

        for i, form in enumerate(forms):
            if form == form_type:
                return {
                    "accession": accessions[i],
                    "date": dates[i],
                    "cik": cik,
                    "company": data.get("name", ticker_label(cik)),
                }
    except Exception as e:
        logger.error(f"get_latest_filing failed: {e}")
    return None


def ticker_label(cik: str) -> str:
    return f"CIK {cik}"


# ── Filing text fetcher ───────────────────────────────────────────────────────

def _get_doc_url(cik: str, accession: str) -> str | None:
    """
    Find the primary HTML document URL for a filing using the -index.htm page.
    Falls back to the SGML composite .txt file if no HTML document is found.
    """
    acc_flat = accession.replace("-", "")
    cik_int  = int(cik)
    base_path = f"/Archives/edgar/data/{cik_int}/{acc_flat}/"

    idx_url = f"https://www.sec.gov{base_path}{accession}-index.htm"
    try:
        r = _retry_get(idx_url, timeout=10)
    except Exception as e:
        logger.error(f"_get_doc_url: index fetch failed: {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    candidates = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]

        # Strip iXBRL viewer wrapper: /ix?doc=/Archives/...
        if "/ix?doc=" in href:
            href = href.split("/ix?doc=", 1)[1]

        # Normalise absolute SEC URLs → relative path
        if href.startswith("https://www.sec.gov"):
            href = href[len("https://www.sec.gov"):]

        # Must be inside this filing's directory
        if not href.startswith(base_path):
            continue

        lower = href.lower().split("/")[-1]

        # Must be an HTML or plain-text document
        if not any(lower.endswith(ext) for ext in (".htm", ".html", ".txt")):
            continue

        # Skip index/viewer files
        if any(kw in lower for kw in ("index", "viewer", "-index")):
            continue

        candidates.append(f"https://www.sec.gov{href}")

    if candidates:
        # Prefer HTML over .txt; within HTML prefer non-exhibits and shorter names
        def _score(url: str) -> tuple:
            name = url.split("/")[-1].lower()
            is_txt     = int(name.endswith(".txt"))
            is_exhibit = int("ex" in name or "exhibit" in name)
            return (is_txt, is_exhibit, len(url))

        candidates.sort(key=_score)
        return candidates[0]

    # Fallback: SGML composite document (always present on EDGAR)
    sgml_url = f"https://www.sec.gov{base_path}{accession}.txt"
    try:
        _edgar_limiter.wait()
        head = requests.head(sgml_url, headers=HEADERS, timeout=5)  # HEAD only — don't download
        if head.status_code == 200:
            return sgml_url
    except Exception:
        pass

    return None


def fetch_filing_text(cik: str, accession: str, max_chars: int = 60_000) -> str:
    """Download a filing and return plain text (capped at max_chars)."""
    cache_key = f"{accession}:{max_chars}"
    if cache_key in _filing_text_cache:
        return _filing_text_cache[cache_key]

    doc_url = _get_doc_url(cik, accession)
    if not doc_url:
        logger.warning(f"No document URL found for {accession}")
        return ""

    try:
        r = _retry_get(doc_url, timeout=30)
    except Exception as e:
        logger.error(f"fetch_filing_text download failed: {e}")
        return ""

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "meta", "link", "noscript"]):
        tag.decompose()

    # Use '\n' as separator so block elements produce line breaks we can split on
    text = soup.get_text(separator="\n")
    # Strip each line, drop empty ones, then rejoin with paragraph breaks
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    text = "\n\n".join(lines)
    # Normalise horizontal whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)
    result = text[:max_chars]
    _filing_text_cache[cache_key] = result
    return result


# ── Passage extraction ────────────────────────────────────────────────────────

def find_relevant_passages(text: str, query: str, top_n: int = 3, window: int = 1500) -> str:
    """
    Slide a window across text and return the top_n highest-scoring windows
    concatenated, scored by query-word overlap.
    """
    query_words = {w.lower() for w in re.findall(r"\w+", query) if len(w) > 3}
    if not query_words:
        return text[:window * top_n]

    step = 300
    scored: list[tuple[int, int]] = []  # (score, start)

    for start in range(0, max(1, len(text) - window), step):
        chunk = text[start : start + window].lower()
        score = sum(1 for w in query_words if w in chunk)
        scored.append((score, start))

    scored.sort(key=lambda x: -x[0])

    # Pick top_n non-overlapping windows
    selected = []
    used_ranges: list[tuple[int, int]] = []

    for _, start in scored:
        end = start + window
        if not any(s < end and start < e for s, e in used_ranges):
            selected.append(text[start:end])
            used_ranges.append((start, end))
        if len(selected) >= top_n:
            break

    return "\n\n---\n\n".join(selected)


# ── Concurrent content fetch (used by fetch_sec_filing tool) ─────────────────

def fetch_filings_content_concurrent(
    filing_refs: list[dict],
    max_chars: int = 60_000,
    max_workers: int = 4,
) -> list[dict]:
    """
    Fetch text for multiple filings concurrently using a thread pool.
    The rate limiter inside fetch_filing_text serialises EDGAR access safely
    even when multiple threads are running — no extra coordination needed.

    filing_refs: list of dicts with at minimum {cik, accession}; any extra
                 fields (company, ticker, …) are preserved in the output.
    Returns list of {cik, accession, text, char_count, error, …extra}.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(ref: dict) -> dict:
        cik       = str(ref.get("cik", ""))
        accession = str(ref.get("accession", ""))
        try:
            text = fetch_filing_text(cik, accession, max_chars)
            return {**ref, "text": text, "char_count": len(text), "error": None}
        except Exception as e:
            return {**ref, "text": "", "char_count": 0, "error": str(e)}

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, ref): ref for ref in filing_refs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                ref = futures[future]
                results.append({**ref, "text": "", "char_count": 0, "error": str(e)})

    return results


# ── Main entry point ──────────────────────────────────────────────────────────

def live_answer(question: str, ticker: str, form_type: str = "10-K") -> dict:
    """
    Fetch the most recent SEC filing for ticker and answer question with GPT-4o mini.

    Returns a dict with keys:
        answer, company, date, accession, doc_url, error
    """
    result = {"answer": "", "company": "", "date": "", "accession": "", "doc_url": "", "error": ""}

    config = get_config()
    api_key = config.llm.api_key
    if not api_key or api_key.startswith("your_"):
        result["error"] = "OpenAI API key not set. Add LLM_API_KEY to your .env file."
        result["answer"] = result["error"]
        return result

    client = OpenAI(api_key=api_key)

    # 1. Ticker → CIK
    cik = ticker_to_cik(ticker.strip().upper())
    if not cik:
        result["error"] = f"Could not find an SEC CIK for ticker '{ticker}'. Check the ticker is correct."
        result["answer"] = result["error"]
        return result

    # 2. Latest filing
    filing = get_latest_filing(cik, form_type)
    if not filing:
        result["error"] = f"No {form_type} filing found on EDGAR for {ticker} (CIK {int(cik)})."
        result["answer"] = result["error"]
        return result

    result["company"] = filing["company"]
    result["date"] = filing["date"]
    result["accession"] = filing["accession"]

    # 3. Fetch text
    text = fetch_filing_text(cik, filing["accession"])
    if not text:
        result["error"] = "Filing was found but its text could not be downloaded from EDGAR."
        result["answer"] = result["error"]
        return result

    # 4. Extract relevant passages
    passage = find_relevant_passages(text, question)

    # 5. GPT-4o mini
    prompt = (
        f"You are analyzing {filing['company']}'s {form_type} filing dated {filing['date']} "
        f"(accession {filing['accession']}).\n\n"
        f"Answer the following question based solely on the excerpts below. "
        f"Be specific and cite facts from the text.\n\n"
        f"QUESTION: {question}\n\n"
        f"FILING EXCERPTS:\n{passage}\n\n"
        f"If the excerpts don't contain enough to fully answer, say so clearly and "
        f"summarize what they do say about the topic."
    )

    try:
        response = client.chat.completions.create(
            model=config.llm.model_name,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        result["answer"] = response.choices[0].message.content
    except Exception as e:
        result["error"] = f"LLM call failed: {e}"
        result["answer"] = result["error"]

    return result
