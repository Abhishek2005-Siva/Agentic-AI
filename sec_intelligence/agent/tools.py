"""
Tool implementations for the SEC Intelligence Agent.
Each function returns a dict with at minimum an 'error' key (None on success).
"""
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ── Standardized filing schema ────────────────────────────────────────────────

def _standardize_filing(raw: dict) -> dict:
    """Convert any EDGAR filing dict into a consistent schema."""
    cik       = str(raw.get("cik", ""))
    accession = str(raw.get("accession", raw.get("accessionNumber", "")))
    try:
        cik_int  = int(cik)
        acc_flat = accession.replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_flat}/"
    except Exception:
        cik_int  = 0
        acc_flat = ""
        base     = ""
    return {
        "cik":         cik,
        "ticker":      raw.get("ticker", ""),
        "company":     raw.get("company", raw.get("name", "")),
        "form_type":   raw.get("form_type", raw.get("form", "")),
        "filing_date": raw.get("filing_date", raw.get("date", "")),
        "accession":   accession,
        "filing_url":  base,
        "index_url":   f"{base}{accession}-index.htm" if base and accession else "",
    }


# ── 1. Fetch SEC Filing ───────────────────────────────────────────────────────

def fetch_sec_filing(
    # Target selection — use exactly one of: date, ticker, cik
    ticker: str = None,
    cik: str = None,
    date: str = None,

    # Filtering (applies to date-based queries)
    form_type: str = None,    # e.g. "8-K"
    form_types: list = None,  # e.g. ["8-K", "8-K/A"]
    company: str = None,      # substring filter on company name

    # Mode
    mode: str = "metadata",   # "metadata" (fast) | "full" (includes filing text)

    # Pagination — only needed when you have more than `limit` results
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """
    Fetch SEC filing metadata (and optionally content) with built-in pagination,
    retry handling, rate limiting, and caching. The tool manages all complexity
    internally — callers just declare what they need.

    Modes
    -----
    "metadata"  — returns the filing list with standardized schema (very fast;
                  the quarterly EDGAR index is cached after the first download).
    "full"      — same as metadata plus extracts the full filing text for every
                  filing in the page, fetched concurrently (4 threads, rate-limited).

    Pagination
    ----------
    When date= is used the result can contain hundreds of filings. Use limit/offset
    to page through them. The response always includes:
      total      — total matching filings for this date + filter combination
      has_more   — True if more pages exist
      next_offset— pass this as `offset` in your next call to get the next page

    Examples
    --------
    # All 8-K filings on a date (metadata only):
    fetch_sec_filing(date="2025-01-13", form_type="8-K")

    # Page 2 of the same query:
    fetch_sec_filing(date="2025-01-13", form_type="8-K", offset=100)

    # Latest 10-K for a ticker with full text:
    fetch_sec_filing(ticker="AAPL", form_type="10-K", mode="full")
    """
    from sec_intelligence.live_search import (
        ticker_to_cik, get_latest_filing, get_filings_paginated,
        fetch_filings_content_concurrent, fetch_filing_text,
    )
    try:
        all_form_types = form_types or ([form_type] if form_type else None)

        # ── Date-based query ──────────────────────────────────────────────────
        if date:
            page = get_filings_paginated(
                date_str=date,
                form_types=all_form_types,
                company_filter=company,
                limit=limit,
                offset=offset,
            )
            filings = [_standardize_filing(f) for f in page["filings"]]

            if mode == "full" and filings:
                refs = [{"cik": f["cik"], "accession": f["accession"]} for f in filings]
                contents = fetch_filings_content_concurrent(refs, max_workers=4)
                content_map = {c["accession"]: c for c in contents}
                for f in filings:
                    c = content_map.get(f["accession"], {})
                    f["text"]        = c.get("text", "")
                    f["char_count"]  = c.get("char_count", 0)
                    f["fetch_error"] = c.get("error")

            return {
                "filings":         filings,
                "returned_count":  len(filings),   # filings in THIS page
                "total_count":     page["total"],   # ALL matching filings (use this for counts)
                "offset":          offset,
                "limit":           limit,
                "has_more":        page["has_more"],
                "next_offset":     page["next_offset"],
                "mode":            mode,
                "error":           None,
            }

        # ── Ticker / CIK query ────────────────────────────────────────────────
        if ticker and not cik:
            cik = ticker_to_cik(ticker.strip().upper())
            if not cik:
                return {
                    "filings": [], "returned_count": 0, "total_count": 0,
                    "error": f"No CIK found for ticker '{ticker}'",
                }

        if cik:
            form = (all_form_types[0] if all_form_types else None) or "10-K"
            raw = get_latest_filing(str(cik), form)
            if not raw:
                return {
                    "filings": [], "returned_count": 0, "total_count": 0,
                    "error": f"No {form} found for CIK {int(cik)}",
                }
            f = _standardize_filing(raw)
            if mode == "full":
                text           = fetch_filing_text(f["cik"], f["accession"])
                f["text"]      = text
                f["char_count"] = len(text)
            return {
                "filings": [f], "returned_count": 1, "total_count": 1,
                "offset": 0, "limit": 1,
                "has_more": False, "next_offset": None,
                "mode": mode, "error": None,
            }

        return {
            "filings": [], "returned_count": 0, "total_count": 0,
            "error": "Provide at least one of: date, ticker, cik",
            "hint": (
                "fetch_sec_filing(date='YYYY-MM-DD', form_type='8-K') "
                "— use next_offset from the response to page through results"
            ),
        }
    except Exception as e:
        logger.exception("fetch_sec_filing failed")
        return {"filings": [], "returned_count": 0, "total_count": 0, "error": str(e)}


# ── 2. Fetch Filing Content ───────────────────────────────────────────────────

def fetch_filing_content(
    filings: list,
    section: str = None,
    max_chars: int = 60_000,
) -> dict:
    """
    Download and extract text for a list of filings in one call.
    Fetches concurrently (4 threads) while respecting EDGAR's rate limit.
    Caching means re-fetching the same accession is instant.

    filings: list of {cik, accession} dicts (from fetch_sec_filing results).
    section: optional section filter e.g. "Item 1A", applied after download.
    Returns {results: [{cik, accession, content, char_count, error}], total, successful, errors}.
    """
    from sec_intelligence.live_search import fetch_filings_content_concurrent
    from sec_intelligence.pipeline import _strip_xbrl, _detect_sections

    if not filings:
        return {"results": [], "total": 0, "successful": 0, "errors": 0, "error": "No filings provided"}

    raw_results = fetch_filings_content_concurrent(filings, max_chars=max_chars, max_workers=4)

    results = []
    errors  = 0
    for r in raw_results:
        text = r.get("text", "")
        err  = r.get("error")
        if err or not text:
            results.append({
                "cik":        r.get("cik", ""),
                "accession":  r.get("accession", ""),
                "content":    "",
                "char_count": 0,
                "error":      err or "empty content",
            })
            errors += 1
            continue

        section_names: list[str] = []
        try:
            clean    = _strip_xbrl(text)
            sections = _detect_sections(clean)
            section_names = [s[0] for s in sections]
            if section:
                su      = section.upper()
                matched = [(n, t) for n, t in sections if su in n.upper()]
                content = matched[0][1][:8000] if matched else clean[:5000]
            else:
                content = clean[:8000]
        except Exception:
            content = text[:8000]

        results.append({
            "cik":                r.get("cik", ""),
            "accession":          r.get("accession", ""),
            "content":            content,
            "char_count":         len(content),
            "sections_available": section_names,
            "error":              None,
        })

    return {
        "results":    results,
        "total":      len(results),
        "successful": len(results) - errors,
        "errors":     errors,
        "error":      None,
    }


# ── Count Filings by Date ─────────────────────────────────────────────────────

def count_filings_by_date(
    date: str,
    form_type: str = None,
    form_types: list = None,
    company: str = None,
) -> dict:
    """
    Return the total count of SEC filings on a date WITHOUT downloading any content.
    Uses the cached quarterly index — instant after the first call for that quarter.

    Use this whenever the question is about HOW MANY filings exist.
    Do NOT use fetch_sec_filing for counting — it only returns a page.

    Returns:
        total_filings  — exact count matching all filters
        by_form_type   — breakdown of counts per form type (top 15)
    """
    from sec_intelligence.live_search import get_filings_for_date
    try:
        all_form_types = form_types or ([form_type] if form_type else None)
        filings = get_filings_for_date(date, all_form_types)

        if company:
            cf = company.upper()
            filings = [f for f in filings if cf in f.get("company", "").upper()]

        type_counts: dict[str, int] = {}
        for f in filings:
            ft = f.get("form_type", "unknown")
            type_counts[ft] = type_counts.get(ft, 0) + 1

        # Sort by count descending, cap at 15 types
        by_type = dict(sorted(type_counts.items(), key=lambda x: -x[1])[:15])

        return {
            "date":           date,
            "total_filings":  len(filings),
            "by_form_type":   by_type,
            "filters_applied": {
                "form_types": all_form_types,
                "company":    company,
            },
            "error": None,
        }
    except Exception as e:
        return {"date": date, "total_filings": 0, "by_form_type": {}, "error": str(e)}


# ── Fetch Market Cap ─────────────────────────────────────────────────────────

def fetch_market_cap(ticker: str, date: str = None) -> dict:
    """Fetch market cap for a ticker (uses yfinance)."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        cap = info.get("marketCap")
        if not cap:
            return {"ticker": ticker, "market_cap": None,
                    "error": "Market cap not available from yfinance"}
        if cap >= 1e12:
            fmt = f"${cap / 1e12:.2f}T"
        elif cap >= 1e9:
            fmt = f"${cap / 1e9:.2f}B"
        else:
            fmt = f"${cap / 1e6:.2f}M"
        return {
            "ticker": ticker,
            "market_cap": cap,
            "market_cap_formatted": fmt,
            "currency": info.get("currency", "USD"),
            "company_name": info.get("longName", ticker),
            "error": None,
        }
    except ImportError:
        return {"ticker": ticker, "market_cap": None,
                "error": "yfinance not installed — run: pip install yfinance"}
    except Exception as e:
        return {"ticker": ticker, "market_cap": None, "error": str(e)}


# ── 3. Fetch From Database ────────────────────────────────────────────────────

def fetch_from_database(ticker: str = None, form_type: str = None) -> dict:
    """Fetch filings already stored in the local SQLite database."""
    try:
        from sec_intelligence.pipeline import LocalDB, DB_PATH
        db = LocalDB(DB_PATH)
        with db._conn() as c:
            q = (
                "SELECT f.ticker, f.company, f.form_type, f.filing_date, f.accession, "
                "COUNT(ch.id) AS chunk_count "
                "FROM filings f LEFT JOIN chunks ch ON ch.filing_id = f.id"
            )
            params = []
            conds = []
            if ticker:
                conds.append("f.ticker = ?")
                params.append(ticker.upper())
            if form_type:
                conds.append("f.form_type = ?")
                params.append(form_type)
            if conds:
                q += " WHERE " + " AND ".join(conds)
            q += " GROUP BY f.id ORDER BY f.filing_date DESC"
            rows = c.execute(q, params).fetchall()
        filings = [dict(r) for r in rows]
        return {"filings": filings, "count": len(filings), "error": None}
    except Exception as e:
        return {"filings": [], "count": 0, "error": str(e)}


# ── 4. Write To Database ──────────────────────────────────────────────────────

def write_to_database(
    cik: str,
    accession: str,
    company: str,
    form_type: str,
    filing_date: str,
    ticker: str = None,
) -> dict:
    """Download a filing from EDGAR, chunk, embed, and store it in the local DB."""
    try:
        from sec_intelligence.pipeline import get_pipeline, LocalDB, DB_PATH

        db = LocalDB(DB_PATH)
        if db.has_accession(accession):
            return {"status": "already_exists", "accession": accession,
                    "chunks": 0, "error": None}

        eff_ticker = ticker or f"CIK{int(cik)}"
        pipeline = get_pipeline()
        log: list = []

        def _emit(msg):
            return msg, []

        for _ in pipeline._ingest_known_filing(
            cik, accession, company, form_type, filing_date, eff_ticker, log, _emit
        ):
            pass

        if db.has_accession(accession):
            chunk_count = len(db.get_chunks(eff_ticker))
            return {"status": "success", "accession": accession,
                    "chunks": chunk_count, "error": None}
        return {"status": "failed", "accession": accession,
                "chunks": 0, "error": "Ingestion did not complete"}
    except Exception as e:
        return {"status": "failed", "accession": accession,
                "chunks": 0, "error": str(e)}


# ── 5. Filing Content Extraction ──────────────────────────────────────────────

def extract_filing_content(
    cik: str,
    accession: str,
    section: str = None,
) -> dict:
    """
    Extract text from a filing, optionally filtered to a specific section.
    Automatically retries with raw HTML fallback if the primary document fails.
    """
    import time
    from sec_intelligence.live_search import fetch_filing_text, _get_doc_url, HEADERS
    from sec_intelligence.pipeline import _strip_xbrl, _detect_sections

    # Attempt 1: normal fetch
    text = fetch_filing_text(cik, accession, max_chars=300_000)

    # Attempt 2: if empty, wait briefly and retry (EDGAR rate-limiting)
    if not text or len(text) < 200:
        time.sleep(1.5)
        text = fetch_filing_text(cik, accession, max_chars=300_000)

    # Attempt 3: if still empty, try fetching the raw index page text directly
    if not text or len(text) < 200:
        try:
            import requests
            from bs4 import BeautifulSoup
            acc_flat = accession.replace("-", "")
            idx_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_flat}/{accession}-index.htm"
            )
            r = requests.get(idx_url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator="\n")[:20000]
        except Exception as e:
            return {
                "content": "",
                "sections_available": [],
                "error": f"All download attempts failed: {e}",
            }

    if not text or len(text) < 100:
        return {"content": "", "sections_available": [], "error": "No readable content found"}

    try:
        clean = _strip_xbrl(text)
        sections = _detect_sections(clean)
        names = [s[0] for s in sections]

        if section:
            su = section.upper()
            matched = [(n, t) for n, t in sections if su in n.upper()]
            content = matched[0][1][:8000] if matched else clean[:5000]
        else:
            content = clean[:8000]

        return {
            "content": content,
            "sections_available": names,
            "total_chars": len(clean),
            "error": None,
        }
    except Exception as e:
        return {"content": "", "sections_available": [], "error": str(e)}


# ── 6. Event Detection ────────────────────────────────────────────────────────

# Role title → canonical abbreviation
_ROLE_CANON: dict[str, str] = {
    "chief executive officer": "CEO",
    "chief financial officer": "CFO",
    "chief operating officer": "COO",
    "chief technology officer": "CTO",
    "chief marketing officer": "CMO",
    "chief legal officer": "CLO",
    "chief accounting officer": "CAO",
    "chief information officer": "CIO",
    "chief revenue officer": "CRO",
    "chief people officer": "CPO",
    "chief human resources officer": "CHRO",
    "general counsel": "General Counsel",
    "vice president": "VP",
    "executive vice president": "EVP",
    "senior vice president": "SVP",
    "president": "President",
    "chairman": "Chairman",
    "director": "Director",
    "treasurer": "Treasurer",
    "secretary": "Secretary",
    "ceo": "CEO",
    "cfo": "CFO",
    "coo": "COO",
    "cto": "CTO",
    "cmo": "CMO",
    "clo": "CLO",
    "cao": "CAO",
    "cio": "CIO",
}

# Matches full role titles (longest first to prefer multi-word matches)
_ROLE_RE = re.compile(
    r'\b(?:'
    r'Chief\s+Executive\s+Officer|Chief\s+Financial\s+Officer|'
    r'Chief\s+Operating\s+Officer|Chief\s+Technology\s+Officer|'
    r'Chief\s+Marketing\s+Officer|Chief\s+Legal\s+Officer|'
    r'Chief\s+Accounting\s+Officer|Chief\s+Information\s+Officer|'
    r'Chief\s+Revenue\s+Officer|Chief\s+People\s+Officer|'
    r'Chief\s+Human\s+Resources\s+Officer|'
    r'Executive\s+Vice\s+President|Senior\s+Vice\s+President|'
    r'General\s+Counsel|Vice\s+President|'
    r'President|Chairman|Director|Treasurer|Secretary|'
    r'CEO|CFO|COO|CTO|CMO|CLO|CAO|CIO|CRO|EVP|SVP|VP'
    r')\b',
    re.IGNORECASE,
)

# Person name: Title Case 2–4 words (min 3 chars each), excluding false-positive phrases
_NAME_RE = re.compile(
    r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})'
    r'|([A-Z][a-z]{2,15}\s+(?:[A-Z]\.\s+)?[A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15})?)',
)

# Words that look like name tokens but are section headers, boilerplate, or months
_FAKE_NAME_TOKENS = frozenset({
    # SEC / legal boilerplate
    "Item", "Section", "Form", "Part", "Exhibit", "Annual", "Quarterly",
    "Financial", "Statements", "Management", "Discussion", "Analysis",
    "Risk", "Factors", "Legal", "Proceedings", "Market", "Information",
    "Properties", "Controls", "Procedures", "Board", "Directors",
    "Executive", "Compensation", "Notes", "Report", "Results",
    "Operations", "Company", "Corporation", "Incorporated", "Limited",
    "United", "States", "Securities", "Exchange", "Commission",
    "Pursuant", "Whereas", "Registrant", "Signature", "Agreement",
    "Effective", "Pursuant", "Certain", "Officer", "Officers",
    # Month names (often capitalized mid-sentence)
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    # Common sentence-start words
    "The", "This", "That", "These", "Those", "Such", "Each",
    "Our", "His", "Her", "Its", "Their", "Your", "Any", "All",
    "New", "Other", "Both", "Each", "Upon", "With", "From",
})


def _norm_role(raw: str) -> str:
    key = re.sub(r"\s+", " ", raw.strip().lower())
    return _ROLE_CANON.get(key, raw.strip())


def _extract_name(snippet: str) -> str:
    """Extract the first plausible person name from a short text snippet."""
    for m in _NAME_RE.finditer(snippet):
        candidate = (m.group(1) or m.group(2) or "").strip()
        if not candidate:
            continue
        words = candidate.split()
        # Reject if any word is a known false-positive token
        if any(w in _FAKE_NAME_TOKENS for w in words):
            continue
        # Reject single words or all-caps acronyms
        if len(words) < 2 or all(w.isupper() for w in words):
            continue
        return candidate
    return ""


def _extract_officer_events(text: str) -> list[dict]:
    """
    Return structured officer-change events extracted from filing text.
    Each event: {event_type, person, role, action, context, position}
    """
    events: list[dict] = []

    # Pattern groups: (action_type, regex)
    patterns = [
        # Appointment: "appointed John Doe as CFO"
        (
            "officer_appointment",
            re.compile(
                r'(?P<action>appoint(?:ed|ment)|named?|elected?|designated?|hired?|promoted?)'
                r'(?P<mid>(?:\s+\w+){0,6}?\s+(?:as\s+)?)'
                r'(?P<role>' + _ROLE_RE.pattern + r')',
                re.IGNORECASE,
            ),
        ),
        # Appointment alt: "John Doe will serve as / has been named CFO"
        (
            "officer_appointment",
            re.compile(
                r'(?P<role>' + _ROLE_RE.pattern + r')'
                r'(?P<mid>(?:\s+\w+){0,6}?\s+)'
                r'(?P<action>appoint(?:ed|ment)|named?|elected?|designated?|hired?|promoted?)',
                re.IGNORECASE,
            ),
        ),
        # Resignation / departure: "CFO John Doe resigned"
        (
            "officer_resignation",
            re.compile(
                r'(?P<role>' + _ROLE_RE.pattern + r')'
                r'(?P<mid>(?:\s+\w+){0,6}?\s+)'
                r'(?P<action>resign(?:ed|ation)|stepping\s+down|stepped\s+down|'
                r'leaving|depart(?:ed|ure|ing)?|terminat(?:ed|ion))',
                re.IGNORECASE,
            ),
        ),
        # Resignation alt: "John Doe resigned as CFO"
        (
            "officer_resignation",
            re.compile(
                r'(?P<action>resign(?:ed|ation)|stepping\s+down|stepped\s+down|'
                r'leaving|depart(?:ed|ure|ing)?|terminat(?:ed|ion))'
                r'(?P<mid>(?:\s+\w+){0,6}?\s+(?:as\s+)?)'
                r'(?P<role>' + _ROLE_RE.pattern + r')',
                re.IGNORECASE,
            ),
        ),
        # Retirement: "CFO John Doe will retire"
        (
            "officer_retirement",
            re.compile(
                r'(?:(?P<role>' + _ROLE_RE.pattern + r')(?P<mid1>(?:\s+\w+){0,6}?\s+))?'
                r'(?P<action>retir(?:ed?|ing|ement))',
                re.IGNORECASE,
            ),
        ),
    ]

    for event_type, pat in patterns:
        for m in pat.finditer(text):
            pos = m.start()
            ctx_start = max(0, pos - 120)
            ctx_end = min(len(text), m.end() + 120)
            snippet = text[ctx_start:ctx_end]

            # Extract role from match groups
            role_raw = m.groupdict().get("role") or ""
            role = _norm_role(role_raw) if role_raw else ""

            # Skip if no role found (catches false positives with no title)
            if not role:
                continue

            # Extract action verb
            action_raw = m.groupdict().get("action") or ""
            action = action_raw.lower().split()[0] if action_raw else ""
            # Normalize action to past-tense verb
            action_map = {
                "appoint": "appointed", "appointed": "appointed",
                "appointment": "appointed", "named": "named",
                "name": "named", "elected": "elected", "elect": "elected",
                "designated": "designated", "designate": "designated",
                "hired": "hired", "hire": "hired",
                "promoted": "promoted", "promote": "promoted",
                "resigned": "resigned", "resign": "resigned",
                "resignation": "resigned", "stepping": "stepping down",
                "stepped": "stepped down", "leaving": "left",
                "departed": "departed", "depart": "departed",
                "departure": "departed", "terminated": "terminated",
                "termination": "terminated", "retir": "retired",
                "retired": "retired", "retiring": "retiring",
                "retirement": "retired",
            }
            action = action_map.get(action, action)

            # Extract person name from surrounding context
            person = _extract_name(snippet)

            events.append({
                "event_type": event_type,
                "person": person,
                "role": role,
                "action": action,
                "context": snippet.strip(),
                "position": pos,
            })

    return events


def _extract_non_officer_events(text: str, target_types: set) -> list[dict]:
    """Extract non-officer structured events."""
    events: list[dict] = []

    simple_patterns: dict[str, list[tuple[str, str]]] = {
        "acquisition": [
            ("acquired", r'acqui(?:red?|ring|sition)\s+(?:of\s+)?(?P<target>[A-Z][\w\s&,\.]{2,40}?)(?:\s+for\s+(?P<amount>\$[\d\.]+[BMK]?))?(?:\s|$|,|\.)'),
            ("merger", r'(?:merger|business\s+combination)\s+(?:with\s+)?(?P<target>[A-Z][\w\s&,\.]{2,30}?)(?:\s+for\s+(?P<amount>\$[\d\.]+[BMK]?))?(?:\s|$|,|\.)'),
        ],
        "earnings": [
            ("reported earnings", r'(?:net\s+income|earnings\s+per\s+share|EPS)\s+(?:of\s+)?(?P<amount>\$[\d,\.]+)'),
            ("reported revenue", r'revenue\s+(?:of|was|were|increased\s+to|decreased\s+to)\s+(?P<amount>\$[\d,\.]+)'),
        ],
        "dividend": [
            ("declared dividend", r'declared?\s+(?:a\s+)?(?:quarterly|annual|special)\s+dividend(?:\s+of\s+(?P<amount>\$[\d\.]+)\s+per\s+share)?'),
        ],
        "restructuring": [
            ("restructuring", r'(?:restructuring|workforce\s+reduction|reduction\s+in\s+force)(?:.*?elimin(?:ate|ating)\s+approximately\s+(?P<count>[\d,]+)\s+(?:positions?|jobs?|employees?))?'),
        ],
        "legal": [
            ("litigation", r'(?:lawsuit|class\s+action|legal\s+proceedings?)\s+(?:against\s+)?(?P<party>[A-Z][\w\s]{2,30}?)(?:\s|$|,|\.)'),
            ("settlement", r'settlement\s+(?:of\s+)?(?P<amount>\$[\d,\.]+)'),
        ],
        "risk": [
            ("material weakness", r'material\s+weakness'),
            ("going concern", r'going\s+concern'),
        ],
        "guidance": [
            ("guidance", r'(?:outlook|guidance|forecast)\s+for\s+(?:fiscal|the\s+full)\s+(?:year\s+)?(?P<period>[\w\s]{1,20})'),
            ("revenue guidance", r'expects?\s+(?:revenue|earnings|sales)\s+(?:of|to\s+be)\s+(?P<amount>\$[\d,\.]+)'),
        ],
    }

    for etype, plist in simple_patterns.items():
        if etype not in target_types:
            continue
        for action_label, pat in plist:
            for m in re.finditer(pat, text, re.IGNORECASE):
                pos = m.start()
                ctx_s = max(0, pos - 80)
                ctx_e = min(len(text), m.end() + 80)
                gd = m.groupdict()
                events.append({
                    "event_type": etype,
                    "action": action_label,
                    "amount": gd.get("amount", ""),
                    "target": gd.get("target", "").strip() if gd.get("target") else "",
                    "context": text[ctx_s:ctx_e].strip(),
                    "position": pos,
                })

    return events


def detect_events(text: str, event_types: list = None) -> dict:
    """
    Detect and classify financial events in filing text.

    Returns structured events — each event has typed fields (event_type, person,
    role, action) rather than raw match text. Officer events without a detectable
    role are filtered out to prevent false positives from section headers.
    """
    if not text:
        return {"events": [], "count": 0, "error": "No text provided"}

    officer_types = {"officer_change", "officer_appointment", "officer_resignation", "officer_retirement"}
    non_officer_types = {"acquisition", "earnings", "dividend", "restructuring", "legal", "risk", "guidance"}

    if event_types:
        # Normalize: "officer_change" expands to all officer subtypes
        requested = set(event_types)
        if "officer_change" in requested:
            requested |= officer_types
        target_officer = bool(requested & officer_types)
        target_non_officer = requested & non_officer_types
    else:
        target_officer = True
        target_non_officer = non_officer_types

    all_events: list[dict] = []

    if target_officer:
        all_events.extend(_extract_officer_events(text))

    if target_non_officer:
        all_events.extend(_extract_non_officer_events(text, target_non_officer))

    # Deduplicate: drop events within 200 chars of a prior event of the same type+role
    all_events.sort(key=lambda x: x["position"])
    deduped: list[dict] = []
    seen: list[tuple[str, str, int]] = []  # (event_type, role_or_action, position)
    for ev in all_events:
        role_key = ev.get("role") or ev.get("action", "")
        close = any(
            ev["event_type"] == s[0]
            and role_key == s[1]
            and abs(ev["position"] - s[2]) < 200
            for s in seen
        )
        if not close:
            # Remove internal position tracker before returning to LLM
            out = {k: v for k, v in ev.items() if k != "position"}
            deduped.append(out)
            seen.append((ev["event_type"], role_key, ev["position"]))

    return {"events": deduped, "count": len(deduped), "error": None}


# ── 7. Named Entity Recognition ───────────────────────────────────────────────

def ner_tool(text: str) -> dict:
    """Extract named entities: companies, people, financial figures, dates, tickers."""
    if not text:
        return {"entities": {}, "total_count": 0, "error": "No text provided"}

    entities: dict = {
        "companies": [],
        "people": [],
        "financial_figures": [],
        "dates": [],
        "locations": [],
        "tickers": [],
    }

    # Financial figures
    for m in re.finditer(
        r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|M|B|T)\b)?',
        text, re.IGNORECASE,
    ):
        entities["financial_figures"].append(m.group(0).strip())

    # Dates
    for m in re.finditer(
        r'\b(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        r'|\b\d{4}-\d{2}-\d{2}\b'
        r'|\bQ[1-4]\s+\d{4}\b'
        r'|\bfiscal\s+(?:year\s+)?\d{4}\b',
        text, re.IGNORECASE,
    ):
        entities["dates"].append(m.group(0))

    # Ticker symbols (in parentheses or after exchange colon)
    for m in re.finditer(
        r'\(([A-Z]{2,5})\)|(?:NASDAQ|NYSE|AMEX):\s*([A-Z]{2,5})', text
    ):
        t = m.group(1) or m.group(2)
        if t:
            entities["tickers"].append(t)

    # People (Mr./Ms./Dr. + Title Case Name)
    for m in re.finditer(
        r'(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text
    ):
        entities["people"].append(m.group(0))

    # Deduplicate, cap at 20 each
    for k in entities:
        entities[k] = list(dict.fromkeys(entities[k]))[:20]

    total = sum(len(v) for v in entities.values())
    return {"entities": entities, "total_count": total, "error": None}


# ── 8. Semantic Search ────────────────────────────────────────────────────────

def semantic_search(query: str, ticker: str = None, top_k: int = 5) -> dict:
    """Hybrid BM25 + vector search over stored filing chunks."""
    try:
        from sec_intelligence.pipeline import (
            get_pipeline, LocalDB, DB_PATH,
            _bm25_scores, _vector_scores, _hybrid_top_k,
        )
        db = LocalDB(DB_PATH)

        if ticker:
            chunks = db.get_chunks(ticker.upper())
        else:
            with db._conn() as c:
                rows = c.execute(
                    "SELECT id, section, text, filing_date, fiscal_year, "
                    "chunk_index, ticker FROM chunks "
                    "ORDER BY filing_date DESC LIMIT 2000"
                ).fetchall()
                chunks = [dict(r) for r in rows]

        if not chunks:
            return {"chunks": [], "total_searched": 0,
                    "error": "No chunks in database. Use write_to_database first."}

        chunk_map = {c["id"]: c for c in chunks}
        pipeline = get_pipeline()
        emb_map = db.get_embeddings(list(chunk_map.keys()))

        bm25 = _bm25_scores(query, chunks)
        vec = _vector_scores(query, chunks, emb_map, pipeline.embed_model)
        top = _hybrid_top_k(bm25, vec, chunk_map, k=top_k)

        return {
            "chunks": [
                {
                    "rank": i + 1,
                    "ticker": c.get("ticker", ""),
                    "section": c["section"],
                    "filing_date": c.get("filing_date", ""),
                    "text": c["text"][:600],
                }
                for i, c in enumerate(top)
            ],
            "total_searched": len(chunks),
            "error": None,
        }
    except Exception as e:
        return {"chunks": [], "total_searched": 0, "error": str(e)}


# ── 9. Filing Type Filter ─────────────────────────────────────────────────────

def filter_by_filing_type(filings: list, form_types: list) -> dict:
    """Filter a list of filing dicts to only those matching the given form types."""
    if not filings:
        return {"filings": [], "count": 0, "filtered_from": 0, "error": None}
    types_upper = {f.upper() for f in form_types}
    kept = [f for f in filings if str(f.get("form_type", "")).upper() in types_upper]
    return {
        "filings": kept,
        "count": len(kept),
        "filtered_from": len(filings),
        "error": None,
    }


# ── 10. Section Classifier ────────────────────────────────────────────────────

def classify_sections(text: str) -> dict:
    """Identify and list SEC filing sections present in text."""
    try:
        from sec_intelligence.pipeline import _detect_sections
        sections = _detect_sections(text)
        return {
            "sections": [
                {
                    "label": label,
                    "char_count": len(content),
                    "preview": content[:120].replace("\n", " "),
                }
                for label, content in sections
            ],
            "count": len(sections),
            "error": None,
        }
    except Exception as e:
        return {"sections": [], "count": 0, "error": str(e)}


# ── 11. Temporal Reasoning ────────────────────────────────────────────────────

def temporal_reasoning(query: str, filing_dates: list) -> dict:
    """Answer temporal questions: latest, earliest, date range, year filter."""
    if not filing_dates:
        return {"answer": "No filing dates provided.", "dates": [], "error": None}
    try:
        parsed = []
        for d in filing_dates:
            try:
                parsed.append(datetime.strptime(str(d)[:10], "%Y-%m-%d"))
            except Exception:
                pass
        if not parsed:
            return {"answer": "Could not parse any of the provided dates.",
                    "dates": [], "error": None}
        parsed.sort()
        ql = query.lower()
        if any(w in ql for w in ("latest", "most recent", "newest")):
            answer = f"Most recent filing: {parsed[-1].strftime('%B %d, %Y')}"
        elif any(w in ql for w in ("earliest", "oldest", "first")):
            answer = f"Earliest filing: {parsed[0].strftime('%B %d, %Y')}"
        elif "range" in ql or "between" in ql or "span" in ql:
            answer = (
                f"Filings span from {parsed[0].strftime('%Y-%m-%d')} "
                f"to {parsed[-1].strftime('%Y-%m-%d')}"
            )
        elif "year" in ql:
            years = sorted({d.year for d in parsed})
            answer = f"Filing years present: {years}"
        else:
            answer = (
                f"{len(parsed)} filing(s): "
                f"{parsed[0].strftime('%Y-%m-%d')} → {parsed[-1].strftime('%Y-%m-%d')}"
            )
        return {
            "answer": answer,
            "dates": [d.strftime("%Y-%m-%d") for d in parsed],
            "count": len(parsed),
            "error": None,
        }
    except Exception as e:
        return {"answer": "", "dates": [], "error": str(e)}


# ── 12. Deduplication + Entity Resolution ─────────────────────────────────────

def deduplicate_entities(entities: list) -> dict:
    """Remove duplicate / near-duplicate entity names, group similar ones."""
    if not entities:
        return {"entities": [], "count": 0, "groups": {}, "error": None}
    normalized: dict[str, str] = {}
    for ent in entities:
        key = re.sub(r"\s+", " ", str(ent).strip().lower())
        key = re.sub(r"[,\.\s]+$", "", key)
        if key and key not in normalized:
            normalized[key] = str(ent).strip()

    # Group by first significant token
    groups: dict[str, list] = {}
    for key, canonical in normalized.items():
        words = [w for w in key.split() if len(w) > 2]
        anchor = words[0] if words else key
        groups.setdefault(anchor, []).append(canonical)

    unique = list(normalized.values())
    return {
        "entities": unique,
        "count": len(unique),
        "groups": {k: v for k, v in groups.items() if len(v) > 1},
        "error": None,
    }


# ── 13. Confidence / Verification ─────────────────────────────────────────────

def verify_confidence(answer: str, context: str, question: str = "") -> dict:
    """Score how well an answer is supported by the provided context."""
    if not answer or not context:
        return {
            "confidence": 0.0,
            "supported": False,
            "explanation": "Answer or context missing.",
            "error": None,
        }
    answer_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", answer.lower()))
    context_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", context.lower()))

    if not answer_words:
        return {
            "confidence": 0.0,
            "supported": False,
            "explanation": "Answer contains no verifiable content words.",
            "error": None,
        }

    overlap = len(answer_words & context_words) / len(answer_words)

    hedging = bool(re.search(
        r"cannot find|don.t know|not mentioned|insufficient|unclear|"
        r"not available|not provided|unable to determine",
        answer, re.IGNORECASE,
    ))

    confidence = round(overlap * (0.5 if hedging else 1.0), 2)
    supported = confidence >= 0.3

    explanation = (
        f"{overlap:.0%} of answer's key words appear in the source context. "
        + ("Hedging language detected — confidence halved." if hedging
           else "No hedging detected.")
    )
    return {
        "confidence": confidence,
        "supported": supported,
        "explanation": explanation,
        "error": None,
    }


# ── Batch tools (process many items in one call) ──────────────────────────────

def batch_extract_filing_content(items: list) -> dict:
    """
    Extract content from multiple filings in a single call.
    items: list of {cik, accession, section (optional)}
    Returns all results; continues past individual failures.
    """
    import time
    results = []
    errors = 0
    for item in items:
        try:
            r = extract_filing_content(
                str(item.get("cik", "")),
                str(item.get("accession", "")),
                item.get("section"),
            )
            results.append({
                "cik": item.get("cik"),
                "accession": item.get("accession"),
                "section_requested": item.get("section"),
                "content": r.get("content", ""),
                "sections_available": r.get("sections_available", []),
                "total_chars": r.get("total_chars", 0),
                "error": r.get("error"),
            })
            if r.get("error"):
                errors += 1
        except Exception as e:
            results.append({
                "cik": item.get("cik"),
                "accession": item.get("accession"),
                "content": "",
                "sections_available": [],
                "error": str(e),
            })
            errors += 1
    return {
        "results": results,
        "total": len(results),
        "successful": len(results) - errors,
        "errors": errors,
        "error": None,
    }


def batch_detect_events(
    items: list = None,
    event_types: list = None,
    filings: list = None,
    date: str = None,
    form_type: str = None,
    page_size: int = 20,
) -> dict:
    """
    Detect events across many filings in one call.

    Three calling modes (use the first one that applies):
      Mode A — date-based (BEST): tool fetches ALL filings for that date internally:
        date="YYYY-MM-DD", form_type="8-K", event_types=[...]
      Mode B — filing refs: pass cik+accession pairs, tool fetches content:
        filings=[{"cik": "...", "accession": "..."}, ...]
      Mode C — raw text: pass extracted text directly:
        items=[{"text": "...", "label": "..."}, ...]

    page_size controls how many filings are fetched per page in Mode A/B (default 20).
    """
    from sec_intelligence.live_search import get_filings_for_date

    # Mode A: fetch all filings for the date internally
    if date and not filings and not items:
        form_types = [form_type] if form_type else None
        raw = get_filings_for_date(date, form_types)
        filings = [{"cik": f["cik"], "accession": f["accession"],
                    "company": f.get("company", ""), "ticker": f.get("ticker", "")}
                   for f in raw]

    if not items and not filings:
        return {
            "error": (
                "Missing required argument. Best option: "
                "batch_detect_events(date=\"YYYY-MM-DD\", form_type=\"8-K\", event_types=[\"officer_change\"]). "
                "Or: filings=[{\"cik\": \"...\", \"accession\": \"...\"}]. "
                "Or: items=[{\"text\": \"...\", \"label\": \"...\"}]."
            ),
            "results": [], "total_items": 0, "total_events": 0,
        }

    # Mode B: extract content from EDGAR page-by-page, then detect
    if filings and not items:
        items = []
        total_pages = (len(filings) + page_size - 1) // page_size
        for page_idx in range(total_pages):
            page = filings[page_idx * page_size : (page_idx + 1) * page_size]
            logger.info(
                f"batch_detect_events: fetching page {page_idx + 1}/{total_pages} "
                f"({len(page)} filings)"
            )
            for f in page:
                try:
                    r = extract_filing_content(cik=f.get("cik", ""), accession=f.get("accession", ""))
                    text = r.get("text") or r.get("content") or ""
                    label = f.get("accession", f.get("cik", ""))
                    meta = {"company": f.get("company", ""), "ticker": f.get("ticker", "")}
                    if text:
                        items.append({"text": text, "label": label, **meta})
                    else:
                        items.append({"text": "", "label": label, "_error": "empty content", **meta})
                except Exception as e:
                    items.append({"text": "", "label": f.get("accession", ""), "_error": str(e)})

    results = []
    total_events = 0
    for item in (items or []):
        text  = item.get("text", "")  if isinstance(item, dict) else str(item)
        label = item.get("label", "") if isinstance(item, dict) else ""
        if not text:
            # Skip empty-content filings entirely — no events, nothing for LLM to read
            continue
        r = detect_events(text, event_types)
        count = r.get("count", 0)
        # Filter: only include filings where at least one event was detected
        if count == 0:
            continue
        results.append({
            "label": label,
            "company": item.get("company", "") if isinstance(item, dict) else "",
            "ticker":  item.get("ticker",  "") if isinstance(item, dict) else "",
            "events": r.get("events", []),
            "count": count,
        })
        total_events += count

    return {
        "results": results,
        "total_items": len(results),
        "total_events": total_events,
        "filings_with_events": len(results),
        "error": None,
    }


def batch_ner_tool(items: list) -> dict:
    """
    Extract named entities from multiple texts in one call.
    items: list of {text, label (optional)}
    """
    results = []
    for item in items:
        text  = item.get("text", "")  if isinstance(item, dict) else str(item)
        label = item.get("label", "") if isinstance(item, dict) else ""
        r = ner_tool(text)
        results.append({
            "label": label,
            "entities": r.get("entities", {}),
            "total_count": r.get("total_count", 0),
            "error": r.get("error"),
        })
    return {
        "results": results,
        "total_items": len(results),
        "error": None,
    }


def batch_fetch_sec_filing(queries: list) -> dict:
    """
    Fetch filing metadata for multiple tickers/CIKs in one call.
    queries: list of {ticker?, cik?, form_type?}
    """
    results = []
    for q in queries:
        r = fetch_sec_filing(
            ticker=q.get("ticker"),
            cik=q.get("cik"),
            form_type=q.get("form_type"),
        )
        results.append({
            "query": q,
            "filings": r.get("filings", []),
            "count": r.get("count", 0),
            "error": r.get("error"),
        })
    return {
        "results": results,
        "total_queries": len(results),
        "error": None,
    }
