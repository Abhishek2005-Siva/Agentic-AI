"""
Workflow Planner Agent.
Uses OpenAI function-calling in an agentic loop to plan and execute tools.
"""
import json
import logging
import re
from typing import Generator

from openai import OpenAI
from sec_intelligence.config import get_config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert SEC financial intelligence analyst.

Before each tool call write 1–2 sentences explaining what you are doing and why.

Rules:
1. Counting filings: use count_filings_by_date — never infer counts from fetch_sec_filing pages.
2. When fetch_sec_filing returns data, report total_count, not returned_count (one page).
3. Multiple filings/texts to process: use the batch_ variant in one call, never loop.
4. Tools manage pagination, retries, rate limiting, and caching — you do not.
5. Never stop before the answer is complete. Exhaust all available data first.
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_sec_filing",
            "description": (
                "Fetch SEC filing metadata (and optionally text) from EDGAR. "
                "Target: date='YYYY-MM-DD' | ticker='AAPL' | cik='...'. "
                "Filters: form_type, form_types, company (substring). "
                "mode='metadata' (default, fast) | mode='full' (includes text, concurrent). "
                "Response: returned_count=this page, total_count=all matches. "
                "Use next_offset to page. Use count_filings_by_date for count-only questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker":     {"type": "string",  "description": "Stock ticker e.g. AAPL, NVDA"},
                    "cik":        {"type": "string",  "description": "10-digit SEC CIK number"},
                    "date":       {"type": "string",  "description": "YYYY-MM-DD — returns all filings on this date"},
                    "form_type":  {"type": "string",  "description": "Single form type filter: 8-K, 10-K, 10-Q, etc."},
                    "form_types": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Multiple form types e.g. ['8-K', '8-K/A']",
                    },
                    "company":    {"type": "string",  "description": "Substring filter on company name"},
                    "mode":       {"type": "string",  "description": "'metadata' (default, fast) or 'full' (includes filing text)"},
                    "limit":      {"type": "integer", "description": "Page size (default 100)"},
                    "offset":     {"type": "integer", "description": "Start offset for pagination (default 0)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_filing_content",
            "description": (
                "Download and extract text from multiple filings in one concurrent call. "
                "Pass filings=[{cik, accession}] from fetch_sec_filing results. "
                "Optional section filter e.g. 'Item 1A'. Cached — re-fetching is instant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filings": {
                        "type": "array",
                        "description": "List of {cik, accession} dicts from fetch_sec_filing results",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cik":       {"type": "string"},
                                "accession": {"type": "string"},
                            },
                            "required": ["cik", "accession"],
                        },
                    },
                    "section": {
                        "type": "string",
                        "description": "Optional section to extract: 'Item 1A', 'Item 7', 'Item 5.02', etc.",
                    },
                },
                "required": ["filings"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_filings_by_date",
            "description": (
                "Count SEC filings on a date without downloading content. "
                "Returns total_filings (exact) and by_form_type breakdown. "
                "Use this for any 'how many' question — fetch_sec_filing returns only a page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date":       {"type": "string",  "description": "YYYY-MM-DD"},
                    "form_type":  {"type": "string",  "description": "Optional: filter to a single form type e.g. '8-K'"},
                    "form_types": {"type": "array", "items": {"type": "string"},
                                   "description": "Optional: filter to multiple form types"},
                    "company":    {"type": "string",  "description": "Optional: substring filter on company name"},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_market_cap",
            "description": "Get the current market capitalization of a company by ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "date": {"type": "string", "description": "Optional date (YYYY-MM-DD) for historical data"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_from_database",
            "description": (
                "Check the local database for already-ingested filings. "
                "Always call this before fetching from EDGAR."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker symbol to look up"},
                    "form_type": {"type": "string", "description": "Filter by form type"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_to_database",
            "description": (
                "Download a filing from EDGAR, chunk it, embed it, and store it locally. "
                "Required before semantic_search can find chunks from this filing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cik": {"type": "string", "description": "SEC CIK number"},
                    "accession": {"type": "string", "description": "Accession number e.g. 0001234567-24-000001"},
                    "company": {"type": "string", "description": "Company name"},
                    "form_type": {"type": "string", "description": "Form type (10-K, 10-Q, etc.)"},
                    "filing_date": {"type": "string", "description": "Filing date (YYYY-MM-DD)"},
                    "ticker": {"type": "string", "description": "Ticker symbol if known"},
                },
                "required": ["cik", "accession", "company", "form_type", "filing_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_filing_content",
            "description": "Extract text from one filing. For 2+ filings use batch_extract_filing_content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cik": {"type": "string", "description": "SEC CIK number"},
                    "accession": {"type": "string", "description": "Accession number"},
                    "section": {"type": "string", "description": "Optional: 'Item 1A', 'Item 7', etc."},
                },
                "required": ["cik", "accession"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_events",
            "description": "Detect financial events in one text. For 2+ texts use batch_detect_events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Filing text to analyze (first 4000 chars recommended)"},
                    "event_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter to specific types: acquisition, earnings, risk, guidance, legal, dividend, restructuring",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ner_tool",
            "description": "Extract named entities from one text. For 2+ texts use batch_ner_tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze (first 4000 chars recommended)"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Hybrid BM25 + vector search over stored filing chunks. "
                "Use this for Q&A — finds the most relevant passages for a query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "ticker": {"type": "string", "description": "Restrict search to a specific ticker (optional)"},
                    "top_k": {"type": "integer", "description": "Number of chunks to return (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_filing_type",
            "description": "Filter a list of filing objects to only include specified form types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filings": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of filing dicts (from fetch_sec_filing)",
                    },
                    "form_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Types to keep e.g. ['10-K', '10-Q']",
                    },
                },
                "required": ["filings", "form_types"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_sections",
            "description": "Identify which SEC filing sections (Item 1, Item 1A, Item 7, etc.) are present in text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Filing text to classify"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "temporal_reasoning",
            "description": "Answer time-based questions: which filing is latest, date ranges, fiscal year filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Temporal question to answer"},
                    "filing_dates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of filing dates in YYYY-MM-DD format",
                    },
                },
                "required": ["query", "filing_dates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deduplicate_entities",
            "description": "Remove duplicates and near-duplicates from a list of entity names, grouping similar ones.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of entity name strings to deduplicate",
                    },
                },
                "required": ["entities"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_confidence",
            "description": "Score how well an answer is grounded in the source context. Returns a confidence value 0.0–1.0.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The answer to verify"},
                    "context": {"type": "string", "description": "Source text the answer should be based on"},
                    "question": {"type": "string", "description": "The original question (optional)"},
                },
                "required": ["answer", "context"],
            },
        },
    },
]

# ── Batch tool definitions (appended to TOOL_DEFINITIONS) ────────────────────

TOOL_DEFINITIONS += [
    {
        "type": "function",
        "function": {
            "name": "batch_extract_filing_content",
            "description": "Extract text from multiple filings in one call. items=[{cik, accession, section?}].",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "List of filings to extract",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cik":       {"type": "string", "description": "SEC CIK"},
                                "accession": {"type": "string", "description": "Accession number"},
                                "section":   {"type": "string", "description": "Optional section e.g. 'Item 1A'"},
                            },
                            "required": ["cik", "accession"],
                        },
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "batch_detect_events",
            "description": (
                "Detect events across many filings. "
                "Mode A: date+form_type — fetches all filings for that date internally (preferred for date queries). "
                "Mode B: filings=[{cik,accession}] — fetches content per ref. "
                "Mode C: items=[{text,label}] — use when you already have text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "✅ BEST: YYYY-MM-DD — tool fetches ALL filings for this date internally. Use this for date-based searches.",
                    },
                    "form_type": {
                        "type": "string",
                        "description": "Filter by form type when using date mode, e.g. '8-K'.",
                    },
                    "filings": {
                        "type": "array",
                        "description": "List of {cik, accession} dicts. Tool fetches content automatically.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cik":       {"type": "string"},
                                "accession": {"type": "string"},
                            },
                            "required": ["cik", "accession"],
                        },
                    },
                    "items": {
                        "type": "array",
                        "description": "List of {text, label} if you already have extracted text.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text":  {"type": "string"},
                                "label": {"type": "string"},
                            },
                            "required": ["text"],
                        },
                    },
                    "event_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter: officer_change, acquisition, earnings, risk, guidance, legal, dividend, restructuring",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "batch_ner_tool",
            "description": "Extract named entities from multiple texts. items=[{text, label?}].",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text":  {"type": "string"},
                                "label": {"type": "string"},
                            },
                            "required": ["text"],
                        },
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "batch_fetch_sec_filing",
            "description": "Fetch filing metadata for multiple tickers/CIKs. queries=[{ticker?,cik?,form_type?}].",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ticker":    {"type": "string"},
                                "cik":       {"type": "string"},
                                "form_type": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["queries"],
            },
        },
    },
]

# ── Tool dispatch ─────────────────────────────────────────────────────────────

from sec_intelligence.agent.tools import (
    fetch_sec_filing, fetch_filing_content, count_filings_by_date,
    fetch_market_cap, fetch_from_database, write_to_database,
    extract_filing_content, detect_events, ner_tool, semantic_search,
    filter_by_filing_type, classify_sections, temporal_reasoning,
    deduplicate_entities, verify_confidence,
    batch_extract_filing_content, batch_detect_events, batch_ner_tool, batch_fetch_sec_filing,
)

_TOOL_MAP = {
    "fetch_sec_filing":             fetch_sec_filing,
    "fetch_filing_content":         fetch_filing_content,
    "count_filings_by_date":        count_filings_by_date,
    "fetch_market_cap":             fetch_market_cap,
    "fetch_from_database":          fetch_from_database,
    "write_to_database":            write_to_database,
    "extract_filing_content":       extract_filing_content,
    "detect_events":                detect_events,
    "ner_tool":                     ner_tool,
    "semantic_search":              semantic_search,
    "filter_by_filing_type":        filter_by_filing_type,
    "classify_sections":            classify_sections,
    "temporal_reasoning":           temporal_reasoning,
    "deduplicate_entities":         deduplicate_entities,
    "verify_confidence":            verify_confidence,
    "batch_extract_filing_content": batch_extract_filing_content,
    "batch_detect_events":          batch_detect_events,
    "batch_ner_tool":               batch_ner_tool,
    "batch_fetch_sec_filing":       batch_fetch_sec_filing,
}

TOOL_ICONS = {
    "fetch_sec_filing":             "📡",
    "fetch_filing_content":         "📄",
    "count_filings_by_date":        "🔢",
    "fetch_market_cap":             "💰",
    "fetch_from_database":          "🗄️",
    "batch_extract_filing_content": "📦",
    "batch_detect_events":          "🔔",
    "batch_ner_tool":               "🏷️",
    "batch_fetch_sec_filing":       "📡",
    "write_to_database":            "💾",
    "extract_filing_content":       "📄",
    "detect_events":                "🔔",
    "ner_tool":                     "🏷️",
    "semantic_search":              "🔍",
    "filter_by_filing_type":        "🗂️",
    "classify_sections":            "📑",
    "temporal_reasoning":           "📅",
    "deduplicate_entities":         "🔗",
    "verify_confidence":            "✅",
}


_GIVE_UP_RE = re.compile(
    r"i was unable|unable to retrieve|could not extract|"
    r"encountered (?:issues|errors)|no information was found|"
    r"i could not find|unfortunately i|i apologize|"
    r"if you would like to explore|please let me know if you",
    re.IGNORECASE,
)

_CONTINUATION_PROMPT = (
    "You stopped before completing the task. "
    "Keep going:\n"
    "1. Retry every failed extraction — call extract_filing_content again for any filing "
    "that returned an error or empty content.\n"
    "2. Continue iterating through the full list — process every remaining filing.\n"
    "3. Do NOT provide a final answer until you have attempted all filings.\n"
    "Resume now."
)


# ── Orchestrator-level enforcement ────────────────────────────────────────────

# Maps single-item tools to their batch equivalents (for violation detection)
_SINGLE_TO_BATCH: dict[str, str] = {
    "extract_filing_content": "batch_extract_filing_content",
    "detect_events":          "batch_detect_events",
    "ner_tool":               "batch_ner_tool",
}

# Auto-paginate fetch_sec_filing when total_count fits comfortably in context
_AUTO_PAGE_MAX = 300


def _auto_paginate_filing(args: dict, first: dict) -> str:
    """
    When fetch_sec_filing returns has_more=True and total_count ≤ _AUTO_PAGE_MAX,
    transparently fetch remaining pages and merge into one result.
    The LLM sees the complete dataset without managing pagination itself.
    """
    from sec_intelligence.agent.tools import fetch_sec_filing as _fsf
    all_filings = list(first.get("filings", []))
    total       = first.get("total_count", 0)
    offset      = first.get("next_offset")
    pages       = 1

    while offset and len(all_filings) < total and pages < 5:
        try:
            page = _fsf(**{**args, "offset": offset})
            all_filings.extend(page.get("filings", []))
            offset = page.get("next_offset")
            pages += 1
        except Exception:
            break

    merged = {
        **first,
        "filings":        all_filings,
        "returned_count": len(all_filings),
        "has_more":       len(all_filings) < total,
        "next_offset":    None,
        "_pages_fetched": pages,
    }
    s = json.dumps(merged, default=str)
    return s[:6000] + "…[truncated]" if len(s) > 6000 else s


def _detect_batch_violation(tool_calls: list) -> str | None:
    """
    Return a correction message if the LLM called the same single-item tool
    more than once in one turn. Injected as a user message so the LLM learns
    for the next iteration — does not interrupt the current turn's execution.
    """
    from collections import Counter
    counts = Counter(tc.function.name for tc in tool_calls)
    for name, n in counts.items():
        batch = _SINGLE_TO_BATCH.get(name)
        if batch and n > 1:
            return (
                f"[Orchestrator] You called `{name}` {n} times in one turn. "
                f"Next time call `{batch}` once with all {n} items — "
                f"single-tool loops waste context and time."
            )
    return None


def _execute_tool(name: str, args: dict, max_retries: int = 2) -> tuple[str, bool]:
    """
    Execute a named tool, retrying up to max_retries times on error.
    Auto-paginates fetch_sec_filing for small datasets.
    Returns (json_result_str, had_error).
    """
    import time
    fn = _TOOL_MAP.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"}), True

    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            result = fn(**args)
            result_str = json.dumps(result, default=str)

            # Auto-paginate fetch_sec_filing for small total datasets
            if (name == "fetch_sec_filing"
                    and isinstance(result, dict)
                    and result.get("has_more")
                    and result.get("total_count", 0) <= _AUTO_PAGE_MAX):
                result_str = _auto_paginate_filing(args, result)

            if len(result_str) > 6000:
                result_str = result_str[:6000] + "…[truncated]"
            had_error = bool(result.get("error")) if isinstance(result, dict) else False
            if had_error and attempt < max_retries:
                last_err = result.get("error", "unknown") if isinstance(result, dict) else ""
                time.sleep(1.0 * (attempt + 1))
                continue
            return result_str, had_error
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
            continue

    return json.dumps({"error": f"Failed after {max_retries + 1} attempts: {last_err}"}), True


def _trim_context(messages: list, keep_recent: int = 1, max_chars: int = 150) -> list:
    """
    Keep only the most recent `keep_recent` tool results in full; truncate all
    older tool results to `max_chars`. This prevents the context from ballooning
    across many iterations while preserving the most relevant recent output.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    to_trim = set(tool_indices[:-keep_recent]) if len(tool_indices) > keep_recent else set()
    result = []
    for i, m in enumerate(messages):
        if i in to_trim:
            content = m.get("content", "")
            if len(content) > max_chars:
                m = {**m, "content": content[:max_chars] + " …[trimmed]"}
        result.append(m)
    return result


def _tool_narrative(name: str, args: dict, result_str: str, had_error: bool) -> str:
    """One-line human-readable summary of a completed tool call."""
    icon = TOOL_ICONS.get(name, "🔧")
    if had_error:
        try:
            err = json.loads(result_str).get("error", result_str[:80])
        except Exception:
            err = result_str[:80]
        return f"❌ `{name}` failed: {err}"
    try:
        r = json.loads(result_str)
    except Exception:
        r = {}
    if name == "fetch_sec_filing":
        filings        = r.get("filings", [])
        returned_count = r.get("returned_count", len(filings))
        total_count    = r.get("total_count", returned_count)
        has_more       = r.get("has_more", False)
        date           = args.get("date", "")
        ticker         = args.get("ticker", "")
        form           = args.get("form_type", "")
        mode           = args.get("mode", "metadata")
        who            = f"on **{date}**" if date else (f"for **{ticker}**" if ticker else "")
        form_str       = f" ({form})" if form else ""
        page_note      = (f" — returned {returned_count} of **{total_count} total**"
                          + (", more pages available" if has_more else "")) if date else ""
        mode_str       = " with text" if mode == "full" else ""
        return f"{icon} Fetched filing metadata{form_str}{mode_str} {who}{page_note}."
    if name == "count_filings_by_date":
        total    = r.get("total_filings", 0)
        date     = r.get("date", args.get("date", ""))
        form     = args.get("form_type", "")
        form_str = f" ({form})" if form else ""
        return f"{icon} **{total:,} total filing(s)**{form_str} filed on **{date}**."
    if name == "fetch_filing_content":
        results    = r.get("results", [])
        successful = r.get("successful", 0)
        total      = r.get("total", len(results))
        errors     = r.get("errors", 0)
        section    = args.get("section", "")
        sec_str    = f" (section: {section})" if section else ""
        return f"{icon} Downloaded content for **{successful}/{total} filing(s)**{sec_str} ({errors} failed)."
    if name == "batch_fetch_sec_filing":
        results = r.get("results", [])
        total = sum(len(x.get("filings", [])) for x in results if isinstance(x, dict))
        return f"{icon} Batch-fetched **{total} filing(s)** across {len(results)} queries."
    if name == "fetch_from_database":
        filings = r.get("filings", [])
        ticker = args.get("ticker", "")
        if filings:
            return f"{icon} Found **{len(filings)} filing(s)** for **{ticker}** in local database."
        return f"{icon} No filings for **{ticker}** in local database — will fetch from EDGAR."
    if name == "extract_filing_content":
        text = r.get("text") or r.get("content") or ""
        words = r.get("words", len(text.split()) if text else 0)
        cik = args.get("cik", "")
        return f"{icon} Extracted filing content — **{words:,} words** (CIK {cik})."
    if name == "batch_extract_filing_content":
        total = r.get("total", 0)
        successful = r.get("successful", 0)
        errors = r.get("errors", total - successful)
        return f"{icon} Extracted content from **{successful}/{total} filing(s)** ({errors} failed)."
    if name == "detect_events":
        events = r.get("events", [])
        return f"{icon} Detected **{len(events)} event(s)** in filing."
    if name == "batch_detect_events":
        total_events = r.get("total_events", 0)
        total_items = r.get("total_items", 0)
        date = args.get("date", "")
        src = f" on {date}" if date else ""
        return f"{icon} Scanned **{total_items} filing(s)**{src} — found **{total_events} event(s)**."
    if name == "ner_tool":
        entities = r.get("entities", {})
        count = sum(len(v) for v in entities.values()) if isinstance(entities, dict) else len(entities)
        return f"{icon} Extracted **{count} named entities** from text."
    if name == "batch_ner_tool":
        total = r.get("total_items", 0)
        return f"{icon} Extracted named entities from **{total} text(s)**."
    if name == "semantic_search":
        chunks = r.get("chunks", [])
        query = (args.get("query") or "")[:50]
        return f"{icon} Search for \"{query}\" → **{len(chunks)} relevant chunk(s)**."
    if name == "write_to_database":
        company = args.get("company", "")
        form = args.get("form_type", "")
        return f"{icon} Saved **{company}** {form} to local database."
    if name == "fetch_market_cap":
        ticker = args.get("ticker", "")
        mc = r.get("market_cap") or r.get("marketCap") or r.get("market_cap_usd")
        if isinstance(mc, (int, float)):
            if mc >= 1e12:
                mc_str = f"${mc / 1e12:.2f}T"
            elif mc >= 1e9:
                mc_str = f"${mc / 1e9:.2f}B"
            else:
                mc_str = f"${mc / 1e6:.2f}M"
            return f"{icon} **{ticker}** market cap: **{mc_str}**."
        return f"{icon} Fetched market cap for **{ticker}**."
    if name == "filter_by_filing_type":
        filings = r.get("filings", r.get("filtered", []))
        types = ", ".join(args.get("form_types", []))
        return f"{icon} Filtered to **{len(filings)} filing(s)** of type {types}."
    if name == "classify_sections":
        sections = r.get("sections", [])
        return f"{icon} Identified **{len(sections)} section(s)** in filing."
    if name == "temporal_reasoning":
        answer = (r.get("answer") or "")[:100]
        return f"{icon} Temporal reasoning: {answer}"
    if name == "deduplicate_entities":
        deduped = r.get("deduplicated", r.get("entities", []))
        orig = r.get("original_count", "?")
        return f"{icon} Deduplicated **{orig} → {len(deduped)} unique entities**."
    if name == "verify_confidence":
        score = r.get("confidence") or r.get("score")
        if isinstance(score, (int, float)):
            return f"{icon} Confidence score: **{score:.0%}**."
        return f"{icon} Confidence verification complete."
    return f"{icon} **{name}** completed."


# ── Agent loop ────────────────────────────────────────────────────────────────

Stage = dict  # keys: step, tool, status, args_preview, result_preview


def _chat_history(history: list, question: str, assistant_msg: str) -> list:
    """
    Return updated Gradio 6 messages-format history.
    history is list[dict] with keys 'role' and 'content'.
    """
    return history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_msg},
    ]


def run_agent(
    question: str,
    history: list,       # list[dict] — Gradio 6 messages format
    max_iterations: int = 25,
) -> Generator[tuple[list, list], None, None]:
    """
    Agentic loop driven by OpenAI function calling.

    Yields (stages, chat_history) on every meaningful state change.
      stages       — list of Stage dicts for the debug panel
      chat_history — list[dict] messages for the Gradio Chatbot
    """
    config = get_config()
    api_key = config.llm.api_key or ""
    if not api_key or api_key.startswith("your_"):
        err = "❌ OpenAI API key not set. Add `LLM_API_KEY=sk-…` to your .env file."
        yield [], _chat_history(history, question, err)
        return

    client = OpenAI(api_key=api_key)

    # Build OpenAI messages from Gradio history (already role/content dicts)
    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role in ("user", "assistant") and content:
            llm_messages.append({"role": role, "content": content})
    llm_messages.append({"role": "user", "content": question})

    stages: list[Stage] = []
    step = 0

    # Show "thinking" immediately
    yield stages, _chat_history(history, question, "🤔 Planning…")

    for iteration in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model=config.llm.model_name,
                messages=llm_messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2000,
            )
        except Exception as e:
            err = f"❌ LLM error: {e}"
            yield stages, _chat_history(history, question, err)
            return

        msg = response.choices[0].message
        finish = response.choices[0].finish_reason

        # No tool calls → check if the LLM is giving up prematurely
        if finish == "stop" or not msg.tool_calls:
            answer = msg.content or "Analysis complete."

            # Detect premature surrender and force continuation
            if _GIVE_UP_RE.search(answer) and iteration < max_iterations - 3:
                step += 1
                retry_stage: Stage = {
                    "step": step,
                    "tool": "🔄 retry-loop",
                    "status": "⏳ running",
                    "args": "Premature stop detected — forcing continuation",
                    "result": "",
                    "error": "",
                }
                stages.append(retry_stage)
                yield stages, _chat_history(
                    history, question,
                    answer + "\n\n_🔄 Incomplete — agent retrying remaining filings…_",
                )
                # Push partial answer + continuation prompt back into the thread
                llm_messages.append({"role": "assistant", "content": answer})
                llm_messages.append({"role": "user", "content": _CONTINUATION_PROMPT})
                retry_stage["status"] = "✅ done"
                retry_stage["result"] = "Continuation injected"
                continue  # re-enter the loop

            yield stages, _chat_history(history, question, answer)
            return

        # Append assistant tool-call message to LLM thread
        llm_messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        tool_results = []

        for tc in msg.tool_calls:
            step += 1
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}

            icon = TOOL_ICONS.get(name, "🔧")
            args_preview = json.dumps(args, default=str)
            if len(args_preview) > 100:
                args_preview = args_preview[:100] + "…"

            stage: Stage = {
                "step": step,
                "tool": f"{icon} {name}",
                "status": "⏳ running",
                "args": args_preview,
                "result": "",
                "error": "",
            }
            stages.append(stage)

            # Live update — running
            yield stages, _chat_history(history, question, _build_thinking(stages, name))

            result_str, had_error = _execute_tool(name, args)

            stage["status"] = "❌ error" if had_error else "✅ done"
            stage["result"] = result_str[:150].replace("\n", " ")
            if had_error:
                try:
                    stage["error"] = json.loads(result_str).get("error", "")[:100]
                except Exception:
                    stage["error"] = result_str[:100]

            # Live update — done
            yield stages, _chat_history(history, question, _build_thinking(stages, None))

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

        llm_messages.extend(tool_results)

    # Max iterations hit — force final answer
    llm_messages.append({
        "role": "user",
        "content": "Summarize your findings and provide a final answer now.",
    })
    try:
        final = client.chat.completions.create(
            model=config.llm.model_name,
            messages=llm_messages,
            max_tokens=1200,
            temperature=0.1,
        )
        answer = final.choices[0].message.content or "Analysis complete (max steps reached)."
    except Exception as e:
        answer = f"Analysis complete (max steps reached). Error in synthesis: {e}"

    yield stages, _chat_history(history, question, answer)


def _build_thinking(stages: list[Stage], running_tool: str | None) -> str:
    """Live status shown in the chatbot while the agent is working."""
    lines = ["🤖 **Agent working…**\n"]
    for s in stages:
        lines.append(f"{s['status']}  **{s['tool']}**")
        if s.get("error"):
            lines.append(f"   ↳ _{s['error']}_")
    if running_tool:
        icon = TOOL_ICONS.get(running_tool, "🔧")
        lines.append(f"⏳ running  **{icon} {running_tool}**")
    return "\n".join(lines)


def generate_plan_text(question: str, history: list) -> str:
    """Preliminary LLM call (no tools) to generate a numbered execution plan."""
    config = get_config()
    api_key = config.llm.api_key or ""
    if not api_key or api_key.startswith("your_"):
        return "❌ API key not set — cannot generate plan."

    client = OpenAI(api_key=api_key)

    plan_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert SEC analyst. Given the user's question, generate a clear "
                "numbered step-by-step execution plan describing EXACTLY which tools you will "
                "call and in what order, with specific expected inputs for each step. "
                "Available tools: fetch_sec_filing, fetch_from_database, extract_filing_content, "
                "batch_extract_filing_content, detect_events, batch_detect_events, ner_tool, "
                "semantic_search, write_to_database, fetch_market_cap, filter_by_filing_type, "
                "classify_sections, temporal_reasoning, deduplicate_entities, verify_confidence. "
                "Do NOT execute tools — only write the plan. Format as a numbered list."
            ),
        },
    ]
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role in ("user", "assistant") and content:
            plan_messages.append({"role": role, "content": content})
    plan_messages.append({
        "role": "user",
        "content": f"Write your execution plan to answer: {question}",
    })

    try:
        resp = client.chat.completions.create(
            model=config.llm.model_name,
            messages=plan_messages,
            max_tokens=500,
            temperature=0.1,
        )
        return resp.choices[0].message.content or "No plan generated."
    except Exception as e:
        return f"Error generating plan: {e}"


def _format_step_detail(
    iteration: int,
    messages_sent: list,
    llm_reply: str,
    tool_calls_info: list,
    tool_results_info: list,
) -> str:
    """Format a full step breakdown: context sent, LLM reply, tool calls+params, results."""
    lines = [f"{'=' * 50}", f"  STEP {iteration} — FULL DETAIL", f"{'=' * 50}\n"]

    lines.append(f"── CONTEXT SENT TO LLM ({len(messages_sent)} messages) ──")
    for idx, m in enumerate(messages_sent):
        role = m.get("role", "")
        content = m.get("content", "") or ""
        prefix = f"[{idx + 1}/{len(messages_sent)}] [{role.upper()}]"
        if role == "tool":
            preview = content[:500] + ("…" if len(content) > 500 else "")
            lines.append(f"\n{prefix}\n{preview}")
        elif role == "assistant":
            tc = m.get("tool_calls", [])
            if tc:
                names = ", ".join(c["function"]["name"] for c in tc)
                lines.append(f"\n{prefix} → called: {names}")
                if content:
                    lines.append(content)
            elif content:
                lines.append(f"\n{prefix}\n{content}")
        elif content:
            lines.append(f"\n{prefix}\n{content}")

    if llm_reply:
        lines.append(f"\n── LLM RESPONSE ──")
        lines.append(llm_reply[:700] + ("…" if len(llm_reply) > 700 else ""))

    if tool_calls_info:
        lines.append(f"\n── TOOLS CALLED ({len(tool_calls_info)}) ──")
        for name, args in tool_calls_info:
            icon = TOOL_ICONS.get(name, "🔧")
            args_str = json.dumps(args, indent=2, default=str)
            if len(args_str) > 400:
                args_str = args_str[:400] + "\n  …(truncated)"
            lines.append(f"\n{icon} {name}")
            lines.append(f"  Args: {args_str}")

    if tool_results_info:
        lines.append(f"\n── TOOL RESULTS ──")
        for name, result_str, had_error in tool_results_info:
            icon = TOOL_ICONS.get(name, "🔧")
            status = "❌ ERROR" if had_error else "✅ OK"
            preview = result_str[:350] + ("…" if len(result_str) > 350 else "")
            lines.append(f"\n{status}  {icon} {name}")
            lines.append(f"  {preview}")

    return "\n".join(lines)


# ── Step-by-step execution (human-in-the-loop) ────────────────────────────────

def init_agent_state(question: str, history: list) -> dict:
    """
    Create a fresh agent state dict for a new question.
    Stores the full LLM thread plus bookkeeping for step-by-step execution.
    The planning instruction is baked into the first user message so no
    separate planning API call is needed.
    """
    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role in ("user", "assistant") and content:
            llm_messages.append({"role": role, "content": content})
    llm_messages.append({
        "role": "user",
        "content": (
            question
            + "\n\nBefore calling any tools, write a concise numbered execution plan "
            "(which tools, in what order, with what key inputs). "
            "Then immediately start executing."
        ),
    })

    return {
        "llm_messages": llm_messages,
        "stages": [],
        "step": 0,
        "iteration": 0,
        "max_iterations": 25,
        "question": question,
        "done": False,
        "answer": "",
        "give_up_count": 0,
        "plan": "",
        "current_step_detail": "",
        "all_step_details": [],
        "tool_error_counts": {},
    }


def run_one_step(state: dict):
    """
    Generator: execute ONE agent iteration and yield (state, display, is_final) tuples.
    Intermediate yields (is_final=False) are sent after each tool call so the UI can
    show live progress. The final yield (is_final=True) carries the complete summary.
    """
    if state.get("done"):
        yield state, state.get("answer", "Done."), True
        return

    config = get_config()
    api_key = config.llm.api_key or ""
    if not api_key or api_key.startswith("your_"):
        state["done"] = True
        state["answer"] = "❌ OpenAI API key not set. Add LLM_API_KEY=sk-… to your .env file."
        yield state, state["answer"], True
        return

    client = OpenAI(api_key=api_key)
    state["iteration"] += 1

    if state["iteration"] > state["max_iterations"]:
        state["done"] = True
        state["answer"] = "⚠️ Maximum iterations reached. Stopping here."
        yield state, state["answer"], True
        return

    # Trim old tool results to keep context lean (OpenAI auto-caches the stable prefix)
    state["llm_messages"] = _trim_context(state["llm_messages"])
    messages_sent = list(state["llm_messages"])

    # ── LLM call ──────────────────────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=config.llm.model_name,
            messages=state["llm_messages"],
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=2000,
        )
    except Exception as e:
        state["done"] = True
        state["answer"] = f"❌ LLM error: {e}"
        yield state, state["answer"], True
        return

    msg = response.choices[0].message
    finish = response.choices[0].finish_reason

    # Capture the first LLM text response as the execution plan
    if state["iteration"] == 1 and msg.content and not state.get("plan"):
        state["plan"] = msg.content

    # ── Final answer (no tool calls) ──────────────────────────────────────────
    if finish == "stop" or not msg.tool_calls:
        answer = msg.content or "Analysis complete."

        if _GIVE_UP_RE.search(answer) and state["iteration"] < state["max_iterations"] - 3:
            state["give_up_count"] = state.get("give_up_count", 0) + 1
            state["llm_messages"].append({"role": "assistant", "content": answer})
            state["llm_messages"].append({"role": "user", "content": _CONTINUATION_PROMPT})
            state["step"] += 1
            state["stages"].append({
                "step": state["step"],
                "tool": "🔄 retry-loop",
                "status": "⚠️ forced",
                "args": "Premature stop detected — continuation injected",
                "result": "",
                "error": "",
            })
            _detail = _format_step_detail(state["iteration"], messages_sent, answer, [], [])
            state["current_step_detail"] = _detail
            state["all_step_details"].append(_detail)
            display = (
                f"{answer}\n\n---\n"
                f"⚠️ **Agent stopped prematurely. Continuation injected automatically.**\n\n"
                f"⏸️ Click **▶ Continue** to retry remaining filings."
            )
            yield state, display, True
            return

        state["done"] = True
        state["answer"] = answer
        _detail = _format_step_detail(state["iteration"], messages_sent, answer, [], [])
        state["current_step_detail"] = _detail
        state["all_step_details"].append(_detail)
        yield state, answer, True
        return

    # ── Execute tool calls ────────────────────────────────────────────────────
    state["llm_messages"].append({
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ],
    })

    tool_results = []
    tool_calls_info: list = []
    tool_results_info: list = []
    narrative_lines: list[str] = []
    n = state["iteration"]

    # Orchestrator: detect and log batch-violation for the next LLM turn
    batch_violation = _detect_batch_violation(msg.tool_calls)

    # Prefix the running display with the LLM's reasoning text (if any)
    llm_preamble = (f"> {msg.content}\n\n") if msg.content else ""

    for tc in msg.tool_calls:
        state["step"] += 1
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except Exception:
            args = {}

        icon = TOOL_ICONS.get(name, "🔧")
        tool_calls_info.append((name, args))
        args_preview = json.dumps(args, default=str)
        if len(args_preview) > 120:
            args_preview = args_preview[:120] + "…"

        stage: Stage = {
            "step": state["step"],
            "tool": f"{icon} {name}",
            "status": "⏳ running",
            "args": args_preview,
            "result": "",
            "error": "",
        }
        state["stages"].append(stage)

        result_str, had_error = _execute_tool(name, args)

        stage["status"] = "❌ error" if had_error else "✅ done"
        stage["result"] = result_str[:200].replace("\n", " ")
        if had_error:
            try:
                stage["error"] = json.loads(result_str).get("error", "")[:120]
            except Exception:
                stage["error"] = result_str[:120]

        tool_results_info.append((name, result_str, had_error))
        tool_results.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

        # Loop-break: if the same tool keeps failing with the same error, force a pivot
        if had_error:
            err_sig = f"{name}:{stage.get('error', '')[:60]}"
            counts = state.setdefault("tool_error_counts", {})
            counts[err_sig] = counts.get(err_sig, 0) + 1
            if counts[err_sig] >= 2:
                correction = (
                    f"STOP RETRYING `{name}` — it has failed {counts[err_sig]} times "
                    f"with the same error: \"{stage.get('error', '')}\". "
                    f"You MUST switch to a completely different approach. "
                    f"If you were calling batch_detect_events with empty args, "
                    f"you must pass either: "
                    f"filings=[{{\"cik\": \"...\", \"accession\": \"...\"}}] "
                    f"or items=[{{\"text\": \"...\", \"label\": \"...\"}}]. "
                    f"Do NOT call this tool again without the required arguments."
                )
                tool_results.append({
                    "role": "user",
                    "content": correction,
                })

        # Live narrative update after each tool
        narrative_lines.append(_tool_narrative(name, args, result_str, had_error))
        running = (
            f"**Step {n}** — in progress…\n\n"
            + llm_preamble
            + "\n".join(f"- {l}" for l in narrative_lines)
            + "\n\n_⏳ Working…_"
        )
        yield state, running, False

    state["llm_messages"].extend(tool_results)

    # Inject batch-violation correction so the LLM sees it on the next iteration
    if batch_violation:
        state["llm_messages"].append({"role": "user", "content": batch_violation})

    _detail = _format_step_detail(
        state["iteration"], messages_sent, msg.content or "",
        tool_calls_info, tool_results_info,
    )
    state["current_step_detail"] = _detail
    state["all_step_details"].append(_detail)

    final_display = (
        f"**Step {n} complete**\n\n"
        + llm_preamble
        + "\n".join(f"- {l}" for l in narrative_lines)
        + f"\n\n⏸️ Click **▶ Next step** or **⚡ Run all steps** to continue."
    )
    yield state, final_display, True
