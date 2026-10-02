# SEC Intelligence Platform

**Live site:** [agentic-ai-eight-alpha.vercel.app](https://agentic-ai-eight-alpha.vercel.app) · **Source:** [`/web`](web)

> The live site is a Vite + React landing page that explains the project. The application itself runs locally, so follow the setup steps below to try it.


An autonomous financial intelligence system that queries SEC EDGAR in real time, extracts structured events and entities from filings, and answers natural-language questions through a multi-tool agentic loop powered by OpenAI function-calling.

---

## Streamlit app

A Streamlit version of the Gradio UI. Paste your OpenAI key in the sidebar; it is kept in your session only. The Gradio app (`app.py`) is unchanged.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Deploy on Streamlit Community Cloud:** at [share.streamlit.io](https://share.streamlit.io) choose this repo, branch `main` and main file `streamlit_app.py`.

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [System Architecture](#2-system-architecture)
3. [Repository Layout](#3-repository-layout)
4. [Setup & Configuration](#4-setup--configuration)
5. [Running the App](#5-running-the-app)
6. [Module Reference](#6-module-reference)
   - [live_search.py — EDGAR HTTP Layer](#61-live_searchpy--edgar-http-layer)
   - [agent/tools.py — Tool Implementations](#62-agenttoolspy--tool-implementations)
   - [agent/planner.py — Agent Orchestrator](#63-agentplannerpy--agent-orchestrator)
   - [pipeline.py — Local RAG Pipeline](#64-pipelinepy--local-rag-pipeline)
   - [app.py — Gradio UI](#65-apppy--gradio-ui)
   - [config.py — Configuration](#66-configpy--configuration)
7. [All Tools Reference](#7-all-tools-reference)
8. [Event Detection System](#8-event-detection-system)
9. [EDGAR Rate Limiting & Caching](#9-edgar-rate-limiting--caching)
10. [Agent Loop Mechanics](#10-agent-loop-mechanics)
11. [Local RAG Pipeline](#11-local-rag-pipeline)
12. [Data Flow Diagrams](#12-data-flow-diagrams)
13. [Environment Variables](#13-environment-variables)

---

## 1. What It Does

- **Real-time EDGAR access** — fetches filing metadata and full text directly from SEC EDGAR with automatic rate limiting, retry, and caching; no pre-processing pipeline required.
- **Agentic question answering** — an OpenAI function-calling loop chooses and calls the right tools, paginates automatically, and never stops until the question is fully answered.
- **Structured event extraction** — classifies officer appointments, resignations, retirements, acquisitions, earnings, dividends, restructurings, legal actions, and risk disclosures into typed, machine-readable structs.
- **Named entity recognition** — extracts people, companies, financial figures, dates, locations, and ticker symbols from filing text.
- **Hybrid semantic search** — BM25 keyword search combined with sentence-transformer vector search over locally stored filing chunks.
- **Local SQLite storage** — no external database required; filings are chunked, embedded, and stored in a single `.sec_data.db` file.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Gradio UI  (app.py)                         │
│  start_chat → run_one_step → continue_step / run_all_steps / stop   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ question + history
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               Agent Orchestrator  (agent/planner.py)                │
│                                                                     │
│  init_agent_state()   — build LLM thread                           │
│  run_one_step()        — one LLM call + all tool calls              │
│  _execute_tool()       — dispatch + auto-retry + auto-paginate      │
│  _detect_batch_violation() — enforce batch-tool usage               │
│  _trim_context()       — prune old tool results                     │
│  _auto_paginate_filing() — transparent pagination for small sets    │
│  _tool_narrative()     — human-readable tool result summaries       │
└──────┬──────────────────────────────────────────────────────────────┘
       │ function calls
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Tool Layer  (agent/tools.py)                       │
│                                                                     │
│  EDGAR tools:   fetch_sec_filing, fetch_filing_content,            │
│                 count_filings_by_date, fetch_market_cap            │
│                                                                     │
│  DB tools:      fetch_from_database, write_to_database             │
│                                                                     │
│  NLP tools:     extract_filing_content, detect_events, ner_tool,   │
│                 classify_sections, semantic_search                  │
│                                                                     │
│  Utility tools: filter_by_filing_type, temporal_reasoning,         │
│                 deduplicate_entities, verify_confidence             │
│                                                                     │
│  Batch tools:   batch_extract_filing_content, batch_detect_events, │
│                 batch_ner_tool, batch_fetch_sec_filing              │
└──────┬──────────────────────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
┌──────────────────────────┐           ┌─────────────────────────────┐
│  EDGAR HTTP Layer        │           │  Local RAG Pipeline         │
│  (live_search.py)        │           │  (pipeline.py)              │
│                          │           │                             │
│  _RateLimiter            │           │  LocalDB (SQLite)           │
│  _retry_get()            │           │  _strip_xbrl()              │
│  _load_quarter_index()   │           │  _detect_sections()         │
│  get_filings_paginated() │           │  _chunk_section()           │
│  fetch_filing_text()     │           │  _bm25_scores()             │
│  fetch_filings_content   │           │  _vector_scores()           │
│    _concurrent()         │           │  _hybrid_top_k()            │
│  ticker_to_cik()         │           │  SECPipeline.run()          │
└──────────────────────────┘           └─────────────────────────────┘
       │
       ▼
   SEC EDGAR (www.sec.gov)
```

### Invariants enforced by the orchestrator (not the LLM)

| Rule | Enforcement point |
|------|-------------------|
| Never loop single-item tools — use batch variants | `_detect_batch_violation()` injects a correction message |
| Never report `returned_count` as the total | Rename in tool contract + system prompt rule |
| Auto-page through results ≤ 300 total | `_auto_paginate_filing()` in `_execute_tool()` |
| Stop retrying a tool with the same error after 2 failures | `tool_error_counts` dict + forced pivot message |
| Never give up prematurely | `_GIVE_UP_RE` pattern + `_CONTINUATION_PROMPT` injection |
| Keep context lean | `_trim_context(keep_recent=1, max_chars=150)` after each step |

---

## 3. Repository Layout

```
Agentic AI/
├── app.py                          # Gradio UI entry point
├── .env                            # API keys (not committed)
├── .env.example                    # Template
├── pyproject.toml                  # Package metadata & dependencies
├── run.sh                          # Convenience launcher
├── examples.py                     # Standalone usage examples
├── examples/
│   ├── interactive_qa.py
│   ├── process_filing.py
│   ├── query_risks.py
│   └── simple_qa.py
├── tests/
│   ├── conftest.py
│   └── test_core.py
└── sec_intelligence/
    ├── __init__.py
    ├── config.py                   # All configuration (Pydantic Settings)
    ├── live_search.py              # EDGAR HTTP layer — rate limiter, fetchers, caches
    ├── pipeline.py                 # Local RAG pipeline — SQLite, chunking, embeddings
    ├── platform.py                 # High-level platform facade
    ├── agent/
    │   ├── __init__.py
    │   ├── tools.py                # All 19 tool implementations
    │   └── planner.py              # Agentic loop, tool dispatch, orchestration
    ├── agents/
    │   └── __init__.py
    ├── core/
    │   └── orchestrator.py
    ├── layers/
    │   ├── ingestion.py
    │   ├── structuring.py
    │   └── extraction.py
    ├── retrieval/
    │   └── __init__.py
    ├── schemas/
    │   └── __init__.py
    ├── storage/
    │   └── models.py
    └── utils/
        └── helpers.py
```

---

## 4. Setup & Configuration

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# Install package with all dependencies
pip install -e .

# Copy the environment template
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
LLM_API_KEY=sk-...          # OpenAI API key (required)
LLM_MODEL_NAME=gpt-4o-mini  # Model to use (default: gpt-4o-mini)
```

Everything else has sensible defaults and is optional.

---

## 5. Running the App

```bash
# Launch the Gradio web UI
python app.py
# Opens at http://localhost:7860

# Or use the convenience script
bash run.sh
```

The UI has three tabs:

| Tab | Purpose |
|-----|---------|
| **Agent** | Chat with the agent; step through or auto-run the agentic loop |
| **Database** | View ingested filings; delete a ticker's data |
| **System** | Health check (API key, EDGAR reachability, SQLite, embedding model) |

---

## 6. Module Reference

### 6.1 `live_search.py` — EDGAR HTTP Layer

All outbound HTTP calls to SEC EDGAR go through this module. It is the only place where network I/O happens for filing retrieval.

#### Module-level constants

| Name | Value | Purpose |
|------|-------|---------|
| `HEADERS` | `{"User-Agent": "...", "Accept-Encoding": "gzip, deflate"}` | Required by EDGAR's fair-use policy |

#### Module-level caches

| Name | Type | Key | Value |
|------|------|-----|-------|
| `_filing_text_cache` | `dict[str, str]` | `"{accession}:{max_chars}"` | Cleaned filing text |
| `_edgar_index_cache` | `dict[str, list[dict]]` | `"YYYY/QTRN"` | All entries for a quarter |
| `_ticker_data` | `list[dict] \| None` | — | Full SEC company ticker list |

#### Class: `_RateLimiter`

Token-bucket rate limiter enforcing EDGAR's 10 req/sec policy (configured to 8 req/sec to leave headroom).

```python
class _RateLimiter:
    def __init__(self, rate: float = 8.0)
    def wait(self) -> None
```

- `__init__(rate)` — initializes bucket with `rate` tokens and a threading lock.
- `wait()` — acquires the lock, refills tokens based on elapsed time, consumes one token. If no token is available, releases the lock and sleeps for exactly the time needed to accumulate one token, then returns.

Module-level singleton: `_edgar_limiter = _RateLimiter(8.0)`

#### `_retry_get(url, timeout=30, max_retries=3) -> requests.Response`

Wraps every EDGAR GET with rate limiting and retry logic.

- Calls `_edgar_limiter.wait()` before every attempt.
- **429 Too Many Requests** — reads `Retry-After` header (defaults to `2^(attempt+1)` seconds) and sleeps.
- **500/502/503/504** — exponential backoff (`2^attempt` seconds) then retry.
- **Timeout** — exponential backoff then retry; raises on final attempt.
- Raises `RuntimeError` after all attempts are exhausted.

#### `_load_ticker_data() -> list[dict]`

Downloads `https://www.sec.gov/files/company_tickers.json` once and caches the result in `_ticker_data`. Each entry has `cik_str` (int), `ticker` (str), `title` (str).

#### `ticker_to_cik(ticker: str) -> str | None`

Returns a zero-padded 10-digit CIK string for a given ticker symbol. Case-insensitive. Returns `None` if the ticker is not in the SEC's master list.

#### `cik_to_ticker(cik: str | int) -> str | None`

Reverse lookup — returns the ticker for a CIK integer, or `None`.

#### `_load_quarter_index(year: int, quarter: int) -> list[dict]`

Downloads EDGAR's `full-index/{year}/QTR{quarter}/company.idx` (~10 MB) and parses it into a list of filing dicts. Cached for the process lifetime in `_edgar_index_cache`.

Each entry: `{company, form_type, cik, filing_date, accession, ticker}`.

After parsing, resolves tickers for all entries in a single pass against `_load_ticker_data()`.

#### `get_filings_for_date(date_str, form_types=None) -> list[dict]`

Thin wrapper over `_load_quarter_index`. Returns all filings whose `filing_date` matches `date_str`. Optionally filters to `form_types`.

#### `get_filings_paginated(date_str, form_types=None, company_filter=None, limit=100, offset=0) -> dict`

Paginates `get_filings_for_date`. Returns:

```python
{
    "filings":     list[dict],   # current page
    "total":       int,          # total matching filings
    "returned":    int,          # count in this page
    "offset":      int,
    "limit":       int,
    "has_more":    bool,
    "next_offset": int | None,
}
```

#### `get_latest_filing(cik: str, form_type: str = "10-K") -> dict | None`

Calls `https://data.sec.gov/submissions/CIK{cik}.json`, scans the `filings.recent` list for the first occurrence of `form_type`, and returns `{accession, date, cik, company}`.

#### `ticker_label(cik: str) -> str`

Returns `"CIK {cik}"` — used as a fallback company name when the real name is unavailable.

#### `_get_doc_url(cik: str, accession: str) -> str | None`

Resolves the primary HTML document URL for a filing:

1. Fetches `{base}/{accession}-index.htm` (the filing index page).
2. Parses all `<a href>` links; strips iXBRL viewer wrappers (`/ix?doc=...`).
3. Keeps only links inside the filing's directory that end in `.htm`, `.html`, or `.txt`, excluding index/viewer files.
4. Scores candidates: plain text > HTML, exhibits ranked lower, shorter filenames ranked higher. Returns the best match.
5. If no candidates found, falls back to the SGML composite `{accession}.txt` (verified with a HEAD request to avoid downloading).

#### `fetch_filing_text(cik: str, accession: str, max_chars: int = 60_000) -> str`

Downloads and cleans a filing's primary document:

1. Checks `_filing_text_cache` (key: `"{accession}:{max_chars}"`); returns cached text if present.
2. Calls `_get_doc_url()` to resolve the document URL.
3. GETs the document via `_retry_get()`.
4. Parses HTML with BeautifulSoup; removes `<script>`, `<style>`, `<meta>`, `<link>`, `<noscript>`.
5. Extracts text with `separator="\n"`, strips blank lines, normalizes horizontal whitespace.
6. Truncates to `max_chars` and stores in cache.

#### `find_relevant_passages(text: str, query: str, top_n: int = 3, window: int = 1500) -> str`

Sliding-window passage extraction:

1. Tokenizes query into words longer than 3 characters.
2. Slides a window of `window` chars across text in steps of 300.
3. Scores each window by the count of query words it contains.
4. Returns the top `top_n` non-overlapping windows joined by `---` separators.

#### `fetch_filings_content_concurrent(filing_refs, max_chars=60_000, max_workers=4) -> list[dict]`

Fetches text for multiple filings using `ThreadPoolExecutor(max_workers=4)`. The `_edgar_limiter` inside `fetch_filing_text` serializes all EDGAR requests safely across threads. Returns a list of `{cik, accession, text, char_count, error, ...}` dicts (extra fields from input refs are preserved).

#### `live_answer(question: str, ticker: str, form_type: str = "10-K") -> dict`

End-to-end single-ticker Q&A without the agent loop:

1. Resolves ticker → CIK via `ticker_to_cik()`.
2. Finds latest filing via `get_latest_filing()`.
3. Downloads text via `fetch_filing_text()`.
4. Extracts relevant passages via `find_relevant_passages()`.
5. Sends to GPT-4o mini with a grounded prompt.

Returns `{answer, company, date, accession, doc_url, error}`.

---

### 6.2 `agent/tools.py` — Tool Implementations

Every public function in this file returns a `dict` with at minimum an `"error"` key (`None` on success). Each function is exposed to the LLM as an OpenAI function-calling tool.

#### Private helpers

##### `_standardize_filing(raw: dict) -> dict`

Normalizes any EDGAR filing dict to the standard schema:

```python
{
    "cik":         str,   # zero-padded 10-digit
    "ticker":      str,
    "company":     str,
    "form_type":   str,
    "filing_date": str,   # YYYY-MM-DD
    "accession":   str,
    "filing_url":  str,   # base directory URL on EDGAR
    "index_url":   str,   # {base}/{accession}-index.htm
}
```

Accepts raw dicts from EDGAR (using either `accessionNumber` or `accession`, `name` or `company`, etc.).

#### Event detection private internals

##### `_ROLE_CANON: dict[str, str]`

Maps every role title variant (lowercase) to its canonical abbreviation. Examples: `"chief financial officer"` → `"CFO"`, `"executive vice president"` → `"EVP"`.

##### `_ROLE_RE: re.Pattern`

Compiled regex that matches any known officer role title in filing text (case-insensitive). Used to anchor officer-event detection — events without a matched role are discarded.

##### `_NAME_RE: re.Pattern`

Matches person names in two forms:
- `Mr.|Ms.|Mrs.|Dr.` followed by one to three Title-Case words of 3+ characters.
- Two to three adjacent Title-Case words of 3+ characters (no prefix required).

##### `_FAKE_NAME_TOKENS: frozenset[str]`

Words that match the name pattern but are not person names: SEC section headings (`Item`, `Section`, `Part`, `Exhibit`), financial boilerplate (`Financial`, `Statements`, `Management`), month names (`January`–`December`), and common sentence-start words (`The`, `This`, `Upon`, etc.).

##### `_norm_role(raw: str) -> str`

Lowercases and normalizes whitespace in `raw`, looks it up in `_ROLE_CANON`, and returns the canonical form (e.g., `"CFO"`). Falls back to the stripped original if not found.

##### `_extract_name(snippet: str) -> str`

Iterates `_NAME_RE` matches in `snippet`. For each candidate, rejects it if:
- Any word is in `_FAKE_NAME_TOKENS`.
- The candidate has fewer than 2 words.
- All words are uppercase acronyms.

Returns the first accepted candidate, or `""`.

##### `_extract_officer_events(text: str) -> list[dict]`

Applies five compiled regex patterns to find officer changes:

| Pattern type | Matches |
|-------------|---------|
| Appointment (action-first) | `"appointed Jane Doe as CFO"` |
| Appointment (role-first) | `"CFO ... appointed"` |
| Resignation (role-first) | `"CFO ... resigned"` |
| Resignation (action-first) | `"resigned as CFO"` |
| Retirement | `"CFO ... retiring"` |

For each match:
1. Extracts role from the `(?P<role>...)` group → normalizes with `_norm_role()`.
2. Discards the event if no role was captured (eliminates section-header false positives).
3. Extracts action verb → normalizes to past-tense canonical form.
4. Extracts person name from a ±120-char context window via `_extract_name()`.
5. Returns `{event_type, person, role, action, context, position}`.

##### `_extract_non_officer_events(text: str, target_types: set) -> list[dict]`

Applies named-group regexes for non-officer event types:

| Event type | Example match | Named groups |
|------------|---------------|--------------|
| `acquisition` | `"acquired Acme Corp for $2B"` | `target`, `amount` |
| `earnings` | `"net income of $500M"` | `amount` |
| `dividend` | `"declared quarterly dividend of $0.25 per share"` | `amount` |
| `restructuring` | `"eliminating approximately 2,000 positions"` | `count` |
| `legal` | `"class action against ..."` | `party` |
| `legal` (settlement) | `"settlement of $50M"` | `amount` |
| `risk` | `"material weakness"` | — |
| `guidance` | `"expects revenue to be $4B"` | `period`, `amount` |

Returns list of `{event_type, action, amount, target, context, position}`.

---

### 6.3 `agent/planner.py` — Agent Orchestrator

#### Constants and configuration

##### `SYSTEM_PROMPT`

574-character static system prompt baked into every LLM call. Contains five rules:

1. Use `count_filings_by_date` for counting — never infer counts from pages.
2. Report `total_count`, not `returned_count`.
3. Use batch tools for multiple items — never loop single-item tools.
4. Tools manage pagination, retries, rate limiting — the LLM does not.
5. Never stop before the answer is complete.

##### `TOOL_DEFINITIONS`

List of 19 OpenAI function-calling tool schemas. Each schema has a `name`, one-line `description`, and a `parameters` JSON Schema object. Tool descriptions are intentionally concise (one line each) to minimize per-call prompt tokens.

##### `_TOOL_MAP: dict[str, Callable]`

Maps tool name strings to their implementation functions imported from `agent/tools.py`. Used by `_execute_tool()`.

##### `TOOL_ICONS: dict[str, str]`

Maps tool names to emoji icons used in the UI's live status display.

##### `_GIVE_UP_RE: re.Pattern`

Matches phrases indicating premature surrender: `"i was unable"`, `"could not extract"`, `"no information was found"`, `"unfortunately i"`, etc.

##### `_CONTINUATION_PROMPT: str`

Injected as a user message when `_GIVE_UP_RE` matches. Forces the LLM to retry every failed extraction and continue iterating.

##### `_SINGLE_TO_BATCH: dict[str, str]`

Maps single-item tools to their batch equivalents:
- `"extract_filing_content"` → `"batch_extract_filing_content"`
- `"detect_events"` → `"batch_detect_events"`
- `"ner_tool"` → `"batch_ner_tool"`

##### `_AUTO_PAGE_MAX: int = 300`

When `fetch_sec_filing` returns `has_more=True` and `total_count ≤ 300`, the orchestrator transparently fetches all remaining pages and merges them. Above 300 the LLM must manage pagination itself.

#### Orchestrator functions

##### `_auto_paginate_filing(args: dict, first: dict) -> str`

Called inside `_execute_tool()` when fetch conditions are met. Loops up to 5 pages, accumulating all filings into `all_filings`. Returns a merged JSON string capped at 6000 chars. The LLM sees one complete result instead of paginated fragments.

##### `_detect_batch_violation(tool_calls: list) -> str | None`

Counts how many times each tool name appears in the current turn's tool calls. If any single-item tool (in `_SINGLE_TO_BATCH`) appears more than once, returns a correction message string. The message is appended as a `role: "user"` message after all tool results, so the LLM sees it on the next iteration. Returns `None` if no violation.

##### `_execute_tool(name: str, args: dict, max_retries: int = 2) -> tuple[str, bool]`

Dispatches a tool call and returns `(json_result_str, had_error)`:

1. Looks up `name` in `_TOOL_MAP`; returns error JSON if not found.
2. Calls the function with `**args`.
3. If `fetch_sec_filing` returns `has_more=True` and `total_count ≤ _AUTO_PAGE_MAX`, calls `_auto_paginate_filing()`.
4. Truncates result JSON to 6000 chars.
5. On error, sleeps `1.0 * (attempt + 1)` seconds and retries up to `max_retries` times.

##### `_trim_context(messages: list, keep_recent: int = 1, max_chars: int = 150) -> list`

Identifies all `role: "tool"` messages. Keeps the most recent `keep_recent` at full length; truncates all older ones to `max_chars` characters (appending `" …[trimmed]"`). This prevents context from ballooning across many iterations while keeping the most recent tool output fully readable.

##### `_tool_narrative(name: str, args: dict, result_str: str, had_error: bool) -> str`

Generates a human-readable one-line summary for each completed tool call, shown in the UI's live status panel. Parses the result JSON to extract the most relevant numbers (counts, totals, error messages) and formats them with bold markdown.

#### Agent loop functions

##### `run_agent(question, history, max_iterations=25) -> Generator`

Streaming agentic loop for the simple (non-step-by-step) mode:

1. Builds LLM messages from Gradio history.
2. On each iteration: calls `client.chat.completions.create()` with `tool_choice="auto"`.
3. If no tool calls and `finish_reason == "stop"`: checks for `_GIVE_UP_RE`, injects `_CONTINUATION_PROMPT` if matched, else yields the final answer.
4. Executes each tool call via `_execute_tool()`, accumulates results.
5. Yields `(stages, chat_history)` after each tool for live UI updates.
6. After `max_iterations`, forces a final answer synthesis call.

##### `_build_thinking(stages: list, running_tool: str | None) -> str`

Formats the live status shown in the chatbot while the agent is working. Lists all completed stages with their status icons; appends a "⏳ running" line for the in-progress tool.

##### `generate_plan_text(question: str, history: list) -> str`

Preliminary no-tool LLM call that generates a numbered execution plan. Called once before the main loop to populate the plan panel in the UI.

##### `_format_step_detail(iteration, messages_sent, llm_reply, tool_calls_info, tool_results_info) -> str`

Formats a complete debug breakdown for one agent iteration: messages sent (with role labels and previews), LLM response text, each tool call with its full args, and each tool result with status and preview. Stored in `state["all_step_details"]` and rendered as color-coded HTML cards in the UI.

##### `init_agent_state(question: str, history: list) -> dict`

Creates a fresh state dict for step-by-step execution. Injects the planning instruction into the first user message so the LLM writes an execution plan before calling any tools. State keys:

| Key | Type | Purpose |
|-----|------|---------|
| `llm_messages` | `list` | Full OpenAI message thread |
| `stages` | `list[Stage]` | Tool call log for UI table |
| `step` | `int` | Cumulative tool call counter |
| `iteration` | `int` | LLM call counter |
| `max_iterations` | `int` | Hard stop at 25 |
| `question` | `str` | Original question |
| `done` | `bool` | Whether the agent has finished |
| `answer` | `str` | Final answer text |
| `give_up_count` | `int` | Times continuation was injected |
| `plan` | `str` | LLM's first-response execution plan |
| `current_step_detail` | `str` | Debug text for the last step |
| `all_step_details` | `list[str]` | Accumulated debug texts |
| `tool_error_counts` | `dict` | `"toolname:error_sig"` → failure count |

##### `run_one_step(state: dict) -> Generator`

Executes exactly one agent iteration (one LLM call + all its tool calls). Yields `(state, display_text, is_final)` tuples:

- After each tool call: `is_final=False`, `display_text` contains the running narrative.
- After the LLM produces a final answer or needs to stop: `is_final=True`.

Key behaviors:
- Calls `_trim_context()` before the LLM call.
- Detects `_GIVE_UP_RE` and injects `_CONTINUATION_PROMPT`.
- Calls `_detect_batch_violation()` and injects the correction message.
- Tracks `tool_error_counts`; if the same tool+error appears ≥ 2 times, appends a forced-pivot message telling the LLM to switch strategies.

##### `_chat_history(history, question, assistant_msg) -> list`

Returns updated Gradio messages-format history by appending the new user/assistant message pair.

---

### 6.4 `pipeline.py` — Local RAG Pipeline

Self-contained pipeline using SQLite (no PostgreSQL) and numpy vectors (no Qdrant). Only external requirement is an OpenAI API key.

#### Constants

| Name | Value | Purpose |
|------|-------|---------|
| `DB_PATH` | `<package_parent>/.sec_data.db` | SQLite database file path |
| `CHUNK_TARGET` | `700` | Target characters per chunk |
| `CHUNK_MAX` | `1200` | Hard maximum characters per chunk |
| `CHUNK_MIN` | `80` | Minimum characters to keep a chunk |
| `_SEC_SECTION_RE` | regex | Matches SEC `ITEM N` headers |

#### Class: `LocalDB`

Thin SQLite wrapper. Schema has three tables: `filings`, `chunks`, `embeddings`.

```
filings:    id, ticker, company, form_type, accession, filing_date, fiscal_year, ingested_at
chunks:     id, filing_id, ticker, section, text, chunk_index, filing_date, fiscal_year
embeddings: chunk_id, vector (BLOB — float32 numpy array serialized with .tobytes())
```

Indexes: `idx_c_ticker` on `chunks(ticker)`, `idx_c_filing` on `chunks(filing_id)`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(db_path=DB_PATH)` | Creates tables if they don't exist |
| `_conn` | `() -> sqlite3.Connection` | Opens a connection with `Row` factory |
| `_init` | `()` | Runs `CREATE TABLE IF NOT EXISTS` DDL |
| `has_ticker` | `(ticker) -> bool` | True if any chunks exist for this ticker |
| `has_accession` | `(accession) -> bool` | True if this accession is in `filings` |
| `get_chunks` | `(ticker) -> list[dict]` | All chunks ordered by `filing_date DESC, chunk_index` |
| `get_embeddings` | `(chunk_ids) -> dict[str, np.ndarray]` | Deserializes stored BLOBs to float32 arrays |
| `get_ticker_meta` | `(ticker) -> dict \| None` | Latest filing row for a ticker |
| `save_filing` | `(filing_id, ticker, ...)` | `INSERT OR IGNORE` into `filings` |
| `save_chunks` | `(rows: list[dict])` | Batch `INSERT OR IGNORE` into `chunks` |
| `save_embeddings` | `(pairs: list[tuple[str, ndarray]])` | Batch `INSERT OR IGNORE`, serializes vectors |
| `delete_ticker` | `(ticker)` | Removes all chunks, embeddings, and filings for a ticker |

#### Text processing functions

##### `_strip_xbrl(text: str) -> str`

Skips past inline XBRL metadata to the actual prose of the filing:

1. Finds all `PART I/II/III` and `ITEM 1` matches.
2. Scans forward to find the first match that is:
   - After position 1000 in the document (past the cover page).
   - Followed by ≥200 chars with word density > 1.5 words per 10 chars (actual prose, not a ToC entry).
3. Returns the text from that match point onward.

##### `_detect_sections(text: str) -> list[tuple[str, str]]`

Uses `_SEC_SECTION_RE` to find all `ITEM N` headers. Returns a list of `(label, section_text)` pairs — each section runs from its header to the next header. Falls back to `[("Filing", text)]` if no headers are found.

##### `_chunk_section(section_text: str) -> list[str]`

Splits a section into chunks:

- If the section has paragraph structure (`\n\n` separators): accumulates paragraphs into a buffer, flushing when the buffer would exceed `CHUNK_MAX`.
- If the section has no paragraph structure: delegates to `_sliding_window()`.

##### `_sliding_window(text: str) -> list[str]`

Character-based chunking with overlap:

- Advances by `CHUNK_TARGET` chars per step.
- Snaps the cut point to the nearest word boundary (looks ±80 chars for a space).
- Adds 100-char overlap between consecutive chunks.
- Discards chunks shorter than `CHUNK_MIN`.

##### `build_chunks(text: str, ticker: str, filing: dict) -> list[dict]`

Orchestrates the full chunking pipeline for a filing:

1. Generates a deterministic `filing_id` from ticker and accession.
2. Calls `_strip_xbrl()` → `_detect_sections()` → `_chunk_section()` for each section.
3. Returns a list of chunk dicts ready for `LocalDB.save_chunks()`, each with a UUID-based `id`, `filing_id`, `ticker`, `section`, `text`, `chunk_index`, `filing_date`, `fiscal_year`.

#### Retrieval functions

##### `_bm25_scores(query: str, chunks: list[dict]) -> dict[str, float]`

Builds a `BM25Okapi` index over all chunk texts, queries it with the tokenized query, and returns `{chunk_id: score}`.

##### `_vector_scores(query, chunks, emb_map, model) -> dict[str, float]`

Encodes the query with the sentence-transformer model. Stacks stored chunk embeddings into a matrix, normalizes both query and chunk vectors, and computes cosine similarity via matrix-vector dot product. Returns `{chunk_id: cosine_sim}`.

##### `_hybrid_top_k(bm25, vec, chunk_map, k=5) -> list[dict]`

Combines BM25 and vector scores with a weighted sum (BM25 weight: 0.35, vector weight: 0.65) after normalizing each by its maximum. Returns the top-k chunk dicts sorted by combined score.

#### Class: `SECPipeline`

```python
class SECPipeline:
    def __init__(self)           # creates LocalDB; _embed is lazy
    embed_model                  # property: loads SentenceTransformer("all-MiniLM-L6-v2") on first access
    def run(question, ticker, form_type="10-K", force_refresh=False) -> Generator
```

##### `SECPipeline.run()`

Generator that yields `(log_text, answer, meta_markdown)` tuples. The caller can stream the log in real time; `answer` and `meta` are empty strings until the final yield.

Flow:

1. Validates OpenAI API key.
2. Optionally deletes existing data if `force_refresh=True`.
3. Checks `LocalDB.has_ticker()` — if cached, skips ingestion.
4. If not cached: calls `ticker_to_cik()` → `get_latest_filing()` → `fetch_filing_text()`.
5. Calls `build_chunks()` to split the text into chunks.
6. Embeds all chunks in one batch with `embed_model.encode()`.
7. Stores filing, chunks, and embeddings in SQLite.
8. Retrieves via `_bm25_scores()` + `_vector_scores()` + `_hybrid_top_k()`.
9. Sends top chunks to GPT-4o mini and yields the answer.

Module-level singleton accessor:

```python
def get_pipeline() -> SECPipeline:
    """Returns the process-level SECPipeline singleton."""
```

---

### 6.5 `app.py` — Gradio UI

Implements the Gradio 6 web interface. All functions are generator-based for streaming support.

#### UI helper functions

##### `_state_to_rows(state: dict | None) -> list`

Converts `state["stages"]` into a list of rows for the Gradio `DataFrame` component: `[step, tool, status, args, result_or_error]`.

##### `_render_step_boxes(state: dict | None) -> str`

Renders all accumulated step details from `state["all_step_details"]` as color-coded HTML cards. Each card has a colored header (`Step N`) and a monospace `<pre>` block with the full debug text. Returns a placeholder message if no steps exist.

##### `_update_last_assistant(history: list, new_content: str) -> list`

Scans the history list in reverse to find the last assistant message and replaces its `content`. Used to update the in-progress assistant bubble without appending a new message.

#### UI event handler functions

##### `start_chat(message, history, state)`

Generator. Called when the user submits a message:

1. Calls `init_agent_state()` to create a fresh state.
2. Immediately yields a "Running Step 1…" placeholder.
3. Calls `run_one_step()` and streams all its yields to the UI.
4. The LLM's first text response (the execution plan) is captured in `state["plan"]` and shown in the plan panel.

Yields 7-tuple: `(chatbot, stages_table, agent_state, continue_btn, stop_btn, plan_box, step_detail_box, run_all_btn)`.

##### `run_all_steps(history, state)`

Generator. Runs the full agentic loop without pausing. Loops `run_one_step()` until `state["done"]` is True, streaming UI updates continuously. Disables the continue/stop/run-all buttons during execution.

##### `continue_step(history, state)`

Generator. Runs exactly one more step after the user clicks "Continue". Shows a "▶ Running Step N…" placeholder while the step executes, then updates the chatbot with the step's result.

##### `stop_agent(history, state)`

Sets `state["done"] = True` and appends `"\n\n⏹ Stopped by user."` to the last assistant message. Non-generator — returns immediately.

#### Secondary tab functions

##### `db_status() -> tuple[list, str]`

Queries the SQLite database and returns a list of rows `[ticker, company, form_type, filing_date, chunk_count]` for the Database tab's table, plus a status message string.

##### `delete_ticker(ticker: str) -> str`

Calls `LocalDB.delete_ticker()` and returns a success or error message string.

##### `health_check() -> list`

Checks four components and returns a list of `[component, status, detail]` rows:

| Component | What is checked |
|-----------|----------------|
| OpenAI API Key | Reads `cfg.llm.api_key`; validates it is set and not the placeholder |
| SEC EDGAR | GET `https://www.sec.gov/files/company_tickers.json` with 5s timeout |
| Local SQLite DB | Opens `LocalDB`; counts filings, chunks, embeddings |
| Embedding Model (MiniLM) | Imports `sentence_transformers`; checks HuggingFace cache for `.bin` files |

---

### 6.6 `config.py` — Configuration

All configuration uses Pydantic Settings and is loaded from environment variables and/or a `.env` file.

#### Config classes

| Class | Purpose | Key fields |
|-------|---------|-----------|
| `SECConfig` | SEC API endpoints | `sec_api_base`, `sec_filings_api`, `filing_types`, `sec_rate_limit_delay` |
| `DatabaseConfig` | PostgreSQL settings (optional) | `db_host`, `db_port`, `db_name`, `db_user`, `db_password`, `db_pool_size`; property `db_url` |
| `VectorStoreConfig` | Qdrant settings (optional) | `qdrant_url`, `qdrant_api_key`, `vectorstore_embedding_model`, `vectorstore_embedding_dim` |
| `ObjectStorageConfig` | File storage paths | `storage_base_path`, `storage_raw_filings_path`, `storage_processed_filings_path` |
| `LLMConfig` | OpenAI / Anthropic | `llm_provider`, `llm_model_name`, `llm_api_key`, `llm_temperature`, `llm_max_tokens` |
| `AgentConfig` | Agent limits | `agent_max_retries`, `agent_timeout_seconds`, `agent_batch_size` |
| `RetrievalConfig` | Hybrid search weights | `retrieval_bm25_weight=0.3`, `retrieval_vector_weight=0.7`, `retrieval_initial_retrieval_k=30`, `retrieval_reranked_k=5` |
| `AppConfig` | Root config | Aggregates all sub-configs; reads `.env` |

#### `get_config() -> AppConfig`

Returns the process-level singleton `AppConfig`. Creates it on first call. All modules import this function rather than constructing their own config.

---

## 7. All Tools Reference

### Single-item tools

| Tool | Inputs | Returns |
|------|--------|---------|
| `fetch_sec_filing` | `ticker?`, `cik?`, `date?`, `form_type?`, `form_types?`, `company?`, `mode?`, `limit?`, `offset?` | `{filings, returned_count, total_count, offset, limit, has_more, next_offset, mode, error}` |
| `fetch_filing_content` | `filings=[{cik,accession}]`, `section?`, `max_chars?` | `{results, total, successful, errors, error}` |
| `count_filings_by_date` | `date`, `form_type?`, `form_types?`, `company?` | `{date, total_filings, by_form_type, filters_applied, error}` |
| `fetch_market_cap` | `ticker`, `date?` | `{market_cap, currency, source, error}` |
| `fetch_from_database` | `ticker?`, `form_type?` | `{filings, count, error}` |
| `write_to_database` | `cik`, `accession`, `company`, `form_type`, `filing_date`, `ticker?` | `{chunks_stored, filing_id, ticker, error}` |
| `extract_filing_content` | `cik`, `accession`, `section?` | `{content, sections_available, total_chars, error}` |
| `detect_events` | `text`, `event_types?` | `{events: [{event_type, person, role, action, context}], count, error}` |
| `ner_tool` | `text` | `{entities: {companies, people, financial_figures, dates, locations, tickers}, total_count, error}` |
| `semantic_search` | `query`, `ticker?`, `top_k?` | `{chunks: [{rank, ticker, section, filing_date, text}], total_searched, error}` |
| `filter_by_filing_type` | `filings`, `form_types` | `{filings, count, filtered_from, error}` |
| `classify_sections` | `text` | `{sections: [{label, char_count, preview}], count, error}` |
| `temporal_reasoning` | `query`, `filing_dates` | `{answer, dates, count, error}` |
| `deduplicate_entities` | `entities: list[str]` | `{entities, count, groups, error}` |
| `verify_confidence` | `answer`, `context`, `question?` | `{confidence, supported, explanation, error}` |

### Batch tools

| Tool | Mode | Inputs | Returns |
|------|------|--------|---------|
| `batch_extract_filing_content` | — | `items=[{cik, accession, section?}]` | `{results, total, successful, errors, error}` |
| `batch_detect_events` | A (date) | `date`, `form_type?`, `event_types?` | `{results (non-empty only), total_items, total_events, filings_with_events, error}` |
| `batch_detect_events` | B (refs) | `filings=[{cik, accession}]`, `event_types?` | same |
| `batch_detect_events` | C (text) | `items=[{text, label?}]`, `event_types?` | same |
| `batch_ner_tool` | — | `items=[{text, label?}]` | `{results, total_items, error}` |
| `batch_fetch_sec_filing` | — | `queries=[{ticker?, cik?, form_type?}]` | `{results, error}` |

### `fetch_sec_filing` pagination contract

```
returned_count  — number of filings in THIS response page (e.g. 100)
total_count     — ALL matching filings for this query (e.g. 4821)
has_more        — True if more pages exist
next_offset     — pass as `offset=` in the next call
```

The LLM must always use `total_count` for counting questions. The orchestrator auto-paginates when `total_count ≤ 300`.

---

## 8. Event Detection System

`detect_events()` returns structured events, never raw match text.

### Event types

| `event_type` | Triggered by | Key fields |
|-------------|-------------|-----------|
| `officer_appointment` | `"appointed"`, `"named"`, `"elected"`, `"designated"`, `"hired"`, `"promoted"` | `person`, `role`, `action` |
| `officer_resignation` | `"resigned"`, `"stepping down"`, `"leaving"`, `"departed"`, `"terminated"` | `person`, `role`, `action` |
| `officer_retirement` | `"retired"`, `"retiring"`, `"retirement"` | `person`, `role`, `action` |
| `acquisition` | `"acquired"`, `"merger"`, `"business combination"` | `action`, `target`, `amount` |
| `earnings` | `"net income"`, `"EPS"`, `"earnings per share"`, `"revenue"` | `action`, `amount` |
| `dividend` | `"declared dividend"` | `action`, `amount` |
| `restructuring` | `"workforce reduction"`, `"reduction in force"` | `action`, `count` |
| `legal` | `"class action"`, `"lawsuit"`, `"settlement"` | `action`, `party`, `amount` |
| `risk` | `"material weakness"`, `"going concern"` | `action` |
| `guidance` | `"outlook"`, `"guidance"`, `"expects revenue"` | `action`, `period`, `amount` |

### Deduplication

After all patterns run, events are sorted by `position`. Events of the same `event_type` and `role/action` within 200 characters of each other are collapsed into one. The `position` field is stripped before returning to the LLM.

### False-positive suppression

- **No role = no event.** Officer events without a matched `_ROLE_RE` group are discarded entirely. This eliminates false positives from section headers like `"Item 5.02. Departure of Directors or Certain Officers"`.
- **`_FAKE_NAME_TOKENS`.** Month names, SEC section vocabulary, and common prepositions are excluded from person name extraction.
- **Minimum word length.** Name candidates require words of 3+ characters, excluding single-syllable words.

### `batch_detect_events` filtering

Two filtering rules are applied before returning results to the LLM:

1. Filings with empty content are silently skipped (not returned as `count=0` entries).
2. Filings where `detect_events` finds zero events are silently dropped.

The LLM only ever sees filings that contain at least one detected event.

---

## 9. EDGAR Rate Limiting & Caching

### Rate limiter

The `_RateLimiter` token-bucket runs at 8 req/sec (EDGAR's limit is 10). Every call to `_retry_get()` consumes one token. Because the limiter uses a threading lock, it is safe across the 4-worker `ThreadPoolExecutor` in `fetch_filings_content_concurrent()` — threads compete for tokens but never violate the rate.

### In-process caches (process lifetime)

| Cache | What is stored | Key |
|-------|---------------|-----|
| `_ticker_data` | SEC's full company ticker JSON (one download per process) | — |
| `_edgar_index_cache` | Quarterly `company.idx` files (~10 MB each) | `"YYYY/QTRN"` |
| `_filing_text_cache` | Cleaned filing text | `"{accession}:{max_chars}"` |

All three caches are module-level dicts. They survive for the process lifetime and are shared across all threads.

### Retry policy

| Error | Wait | Max retries |
|-------|------|-------------|
| HTTP 429 | `Retry-After` header (default: `2^(attempt+1)` s) | 3 |
| HTTP 500/502/503/504 | `2^attempt` seconds | 3 |
| `requests.Timeout` | `2^attempt` seconds | 3 |

---

## 10. Agent Loop Mechanics

### Execution modes

| Mode | Function | Description |
|------|----------|-------------|
| Auto (streaming) | `run_agent()` | Runs all iterations internally; yields UI updates after each tool |
| Step-by-step | `run_one_step()` | User clicks "Continue" to advance one iteration at a time |
| Run all | `run_all_steps()` | Calls `run_one_step()` in a loop until done |

### Context management

Each LLM call sees:

```
[system]       SYSTEM_PROMPT (static, 574 chars)
[user]         Prior conversation history (from Gradio history)
[user]         Current question + planning instruction
[assistant]    Tool calls for step 1
[tool × N]     Tool results for step 1   ← trimmed to 150 chars after step 2+
[assistant]    Tool calls for step 2
[tool × N]     Tool results for step 2   ← trimmed to 150 chars after step 3+
...
[tool × N]     Tool results for current step   ← FULL (keep_recent=1)
```

`_trim_context(keep_recent=1, max_chars=150)` runs before every LLM call, keeping only the most recent tool output at full length.

### Premature stop detection

If the LLM produces text matching `_GIVE_UP_RE` (e.g., `"I was unable to retrieve"`) and there are still iterations remaining, `_CONTINUATION_PROMPT` is appended as a `role: "user"` message. This forces the LLM to retry every failed extraction before giving a final answer.

### Repeated-error detection

`state["tool_error_counts"]` tracks `"toolname:error_signature"` → failure count. If the same tool fails with the same error ≥ 2 times, a forced-pivot message is injected telling the LLM to change strategy entirely.

---

## 11. Local RAG Pipeline

The `SECPipeline` enables Q&A over filings stored in the local SQLite database (distinct from the real-time EDGAR path used by the agent tools).

### Ingestion flow

```
ticker
  ↓  ticker_to_cik()
CIK
  ↓  get_latest_filing()
filing metadata {accession, date, company}
  ↓  fetch_filing_text()
raw HTML → cleaned text (60,000 chars)
  ↓  _strip_xbrl()
XBRL skipped → prose text
  ↓  _detect_sections()
[(label, section_text), ...]
  ↓  _chunk_section() per section
chunks (700–1200 chars, 100-char overlap)
  ↓  SentenceTransformer.encode()
float32 embeddings (384-dim, MiniLM)
  ↓  LocalDB.save_filing() + save_chunks() + save_embeddings()
SQLite .sec_data.db
```

### Retrieval flow

```
query
  ↓
LocalDB.get_chunks(ticker)   [all chunks for the ticker]
LocalDB.get_embeddings()     [all corresponding vectors]
  ↓
_bm25_scores()               BM25Okapi keyword scores
_vector_scores()             Cosine similarity against query embedding
  ↓
_hybrid_top_k(k=5)           Combined: 0.35×BM25 + 0.65×vector
  ↓
Top-5 chunks → GPT-4o mini context window → answer
```

### Hybrid scoring formula

```
combined_score(chunk) = (bm25_score / max_bm25) * 0.35
                      + (cosine_sim / max_cosine) * 0.65
```

Both scores are normalized to [0,1] before combining. BM25 captures keyword precision; vector similarity captures semantic relevance.

---

## 12. Data Flow Diagrams

### Real-time agent query (no local DB required)

```
User question
    │
    ▼
init_agent_state()  ─── builds LLM thread
    │
    ▼
run_one_step() loop ─── up to 25 iterations
    │
    ├─ LLM call (OpenAI) ─── selects tools + args
    │
    ├─ _execute_tool()
    │       │
    │       ├─ fetch_sec_filing
    │       │       └─ get_filings_paginated()
    │       │               └─ _load_quarter_index() ─── SEC EDGAR (cached)
    │       │
    │       ├─ fetch_filing_content / extract_filing_content
    │       │       └─ fetch_filings_content_concurrent()
    │       │               └─ fetch_filing_text() ─── SEC EDGAR (cached)
    │       │
    │       ├─ detect_events / batch_detect_events
    │       │       └─ _extract_officer_events()
    │       │          _extract_non_officer_events()
    │       │
    │       └─ [other tools]
    │
    └─ Final answer → Gradio UI
```

### Local RAG query (requires prior `write_to_database`)

```
User question
    │
    ▼
semantic_search() tool
    │
    ├─ LocalDB.get_chunks(ticker)
    ├─ LocalDB.get_embeddings()
    ├─ _bm25_scores()
    ├─ _vector_scores()  ─── SentenceTransformer embedding (lazy-loaded)
    └─ _hybrid_top_k()
            │
            ▼
    Top-5 chunks → LLM context → answer
```

---

## 13. Environment Variables

All variables are read from `.env` (or shell environment). Only `LLM_API_KEY` is required.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | *(required)* | OpenAI API key (`sk-...`) |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | OpenAI model ID |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_MAX_TOKENS` | `4096` | Max tokens per LLM response |
| `DB_HOST` | `localhost` | PostgreSQL host (optional, not used by default pipeline) |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `sec_intelligence` | PostgreSQL database name |
| `DB_USER` | `sec_user` | PostgreSQL user |
| `DB_PASSWORD` | `sec_password` | PostgreSQL password |
| `VECTORSTORE_QDRANT_URL` | `http://localhost:6333` | Qdrant URL (optional) |
| `VECTORSTORE_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RETRIEVAL_BM25_WEIGHT` | `0.3` | BM25 weight in hybrid search |
| `RETRIEVAL_VECTOR_WEIGHT` | `0.7` | Vector weight in hybrid search |
| `RETRIEVAL_INITIAL_RETRIEVAL_K` | `30` | Candidates before reranking |
| `RETRIEVAL_RERANKED_K` | `5` | Final chunks after reranking |
| `DEBUG` | `false` | Enable debug logging |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Disclaimer

This system uses AI-generated analysis. Always verify critical financial information directly against original SEC filings at [sec.gov](https://www.sec.gov). Not investment advice.
