"""
Self-contained local RAG pipeline.
Uses SQLite (no PostgreSQL required) + numpy vectors (no Qdrant required).
Only external dependency: OpenAI API key.

Flow:
  1. Check SQLite for ticker data
  2. If missing → fetch from SEC EDGAR → chunk → embed → store
  3. BM25 + vector hybrid retrieval → top-k chunks
  4. GPT-4o mini answers using those chunks
"""
import re
import sqlite3
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Generator

import numpy as np
from openai import OpenAI

from sec_intelligence.config import get_config
from sec_intelligence.live_search import (
    ticker_to_cik, get_latest_filing, fetch_filing_text, get_filings_for_date,
)

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / ".sec_data.db"

# ── Chunking constants ────────────────────────────────────────────────────────
CHUNK_TARGET = 700    # target chars per chunk
CHUNK_MAX    = 1200   # hard max chars
CHUNK_MIN    = 80     # discard shorter than this

# SEC section headers (regex)
_SEC_SECTION_RE = re.compile(
    r'(?:^|\n)\s*'
    r'(ITEM\s+\d+[A-C]?(?:\s*[.:\-–]|\s+(?:(?!ITEM)[A-Z]))[^\n]{0,80})',
    re.IGNORECASE | re.MULTILINE,
)


# ── SQLite store ──────────────────────────────────────────────────────────────

class LocalDB:
    """Thin SQLite wrapper — no external services needed."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS filings (
                    id          TEXT PRIMARY KEY,
                    ticker      TEXT NOT NULL,
                    company     TEXT,
                    form_type   TEXT,
                    accession   TEXT UNIQUE,
                    filing_date TEXT,
                    fiscal_year INTEGER,
                    ingested_at TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id          TEXT PRIMARY KEY,
                    filing_id   TEXT NOT NULL,
                    ticker      TEXT NOT NULL,
                    section     TEXT,
                    text        TEXT NOT NULL,
                    chunk_index INTEGER,
                    filing_date TEXT,
                    fiscal_year INTEGER
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    vector   BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_c_ticker  ON chunks(ticker);
                CREATE INDEX IF NOT EXISTS idx_c_filing  ON chunks(filing_id);
            """)

    # ── reads ─────────────────────────────────────────────────────────────────

    def has_ticker(self, ticker: str) -> bool:
        with self._conn() as c:
            n = c.execute(
                "SELECT COUNT(*) FROM chunks WHERE ticker=?", (ticker.upper(),)
            ).fetchone()[0]
            return n > 0

    def has_accession(self, accession: str) -> bool:
        with self._conn() as c:
            n = c.execute(
                "SELECT COUNT(*) FROM filings WHERE accession=?", (accession,)
            ).fetchone()[0]
            return n > 0

    def get_chunks(self, ticker: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, section, text, filing_date, fiscal_year, chunk_index "
                "FROM chunks WHERE ticker=? ORDER BY filing_date DESC, chunk_index",
                (ticker.upper(),),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_embeddings(self, chunk_ids: list[str]) -> dict[str, np.ndarray]:
        if not chunk_ids:
            return {}
        ph = ",".join("?" * len(chunk_ids))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT chunk_id, vector FROM embeddings WHERE chunk_id IN ({ph})",
                chunk_ids,
            ).fetchall()
        return {r["chunk_id"]: np.frombuffer(r["vector"], dtype=np.float32) for r in rows}

    def get_ticker_meta(self, ticker: str) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT company, form_type, filing_date, accession "
                "FROM filings WHERE ticker=? ORDER BY filing_date DESC LIMIT 1",
                (ticker.upper(),),
            ).fetchone()
            return dict(row) if row else None

    # ── writes ────────────────────────────────────────────────────────────────

    def save_filing(self, filing_id, ticker, company, form_type,
                    accession, filing_date, fiscal_year):
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO filings "
                "(id,ticker,company,form_type,accession,filing_date,fiscal_year,ingested_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (filing_id, ticker.upper(), company, form_type, accession,
                 filing_date, fiscal_year, datetime.utcnow().isoformat()),
            )

    def save_chunks(self, rows: list[dict]):
        with self._conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO chunks "
                "(id,filing_id,ticker,section,text,chunk_index,filing_date,fiscal_year) "
                "VALUES(:id,:filing_id,:ticker,:section,:text,:chunk_index,:filing_date,:fiscal_year)",
                rows,
            )

    def save_embeddings(self, pairs: list[tuple[str, np.ndarray]]):
        with self._conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO embeddings(chunk_id,vector) VALUES(?,?)",
                [(cid, vec.tobytes()) for cid, vec in pairs],
            )

    def delete_ticker(self, ticker: str):
        """Remove all data for a ticker (used for force-refresh)."""
        with self._conn() as c:
            chunk_ids = [r[0] for r in c.execute(
                "SELECT id FROM chunks WHERE ticker=?", (ticker.upper(),)).fetchall()]
            if chunk_ids:
                ph = ",".join("?" * len(chunk_ids))
                c.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({ph})", chunk_ids)
            c.execute("DELETE FROM chunks WHERE ticker=?", (ticker.upper(),))
            c.execute("DELETE FROM filings WHERE ticker=?", (ticker.upper(),))


# ── Text processing ───────────────────────────────────────────────────────────

def _strip_xbrl(text: str) -> str:
    """
    Skip past XBRL inline metadata to the actual human-readable filing.
    Real content starts at the first substantial SEC section.
    """
    # Find all ITEM or PART matches
    matches = list(re.finditer(r'\b(PART\s+I{1,3}|ITEM\s+1)\b', text, re.IGNORECASE))
    if not matches:
        return text

    # The first hit is usually in the ToC (short lines).
    # Walk forward to find a hit followed by ≥200 chars of prose.
    for m in matches:
        tail = text[m.start():m.start() + 400]
        # Real section: next 200 chars should have many word characters
        word_density = len(re.findall(r'[a-zA-Z]{4,}', tail)) / max(1, len(tail) / 10)
        if word_density > 1.5 and m.start() > 1000:
            return text[m.start():]

    return text[matches[0].start():]


def _detect_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text into (section_label, section_text) pairs using SEC item headers.
    Falls back to unsectioned text if no headers found.
    """
    boundaries = [(m.start(), m.group(1).strip()) for m in _SEC_SECTION_RE.finditer(text)]

    if not boundaries:
        return [("Filing", text)]

    sections = []
    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) > CHUNK_MIN:
            sections.append((label, section_text))

    return sections or [("Filing", text)]


def _chunk_section(section_text: str) -> list[str]:
    """
    Split a section into overlapping chunks roughly CHUNK_TARGET chars each.
    Splits on paragraph boundaries where possible; falls back to word-boundary
    sliding window when the text has no paragraph breaks.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', section_text) if p.strip()]

    if len(paragraphs) <= 1:
        # No paragraph structure — sliding window over raw text
        return _sliding_window(section_text)

    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if buf and len(candidate) > CHUNK_MAX:
            if len(buf) >= CHUNK_MIN:
                chunks.append(buf.strip())
            buf = para          # start new chunk (no overlap for simplicity)
        else:
            buf = candidate

    if buf and len(buf) >= CHUNK_MIN:
        chunks.append(buf.strip())

    return chunks


def _sliding_window(text: str) -> list[str]:
    """Character-based chunking with word-boundary snapping and overlap."""
    chunks: list[str] = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = min(start + CHUNK_TARGET, len(text))

        # Snap to nearest word boundary
        if end < len(text):
            snap = text.rfind(" ", start + CHUNK_TARGET // 2, end + 80)
            if snap > start:
                end = snap

        chunk = text[start:end].strip()
        if len(chunk) >= CHUNK_MIN:
            chunks.append(chunk)

        overlap = min(100, end - start)
        start = end - overlap
        if start >= len(text) - CHUNK_MIN:
            break

    return chunks


def build_chunks(text: str, ticker: str, filing: dict) -> list[dict]:
    """Turn raw filing text into chunk dicts ready for DB insertion."""
    filing_id = f"f_{ticker}_{filing['accession'].replace('-', '')}"
    clean = _strip_xbrl(text)
    sections = _detect_sections(clean)

    rows: list[dict] = []
    idx = 0

    for section_label, section_text in sections:
        for chunk_text in _chunk_section(section_text):
            if len(chunk_text) < CHUNK_MIN:
                continue
            rows.append({
                "id": f"c_{ticker}_{idx}_{uuid.uuid4().hex[:6]}",
                "filing_id": filing_id,
                "ticker": ticker.upper(),
                "section": section_label[:120],
                "text": chunk_text,
                "chunk_index": idx,
                "filing_date": filing["date"],
                "fiscal_year": int(filing["date"][:4]),
            })
            idx += 1

    return rows


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _bm25_scores(query: str, chunks: list[dict]) -> dict[str, float]:
    from rank_bm25 import BM25Okapi
    texts = [c["text"] for c in chunks]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    return {chunks[i]["id"]: float(scores[i]) for i in range(len(chunks))}


def _vector_scores(
    query: str, chunks: list[dict], emb_map: dict[str, np.ndarray], model
) -> dict[str, float]:
    q_vec = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)

    valid = [(c, emb_map[c["id"]]) for c in chunks if c["id"] in emb_map]
    if not valid:
        return {}

    ids = [c["id"] for c, _ in valid]
    vecs = np.stack([v for _, v in valid])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    sims = (vecs / norms) @ q_norm

    return {ids[i]: float(sims[i]) for i in range(len(ids))}


def _hybrid_top_k(
    bm25: dict[str, float],
    vec: dict[str, float],
    chunk_map: dict[str, dict],
    k: int = 5,
) -> list[dict]:
    all_ids = set(bm25) | set(vec)
    max_b = max(bm25.values(), default=1) or 1
    max_v = max(vec.values(), default=1) or 1

    combined = {
        cid: (bm25.get(cid, 0) / max_b) * 0.35 + (vec.get(cid, 0) / max_v) * 0.65
        for cid in all_ids
    }
    top = sorted(combined, key=combined.__getitem__, reverse=True)[:k]
    return [chunk_map[cid] for cid in top if cid in chunk_map]


# ── Main pipeline ─────────────────────────────────────────────────────────────

class SECPipeline:
    """
    Orchestrates: DB check → EDGAR fetch → chunk → embed → store → retrieve → answer.
    Designed as a generator so the caller can stream status log messages.
    """

    def __init__(self):
        self.db = LocalDB()
        self.config = get_config()
        self._embed = None  # lazy-loaded

    @property
    def embed_model(self):
        if self._embed is None:
            from sentence_transformers import SentenceTransformer
            self._embed = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed

    # ── public entry point ────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        ticker: str,
        form_type: str = "10-K",
        force_refresh: bool = False,
    ) -> Generator[tuple[str, str, str], None, None]:
        """
        Yields (log_text, answer, meta_markdown) tuples.
        answer and meta are empty strings until the final yield.
        """
        log: list[str] = []

        def emit(msg: str, *, answer="", meta=""):
            ts = datetime.now().strftime("%H:%M:%S")
            log.append(f"[{ts}]  {msg}")
            return "\n".join(log), answer, meta

        ticker = ticker.strip().upper()

        # ── validate config ───────────────────────────────────────────────────
        api_key = self.config.llm.api_key or ""
        if not api_key or api_key.startswith("your_"):
            yield emit("❌  OpenAI API key not set. Add LLM_API_KEY=sk-... to your .env file.")
            return

        # ── check / populate DB ───────────────────────────────────────────────
        if force_refresh and self.db.has_ticker(ticker):
            yield emit(f"♻️  Force-refresh: removing existing {ticker} data...")
            self.db.delete_ticker(ticker)

        if self.db.has_ticker(ticker):
            meta_row = self.db.get_ticker_meta(ticker)
            yield emit(
                f"✅  Found existing {ticker} data in local database "
                f"({meta_row['filing_date'] if meta_row else '?'})."
            )
        else:
            yield from self._ingest(ticker, form_type, log, emit)
            # Check ingestion succeeded
            if not self.db.has_ticker(ticker):
                return

        # ── retrieve ──────────────────────────────────────────────────────────
        yield emit("📂  Loading chunks from database...")
        chunks = self.db.get_chunks(ticker)
        yield emit(f"📦  {len(chunks)} chunks available for {ticker}.")

        chunk_map = {c["id"]: c for c in chunks}

        yield emit("🔍  Running BM25 keyword search...")
        bm25 = _bm25_scores(question, chunks)
        top_bm25 = sorted(bm25, key=bm25.__getitem__, reverse=True)[:5]
        yield emit(
            f"📊  BM25 top match score: {bm25[top_bm25[0]]:.3f} — "
            f"section: {chunk_map[top_bm25[0]]['section'][:60]}"
        )

        yield emit("🧠  Running semantic vector search...")
        emb_map = self.db.get_embeddings(list(chunk_map.keys()))
        if not emb_map:
            yield emit("⚠️  No embeddings found — re-embedding now...")
            texts = [c["text"] for c in chunks]
            vecs = self.embed_model.encode(texts, show_progress_bar=False,
                                            convert_to_numpy=True).astype(np.float32)
            self.db.save_embeddings([(chunks[i]["id"], vecs[i]) for i in range(len(chunks))])
            emb_map = self.db.get_embeddings(list(chunk_map.keys()))

        vec = _vector_scores(question, chunks, emb_map, self.embed_model)
        top_vec = sorted(vec, key=vec.__getitem__, reverse=True)[:1]
        yield emit(
            f"🎯  Vector top match score: {vec[top_vec[0]]:.3f} — "
            f"section: {chunk_map[top_vec[0]]['section'][:60]}"
            if top_vec else "🎯  Vector search returned no results."
        )

        yield emit("⚖️  Fusing BM25 + vector scores (35% / 65%)...")
        top_chunks = _hybrid_top_k(bm25, vec, chunk_map, k=5)
        yield emit(f"✂️  Selected top {len(top_chunks)} chunks for context window.")

        for i, c in enumerate(top_chunks, 1):
            yield emit(f"   Chunk {i}: [{c['section'][:50]}] {c['text'][:80]}…")

        # ── answer ────────────────────────────────────────────────────────────
        yield emit(f"💬  Sending question + {len(top_chunks)} chunks to GPT-4o mini...")

        meta_row = self.db.get_ticker_meta(ticker) or {}
        context = "\n\n---\n\n".join(
            f"[{c['section']}]\n{c['text']}" for c in top_chunks
        )

        try:
            answer = self._generate_answer(question, context, meta_row)
        except Exception as e:
            yield emit(f"❌  GPT-4o mini call failed: {e}")
            return

        meta_md = (
            f"**Company:** {meta_row.get('company', ticker)}  \n"
            f"**Filing:** {meta_row.get('form_type', form_type)} — {meta_row.get('filing_date', '?')}  \n"
            f"**Accession:** {meta_row.get('accession', 'N/A')}"
        )

        log.append("")
        log.append("✅  Pipeline complete.")
        yield "\n".join(log), answer, meta_md

    # ── ingestion ─────────────────────────────────────────────────────────────

    def _ingest(self, ticker, form_type, log, emit):
        yield emit(f"🔎  No local data for {ticker}. Fetching from SEC EDGAR...")

        yield emit("🌐  Looking up CIK on SEC EDGAR...")
        cik = ticker_to_cik(ticker)
        if not cik:
            yield emit(f"❌  Could not find CIK for '{ticker}'. Check the ticker symbol.")
            return
        yield emit(f"✅  CIK found: {int(cik)}")

        yield emit(f"📋  Looking up latest {form_type} filing...")
        filing = get_latest_filing(cik, form_type)
        if not filing:
            yield emit(f"❌  No {form_type} found on EDGAR for {ticker}.")
            return
        yield emit(
            f"✅  Found: {filing['accession']}  "
            f"({filing['company']} — filed {filing['date']})"
        )

        yield emit("⬇️   Downloading filing HTML from SEC EDGAR (may take ~10s)...")
        text = fetch_filing_text(cik, filing["accession"], max_chars=300_000)
        if not text or len(text) < 500:
            yield emit("❌  Could not download filing text. SEC EDGAR may be throttling — retry in a moment.")
            return
        yield emit(f"📄  Downloaded {len(text):,} characters.")

        yield emit("✂️   Stripping XBRL header and detecting SEC sections...")
        chunks = build_chunks(text, ticker, filing)
        if not chunks:
            yield emit("❌  No chunks produced — the document may be in an unexpected format.")
            return
        yield emit(f"📦  Created {len(chunks)} chunks across {len(set(c['section'] for c in chunks))} sections.")

        yield emit("🧬  Loading embedding model (first run downloads ~90 MB, cached after)...")
        texts = [c["text"] for c in chunks]
        vecs = self.embed_model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True, batch_size=32
        ).astype(np.float32)
        yield emit(f"✅  Embedded {len(vecs)} chunks (dim={vecs.shape[1]}).")

        yield emit("💾  Storing chunks and embeddings in local database...")
        filing_id = f"f_{ticker}_{filing['accession'].replace('-', '')}"
        self.db.save_filing(
            filing_id, ticker, filing["company"], form_type,
            filing["accession"], filing["date"], int(filing["date"][:4]),
        )
        self.db.save_chunks(chunks)
        self.db.save_embeddings([(chunks[i]["id"], vecs[i]) for i in range(len(chunks))])
        yield emit(f"✅  Stored {len(chunks)} chunks in .sec_data.db — won't re-download next time.")

    # ── known-filing ingestion (used by bulk date ingest) ─────────────────────

    def _ingest_known_filing(
        self,
        cik: str,
        accession: str,
        company: str,
        form_type: str,
        filing_date: str,
        ticker: str,
        log: list,
        emit,
    ):
        """
        Generator: ingest one filing whose metadata is already known.
        Yields whatever `emit(msg)` returns (caller controls the tuple shape).
        """
        if self.db.has_accession(accession):
            yield emit(f"   ⏭️  Already in DB — skipping.")
            return

        yield emit(f"   ⬇️  Downloading {form_type} text from EDGAR...")
        text = fetch_filing_text(cik, accession, max_chars=300_000)
        if not text or len(text) < 500:
            yield emit(f"   ⚠️  Download failed or document too short — skipping.")
            return
        yield emit(f"   📄  {len(text):,} chars downloaded.")

        filing_dict = {"accession": accession, "date": filing_date}
        chunks = build_chunks(text, ticker, filing_dict)
        if not chunks:
            yield emit(f"   ⚠️  No usable chunks produced — skipping.")
            return
        n_sections = len(set(c["section"] for c in chunks))
        yield emit(f"   ✂️  {len(chunks)} chunks across {n_sections} sections.")

        vecs = self.embed_model.encode(
            [c["text"] for c in chunks],
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
        ).astype(np.float32)

        filing_id = f"f_{ticker}_{accession.replace('-', '')}"
        self.db.save_filing(
            filing_id, ticker, company, form_type,
            accession, filing_date, int(filing_date[:4]),
        )
        self.db.save_chunks(chunks)
        self.db.save_embeddings([(chunks[i]["id"], vecs[i]) for i in range(len(chunks))])
        yield emit(f"   ✅  Stored {len(chunks)} chunks.")

    # ── bulk date ingestion ───────────────────────────────────────────────────

    def ingest_by_date(
        self,
        date_str: str,
        form_types: list[str] | None = None,
        max_per_type: int = 20,
    ) -> Generator[tuple[str, list], None, None]:
        """
        Yields (log_text, summary_rows) as filings are discovered and ingested.
        summary_rows: list of [ticker, company, form_type, filing_date, status]
        """
        from collections import defaultdict

        log: list[str] = []
        summary: list[list] = []

        def emit(msg: str) -> tuple[str, list]:
            ts = datetime.now().strftime("%H:%M:%S")
            log.append(f"[{ts}]  {msg}")
            return "\n".join(log), list(summary)

        yield emit(f"🔍  Discovering SEC filings for {date_str}...")

        try:
            filings = get_filings_for_date(date_str, form_types or None)
        except Exception as e:
            yield emit(f"❌  Failed to fetch EDGAR full-index: {e}")
            return

        if not filings:
            yield emit(f"ℹ️   No filings found for {date_str} with the selected filters.")
            return

        yield emit(f"📋  Found {len(filings)} filings on {date_str}.")

        # Cap per form type so a single day doesn't flood the DB
        if max_per_type > 0:
            type_counts: dict = defaultdict(int)
            capped = []
            for f in filings:
                ft = f["form_type"]
                if type_counts[ft] < max_per_type:
                    capped.append(f)
                    type_counts[ft] += 1
            if len(capped) < len(filings):
                yield emit(
                    f"⚠️   Capped to {len(capped)} filings "
                    f"({max_per_type} per form type to limit runtime)."
                )
            filings = capped

        total = len(filings)
        yield emit(f"⚙️   Ingesting {total} filings — this may take several minutes...")

        for i, f in enumerate(filings, 1):
            raw_ticker = f.get("ticker") or ""
            company    = f["company"]
            form_type  = f["form_type"]
            cik        = f["cik"]
            accession  = f["accession"]
            filing_date = f["filing_date"]
            eff_ticker = raw_ticker or f"CIK{int(cik)}"

            yield emit(
                f"\n[{i}/{total}]  {eff_ticker} — {form_type} — {company[:40]}"
            )

            status = "skipped (cached)"
            if self.db.has_accession(accession):
                yield emit(f"   ⏭️  Already in DB.")
            else:
                try:
                    for step in self._ingest_known_filing(
                        cik, accession, company, form_type,
                        filing_date, eff_ticker, log, emit,
                    ):
                        yield step
                    status = (
                        "✅ ingested" if self.db.has_accession(accession) else "⚠️ failed"
                    )
                except Exception as e:
                    yield emit(f"   ❌  Unexpected error: {e}")
                    status = "❌ error"

            summary.append([eff_ticker, company[:40], form_type, filing_date, status])
            yield emit(f"   {i}/{total} processed.")

        log.append("")
        log.append(f"🎉  Bulk ingestion complete — {total} filings processed.")
        yield "\n".join(log), list(summary)

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _generate_answer(self, question: str, context: str, meta: dict) -> str:
        client = OpenAI(api_key=self.config.llm.api_key)
        company = meta.get("company", "the company")
        date = meta.get("filing_date", "unknown date")

        prompt = (
            f"You are a financial analyst reviewing {company}'s SEC filing dated {date}.\n\n"
            f"Answer the question below using ONLY the provided SEC filing excerpts. "
            f"Be specific, quote relevant facts, and note if the excerpts don't fully cover the question.\n\n"
            f"QUESTION: {question}\n\n"
            f"SEC FILING EXCERPTS:\n{context}\n\n"
            f"ANSWER:"
        )
        resp = client.chat.completions.create(
            model=self.config.llm.model_name,
            max_tokens=900,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()


# ── Singleton ─────────────────────────────────────────────────────────────────

_pipeline: SECPipeline | None = None


def get_pipeline() -> SECPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SECPipeline()
    return _pipeline
