"""
SEC Intelligence Agent — Gradio UI
Run:  python app.py
Open: http://localhost:7860
"""
import gradio as gr


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state_to_rows(state: dict | None) -> list:
    if not state:
        return []
    return [
        [s["step"], s["tool"], s["status"], s["args"],
         s["result"] or s.get("error", "")]
        for s in state.get("stages", [])
    ]


def _render_step_boxes(state: dict | None) -> str:
    """Render all accumulated step details as colour-coded HTML cards."""
    if not state:
        return "<p style='color:#888;font-style:italic;padding:8px;'>Step details appear here after each agent iteration.</p>"
    steps = state.get("all_step_details", [])
    if not steps:
        return "<p style='color:#888;font-style:italic;padding:8px;'>Step details appear here after each agent iteration.</p>"

    import html as _html
    _COLORS = ["#1565c0", "#2e7d32", "#e65100", "#6a1b9a", "#00838f", "#c62828", "#f57f17"]

    parts = ['<div style="display:flex;flex-direction:column;gap:10px;padding:4px;">']
    for i, detail in enumerate(steps):
        color = _COLORS[i % len(_COLORS)]
        escaped = _html.escape(detail)
        parts.append(
            f'<div style="border:1px solid #333;border-radius:8px;overflow:hidden;">'
            f'<div style="background:{color};color:#fff;padding:6px 14px;font-weight:600;'
            f'font-size:13px;letter-spacing:.3px;">Step {i + 1}</div>'
            f'<pre style="margin:0;padding:10px 14px;font-size:11.5px;line-height:1.6;'
            f'background:#111;color:#e8e8e8;overflow-x:auto;white-space:pre-wrap;'
            f'word-wrap:break-word;">{escaped}</pre>'
            f'</div>'
        )
    parts.append('</div>')
    return ''.join(parts)


def _update_last_assistant(history: list, new_content: str) -> list:
    """Replace the content of the last assistant message in history."""
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "assistant":
            updated = list(history)
            updated[i] = {"role": "assistant", "content": new_content}
            return updated
    return history


# ── Step 1: user sends a message ──────────────────────────────────────────────

def start_chat(message: str, history: list, state: dict):
    """
    Initialise the agent state and run Step 1.
    Generator — yields (chatbot, stages_table, agent_state,
                        continue_btn_update, stop_btn_update, plan_box, step_detail_box).
    """
    if not message or not message.strip():
        yield history, _state_to_rows(state), state, \
              gr.update(interactive=False), gr.update(interactive=False), \
              gr.update(), _render_step_boxes(state), gr.update(interactive=False)
        return

    from sec_intelligence.agent.planner import init_agent_state, run_one_step

    question = message.strip()
    new_state = init_agent_state(question, history)

    # Show "thinking" immediately
    pending = history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": "🤔 Running Step 1…"},
    ]
    yield pending, [], new_state, \
          gr.update(interactive=False), gr.update(interactive=False), "", \
          _render_step_boxes(None), gr.update(interactive=False)

    # Run step 1 — the LLM's first text response becomes the plan
    for new_state, display, is_final in run_one_step(new_state):
        plan_text = new_state.get("plan", "")
        active = (not new_state["done"]) if is_final else False
        new_history = history + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": display},
        ]
        yield new_history, _state_to_rows(new_state), new_state, \
              gr.update(interactive=active), gr.update(interactive=active), \
              plan_text, _render_step_boxes(new_state), gr.update(interactive=active)


# ── Run All: run every remaining step without pausing ─────────────────────────

def run_all_steps(history: list, state: dict):
    """
    Run all remaining agent steps automatically without human confirmation.
    Generator — same outputs as start_chat.
    """
    if not state or state.get("done"):
        yield history, _state_to_rows(state), state, \
              gr.update(interactive=False), gr.update(interactive=False), \
              gr.update(), _render_step_boxes(state), gr.update(interactive=False)
        return

    from sec_intelligence.agent.planner import run_one_step

    plan_text = state.get("plan", "")

    while not state.get("done"):
        for state, display, _ in run_one_step(state):
            history = _update_last_assistant(history, display)
            yield history, _state_to_rows(state), state, \
                  gr.update(interactive=False), gr.update(interactive=False), \
                  plan_text, _render_step_boxes(state), gr.update(interactive=False)

    yield history, _state_to_rows(state), state, \
          gr.update(interactive=False), gr.update(interactive=False), \
          plan_text, _render_step_boxes(state), gr.update(interactive=False)


# ── Continue: user confirms next step ─────────────────────────────────────────

def continue_step(history: list, state: dict):
    """
    Run the next agent step after user confirmation.
    Generator — same outputs as start_chat.
    """
    if not state or state.get("done"):
        yield history, _state_to_rows(state), state, \
              gr.update(interactive=False), gr.update(interactive=False), \
              gr.update(), _render_step_boxes(state), gr.update(interactive=False)
        return

    from sec_intelligence.agent.planner import run_one_step

    plan_text = state.get("plan", "")

    # Show "running next step" while working
    interim = _update_last_assistant(
        history,
        f"▶ Running Step {state['iteration'] + 1}…",
    )
    yield interim, _state_to_rows(state), state, \
          gr.update(interactive=False), gr.update(interactive=False), \
          plan_text, _render_step_boxes(state), gr.update(interactive=False)

    for state, display, is_final in run_one_step(state):
        active = (not state["done"]) if is_final else False
        new_history = _update_last_assistant(history, display)
        yield new_history, _state_to_rows(state), state, \
              gr.update(interactive=active), gr.update(interactive=active), \
              plan_text, _render_step_boxes(state), gr.update(interactive=active)


# ── Stop: user aborts ─────────────────────────────────────────────────────────

def stop_agent(history: list, state: dict):
    """End the loop immediately."""
    if state:
        state["done"] = True
    new_history = _update_last_assistant(
        history,
        (_update_last_assistant(history, "")[-1]["content"]
         if history else "") + "\n\n⏹ **Stopped by user.**",
    )
    # Rebuild cleanly
    if history:
        last_content = history[-1].get("content", "")
        if history[-1].get("role") == "assistant":
            new_history = history[:-1] + [
                {"role": "assistant",
                 "content": last_content + "\n\n⏹ **Stopped by user.**"}
            ]
        else:
            new_history = history
    else:
        new_history = history
    return new_history, _state_to_rows(state), state, \
           gr.update(interactive=False), gr.update(interactive=False), \
           state.get("plan", "") if state else "", \
           _render_step_boxes(state), gr.update(interactive=False)


# ── DB helpers (used by secondary tabs) ──────────────────────────────────────

def db_status():
    try:
        from sec_intelligence.pipeline import LocalDB, DB_PATH
        db = LocalDB(DB_PATH)
        with db._conn() as c:
            rows = c.execute(
                "SELECT f.ticker, f.company, f.form_type, f.filing_date, "
                "COUNT(ch.id) AS chunks "
                "FROM filings f LEFT JOIN chunks ch ON ch.filing_id = f.id "
                "GROUP BY f.id ORDER BY f.filing_date DESC"
            ).fetchall()
        if not rows:
            return [["—", "—", "—", "—", 0]], "Database is empty."
        data = [[r["ticker"], r["company"], r["form_type"],
                 r["filing_date"], r["chunks"]] for r in rows]
        return data, f"{len(data)} filing(s) in `.sec_data.db`"
    except Exception as e:
        return [], f"Error: {e}"


def delete_ticker(ticker: str):
    if not ticker.strip():
        return "Enter a ticker first."
    try:
        from sec_intelligence.pipeline import LocalDB, DB_PATH
        LocalDB(DB_PATH).delete_ticker(ticker.strip().upper())
        return f"✅ All data for {ticker.upper()} removed."
    except Exception as e:
        return f"❌ {e}"


def health_check():
    rows = []

    # OpenAI key
    try:
        from sec_intelligence.config import get_config
        cfg = get_config()
        key = cfg.llm.api_key or ""
        if key and not key.startswith("your_"):
            rows.append(["OpenAI API Key", "✅ Set",
                         f"Model: {cfg.llm.model_name}  key: {key[:8]}…"])
        else:
            rows.append(["OpenAI API Key", "❌ Missing",
                         "Set LLM_API_KEY=sk-… in .env"])
    except Exception as e:
        rows.append(["OpenAI API Key", "❌ Error", str(e)])

    # SEC EDGAR
    try:
        import requests
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "SEC Intelligence Platform research@secplatform.com"},
            timeout=5,
        )
        rows.append(["SEC EDGAR", "✅ Reachable", f"HTTP {r.status_code}"])
    except Exception as e:
        rows.append(["SEC EDGAR", "❌ Unreachable", str(e)[:80]])

    # SQLite DB
    try:
        from sec_intelligence.pipeline import LocalDB, DB_PATH
        db = LocalDB(DB_PATH)
        with db._conn() as c:
            nf = c.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
            nc = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            ne = c.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        rows.append(["Local SQLite DB", "✅ Ready",
                     f"{nf} filings · {nc} chunks · {ne} embeddings"])
    except Exception as e:
        rows.append(["Local SQLite DB", "❌ Error", str(e)])

    # Embedding model
    try:
        from sentence_transformers import SentenceTransformer
        import pathlib
        cache = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
        cached = any("MiniLM" in str(p) for p in cache.rglob("*.bin")) if cache.exists() else False
        rows.append(["Embedding Model (MiniLM)", "✅ Available",
                     "Cached locally" if cached else "Will download ~90 MB on first use"])
    except Exception as e:
        rows.append(["Embedding Model (MiniLM)", "❌ Error", str(e)])

    # yfinance (optional)
    try:
        import yfinance
        rows.append(["yfinance (market cap)", "✅ Installed", f"v{yfinance.__version__}"])
    except ImportError:
        rows.append(["yfinance (market cap)", "⚠️ Not installed",
                     "pip install yfinance  — needed for fetch_market_cap"])

    # PostgreSQL (optional)
    try:
        from sec_intelligence.storage import get_postgres_db
        import sqlalchemy
        db2 = get_postgres_db()
        s = db2.get_session()
        s.execute(sqlalchemy.text("SELECT 1"))
        s.close()
        rows.append(["PostgreSQL (optional)", "✅ Connected", "Full pipeline available"])
    except Exception:
        rows.append(["PostgreSQL (optional)", "⚠️ Not running",
                     "Not needed for the agent"])

    # Qdrant (optional)
    try:
        from sec_intelligence.storage import get_vector_store
        vs = get_vector_store()
        vs.client.get_collections()
        rows.append(["Qdrant (optional)", "✅ Connected", "Full pipeline available"])
    except Exception:
        rows.append(["Qdrant (optional)", "⚠️ Not running", "Not needed for the agent"])

    return rows


# ── UI ────────────────────────────────────────────────────────────────────────

_EXAMPLE_PROMPTS = [
    "What are NVDA's main AI-related risks in their latest 10-K?",
    "Fetch all 8-K filings from 2025-01-13 and tell me what events occurred.",
    "What was Apple's revenue and net income in their most recent 10-K?",
    "Search for information about Microsoft's cloud segment in stored filings.",
    "What is Tesla's market cap? Also show me their latest 10-K risk factors.",
    "Find all filings for GOOGL in the database and summarize what's stored.",
    "Extract named entities from META's latest 10-K.",
    "What cybersecurity incidents did Meta disclose in their annual report?",
]

with gr.Blocks(title="SEC Intelligence Agent") as demo:

    # ── Persistent agent state (survives between button clicks) ───────────────
    agent_state = gr.State(None)

    # ── Header ─────────────────────────────────────────────────────────────────
    gr.Markdown(
        "# 🏛️ SEC Intelligence Agent\n"
        "_Step-by-step agentic loop — you confirm each iteration before it continues_"
    )

    # ── Tab 1: Main chat ───────────────────────────────────────────────────────
    with gr.Tab("💬 Agent Chat"):
        with gr.Row(equal_height=True):

            # ── Left: chat + controls ─────────────────────────────────────────
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    label="",
                    height=480,
                    show_label=False,
                )

                # Input row
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask anything about SEC filings…",
                        label="",
                        scale=7,
                        container=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send ▶", variant="primary", scale=1, min_width=80)

                # Step-control row
                with gr.Row():
                    continue_btn = gr.Button(
                        "▶ Next step",
                        variant="primary",
                        interactive=False,
                        scale=2,
                    )
                    run_all_btn = gr.Button(
                        "⚡ Run all steps",
                        variant="primary",
                        interactive=False,
                        scale=2,
                    )
                    stop_btn = gr.Button(
                        "⏹ Stop",
                        variant="stop",
                        interactive=False,
                        scale=1,
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        scale=1,
                    )

                gr.Markdown(
                    "_After each step the agent **pauses** and waits for you to click "
                    "**▶ Continue** or **⏹ Stop**._"
                )

            # ── Right: plan + live debug panel ────────────────────────────────
            with gr.Column(scale=4):
                gr.Markdown("### 📋 Execution Plan")
                plan_box = gr.Textbox(
                    label="",
                    show_label=False,
                    lines=7,
                    interactive=False,
                    placeholder="Execution plan appears here after your first message…",
                )
                gr.Markdown("### 🔍 Agent Pipeline — Step Trace")
                stages_table = gr.Dataframe(
                    headers=["#", "Tool", "Status", "Arguments", "Result"],
                    datatype=["number", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    label="",
                    show_label=False,
                )
                gr.Markdown(
                    "_Each row = one tool call. "
                    "⏳ running → ✅ done / ❌ error / ⚠️ forced_"
                )

        # ── Step detail — full-width panel below main row ─────────────────────
        gr.Markdown("### 📝 Step Detail — Full Briefing")
        step_detail_box = gr.HTML(
            value="<p style='color:#888;font-style:italic;padding:8px;'>Step details appear here after each agent iteration.</p>",
        )

        # ── Shared outputs list ───────────────────────────────────────────────
        _CHAT_OUTPUTS = [chatbot, stages_table, agent_state, continue_btn, stop_btn,
                         plan_box, step_detail_box, run_all_btn]

        # Send button / Enter key → start_chat
        send_btn.click(
            fn=start_chat,
            inputs=[msg_box, chatbot, agent_state],
            outputs=_CHAT_OUTPUTS,
        ).then(fn=lambda: "", outputs=msg_box)

        msg_box.submit(
            fn=start_chat,
            inputs=[msg_box, chatbot, agent_state],
            outputs=_CHAT_OUTPUTS,
        ).then(fn=lambda: "", outputs=msg_box)

        # Continue button → continue_step
        continue_btn.click(
            fn=continue_step,
            inputs=[chatbot, agent_state],
            outputs=_CHAT_OUTPUTS,
        )

        # Run All button → run_all_steps (no pause between steps)
        run_all_btn.click(
            fn=run_all_steps,
            inputs=[chatbot, agent_state],
            outputs=_CHAT_OUTPUTS,
        )

        # Stop button → stop_agent
        stop_btn.click(
            fn=stop_agent,
            inputs=[chatbot, agent_state],
            outputs=_CHAT_OUTPUTS,
            queue=False,
        )

        # Clear → reset everything
        clear_btn.click(
            fn=lambda: ([], [], None,
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        "",
                        "<p style='color:#888;font-style:italic;padding:8px;'>Step details appear here after each agent iteration.</p>",
                        gr.update(interactive=False)),
            outputs=_CHAT_OUTPUTS,
            queue=False,
        )

    # ── Tab 2: Local database browser ─────────────────────────────────────────
    with gr.Tab("🗄️ Local Database"):
        gr.Markdown(
            "Filings stored in `.sec_data.db` (SQLite + numpy embeddings).  \n"
            "The agent auto-populates this when it writes filings — no manual steps needed."
        )
        db_refresh_btn = gr.Button("Refresh", variant="secondary")
        db_table = gr.Dataframe(
            headers=["Ticker", "Company", "Type", "Filed", "Chunks"],
            datatype=["str", "str", "str", "str", "number"],
            interactive=False,
            wrap=True,
        )
        db_info = gr.Markdown()
        db_refresh_btn.click(fn=db_status, outputs=[db_table, db_info])

        gr.Markdown("---\n### Delete a ticker's data")
        with gr.Row():
            del_ticker_box = gr.Textbox(
                label="Ticker to delete", placeholder="NVDA", scale=1
            )
            del_btn = gr.Button("Delete", variant="stop", scale=1)
        del_status = gr.Markdown()
        del_btn.click(fn=delete_ticker, inputs=del_ticker_box, outputs=del_status)

    # ── Tab 3: System status ───────────────────────────────────────────────────
    with gr.Tab("🔌 System Status"):
        gr.Markdown("Health check for all services and dependencies.")
        hc_btn = gr.Button("Run Health Check", variant="primary")
        hc_table = gr.Dataframe(
            headers=["Service", "Status", "Detail"],
            datatype=["str", "str", "str"],
            interactive=False,
            wrap=True,
        )
        hc_btn.click(fn=health_check, outputs=hc_table)
        gr.Markdown(
            "| Service | Required for |\n"
            "|---|---|\n"
            "| **OpenAI API key** | All LLM calls |\n"
            "| **SEC EDGAR** | Fetching filings (public API) |\n"
            "| **Local SQLite** | Storing chunks (auto-created) |\n"
            "| **Embedding model** | Semantic search (auto-downloaded ~90 MB) |\n"
            "| **yfinance** | `fetch_market_cap` tool |\n"
            "| **PostgreSQL / Qdrant** | Optional full pipeline only |"
        )

    gr.Markdown(
        "_SEC Intelligence Agent v0.2 · "
        "No Docker required · "
        "All storage is local SQLite + numpy_"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
