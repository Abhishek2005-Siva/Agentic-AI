"""
SEC Intelligence Agent — Streamlit UI
Run:  streamlit run streamlit_app.py

Same agent as the Gradio app (app.py): the planner in sec_intelligence.agent
drives an OpenAI function-calling loop over live SEC EDGAR tools. Visitors paste
their own OpenAI key in the sidebar; it is kept in their session only.
"""
import threading

import streamlit as st

st.set_page_config(page_title="SEC Intelligence Agent", page_icon="📈", layout="wide")

# The agent reads its API key from a process-wide config object. To keep one
# visitor's key from ever being used for another visitor's request, every agent
# step runs while holding this lock, with the key set just before it runs.
_AGENT_LOCK = threading.Lock()

EXAMPLES = [
    "What were Apple's most recent 10-K risk factors about supply chain?",
    "Which companies announced a CEO change in 8-K filings this month?",
    "Summarize the latest 10-Q for NVIDIA and list key financial figures.",
]


def _init_session() -> None:
    defaults = {"history": [], "agent": None, "run_mode": None, "pending": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset() -> None:
    st.session_state.history = []
    st.session_state.agent = None
    st.session_state.run_mode = None
    st.session_state.pending = None


def _set_last_assistant(content: str) -> None:
    history = st.session_state.history
    if history and history[-1]["role"] == "assistant":
        history[-1]["content"] = content
    else:
        history.append({"role": "assistant", "content": content})


def _render_trace(agent: dict) -> None:
    """Plan, tool calls and per-step detail for the current question."""
    plan = agent.get("plan")
    stages = agent.get("stages", [])
    details = agent.get("all_step_details", [])
    if not (plan or stages or details):
        return
    with st.expander("Agent trace", expanded=False):
        if plan:
            st.markdown("**Plan**")
            st.markdown(plan)
        if stages:
            st.markdown("**Tool calls**")
            st.dataframe(
                [
                    {
                        "step": s["step"],
                        "tool": s["tool"],
                        "status": s["status"],
                        "args": str(s["args"]),
                        "result": str(s["result"] or s.get("error", "")),
                    }
                    for s in stages
                ],
                width="stretch",
                hide_index=True,
            )
        for i, detail in enumerate(details, start=1):
            st.markdown(f"**Step {i}**")
            st.code(detail, language="text")


def _run_steps(api_key: str, placeholder) -> None:
    """Run one step (step mode) or all remaining steps, streaming into placeholder."""
    from sec_intelligence.agent.planner import run_one_step
    from sec_intelligence.config import get_config

    agent = st.session_state.agent
    step_by_step = st.session_state.run_mode == "step"
    st.session_state.run_mode = None

    with _AGENT_LOCK:
        get_config().llm.llm_api_key = api_key
        try:
            while not agent.get("done"):
                for agent, display, _ in run_one_step(agent):
                    placeholder.markdown(display)
                    _set_last_assistant(display)
                if step_by_step:
                    break
        finally:
            get_config().llm.llm_api_key = None
    st.session_state.agent = agent


def main() -> None:
    _init_session()

    with st.sidebar:
        st.title("📈 SEC Intelligence")
        st.caption("An agent that searches SEC EDGAR live and answers in plain language.")
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-…",
            help="Used only for your requests in this browser session. Never stored.",
        )
        auto = st.toggle("Run all steps automatically", value=True)
        st.button("Clear conversation", on_click=_reset, width="stretch")
        st.divider()
        st.caption("Source: [github.com/Abhishek2005-Siva/Agentic-AI](https://github.com/Abhishek2005-Siva/Agentic-AI)")

    st.header("Ask about any SEC filing")

    if not st.session_state.history:
        st.write("Try one of these:")
        for example in EXAMPLES:
            if st.button(example, key=example):
                st.session_state.pending = example
                st.rerun()

    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    agent = st.session_state.agent
    if agent:
        _render_trace(agent)

    question = st.chat_input("Ask a question, e.g. “Latest 8-K from Tesla?”")
    question = question or st.session_state.pending
    st.session_state.pending = None

    if question:
        if not api_key:
            st.warning("Paste your OpenAI API key in the sidebar first.")
            st.stop()
        from sec_intelligence.agent.planner import init_agent_state

        previous = list(st.session_state.history)
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.agent = init_agent_state(question, previous)
        st.session_state.run_mode = "all" if auto else "step"
        st.rerun()

    agent = st.session_state.agent
    if agent and not agent.get("done"):
        if st.session_state.run_mode:
            if not api_key:
                st.warning("Paste your OpenAI API key in the sidebar to continue.")
                st.stop()
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("🤔 Working…")
                _run_steps(api_key, placeholder)
            st.rerun()
        else:
            cols = st.columns(3)
            if cols[0].button("Continue ▶", type="primary", width="stretch"):
                st.session_state.run_mode = "step"
                st.rerun()
            if cols[1].button("Run all ⏩", width="stretch"):
                st.session_state.run_mode = "all"
                st.rerun()
            if cols[2].button("Stop ⏹", width="stretch"):
                agent["done"] = True
                _set_last_assistant("Stopped by user.")
                st.rerun()


main()
