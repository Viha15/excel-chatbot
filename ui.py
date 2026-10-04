"""Streamlit front end. It is a thin client of the FastAPI service: all parsing,
planning and calculation happen in the API, so the UI and the API give identical answers.

Run the API first (uv run uvicorn src.main:app), then: uv run streamlit run ui.py
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
# No read timeout on /chat: the API already bounds each LLM call (LLM_TIMEOUT_SECONDS) and
# returns a clear error, so the UI must not give up first.
CHAT_TIMEOUT = httpx.Timeout(connect=5, read=None, write=60, pool=5)

st.set_page_config(page_title="Excel Chatbot", page_icon="📊", layout="wide")
state = st.session_state
state.setdefault("workbook", None)  # upload response from the API
state.setdefault("upload_key", None)  # which file that response belongs to
state.setdefault("messages", [])


def api_health() -> dict | None:
    """Call GET /health and return its JSON, or None if the API cannot be reached.

    Used by the sidebar on every rerun to show whether the API is connected.
    """
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).json()
    except httpx.HTTPError:
        return None


def upload(file) -> None:
    """Send the chosen file to POST /upload.

    On success, stores the API's response (workbook id and sheets) in the session state and
    starts a new conversation; on failure, shows the API's error message in the sidebar.
    Called when the user picks a file that has not been uploaded yet.
    """
    try:
        r = httpx.post(f"{API_URL}/upload", files={"file": (file.name, file.getvalue())}, timeout=120)
    except httpx.HTTPError as exc:
        st.sidebar.error(f"Could not reach the API: {exc}")
        return
    if r.status_code != 200:
        st.sidebar.error(r.json().get("detail", r.text))
        return
    state.workbook, state.upload_key, state.messages = r.json(), (file.name, file.size), []


def ask(question: str) -> dict:
    """Send a question about the uploaded workbook to POST /chat and return the reply.

    Never raises: any failure is returned as {"error": "..."}, so the reply can always be
    stored in the chat history and rendered by show_reply().
    """
    try:
        r = httpx.post(
            f"{API_URL}/chat",
            timeout=CHAT_TIMEOUT,
            json={"workbook_id": state.workbook["workbook_id"], "question": question},
        )
    except httpx.ConnectError:
        return {"error": f"Could not connect to the API at {API_URL}. Is uvicorn still running?"}
    except httpx.HTTPError as exc:
        return {"error": f"Request to the API failed: {exc!r}"}
    if r.status_code != 200:
        return {"error": r.json().get("detail", r.text)}
    return r.json()


def show_reply(reply: dict) -> None:
    """Render one assistant message from a /chat reply (or an error dict from ask()).

    Shows the answer, any guardrail warnings, the result table, the sources, and a
    collapsible "How this was computed" section with the steps, the plan, rejected
    attempts and token usage. Used for new replies and when replaying the history.
    """
    if "error" in reply:
        st.error(reply["error"])
        return
    st.markdown(reply["answer"])
    for note in reply.get("guardrails", []):
        st.warning(note)
    result = reply.get("result")
    if result and result["rows"]:
        df = pd.DataFrame(result["rows"], columns=result["columns"])
        st.dataframe(df.style.format(precision=2, thousands=","), hide_index=True, width="stretch")
        if result["total_rows"] > len(df):
            st.caption(f"Showing {len(df)} of {result['total_rows']} rows.")
    if reply.get("sources"):
        st.caption("Sources: " + "; ".join(_source_text(s) for s in reply["sources"]))
    with st.expander("How this was computed"):
        if result:
            st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(result["steps"], 1)))
        st.json(reply.get("plan") or {}, expanded=False)
        for i, err in enumerate(reply.get("failed_attempts", []), 1):
            st.caption(f"Attempt {i} rejected, model retried: {err}")
        usage = reply.get("usage", {})
        st.caption(
            f"LLM tokens: {usage.get('prompt_tokens', 0)} prompt, "
            f"{usage.get('completion_tokens', 0)} completion"
        )


def _source_text(s: dict) -> str:
    """Format one source for display, e.g. "Sales_Data (23 rows; Excel rows 2, 5, 9…)".

    Lists at most the first 10 Excel row numbers. Helper for show_reply().
    """
    text = f"{s['sheet']} ({s['rows_matched']} rows"
    if s["excel_rows"]:
        rows = ", ".join(map(str, s["excel_rows"][:10]))
        more = "…" if s["rows_matched"] > 10 else ""
        text += f"; Excel rows {rows}{more}"
    return text + ")"


# --- sidebar: API status and workbook ---------------------------------------------------
with st.sidebar:
    st.header("Workbook")
    health = api_health()
    if health:
        st.success(f"API connected · LLM: {health['llm_provider']}")
    else:
        st.error(f"API not reachable at {API_URL}. Start it with `uv run uvicorn src.main:app`.")

    file = st.file_uploader("Upload an .xlsx file", type=["xlsx"], disabled=health is None)
    if file and state.upload_key != (file.name, file.size):
        with st.spinner("Reading workbook…"):
            upload(file)

    if state.workbook:
        st.caption(state.workbook["filename"])
        for sheet in state.workbook["sheets"]:
            with st.expander(f"{sheet['name']} · {sheet['rows']} rows"):
                st.dataframe(pd.DataFrame(sheet["columns"]), hide_index=True, width="stretch")
        if st.button("Clear conversation"):
            state.messages = []
            st.rerun()

# --- main: chat --------------------------------------------------------------------------
st.title("📊 Excel Chatbot")

if not state.workbook:
    st.info("Upload a workbook in the sidebar, then ask questions about it in plain language.")
    st.stop()

for message in state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            show_reply(message["content"])

question = st.chat_input("Ask a question about the workbook")
if question:
    state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking… (a local model can take a minute or two)"):
            reply = ask(question)
        show_reply(reply)
    state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
