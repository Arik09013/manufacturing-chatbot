"""
Streamlit chat frontend for the manufacturing chatbot MVP.

Features:
  - Chat message box
  - Plain-language LLM answer display
  - Expandable SHAP explanation panel
  - Confidence indicator
  - Structured payload viewer
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Manufacturing Anomaly Chatbot",
    page_icon="🏭",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Settings")
    selected_machine = st.selectbox(
        "Default machine",
        ["auto-detect", "machine_1", "machine_2", "machine_3"],
    )
    query_time_str = st.text_input(
        "Query time (ISO, e.g. 2026-05-01T15:30:00)",
        placeholder="leave blank for auto",
    )
    st.markdown("---")
    st.markdown("**Example queries:**")
    st.markdown("- Why did line 1 slow down at 15:30?")
    st.markdown("- What caused the alarm on machine_2 at 10:00?")
    st.markdown("- Is machine_3 running normally?")
    st.markdown("---")
    st.caption("MVP — Anomaly detection + SHAP + Claude synthesis")

# ── Main chat area ────────────────────────────────────────────────────────────

st.title("Manufacturing Anomaly Chatbot")
st.caption("Ask a natural-language question about any machine or production event.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("payload"):
            _render_details(msg["payload"])


def _render_details(payload: dict) -> None:
    is_anomaly = payload.get("is_anomaly", False)
    conf = payload.get("confidence", {})

    # Confidence badge
    band = conf.get("band", "unknown")
    score = conf.get("score", 0.0)
    badge_color = {"high": "green", "medium": "orange", "low": "red"}.get(band, "gray")
    st.markdown(
        f"**Confidence:** :{badge_color}[{band.upper()} {score:.0%}]"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Anomaly probability", f"{payload.get('anomaly_prob', 0):.0%}")
        st.metric("Anomaly type", payload.get("anomaly_type") or "none")
    with col2:
        st.metric("Machine", payload.get("machine_id", "—"))
        st.metric(
            "Window",
            payload.get("window_start", "")[:16].replace("T", " "),
        )

    if is_anomaly:
        with st.expander("Root causes & recommendation", expanded=True):
            causes = payload.get("causes", [])
            for c in causes:
                st.markdown(
                    f"**{c['rank']}.** {c['cause']} — *evidence: {c['evidence_strength']}*"
                )
            rec = payload.get("recommendation", {})
            if rec:
                st.markdown("---")
                urgency = rec.get("urgency", "medium")
                urgency_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(
                    urgency, "🔵"
                )
                st.markdown(f"**{urgency_icon} Primary action:** {rec.get('primary', '')}")
                st.markdown(f"**Secondary action:** {rec.get('secondary', '')}")

    with st.expander("SHAP explanation (sensor signals)", expanded=False):
        st.text(payload.get("shap_text", "No SHAP data available."))
        drivers = payload.get("shap_drivers", [])
        if drivers:
            import pandas as pd
            df = pd.DataFrame(drivers)[["feature", "shap", "direction", "magnitude"]]
            df["shap"] = df["shap"].round(4)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Raw pipeline payload", expanded=False):
        display_payload = {
            k: v for k, v in payload.items()
            if k not in ("raw_shap",)
        }
        st.json(display_payload)


# Dummy reference to render_details for message history (before it's defined)
# Redefine after function to avoid forward-reference issues in the history loop
for i, msg in enumerate(st.session_state.messages):
    pass  # already rendered above, no-op


# ── Input handler ─────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about a machine anomaly…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running diagnostic pipeline…"):
            try:
                from src.api.pipeline import run_pipeline
                from src.chat.synthesize import synthesize
                from src.chat.intent import OutOfScopeError

                machine_id = None if selected_machine == "auto-detect" else selected_machine
                query_time = None
                if query_time_str.strip():
                    from datetime import datetime
                    query_time = datetime.fromisoformat(query_time_str.strip())

                payload = run_pipeline(
                    question=prompt,
                    machine_id=machine_id,
                    query_time=query_time,
                )
                answer = synthesize(payload)
                st.markdown(answer)
                _render_details(payload)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "payload": payload,
                })

            except OutOfScopeError as exc:
                msg = str(exc)
                st.info(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

            except Exception as exc:
                err = f"Pipeline error: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
