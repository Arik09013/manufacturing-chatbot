"""
Streamlit chat frontend — professional dark industrial theme.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="ManufactAI — Anomaly Intelligence",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0d1117;
    color: #c9d1d9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ── Header brand ── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0 1.5rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1.5rem;
}
.brand-logo {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}
.brand-logo span {
    color: #00d2ff;
}
.brand-tag {
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8b949e;
    background: #161b22;
    border: 1px solid #30363d;
    padding: 2px 6px;
    border-radius: 4px;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #2ed573;
    box-shadow: 0 0 6px #2ed573;
    display: inline-block;
    margin-left: auto;
}

/* ── Sidebar section labels ── */
.sidebar-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8b949e;
    margin: 1.2rem 0 0.5rem 0;
}

/* ── Quick-query pills ── */
.query-pill {
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 0.72rem;
    color: #8b949e;
    margin: 3px 0;
    width: 100%;
    transition: border-color 0.15s;
}
.query-pill:hover { border-color: #00d2ff; color: #c9d1d9; }

/* ── Page title ── */
.page-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 2px;
}
.page-subtitle {
    font-size: 0.8rem;
    color: #8b949e;
    margin-bottom: 1.5rem;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
}

/* ── Diagnostic result card ── */
.diag-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 0.8rem 0;
    position: relative;
    overflow: hidden;
}
.diag-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.diag-card.anomaly::before  { background: linear-gradient(90deg, #ff4757, #ff6b81); }
.diag-card.normal::before   { background: linear-gradient(90deg, #2ed573, #7bed9f); }
.diag-card.warning::before  { background: linear-gradient(90deg, #ffa502, #ffcc5c); }

/* ── Stat grid inside cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin: 0.9rem 0;
}
.stat-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 7px;
    padding: 0.6rem 0.8rem;
}
.stat-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 3px;
}
.stat-value {
    font-size: 1rem;
    font-weight: 600;
    color: #c9d1d9;
    font-family: 'JetBrains Mono', monospace;
}
.stat-value.anomaly { color: #ff4757; }
.stat-value.normal  { color: #2ed573; }
.stat-value.warn    { color: #ffa502; }

/* ── Confidence bar ── */
.conf-wrap { margin: 0.9rem 0 0.4rem; }
.conf-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}
.conf-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #8b949e;
}
.conf-pct {
    font-size: 0.85rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.conf-track {
    height: 5px;
    background: #21262d;
    border-radius: 3px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}

/* ── Cause list ── */
.cause-list { margin: 0.6rem 0; }
.cause-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 0.55rem 0.7rem;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 7px;
    margin-bottom: 6px;
}
.cause-rank {
    font-size: 0.68rem;
    font-weight: 700;
    color: #8b949e;
    background: #21262d;
    border-radius: 4px;
    padding: 1px 6px;
    min-width: 22px;
    text-align: center;
    margin-top: 1px;
    flex-shrink: 0;
}
.cause-text { font-size: 0.82rem; color: #c9d1d9; line-height: 1.4; }
.cause-badge {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 1px 6px;
    border-radius: 3px;
    margin-left: auto;
    flex-shrink: 0;
    align-self: center;
}
.badge-high   { background: #ff47571a; color: #ff4757; border: 1px solid #ff475733; }
.badge-medium { background: #ffa5021a; color: #ffa502; border: 1px solid #ffa50233; }
.badge-low    { background: #8b949e1a; color: #8b949e; border: 1px solid #8b949e33; }

/* ── Action box ── */
.action-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 7px;
    padding: 0.7rem 0.9rem;
    margin-top: 0.5rem;
}
.action-urgency {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.urgency-critical { color: #ff4757; }
.urgency-high     { color: #ffa502; }
.urgency-medium   { color: #ffcc5c; }
.action-primary { font-size: 0.83rem; color: #c9d1d9; margin-bottom: 4px; }
.action-secondary { font-size: 0.78rem; color: #8b949e; }

/* ── Section titles inside cards ── */
.section-title {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8b949e;
    margin: 1rem 0 0.5rem;
    padding-top: 0.8rem;
    border-top: 1px solid #21262d;
}

/* ── Streamlit overrides ── */
.stSelectbox label, .stTextInput label {
    color: #8b949e !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 7px !important;
}
[data-testid="stChatInputTextArea"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 10px !important;
}
.stButton > button {
    background: #161b22;
    border: 1px solid #30363d;
    color: #c9d1d9;
    border-radius: 7px;
    font-size: 0.78rem;
}
.stButton > button:hover {
    border-color: #00d2ff;
    color: #00d2ff;
}
.stExpander {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 7px !important;
}
[data-testid="stExpanderToggleIcon"] { color: #8b949e !important; }
hr { border-color: #21262d !important; }

/* ── Out-of-scope info box ── */
.stAlert {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _conf_color(band: str) -> str:
    return {"high": "#2ed573", "medium": "#ffa502", "low": "#ff4757"}.get(band, "#8b949e")

def _urgency_label(urgency: str) -> str:
    icons = {"critical": "● CRITICAL", "high": "● HIGH", "medium": "● MEDIUM"}
    return icons.get(urgency, "● MEDIUM")

def _shap_chart(drivers: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [d["feature"].replace("_", " ") for d in drivers]
    values = [d["shap"] for d in drivers]
    colors = ["#ff4757" if v > 0 else "#00d2ff" for v in values]

    fig, ax = plt.subplots(figsize=(6, max(2.2, len(labels) * 0.52)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   height=0.55, edgecolor="none")

    ax.axvline(0, color="#30363d", linewidth=0.8, zorder=0)
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.set_xlabel("SHAP value", color="#8b949e", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    ax.xaxis.label.set_color("#8b949e")
    ax.grid(axis="x", color="#21262d", linewidth=0.5, alpha=0.6)

    # Value labels
    for bar, v in zip(bars, values[::-1]):
        ax.text(
            v + (0.003 if v >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            color="#8b949e",
            fontsize=7.5,
        )

    plt.tight_layout(pad=0.4)
    return fig


def _render_details(payload: dict) -> None:
    is_anomaly = payload.get("is_anomaly", False)
    conf       = payload.get("confidence", {})
    band       = conf.get("band", "unknown")
    score      = conf.get("score", 0.0)
    atype      = payload.get("anomaly_type") or "—"
    machine    = payload.get("machine_id", "—")
    prob       = payload.get("anomaly_prob", 0.0)
    window     = payload.get("window_start", "")[:16].replace("T", " ")

    card_class = "anomaly" if is_anomaly else "normal"
    status_text = f"ANOMALY · {atype.upper().replace('_', ' ')}" if is_anomaly else "NORMAL OPERATION"
    status_color = "#ff4757" if is_anomaly else "#2ed573"
    prob_class = "anomaly" if prob > 0.6 else ("warn" if prob > 0.3 else "normal")

    # ── Card header ──
    st.markdown(f"""
    <div class="diag-card {card_class}">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">
        <span style="font-size:0.7rem;font-weight:700;letter-spacing:2px;
                     text-transform:uppercase;color:{status_color};">{status_text}</span>
        <span style="font-size:0.7rem;color:#8b949e;font-family:'JetBrains Mono',monospace;">{window}</span>
      </div>

      <div class="stat-grid">
        <div class="stat-box">
          <div class="stat-label">Machine</div>
          <div class="stat-value">{machine}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Anomaly Prob</div>
          <div class="stat-value {prob_class}">{prob:.0%}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Anomaly Type</div>
          <div class="stat-value" style="font-size:0.82rem;">{atype.replace("_"," ").title()}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Confidence</div>
          <div class="stat-value" style="color:{_conf_color(band)};">{band.upper()}</div>
        </div>
      </div>

      <div class="conf-wrap">
        <div class="conf-label-row">
          <span class="conf-label">Confidence score</span>
          <span class="conf-pct" style="color:{_conf_color(band)};">{score:.0%}</span>
        </div>
        <div class="conf-track">
          <div class="conf-fill" style="width:{score*100:.1f}%;
               background:linear-gradient(90deg,{_conf_color(band)}88,{_conf_color(band)});"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Causes & recommendation ──
    if is_anomaly:
        causes = payload.get("causes", [])
        rec    = payload.get("recommendation", {})
        urgency = rec.get("urgency", "medium")

        with st.expander("Root Causes & Recommended Action", expanded=True):
            badge_map = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}
            cause_html = '<div class="cause-list">'
            for c in causes:
                bc = badge_map.get(c["evidence_strength"], "badge-low")
                cause_html += f"""
                <div class="cause-item">
                  <span class="cause-rank">#{c['rank']}</span>
                  <span class="cause-text">{c['cause']}</span>
                  <span class="cause-badge {bc}">{c['evidence_strength'].upper()}</span>
                </div>"""
            cause_html += "</div>"

            uc = f"urgency-{urgency}"
            ul = _urgency_label(urgency)
            action_html = f"""
            <div class="action-box">
              <div class="action-urgency {uc}">{ul}</div>
              <div class="action-primary">⚡ {rec.get('primary','')}</div>
              <div class="action-secondary">↳ {rec.get('secondary','')}</div>
            </div>"""

            st.markdown(cause_html + action_html, unsafe_allow_html=True)

    # ── SHAP chart ──
    drivers = payload.get("shap_drivers", [])
    with st.expander("SHAP Feature Contributions", expanded=is_anomaly):
        if drivers:
            col_chart, col_table = st.columns([3, 2])
            with col_chart:
                fig = _shap_chart(drivers)
                st.pyplot(fig, use_container_width=True)
            with col_table:
                import pandas as pd
                df = pd.DataFrame(drivers)[["feature", "shap", "direction"]].copy()
                df.columns = ["Feature", "SHAP", "Direction"]
                df["SHAP"] = df["SHAP"].map(lambda x: f"{x:+.4f}")
                df["Direction"] = df["Direction"].str.replace("_", " ")
                st.dataframe(df, hide_index=True, use_container_width=True,
                             height=min(220, 38 + len(df) * 35))
        else:
            st.caption("No SHAP data available.")

    # ── Raw payload ──
    with st.expander("Raw Pipeline Payload"):
        st.json({k: v for k, v in payload.items() if k != "raw_shap"})


def _render_param_advice(rec: dict) -> None:
    """Render a parameter recommendation card."""
    if "error" in rec:
        st.warning(rec["error"])
        return

    mat   = rec.get("material_display", rec.get("material", ""))
    thick = rec.get("thickness_mm", "?")
    band  = rec.get("band", "")
    eff   = rec.get("efficiency_score", 0)
    dep   = rec.get("deposition_rate_g_per_min", 0)
    p     = rec.get("params", {})
    note  = rec.get("notes", "")

    eff_color = "#2ed573" if eff >= 8 else ("#ffa502" if eff >= 6 else "#ff4757")
    eff_bar   = int(eff / 10 * 100)

    st.markdown(f"""
    <div class="diag-card normal">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.7rem;">
        <span style="font-size:0.7rem;font-weight:700;letter-spacing:2px;
                     text-transform:uppercase;color:#00d2ff;">PARAMETER RECOMMENDATION</span>
        <span style="font-size:0.7rem;color:#8b949e;">{mat} · {thick} mm · {band}</span>
      </div>

      <div class="stat-grid">
        <div class="stat-box">
          <div class="stat-label">Current (A)</div>
          <div class="stat-value" style="color:#00d2ff;">
            {p.get('welding_current',['?','?'])[0]}–{p.get('welding_current',['?','?'])[1]}
          </div>
          <div style="font-size:0.62rem;color:#8b949e;">optimal: {rec.get('optimal_current','?')} A</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Voltage (V)</div>
          <div class="stat-value" style="color:#00d2ff;">
            {p.get('arc_voltage',['?','?'])[0]}–{p.get('arc_voltage',['?','?'])[1]}
          </div>
          <div style="font-size:0.62rem;color:#8b949e;">optimal: {rec.get('optimal_voltage','?')} V</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Speed (mm/min)</div>
          <div class="stat-value" style="color:#00d2ff;">
            {p.get('welding_speed',['?','?'])[0]}–{p.get('welding_speed',['?','?'])[1]}
          </div>
          <div style="font-size:0.62rem;color:#8b949e;">optimal: {rec.get('optimal_speed','?')}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Wire Feed (m/min)</div>
          <div class="stat-value" style="color:#00d2ff;">
            {p.get('wire_feed_rate',['?','?'])[0]}–{p.get('wire_feed_rate',['?','?'])[1]}
          </div>
          <div style="font-size:0.62rem;color:#8b949e;">optimal: {rec.get('optimal_wfr','?')}</div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin:0.7rem 0;">
        <div class="stat-box">
          <div class="stat-label">Gas Flow (L/min)</div>
          <div class="stat-value" style="color:#00d2ff;">
            {p.get('shielding_gas_flow',['?','?'])[0]}–{p.get('shielding_gas_flow',['?','?'])[1]}
          </div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Gas Mix</div>
          <div class="stat-value" style="font-size:0.75rem;color:#c9d1d9;">{p.get('gas_mix','—')}</div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-bottom:0.9rem;">
        <div class="stat-box">
          <div class="stat-label">Wire Diameter</div>
          <div class="stat-value" style="color:#00d2ff;">{p.get('wire_diameter','1.2')} mm</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Deposition Rate</div>
          <div class="stat-value" style="color:#2ed573;">~{dep} g/min</div>
        </div>
      </div>

      <div class="conf-wrap">
        <div class="conf-label-row">
          <span class="conf-label">Efficiency Score</span>
          <span class="conf-pct" style="color:{eff_color};">{eff}/10</span>
        </div>
        <div class="conf-track">
          <div class="conf-fill"
               style="width:{eff_bar}%;background:linear-gradient(90deg,{eff_color}88,{eff_color});"></div>
        </div>
      </div>

      <div style="margin-top:0.8rem;padding:0.6rem 0.8rem;background:#0d1117;
                  border:1px solid #21262d;border-radius:7px;
                  font-size:0.78rem;color:#8b949e;line-height:1.5;">
        <span style="color:#00d2ff;font-weight:600;">Note: </span>{note}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Heat input range
    hi = p.get("heat_input_range", [0, 0])
    st.caption(f"Heat input range: {hi[0]}–{hi[1]} kJ/mm")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="brand-header">
      <div>
        <div class="brand-logo">Weld<span>.AI</span></div>
        <div style="font-size:0.62rem;color:#8b949e;margin-top:2px;
                    letter-spacing:1px;">WELDING INTELLIGENCE SYSTEM</div>
      </div>
      <div>
        <span class="brand-tag">MVP</span><br>
        <span class="status-dot" title="System online"></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Welding Station</div>', unsafe_allow_html=True)
    selected_machine = st.selectbox(
        "Welding Station",
        ["auto-detect", "station_1", "station_2", "station_3"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Query Time</div>', unsafe_allow_html=True)
    query_time_str = st.text_input(
        "Query Time",
        placeholder="2026-05-01T14:00:00  (blank = auto)",
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Example Queries</div>', unsafe_allow_html=True)
    example_queries = [
        "Why did station_1 stop at 14:00?",
        "Arc instability on station_2 — cause?",
        "Best settings for 5mm mild steel",
        "Optimal speed for 3mm aluminum?",
        "Is station_3 running normally?",
        "What current for 8mm stainless?",
    ]
    for q in example_queries:
        short = q[:46] + ("…" if len(q) > 46 else "")
        st.markdown(f'<div class="query-pill">{short}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border-top:1px solid #21262d;padding-top:1rem;">
      <div style="font-size:0.65rem;color:#8b949e;line-height:1.8;">
        <b style="color:#30363d;">Process</b><br>
        MIG / MAG / GMAW<br>
        <b style="color:#30363d;">Pipeline</b><br>
        RandomForest &middot; SHAP &middot; Claude<br>
        <b style="color:#30363d;">Materials</b><br>
        Mild Steel &middot; SS &middot; Aluminum<br>
        <b style="color:#30363d;">Eval</b><br>
        F1 = 0.99 &middot; AUC = 1.00
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-title">Welding Intelligence Console</div>
<div class="page-subtitle">
  Ask about welding anomalies, fault root-causes, or get optimal parameter recommendations.
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("payload"):
            p = msg["payload"]
            if p.get("pipeline_type") == "parameter_advice":
                _render_param_advice(p)
            else:
                _render_details(p)

# ── Input handler ─────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about a weld fault or request parameter recommendations…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running diagnostic pipeline…"):
            try:
                from src.api.pipeline import run_pipeline, run_param_pipeline
                from src.chat.synthesize import synthesize, _fallback_text
                from src.chat.intent import OutOfScopeError
                from src.reasoning.param_advisor import parse_param_query

                machine_id = None if selected_machine == "auto-detect" else selected_machine
                query_time = None
                if query_time_str.strip():
                    from datetime import datetime
                    query_time = datetime.fromisoformat(query_time_str.strip())

                # Route: parameter query or anomaly query?
                if parse_param_query(prompt) is not None:
                    payload = run_param_pipeline(prompt)
                    rec = payload
                    mat   = rec.get("material_display", rec.get("material", "material"))
                    thick = rec.get("thickness_mm", "?")
                    eff   = rec.get("efficiency_score", "?")
                    answer = (
                        f"Here are the recommended MIG/MAG parameters for "
                        f"**{mat} ({thick} mm)**. "
                        f"Efficiency score: **{eff}/10**. "
                        f"Optimal speed: **{rec.get('optimal_speed','?')} mm/min**, "
                        f"current: **{rec.get('optimal_current','?')} A**, "
                        f"voltage: **{rec.get('optimal_voltage','?')} V**."
                    )
                    st.markdown(answer)
                    _render_param_advice(payload)
                else:
                    payload = run_pipeline(
                        question=prompt,
                        machine_id=machine_id,
                        query_time=query_time,
                    )
                    try:
                        answer = synthesize(payload)
                    except Exception:
                        answer = _fallback_text(payload)
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
