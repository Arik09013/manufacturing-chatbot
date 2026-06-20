"""
End-to-end inference pipeline.

Given a user question (with optional machine_id and timestamp filter),
runs: load -> preprocess -> fuse -> detect -> explain -> reason -> confidence
and returns a structured payload ready for LLM synthesis.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

MACHINES = ["station_1", "station_2", "station_3"]
_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "anomaly.joblib"
_WINDOW_MINUTES = 30


@lru_cache(maxsize=1)
def _load_bundle():
    """Load the trained detector+scaler bundle once, cache it."""
    bundle = joblib.load(_MODEL_PATH)
    return bundle["detector"], bundle["scaler"]


def _parse_intent(question: str) -> dict:
    """
    Lightweight NL intent parser: extract machine_id and approximate timestamp.
    Returns dict with keys: machine_id (or None), query_time (or None).
    """
    question_lower = question.lower()

    # Machine / station detection
    machine_id = None
    for m in MACHINES:
        if m.replace("_", " ") in question_lower or m in question_lower:
            machine_id = m
            break
    # Shorthand "station 1/2/3" or "line 1/2/3"
    if machine_id is None:
        match = re.search(r"(?:station|line)\s*(\d)", question_lower)
        if match:
            machine_id = f"station_{match.group(1)}"

    # Time extraction — simple patterns: "at HH:MM", "around HH:MM"
    query_time = None
    time_match = re.search(r"(?:at|around|near|after)\s+(\d{1,2}):(\d{2})", question_lower)
    if time_match:
        h, m = int(time_match.group(1)), int(time_match.group(2))
        # Use 2026-05-01 as the default date (matching synthetic data)
        query_time = datetime(2026, 5, 1, h, m, 0)

    return {"machine_id": machine_id, "query_time": query_time}


def run_pipeline(
    question: str,
    machine_id: Optional[str] = None,
    query_time: Optional[datetime] = None,
) -> dict:
    """
    Run the full inference pipeline for a user question.

    Parameters
    ----------
    question   : raw operator question
    machine_id : override machine filter (or None to auto-detect from question)
    query_time : override time window (or None to use most recent anomaly window)

    Returns
    -------
    Structured payload dict for LLM synthesis.
    """
    from src.data.loaders import load_sensors, load_logs, load_ground_truth
    from src.preprocess.sensor import preprocess_sensors
    from src.preprocess.logs import preprocess_logs
    from src.fusion.fuse import fuse
    from src.model.anomaly import get_feature_matrix
    from src.explain.shap_explainer import AnomalyExplainer
    from src.reasoning.root_cause import identify_causes
    from src.reasoning.recommend import get_recommendation
    from src.reasoning.confidence import compute_confidence
    from src.chat.intent import classify_intent, OutOfScopeError

    # --- 0. Intent guard ---
    intent_result = classify_intent(question)
    if not intent_result.in_scope:
        raise OutOfScopeError(question, intent_result)

    # --- 1. Parse intent ---
    intent = _parse_intent(question)
    if machine_id is None:
        machine_id = intent["machine_id"] or MACHINES[0]
    if query_time is None:
        query_time = intent["query_time"]

    logger.info("Pipeline: machine=%s, time=%s", machine_id, query_time)

    # --- 2. Load data ---
    sensors = load_sensors(machine_id=machine_id)
    logs = load_logs(machine_id=machine_id)
    gt = load_ground_truth()

    # --- 3. Preprocess ---
    detector, scaler = _load_bundle()
    windowed, _ = preprocess_sensors(sensors, scaler=scaler, fit_scaler_on_data=False)
    log_features = preprocess_logs(logs)

    # --- 4. Fuse (no embeddings for inference speed) ---
    fused = fuse(windowed, log_features, None, gt, include_embeddings=False)

    # --- 5. Select the target window ---
    if query_time is not None:
        target = _window_at(fused, query_time)
    else:
        # Default: most recent anomaly detected, or last window
        preds = detector.predict_proba(fused)
        max_idx = int(preds.argmax())
        target = fused.iloc[[max_idx]]

    target = target.reset_index(drop=True)

    # --- 6. Detect ---
    anomaly_prob = float(detector.predict_proba(target)[0])
    is_anomaly = bool(detector.predict(target)[0] == 1)

    anomaly_type = ""
    if is_anomaly:
        anomaly_type = str(target["anomaly_type"].iloc[0]) if "anomaly_type" in target.columns else "unknown"
        if not anomaly_type or anomaly_type == "nan":
            anomaly_type = "unknown"

    # --- 7. SHAP explain ---
    explainer = AnomalyExplainer(detector, top_n=5)
    shap_result = explainer.explain_row(target)

    # --- 7b. LIME explain (independent local surrogate; best-effort) ---
    lime_drivers = []
    try:
        from src.explain.lime_explainer import get_default_lime_explainer
        lime_result = get_default_lime_explainer(top_n=5).explain_row(target)
        lime_drivers = lime_result["lime_drivers"]
    except Exception:
        logger.warning("LIME explanation unavailable", exc_info=True)

    # --- 8. Root cause ---
    log_codes = str(target["unique_event_codes"].iloc[0]) if "unique_event_codes" in target.columns else ""
    causes = identify_causes(anomaly_type, shap_result["shap_drivers"], log_codes)

    # --- 9. Recommendation ---
    recommendation = get_recommendation(anomaly_type) if is_anomaly else {}

    # --- 10. Confidence ---
    confidence = compute_confidence(anomaly_prob, shap_result["shap_drivers"])

    window_start = target["window_start"].iloc[0]
    window_end = target["window_end"].iloc[0]

    # --- 7c. Operator-note attention visualization (best-effort) ---
    # Use the note inside the window if present, else the nearest note for this
    # machine within 90 min of the window midpoint (notes are logged sparsely and
    # rarely align exactly to a fused-window boundary).
    note_text, note_attention = "", None
    try:
        from src.data.loaders import load_notes
        notes = load_notes(machine_id=machine_id)
        if len(notes):
            in_win = notes[(notes["timestamp"] >= window_start) & (notes["timestamp"] <= window_end)]
            if len(in_win):
                note_text = str(in_win.iloc[0]["note_text"])
            else:
                midpoint = window_start + (window_end - window_start) / 2
                gap = (notes["timestamp"] - midpoint).abs()
                nearest = notes.iloc[[gap.argmin()]]
                if gap.min() <= timedelta(minutes=90):
                    note_text = str(nearest.iloc[0]["note_text"])
        if note_text:
            from src.explain.attention_explainer import explain_attention
            note_attention = explain_attention(note_text)
    except Exception:
        logger.warning("Attention visualization unavailable", exc_info=True)

    return {
        "question":       question,
        "machine_id":     machine_id,
        "window_start":   str(window_start),
        "window_end":     str(window_end),
        "is_anomaly":     is_anomaly,
        "anomaly_type":   anomaly_type,
        "anomaly_prob":   round(anomaly_prob, 4),
        "confidence":     confidence,
        "causes":         causes,
        "recommendation": recommendation,
        "shap_drivers":   shap_result["shap_drivers"],
        "shap_text":      explainer.explain_text(shap_result),
        "lime_drivers":   lime_drivers,
        "note_text":      note_text,
        "note_attention": note_attention,
    }


_STATION_RE = re.compile(r"\b(?:station|line|machine)[_\s]?\d\b|\b[sm][123]\b")

# Manufacturing domains OUTSIDE welding. A question that names one of these is
# answered by the general-manufacturing LLM advisor — it must NOT bleed into the
# welding parameter optimiser or the welding knowledge base (which would give a
# welding-flavoured answer to, e.g., an injection-moulding question).
_OTHER_MANUFACTURING_TERMS = (
    "machining", "cnc", "milling", " mill ", "lathe", "turning", "drilling",
    "boring", "grinding", "spindle", "feed rate", "cutting speed", "end mill",
    "tool wear", "chatter", "swarf",
    "injection molding", "injection moulding", "molding", "moulding",
    "mold", "mould", "sink mark", "short shot", "flash ", "thermoplastic",
    "abs", "polypropylene", "polymer", "resin", "plastic",
    "casting", "die cast", "sand cast", "investment cast",
    "forging", "extrusion", "extrude", "rolling", "sheet metal", "stamping",
    "press brake", "springback", "blanking", "deep drawing", "tonnage",
    "additive", "3d print", "fdm", "fff", "sla", "sls", "dmls", "infill",
    "heat treatment", "annealing", "quenching", "tempering", "carburizing",
    "anodizing", "deburring", "honing", "gd&t", "metrology", "cmm",
    "six sigma", "lean", "kaizen", "5s", "kanban", "poka yoke",
    "oee", "overall equipment effectiveness", "takt", "cycle time",
    "assembly line", "bill of materials", "g-code", "gcode",
    "titanium", "brass", "composite", "ceramic",
)

# Strong welding markers — used only to decide whether an unmatched question that
# slipped past the welding knowledge base should still get a welding persona or
# fall through to the general-manufacturing advisor.
_WELDING_MARKERS = (
    "weld", "arc", "mig", "mag", "tig", "smaw", "gmaw", "gtaw", "fcaw", "mma",
    "bead", "filler", "electrode", "shielding gas", "wire feed", "spatter",
    "porosity", "undercut", "penetration", "heat input", "wps", "stick weld",
    "mild steel", "stainless",
)


def _is_other_manufacturing_domain(question: str) -> bool:
    """True if the question is clearly a non-welding manufacturing domain."""
    q = f" {question.lower()} "
    return any(term in q for term in _OTHER_MANUFACTURING_TERMS)


def _looks_welding(question: str) -> bool:
    """True if the question carries a clear welding marker."""
    q = question.lower()
    return any(term in q for term in _WELDING_MARKERS)


def route_question(question: str) -> str:
    """
    Decide which pipeline a question belongs to.

    Returns one of:
      "param"     — material+thickness parameter optimisation (grid-search advisor)
      "anomaly"   — diagnose a specific station/time event (ML + SHAP pipeline)
      "knowledge" — general welding advice (defects, troubleshooting, cost, etc.)
      "general"   — any other manufacturing question (machining, molding, casting,
                    forming, additive, quality, ops) answered by the LLM advisor
    """
    from src.reasoning.param_advisor import parse_param_query

    # Non-welding manufacturing domains are routed to the general advisor FIRST,
    # so a CNC/molding/stamping query never reaches the welding-specific tools.
    if _is_other_manufacturing_domain(question):
        return "general"
    if parse_param_query(question) is not None:
        return "param"
    if _STATION_RE.search(question.lower()):
        return "anomaly"
    return "knowledge"


def run_general_pipeline(question: str) -> dict:
    """
    Route a general (non-welding) manufacturing question to the LLM advisor.

    Runs the intent guard first, then returns a payload with
    pipeline_type='general_manufacturing'. There is no deterministic grounding
    here — the LLM answers from general manufacturing-engineering knowledge, and
    the prompt instructs it to flag specific numbers as starting points to verify.
    """
    from src.chat.intent import classify_intent, OutOfScopeError

    intent_result = classify_intent(question)
    if not intent_result.in_scope:
        raise OutOfScopeError(question, intent_result)

    return {
        "question":         question,
        "pipeline_type":    "general_manufacturing",
        "knowledge_domain": "general",
    }


# Integration-intent keywords — used to keep the robotics-engineer persona when a
# question is clearly about the Isaac Sim / sim build but matches no curated entry.
_INTEGRATION_INTENT_TERMS = (
    "isaac sim", "isaac lab", "isaac", "omniverse", "nvidia",
    "simulation", "simulate", "digital twin", "manipulator", "urdf", "usd",
    "rmpflow", "omnigraph", "depth camera", "rgbd", "rgb-d", "lidar",
    "point cloud", "force/torque", "force torque", "f/t sensor", "6-axis",
    "teleop", "teleoperation", "vision pro", "cloudxr", "headset",
    "hand tracking", "retarget", "ros2", "ros 2", "daq", "data acquisition",
    "ethercat", "realsense", "ouster", "seam tracking", "weld cell", "welding cell",
    "kinematics", "inverse kinematics", "ik solver", "motion planning",
    "trajectory planning", "moveit", "cumotion", "isaac manipulator",
    "collision avoidance", "force control", "standoff", "seam finding",
    "weld pool", "fault injection", "sim to real", "sim-to-real", "sim2real",
    "reality gap", "domain randomization", "omni.ui", "viewport", "spatial ui",
    "novel contribution", "research contribution", "why simulation", "simulator",
)


def _is_integration_question(question: str) -> bool:
    """True if the question is about the Isaac Sim / robotics-integration build."""
    q = question.lower()
    return any(term in q for term in _INTEGRATION_INTENT_TERMS)


def run_knowledge_pipeline(question: str) -> dict:
    """
    Route a general welding-knowledge question (defect / troubleshooting / quality /
    productivity / cost / comparison) to the RAG-grounded knowledge layer.

    Runs the intent guard first (so out-of-scope questions are rejected here too),
    then performs hybrid RAG retrieval (semantic + lexical) over the welding
    knowledge corpus and returns a payload with pipeline_type='knowledge_advice'.
    Keyword topic matching is retained as a routing/domain signal and as a
    grounding fallback when the embedding model is unavailable.
    """
    from src.chat.intent import classify_intent, OutOfScopeError
    from src.reasoning.knowledge import match_knowledge, build_knowledge_summary

    intent_result = classify_intent(question)
    if not intent_result.in_scope:
        raise OutOfScopeError(question, intent_result)

    topics = match_knowledge(question)

    # --- RAG retrieval (hybrid semantic + lexical over KB entries + reference
    # documents). Best-effort: if the local embedding model can't load (e.g. a
    # first run with no network), fall back to keyword grounding so the route
    # still answers. ---
    rag_passages: list[dict] = []
    try:
        from src.rag.retriever import retrieve
        rag_passages = retrieve(question, top_k=4)
    except Exception:
        logger.warning("RAG retrieval unavailable; falling back to keyword grounding",
                       exc_info=True)

    # A confident semantic hit counts as grounding even when keyword matching
    # found nothing.
    strong_rag = bool(rag_passages) and rag_passages[0]["score"] >= 0.35

    # No grounding at all (no curated entry, no strong retrieval), not an
    # integration question, and no welding marker → general manufacturing
    # question that landed here by the catch-all. Hand it to the general LLM
    # advisor instead of answering it in a welding-engineer persona.
    if (not topics
            and not strong_rag
            and not _is_integration_question(question)
            and not _looks_welding(question)):
        return run_general_pipeline(question)

    # Simulation & robotics-integration entries (Isaac Sim, sensors, teleop, DAQ)
    # use a different engineer persona at synthesis time. Resolve the domain from
    # the matched entries / retrieved passages first; if nothing matched, fall
    # back to intent keywords so an integration question still gets the robotics
    # persona (not the generic welding-engineer fallback).
    _INTEGRATION_CATEGORIES = {"simulation", "integration", "hardware"}
    knowledge_domain = "welding"
    if topics and any(t.get("category") in _INTEGRATION_CATEGORIES for t in topics):
        knowledge_domain = "integration"
    elif rag_passages and any(p.get("category") in _INTEGRATION_CATEGORIES for p in rag_passages):
        knowledge_domain = "integration"
    elif not topics and _is_integration_question(question):
        knowledge_domain = "integration"

    return {
        "question":        question,
        "pipeline_type":   "knowledge_advice",
        "matched_topics":  topics,
        "rag_passages":    rag_passages,
        "retrieval":       "hybrid_rag" if rag_passages else "keyword",
        "knowledge_domain": knowledge_domain,
        "summary_text":    build_knowledge_summary(question, topics),
    }


def run_param_pipeline(question: str, context: dict | None = None) -> dict:
    """
    Route parameter-optimisation queries to the welding param advisor.
    Returns a payload with pipeline_type='parameter_advice'.

    `context` is the parameter state carried from earlier turns in the same
    conversation ({material, thickness_mm, process, overrides, bounds}); a
    follow-up ("now stay in 1.5 diameter and more than 21 voltage") inherits the
    material/thickness it doesn't restate and accumulates pinned parameters from
    it. The returned payload includes an updated `param_context` to persist for
    the next turn.
    """
    from src.reasoning.param_advisor import (
        parse_param_query, recommend_parameters, merge_param_context,
    )

    parsed = parse_param_query(question)
    if parsed is None:
        # Generic request — default to medium mild steel, GMAW
        parsed = {"material": "mild_steel", "thickness_mm": 5.0, "process": None,
                  "overrides": {}, "bounds": {}, "override_ranges": {},
                  "defaulted": {"material": True, "thickness": True}}

    # Carry conversation state forward (material/thickness/process + pinned params).
    parsed = merge_param_context(parsed, context)

    rec = recommend_parameters(
        material=parsed["material"],
        thickness_mm=parsed["thickness_mm"],
        process=parsed.get("process") or "GMAW",
        overrides=parsed.get("overrides"),
        bounds=parsed.get("bounds"),
        override_ranges=parsed.get("override_ranges"),
    )
    rec["question"] = question
    rec["pipeline_type"] = "parameter_advice"
    # Surface which anchors we had to assume (the operator pinned a parameter but
    # didn't name a material/thickness), so synthesis can state the assumption.
    if "error" not in rec:
        rec["defaulted"] = parsed.get("defaulted", {})
        rec["carried"] = parsed.get("carried", [])
        # Persist the resolved state so the next follow-up can build on it.
        rec["param_context"] = {
            "material":     rec["material"],
            "thickness_mm": rec["thickness_mm"],
            "process":      parsed.get("process"),
            "overrides":    parsed.get("overrides", {}),
            "bounds":       parsed.get("bounds", {}),
            "override_ranges": parsed.get("override_ranges", {}),
        }
    return rec


def _window_at(fused: pd.DataFrame, query_time: datetime) -> pd.DataFrame:
    """Return the window row that contains query_time, or the closest one."""
    mask = (fused["window_start"] <= query_time) & (fused["window_end"] > query_time)
    if mask.any():
        return fused[mask].iloc[[0]]
    # Fallback: closest window by start time
    diffs = (fused["window_start"] - query_time).abs()
    return fused.iloc[[diffs.argmin()]]
