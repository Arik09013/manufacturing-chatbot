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

MACHINES = ["machine_1", "machine_2", "machine_3"]
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

    # Machine detection
    machine_id = None
    for m in MACHINES:
        if m.replace("_", " ") in question_lower or m in question_lower:
            machine_id = m
            break
    # Shorthand "line 1/2/3" → "machine_1/2/3"
    if machine_id is None:
        match = re.search(r"line\s*(\d)", question_lower)
        if match:
            machine_id = f"machine_{match.group(1)}"

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

    # --- 8. Root cause ---
    log_codes = str(target["unique_event_codes"].iloc[0]) if "unique_event_codes" in target.columns else ""
    causes = identify_causes(anomaly_type, shap_result["shap_drivers"], log_codes)

    # --- 9. Recommendation ---
    recommendation = get_recommendation(anomaly_type) if is_anomaly else {}

    # --- 10. Confidence ---
    confidence = compute_confidence(anomaly_prob, shap_result["shap_drivers"])

    window_start = target["window_start"].iloc[0]
    window_end = target["window_end"].iloc[0]

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
    }


def _window_at(fused: pd.DataFrame, query_time: datetime) -> pd.DataFrame:
    """Return the window row that contains query_time, or the closest one."""
    mask = (fused["window_start"] <= query_time) & (fused["window_end"] > query_time)
    if mask.any():
        return fused[mask].iloc[[0]]
    # Fallback: closest window by start time
    diffs = (fused["window_start"] - query_time).abs()
    return fused.iloc[[diffs.argmin()]]
