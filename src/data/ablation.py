"""
Multimodal Feature Mapping and Ablation Dataset Builder.

Provides:
  1. Authoritative programmatic modality-to-feature mapping:
     - SENSOR: 30 continuous/statistical telemetry features
     - LOGS: 9 event counts and status flags
     - OPERATOR_NOTES: 1 operator note presence flag ('has_note')
     - ENGINEERING_CONTEXT: 0 classifier features (RAG retrieval only)
  2. Reusable ablation feature selectors:
     - FULL (40 features)
     - SENSOR_ONLY (30 features)
     - WITHOUT_SENSOR (10 features)
     - WITHOUT_LOGS (31 features)
     - WITHOUT_OPERATOR_NOTES (39 features)
     - WITHOUT_ENGINEERING_CONTEXT (Marked NOT APPLICABLE)
  3. DistilBERT text ablation serializer.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.model.anomaly import (
    ALL_FEATURE_COLS,
    LOG_FEATURE_COLS,
    NOTE_FEATURE_COLS,
    SENSOR_FEATURE_COLS,
)
from src.model.bert_detector import _SENSORS

logger = logging.getLogger(__name__)

# Modality definitions
MODALITY_MAPPING: Dict[str, List[str]] = {
    "sensor": SENSOR_FEATURE_COLS,
    "logs": LOG_FEATURE_COLS,
    "operator_notes": NOTE_FEATURE_COLS,
    "engineering_context": [],  # Used exclusively by RAG, not in classifier
}

# Standard ablation condition names
ABLATION_CONDITIONS = [
    "FULL",
    "SENSOR_ONLY",
    "WITHOUT_SENSOR",
    "WITHOUT_LOGS",
    "WITHOUT_OPERATOR_NOTES",
    "WITHOUT_ENGINEERING_CONTEXT",
]


def get_ablation_columns(condition: str) -> Optional[List[str]]:
    """
    Return the exact list of feature columns for a given ablation condition.
    Returns None if the condition is not applicable to the classifier.
    """
    cond_upper = condition.upper()
    if cond_upper == "FULL":
        return list(ALL_FEATURE_COLS)
    elif cond_upper == "SENSOR_ONLY":
        return list(SENSOR_FEATURE_COLS)
    elif cond_upper == "WITHOUT_SENSOR":
        return list(LOG_FEATURE_COLS + NOTE_FEATURE_COLS)
    elif cond_upper == "WITHOUT_LOGS":
        return list(SENSOR_FEATURE_COLS + NOTE_FEATURE_COLS)
    elif cond_upper == "WITHOUT_OPERATOR_NOTES":
        return list(SENSOR_FEATURE_COLS + LOG_FEATURE_COLS)
    elif cond_upper == "WITHOUT_ENGINEERING_CONTEXT":
        # Engineering documents are RAG-only, not classifier features
        return None
    else:
        raise ValueError(f"Unknown ablation condition: '{condition}'. Valid conditions: {ABLATION_CONDITIONS}")


def build_ablation_matrix(
    df: pd.DataFrame,
    condition: str,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Extract feature matrix X for a given ablation condition.
    Returns (X, feature_names) or (None, None) if not applicable.
    """
    cols = get_ablation_columns(condition)
    if cols is None:
        return None, None

    available = [c for c in cols if c in df.columns]
    X = df[available].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    return X, available


def ablated_window_to_text(row: pd.Series, condition: str) -> Optional[str]:
    """
    Serialise one fused window into natural-language text reflecting the ablation condition.
    Returns None if the condition is not applicable.
    """
    cond_upper = condition.upper()
    if cond_upper == "WITHOUT_ENGINEERING_CONTEXT":
        return None

    include_sensor = cond_upper in ("FULL", "SENSOR_ONLY", "WITHOUT_LOGS", "WITHOUT_OPERATOR_NOTES")
    include_logs = cond_upper in ("FULL", "WITHOUT_SENSOR", "WITHOUT_OPERATOR_NOTES")
    include_notes = cond_upper in ("FULL", "WITHOUT_SENSOR", "WITHOUT_LOGS")

    parts = ["Welding station window."]

    if include_sensor:
        for col, label, unit in _SENSORS:
            mean = row.get(f"{col}_mean")
            std = row.get(f"{col}_std")
            rng = row.get(f"{col}_range")
            if mean is None or pd.isna(mean):
                continue
            seg = f"{label} mean {float(mean):.1f}{unit}"
            if std is not None and not pd.isna(std):
                seg += f" std {float(std):.1f}"
            if rng is not None and not pd.isna(rng):
                seg += f" range {float(rng):.1f}"
            parts.append(seg + ".")

    if include_logs:
        n_alarm = int(row.get("n_alarm", 0) or 0)
        n_warn = int(row.get("n_warning", 0) or 0)
        n_diag = int(row.get("n_diagnostic", 0) or 0)
        parts.append(f"Log: {n_alarm} alarms, {n_warn} warnings, {n_diag} diagnostics.")
        codes = str(row.get("unique_event_codes", "") or "").strip()
        parts.append(f"Event codes: {codes if codes and codes != 'nan' else 'none'}.")

    if include_notes:
        has_note = bool(row.get("has_note", False))
        parts.append("Operator note present." if has_note else "No operator note.")

    return " ".join(parts)


def build_ablation_texts(df: pd.DataFrame, condition: str) -> Optional[List[str]]:
    """
    Serialise every row of a fused DataFrame to text under an ablation condition.
    Returns None if not applicable.
    """
    if condition.upper() == "WITHOUT_ENGINEERING_CONTEXT":
        return None
    return [ablated_window_to_text(r, condition) for _, r in df.iterrows()]
