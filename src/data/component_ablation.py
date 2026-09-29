"""
Component Ablation Specification and Matrix Construction.

Defines the comprehensive pipeline component inventory and provides
controlled data matrix preparation and text serialization for Step 5
Component Ablation Experiments.

Pipeline Components:
  1. Preprocessing: StandardScaler (train-only fit)
  2. Feature Engineering:
     - Range-derived peak-to-peak amplitude features (*_range)
     - Physics/domain-derived heat input composite index (heat_input_*)
     - Granular event type log counts (n_*)
     - Second-moment statistical dispersion (*_std)
     - Extreme value bounds (*_min, *_max)
  3. Class Imbalance Handling:
     - Inverse frequency loss weighting (class_weight='balanced')
  4. Decision / Threshold:
     - Standard probability classification threshold (theta = 0.5)
  5. Out-of-Scope / Non-Inference Components (marked N/A):
     - Post-hoc explainers (SHAP, LIME, Attention)
     - Reasoning engines (Root Cause, Parameter Advisor, Confidence)
     - RAG knowledge retrieval and LLM chat synthesis
     - Feature selection / Dimensionality reduction (not implemented)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.model.anomaly import (
    ALL_FEATURE_COLS,
    LOG_FEATURE_COLS,
    NOTE_FEATURE_COLS,
    SENSOR_FEATURE_COLS,
    get_labels,
)

# -------------------------------------------------------------------------
# Feature Subsets
# -------------------------------------------------------------------------

RANGE_COLS = [c for c in ALL_FEATURE_COLS if c.endswith("_range")]
STD_COLS = [c for c in ALL_FEATURE_COLS if c.endswith("_std")]
MIN_MAX_COLS = [c for c in ALL_FEATURE_COLS if c.endswith("_min") or c.endswith("_max")]
HEAT_INPUT_COLS = [c for c in ALL_FEATURE_COLS if c.startswith("heat_input_")]
LOG_COUNT_COLS = [c for c in LOG_FEATURE_COLS if c.startswith("n_")]

# -------------------------------------------------------------------------
# Candidate Component Inventory (Complete Audit)
# -------------------------------------------------------------------------

COMPONENT_INVENTORY: List[Dict[str, Any]] = [
    {
        "component_name": "STANDARD_SCALER",
        "component_type": "Preprocessing",
        "implementation_location": "src/data/splits.py:prepare_tabular_data",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": True,
        "ablation_method": "Pass raw unscaled feature matrix (scale=False)",
        "expected_behavior": "Tree models (RF) are invariant to monotonic scaling; linear models (LR) suffer due to unscaled regularization penalties across feature scales",
        "quantitative_or_na": "QUANTITATIVE",
    },
    {
        "component_name": "RANGE_FEATURES",
        "component_type": "Feature Engineering",
        "implementation_location": "src/preprocess/sensor.py:segment_windows",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": True,
        "ablation_method": "Remove 6 window peak-to-peak amplitude features (*_range)",
        "expected_behavior": "Removes direct dynamic range information; models must rely on std and min/max bounds",
        "quantitative_or_na": "QUANTITATIVE",
    },
    {
        "component_name": "DOMAIN_HEAT_INPUT",
        "component_type": "Feature Engineering (Physics)",
        "implementation_location": "src/data/generate_synthetic.py / src/model/anomaly.py",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": True,
        "ablation_method": "Remove 5 physics-derived heat input features (heat_input_*)",
        "expected_behavior": "Assesses whether domain-composite quality index H = eta*(V*I)/v provides independent predictive power over raw current, voltage, and speed",
        "quantitative_or_na": "QUANTITATIVE",
    },
    {
        "component_name": "LOG_EVENT_COUNTS",
        "component_type": "Feature Engineering (Logs)",
        "implementation_location": "src/preprocess/logs.py:aggregate_windows",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": True,
        "ablation_method": "Remove 7 granular event count features (n_*), retaining boolean status flags (has_alarm, has_warning)",
        "expected_behavior": "Tests whether granular count aggregations provide marginal value beyond binary operational state indicators",
        "quantitative_or_na": "QUANTITATIVE",
    },
    {
        "component_name": "STATISTICAL_SPREAD_STD",
        "component_type": "Feature Engineering (Statistics)",
        "implementation_location": "src/preprocess/sensor.py:segment_windows",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": True,
        "ablation_method": "Remove 6 second-moment standard deviation features (*_std)",
        "expected_behavior": "Removes sensor signal dispersion; forces model to rely on location and extrema bounds",
        "quantitative_or_na": "QUANTITATIVE",
    },
    {
        "component_name": "EXTREME_BOUNDS_MIN_MAX",
        "component_type": "Feature Engineering (Statistics)",
        "implementation_location": "src/preprocess/sensor.py:segment_windows",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": True,
        "ablation_method": "Remove 12 minimum and maximum extreme value boundary features (*_min, *_max)",
        "expected_behavior": "Removes extreme value indicators; forces model to rely on mean and spread",
        "quantitative_or_na": "QUANTITATIVE",
    },
    {
        "component_name": "BALANCED_CLASS_WEIGHTS",
        "component_type": "Class Imbalance Handling",
        "implementation_location": "src/model/baselines.py / src/model/anomaly.py",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": True,
        "ablation_method": "Set class_weight=None (uniform loss weighting)",
        "expected_behavior": "Tests resilience under severe class imbalance (~1.8% anomalies); unweighted classifiers may collapse toward majority-class predictions",
        "quantitative_or_na": "QUANTITATIVE",
    },
    {
        "component_name": "DECISION_THRESHOLD",
        "component_type": "Decision / Inference",
        "implementation_location": "src/model/anomaly.py / scikit-learn default",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": True,
        "ablation_method": "Threshold sensitivity sweep over theta in [0.1, 0.9]",
        "expected_behavior": "Quantifies precision/recall trade-off across decision cutoffs",
        "quantitative_or_na": "QUANTITATIVE",
    },
    {
        "component_name": "FEATURE_SELECTION_DIM_REDUCTION",
        "component_type": "Feature Redundancy / Reduction",
        "implementation_location": "None (all available features retained)",
        "enabled_in_full_pipeline": False,
        "affects_predictive_inference": False,
        "ablation_method": "Not applicable — no feature selection or dimensionality reduction module exists in the active pipeline",
        "expected_behavior": "N/A",
        "quantitative_or_na": "NOT_APPLICABLE",
    },
    {
        "component_name": "OVERSAMPLING_UNDERSAMPLING",
        "component_type": "Class Imbalance Resampling",
        "implementation_location": "None (algorithmic class_weight used exclusively)",
        "enabled_in_full_pipeline": False,
        "affects_predictive_inference": False,
        "ablation_method": "Not applicable — data resampling (SMOTE, downsampling) is not implemented in the pipeline",
        "expected_behavior": "N/A",
        "quantitative_or_na": "NOT_APPLICABLE",
    },
    {
        "component_name": "SHAP_EXPLAINER",
        "component_type": "Post-Hoc Explainability",
        "implementation_location": "src/explain/shap_explainer.py",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": False,
        "ablation_method": "Not applicable — SHAP calculates post-hoc feature attributions after classification; zero predictive inference role",
        "expected_behavior": "N/A",
        "quantitative_or_na": "NOT_APPLICABLE",
    },
    {
        "component_name": "LIME_EXPLAINER",
        "component_type": "Post-Hoc Explainability",
        "implementation_location": "src/explain/lime_explainer.py",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": False,
        "ablation_method": "Not applicable — LIME fits local surrogate models for post-hoc explanation only",
        "expected_behavior": "N/A",
        "quantitative_or_na": "NOT_APPLICABLE",
    },
    {
        "component_name": "ATTENTION_EXPLAINER",
        "component_type": "Post-Hoc Explainability",
        "implementation_location": "src/explain/attention_explainer.py",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": False,
        "ablation_method": "Not applicable — extracts attention weights for operator-note visualization",
        "expected_behavior": "N/A",
        "quantitative_or_na": "NOT_APPLICABLE",
    },
    {
        "component_name": "ROOT_CAUSE_REASONING",
        "component_type": "Decision Reasoning",
        "implementation_location": "src/reasoning/root_cause.py",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": False,
        "ablation_method": "Not applicable — rule-based root cause identification triggers only after an anomaly is detected",
        "expected_behavior": "N/A",
        "quantitative_or_na": "NOT_APPLICABLE",
    },
    {
        "component_name": "PHYSICS_PARAM_ADVISOR",
        "component_type": "Advisory / Recommendation",
        "implementation_location": "src/reasoning/param_advisor.py",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": False,
        "ablation_method": "Not applicable — calculates physics parameter adjustments for operators after detection",
        "expected_behavior": "N/A",
        "quantitative_or_na": "NOT_APPLICABLE",
    },
    {
        "component_name": "CONFIDENCE_SCORING",
        "component_type": "Post-Hoc Heuristic",
        "implementation_location": "src/reasoning/confidence.py",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": False,
        "ablation_method": "Not applicable — computes heuristic confidence for LLM explanation payloads",
        "expected_behavior": "N/A",
        "quantitative_or_na": "NOT_APPLICABLE",
    },
    {
        "component_name": "RAG_RETRIEVAL_INDEX",
        "component_type": "External Knowledge Retrieval",
        "implementation_location": "src/rag/",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": False,
        "ablation_method": "Not applicable — engineering documents are utilized exclusively in RAG retrieval for advisory questions, never as model features",
        "expected_behavior": "N/A",
        "quantitative_or_na": "NOT_APPLICABLE",
    },
    {
        "component_name": "LLM_PROMPT_SYNTHESIZER",
        "component_type": "Conversational UI",
        "implementation_location": "src/chat/",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": False,
        "ablation_method": "Not applicable — conversational formatting and user interaction wrapper",
        "expected_behavior": "N/A",
        "quantitative_or_na": "NOT_APPLICABLE",
    },
    {
        "component_name": "TIME_SERIES_DENOISING",
        "component_type": "ETL Preprocessing",
        "implementation_location": "src/preprocess/sensor.py:denoise",
        "enabled_in_full_pipeline": True,
        "affects_predictive_inference": True,
        "ablation_method": "Baked into fused.parquet generation; ablaing would alter master synthetic data generation",
        "expected_behavior": "N/A for post-fusion evaluation; documented as fixed raw ETL transformation",
        "quantitative_or_na": "NOT_APPLICABLE",
    },
]

# -------------------------------------------------------------------------
# Component Ablation Experimental Conditions
# -------------------------------------------------------------------------

COMPONENT_CONDITIONS: Dict[str, Dict[str, Any]] = {
    "FULL_COMPONENTS": {
        "description": "Full reference pipeline (40 features, StandardScaler, balanced class weights)",
        "excluded_features": [],
        "scale": True,
        "class_weight": "balanced",
        "n_features": 40,
    },
    "WITHOUT_STANDARD_SCALER": {
        "description": "StandardScaler disabled — raw unscaled tabular features",
        "excluded_features": [],
        "scale": False,
        "class_weight": "balanced",
        "n_features": 40,
    },
    "WITHOUT_RANGE_FEATURES": {
        "description": "Ablate 6 window peak-to-peak amplitude features (*_range)",
        "excluded_features": RANGE_COLS,
        "scale": True,
        "class_weight": "balanced",
        "n_features": 34,
    },
    "WITHOUT_DOMAIN_HEAT_INPUT": {
        "description": "Ablate 5 physics-derived heat input features (heat_input_*)",
        "excluded_features": HEAT_INPUT_COLS,
        "scale": True,
        "class_weight": "balanced",
        "n_features": 35,
    },
    "WITHOUT_LOG_EVENT_COUNTS": {
        "description": "Ablate 7 event type counts (n_*), retaining boolean status flags",
        "excluded_features": LOG_COUNT_COLS,
        "scale": True,
        "class_weight": "balanced",
        "n_features": 33,
    },
    "WITHOUT_STD_FEATURES": {
        "description": "Ablate 6 second-moment standard deviation features (*_std)",
        "excluded_features": STD_COLS,
        "scale": True,
        "class_weight": "balanced",
        "n_features": 34,
    },
    "WITHOUT_MIN_MAX_FEATURES": {
        "description": "Ablate 12 extreme value boundary features (*_min, *_max)",
        "excluded_features": MIN_MAX_COLS,
        "scale": True,
        "class_weight": "balanced",
        "n_features": 28,
    },
    "WITHOUT_BALANCED_CLASS_WEIGHTS": {
        "description": "Ablate class_weight='balanced' — uniform class weighting",
        "excluded_features": [],
        "scale": True,
        "class_weight": None,
        "n_features": 40,
    },
}


def get_component_columns(condition: str) -> List[str]:
    """Return the ordered list of feature column names for a given condition."""
    if condition not in COMPONENT_CONDITIONS:
        raise ValueError(
            f"Unknown condition '{condition}'. Valid conditions: {list(COMPONENT_CONDITIONS.keys())}"
        )
    excluded = set(COMPONENT_CONDITIONS[condition]["excluded_features"])
    return [c for c in ALL_FEATURE_COLS if c not in excluded]


def build_component_matrix(
    df: pd.DataFrame, condition: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract raw numeric feature matrix X for the specified component condition.
    Guarantees no target or metadata leakage.
    """
    cols = get_component_columns(condition)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required feature columns for '{condition}': {missing}"
        )

    X = df[cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    return X, cols


def prepare_component_tabular_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    condition: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """
    Extract train and test feature matrices and labels for a component condition.

    CRITICAL LEAKAGE REQUIREMENT:
    - If scaling is active, StandardScaler is fitted ONLY on train data.
    - Test data is transformed using fitted scaler.
    - If condition is WITHOUT_STANDARD_SCALER, raw features are returned.
    """
    cfg = COMPONENT_CONDITIONS[condition]
    should_scale = cfg["scale"]

    X_train_raw, feat_names = build_component_matrix(train_df, condition)
    y_train = get_labels(train_df)

    X_test_raw, _ = build_component_matrix(test_df, condition)
    y_test = get_labels(test_df)

    scaler = None
    if should_scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        X_train = X_train_raw
        X_test = X_test_raw

    return X_train, y_train, X_test, y_test, feat_names, scaler


# -------------------------------------------------------------------------
# DistilBERT Text Serialization under Component Ablations
# -------------------------------------------------------------------------

_SENSORS_CONFIG = [
    ("welding_current", "current", "A"),
    ("arc_voltage", "voltage", "V"),
    ("welding_speed", "speed", "mm/min"),
    ("wire_feed_rate", "wire feed", "m/min"),
    ("shielding_gas_flow", "gas flow", "L/min"),
    ("heat_input", "heat input", "kJ/mm"),
]


def component_window_to_text(row: pd.Series, condition: str) -> str:
    """
    Serialize one window to text according to component ablation condition.
    """
    omit_range = condition == "WITHOUT_RANGE_FEATURES"
    omit_heat = condition == "WITHOUT_DOMAIN_HEAT_INPUT"
    omit_log_counts = condition == "WITHOUT_LOG_EVENT_COUNTS"
    omit_std = condition == "WITHOUT_STD_FEATURES"
    omit_min_max = condition == "WITHOUT_MIN_MAX_FEATURES"

    parts = ["Welding station window."]

    for col, label, unit in _SENSORS_CONFIG:
        if omit_heat and col == "heat_input":
            continue

        mean = row.get(f"{col}_mean")
        std = row.get(f"{col}_std")
        rng = row.get(f"{col}_range")
        c_min = row.get(f"{col}_min")
        c_max = row.get(f"{col}_max")

        if mean is None or pd.isna(mean):
            continue

        seg = f"{label} mean {float(mean):.1f}{unit}"
        if not omit_std and std is not None and not pd.isna(std):
            seg += f" std {float(std):.1f}"
        if not omit_range and rng is not None and not pd.isna(rng):
            seg += f" range {float(rng):.1f}"
        if not omit_min_max and c_min is not None and c_max is not None:
            seg += f" min {float(c_min):.1f} max {float(c_max):.1f}"
        parts.append(seg + ".")

    if not omit_log_counts:
        n_alarm = int(row.get("n_alarm", 0) or 0)
        n_warn = int(row.get("n_warning", 0) or 0)
        n_diag = int(row.get("n_diagnostic", 0) or 0)
        parts.append(f"Log: {n_alarm} alarms, {n_warn} warnings, {n_diag} diagnostics.")
    else:
        has_alarm = bool(row.get("has_alarm", False))
        has_warn = bool(row.get("has_warning", False))
        alarm_str = "Alarm active." if has_alarm else "No alarm."
        warn_str = "Warning active." if has_warn else "No warning."
        parts.append(f"Log: {alarm_str} {warn_str}")

    codes = str(row.get("unique_event_codes", "") or "").strip()
    parts.append(f"Event codes: {codes if codes and codes != 'nan' else 'none'}.")

    has_note = bool(row.get("has_note", False))
    parts.append("Operator note present." if has_note else "No operator note.")

    return " ".join(parts)


def build_component_texts(df: pd.DataFrame, condition: str) -> List[str]:
    """Serialize entire DataFrame to list of texts under component condition."""
    return [component_window_to_text(r, condition) for _, r in df.iterrows()]


def export_component_inventory(out_path: Path) -> Path:
    """Export the component inventory to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(COMPONENT_INVENTORY, f, indent=2)
    return out_path
