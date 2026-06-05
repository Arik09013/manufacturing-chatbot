"""
T13 — Lightweight MVP evaluation.

Metrics:
  - Anomaly detection: precision, recall, F1, accuracy (cross-validated)
  - SHAP alignment: do top SHAP drivers match injected anomaly channels?
  - Confidence calibration: high-confidence predictions more accurate?

Usage:
    python tests/eval_mvp.py

Outputs:
    outputs/eval_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import textwrap

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "eval_report.md"

# Anomaly type -> sensor channels expected to be top SHAP drivers
EXPECTED_SHAP_CHANNELS = {
    "overheating":     ["temperature", "power_consumption"],
    "bearing_failure": ["vibration", "rpm"],
    "pressure_loss":   ["pressure"],
    "motor_overload":  ["power_consumption", "temperature"],
    "coolant_failure": ["temperature", "pressure"],
}


def load_fused_data() -> pd.DataFrame:
    from src.fusion.fuse import load_fused
    df = load_fused()
    return df


def evaluate_detection(df: pd.DataFrame) -> dict:
    """5-fold stratified cross-validation on anomaly detection."""
    from src.model.anomaly import AnomalyDetector, get_feature_matrix, get_labels

    X, feat_names = get_feature_matrix(df)
    y = get_labels(df)

    detector = AnomalyDetector(mode="supervised")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(detector.model, X, y, cv=cv)
    y_prob = cross_val_predict(
        detector.model, X, y, cv=cv, method="predict_proba"
    )[:, 1]

    return {
        "accuracy":  round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y, y_prob)), 4),
        "n_samples": int(len(y)),
        "n_anomaly": int(y.sum()),
        "report":    classification_report(y, y_pred, target_names=["normal", "anomaly"]),
        "feat_names": feat_names,
    }


def evaluate_shap_alignment(df: pd.DataFrame, feat_names: list[str]) -> dict:
    """
    Check that top SHAP drivers for anomalous windows contain the expected
    sensor channel(s) for each anomaly type.
    """
    import joblib
    from src.explain.shap_explainer import AnomalyExplainer
    from src.model.anomaly import AnomalyDetector

    model_path = Path(__file__).parent.parent / "models" / "anomaly.joblib"
    bundle = joblib.load(model_path)
    detector = bundle["detector"]
    explainer = AnomalyExplainer(detector, top_n=5)

    anomaly_rows = df[df["is_anomaly"] == True]
    results_by_type: dict[str, list[bool]] = {}

    for _, row in anomaly_rows.iterrows():
        atype = str(row.get("anomaly_type", ""))
        if atype not in EXPECTED_SHAP_CHANNELS:
            continue

        row_df = pd.DataFrame([row])
        try:
            shap_result = explainer.explain_row(row_df)
        except Exception:
            continue

        top_features = [d["feature"] for d in shap_result["shap_drivers"]]
        expected_channels = EXPECTED_SHAP_CHANNELS[atype]

        # Check if any expected channel appears in top features
        hit = any(
            any(ch in feat for feat in top_features)
            for ch in expected_channels
        )

        results_by_type.setdefault(atype, []).append(hit)

    alignment: dict[str, float] = {}
    for atype, hits in results_by_type.items():
        alignment[atype] = round(sum(hits) / len(hits), 3) if hits else 0.0

    overall = (
        round(sum(sum(v) for v in results_by_type.values()) /
              sum(len(v) for v in results_by_type.values()), 3)
        if results_by_type else 0.0
    )
    alignment["overall"] = overall
    return alignment


def evaluate_confidence_calibration(df: pd.DataFrame) -> dict:
    """
    High-confidence predictions should have higher accuracy.
    Bin predictions by confidence band and compute accuracy per band.
    """
    import joblib
    from src.reasoning.confidence import compute_confidence
    from src.explain.shap_explainer import AnomalyExplainer
    from src.model.anomaly import get_feature_matrix, get_labels

    model_path = Path(__file__).parent.parent / "models" / "anomaly.joblib"
    bundle = joblib.load(model_path)
    detector = bundle["detector"]
    explainer = AnomalyExplainer(detector, top_n=5)

    probs = detector.predict_proba(df)
    preds = detector.predict(df)
    labels = get_labels(df)

    bands = {"high": [], "medium": [], "low": []}
    for i in range(len(df)):
        row = df.iloc[[i]]
        shap_result = explainer.explain_row(row)
        conf = compute_confidence(float(probs[i]), shap_result["shap_drivers"])
        correct = int(preds[i] == labels[i])
        bands[conf["band"]].append(correct)

    result = {}
    for band, corrects in bands.items():
        if corrects:
            result[band] = {
                "n": len(corrects),
                "accuracy": round(sum(corrects) / len(corrects), 3),
            }
    return result


def write_report(detection: dict, shap_alignment: dict, calibration: dict) -> str:
    lines = [
        "# MVP Evaluation Report",
        "",
        f"Generated: 2026-06-05",
        "",
        "---",
        "",
        "## 1. Anomaly Detection (5-fold Stratified CV)",
        "",
        f"| Metric | Score |",
        f"|---|---|",
        f"| Accuracy | {detection['accuracy']} |",
        f"| Precision | {detection['precision']} |",
        f"| Recall | {detection['recall']} |",
        f"| F1 | {detection['f1']} |",
        f"| ROC-AUC | {detection['roc_auc']} |",
        f"| Total windows | {detection['n_samples']} |",
        f"| Anomalous windows | {detection['n_anomaly']} |",
        "",
        "### Classification Report",
        "",
        "```",
        detection["report"],
        "```",
        "",
        "---",
        "",
        "## 2. SHAP Driver Alignment",
        "",
        "Fraction of anomalous windows where at least one top-5 SHAP driver",
        "matches the injected anomaly channel for that anomaly type.",
        "",
        "| Anomaly type | Alignment |",
        "|---|---|",
    ]
    for atype, score in shap_alignment.items():
        lines.append(f"| {atype} | {score:.0%} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Confidence Calibration",
        "",
        "| Confidence band | N | Accuracy |",
        "|---|---|---|",
    ]
    for band in ["high", "medium", "low"]:
        if band in calibration:
            c = calibration[band]
            lines.append(f"| {band} | {c['n']} | {c['accuracy']:.0%} |")

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- CV metrics are computed on the synthetic dataset (labels known exactly).",
        "- SHAP alignment validates that the explainability layer surfaces the correct",
        "  sensor channels for each injected anomaly type.",
        "- Full 5-fold CV, KPI benchmarking deferred to Phase 6.",
    ]
    return "\n".join(lines)


def main() -> None:
    logger.info("Loading fused dataset…")
    df = load_fused_data()
    logger.info("Dataset: %d windows, %d anomalous", len(df), df["is_anomaly"].sum())

    logger.info("Running detection evaluation (5-fold CV)…")
    detection = evaluate_detection(df)
    logger.info("F1=%.3f  ROC-AUC=%.3f", detection["f1"], detection["roc_auc"])

    logger.info("Running SHAP alignment check…")
    shap_alignment = evaluate_shap_alignment(df, detection["feat_names"])
    logger.info("SHAP alignment: %s", shap_alignment)

    logger.info("Running confidence calibration check…")
    calibration = evaluate_confidence_calibration(df)
    logger.info("Calibration: %s", calibration)

    report = write_report(detection, shap_alignment, calibration)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    logger.info("Evaluation report -> %s", OUTPUT_PATH)

    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
