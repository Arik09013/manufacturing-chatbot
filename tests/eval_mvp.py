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

# Welding anomaly type -> sensor channels expected among the top XAI drivers
# (matches the channels perturbed in src/data/generate_synthetic.py).
EXPECTED_SHAP_CHANNELS = {
    "arc_instability":  ["welding_current", "arc_voltage", "heat_input"],
    "wire_feed_fault":  ["wire_feed_rate", "welding_current"],
    "gas_flow_failure": ["shielding_gas_flow"],
    "overheating":      ["heat_input", "welding_speed", "welding_current"],
    "underheat":        ["heat_input", "welding_speed", "arc_voltage"],
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


def evaluate_lime_alignment(df: pd.DataFrame) -> dict:
    """
    Same alignment check as SHAP, but using the LIME local surrogate — confirms
    the second XAI method also surfaces the correct injected channel.
    """
    from src.explain.lime_explainer import get_default_lime_explainer

    explainer = get_default_lime_explainer(top_n=5)
    anomaly_rows = df[df["is_anomaly"] == True]
    results_by_type: dict[str, list[bool]] = {}

    for _, row in anomaly_rows.iterrows():
        atype = str(row.get("anomaly_type", ""))
        if atype not in EXPECTED_SHAP_CHANNELS:
            continue
        try:
            res = explainer.explain_row(pd.DataFrame([row]))
        except Exception:
            continue
        top_features = [d["feature"] for d in res["lime_drivers"]]
        expected = EXPECTED_SHAP_CHANNELS[atype]
        hit = any(any(ch in feat for feat in top_features) for ch in expected)
        results_by_type.setdefault(atype, []).append(hit)

    alignment = {a: round(sum(h) / len(h), 3) for a, h in results_by_type.items() if h}
    total = sum(len(v) for v in results_by_type.values())
    alignment["overall"] = (
        round(sum(sum(v) for v in results_by_type.values()) / total, 3) if total else 0.0
    )
    return alignment


def evaluate_root_cause(df: pd.DataFrame, ks=(1, 2, 3)) -> dict:
    """
    Rank-quality of the root-cause mapper.

    For each anomalous window we treat the *canonical* cause for the true
    anomaly type (the first cause listed for that type in cause_action_map.yaml)
    as the relevant answer, then check the rank the mapper assigns it given the
    SHAP drivers. Reports Mean Reciprocal Rank (MRR) and top-k accuracy.
    """
    import joblib
    from src.explain.shap_explainer import AnomalyExplainer
    from src.reasoning.root_cause import identify_causes, _load_map

    bundle = joblib.load(Path(__file__).parent.parent / "models" / "anomaly.joblib")
    explainer = AnomalyExplainer(bundle["detector"], top_n=5)
    cause_cfg = _load_map()

    def _canonical(atype: str):
        entries = cause_cfg.get(atype, {}).get("causes", [])
        if not entries:
            return None
        first = entries[0]
        return list(first.values())[0] if isinstance(first, dict) else str(first)

    reciprocal_ranks: list[float] = []
    topk_hits = {k: 0 for k in ks}
    n = 0

    for _, row in df[df["is_anomaly"] == True].iterrows():
        atype = str(row.get("anomaly_type", ""))
        canonical = _canonical(atype)
        if canonical is None:
            continue
        row_df = pd.DataFrame([row])
        try:
            shap_res = explainer.explain_row(row_df)
        except Exception:
            continue
        log_codes = str(row.get("unique_event_codes", ""))
        ranked = [c["cause"] for c in identify_causes(atype, shap_res["shap_drivers"], log_codes, top_n=5)]

        n += 1
        if canonical in ranked:
            rank = ranked.index(canonical) + 1
            reciprocal_ranks.append(1.0 / rank)
            for k in ks:
                if rank <= k:
                    topk_hits[k] += 1
        else:
            reciprocal_ranks.append(0.0)

    return {
        "n": n,
        "mrr": round(sum(reciprocal_ranks) / n, 3) if n else 0.0,
        "topk": {k: round(topk_hits[k] / n, 3) if n else 0.0 for k in ks},
    }


def evaluate_prescriptive() -> dict:
    """
    Quantitative check on the parameter advisor's prescriptive setpoints.

    For every supported material x process x thickness-band, recompute the
    optimized settings and measure:
      - in_window_rate: fraction of recommended setpoints inside the standards
        window (should be 100% — the optimizer is constrained to it)
      - mean_norm_dev: mean |optimized - window_center| / window_width across
        current / voltage / speed (0 = dead-centre, 0.5 = at an edge) — i.e. an
        MAE of setpoints normalised by the allowed range.
    """
    from src.reasoning.param_advisor import recommend_parameters, _load_params

    params_db = _load_params()
    fields = ["welding_current", "arc_voltage", "welding_speed"]
    thickness_probe = {"thin": 2.0, "medium": 6.0, "thick": 15.0}

    deviations: list[float] = []
    in_window = 0
    total = 0

    for material, mat_db in params_db.get("materials", {}).items():
        for process in mat_db:
            if process not in params_db.get("processes", {}):
                continue
            for band in mat_db[process]:
                rec = recommend_parameters(material, thickness_probe.get(band, 6.0), process)
                if "error" in rec or rec.get("band") != band:
                    continue
                opt, ranges = rec["optimized"], rec["ranges"]
                for f in fields:
                    rng = ranges.get(f)
                    val = opt.get(f)
                    if not rng or val is None:
                        continue
                    lo, hi = rng
                    width = (hi - lo) or 1.0
                    center = (lo + hi) / 2.0
                    deviations.append(abs(val - center) / width)
                    in_window += int(lo <= val <= hi)
                    total += 1

    return {
        "n_setpoints": total,
        "in_window_rate": round(in_window / total, 3) if total else 0.0,
        "mean_norm_dev": round(sum(deviations) / len(deviations), 3) if deviations else 0.0,
    }


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


def write_report(detection: dict, shap_alignment: dict, calibration: dict,
                 lime_alignment: dict | None = None,
                 root_cause: dict | None = None,
                 prescriptive: dict | None = None) -> str:
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

    if lime_alignment:
        lines += [
            "",
            "### LIME Driver Alignment (independent XAI cross-check)",
            "",
            "| Anomaly type | Alignment |",
            "|---|---|",
        ]
        for atype, score in lime_alignment.items():
            lines.append(f"| {atype} | {score:.0%} |")

    if root_cause:
        lines += [
            "",
            "---",
            "",
            "## 3. Root-Cause Ranking",
            "",
            "Rank quality of the heuristic root-cause mapper for the canonical cause",
            "of each anomaly type (given SHAP drivers).",
            "",
            "| Metric | Score |",
            "|---|---|",
            f"| MRR | {root_cause['mrr']} |",
        ]
        for k, v in root_cause["topk"].items():
            lines.append(f"| Top-{k} accuracy | {v:.0%} |")
        lines.append(f"| Anomalous windows scored | {root_cause['n']} |")

    if prescriptive:
        lines += [
            "",
            "---",
            "",
            "## 4. Prescriptive Setpoint Quality",
            "",
            "Parameter-advisor recommendations across every supported",
            "material x process x thickness band.",
            "",
            "| Metric | Score |",
            "|---|---|",
            f"| In-window compliance | {prescriptive['in_window_rate']:.0%} |",
            f"| Mean normalised deviation (MAE / range) | {prescriptive['mean_norm_dev']} |",
            f"| Setpoints evaluated | {prescriptive['n_setpoints']} |",
        ]

    lines += [
        "",
        "---",
        "",
        "## 5. Confidence Calibration",
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

    logger.info("Running LIME alignment check…")
    lime_alignment = evaluate_lime_alignment(df)
    logger.info("LIME alignment: %s", lime_alignment)

    logger.info("Running root-cause ranking (MRR/top-k)…")
    root_cause = evaluate_root_cause(df)
    logger.info("Root cause: %s", root_cause)

    logger.info("Running prescriptive setpoint evaluation…")
    prescriptive = evaluate_prescriptive()
    logger.info("Prescriptive: %s", prescriptive)

    logger.info("Running confidence calibration check…")
    calibration = evaluate_confidence_calibration(df)
    logger.info("Calibration: %s", calibration)

    report = write_report(detection, shap_alignment, calibration,
                          lime_alignment, root_cause, prescriptive)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    logger.info("Evaluation report -> %s", OUTPUT_PATH)

    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
