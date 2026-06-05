"""
Confidence scoring: derive a 0–1 confidence value and qualitative band
from the detector's anomaly probability and the SHAP driver agreement.

Formula:
  base_confidence = anomaly_prob  (from RF predict_proba)
  shap_agreement  = fraction of top-N SHAP drivers pointing toward anomaly
  confidence      = 0.7 * base_confidence + 0.3 * shap_agreement

Qualitative bands:
  high    >= 0.75
  medium  >= 0.45
  low     <  0.45
"""

from __future__ import annotations

import numpy as np


def compute_confidence(
    anomaly_prob: float,
    shap_drivers: list[dict],
    top_n: int = 5,
) -> dict:
    """
    Parameters
    ----------
    anomaly_prob : float in [0, 1] from AnomalyDetector.predict_proba
    shap_drivers : list of dicts from AnomalyExplainer (top-N entries)
    top_n        : how many drivers to consider for agreement score

    Returns
    -------
    dict with keys: score (float), band (str), anomaly_prob, shap_agreement
    """
    drivers_used = shap_drivers[:top_n]

    if drivers_used:
        toward_anomaly = sum(
            1 for d in drivers_used if d["direction"] == "toward_anomaly"
        )
        shap_agreement = toward_anomaly / len(drivers_used)
    else:
        shap_agreement = 0.0

    score = 0.7 * anomaly_prob + 0.3 * shap_agreement
    score = float(np.clip(score, 0.0, 1.0))

    return {
        "score":          round(score, 4),
        "band":           _band(score),
        "anomaly_prob":   round(anomaly_prob, 4),
        "shap_agreement": round(shap_agreement, 4),
    }


def _band(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"
