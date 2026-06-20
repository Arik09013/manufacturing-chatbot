"""
LIME explainability wrapper for the AnomalyDetector.

Complements the SHAP explainer (`shap_explainer.py`). Where SHAP gives exact,
game-theoretic feature attributions for the tree model, LIME builds a *local
linear surrogate* around a single prediction by perturbing the input — a second,
independent view of "why this window was flagged".

Design note: LIME weights here are for class 1 (anomaly). Positive weight =
this feature condition pushed the prediction toward anomaly; negative = toward
normal. Showing SHAP and LIME side by side satisfies the thesis requirement of
multiple XAI methods and lets the operator cross-check the explanation.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "anomaly.joblib"


class LimeAnomalyExplainer:
    """
    Wraps a trained AnomalyDetector with LIME explanations.

    Parameters
    ----------
    detector : AnomalyDetector
        Fitted detector (its `.model` must expose predict_proba).
    training_data : np.ndarray
        Representative feature matrix used by LIME to learn perturbation
        statistics (means/stds and discretisation bins).
    top_n : int
        Number of top LIME drivers to return per explanation.
    """

    def __init__(self, detector, training_data: np.ndarray, top_n: int = 5):
        from lime.lime_tabular import LimeTabularExplainer

        self.detector = detector
        self.top_n = top_n
        self.feature_names: list[str] = detector.feature_names

        self._explainer = LimeTabularExplainer(
            training_data=np.asarray(training_data, dtype=np.float64),
            feature_names=self.feature_names,
            class_names=["normal", "anomaly"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

    def _feature_from_condition(self, condition: str) -> str:
        """
        LIME returns conditions like 'welding_current_std > 5.20' or
        '1.0 < heat_input_mean <= 2.0'. Map back to the bare feature name.
        """
        # Longest feature name that appears in the condition string wins
        matches = [f for f in self.feature_names if f in condition]
        if matches:
            return max(matches, key=len)
        # Fallback: first token that looks like an identifier
        m = re.search(r"[A-Za-z_][A-Za-z0-9_]+", condition)
        return m.group(0) if m else condition

    def explain_row(self, row: pd.DataFrame) -> dict:
        """
        Compute a LIME explanation for a single prediction row.

        Returns
        -------
        dict with keys:
          anomaly_prob : float
          lime_drivers : list of {feature, condition, weight, direction, magnitude}
        """
        from src.model.anomaly import get_feature_matrix

        X, _ = get_feature_matrix(row)
        exp = self._explainer.explain_instance(
            X[0].astype(np.float64),
            self.detector.model.predict_proba,
            num_features=self.top_n,
            labels=(1,),
        )

        prob = float(self.detector.predict_proba(row)[0])
        drivers = []
        for condition, weight in exp.as_list(label=1):
            drivers.append({
                "feature":   self._feature_from_condition(condition),
                "condition": condition,
                "weight":    round(float(weight), 4),
                "direction": "toward_anomaly" if weight > 0 else "toward_normal",
                "magnitude": round(abs(float(weight)), 4),
            })
        return {"anomaly_prob": round(prob, 4), "lime_drivers": drivers}

    def explain_text(self, lime_result: dict) -> str:
        """Concise plain-text rendering of a LIME explanation."""
        lines = [f"Anomaly probability: {lime_result['anomaly_prob']:.0%} (LIME local view):"]
        for d in lime_result["lime_drivers"]:
            direction = "anomaly" if d["direction"] == "toward_anomaly" else "normal"
            lines.append(f"  - {d['condition']}: weight={d['weight']:+.3f} (toward {direction})")
        return "\n".join(lines)


@lru_cache(maxsize=1)
def _background_sample(n: int = 500) -> np.ndarray:
    """Load a cached representative feature sample for LIME perturbation stats."""
    from src.fusion.fuse import load_fused
    from src.model.anomaly import get_feature_matrix

    df = load_fused()
    X, _ = get_feature_matrix(df)
    if len(X) > n:
        idx = np.random.RandomState(42).choice(len(X), n, replace=False)
        X = X[idx]
    return X


@lru_cache(maxsize=1)
def get_default_lime_explainer(top_n: int = 5) -> "LimeAnomalyExplainer":
    """Build (once, cached) a LIME explainer from the saved model + a data sample."""
    import joblib

    bundle = joblib.load(_MODEL_PATH)
    detector = bundle["detector"]
    return LimeAnomalyExplainer(detector, _background_sample(), top_n=top_n)
