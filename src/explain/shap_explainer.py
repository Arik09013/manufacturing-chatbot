"""
SHAP explainability wrapper for the AnomalyDetector.

Uses TreeExplainer for exact, fast SHAP values on the RandomForest.
Produces:
  - per-prediction SHAP values (one value per feature)
  - a structured explanation payload: top-N drivers with direction + magnitude
  - an optional static waterfall plot saved to outputs/

Design note: SHAP values here represent contributions to anomaly probability
(class 1). Positive value = pushes toward anomaly; negative = toward normal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OUTPUTS = Path(__file__).parent.parent.parent / "outputs"


class AnomalyExplainer:
    """
    Wraps a trained AnomalyDetector with SHAP explanations.

    Parameters
    ----------
    detector : AnomalyDetector
        Fitted AnomalyDetector instance (must use a tree-based model).
    top_n : int
        Number of top SHAP drivers to include in each explanation.
    """

    def __init__(self, detector, top_n: int = 5):
        import shap

        self.detector = detector
        self.top_n = top_n
        self._explainer: Optional[shap.TreeExplainer] = None
        self._feature_names: list[str] = detector.feature_names

        # Initialise TreeExplainer on the underlying sklearn model
        self._explainer = shap.TreeExplainer(
            detector.model,
            feature_perturbation="tree_path_dependent",
        )

    def explain_row(self, row: pd.DataFrame) -> dict:
        """
        Compute SHAP values for a single prediction row.

        Parameters
        ----------
        row : pd.DataFrame
            Single-row DataFrame (same schema as training features).

        Returns
        -------
        dict with keys:
          anomaly_prob    : float
          shap_values     : list of {feature, value, shap} for top_n drivers
          base_value      : float (SHAP base value = expected model output)
          raw_shap        : np.ndarray of all SHAP values
        """
        from src.model.anomaly import get_feature_matrix

        X, _ = get_feature_matrix(row)
        # shap_values shape: (n_rows, n_features, n_classes) or (n_rows, n_features)
        sv = self._explainer.shap_values(X)

        # Handle both old SHAP (list of arrays) and new SHAP (3D ndarray)
        # We want class 1 (anomaly) SHAP values, shape (n_features,)
        if isinstance(sv, list):
            # Old API: list[class_idx] -> (n_samples, n_features)
            shap_vals = np.asarray(sv[1])[0]
        elif hasattr(sv, "ndim") and sv.ndim == 3:
            # New API: (n_samples, n_features, n_classes)
            shap_vals = sv[0, :, 1]
        else:
            shap_vals = np.asarray(sv)[0]

        shap_vals = shap_vals.ravel()

        prob = float(self.detector.predict_proba(row)[0])
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base = float(np.asarray(ev).ravel()[1])
        else:
            base = float(ev)

        # Build top-N drivers
        abs_vals = np.abs(shap_vals)
        top_idx = np.argsort(abs_vals)[::-1][: self.top_n]

        drivers = []
        for i in top_idx:
            feat = self._feature_names[i] if i < len(self._feature_names) else f"feat_{i}"
            sv_i = float(shap_vals[i])
            fv_i = float(X[0, i]) if X.shape[1] > i else 0.0
            drivers.append({
                "feature":    feat,
                "shap":       round(sv_i, 4),
                "direction":  "toward_anomaly" if sv_i > 0 else "toward_normal",
                "magnitude":  round(abs(sv_i), 4),
                "feature_value": round(fv_i, 4),
            })

        return {
            "anomaly_prob": round(prob, 4),
            "base_value":   round(base, 4),
            "shap_drivers": drivers,
            "raw_shap":     shap_vals,
        }

    def explain_text(self, shap_result: dict) -> str:
        """
        Convert SHAP result dict to a concise plain-English explanation paragraph.

        Example output:
          "The top contributing factors to this anomaly (prob=0.87) were:
           temperature_mean (+0.34, toward anomaly),
           vibration_max (+0.21, toward anomaly),
           has_alarm (+0.15, toward anomaly)."
        """
        prob = shap_result["anomaly_prob"]
        drivers = shap_result["shap_drivers"]

        lines = [f"Anomaly probability: {prob:.0%}."]
        lines.append("Top contributing sensor/log signals:")
        for d in drivers:
            direction = "anomaly" if d["direction"] == "toward_anomaly" else "normal"
            lines.append(
                f"  - {d['feature']}: SHAP={d['shap']:+.3f} (toward {direction})"
            )
        return "\n".join(lines)

    def plot_waterfall(
        self,
        shap_result: dict,
        row: pd.DataFrame,
        filename: str = "shap_plot.png",
    ) -> Path:
        """Save a SHAP waterfall plot to outputs/."""
        import shap
        import matplotlib.pyplot as plt
        from src.model.anomaly import get_feature_matrix

        X, _ = get_feature_matrix(row)
        sv = self._explainer.shap_values(X)
        if isinstance(sv, list):
            sv_class1_row = np.asarray(sv[1])[0]
        elif hasattr(sv, "ndim") and sv.ndim == 3:
            sv_class1_row = sv[0, :, 1]
        else:
            sv_class1_row = np.asarray(sv)[0]

        ev = self._explainer.expected_value
        base = float(np.asarray(ev).ravel()[1]) if isinstance(ev, (list, np.ndarray)) else float(ev)

        expl = shap.Explanation(
            values=sv_class1_row,
            base_values=base,
            data=X[0],
            feature_names=self._feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(expl, max_display=10, show=False)
        _OUTPUTS.mkdir(parents=True, exist_ok=True)
        out_path = _OUTPUTS / filename
        plt.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close()
        logger.info("SHAP plot saved -> %s", out_path)
        return out_path
