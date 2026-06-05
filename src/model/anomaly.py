"""
Anomaly detection over fused feature vectors.

MVP uses a RandomForestClassifier (supervised, because we have ground-truth
labels from the synthetic data). IsolationForest is available as a fallback
for unsupervised deployment.

Feature columns used:
  - All sensor aggregate columns (mean, std, min, max, range)
  - Log feature columns (n_* counts, has_alarm, has_warning)
  - Note presence flag (has_note)
  - Note embeddings are excluded by default (dimensionality vs. dataset size)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

_SENSOR_COLS = [
    "temperature", "vibration", "pressure", "rpm", "power_consumption"
]
_STATS = ["mean", "std", "min", "max", "range"]
SENSOR_FEATURE_COLS = [f"{c}_{s}" for c in _SENSOR_COLS for s in _STATS]

LOG_FEATURE_COLS = [
    "n_events", "n_alarm", "n_warning", "n_maintenance",
    "n_production", "n_diagnostic", "n_operational",
    "has_alarm", "has_warning",
]

NOTE_FEATURE_COLS = ["has_note"]

ALL_FEATURE_COLS = SENSOR_FEATURE_COLS + LOG_FEATURE_COLS + NOTE_FEATURE_COLS


def get_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Extract feature matrix X from a fused DataFrame.
    Returns (X, feature_names) using only the columns that exist.
    """
    available = [c for c in ALL_FEATURE_COLS if c in df.columns]
    X = df[available].values.astype(np.float32)
    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)
    return X, available


def get_labels(df: pd.DataFrame) -> np.ndarray:
    return df["is_anomaly"].astype(int).values


class AnomalyDetector:
    """
    Wraps either a supervised RandomForestClassifier or an unsupervised
    IsolationForest.  The supervised mode is used when labels are available.
    """

    def __init__(self, mode: str = "supervised", **kwargs):
        assert mode in ("supervised", "unsupervised")
        self.mode = mode
        self.feature_names: list[str] = []

        if mode == "supervised":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", None),
                min_samples_leaf=kwargs.get("min_samples_leaf", 2),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = IsolationForest(
                n_estimators=kwargs.get("n_estimators", 200),
                contamination=kwargs.get("contamination", "auto"),
                random_state=42,
                n_jobs=-1,
            )

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        X, self.feature_names = get_feature_matrix(df)
        if self.mode == "supervised":
            y = get_labels(df)
            self.model.fit(X, y)
            logger.info(
                "RF trained: %d samples, %d features, %.1f%% anomaly",
                len(y), X.shape[1], 100 * y.mean(),
            )
        else:
            self.model.fit(X)
            logger.info("IsolationForest trained: %d samples", len(X))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary anomaly labels (0/1)."""
        X, _ = get_feature_matrix(df)
        if self.mode == "supervised":
            return self.model.predict(X)
        else:
            raw = self.model.predict(X)
            return (raw == -1).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly probability in [0, 1]."""
        X, _ = get_feature_matrix(df)
        if self.mode == "supervised":
            return self.model.predict_proba(X)[:, 1]
        else:
            # IsolationForest score: more negative = more anomalous
            scores = self.model.score_samples(X)
            # Normalise to [0, 1] with sigmoid-like transform
            return 1 / (1 + np.exp(scores * 5))

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None
