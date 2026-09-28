"""
Conventional Baseline Models for Welding Anomaly Detection.

Implements and registers standard comparative baseline classifiers to benchmark
against the Random Forest reference model:
  1. Random Forest (Reference supervised baseline)
  2. Logistic Regression (Linear baseline with balanced weighting)
  3. Support Vector Machine (RBF kernel with balanced weighting)
  4. Isolation Forest (Unsupervised baseline from anomaly.py)
  5. Gradient Boosting Models (XGBoost, LightGBM, CatBoost) - checked dynamically

All models adhere to strict fairness requirements:
  - Identical feature matrices and column ordering
  - Preprocessing (StandardScaler) fitted strictly on training data
  - Identical train/test sample partitions
"""

from __future__ import annotations

import importlib.util
import logging
import pickle
import time
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def is_package_available(package_name: str) -> bool:
    """Check if a Python package is installed and importable."""
    return importlib.util.find_spec(package_name) is not None


class BaselineModelSpec:
    """Specification and factory for a baseline model."""

    def __init__(
        self,
        name: str,
        category: str,
        factory: Optional[Callable[[], Any]],
        config: Dict[str, Any],
        is_supervised: bool = True,
        class_imbalance_strategy: str = "class_weight='balanced'",
        notes: str = "",
    ):
        self.name = name
        self.category = category
        self.factory = factory
        self.config = config
        self.is_supervised = is_supervised
        self.class_imbalance_strategy = class_imbalance_strategy
        self.notes = notes

    @property
    def is_available(self) -> bool:
        return self.factory is not None

    def instantiate(self) -> Any:
        if not self.is_available:
            raise RuntimeError(f"Model '{self.name}' is not available: {self.notes}")
        return self.factory()


def _create_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _create_lr() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )


def _create_svc() -> SVC:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )


def _create_if() -> IsolationForest:
    return IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )


def get_baseline_registry() -> Dict[str, BaselineModelSpec]:
    """
    Return dictionary of all candidate baseline models, both available and unavailable.
    """
    registry: Dict[str, BaselineModelSpec] = {
        "Random Forest": BaselineModelSpec(
            name="Random Forest",
            category="Tree Ensemble (Supervised)",
            factory=_create_rf,
            config={
                "n_estimators": 200,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
            is_supervised=True,
            class_imbalance_strategy="class_weight='balanced' (inversely proportional to class frequencies)",
            notes="Primary reference baseline from MVP implementation.",
        ),
        "Logistic Regression": BaselineModelSpec(
            name="Logistic Regression",
            category="Linear Model (Supervised)",
            factory=_create_lr,
            config={
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42,
                "solver": "lbfgs",
            },
            is_supervised=True,
            class_imbalance_strategy="class_weight='balanced' (heuristically scales loss inversely to class frequencies)",
            notes="Standard conventional linear classifier with L2 regularization.",
        ),
        "Support Vector Machine": BaselineModelSpec(
            name="Support Vector Machine",
            category="Kernel Method (Supervised)",
            factory=_create_svc,
            config={
                "C": 1.0,
                "kernel": "rbf",
                "class_weight": "balanced",
                "probability": True,
                "random_state": 42,
            },
            is_supervised=True,
            class_imbalance_strategy="class_weight='balanced' (scales penalty parameter C per class)",
            notes="RBF kernel support vector classifier with Platt scaling for probability estimation.",
        ),
        "Isolation Forest": BaselineModelSpec(
            name="Isolation Forest",
            category="Tree Ensemble (Unsupervised)",
            factory=_create_if,
            config={
                "n_estimators": 200,
                "contamination": "auto",
                "random_state": 42,
                "n_jobs": -1,
            },
            is_supervised=False,
            class_imbalance_strategy="None (unsupervised anomaly isolation based on tree path length)",
            notes="Unsupervised anomaly detector from src/model/anomaly.py.",
        ),
    }

    # Dynamic check for external gradient boosting libraries
    if is_package_available("xgboost"):
        import xgboost as xgb

        registry["XGBoost"] = BaselineModelSpec(
            name="XGBoost",
            category="Gradient Boosted Trees (Supervised)",
            factory=lambda: xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=1.0,
                random_state=42,
                n_jobs=-1,
            ),
            config={
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "scale_pos_weight": "auto",
                "random_state": 42,
            },
            is_supervised=True,
            class_imbalance_strategy="scale_pos_weight = (n_neg / n_pos)",
            notes="Installed gradient boosted decision trees.",
        )
    else:
        registry["XGBoost"] = BaselineModelSpec(
            name="XGBoost",
            category="Gradient Boosted Trees (Supervised)",
            factory=None,
            config={},
            is_supervised=True,
            class_imbalance_strategy="scale_pos_weight",
            notes="Not executed — dependency unavailable in virtual environment.",
        )

    if is_package_available("lightgbm"):
        import lightgbm as lgb

        registry["LightGBM"] = BaselineModelSpec(
            name="LightGBM",
            category="Gradient Boosted Trees (Supervised)",
            factory=lambda: lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            config={
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "class_weight": "balanced",
                "random_state": 42,
            },
            is_supervised=True,
            class_imbalance_strategy="class_weight='balanced'",
            notes="Installed LightGBM classifier.",
        )
    else:
        registry["LightGBM"] = BaselineModelSpec(
            name="LightGBM",
            category="Gradient Boosted Trees (Supervised)",
            factory=None,
            config={},
            is_supervised=True,
            class_imbalance_strategy="class_weight='balanced'",
            notes="Not executed — dependency unavailable in virtual environment.",
        )

    if is_package_available("catboost"):
        import catboost as cb

        registry["CatBoost"] = BaselineModelSpec(
            name="CatBoost",
            category="Gradient Boosted Trees (Supervised)",
            factory=lambda: cb.CatBoostClassifier(
                iterations=100,
                depth=4,
                learning_rate=0.1,
                auto_class_weights="Balanced",
                random_seed=42,
                verbose=0,
            ),
            config={
                "iterations": 100,
                "depth": 4,
                "learning_rate": 0.1,
                "auto_class_weights": "Balanced",
                "random_seed": 42,
            },
            is_supervised=True,
            class_imbalance_strategy="auto_class_weights='Balanced'",
            notes="Installed CatBoost classifier.",
        )
    else:
        registry["CatBoost"] = BaselineModelSpec(
            name="CatBoost",
            category="Gradient Boosted Trees (Supervised)",
            factory=None,
            config={},
            is_supervised=True,
            class_imbalance_strategy="auto_class_weights='Balanced'",
            notes="Not executed — dependency unavailable in virtual environment.",
        )

    return registry


def compute_model_complexity(model: Any) -> Dict[str, Any]:
    """
    Extract structural complexity and serialized footprint of a fitted model.
    """
    # 1. Serialized byte size
    try:
        serialized_size = len(pickle.dumps(model))
    except Exception:
        serialized_size = None

    # 2. Structural parameters / nodes
    complexity_info: Dict[str, Any] = {
        "serialized_size_bytes": serialized_size,
        "structural_parameters": None,
        "complexity_description": "N/A",
    }

    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        n_params = int(model.coef_.size + model.intercept_.size)
        complexity_info["structural_parameters"] = n_params
        complexity_info["complexity_description"] = f"{n_params} linear coefficients + bias"
    elif hasattr(model, "support_vectors_"):
        n_sv = int(model.support_vectors_.shape[0])
        total_sv_elements = int(model.support_vectors_.size)
        complexity_info["structural_parameters"] = total_sv_elements
        complexity_info["complexity_description"] = f"{n_sv} support vectors ({total_sv_elements} float coordinates)"
    elif hasattr(model, "estimators_"):
        total_nodes = int(sum(est.tree_.node_count for est in model.estimators_))
        n_trees = len(model.estimators_)
        complexity_info["structural_parameters"] = total_nodes
        complexity_info["complexity_description"] = f"{total_nodes} decision tree nodes across {n_trees} trees"

    return complexity_info


def fit_and_evaluate_model(
    model_spec: BaselineModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float, Dict[str, Any]]:
    """
    Fit a baseline model on (X_train, y_train) and predict on X_test.

    Returns
    -------
    (y_pred, y_prob, train_time_sec, inference_time_sec, complexity_info)
    """
    model = model_spec.instantiate()

    # 1. Training with timing
    t0 = time.perf_counter()
    if model_spec.is_supervised:
        # Handle gradient boosting dynamic scale_pos_weight if needed
        if model_spec.name == "XGBoost" and hasattr(model, "scale_pos_weight"):
            n_pos = int(np.sum(y_train))
            n_neg = int(len(y_train) - n_pos)
            if n_pos > 0:
                model.set_params(scale_pos_weight=n_neg / n_pos)
        model.fit(X_train, y_train)
    else:
        # Unsupervised (Isolation Forest)
        model.fit(X_train)
    train_time = time.perf_counter() - t0

    # 2. Inference with timing
    t1 = time.perf_counter()
    if model_spec.is_supervised:
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-scores))
        else:
            y_prob = y_pred.astype(float)
    else:
        # Unsupervised anomaly detection (Isolation Forest)
        raw_pred = model.predict(X_test)
        y_pred = (raw_pred == -1).astype(int)
        scores = model.score_samples(X_test)
        # Normalise to [0, 1] using standard sigmoid transform from anomaly.py
        y_prob = 1 / (1 + np.exp(scores * 5))
    inference_time = time.perf_counter() - t1

    # 3. Model complexity & size
    complexity = compute_model_complexity(model)

    return y_pred, y_prob, train_time, inference_time, complexity
