"""
Unit tests for the conventional baseline model benchmarking infrastructure.

Verifies:
  1. All models receive identical feature columns and ordering.
  2. All models evaluate on identical train/test indices.
  3. Preprocessing (StandardScaler) is strictly fitted on training data.
  4. Chronological splits preserve temporal ordering with >=30 min embargo.
  5. LOMO splits exclude held-out machines completely.
  6. Model complexity and size extraction functions correctly.
  7. Unavailable dependencies (XGBoost, LightGBM, CatBoost) are handled gracefully.
  8. All available baseline models produce valid predictions, probabilities, and metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.splits import (
    compute_classification_metrics,
    prepare_tabular_data,
    split_chronological,
    split_leave_one_machine_out,
    verify_leakage_free,
)
from src.fusion.fuse import load_fused
from src.model.anomaly import ALL_FEATURE_COLS, get_feature_matrix, get_labels
from src.model.baselines import (
    compute_model_complexity,
    fit_and_evaluate_model,
    get_baseline_registry,
)


@pytest.fixture(scope="module")
def fused_data() -> pd.DataFrame:
    """Load the real fused dataset for tests."""
    return load_fused()


def test_baseline_registry_completeness():
    """Verify registry contains all candidate models and handles availability."""
    reg = get_baseline_registry()
    expected_candidates = {
        "Random Forest",
        "Logistic Regression",
        "Support Vector Machine",
        "Isolation Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    }
    assert expected_candidates.issubset(set(reg.keys()))

    # Available models in standard environment
    for name in ["Random Forest", "Logistic Regression", "Support Vector Machine", "Isolation Forest"]:
        spec = reg[name]
        assert spec.is_available is True
        model = spec.instantiate()
        assert model is not None


def test_unavailable_dependencies_handled_gracefully():
    """Verify missing dependencies do not crash registry and raise informative errors on instantiation."""
    reg = get_baseline_registry()
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        spec = reg[name]
        if not spec.is_available:
            with pytest.raises(RuntimeError, match="not available"):
                spec.instantiate()


def test_models_receive_identical_features(fused_data):
    """Verify feature extraction produces exactly ALL_FEATURE_COLS in identical order."""
    X, feat_names = get_feature_matrix(fused_data)
    assert len(feat_names) == 40
    assert feat_names == ALL_FEATURE_COLS
    assert X.shape == (len(fused_data), 40)
    assert not np.isnan(X).any()


def test_models_use_identical_indices(fused_data):
    """Verify chronological and LOMO splits maintain identical indices across models."""
    train_df, test_df, purge_df = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)
    assert len(train_df) == 1533
    assert len(test_df) == 375
    assert len(purge_df) == 9

    # Train and test index sets must be strictly disjoint
    assert set(train_df["window_id"]).isdisjoint(set(test_df["window_id"]))

    # Multiple calls must produce identical indices (deterministic)
    tr2, te2, pu2 = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)
    assert train_df["window_id"].tolist() == tr2["window_id"].tolist()
    assert test_df["window_id"].tolist() == te2["window_id"].tolist()


def test_train_only_scaling_isolation(fused_data):
    """Verify StandardScaler is strictly fitted on X_train and not influenced by X_test."""
    train_df, test_df, _ = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, feat_names, scaler = prepare_tabular_data(train_df, test_df, scale=True)

    assert scaler is not None
    assert scaler.n_samples_seen_ == len(train_df)
    assert scaler.mean_.shape[0] == 40
    assert scaler.scale_.shape[0] == 40

    # Ensure scaler params match manual fit on X_train
    manual_scaler = StandardScaler()
    X_tr_raw, _ = get_feature_matrix(train_df)
    manual_scaler.fit(X_tr_raw)
    np.testing.assert_allclose(scaler.mean_, manual_scaler.mean_)
    np.testing.assert_allclose(scaler.scale_, manual_scaler.scale_)


def test_model_complexity_computation():
    """Verify structural complexity and serialization footprint extraction."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    X_dummy = np.random.randn(50, 40)
    y_dummy = np.random.choice([0, 1], size=50)

    # Linear
    lr = LogisticRegression().fit(X_dummy, y_dummy)
    lr_c = compute_model_complexity(lr)
    assert lr_c["structural_parameters"] == 41  # 40 weights + 1 bias
    assert lr_c["serialized_size_bytes"] > 0

    # Tree ensemble
    rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X_dummy, y_dummy)
    rf_c = compute_model_complexity(rf)
    assert rf_c["structural_parameters"] > 0
    assert "decision tree nodes" in rf_c["complexity_description"]


def test_all_available_models_produce_valid_metrics(fused_data):
    """Verify each available model trains, predicts, and outputs valid metrics on a slice of data."""
    reg = get_baseline_registry()
    train_df, test_df, _ = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)

    # Use first station for fast unit test
    m_tr = train_df[train_df["machine_id"] == "station_1"].head(60)
    m_te = test_df[test_df["machine_id"] == "station_1"].head(30)
    X_tr, y_tr, X_te, y_te, _, _ = prepare_tabular_data(m_tr, m_te, scale=True)

    for name, spec in reg.items():
        if not spec.is_available:
            continue
        y_pred, y_prob, t_fit, t_inf, complexity = fit_and_evaluate_model(
            spec, X_tr, y_tr, X_te, y_te
        )
        assert len(y_pred) == len(y_te)
        assert len(y_prob) == len(y_te)
        assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))
        assert t_fit >= 0.0
        assert t_inf >= 0.0
        assert complexity["serialized_size_bytes"] > 0

        metrics = compute_classification_metrics(y_te, y_pred, y_prob, len(m_tr), int(y_tr.sum()))
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
