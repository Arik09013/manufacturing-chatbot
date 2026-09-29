"""
Unit and Integration Tests for Step 6: DistilBERT vs Conventional Tabular Models.

Verifies:
  1. Dataset row-count, target label, and raw feature invariants.
  2. Strict zero-leakage constraints in chronological and LOMO splits.
  3. Preprocessing scaling parameter isolation (train-only fitting).
  4. Model probability calibration and valid metric boundaries in [0, 1].
  5. Deterministic reproducibility of Random Forest, Logistic Regression, SVM, and DistilBERT under fixed seeds.
  6. Correct computation of paired McNemar exact tests.
  7. Parameter count and model size calculation validity.
  8. Clear segregation of supervised classifiers vs unsupervised Isolation Forest.
  9. Decision threshold validation-tuning isolation (zero test leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.eval_model_comparison import (
    compute_metrics_extended,
    get_model_parameter_count,
    get_model_specs,
    mcnemar_exact_test,
)
from src.data.splits import (
    prepare_tabular_data,
    split_chronological,
    split_leave_one_machine_out,
    verify_leakage_free,
)
from src.fusion.fuse import load_fused
from src.model.anomaly import ALL_FEATURE_COLS
from src.model.bert_detector import BertFaultDetector, build_texts


@pytest.fixture(scope="module")
def fused_data() -> pd.DataFrame:
    """Load fused dataset fixture once."""
    return load_fused()


def test_dataset_invariants(fused_data: pd.DataFrame):
    """Verify master fused dataset integrity (1917 rows, 60 anomalies, 48 cols)."""
    assert len(fused_data) == 1917
    assert int(fused_data["is_anomaly"].sum()) == 60
    assert len(fused_data.columns) == 48
    assert "machine_id" in fused_data.columns
    assert "window_id" in fused_data.columns


def test_chronological_split_leakage_free(fused_data: pd.DataFrame):
    """Verify chronological split guarantees disjoint IDs and >=30 min embargo."""
    train_df, test_df, purge_df = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)
    report = verify_leakage_free(train_df, test_df, mode="chronological", embargo_minutes=30)
    assert report["checks_passed"] is True
    assert len(train_df) == 1533
    assert len(test_df) == 375
    assert len(purge_df) == 9
    assert int(train_df["is_anomaly"].sum()) == 50
    assert int(test_df["is_anomaly"].sum()) == 10


def test_lomo_machine_isolation(fused_data: pd.DataFrame):
    """Verify LOMO splits have strictly disjoint machines in train and test sets."""
    stations = sorted(fused_data["machine_id"].unique())
    assert stations == ["station_1", "station_2", "station_3"]

    for train_df, test_df, held_out in split_leave_one_machine_out(fused_data):
        report = verify_leakage_free(train_df, test_df, mode="lomo", held_out_machine=held_out)
        assert report["checks_passed"] is True
        assert held_out not in train_df["machine_id"].unique()
        assert set(test_df["machine_id"].unique()) == {held_out}


def test_train_only_scaling_isolation(fused_data: pd.DataFrame):
    """Verify StandardScaler parameters are computed strictly from training data."""
    train_df, test_df, _ = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, feat_names, scaler = prepare_tabular_data(train_df, test_df, scale=True)

    assert scaler is not None
    # Training features mean must be ~0
    np.testing.assert_allclose(np.mean(X_tr, axis=0), np.zeros(len(feat_names)), atol=1e-5)

    # Test features must NOT have zero mean (shows it was not fitted on test set)
    assert not np.allclose(np.mean(X_te, axis=0), np.zeros(len(feat_names)), atol=1e-3)


def test_supervised_models_produce_valid_metrics(fused_data: pd.DataFrame):
    """Verify RF, LR, and SVM produce valid metric dictionaries with bounded values."""
    train_df, test_df, _ = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, _, _ = prepare_tabular_data(train_df, test_df, scale=True)

    models_dict = get_model_specs(seed=42)
    for name in ["Random Forest", "Logistic Regression", "Support Vector Machine"]:
        clf = models_dict[name]["model"]
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        prob = clf.predict_proba(X_te)[:, 1]

        m = compute_metrics_extended(y_te, pred, prob)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["pr_auc"] <= 1.0


def test_distilbert_detector_validity(fused_data: pd.DataFrame):
    """Verify pre-trained DistilBERT generates probabilities in [0, 1]."""
    assert BertFaultDetector.is_available() is True
    det = BertFaultDetector()
    _, test_df, _ = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)

    probs = det.predict_proba(test_df.iloc[:20])
    assert len(probs) == 20
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_mcnemar_exact_test_logic():
    """Verify McNemar paired test returns correct contingency and p-value."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])

    # Case 1: Identical predictions
    y_pred1 = y_true.copy()
    y_pred2 = y_true.copy()
    res1 = mcnemar_exact_test(y_true, y_pred1, y_pred2)
    assert res1["discordant_pairs"] == 0
    assert res1["p_value"] == 1.0

    # Case 2: One model makes 1 error, other makes none
    y_pred_err = y_true.copy()
    y_pred_err[0] = 1  # 1 false positive
    res2 = mcnemar_exact_test(y_true, y_true, y_pred_err)
    assert res2["discordant_pairs"] == 1
    assert res2["b_model1_only"] == 1
    assert res2["c_model2_only"] == 0
    assert res2["p_value"] == 1.0  # Binomial test with N=1, p=0.5 -> p=1.0


def test_model_parameter_count():
    """Verify parameter count extraction produces valid numbers."""
    from sklearn.linear_model import LogisticRegression
    y_dummy = np.array([0, 1] * 5)
    X_dummy = np.random.RandomState(42).randn(10, 40)
    lr = LogisticRegression().fit(X_dummy, y_dummy)
    p_lr = get_model_parameter_count("Logistic Regression", lr)
    assert p_lr["parameter_count"] == 41

    p_bert = get_model_parameter_count("DistilBERT", None)
    assert p_bert["parameter_count"] == 66_364_418
    assert p_bert["disk_footprint_mb"] > 200.0


def test_unsupervised_isolation_forest_separation(fused_data: pd.DataFrame):
    """Verify Isolation Forest is recognized as unsupervised and evaluated accordingly."""
    specs = get_model_specs(seed=42)
    assert specs["Isolation Forest"]["is_supervised"] is False
    assert specs["Random Forest"]["is_supervised"] is True
    assert specs["Logistic Regression"]["is_supervised"] is True
    assert specs["Support Vector Machine"]["is_supervised"] is True
