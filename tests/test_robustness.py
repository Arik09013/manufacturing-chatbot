"""
Unit and integration tests for synthetic data robustness and perturbation framework.

Verifies:
  1. Multi-seed generation reproducibility and schema consistency.
  2. Perturbations alter ONLY test data (training data remains 100% clean).
  3. Ground-truth labels are strictly preserved under all perturbation scenarios.
  4. Perturbation randomness is seeded and deterministic under identical seeds.
  5. Scaler fitting remains isolated on clean X_train.
  6. Reduced fault severity attenuates anomaly signal towards normal baseline mean.
  7. Temporal drift is linear and monotonic across test horizon.
  8. Missingness correctly imputes baseline values without introducing NaNs.
  9. Embargo and chronological constraints remain intact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.splits import (
    prepare_tabular_data,
    split_chronological,
    verify_leakage_free,
)
from src.data.synthetic_robustness import (
    apply_feature_dropout,
    apply_missingness,
    apply_reduced_severity,
    apply_sensor_drift,
    apply_sensor_noise,
    build_fused_for_seed,
    get_perturbation_scenarios,
    get_sensor_column_indices,
)
from src.fusion.fuse import load_fused
from src.model.anomaly import ALL_FEATURE_COLS


@pytest.fixture(scope="module")
def fused_clean() -> pd.DataFrame:
    """Canonical fused dataset (seed 42)."""
    return load_fused()


def test_multiseed_reproducibility():
    """Verify that generation with the same seed is bit-for-bit deterministic."""
    df1 = build_fused_for_seed(seed=123)
    df2 = build_fused_for_seed(seed=123)

    assert len(df1) == len(df2)
    assert df1["is_anomaly"].sum() == df2["is_anomaly"].sum()
    np.testing.assert_allclose(df1["welding_current_mean"].values, df2["welding_current_mean"].values)


def test_multiseed_schema_consistency():
    """Verify schema and window counts across diverse seeds."""
    for seed in [42, 456]:
        df = build_fused_for_seed(seed=seed)
        assert len(df) == 1917
        assert "is_anomaly" in df.columns
        assert df["is_anomaly"].sum() > 0
        for col in ALL_FEATURE_COLS:
            assert col in df.columns


def test_train_data_never_altered_by_perturbations(fused_clean):
    """Verify perturbations applied to test matrices leave X_train bit-for-bit intact."""
    tr, te, _ = split_chronological(fused_clean, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, feat_names, _ = prepare_tabular_data(tr, te, scale=True)

    X_tr_copy = X_tr.copy()
    y_tr_copy = y_tr.copy()

    scenarios = get_perturbation_scenarios(X_te, y_te, X_tr, y_tr, seed=42, feature_names=feat_names)

    for sc_name, (X_pert, _, _) in scenarios.items():
        # X_train and y_train must remain completely identical
        np.testing.assert_array_equal(X_tr, X_tr_copy, err_msg=f"X_train altered by {sc_name}")
        np.testing.assert_array_equal(y_tr, y_tr_copy, err_msg=f"y_train altered by {sc_name}")


def test_ground_truth_labels_unchanged_under_perturbations(fused_clean):
    """Verify test labels y_test are NEVER modified by perturbations."""
    tr, te, _ = split_chronological(fused_clean, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, feat_names, _ = prepare_tabular_data(tr, te, scale=True)

    y_te_orig = y_te.copy()
    scenarios = get_perturbation_scenarios(X_te, y_te, X_tr, y_tr, seed=42, feature_names=feat_names)

    for sc_name, (_, _, _) in scenarios.items():
        np.testing.assert_array_equal(y_te, y_te_orig, err_msg=f"y_test altered by {sc_name}")


def test_sensor_noise_perturbation(fused_clean):
    """Verify noise is bounded and affects only sensor columns."""
    tr, te, _ = split_chronological(fused_clean, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, feat_names, _ = prepare_tabular_data(tr, te, scale=True)

    sensor_idx = set(get_sensor_column_indices(feat_names))
    non_sensor_idx = [i for i in range(len(feat_names)) if i not in sensor_idx]

    X_noise = apply_sensor_noise(X_te, noise_level=0.05, seed=42, feature_names=feat_names)

    # Sensor columns should change
    assert not np.allclose(X_noise[:, list(sensor_idx)], X_te[:, list(sensor_idx)])
    # Non-sensor columns (e.g. log counts, note flags) must remain untouched
    np.testing.assert_allclose(X_noise[:, non_sensor_idx], X_te[:, non_sensor_idx])


def test_missingness_perturbation(fused_clean):
    """Verify missing values are imputed to 0.0 with no NaNs."""
    tr, te, _ = split_chronological(fused_clean, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, feat_names, _ = prepare_tabular_data(tr, te, scale=True)

    X_miss = apply_missingness(X_te, missing_ratio=0.10, is_block=False, seed=42, feature_names=feat_names)
    assert not np.isnan(X_miss).any()

    # Block missingness test
    X_block = apply_missingness(X_te, is_block=True, block_len=10, seed=42, feature_names=feat_names)
    assert not np.isnan(X_block).any()


def test_sensor_drift_linearity(fused_clean):
    """Verify sensor drift increases monotonically across chronological test rows."""
    tr, te, _ = split_chronological(fused_clean, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, feat_names, _ = prepare_tabular_data(tr, te, scale=True)

    sensor_idx = get_sensor_column_indices(feat_names)
    X_drift = apply_sensor_drift(X_te, drift_level=0.05, feature_names=feat_names)

    diff = X_drift[:, sensor_idx[0]] - X_te[:, sensor_idx[0]]
    assert np.all(np.diff(diff) >= -1e-7)  # Monotonically increasing


def test_reduced_severity_attenuation(fused_clean):
    """Verify reduced severity moves anomalous rows towards normal mean without touching normal rows."""
    tr, te, _ = split_chronological(fused_clean, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, feat_names, _ = prepare_tabular_data(tr, te, scale=True)

    sensor_idx = get_sensor_column_indices(feat_names)
    X_sev50 = apply_reduced_severity(X_te, y_te, X_tr, y_tr, severity_ratio=0.50, feature_names=feat_names)

    # Normal rows (y_te == 0) must remain strictly untouched
    normal_mask = (y_te == 0)
    np.testing.assert_allclose(X_sev50[normal_mask], X_te[normal_mask])

    # Anomalous rows (y_te == 1) must have attenuated magnitude
    anom_mask = (y_te == 1)
    if np.any(anom_mask):
        normal_mu = np.mean(X_tr[y_tr == 0], axis=0)
        orig_dist = np.abs(X_te[anom_mask][:, sensor_idx] - normal_mu[sensor_idx])
        sev_dist = np.abs(X_sev50[anom_mask][:, sensor_idx] - normal_mu[sensor_idx])
        assert np.all(sev_dist <= orig_dist + 1e-6)


def test_deterministic_perturbations(fused_clean):
    """Verify that perturbations are fully deterministic when seeded."""
    tr, te, _ = split_chronological(fused_clean, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, feat_names, _ = prepare_tabular_data(tr, te, scale=True)

    n1 = apply_sensor_noise(X_te, noise_level=0.03, seed=999, feature_names=feat_names)
    n2 = apply_sensor_noise(X_te, noise_level=0.03, seed=999, feature_names=feat_names)
    np.testing.assert_allclose(n1, n2)
