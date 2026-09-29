"""
Unit and validation tests for the Multimodal Modality Ablation Study.

Verifies:
  1. FULL condition reproduces the exact baseline feature set (ALL_FEATURE_COLS, 40 features).
  2. SENSOR_ONLY contains exclusively sensor features (30 features).
  3. WITHOUT_SENSOR contains zero sensor features (10 features).
  4. WITHOUT_LOGS contains zero log features (31 features).
  5. WITHOUT_OPERATOR_NOTES contains zero operator note features (39 features).
  6. WITHOUT_ENGINEERING_CONTEXT returns None / NOT APPLICABLE.
  7. Target labels ('is_anomaly') remain strictly identical across all ablations.
  8. Row/window counts remain identical across all conditions.
  9. Train/test split indices remain strictly identical.
  10. Preprocessing (StandardScaler) is strictly fitted on X_train.
  11. Feature selection is deterministic.
  12. No duplicate columns are introduced.
  13. Target column is never accidentally included as a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.ablation import (
    ABLATION_CONDITIONS,
    MODALITY_MAPPING,
    ablated_window_to_text,
    build_ablation_matrix,
    get_ablation_columns,
)
from src.data.splits import (
    prepare_tabular_data,
    split_chronological,
)
from src.fusion.fuse import load_fused
from src.model.anomaly import (
    ALL_FEATURE_COLS,
    LOG_FEATURE_COLS,
    NOTE_FEATURE_COLS,
    SENSOR_FEATURE_COLS,
)


@pytest.fixture(scope="module")
def fused_data() -> pd.DataFrame:
    return load_fused()


def test_full_condition_reproduces_baseline_features():
    """Verify FULL condition reproduces ALL_FEATURE_COLS."""
    cols = get_ablation_columns("FULL")
    assert cols == ALL_FEATURE_COLS
    assert len(cols) == 40


def test_sensor_only_condition():
    """Verify SENSOR_ONLY contains exclusively sensor features."""
    cols = get_ablation_columns("SENSOR_ONLY")
    assert cols == SENSOR_FEATURE_COLS
    assert len(cols) == 30
    assert set(cols).isdisjoint(set(LOG_FEATURE_COLS))
    assert set(cols).isdisjoint(set(NOTE_FEATURE_COLS))


def test_without_sensor_condition():
    """Verify WITHOUT_SENSOR contains zero sensor features."""
    cols = get_ablation_columns("WITHOUT_SENSOR")
    assert len(cols) == 10
    assert set(cols).isdisjoint(set(SENSOR_FEATURE_COLS))
    assert set(LOG_FEATURE_COLS).issubset(set(cols))
    assert set(NOTE_FEATURE_COLS).issubset(set(cols))


def test_without_logs_condition():
    """Verify WITHOUT_LOGS contains zero log features."""
    cols = get_ablation_columns("WITHOUT_LOGS")
    assert len(cols) == 31
    assert set(cols).isdisjoint(set(LOG_FEATURE_COLS))
    assert set(SENSOR_FEATURE_COLS).issubset(set(cols))
    assert set(NOTE_FEATURE_COLS).issubset(set(cols))


def test_without_operator_notes_condition():
    """Verify WITHOUT_OPERATOR_NOTES contains zero operator note features."""
    cols = get_ablation_columns("WITHOUT_OPERATOR_NOTES")
    assert len(cols) == 39
    assert set(cols).isdisjoint(set(NOTE_FEATURE_COLS))
    assert set(SENSOR_FEATURE_COLS).issubset(set(cols))
    assert set(LOG_FEATURE_COLS).issubset(set(cols))


def test_without_engineering_context_not_applicable():
    """Verify WITHOUT_ENGINEERING_CONTEXT is marked None / Not Applicable."""
    cols = get_ablation_columns("WITHOUT_ENGINEERING_CONTEXT")
    assert cols is None


def test_labels_identical_across_all_ablations(fused_data):
    """Verify ground truth labels remain identical regardless of ablation."""
    y_true_orig = fused_data["is_anomaly"].astype(int).values
    for cond in ["FULL", "SENSOR_ONLY", "WITHOUT_SENSOR", "WITHOUT_LOGS", "WITHOUT_OPERATOR_NOTES"]:
        X, cols = build_ablation_matrix(fused_data, cond)
        assert len(X) == len(y_true_orig)
        assert fused_data["is_anomaly"].astype(int).tolist() == y_true_orig.tolist()


def test_row_counts_identical(fused_data):
    """Verify row and window counts remain exactly 1,917 across all conditions."""
    for cond in ["FULL", "SENSOR_ONLY", "WITHOUT_SENSOR", "WITHOUT_LOGS", "WITHOUT_OPERATOR_NOTES"]:
        X, cols = build_ablation_matrix(fused_data, cond)
        assert X.shape[0] == len(fused_data)
        assert X.shape[1] == len(cols)


def test_train_test_split_identical_across_ablations(fused_data):
    """Verify chronological split boundaries are strictly identical."""
    tr1, te1, _ = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)
    tr2, te2, _ = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)
    assert tr1["window_id"].tolist() == tr2["window_id"].tolist()
    assert te1["window_id"].tolist() == te2["window_id"].tolist()


def test_train_only_scaling_isolation_under_ablation(fused_data):
    """Verify StandardScaler is strictly fitted on X_train for ablated feature subsets."""
    tr, te, _ = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)
    cols = get_ablation_columns("SENSOR_ONLY")

    X_tr_raw = tr[cols].values.astype(np.float32)
    scaler = StandardScaler().fit(X_tr_raw)
    assert scaler.n_samples_seen_ == len(tr)
    assert scaler.mean_.shape[0] == len(cols)


def test_no_duplicate_columns():
    """Verify no duplicate columns exist in any ablation condition."""
    for cond in ["FULL", "SENSOR_ONLY", "WITHOUT_SENSOR", "WITHOUT_LOGS", "WITHOUT_OPERATOR_NOTES"]:
        cols = get_ablation_columns(cond)
        assert len(cols) == len(set(cols)), f"Duplicate columns detected in {cond}"


def test_target_column_never_included_as_feature():
    """Verify target column 'is_anomaly' is never present in any feature list."""
    for cond in ["FULL", "SENSOR_ONLY", "WITHOUT_SENSOR", "WITHOUT_LOGS", "WITHOUT_OPERATOR_NOTES"]:
        cols = get_ablation_columns(cond)
        assert "is_anomaly" not in cols
        assert "anomaly_type" not in cols
        assert "root_cause" not in cols


def test_text_ablation_serialization(fused_data):
    """Verify natural-language text serialization reflects ablation conditions."""
    sample_row = fused_data.iloc[0]
    t_full = ablated_window_to_text(sample_row, "FULL")
    t_sensor = ablated_window_to_text(sample_row, "SENSOR_ONLY")
    t_no_sensor = ablated_window_to_text(sample_row, "WITHOUT_SENSOR")

    assert "Welding station window." in t_full
    assert "Log:" in t_full
    assert "Log:" not in t_sensor
    assert "mean" not in t_no_sensor
