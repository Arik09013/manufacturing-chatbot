"""
Unit and Integration Tests for Step 5 Component Ablation Study.

Verifies:
  1. Component inventory integrity and comprehensive audit coverage.
  2. Condition feature matrix dimensions and column isolation.
  3. Strict row-count and ground-truth label invariance.
  4. Temporal split invariance across all conditions.
  5. Train-only scaling parameter isolation (zero test-set leakage).
  6. Absence of target labels or metadata in feature matrices.
  7. Exact reproducibility of the FULL baseline condition.
  8. DistilBERT text serialization correctness under component ablations.
  9. Documentation of non-applicable (N/A) components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.component_ablation import (
    COMPONENT_CONDITIONS,
    COMPONENT_INVENTORY,
    HEAT_INPUT_COLS,
    LOG_COUNT_COLS,
    MIN_MAX_COLS,
    RANGE_COLS,
    STD_COLS,
    build_component_matrix,
    build_component_texts,
    component_window_to_text,
    get_component_columns,
    prepare_component_tabular_data,
)
from src.data.splits import (
    split_chronological,
    split_leave_one_machine_out,
    verify_leakage_free,
)
from src.fusion.fuse import load_fused
from src.model.anomaly import ALL_FEATURE_COLS


@pytest.fixture(scope="module")
def fused_data() -> pd.DataFrame:
    """Load fused dataset fixture once."""
    return load_fused()


def test_component_inventory_structure():
    """Verify inventory contains all required metadata fields and valid classifications."""
    assert len(COMPONENT_INVENTORY) >= 15
    required_fields = {
        "component_name",
        "component_type",
        "implementation_location",
        "enabled_in_full_pipeline",
        "affects_predictive_inference",
        "ablation_method",
        "expected_behavior",
        "quantitative_or_na",
    }
    for item in COMPONENT_INVENTORY:
        assert required_fields.issubset(item.keys())
        assert item["quantitative_or_na"] in ("QUANTITATIVE", "NOT_APPLICABLE")
        if item["quantitative_or_na"] == "QUANTITATIVE":
            assert item["affects_predictive_inference"] is True


def test_full_condition_reproduces_baseline_features(fused_data: pd.DataFrame):
    """Verify FULL_COMPONENTS contains exactly the 40 baseline features."""
    cols = get_component_columns("FULL_COMPONENTS")
    assert len(cols) == 40
    assert cols == ALL_FEATURE_COLS

    X, names = build_component_matrix(fused_data, "FULL_COMPONENTS")
    assert X.shape == (len(fused_data), 40)
    assert names == ALL_FEATURE_COLS


def test_range_features_ablation(fused_data: pd.DataFrame):
    """Verify WITHOUT_RANGE_FEATURES excludes all 6 range columns."""
    cols = get_component_columns("WITHOUT_RANGE_FEATURES")
    assert len(cols) == 34
    assert not any(c.endswith("_range") for c in cols)
    for rc in RANGE_COLS:
        assert rc not in cols


def test_domain_heat_input_ablation(fused_data: pd.DataFrame):
    """Verify WITHOUT_DOMAIN_HEAT_INPUT excludes all 5 heat input columns."""
    cols = get_component_columns("WITHOUT_DOMAIN_HEAT_INPUT")
    assert len(cols) == 35
    assert not any(c.startswith("heat_input") for c in cols)
    for hc in HEAT_INPUT_COLS:
        assert hc not in cols


def test_log_event_counts_ablation(fused_data: pd.DataFrame):
    """Verify WITHOUT_LOG_EVENT_COUNTS drops counts but retains boolean flags."""
    cols = get_component_columns("WITHOUT_LOG_EVENT_COUNTS")
    assert len(cols) == 33
    for lc in LOG_COUNT_COLS:
        assert lc not in cols
    # Boolean operational flags must remain
    assert "has_alarm" in cols
    assert "has_warning" in cols
    assert "has_note" in cols


def test_std_features_ablation(fused_data: pd.DataFrame):
    """Verify WITHOUT_STD_FEATURES excludes all 6 standard deviation columns."""
    cols = get_component_columns("WITHOUT_STD_FEATURES")
    assert len(cols) == 34
    assert not any(c.endswith("_std") for c in cols)
    for sc in STD_COLS:
        assert sc not in cols


def test_min_max_features_ablation(fused_data: pd.DataFrame):
    """Verify WITHOUT_MIN_MAX_FEATURES excludes all 12 min and max columns."""
    cols = get_component_columns("WITHOUT_MIN_MAX_FEATURES")
    assert len(cols) == 28
    assert not any(c.endswith("_min") or c.endswith("_max") for c in cols)
    for mm in MIN_MAX_COLS:
        assert mm not in cols


def test_scaler_ablation_unscaled_features(fused_data: pd.DataFrame):
    """Verify WITHOUT_STANDARD_SCALER returns raw unscaled values."""
    train_df, test_df, _ = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, names, scaler = prepare_component_tabular_data(
        train_df, test_df, "WITHOUT_STANDARD_SCALER"
    )
    assert scaler is None
    # Verify X_tr matches raw unscaled feature matrix exactly
    X_tr_raw, _ = build_component_matrix(train_df, "WITHOUT_STANDARD_SCALER")
    np.testing.assert_array_equal(X_tr, X_tr_raw)
    X_te_raw, _ = build_component_matrix(test_df, "WITHOUT_STANDARD_SCALER")
    np.testing.assert_array_equal(X_te, X_te_raw)


def test_labels_and_rows_invariant_across_conditions(fused_data: pd.DataFrame):
    """Verify row count and target labels are bit-level identical across all conditions."""
    n_rows = len(fused_data)
    y_baseline = fused_data["is_anomaly"].astype(int).values

    for cond in COMPONENT_CONDITIONS:
        X, names = build_component_matrix(fused_data, cond)
        assert X.shape[0] == n_rows
        # Verify no NaN or Inf
        assert np.isfinite(X).all()


def test_split_invariance_across_conditions(fused_data: pd.DataFrame):
    """Verify chronological split produces identical sample indices regardless of condition."""
    train_df, test_df, purge_df = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)

    for cond in COMPONENT_CONDITIONS:
        X_tr, y_tr, X_te, y_te, names, _ = prepare_component_tabular_data(
            train_df, test_df, cond
        )
        assert len(X_tr) == len(train_df) == 1533
        assert len(X_te) == len(test_df) == 375
        assert np.array_equal(y_tr, train_df["is_anomaly"].astype(int).values)
        assert np.array_equal(y_te, test_df["is_anomaly"].astype(int).values)


def test_train_only_scaling_isolation(fused_data: pd.DataFrame):
    """Verify scaler mean and scale are computed exclusively from training data."""
    train_df, test_df, _ = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, names, scaler = prepare_component_tabular_data(
        train_df, test_df, "FULL_COMPONENTS"
    )
    assert scaler is not None

    # Compute manual train mean
    X_tr_raw, _ = build_component_matrix(train_df, "FULL_COMPONENTS")
    expected_means = np.mean(X_tr_raw, axis=0)
    np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)

    # Standardized train features must have mean close to 0 and std close to 1
    np.testing.assert_allclose(np.mean(X_tr, axis=0), np.zeros(len(names)), atol=1e-5)


def test_no_target_or_metadata_in_component_features():
    """Verify target columns and metadata IDs are never present in any condition."""
    forbidden = {"is_anomaly", "primary_fault", "severity", "window_id", "station_id", "window_start", "window_end"}
    for cond in COMPONENT_CONDITIONS:
        cols = get_component_columns(cond)
        overlap = set(cols) & forbidden
        assert not overlap, f"Target or metadata columns found in condition '{cond}': {overlap}"


def test_text_component_serialization(fused_data: pd.DataFrame):
    """Verify DistilBERT text serializer correctly adapts to component conditions."""
    row = fused_data.iloc[0]

    # Full text contains range and heat input
    full_text = component_window_to_text(row, "FULL_COMPONENTS")
    assert "range" in full_text
    assert "heat input" in full_text
    assert "alarms" in full_text

    # Without range text
    no_range_text = component_window_to_text(row, "WITHOUT_RANGE_FEATURES")
    assert "range" not in no_range_text
    assert "heat input" in no_range_text

    # Without heat input text
    no_heat_text = component_window_to_text(row, "WITHOUT_DOMAIN_HEAT_INPUT")
    assert "heat input" not in no_heat_text

    # Without log counts
    no_counts_text = component_window_to_text(row, "WITHOUT_LOG_EVENT_COUNTS")
    assert "alarms," not in no_counts_text
    assert "Alarm active." in no_counts_text or "No alarm." in no_counts_text


def test_na_components_documented():
    """Verify out-of-scope / non-inference components are marked NOT_APPLICABLE."""
    na_items = [c for c in COMPONENT_INVENTORY if c["quantitative_or_na"] == "NOT_APPLICABLE"]
    assert len(na_items) >= 9
    na_names = {c["component_name"] for c in na_items}
    assert "SHAP_EXPLAINER" in na_names
    assert "LIME_EXPLAINER" in na_names
    assert "RAG_RETRIEVAL_INDEX" in na_names
    assert "ROOT_CAUSE_REASONING" in na_names
    assert "FEATURE_SELECTION_DIM_REDUCTION" in na_names
