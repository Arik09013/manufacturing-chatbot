"""
Unit tests for data leakage prevention and time-aware splitting.

Verifies:
  1. No window ID overlap between train and test sets.
  2. Strict chronological separation and >=30 min embargo gap per machine.
  3. Strict machine isolation in Leave-One-Machine-Out splits.
  4. Detection and rejection of simulated data leakage (DataLeakageError).
  5. Scaler fitting strictly on training feature matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
import pytest

from src.data.splits import (
    DataLeakageError,
    prepare_tabular_data,
    split_chronological,
    split_leave_one_machine_out,
    verify_leakage_free,
)
from src.fusion.fuse import load_fused


@pytest.fixture(scope="module")
def fused_data() -> pd.DataFrame:
    return load_fused()


def test_chronological_split_proportions_and_embargo(fused_data: pd.DataFrame):
    """Verify chronological split sizes, anomaly presence, and 30-min embargo."""
    train_df, test_df, purge_df = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)

    # 1,533 train (511 * 3), 375 test (125 * 3), 9 purged (3 * 3)
    assert len(train_df) == 1533
    assert len(test_df) == 375
    assert len(purge_df) == 9
    assert len(train_df) + len(test_df) + len(purge_df) == len(fused_data)

    # Anomalies exist in both splits
    assert train_df["is_anomaly"].sum() == 50
    assert test_df["is_anomaly"].sum() == 10
    assert purge_df["is_anomaly"].sum() == 0  # no anomalies discarded

    # Per machine checks
    for m in fused_data["machine_id"].unique():
        m_tr = train_df[train_df["machine_id"] == m]
        m_te = test_df[test_df["machine_id"] == m]

        assert len(m_tr) == 511
        assert len(m_te) == 125
        assert m_tr["is_anomaly"].sum() > 0
        assert m_te["is_anomaly"].sum() > 0

        # Exact minimum separation >= 30 min
        gap = m_te["window_start"].min() - m_tr["window_end"].max()
        assert gap >= pd.Timedelta(minutes=30)


def test_chronological_split_no_window_id_overlap(fused_data: pd.DataFrame):
    """Verify zero overlap of window IDs."""
    train_df, test_df, purge_df = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)

    tr_ids = set(train_df["window_id"])
    te_ids = set(test_df["window_id"])
    pu_ids = set(purge_df["window_id"])

    assert len(tr_ids & te_ids) == 0
    assert len(tr_ids & pu_ids) == 0
    assert len(te_ids & pu_ids) == 0


def test_lomo_strict_machine_isolation(fused_data: pd.DataFrame):
    """Verify that Leave-One-Machine-Out contains zero cross-machine leakage."""
    folds = list(split_leave_one_machine_out(fused_data))
    assert len(folds) == 3

    for train_df, test_df, held_out in folds:
        assert set(test_df["machine_id"].unique()) == {held_out}
        assert held_out not in train_df["machine_id"].unique()
        assert len(set(train_df["window_id"]) & set(test_df["window_id"])) == 0
        assert len(train_df) == 1278
        assert len(test_df) == 639
        assert test_df["is_anomaly"].sum() == 20
        assert train_df["is_anomaly"].sum() == 40


def test_leakage_detector_catches_id_overlap(fused_data: pd.DataFrame):
    """Simulate a leaky split with an overlapping window and assert DataLeakageError."""
    train_df = fused_data.iloc[:100].copy()
    test_df = fused_data.iloc[99:150].copy()  # index 99 is duplicated

    with pytest.raises(DataLeakageError, match="identical window IDs"):
        verify_leakage_free(train_df, test_df, mode="chronological")


def test_leakage_detector_catches_embargo_violation(fused_data: pd.DataFrame):
    """Simulate an overlapping temporal window without 30-min embargo and assert DataLeakageError."""
    m1 = fused_data[fused_data["machine_id"] == "station_1"].sort_values("window_start")
    # Window 0: 06:00-06:30, Window 1: 06:15-06:45 (shares 15 min)
    train_df = m1.iloc[[0]].copy()
    test_df = m1.iloc[[1]].copy()

    with pytest.raises(DataLeakageError, match="Data leakage on machine|Embargo violation"):
        verify_leakage_free(train_df, test_df, mode="chronological", embargo_minutes=30)


def test_leakage_detector_catches_lomo_machine_leak(fused_data: pd.DataFrame):
    """Simulate a LOMO split where the test machine appears in training."""
    train_df = fused_data.copy()  # contains station_1
    test_df = fused_data[fused_data["machine_id"] == "station_1"].copy()
    test_df["window_id"] = test_df["window_id"] + "_test_only"

    with pytest.raises(DataLeakageError, match="LOMO leakage"):
        verify_leakage_free(train_df, test_df, mode="lomo", held_out_machine="station_1")


def test_scaler_fitted_strictly_on_train(fused_data: pd.DataFrame):
    """Verify that StandardScaler parameters (mean_, var_) depend ONLY on training data."""
    train_df, test_df, _ = split_chronological(fused_data, train_ratio=0.8, embargo_minutes=30)

    # 1. Fit scaler on train only
    X_tr_1, _, _, _, _, scaler_1 = prepare_tabular_data(train_df, test_df, scale=True)

    # 2. Modify test_df with extreme values
    test_df_corrupted = test_df.copy()
    test_df_corrupted["welding_current_mean"] = 99999.0

    # 3. Fit scaler on train again
    X_tr_2, _, _, _, _, scaler_2 = prepare_tabular_data(train_df, test_df_corrupted, scale=True)

    # The fitted scaler statistics must be 100% identical regardless of test set values
    assert scaler_1 is not None and scaler_2 is not None
    assert (scaler_1.mean_ == scaler_2.mean_).all()
    assert (scaler_1.var_ == scaler_2.var_).all()
    assert (X_tr_1 == X_tr_2).all()
