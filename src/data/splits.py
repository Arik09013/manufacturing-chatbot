"""
Leakage-safe, time-aware dataset splitting and verification utilities.

Provides:
  1. Machine-wise chronological holdout with temporal embargo/purge gap.
  2. Leave-One-Machine-Out (LOMO) cross-validation splits.
  3. Strict leakage verification (window ID overlap, temporal gap, machine isolation).
  4. Train-only feature scaling and matrix preparation.
"""

from __future__ import annotations

import logging
from typing import Generator, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataLeakageError(ValueError):
    """Raised when a train/test split violates data leakage constraints."""
    pass


def verify_leakage_free(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str = "chronological",
    embargo_minutes: int = 30,
    held_out_machine: Optional[str] = None,
) -> dict:
    """
    Verify that a train/test split has zero data leakage.

    Checks:
      1. Disjoint window IDs (no exact window appears in both train and test).
      2. In chronological mode:
         - For every machine, all test windows occur strictly AFTER training windows.
         - The temporal gap between the end of training data and beginning of test data
           is at least `embargo_minutes`.
      3. In LOMO mode:
         - The set of machine IDs in train and test are strictly disjoint.
         - If `held_out_machine` is specified, test contains only that machine.
      4. Label sanity: both train and test contain at least one normal and one anomaly
         sample (required for valid ROC-AUC / PR-AUC computation).

    Raises
    ------
    DataLeakageError if any check fails.
    """
    report: dict = {
        "mode": mode,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "checks_passed": False,
        "machine_reports": {},
    }

    # 1. Window ID overlap check
    train_ids = set(train_df["window_id"].unique())
    test_ids = set(test_df["window_id"].unique())
    overlap_ids = train_ids & test_ids
    if overlap_ids:
        msg = f"Data leakage detected: {len(overlap_ids)} identical window IDs appear in both train and test sets!"
        logger.error(msg)
        raise DataLeakageError(msg)

    # 2. Mode-specific checks
    if mode == "chronological":
        machines = sorted(set(train_df["machine_id"].unique()) & set(test_df["machine_id"].unique()))
        embargo_td = pd.Timedelta(minutes=embargo_minutes)

        for m in machines:
            m_tr = train_df[train_df["machine_id"] == m]
            m_te = test_df[test_df["machine_id"] == m]

            if m_tr.empty or m_te.empty:
                continue

            last_train_start = m_tr["window_start"].max()
            last_train_end = m_tr["window_end"].max()
            first_test_start = m_te["window_start"].min()
            first_test_end = m_te["window_end"].min()

            # Check strict temporal ordering
            if first_test_start <= last_train_start:
                msg = (
                    f"Data leakage on machine {m}: test window starts at {first_test_start} "
                    f"which is <= last train window start at {last_train_start}."
                )
                logger.error(msg)
                raise DataLeakageError(msg)

            # Check embargo duration
            actual_gap = first_test_start - last_train_end
            if actual_gap < embargo_td:
                msg = (
                    f"Embargo violation on machine {m}: gap between last train end ({last_train_end}) "
                    f"and first test start ({first_test_start}) is {actual_gap}, "
                    f"which is less than required {embargo_minutes} minutes ({embargo_td})."
                )
                logger.error(msg)
                raise DataLeakageError(msg)

            report["machine_reports"][m] = {
                "train_windows": len(m_tr),
                "test_windows": len(m_te),
                "train_anomalies": int(m_tr["is_anomaly"].sum()),
                "test_anomalies": int(m_te["is_anomaly"].sum()),
                "last_train_end": str(last_train_end),
                "first_test_start": str(first_test_start),
                "actual_embargo_gap_minutes": actual_gap.total_seconds() / 60.0,
                "embargo_passed": True,
            }

    elif mode == "lomo":
        train_machines = set(train_df["machine_id"].unique())
        test_machines = set(test_df["machine_id"].unique())
        shared_machines = train_machines & test_machines

        if shared_machines:
            msg = f"LOMO leakage: machine(s) {shared_machines} appear in both train and test sets!"
            logger.error(msg)
            raise DataLeakageError(msg)

        if held_out_machine and test_machines != {held_out_machine}:
            msg = f"LOMO error: expected held-out machine '{held_out_machine}', but test contains {test_machines}."
            logger.error(msg)
            raise DataLeakageError(msg)

        report["train_machines"] = sorted(train_machines)
        report["test_machines"] = sorted(test_machines)

    # 3. Label representation check
    if train_df["is_anomaly"].nunique() < 2:
        logger.warning("Train set contains only one class!")
    if test_df["is_anomaly"].nunique() < 2:
        logger.warning("Test set contains only one class; ROC-AUC will be undefined.")

    report["checks_passed"] = True
    return report


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    embargo_minutes: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split windows into train, purged, and test sets per machine chronologically.

    For each machine:
      1. Sort windows strictly by `window_start`.
      2. Allocate the first ~80% of windows to training.
      3. Discard / purge all windows whose time span overlaps with the end of training
         data OR falls within the `embargo_minutes` buffer after the last training window.
      4. Allocate the remaining windows to testing.
      5. Verify zero temporal overlap and minimum timestamp separation.

    Returns
    -------
    (train_df, test_df, purge_df)
    """
    train_pieces = []
    test_pieces = []
    purge_pieces = []

    machines = sorted(df["machine_id"].unique())
    embargo_td = pd.Timedelta(minutes=embargo_minutes)

    for m in machines:
        m_df = df[df["machine_id"] == m].sort_values("window_start").reset_index(drop=True)
        total_m = len(m_df)
        n_train = int(total_m * train_ratio)

        # Baseline chronological cut
        candidate_train = m_df.iloc[:n_train].copy()

        # Check anomaly presence: if naive cut has 0 anomalies in test, step back
        anom_in_test_pool = m_df.iloc[n_train:]["is_anomaly"].sum()
        if anom_in_test_pool == 0 and m_df["is_anomaly"].sum() > 1:
            # Step earlier until test has at least 1 anomaly
            logger.warning(
                "Machine %s naive 80/20 cut leaves test with 0 anomalies. Searching earlier chronological boundary.",
                m,
            )
            anom_indices = m_df[m_df["is_anomaly"] == True].index.tolist()
            # Pick a cut before the last anomaly so at least 1 anomaly is in test
            last_anom_idx = anom_indices[-1]
            n_train = min(n_train, last_anom_idx)
            candidate_train = m_df.iloc[:n_train].copy()

        t_train_last_start = candidate_train["window_start"].iloc[-1]
        t_train_end = candidate_train["window_end"].iloc[-1]
        t_test_min_start = t_train_end + embargo_td

        # Purged windows: windows that overlap with train or fall in the embargo buffer
        mask_purge = (m_df["window_start"] > t_train_last_start) & (m_df["window_start"] < t_test_min_start)
        mask_test = m_df["window_start"] >= t_test_min_start

        m_purge = m_df[mask_purge].copy()
        m_test = m_df[mask_test].copy()

        train_pieces.append(candidate_train)
        purge_pieces.append(m_purge)
        test_pieces.append(m_test)

        logger.info(
            "Machine %s chronological split: Train=%d (anom=%d), Purged=%d (anom=%d), Test=%d (anom=%d). Embargo gap: %s",
            m,
            len(candidate_train),
            candidate_train["is_anomaly"].sum(),
            len(m_purge),
            m_purge["is_anomaly"].sum(),
            len(m_test),
            m_test["is_anomaly"].sum(),
            m_test["window_start"].iloc[0] - t_train_end if len(m_test) else "N/A",
        )

    train_df = pd.concat(train_pieces, ignore_index=True)
    purge_df = pd.concat(purge_pieces, ignore_index=True)
    test_df = pd.concat(test_pieces, ignore_index=True)

    # Run strict verification
    verify_leakage_free(train_df, test_df, mode="chronological", embargo_minutes=embargo_minutes)

    return train_df, test_df, purge_df


def split_leave_one_machine_out(
    df: pd.DataFrame,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame, str], None, None]:
    """
    Generate Leave-One-Machine-Out splits.

    Yields
    ------
    (train_df, test_df, held_out_machine_id)
    """
    machines = sorted(df["machine_id"].unique())
    for held_out in machines:
        train_df = df[df["machine_id"] != held_out].copy().reset_index(drop=True)
        test_df = df[df["machine_id"] == held_out].copy().reset_index(drop=True)

        verify_leakage_free(
            train_df,
            test_df,
            mode="lomo",
            held_out_machine=held_out,
        )
        yield train_df, test_df, held_out


def prepare_tabular_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], Optional[StandardScaler]]:
    """
    Extract tabular features and labels from train and test sets.

    CRITICAL LEAKAGE FIX:
    StandardScaler is fitted ONLY on X_train.
    X_test is transformed using the fitted scaler parameters (mean_, var_).

    Returns
    -------
    (X_train, y_train, X_test, y_test, feature_names, fitted_scaler)
    """
    from src.model.anomaly import get_feature_matrix, get_labels

    X_train_raw, feat_names = get_feature_matrix(train_df)
    y_train = get_labels(train_df)

    X_test_raw, _ = get_feature_matrix(test_df)
    y_test = get_labels(test_df)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        X_train = X_train_raw
        X_test = X_test_raw

    return X_train, y_train, X_test, y_test, feat_names, scaler


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    n_train: int = 0,
    n_anom_train: int = 0,
) -> dict:
    """Compute comprehensive classification and ranking metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    has_both_classes = len(np.unique(y_true)) > 1

    roc_auc = None
    pr_auc = None
    if y_prob is not None and has_both_classes:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = None
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = None

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": cm,
        "n_train": int(n_train),
        "n_test": int(len(y_true)),
        "n_anomalies_train": int(n_anom_train),
        "n_anomalies_test": int(np.sum(y_true)),
    }
