"""
Feature fusion: concatenate sensor-window features, log features,
and note embeddings into a single feature record per window.

Also annotates each window with its ground-truth label (is_anomaly,
anomaly_type, root_cause) using an interval join against ground_truth.csv.

Output: data/processed/fused.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_OUT = Path(__file__).parent.parent.parent / "data" / "processed"


def label_windows(
    fused: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add is_anomaly, anomaly_type, root_cause columns to each window.

    A window is labelled anomalous if any GT anomaly window overlaps
    with more than half of the sensor window's duration.
    """
    fused = fused.copy()
    fused["is_anomaly"]   = False
    fused["anomaly_type"] = ""
    fused["root_cause"]   = ""

    for _, gt_row in ground_truth.iterrows():
        m = gt_row["machine_id"]
        gt_start = gt_row["window_start"]
        gt_end   = gt_row["window_end"]

        mask = fused["machine_id"] == m
        # Correct overlap: max(win_start, gt_start) to min(win_end, gt_end)
        overlap_start = fused.loc[mask, "window_start"].clip(lower=gt_start)
        overlap_end   = fused.loc[mask, "window_end"].clip(upper=gt_end)
        overlap_dur   = (overlap_end - overlap_start).dt.total_seconds().clip(lower=0)
        win_dur       = (
            fused.loc[mask, "window_end"] - fused.loc[mask, "window_start"]
        ).dt.total_seconds()

        # Label windows where ≥50 % of the window is covered by anomaly
        anomaly_mask = mask & (overlap_dur / win_dur >= 0.5)
        fused.loc[anomaly_mask, "is_anomaly"]   = True
        fused.loc[anomaly_mask, "anomaly_type"] = str(gt_row["anomaly_type"])
        fused.loc[anomaly_mask, "root_cause"]   = str(gt_row["root_cause"])

    return fused


def fuse(
    sensor_windows: pd.DataFrame,
    log_features: pd.DataFrame,
    notes_embedded: Optional[pd.DataFrame],
    ground_truth: pd.DataFrame,
    include_embeddings: bool = True,
) -> pd.DataFrame:
    """
    Full fusion pipeline.

    Parameters
    ----------
    sensor_windows  : output of preprocess_sensors (one or all machines)
    log_features    : output of preprocess_logs (one or all machines)
    notes_embedded  : output of preprocess_notes (one or all machines), or None
    ground_truth    : from load_ground_truth()
    include_embeddings : whether to include note embedding dims in output

    Returns
    -------
    fused DataFrame with sensor + log + (optional) note features + labels
    """
    from src.fusion.align import align_logs_to_windows, align_notes_to_windows

    # Step 1: align logs
    df = align_logs_to_windows(sensor_windows, log_features)
    logger.debug("After log alignment: %s", df.shape)

    # Step 2: align notes (optional)
    if notes_embedded is not None and include_embeddings:
        df = align_notes_to_windows(df, notes_embedded)
        logger.debug("After note alignment: %s", df.shape)
    else:
        df["has_note"] = False

    # Step 3: label with ground truth
    df = label_windows(df, ground_truth)
    logger.debug("After labelling: %s anomalous windows", df["is_anomaly"].sum())

    return df


def save_fused(df: pd.DataFrame, out_dir: Path = _OUT) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fused.parquet"
    df.to_parquet(path, index=False)
    logger.info("Fused dataset saved -> %s  (%s rows)", path, len(df))
    return path


def load_fused(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or (_OUT / "fused.parquet")
    return pd.read_parquet(p)
