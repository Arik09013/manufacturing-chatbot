"""
Sensor time-series preprocessing.

Pipeline per machine:
  1. Impute missing values (linear interpolation)
  2. Denoise with rolling mean (5-minute window)
  3. Normalise to zero-mean / unit-variance (StandardScaler, fit on normal windows)
  4. Segment into fixed 30-minute windows and extract statistical features

Outputs a DataFrame with one row per (machine, window) containing
aggregate features: mean, std, min, max, range for each sensor channel.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SENSOR_COLS = [
    "welding_current",
    "arc_voltage",
    "welding_speed",
    "wire_feed_rate",
    "shielding_gas_flow",
    "heat_input",
]
WINDOW_MINUTES = 30
STEP_MINUTES = 15        # 50 % overlap
DENOISE_WINDOW = 5       # rolling mean over 5 samples (= 5 minutes at 1-min freq)


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Linear interpolation for gaps, forward/back fill for edge values."""
    df = df.copy()
    df[SENSOR_COLS] = (
        df[SENSOR_COLS]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    return df


def denoise(df: pd.DataFrame, window: int = DENOISE_WINDOW) -> pd.DataFrame:
    """Apply rolling mean smoothing; preserves original length."""
    df = df.copy()
    df[SENSOR_COLS] = (
        df[SENSOR_COLS]
        .rolling(window=window, min_periods=1, center=True)
        .mean()
    )
    return df


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the provided (clean/normal) sensor data."""
    scaler = StandardScaler()
    scaler.fit(df[SENSOR_COLS].values)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    df[SENSOR_COLS] = scaler.transform(df[SENSOR_COLS].values)
    return df


def segment_windows(
    df: pd.DataFrame,
    window_minutes: int = WINDOW_MINUTES,
    step_minutes: int = STEP_MINUTES,
) -> pd.DataFrame:
    """
    Slice the time-series into fixed-width windows and compute per-window
    aggregate statistics.

    Returns DataFrame with columns:
      window_id, machine_id, window_start, window_end,
      <channel>_mean, <channel>_std, <channel>_min, <channel>_max, <channel>_range
      for each sensor channel.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    machine_id = df["machine_id"].iloc[0]

    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]
    window_td = pd.Timedelta(minutes=window_minutes)
    step_td = pd.Timedelta(minutes=step_minutes)

    rows = []
    w_start = start
    window_idx = 0
    while w_start + window_td <= end:
        w_end = w_start + window_td
        mask = (df["timestamp"] >= w_start) & (df["timestamp"] < w_end)
        chunk = df.loc[mask, SENSOR_COLS]

        if len(chunk) == 0:
            w_start += step_td
            continue

        row: dict = {
            "window_id":    f"{machine_id}_w{window_idx:05d}",
            "machine_id":   machine_id,
            "window_start": w_start,
            "window_end":   w_end,
        }
        for col in SENSOR_COLS:
            vals = chunk[col].values
            row[f"{col}_mean"]  = float(np.mean(vals))
            row[f"{col}_std"]   = float(np.std(vals))
            row[f"{col}_min"]   = float(np.min(vals))
            row[f"{col}_max"]   = float(np.max(vals))
            row[f"{col}_range"] = float(np.max(vals) - np.min(vals))

        rows.append(row)
        w_start += step_td
        window_idx += 1

    return pd.DataFrame(rows)


def preprocess_sensors(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler_on_data: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Full preprocessing pipeline for one machine's sensor data.

    Returns (windowed_features_df, fitted_scaler).
    Pass a pre-fitted scaler to reuse the same normalisation across
    train/inference splits.
    """
    df = impute(df)
    df = denoise(df)

    if scaler is None and fit_scaler_on_data:
        scaler = fit_scaler(df)

    if scaler is not None:
        df = apply_scaler(df, scaler)

    windowed = segment_windows(df)
    return windowed, scaler
