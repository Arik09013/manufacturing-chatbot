"""
Production-log preprocessing.

Produces a per-window feature record by counting event types and flagging
the presence of alarms/warnings in each time window.

Outputs a DataFrame with one row per (machine, window) with columns:
  window_id, machine_id, window_start, window_end,
  n_events, n_alarms, n_warnings, n_maintenance, n_production, n_diagnostic,
  has_alarm, has_warning,
  unique_event_codes  (comma-separated, for downstream rule lookup)
"""

from __future__ import annotations

import pandas as pd

WINDOW_MINUTES = 30
STEP_MINUTES = 15

_EVENT_TYPE_COLS = ["alarm", "warning", "maintenance", "production", "diagnostic", "operational"]


def clean_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup exact duplicate event rows, ensure timestamp is datetime."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["timestamp", "machine_id", "event_code"])
    return df.sort_values("timestamp").reset_index(drop=True)


def aggregate_windows(
    df: pd.DataFrame,
    window_minutes: int = WINDOW_MINUTES,
    step_minutes: int = STEP_MINUTES,
) -> pd.DataFrame:
    """
    For each sliding window, count events by type and collect event codes.
    Only rows belonging to the given machine_id are processed.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    machine_id = str(df["machine_id"].iloc[0])

    start = df["timestamp"].min().floor("min")
    end = df["timestamp"].max().ceil("min")
    window_td = pd.Timedelta(minutes=window_minutes)
    step_td = pd.Timedelta(minutes=step_minutes)

    rows = []
    w_start = start
    window_idx = 0
    while w_start + window_td <= end + step_td:
        w_end = w_start + window_td
        mask = (df["timestamp"] >= w_start) & (df["timestamp"] < w_end)
        chunk = df[mask]

        row: dict = {
            "machine_id":         machine_id,
            "window_start":       w_start,
            "window_end":         w_end,
            "n_events":           len(chunk),
        }

        for etype in _EVENT_TYPE_COLS:
            row[f"n_{etype}"] = int((chunk["event_type"] == etype).sum())

        row["has_alarm"]   = bool((chunk["event_type"] == "alarm").any())
        row["has_warning"] = bool((chunk["event_type"] == "warning").any())
        row["unique_event_codes"] = ",".join(
            sorted(chunk["event_code"].astype(str).unique())
        ) if len(chunk) > 0 else ""

        rows.append(row)
        w_start += step_td
        window_idx += 1

    return pd.DataFrame(rows)


def preprocess_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Full log preprocessing for one machine."""
    df = clean_logs(df)
    return aggregate_windows(df)
