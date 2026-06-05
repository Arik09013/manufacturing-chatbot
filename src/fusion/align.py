"""
Temporal alignment: map every log-feature row and note embedding row
onto the nearest sensor-window using merge_asof (backward match).

Each sensor window is the "anchor." Log and note records are aligned
to the window whose window_start is closest but <= the event timestamp.
"""

from __future__ import annotations

import pandas as pd


def align_logs_to_windows(
    sensor_windows: pd.DataFrame,
    log_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join log feature rows onto sensor windows by machine + timestamp.

    sensor_windows must have: machine_id, window_start (sorted asc)
    log_features   must have: machine_id, window_start

    Returns sensor_windows with log columns merged in.
    Missing log rows (no events in that window) are filled with 0 / False.
    """
    machines = sensor_windows["machine_id"].unique()
    pieces = []

    for m in machines:
        sw = sensor_windows[sensor_windows["machine_id"] == m].copy()
        lf = log_features[log_features["machine_id"] == m].copy()

        if lf.empty:
            # Fill log columns with defaults
            for col in _log_fill_defaults():
                sw[col] = _log_fill_defaults()[col]
            pieces.append(sw)
            continue

        sw = sw.sort_values("window_start").reset_index(drop=True)
        lf = lf.sort_values("window_start").reset_index(drop=True)

        # Merge on exact window_start match (both were built with same params)
        # Exclude columns already present in sensor_windows to avoid _x/_y suffix conflicts
        existing = set(sw.columns)
        log_cols = [
            c for c in lf.columns
            if c not in ("machine_id", "window_start") and c not in existing
        ]
        log_cols = ["window_start"] + log_cols
        merged = sw.merge(
            lf[log_cols],
            on="window_start",
            how="left",
        )

        _fill_log_defaults(merged)
        pieces.append(merged)

    return pd.concat(pieces, ignore_index=True)


def align_notes_to_windows(
    sensor_windows: pd.DataFrame,
    notes_embedded: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each sensor window, aggregate any note embeddings that fall
    inside [window_start, window_end) by averaging.

    Returns sensor_windows with columns:
      has_note (bool), note_emb_mean_0 … note_emb_mean_383
    """
    emb_cols = [c for c in notes_embedded.columns if c.startswith("emb_")]
    n_dim = len(emb_cols)

    machines = sensor_windows["machine_id"].unique()
    pieces = []

    for m in machines:
        sw = sensor_windows[sensor_windows["machine_id"] == m].copy().reset_index(drop=True)
        notes_m = notes_embedded[notes_embedded["machine_id"] == m].copy()

        mean_embs = []
        has_notes = []

        for _, win in sw.iterrows():
            mask = (
                (notes_m["timestamp"] >= win["window_start"]) &
                (notes_m["timestamp"] < win["window_end"])
            )
            in_win = notes_m[mask]
            if len(in_win) > 0 and emb_cols:
                avg = in_win[emb_cols].mean(axis=0).values
                has_notes.append(True)
            else:
                avg = np.zeros(n_dim, dtype=np.float32) if n_dim > 0 else np.array([])
                has_notes.append(False)
            mean_embs.append(avg)

        sw["has_note"] = has_notes
        if n_dim > 0:
            emb_df = pd.DataFrame(
                mean_embs,
                columns=[f"note_emb_{i}" for i in range(n_dim)],
                index=sw.index,
            )
            sw = pd.concat([sw, emb_df], axis=1)

        pieces.append(sw)

    return pd.concat(pieces, ignore_index=True)


def _log_fill_defaults() -> dict:
    return {
        "n_events": 0, "n_alarm": 0, "n_warning": 0, "n_maintenance": 0,
        "n_production": 0, "n_diagnostic": 0, "n_operational": 0,
        "has_alarm": False, "has_warning": False, "unique_event_codes": "",
    }


def _fill_log_defaults(df: pd.DataFrame) -> None:
    defaults = _log_fill_defaults()
    for col, val in defaults.items():
        if col in df.columns:
            if isinstance(val, bool):
                df[col] = df[col].fillna(val)
            elif isinstance(val, int):
                df[col] = df[col].fillna(val).astype(int)
            else:
                df[col] = df[col].fillna(val)


# numpy is used inside the function body above
import numpy as np  # noqa: E402 (placed after helpers that reference it)
