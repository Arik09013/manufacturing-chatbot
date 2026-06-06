"""
Load each raw CSV modality into a clean, typed pandas DataFrame.

All timestamps are parsed to UTC-naive datetime64[ns].
Column dtypes are enforced after loading. Schema validation (Pydantic)
is opt-in via validate=True and runs only when pydantic is installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default paths relative to project root
_RAW = Path(__file__).parent.parent.parent / "data" / "raw"

SENSOR_DTYPES = {
    "machine_id":          "category",
    "welding_current":     "float32",
    "arc_voltage":         "float32",
    "welding_speed":       "float32",
    "wire_feed_rate":      "float32",
    "shielding_gas_flow":  "float32",
    "heat_input":          "float32",
}

LOG_DTYPES = {
    "machine_id":  "category",
    "event_code":  "category",
    "event_type":  "category",
    "description": "string",
}

NOTE_DTYPES = {
    "machine_id":  "category",
    "operator_id": "category",
    "note_text":   "string",
}

GT_DTYPES = {
    "window_id":    "string",
    "machine_id":   "category",
    "is_anomaly":   "bool",
    "anomaly_type": "category",
    "root_cause":   "string",
}


def _parse_timestamps(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_datetime(df[col], utc=False)
    return df


def _cast(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df


def _validate_rows(df: pd.DataFrame, schema_cls, label: str) -> None:
    """Run Pydantic validation on a sample (first 5 rows + last row)."""
    try:
        indices = list(df.index[:5]) + [df.index[-1]]
        for i in set(indices):
            row = df.loc[i].to_dict()
            schema_cls(**row)
    except Exception as exc:
        raise ValueError(f"{label} validation failed: {exc}") from exc


def load_sensors(
    path: Optional[Path] = None,
    machine_id: Optional[str] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """Load sensors.csv. Filter by machine_id if given."""
    path = path or _RAW / "sensors.csv"
    df = pd.read_csv(path)
    df = _parse_timestamps(df, ["timestamp"])
    df = _cast(df, SENSOR_DTYPES)

    if machine_id:
        df = df[df["machine_id"] == machine_id].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    if validate:
        from src.data.schemas import SensorReading
        _validate_rows(df, SensorReading, "SensorReading")

    logger.debug("Loaded sensors: %s rows", len(df))
    return df


def load_logs(
    path: Optional[Path] = None,
    machine_id: Optional[str] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """Load logs.csv."""
    path = path or _RAW / "logs.csv"
    df = pd.read_csv(path)
    df = _parse_timestamps(df, ["timestamp"])
    df = _cast(df, LOG_DTYPES)

    if machine_id:
        df = df[df["machine_id"] == machine_id].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    if validate:
        from src.data.schemas import LogEntry
        _validate_rows(df, LogEntry, "LogEntry")

    logger.debug("Loaded logs: %s rows", len(df))
    return df


def load_notes(
    path: Optional[Path] = None,
    machine_id: Optional[str] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """Load notes.csv."""
    path = path or _RAW / "notes.csv"
    df = pd.read_csv(path)
    df = _parse_timestamps(df, ["timestamp"])
    df = _cast(df, NOTE_DTYPES)

    if machine_id:
        df = df[df["machine_id"] == machine_id].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    if validate:
        from src.data.schemas import OperatorNote
        _validate_rows(df, OperatorNote, "OperatorNote")

    logger.debug("Loaded notes: %s rows", len(df))
    return df


def load_ground_truth(
    path: Optional[Path] = None,
    machine_id: Optional[str] = None,
    validate: bool = False,
) -> pd.DataFrame:
    """Load ground_truth.csv."""
    path = path or _RAW / "ground_truth.csv"
    df = pd.read_csv(path)
    df = _parse_timestamps(df, ["window_start", "window_end"])
    df = _cast(df, GT_DTYPES)

    if machine_id:
        df = df[df["machine_id"] == machine_id].copy()

    df = df.sort_values(["machine_id", "window_start"]).reset_index(drop=True)

    if validate:
        from src.data.schemas import GroundTruthWindow
        _validate_rows(df, GroundTruthWindow, "GroundTruthWindow")

    logger.debug("Loaded ground truth: %s rows", len(df))
    return df


def load_all(
    raw_dir: Optional[Path] = None,
    machine_id: Optional[str] = None,
    validate: bool = False,
) -> dict[str, pd.DataFrame]:
    """Convenience: load all four modalities at once."""
    d = raw_dir or _RAW
    return {
        "sensors":      load_sensors(d / "sensors.csv", machine_id, validate),
        "logs":         load_logs(d / "logs.csv", machine_id, validate),
        "notes":        load_notes(d / "notes.csv", machine_id, validate),
        "ground_truth": load_ground_truth(d / "ground_truth.csv", machine_id, validate),
    }
