"""
Pydantic v2 schemas for the three data modalities + ground truth.

These are used for single-row validation at load time (optional) and as
the authoritative type contract for the rest of the pipeline.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


MACHINE_IDS = {"machine_1", "machine_2", "machine_3"}
ANOMALY_TYPES = {
    "overheating",
    "bearing_failure",
    "pressure_loss",
    "motor_overload",
    "coolant_failure",
}
EVENT_TYPES = {"warning", "alarm", "diagnostic", "production", "operational", "maintenance"}


class SensorReading(BaseModel):
    timestamp: datetime
    machine_id: str
    temperature: float = Field(ge=-10.0, le=300.0)
    vibration: float = Field(ge=0.0, le=20.0)
    pressure: float = Field(ge=0.0, le=20.0)
    rpm: float = Field(ge=0.0, le=3000.0)
    power_consumption: float = Field(ge=0.0, le=500.0)

    @field_validator("machine_id")
    @classmethod
    def valid_machine(cls, v: str) -> str:
        if v not in MACHINE_IDS:
            raise ValueError(f"Unknown machine_id: {v!r}")
        return v


class LogEntry(BaseModel):
    timestamp: datetime
    machine_id: str
    event_code: str
    event_type: str
    description: str

    @field_validator("machine_id")
    @classmethod
    def valid_machine(cls, v: str) -> str:
        if v not in MACHINE_IDS:
            raise ValueError(f"Unknown machine_id: {v!r}")
        return v


class OperatorNote(BaseModel):
    timestamp: datetime
    machine_id: str
    operator_id: str
    note_text: str = Field(min_length=1)

    @field_validator("machine_id")
    @classmethod
    def valid_machine(cls, v: str) -> str:
        if v not in MACHINE_IDS:
            raise ValueError(f"Unknown machine_id: {v!r}")
        return v


class GroundTruthWindow(BaseModel):
    window_id: str
    window_start: datetime
    window_end: datetime
    machine_id: str
    is_anomaly: bool
    anomaly_type: str
    root_cause: str

    @field_validator("machine_id")
    @classmethod
    def valid_machine(cls, v: str) -> str:
        if v not in MACHINE_IDS:
            raise ValueError(f"Unknown machine_id: {v!r}")
        return v

    @field_validator("anomaly_type")
    @classmethod
    def valid_anomaly_type(cls, v: str) -> str:
        if v not in ANOMALY_TYPES:
            raise ValueError(f"Unknown anomaly_type: {v!r}")
        return v
