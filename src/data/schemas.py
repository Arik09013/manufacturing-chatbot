"""Pydantic v2 schemas for welding data modalities."""

from datetime import datetime
from pydantic import BaseModel, Field, field_validator

STATION_IDS = {"station_1", "station_2", "station_3"}
ANOMALY_TYPES = {
    "arc_instability",
    "wire_feed_fault",
    "gas_flow_failure",
    "overheating",
    "underheat",
}


class SensorReading(BaseModel):
    timestamp:          datetime
    machine_id:         str
    welding_current:    float = Field(ge=0.0,  le=400.0)
    arc_voltage:        float = Field(ge=0.0,  le=50.0)
    welding_speed:      float = Field(ge=0.0,  le=800.0)
    wire_feed_rate:     float = Field(ge=0.0,  le=20.0)
    shielding_gas_flow: float = Field(ge=0.0,  le=30.0)
    heat_input:         float = Field(ge=0.0,  le=2.0)

    @field_validator("machine_id")
    @classmethod
    def valid_machine(cls, v: str) -> str:
        if v not in STATION_IDS:
            raise ValueError(f"Unknown machine_id: {v!r}")
        return v


class LogEntry(BaseModel):
    timestamp:   datetime
    machine_id:  str
    event_code:  str
    event_type:  str
    description: str

    @field_validator("machine_id")
    @classmethod
    def valid_machine(cls, v: str) -> str:
        if v not in STATION_IDS:
            raise ValueError(f"Unknown machine_id: {v!r}")
        return v


class OperatorNote(BaseModel):
    timestamp:   datetime
    machine_id:  str
    operator_id: str
    note_text:   str = Field(min_length=1)

    @field_validator("machine_id")
    @classmethod
    def valid_machine(cls, v: str) -> str:
        if v not in STATION_IDS:
            raise ValueError(f"Unknown machine_id: {v!r}")
        return v


class GroundTruthWindow(BaseModel):
    window_id:    str
    window_start: datetime
    window_end:   datetime
    machine_id:   str
    is_anomaly:   bool
    anomaly_type: str
    root_cause:   str

    @field_validator("machine_id")
    @classmethod
    def valid_machine(cls, v: str) -> str:
        if v not in STATION_IDS:
            raise ValueError(f"Unknown machine_id: {v!r}")
        return v

    @field_validator("anomaly_type")
    @classmethod
    def valid_anomaly_type(cls, v: str) -> str:
        if v not in ANOMALY_TYPES:
            raise ValueError(f"Unknown anomaly_type: {v!r}")
        return v
