"""
Generate a synthetic tri-modal manufacturing dataset with injected anomalies.

Outputs (to data/raw/):
  sensors.csv      — 1-minute sensor readings for 3 machines over 7 days
  logs.csv         — Structured event log entries
  notes.csv        — Free-text operator shift notes
  ground_truth.csv — Per-window anomaly labels with root causes
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ── Constants ────────────────────────────────────────────────────────────────

MACHINES = ["machine_1", "machine_2", "machine_3"]
START = datetime(2026, 5, 1, 0, 0, 0)
END = datetime(2026, 5, 7, 23, 59, 0)
FREQ_MINUTES = 1
ANOMALIES_PER_MACHINE = 10
ANOMALY_DURATION_MINUTES = 30

ANOMALY_TYPES = [
    "overheating",
    "bearing_failure",
    "pressure_loss",
    "motor_overload",
    "coolant_failure",
]

# Normal operating ranges (mean, std)
NORMAL_PARAMS = {
    "temperature":        (70.0, 3.0),   # °C
    "vibration":          (0.30, 0.05),  # mm/s
    "pressure":           (4.0,  0.2),   # bar
    "rpm":                (1500, 20.0),  # RPM
    "power_consumption":  (55.0, 4.0),   # kW
}

# Anomaly deltas applied to mean (additive)
ANOMALY_DELTAS = {
    "overheating": {
        "temperature":       (30, 15),
        "power_consumption": (18, 5),
    },
    "bearing_failure": {
        "vibration":  (3.0, 0.5),
        "rpm":        (-150, 30),
    },
    "pressure_loss": {
        "pressure": (-2.5, 0.3),
    },
    "motor_overload": {
        "power_consumption": (32, 5),
        "temperature":       (20, 5),
    },
    "coolant_failure": {
        "temperature": (25, 8),
        "pressure":    (-1.5, 0.3),
    },
}

ROOT_CAUSES = {
    "overheating":     "Cooling system blockage or fan failure",
    "bearing_failure": "Worn bearing race or insufficient lubrication",
    "pressure_loss":   "Hydraulic seal leak or valve malfunction",
    "motor_overload":  "Mechanical jam or excessive load on drive shaft",
    "coolant_failure": "Coolant pump failure or burst coolant line",
}

LOG_EVENT_MAP = {
    "overheating": [
        ("WARN_TEMP_HIGH",   "warning",     "Temperature exceeded upper threshold"),
        ("ALARM_OVERHEAT",   "alarm",       "Overtemperature alarm triggered"),
        ("FAN_SPEED_DROP",   "diagnostic",  "Cooling fan speed below setpoint"),
    ],
    "bearing_failure": [
        ("WARN_VIB_HIGH",    "warning",     "Vibration exceeded normal band"),
        ("MAINT_BEARING",    "maintenance", "Bearing wear indicator active"),
        ("RPM_FLUCTUATION",  "diagnostic",  "RPM instability detected"),
    ],
    "pressure_loss": [
        ("WARN_PRES_LOW",    "warning",     "Hydraulic pressure below minimum"),
        ("ALARM_PRES_LOSS",  "alarm",       "Pressure loss alarm triggered"),
        ("VALVE_CHECK",      "diagnostic",  "Valve position sensor mismatch"),
    ],
    "motor_overload": [
        ("ALARM_OVERLOAD",   "alarm",       "Motor current exceeded rated limit"),
        ("WARN_POWER_HIGH",  "warning",     "Power consumption above setpoint"),
        ("THERMAL_TRIP",     "alarm",       "Motor thermal protection activated"),
    ],
    "coolant_failure": [
        ("WARN_COOLANT_TEMP","warning",     "Coolant temperature rising"),
        ("ALARM_COOLANT",    "alarm",       "Coolant flow rate critically low"),
        ("PUMP_FAULT",       "diagnostic",  "Coolant pump fault code 0x3F"),
    ],
}

OPERATOR_NOTE_TEMPLATES = {
    "overheating": [
        "Machine {m} running hot, coolant check needed",
        "Noticed {m} temperature climbing mid-shift, reported to maintenance",
        "High temp warning on {m} around {t}, fan might be blocked",
        "{m} overheating again — same issue as last week",
    ],
    "bearing_failure": [
        "Unusual grinding noise from {m} bearing area",
        "{m} vibration seems higher than normal, monitoring closely",
        "Flagged {m} for bearing inspection at next PM window",
        "Vibration on {m} spiked near {t}, rough feel through handguard",
    ],
    "pressure_loss": [
        "Pressure drop on {m} hydraulic circuit, checked seals",
        "{m} losing pressure intermittently around {t}",
        "Low pressure alarm on {m}, opened maintenance ticket",
        "Found minor leak on {m} hydraulic line, patched temporarily",
    ],
    "motor_overload": [
        "{m} tripped overload protection at {t}, reset and resumed",
        "Motor on {m} running hot, load may be too high",
        "Thermal trip on {m} during peak run — load reduced for now",
        "Overload fault on {m} at {t}, checked mechanical jam — cleared",
    ],
    "coolant_failure": [
        "Coolant low on {m}, topped up but still alarming",
        "{m} coolant pump making noise, flagged for inspection",
        "Coolant flow alarm on {m} near {t}, possible pump issue",
        "{m} coolant system acting up — temp rising despite refill",
    ],
}

NORMAL_LOG_EVENTS = [
    ("PROD_START",      "production",  "Production run started"),
    ("PROD_STOP",       "production",  "Production run stopped"),
    ("SHIFT_CHANGE",    "operational", "Operator shift changeover"),
    ("MAINT_ROUTINE",   "maintenance", "Routine preventive maintenance performed"),
    ("SETPOINT_CHANGE", "operational", "Process setpoint adjusted by operator"),
    ("SENSOR_CALIB",    "diagnostic",  "Sensor calibration check completed"),
    ("BATCH_COMPLETE",  "production",  "Batch production cycle completed"),
]

OPERATORS = ["OP001", "OP002", "OP003", "OP004"]

OUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _time_range() -> list[datetime]:
    times = []
    t = START
    while t <= END:
        times.append(t)
        t += timedelta(minutes=FREQ_MINUTES)
    return times


def _schedule_anomalies(times: list[datetime]) -> list[dict]:
    """Pick non-overlapping anomaly windows across the week."""
    n = len(times)
    window = ANOMALY_DURATION_MINUTES
    # Minimum gap between anomalies to avoid overlap
    min_gap = window + 60
    anomalies = []
    used_indices: set[int] = set()

    for _ in range(ANOMALIES_PER_MACHINE):
        for attempt in range(1000):
            start_idx = rng.integers(window, n - window)
            # Check for overlap with existing anomalies
            conflict = any(
                abs(int(start_idx) - a["start_idx"]) < min_gap
                for a in anomalies
            )
            if not conflict:
                atype = random.choice(ANOMALY_TYPES)
                anomalies.append({
                    "start_idx": int(start_idx),
                    "end_idx":   int(start_idx) + window,
                    "type":      atype,
                })
                break

    anomalies.sort(key=lambda x: x["start_idx"])
    return anomalies


def _build_sensor_series(times: list[datetime], anomalies: list[dict]) -> pd.DataFrame:
    n = len(times)
    channels = list(NORMAL_PARAMS.keys())
    data = {}

    for ch in channels:
        mu, sigma = NORMAL_PARAMS[ch]
        # Baseline: smooth walk + noise
        noise = rng.normal(0, sigma, n)
        drift = np.cumsum(rng.normal(0, sigma * 0.02, n))
        drift -= drift.mean()
        series = mu + noise + drift
        data[ch] = series.copy()

    # Inject anomalies
    for a in anomalies:
        deltas = ANOMALY_DELTAS[a["type"]]
        s, e = a["start_idx"], a["end_idx"]
        for ch, (delta_mu, delta_sigma) in deltas.items():
            spike = rng.normal(delta_mu, delta_sigma, e - s)
            data[ch][s:e] += spike

    # Clip to physically plausible ranges
    clips = {
        "temperature":       (10.0, 200.0),
        "vibration":         (0.0,  10.0),
        "pressure":          (0.0,  10.0),
        "rpm":               (0.0,  2000.0),
        "power_consumption": (0.0,  150.0),
    }
    for ch, (lo, hi) in clips.items():
        data[ch] = np.clip(data[ch], lo, hi)

    df = pd.DataFrame({"timestamp": times})
    for ch in channels:
        df[ch] = np.round(data[ch], 3)

    return df


def _build_log_entries(
    machine: str, times: list[datetime], anomalies: list[dict]
) -> list[dict]:
    entries = []

    # Normal operational events (roughly every 2–4 hours)
    for i in range(0, len(times), rng.integers(120, 240)):
        event = random.choice(NORMAL_LOG_EVENTS)
        entries.append({
            "timestamp":   times[i].isoformat(),
            "machine_id":  machine,
            "event_code":  event[0],
            "event_type":  event[1],
            "description": event[2],
        })

    # Anomaly-correlated events
    for a in anomalies:
        related = LOG_EVENT_MAP[a["type"]]
        for offset_minutes in [0, 5, 15]:
            idx = min(a["start_idx"] + offset_minutes, len(times) - 1)
            event = related[min(offset_minutes // 5, len(related) - 1)]
            entries.append({
                "timestamp":   times[idx].isoformat(),
                "machine_id":  machine,
                "event_code":  event[0],
                "event_type":  event[1],
                "description": event[2],
            })

    entries.sort(key=lambda x: x["timestamp"])
    return entries


def _build_notes(
    machine: str, times: list[datetime], anomalies: list[dict]
) -> list[dict]:
    notes = []
    # ~1 note per anomaly, offset 0–20 minutes after anomaly start
    for a in anomalies:
        if rng.random() < 0.85:  # 85% chance an operator noticed
            offset = int(rng.integers(0, 20))
            idx = min(a["start_idx"] + offset, len(times) - 1)
            t_str = times[idx].strftime("%H:%M")
            template = random.choice(OPERATOR_NOTE_TEMPLATES[a["type"]])
            text = template.format(m=machine, t=t_str)
            notes.append({
                "timestamp":  times[idx].isoformat(),
                "machine_id": machine,
                "operator_id": random.choice(OPERATORS),
                "note_text":  text,
            })
    # Add a few routine notes not tied to anomalies
    for _ in range(3):
        idx = int(rng.integers(0, len(times)))
        notes.append({
            "timestamp":  times[idx].isoformat(),
            "machine_id": machine,
            "operator_id": random.choice(OPERATORS),
            "note_text":  f"{machine} running normally, no issues to report.",
        })
    notes.sort(key=lambda x: x["timestamp"])
    return notes


def _build_ground_truth(
    machine: str, times: list[datetime], anomalies: list[dict]
) -> list[dict]:
    rows = []
    for i, a in enumerate(anomalies):
        rows.append({
            "window_id":    f"{machine}_w{i:03d}",
            "window_start": times[a["start_idx"]].isoformat(),
            "window_end":   times[min(a["end_idx"], len(times) - 1)].isoformat(),
            "machine_id":   machine,
            "is_anomaly":   True,
            "anomaly_type": a["type"],
            "root_cause":   ROOT_CAUSES[a["type"]],
        })
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def generate(out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    times = _time_range()
    print(f"Time range: {times[0]} to {times[-1]}  ({len(times):,} minutes)")

    all_sensors: list[pd.DataFrame] = []
    all_logs:    list[dict] = []
    all_notes:   list[dict] = []
    all_gt:      list[dict] = []

    for machine in MACHINES:
        print(f"  Generating {machine}…")
        anomalies = _schedule_anomalies(times)

        sensor_df = _build_sensor_series(times, anomalies)
        sensor_df.insert(0, "machine_id", machine)
        all_sensors.append(sensor_df)

        all_logs.extend(_build_log_entries(machine, times, anomalies))
        all_notes.extend(_build_notes(machine, times, anomalies))
        all_gt.extend(_build_ground_truth(machine, times, anomalies))

    # Write sensors.csv
    sensors = pd.concat(all_sensors, ignore_index=True)
    sensors.to_csv(out_dir / "sensors.csv", index=False)
    print(f"sensors.csv      -> {len(sensors):,} rows")

    # Write logs.csv
    logs = pd.DataFrame(all_logs).sort_values("timestamp").reset_index(drop=True)
    logs.to_csv(out_dir / "logs.csv", index=False)
    print(f"logs.csv         -> {len(logs):,} rows")

    # Write notes.csv
    notes = pd.DataFrame(all_notes).sort_values("timestamp").reset_index(drop=True)
    notes.to_csv(out_dir / "notes.csv", index=False)
    print(f"notes.csv        -> {len(notes):,} rows")

    # Write ground_truth.csv
    gt = pd.DataFrame(all_gt).sort_values(["machine_id", "window_start"]).reset_index(drop=True)
    gt.to_csv(out_dir / "ground_truth.csv", index=False)
    print(f"ground_truth.csv -> {len(gt):,} rows")

    print(f"\nAll files written to {out_dir.resolve()}")


if __name__ == "__main__":
    generate()
