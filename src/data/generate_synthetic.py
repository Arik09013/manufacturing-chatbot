"""
Generate a synthetic tri-modal welding dataset with injected anomalies.

Outputs (to data/raw/):
  sensors.csv      — 1-minute welding sensor readings for 3 stations over 7 days
  logs.csv         — Structured event log entries
  notes.csv        — Free-text operator notes
  ground_truth.csv — Per-window anomaly labels with root causes

Sensor channels (MIG/MAG welding):
  welding_current   (A)      — arc current
  arc_voltage       (V)      — arc voltage
  welding_speed     (mm/min) — torch travel speed
  wire_feed_rate    (m/min)  — wire spool feed rate
  shielding_gas_flow (L/min) — protective gas flow
  heat_input        (kJ/mm)  — derived quality index: (I*U*0.8)/(v*1000)
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────

MACHINES = ["station_1", "station_2", "station_3"]
START = datetime(2026, 5, 1, 6, 0, 0)   # 06:00 shift start
END   = datetime(2026, 5, 7, 22, 0, 0)  # 22:00 shift end
FREQ_MINUTES        = 1
ANOMALIES_PER_MACHINE = 10
ANOMALY_DURATION_MINUTES = 30

ANOMALY_TYPES = [
    "arc_instability",
    "wire_feed_fault",
    "gas_flow_failure",
    "overheating",
    "underheat",
]

# Normal operating ranges for MIG/MAG on mild steel (mean, std)
NORMAL_PARAMS = {
    "welding_current":    (160.0, 10.0),   # A
    "arc_voltage":        (24.0,  1.0),    # V
    "welding_speed":      (350.0, 20.0),   # mm/min
    "wire_feed_rate":     (8.0,   0.3),    # m/min
    "shielding_gas_flow": (15.0,  0.5),    # L/min
    "heat_input":         (0.53,  0.04),   # kJ/mm (derived from I, U, v)
}

# Anomaly deltas — additive to mean value during anomaly windows
ANOMALY_DELTAS = {
    "arc_instability": {
        "welding_current":  (0,   50.0),   # high fluctuation
        "arc_voltage":      (0,   6.0),    # high fluctuation
        "heat_input":       (0,   0.12),
    },
    "wire_feed_fault": {
        "wire_feed_rate":   (-4.5, 0.5),   # significant drop
        "welding_current":  (-35,  8.0),   # current follows wire
        "heat_input":       (-0.15, 0.04),
    },
    "gas_flow_failure": {
        "shielding_gas_flow": (-11.0, 1.0), # near-zero flow
    },
    "overheating": {
        "welding_current":  (+60,  12.0),  # current too high
        "welding_speed":    (-90,  15.0),  # moving too slow
        "heat_input":       (+0.3, 0.06),  # heat input spikes
    },
    "underheat": {
        "welding_speed":    (+140, 20.0),  # moving too fast
        "arc_voltage":      (-5,   1.0),   # voltage too low
        "heat_input":       (-0.2, 0.04),  # insufficient heat input
    },
}

ROOT_CAUSES = {
    "arc_instability":  "Contaminated wire surface, worn contact tip, or unstable power supply",
    "wire_feed_fault":  "Wire drive motor slip, blocked liner, or spool tangling",
    "gas_flow_failure": "Empty shielding gas cylinder or solenoid valve fault",
    "overheating":      "Travel speed too low — heat input exceeded threshold for material thickness",
    "underheat":        "Travel speed too high — insufficient heat input, risk of incomplete fusion",
}

LOG_EVENT_MAP = {
    "arc_instability": [
        ("WARN_ARC_UNSTABLE", "warning",     "Arc current fluctuation detected"),
        ("ALARM_ARC_LOSS",    "alarm",       "Arc loss — weld interrupted"),
        ("ARC_FAULT_CODE",    "diagnostic",  "Arc fault code 0x21 logged"),
    ],
    "wire_feed_fault": [
        ("WARN_WIRE_FEED",    "warning",     "Wire feed rate below setpoint"),
        ("ALARM_WIRE_STOP",   "alarm",       "Wire feed stopped — arc broken"),
        ("MOTOR_FAULT",       "diagnostic",  "Wire drive motor current fault"),
    ],
    "gas_flow_failure": [
        ("WARN_GAS_LOW",      "warning",     "Shielding gas flow below minimum"),
        ("ALARM_GAS_FAIL",    "alarm",       "Gas flow critical — weld quality at risk"),
        ("SOLENOID_FAULT",    "diagnostic",  "Gas solenoid valve fault code 0x08"),
    ],
    "overheating": [
        ("WARN_HEAT_HIGH",    "warning",     "Heat input above upper control limit"),
        ("ALARM_BURN_THROUGH","alarm",       "Burn-through risk — heat input critical"),
        ("SPEED_ALERT",       "diagnostic",  "Travel speed below minimum setpoint"),
    ],
    "underheat": [
        ("WARN_HEAT_LOW",     "warning",     "Heat input below lower control limit"),
        ("QUALITY_ALERT",     "alarm",       "Quality hold — insufficient penetration risk"),
        ("FUSION_CHECK",      "diagnostic",  "Incomplete fusion check triggered"),
    ],
}

OPERATOR_NOTE_TEMPLATES = {
    "arc_instability": [
        "{m} arc was flickering badly around {t}, checked tip",
        "Unstable arc on {m} near {t}, may need new contact tip",
        "{m} weld quality inconsistent — arc jumping at {t}",
        "Lots of spatter from {m} around {t}, arc not stable",
    ],
    "wire_feed_fault": [
        "Wire jammed on {m} at {t}, had to clear liner",
        "{m} wire feed stopped at {t} — restarted manually",
        "Wire drive slipping on {m}, replaced pressure roller",
        "{m} ran out of wire mid-bead at {t}, restarted weld",
    ],
    "gas_flow_failure": [
        "Gas bottle empty on {m} around {t}, swapped cylinder",
        "{m} gas alarm at {t} — solenoid may be stuck",
        "Porous welds from {m} near {t}, no gas flow",
        "Checked {m} gas line at {t}, found kink in hose",
    ],
    "overheating": [
        "Burn-through on thin section at {m} around {t}",
        "{m} operator reduced speed at {t} — heat buildup",
        "Distortion on part welded at {m} near {t}, heat too high",
        "{m} weld discoloration noted at {t}, possible overheating",
    ],
    "underheat": [
        "Incomplete fusion on parts from {m} at {t}, quality hold",
        "{m} running too fast at {t} — poor penetration noted",
        "QC flagged {m} welds from {t} — lack of fusion",
        "{m} voltage too low at {t}, increased setpoint",
    ],
}

NORMAL_LOG_EVENTS = [
    ("WELD_START",      "production",  "Welding program started"),
    ("WELD_STOP",       "production",  "Welding program stopped normally"),
    ("SHIFT_CHANGE",    "operational", "Operator shift changeover"),
    ("PARAM_CHECK",     "maintenance", "Welding parameters verified by supervisor"),
    ("TIP_CHANGE",      "maintenance", "Contact tip replaced (scheduled)"),
    ("WIRE_CHANGE",     "operational", "Wire spool changed"),
    ("GAS_CHECK",       "operational", "Shielding gas level checked — OK"),
    ("CALIB_DONE",      "diagnostic",  "Current and voltage sensor calibration completed"),
]

OPERATORS = ["WLD01", "WLD02", "WLD03", "WLD04"]
OUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _time_range() -> list[datetime]:
    times, t = [], START
    while t <= END:
        times.append(t)
        t += timedelta(minutes=FREQ_MINUTES)
    return times


def _schedule_anomalies(times: list[datetime]) -> list[dict]:
    n, window, min_gap = len(times), ANOMALY_DURATION_MINUTES, ANOMALY_DURATION_MINUTES + 60
    anomalies = []
    for _ in range(ANOMALIES_PER_MACHINE):
        for _ in range(1000):
            start_idx = int(rng.integers(window, n - window))
            if not any(abs(start_idx - a["start_idx"]) < min_gap for a in anomalies):
                anomalies.append({
                    "start_idx": start_idx,
                    "end_idx":   start_idx + window,
                    "type":      random.choice(ANOMALY_TYPES),
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
        noise = rng.normal(0, sigma, n)
        drift = np.cumsum(rng.normal(0, sigma * 0.015, n))
        drift -= drift.mean()
        data[ch] = (mu + noise + drift).copy()

    for a in anomalies:
        deltas = ANOMALY_DELTAS[a["type"]]
        s, e = a["start_idx"], a["end_idx"]
        for ch, (delta_mu, delta_sigma) in deltas.items():
            data[ch][s:e] += rng.normal(delta_mu, delta_sigma, e - s)

    # Recompute heat_input from actual current/voltage/speed to keep physical consistency
    speed_mm_s = np.clip(data["welding_speed"], 10, 800) / 60.0
    data["heat_input"] = (
        np.clip(data["welding_current"], 0, 400) *
        np.clip(data["arc_voltage"], 0, 50) *
        0.8
    ) / (speed_mm_s * 1000)

    clips = {
        "welding_current":    (0.0,  400.0),
        "arc_voltage":        (0.0,   50.0),
        "welding_speed":      (10.0, 800.0),
        "wire_feed_rate":     (0.0,   20.0),
        "shielding_gas_flow": (0.0,   30.0),
        "heat_input":         (0.0,    2.0),
    }
    for ch, (lo, hi) in clips.items():
        data[ch] = np.clip(data[ch], lo, hi)

    df = pd.DataFrame({"timestamp": times})
    for ch in channels:
        df[ch] = np.round(data[ch], 3)
    return df


def _build_log_entries(machine, times, anomalies):
    entries = []
    for i in range(0, len(times), int(rng.integers(90, 180))):
        ev = random.choice(NORMAL_LOG_EVENTS)
        entries.append({"timestamp": times[i].isoformat(), "machine_id": machine,
                        "event_code": ev[0], "event_type": ev[1], "description": ev[2]})
    for a in anomalies:
        related = LOG_EVENT_MAP[a["type"]]
        for offset in [0, 5, 15]:
            idx = min(a["start_idx"] + offset, len(times) - 1)
            ev = related[min(offset // 5, len(related) - 1)]
            entries.append({"timestamp": times[idx].isoformat(), "machine_id": machine,
                            "event_code": ev[0], "event_type": ev[1], "description": ev[2]})
    entries.sort(key=lambda x: x["timestamp"])
    return entries


def _build_notes(machine, times, anomalies):
    notes = []
    for a in anomalies:
        if rng.random() < 0.85:
            idx = min(a["start_idx"] + int(rng.integers(0, 20)), len(times) - 1)
            t_str = times[idx].strftime("%H:%M")
            text = random.choice(OPERATOR_NOTE_TEMPLATES[a["type"]]).format(m=machine, t=t_str)
            notes.append({"timestamp": times[idx].isoformat(), "machine_id": machine,
                          "operator_id": random.choice(OPERATORS), "note_text": text})
    for _ in range(3):
        idx = int(rng.integers(0, len(times)))
        notes.append({"timestamp": times[idx].isoformat(), "machine_id": machine,
                      "operator_id": random.choice(OPERATORS),
                      "note_text": f"{machine} welding parameters within spec, no issues."})
    notes.sort(key=lambda x: x["timestamp"])
    return notes


def _build_ground_truth(machine, times, anomalies):
    return [
        {
            "window_id":    f"{machine}_w{i:03d}",
            "window_start": times[a["start_idx"]].isoformat(),
            "window_end":   times[min(a["end_idx"], len(times) - 1)].isoformat(),
            "machine_id":   machine,
            "is_anomaly":   True,
            "anomaly_type": a["type"],
            "root_cause":   ROOT_CAUSES[a["type"]],
        }
        for i, a in enumerate(anomalies)
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    times = _time_range()
    print(f"Time range: {times[0]} to {times[-1]}  ({len(times):,} minutes)")

    all_sensors, all_logs, all_notes, all_gt = [], [], [], []

    for machine in MACHINES:
        print(f"  Generating {machine}...")
        anomalies = _schedule_anomalies(times)
        sensor_df = _build_sensor_series(times, anomalies)
        sensor_df.insert(0, "machine_id", machine)
        all_sensors.append(sensor_df)
        all_logs.extend(_build_log_entries(machine, times, anomalies))
        all_notes.extend(_build_notes(machine, times, anomalies))
        all_gt.extend(_build_ground_truth(machine, times, anomalies))

    sensors = pd.concat(all_sensors, ignore_index=True)
    sensors.to_csv(out_dir / "sensors.csv", index=False)
    print(f"sensors.csv      -> {len(sensors):,} rows")

    logs = pd.DataFrame(all_logs).sort_values("timestamp").reset_index(drop=True)
    logs.to_csv(out_dir / "logs.csv", index=False)
    print(f"logs.csv         -> {len(logs):,} rows")

    notes = pd.DataFrame(all_notes).sort_values("timestamp").reset_index(drop=True)
    notes.to_csv(out_dir / "notes.csv", index=False)
    print(f"notes.csv        -> {len(notes):,} rows")

    gt = pd.DataFrame(all_gt).sort_values(["machine_id", "window_start"]).reset_index(drop=True)
    gt.to_csv(out_dir / "ground_truth.csv", index=False)
    print(f"ground_truth.csv -> {len(gt):,} rows")
    print(f"\nAll files written to {out_dir.resolve()}")


if __name__ == "__main__":
    generate()
