"""
Welding Parameter Advisor.

Given: material type + thickness (+ optional process override)
Returns: optimal parameter ranges, efficiency score, and plain-language guidance.

Also provides a "current vs optimal" comparison when live sensor data is supplied.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG = Path(__file__).parent.parent.parent / "config" / "welding_params.yaml"

MATERIAL_ALIASES = {
    "steel":           "mild_steel",
    "mild steel":      "mild_steel",
    "carbon steel":    "mild_steel",
    "ms":              "mild_steel",
    "stainless":       "stainless_steel",
    "stainless steel": "stainless_steel",
    "ss":              "stainless_steel",
    "inox":            "stainless_steel",
    "aluminium":       "aluminum",
    "aluminum":        "aluminum",
    "al":              "aluminum",
    "alu":             "aluminum",
}

THICKNESS_BANDS = {
    "mild_steel":       [(1, 3, "thin"), (3, 8, "medium"), (8, 999, "thick")],
    "stainless_steel":  [(1, 3, "thin"), (3, 8, "medium"), (8, 999, "thick")],
    "aluminum":         [(1, 4, "thin"), (4, 12, "medium"), (12, 999, "thick")],
}


@lru_cache(maxsize=1)
def _load_params() -> dict:
    with open(_CONFIG, "r") as f:
        return yaml.safe_load(f)


def _band_for_thickness(material: str, thickness_mm: float) -> str:
    for lo, hi, label in THICKNESS_BANDS.get(material, []):
        if lo <= thickness_mm < hi:
            return label
    return "medium"


def recommend_parameters(
    material: str,
    thickness_mm: float,
    process: str = "MIG_MAG",
) -> dict:
    """
    Return recommended welding parameters for a material + thickness combination.

    Parameters
    ----------
    material      : e.g. "mild_steel", "stainless_steel", "aluminum"
    thickness_mm  : workpiece thickness in mm
    process       : welding process key (default: "MIG_MAG")

    Returns
    -------
    dict with keys:
      material, thickness_mm, band, params (dict), efficiency_score,
      optimal_speed (midpoint of speed range),
      optimal_current (midpoint of current range),
      notes, summary_text
    """
    mat = MATERIAL_ALIASES.get(material.lower().strip(), material.lower().replace(" ", "_"))
    params_db = _load_params()
    process_db = params_db.get(process, params_db.get("MIG_MAG", {}))
    mat_db = process_db.get(mat)

    if mat_db is None:
        return {
            "error": f"Material '{material}' not found. Supported: mild_steel, stainless_steel, aluminum.",
            "material": material,
            "thickness_mm": thickness_mm,
        }

    band = _band_for_thickness(mat, thickness_mm)
    p = mat_db[band]

    # Midpoints for "optimal" single values
    opt_current = round((p["welding_current"][0] + p["welding_current"][1]) / 2)
    opt_voltage = round((p["arc_voltage"][0] + p["arc_voltage"][1]) / 2, 1)
    opt_speed   = round((p["welding_speed"][0] + p["welding_speed"][1]) / 2)
    opt_wfr     = round((p["wire_feed_rate"][0] + p["wire_feed_rate"][1]) / 2, 1)
    opt_gas     = round((p["shielding_gas_flow"][0] + p["shielding_gas_flow"][1]) / 2)

    # Deposition rate estimate (g/min) ≈ wire_feed_rate * π*(d/2)² * density * 1000
    wire_d = p.get("wire_diameter", 1.2)
    density = 7.85 if "steel" in mat else 2.70  # g/cm³
    dep_rate = round(opt_wfr * 100 * 3.14159 * (wire_d / 20) ** 2 * density, 1)

    summary = _build_summary(mat, thickness_mm, band, p, opt_current, opt_voltage,
                             opt_speed, opt_wfr, dep_rate)

    return {
        "material":         mat,
        "material_display": mat.replace("_", " ").title(),
        "thickness_mm":     thickness_mm,
        "band":             band,
        "process":          process,
        "params":           p,
        "efficiency_score": p.get("efficiency_score", 7),
        "optimal_current":  opt_current,
        "optimal_voltage":  opt_voltage,
        "optimal_speed":    opt_speed,
        "optimal_wfr":      opt_wfr,
        "optimal_gas":      opt_gas,
        "deposition_rate_g_per_min": dep_rate,
        "notes":            p.get("notes", ""),
        "summary_text":     summary,
    }


def compare_to_optimal(live_readings: dict, recommendation: dict) -> dict:
    """
    Compare live sensor averages to optimal parameter ranges.

    Parameters
    ----------
    live_readings  : dict with keys welding_current, arc_voltage, welding_speed,
                     wire_feed_rate, shielding_gas_flow (averaged over recent window)
    recommendation : output of recommend_parameters()

    Returns
    -------
    dict with per-parameter status: "ok" | "too_high" | "too_low" and deviation %
    """
    p = recommendation.get("params", {})
    check_map = {
        "welding_current":    "welding_current",
        "arc_voltage":        "arc_voltage",
        "welding_speed":      "welding_speed",
        "wire_feed_rate":     "wire_feed_rate",
        "shielding_gas_flow": "shielding_gas_flow",
    }
    results = {}
    for field, param_key in check_map.items():
        actual = live_readings.get(field)
        if actual is None or param_key not in p:
            continue
        lo, hi = p[param_key]
        mid = (lo + hi) / 2
        if actual < lo:
            dev = round((lo - actual) / mid * 100, 1)
            results[field] = {"status": "too_low",  "actual": round(actual, 1),
                              "range": [lo, hi], "deviation_pct": dev}
        elif actual > hi:
            dev = round((actual - hi) / mid * 100, 1)
            results[field] = {"status": "too_high", "actual": round(actual, 1),
                              "range": [lo, hi], "deviation_pct": dev}
        else:
            results[field] = {"status": "ok",       "actual": round(actual, 1),
                              "range": [lo, hi], "deviation_pct": 0.0}
    return results


def parse_param_query(question: str) -> dict | None:
    """
    Extract material and thickness from a natural-language parameter query.
    Returns None if the question doesn't look like a parameter request.

    Examples:
      "best settings for 5mm mild steel"   -> {material: "mild_steel", thickness_mm: 5.0}
      "what speed for 3mm aluminum"         -> {material: "aluminum",   thickness_mm: 3.0}
      "optimal current for 10mm stainless"  -> {material: "stainless_steel", thickness_mm: 10.0}
    """
    q = question.lower()

    # Parameter intent keywords
    param_triggers = [
        "best setting", "optimal setting", "recommend setting",
        "what speed", "what current", "what voltage", "what parameter",
        "best parameter", "optimal parameter", "best speed",
        "best current", "best voltage", "best wire", "best gas",
        "optimal current", "optimal speed", "optimal voltage",
        "optimal feed", "optimal gas",
        "how fast", "what feed", "efficiency", "optimal weld",
    ]
    if not any(t in q for t in param_triggers):
        return None

    # Extract thickness (number followed by mm or just a number near material)
    thickness_match = re.search(r"(\d+(?:\.\d+)?)\s*mm", q)
    if not thickness_match:
        thickness_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:millimetre|millimeter)", q)
    thickness_mm = float(thickness_match.group(1)) if thickness_match else None

    # Extract material
    material = None
    for alias in sorted(MATERIAL_ALIASES.keys(), key=len, reverse=True):
        if alias in q:
            material = MATERIAL_ALIASES[alias]
            break

    if material is None and thickness_mm is None:
        return None

    return {
        "material":     material or "mild_steel",
        "thickness_mm": thickness_mm or 5.0,
    }


def _build_summary(mat, thickness_mm, band, p, cur, volt, spd, wfr, dep) -> str:
    mat_label = mat.replace("_", " ").title()
    return (
        f"Recommended MIG/MAG parameters for {mat_label} ({thickness_mm} mm — {band}):\n"
        f"  Current:     {p['welding_current'][0]}–{p['welding_current'][1]} A  (optimal: {cur} A)\n"
        f"  Voltage:     {p['arc_voltage'][0]}–{p['arc_voltage'][1]} V  (optimal: {volt} V)\n"
        f"  Speed:       {p['welding_speed'][0]}–{p['welding_speed'][1]} mm/min  (optimal: {spd} mm/min)\n"
        f"  Wire feed:   {p['wire_feed_rate'][0]}–{p['wire_feed_rate'][1]} m/min  (optimal: {wfr} m/min)\n"
        f"  Gas flow:    {p['shielding_gas_flow'][0]}–{p['shielding_gas_flow'][1]} L/min\n"
        f"  Gas mix:     {p.get('gas_mix','—')}\n"
        f"  Wire diam:   {p.get('wire_diameter','1.2')} mm\n"
        f"  Heat input:  {p['heat_input_range'][0]}–{p['heat_input_range'][1]} kJ/mm\n"
        f"  Deposition:  ~{dep} g/min\n"
        f"  Efficiency:  {p.get('efficiency_score',7)}/10\n"
        f"  Note: {p.get('notes','')}"
    )
