"""
Welding Parameter Advisor.

Given: material + thickness + process (MIG / MAG / TIG / SMAW)
Returns a structured recommendation containing:
  - the standards window (allowed ranges) for the combination
  - optimized settings found via a deterministic grid search that maximizes
    a physically-grounded objective (deposition throughput, weighted by how
    well the resulting heat input sits inside the standards window)
  - computed metrics (heat input, deposition rate) for the optimized settings
  - a per-parameter sensitivity table: nudge each optimized value up/down one
    step, recompute the metrics, and attach qualitative effect labels

Every number here is computed in Python from physics formulas and the YAML
standards tables. Nothing is produced by an LLM — the synthesis layer only
narrates these pre-computed facts.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG = Path(__file__).parent.parent.parent / "config" / "welding_params.yaml"
_EFFECTS_CONFIG = Path(__file__).parent.parent.parent / "config" / "parameter_effects.yaml"

_GRID_STEPS = 7   # grid-search resolution per axis (small grid, as specified)

MATERIAL_ALIASES = {
    "steel":           "mild_steel",
    "mild steel":      "mild_steel",
    "carbon steel":    "mild_steel",
    "ms":              "mild_steel",
    "stainless":       "stainless_steel",
    "stainless steel": "stainless_steel",
    "ss":              "stainless_steel",
    "ss304":           "stainless_steel",
    "ss316":           "stainless_steel",
    "304":             "stainless_steel",
    "316":             "stainless_steel",
    "inox":            "stainless_steel",
    "aluminium":       "aluminum",
    "aluminum":        "aluminum",
    "al":              "aluminum",
    "alu":             "aluminum",
}

# Metals commonly asked about but NOT in the standards tables. Naming one of these
# should produce an honest "not supported" error, never a fabricated default.
UNSUPPORTED_MATERIALS = frozenset({
    "titanium", "copper", "brass", "bronze", "cast iron", "nickel",
    "inconel", "magnesium", "zinc", "lead", "gold", "silver", "tungsten carbide",
})

THICKNESS_BANDS = {
    "mild_steel":       [(1, 3, "thin"), (3, 8, "medium"), (8, 999, "thick")],
    "stainless_steel":  [(1, 3, "thin"), (3, 8, "medium"), (8, 999, "thick")],
    "aluminum":         [(1, 4, "thin"), (4, 12, "medium"), (12, 999, "thick")],
}

_DEFAULT_PROCESS = "GMAW"


# ── Config loading ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_params() -> dict:
    with open(_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_effects() -> dict:
    with open(_EFFECTS_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _process_alias_map() -> dict:
    """Build a flat {alias_lower: process_family_key} lookup from the YAML."""
    out = {}
    for family, meta in _load_params().get("processes", {}).items():
        out[family.lower()] = family
        for alias in meta.get("aliases", []):
            out[alias.lower()] = family
    return out


def _resolve_process(process: str) -> tuple[str, str] | None:
    """
    Resolve a user-facing process string (e.g. "MIG", "stick", "gtaw") to
    (process_family_key, user_facing_label). Returns None if unrecognised.
    """
    key = process.lower().strip()
    family = _process_alias_map().get(key)
    if family is None:
        return None
    label = process.upper().strip() if len(process.strip()) <= 5 else family
    return family, label


def _band_for_thickness(material: str, thickness_mm: float) -> str:
    for lo, hi, label in THICKNESS_BANDS.get(material, []):
        if lo <= thickness_mm < hi:
            return label
    return "medium"


# ── Physics ───────────────────────────────────────────────────────────────────

def _heat_input(current: float, voltage: float, speed: float, arc_efficiency: float) -> float:
    """HI = (eta * 60 * I * V) / (1000 * S)  ->  kJ/mm  (S in mm/min)."""
    return (arc_efficiency * 60 * current * voltage) / (1000 * speed)


def _deposition_rate(wire_feed_rate, wire_diameter_mm, density_g_cm3, deposition_efficiency):
    """
    g/min from wire feed speed (m/min), wire cross-section, density, and the
    process's deposition (transfer) efficiency. Returns None if there is no
    continuous wire feed (TIG / SMAW).
    """
    if wire_feed_rate is None:
        return None
    cross_section_mm2 = math.pi * (wire_diameter_mm / 2.0) ** 2
    return wire_feed_rate * cross_section_mm2 * density_g_cm3 * deposition_efficiency


def _heat_input_window_score(hi: float, hi_range: list[float]) -> float:
    """
    1.0 at the centre of the standards window, tapering to 0.7 at its edges,
    and 0.0 (disqualified) outside it — enforces "stay inside the standards
    window" while preferring the thermal sweet spot (penetration vs distortion).
    """
    lo, hi_max = hi_range
    if hi < lo or hi > hi_max:
        return 0.0
    mid = (lo + hi_max) / 2.0
    half_width = (hi_max - lo) / 2.0
    if half_width == 0:
        return 1.0
    return 1.0 - 0.3 * abs(hi - mid) / half_width


def _window_distance(hi: float, hi_range: list[float]) -> float:
    lo, hi_max = hi_range
    if hi < lo:
        return lo - hi
    if hi > hi_max:
        return hi - hi_max
    return 0.0


# ── Grid-search optimizer ─────────────────────────────────────────────────────

def _apply_bound(value: float, bound: dict) -> float:
    """Clamp a value to satisfy a one-sided user constraint ('more than X' / 'under X')."""
    if bound["op"] == "min":
        return max(value, bound["value"])
    if bound["op"] == "max":
        return min(value, bound["value"])
    return value


def _optimize(p: dict, arc_efficiency: float, deposition_efficiency: float,
              wire_diameter, density_g_cm3, axis_overrides: dict | None = None,
              axis_bounds: dict | None = None) -> dict:
    """
    Small grid search over (current, speed). Voltage and wire-feed are derived
    from current by linear interpolation at the same fractional position within
    their respective ranges — physically justified because in constant-voltage
    GMAW, voltage and wire-feed speed track with current.

    Objective = deposition_rate * heat_input_window_score   (wire-fed processes)
              = heat_input_window_score                     (TIG / SMAW — no
                                                              deposition metric;
                                                              maximize closeness
                                                              to the thermal
                                                              sweet spot instead)

    `axis_overrides` pins one or more process axes to user-supplied values
    (welding_current / arc_voltage / welding_speed / wire_feed_rate): the pinned
    axis is held fixed and the optimizer searches only the remaining free axes,
    so the answer is "given the value YOU fixed, here is the best the rest can be".

    `axis_bounds` applies one-sided constraints ({field: {"op": "min"|"max",
    "value": x}}) from comparator phrasing ("more than 21 voltage"): each candidate
    value for that axis is clamped to satisfy the bound, so the search stays on the
    operator's side of the limit.
    """
    axis_overrides = axis_overrides or {}
    axis_bounds = axis_bounds or {}
    i_lo, i_hi = p["welding_current"]
    v_lo, v_hi = p["arc_voltage"]
    s_lo, s_hi = p["welding_speed"]
    hi_range = p["heat_input_range"]
    wfr_range = p.get("wire_feed_rate")

    n = _GRID_STEPS

    def _grid_for(lo, hi, field):
        if field in axis_overrides:
            return [axis_overrides[field]]
        grid = [lo + k * (hi - lo) / (n - 1) for k in range(n)]
        if field in axis_bounds:
            grid = [_apply_bound(v, axis_bounds[field]) for v in grid]
        return grid

    i_grid = _grid_for(i_lo, i_hi, "welding_current")
    s_grid = _grid_for(s_lo, s_hi, "welding_speed")

    candidates = []
    for current in i_grid:
        frac = (current - i_lo) / (i_hi - i_lo) if i_hi > i_lo else 0.5
        frac = min(max(frac, 0.0), 1.0)   # clamp so a pinned out-of-window current doesn't extrapolate V/WFS
        if "arc_voltage" in axis_overrides:
            voltage = axis_overrides["arc_voltage"]
        else:
            voltage = v_lo + frac * (v_hi - v_lo)
            if "arc_voltage" in axis_bounds:
                voltage = _apply_bound(voltage, axis_bounds["arc_voltage"])
        if "wire_feed_rate" in axis_overrides:
            wfr = axis_overrides["wire_feed_rate"]
        else:
            wfr = (wfr_range[0] + frac * (wfr_range[1] - wfr_range[0])) if wfr_range else None
            if wfr is not None and "wire_feed_rate" in axis_bounds:
                wfr = _apply_bound(wfr, axis_bounds["wire_feed_rate"])

        for speed in s_grid:
            heat_input = _heat_input(current, voltage, speed, arc_efficiency)
            window = _heat_input_window_score(heat_input, hi_range)
            dep = _deposition_rate(wfr, wire_diameter, density_g_cm3, deposition_efficiency)

            objective = (dep * window) if dep is not None else window
            candidates.append({
                "current": current, "voltage": voltage, "speed": speed, "wire_feed_rate": wfr,
                "heat_input": heat_input, "deposition_rate": dep,
                "window_score": window, "objective": objective,
            })

    in_window = [c for c in candidates if c["window_score"] > 0]
    if in_window:
        best = max(in_window, key=lambda c: c["objective"])
    else:
        # No grid point lands inside the standards heat-input window (shouldn't
        # happen with consistent tables) — fall back to the closest one rather
        # than fabricating an in-window result.
        best = min(candidates, key=lambda c: _window_distance(c["heat_input"], hi_range))

    return best


# ── Sensitivity / "explain why" layer ─────────────────────────────────────────

def _step_for(rng: list[float], minimum: float, ndigits: int = 0) -> float:
    """~10% of the range width, floored at `minimum`, rounded for readability."""
    span = rng[1] - rng[0]
    step = max(minimum, span * 0.1)
    return round(step, ndigits) if ndigits else round(step)


def _direction(new_value: float, base_value: float, eps: float = 1e-6) -> str:
    if new_value > base_value + eps:
        return "higher"
    if new_value < base_value - eps:
        return "lower"
    return "unchanged"


def _round_or_none(value, ndigits=1):
    return round(value, ndigits) if value is not None else None


def _build_sensitivity(best: dict, p: dict, arc_efficiency: float,
                       deposition_efficiency: float, wire_diameter, density_g_cm3,
                       effects_db: dict) -> list[dict]:
    base = {
        "current": best["current"], "voltage": best["voltage"],
        "speed": best["speed"], "wire_feed_rate": best["wire_feed_rate"],
    }
    base_heat_input = best["heat_input"]

    specs = [
        ("welding_current", "current",  p["welding_current"], _step_for(p["welding_current"], 5)),
        ("arc_voltage",     "voltage",  p["arc_voltage"],     _step_for(p["arc_voltage"], 0.5, 1)),
        ("welding_speed",   "speed",    p["welding_speed"],   _step_for(p["welding_speed"], 10)),
    ]
    if base["wire_feed_rate"] is not None:
        specs.append((
            "wire_feed_rate", "wire_feed_rate",
            p["wire_feed_rate"], _step_for(p["wire_feed_rate"], 0.5, 1),
        ))

    heat_input_effects = effects_db.get("heat_input", {})

    def _recompute(field: str, value: float) -> dict:
        vals = dict(base)
        vals[field] = value
        heat_input = _heat_input(vals["current"], vals["voltage"], vals["speed"], arc_efficiency)
        dep = _deposition_rate(vals["wire_feed_rate"], wire_diameter, density_g_cm3, deposition_efficiency)
        hi_dir = _direction(heat_input, base_heat_input)
        return {
            "value": round(value, 2),
            "heat_input": round(heat_input, 3),
            "deposition_rate": _round_or_none(dep),
            "heat_input_change": hi_dir,
            "heat_input_effects": heat_input_effects.get(
                "increase" if hi_dir == "higher" else "decrease", []
            ) if hi_dir != "unchanged" else [],
        }

    table = []
    for param_name, field, rng, step in specs:
        lo, hi = rng
        base_val = base[field]
        # When the operator pinned a value outside the standards window, let the
        # nudge follow the base rather than snapping back into the window (which
        # would otherwise make "raise" land below the base value).
        hi_clamp = max(hi, base_val)
        lo_clamp = min(lo, base_val)
        up_val = min(round(base_val + step, 4), hi_clamp)
        down_val = max(round(base_val - step, 4), lo_clamp)

        table.append({
            "parameter": param_name,
            "base_value": round(base_val, 2),
            "step": step,
            "up":   _recompute(field, up_val),
            "down": _recompute(field, down_val),
            "effects": {
                "increase": effects_db.get(param_name, {}).get("increase", []),
                "decrease": effects_db.get(param_name, {}).get("decrease", []),
            },
        })
    return table


# ── Main entry point ──────────────────────────────────────────────────────────

def recommend_parameters(
    material: str,
    thickness_mm: float,
    process: str = _DEFAULT_PROCESS,
    overrides: dict | None = None,
    bounds: dict | None = None,
    override_ranges: dict | None = None,
) -> dict:
    """
    Return an optimized welding-parameter recommendation for a
    material + thickness + process combination.

    `overrides` lets the operator pin one or more parameters and ask the advisor
    to recompute the rest around them ("if I use 1.5 mm wire, what should the
    other settings be?"). Supported keys: welding_current, arc_voltage,
    welding_speed, wire_feed_rate (process axes — held fixed during grid search)
    and wire_diameter / electrode_diameter / tungsten_diameter (consumable size —
    feeds the deposition-rate calculation). Any pinned value sitting outside the
    standards window is still honored, but flagged in `override_warnings`.

    Returns a dict — see module docstring for the shape, or `error` +
    `supported_processes` if the combination isn't in the standards tables
    (e.g. SMAW is not standard practice for aluminum).
    """
    params_db = _load_params()
    effects_db = _load_effects()

    mat = MATERIAL_ALIASES.get(material.lower().strip(), material.lower().replace(" ", "_"))
    mat_db = params_db.get("materials", {}).get(mat)
    if mat_db is None:
        return {
            "error": f"Material '{material}' not recognised. Supported: mild_steel, stainless_steel, aluminum.",
            "material": material,
            "thickness_mm": thickness_mm,
        }

    resolved = _resolve_process(process)
    if resolved is None:
        return {
            "error": f"Process '{process}' not recognised. Supported: MIG, MAG, TIG, SMAW.",
            "material": mat, "thickness_mm": thickness_mm,
        }
    process_family, process_label = resolved
    process_meta = params_db.get("processes", {}).get(process_family, {})

    process_table = mat_db.get(process_family)
    if process_table is None:
        supported = [k for k in mat_db if k in params_db.get("processes", {})]
        return {
            "error": (
                f"{process_label} is not standard industrial practice for "
                f"{mat.replace('_', ' ')}. Supported processes for this material: "
                f"{', '.join(supported)}."
            ),
            "material": mat, "material_display": mat.replace("_", " ").title(),
            "thickness_mm": thickness_mm, "process": process_label,
            "supported_processes": supported,
        }

    band = _band_for_thickness(mat, thickness_mm)
    p = process_table[band]

    arc_efficiency = process_meta.get("arc_efficiency", 0.8)
    deposition_efficiency = process_meta.get("deposition_efficiency", 0.9)
    table_diameter = p.get("wire_diameter") or p.get("tungsten_diameter") or p.get("electrode_diameter") or 1.2
    density = mat_db.get("density_g_cm3", 7.85)

    # --- Split user-pinned parameters into process axes vs consumable size ---
    overrides = overrides or {}
    bounds = bounds or {}
    override_ranges = override_ranges or {}
    axis_overrides = {k: v for k, v in overrides.items() if k in _AXIS_FIELDS}
    axis_bounds = {k: v for k, v in bounds.items() if k in _AXIS_FIELDS}
    axis_ranges = {k: v for k, v in override_ranges.items() if k in _AXIS_FIELDS}
    diameter_override = next(
        (overrides[k] for k in _DIAMETER_FIELDS if k in overrides), None
    )
    # A diameter given with a comparator ("at least 1.5 mm wire") is still a single
    # value for the deposition calc — take it as an exact diameter.
    if diameter_override is None:
        diameter_override = next(
            (bounds[k]["value"] for k in _DIAMETER_FIELDS if k in bounds), None
        )
    # A pinned wire-feed rate only makes sense for wire-fed processes; drop it for
    # TIG/SMAW (where the table has no wire_feed_rate) rather than fabricating one.
    if p.get("wire_feed_rate") is None:
        axis_overrides.pop("wire_feed_rate", None)
        axis_bounds.pop("wire_feed_rate", None)

    wire_diameter = diameter_override if diameter_override is not None else table_diameter

    # --- Optimize (grid search, holding any pinned axes fixed / bounded) ---
    best = _optimize(p, arc_efficiency, deposition_efficiency, wire_diameter, density,
                     axis_overrides=axis_overrides, axis_bounds=axis_bounds)

    applied_overrides = dict(axis_overrides)
    if diameter_override is not None:
        applied_overrides["wire_diameter"] = diameter_override
    override_warnings = _override_warnings(p, axis_overrides, axis_bounds,
                                           diameter_override, table_diameter, axis_ranges)
    user_fixed = _user_fixed_list(p, axis_overrides, axis_bounds, diameter_override,
                                  axis_ranges)

    # What actually changes when you swap wire diameter? The current/voltage/speed
    # windows are set by the joint (material + thickness), NOT the wire, so they do
    # not move. What changes is metal deposited per unit wire-feed. Give the operator
    # the concrete, computed wire-feed adjustment that holds the SAME deposit — that
    # is the real answer to "what's the parameter if I use 1.5 instead of 1.2".
    consumable_change = _consumable_change(
        p, table_diameter, diameter_override, best, deposition_efficiency, density
    )

    optimized = {
        "welding_current": round(best["current"]),
        "arc_voltage":     round(best["voltage"], 1),
        "welding_speed":   round(best["speed"]),
        "wire_feed_rate":  _round_or_none(best["wire_feed_rate"]),
    }
    computed_metrics = {
        "heat_input_kj_per_mm": round(best["heat_input"], 3),
        "heat_input_range":     p["heat_input_range"],
        "deposition_rate_g_per_min": _round_or_none(best["deposition_rate"]),
    }

    ranges = {
        "welding_current":    p["welding_current"],
        "arc_voltage":        p["arc_voltage"],
        "welding_speed":      p["welding_speed"],
        "wire_feed_rate":     p.get("wire_feed_rate"),
        "shielding_gas_flow": p.get("shielding_gas_flow"),
    }

    # --- Sensitivity / explain-why ---
    sensitivity = _build_sensitivity(best, p, arc_efficiency, deposition_efficiency,
                                     wire_diameter, density, effects_db)

    consumable_field = (
        "wire_diameter" if "wire_diameter" in p else
        "tungsten_diameter" if "tungsten_diameter" in p else
        "electrode_diameter" if "electrode_diameter" in p else None
    )
    # If the operator pinned the consumable size, report THEIR value (not the
    # table-typical) for the matching consumable field.
    consumable_size = diameter_override if diameter_override is not None else (
        p.get(consumable_field) if consumable_field else None
    )

    summary = _build_summary(mat, thickness_mm, band, process_label, p, optimized,
                             computed_metrics, consumable_field, consumable_size,
                             user_fixed, override_warnings, consumable_change)

    return {
        "material":         mat,
        "material_display": mat.replace("_", " ").title(),
        "thickness_mm":     thickness_mm,
        "band":             band,
        "process":          process_label,
        "process_family":   process_family,
        "process_display":  process_meta.get("display_name", process_family),
        "arc_efficiency":   arc_efficiency,
        "deposition_efficiency": deposition_efficiency,
        "ranges":           ranges,
        "optimized":        optimized,
        "computed_metrics": computed_metrics,
        "objective_score":  round(best["objective"], 4),
        "efficiency_score": p.get("efficiency_score", 7),
        "sensitivity":      sensitivity,
        "gas_mix":          p.get("gas_mix"),
        "wire_diameter":    consumable_size if consumable_field == "wire_diameter" else p.get("wire_diameter"),
        "tungsten_diameter": consumable_size if consumable_field == "tungsten_diameter" else p.get("tungsten_diameter"),
        "electrode_diameter": consumable_size if consumable_field == "electrode_diameter" else p.get("electrode_diameter"),
        "electrode_type":   p.get("electrode_type"),
        "consumable_field": consumable_field,
        "notes":            p.get("notes", ""),
        "summary_text":     summary,

        # User-pinned parameters ("if I fix X, recompute the rest") + honest flags
        "overrides":          applied_overrides,   # {field: value} actually applied
        "bounds":             axis_bounds,         # {field: {op, value}} one-sided constraints applied
        "override_ranges":    axis_ranges,         # {field: (lo, hi)} axes pinned as a band (midpoint used)
        "user_fixed":         user_fixed,          # display-ready list of pinned params
        "override_warnings":  override_warnings,   # pinned values outside the standards window
        "consumable_change":  consumable_change,   # effect of a wire-diameter swap (or None)
        "defaulted":          {},                  # filled in by run_param_pipeline when material/thickness assumed

        # legacy flat aliases — kept so existing UI code paths don't break
        "params":           ranges,
        "optimal_current":  optimized["welding_current"],
        "optimal_voltage":  optimized["arc_voltage"],
        "optimal_speed":    optimized["welding_speed"],
        "optimal_wfr":      optimized["wire_feed_rate"],
        "deposition_rate_g_per_min": computed_metrics["deposition_rate_g_per_min"],
    }


def compare_to_optimal(live_readings: dict, recommendation: dict) -> dict:
    """
    Compare live sensor averages to the standards-window ranges.

    Parameters
    ----------
    live_readings  : dict with keys welding_current, arc_voltage, welding_speed,
                     wire_feed_rate, shielding_gas_flow (averaged over recent window)
    recommendation : output of recommend_parameters()

    Returns
    -------
    dict with per-parameter status: "ok" | "too_high" | "too_low" and deviation %
    """
    ranges = recommendation.get("ranges", {})
    results = {}
    for field, rng in ranges.items():
        actual = live_readings.get(field)
        if actual is None or rng is None:
            continue
        lo, hi = rng
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


# ── Parameter-override parsing ("if I fix X, what should the rest be?") ────────

# A user can pin one or more parameters and ask the advisor to recompute the
# others around them, e.g. "if I used 1.5 wire diameter then give me other values".
# Each pattern's first capturing group is the numeric value. Patterns are tried
# top-to-bottom and the matched span is masked out before the next pattern (and
# before thickness/material detection) so a "300 mm/min" speed is never mistaken
# for a "300 mm" thickness.
_DIAMETER_FIELDS = ("wire_diameter", "tungsten_diameter", "electrode_diameter")
_AXIS_FIELDS = ("welding_current", "arc_voltage", "welding_speed", "wire_feed_rate")

_OVERRIDE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Consumable diameter (mm-scale) — matched first so its "mm" isn't read as thickness.
    ("wire_diameter",       re.compile(r"(\d+(?:\.\d+)?)\s*mm\s+wire(?:\s*(?:diameter|dia|size))?\b")),
    ("wire_diameter",       re.compile(r"\bwire\s*(?:diameter|dia|size)?\s*(?:of|=|:|is|at|to)?\s*(\d+(?:\.\d+)?)\s*mm\b")),
    ("wire_diameter",       re.compile(r"(\d+(?:\.\d+)?)\s+wire\s*(?:diameter|dia|size)\b")),
    ("wire_diameter",       re.compile(r"\bwire\s*(?:diameter|dia|size)\s*(?:of|=|:|is|at|to)?\s*(\d+(?:\.\d+)?)")),
    # Bare "diameter"/"dia" with no "wire" word — common in a follow-up where "wire"
    # was established earlier in the conversation ("stay in 1.5 diameter").
    ("wire_diameter",       re.compile(r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:diameter|dia)\b")),
    ("wire_diameter",       re.compile(r"\b(?:diameter|dia)\s*(?:of|=|:|is|at|to)?\s*(\d+(?:\.\d+)?)\s*(?:mm)?\b")),
    ("electrode_diameter",  re.compile(r"(\d+(?:\.\d+)?)\s*mm\s+electrode\b")),
    ("electrode_diameter",  re.compile(r"\belectrode\s*(?:diameter|dia|size)\s*(?:of|=|:|is|at|to)?\s*(\d+(?:\.\d+)?)")),
    ("tungsten_diameter",   re.compile(r"\btungsten\s*(?:diameter|dia|size)?\s*(?:of|=|:|is|at|to)?\s*(\d+(?:\.\d+)?)\s*mm\b")),
    # Welding speed (mm/min) — matched before thickness so "mm/min" isn't read as thickness.
    ("welding_speed",       re.compile(r"(\d+(?:\.\d+)?)\s*mm\s*/\s*min")),
    ("welding_speed",       re.compile(r"\b(?:welding|travel)\s*speed\s*(?:of|=|:|is|at|to)?\s*(\d+(?:\.\d+)?)")),
    # Value-before-keyword speed ("use 450 speed") — the number precedes "speed" and
    # there is no compact unit token (the unit is mm/min), so the patterns above miss it.
    ("welding_speed",       re.compile(r"(\d+(?:\.\d+)?)\s*(?:mm\s*/\s*min\s*)?(?:welding\s*|travel\s*)?speed\b")),
    # Wire-feed rate (m/min).
    ("wire_feed_rate",      re.compile(r"(\d+(?:\.\d+)?)\s*m\s*/\s*min")),
    ("wire_feed_rate",      re.compile(r"\bwire\s*feed(?:\s*(?:rate|speed))?\s*(?:of|=|:|is|at|to)?\s*(\d+(?:\.\d+)?)")),
    # Current (A) and voltage (V).
    ("welding_current",     re.compile(r"(\d+(?:\.\d+)?)\s*(?:a|amp|amps|amperes|amperage)\b")),
    ("welding_current",     re.compile(r"\b(?:welding\s*)?current\s*(?:of|=|:|is|at|to)?\s*(\d+(?:\.\d+)?)")),
    ("arc_voltage",         re.compile(r"(\d+(?:\.\d+)?)\s*(?:v|volt|volts|voltage)\b")),
    ("arc_voltage",         re.compile(r"\b(?:arc\s*)?voltage\s*(?:of|=|:|is|at|to)?\s*(\d+(?:\.\d+)?)")),
]


# Two-sided RANGE phrasings ("use 400-500 speed", "current 200 to 250", "24-26 V").
# Each regex captures TWO numbers (low, high); the advisor pins the axis to the band
# MIDPOINT and remembers [low, high] for display ("400–500 mm/min (≈450)"). Tried
# before the single-value patterns so "400-500 speed" is read as a band, not as a
# stray "500 speed". Speed patterns precede wire-feed so a "400-500 mm/min" travel
# speed is never mis-parsed through the trailing "m/min".
_RANGE_SEP = r"(?:\s*(?:-|–|—|to|\.\.)\s*)"
_RANGE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Welding speed (mm/min) — value-before-keyword, bare mm/min tail, or keyword-first.
    ("welding_speed",   re.compile(
        rf"(\d+(?:\.\d+)?){_RANGE_SEP}(\d+(?:\.\d+)?)\s*(?:mm\s*/\s*min\s*)?(?:welding\s*|travel\s*)?speed\b")),
    ("welding_speed",   re.compile(
        rf"(\d+(?:\.\d+)?){_RANGE_SEP}(\d+(?:\.\d+)?)\s*mm\s*/\s*min")),
    ("welding_speed",   re.compile(
        rf"\b(?:welding\s*|travel\s*)?speed\s*(?:of|=|:|is|at|to|between|from|in)?\s*(\d+(?:\.\d+)?){_RANGE_SEP}(\d+(?:\.\d+)?)")),
    # Current (A).
    ("welding_current", re.compile(
        rf"(\d+(?:\.\d+)?){_RANGE_SEP}(\d+(?:\.\d+)?)\s*(?:a|amp|amps|amperes|amperage)\b")),
    ("welding_current", re.compile(
        rf"\b(?:welding\s*)?current\s*(?:of|=|:|is|at|to|between|from|in)?\s*(\d+(?:\.\d+)?){_RANGE_SEP}(\d+(?:\.\d+)?)")),
    # Voltage (V).
    ("arc_voltage",     re.compile(
        rf"(\d+(?:\.\d+)?){_RANGE_SEP}(\d+(?:\.\d+)?)\s*(?:v|volt|volts|voltage)\b")),
    ("arc_voltage",     re.compile(
        rf"\b(?:arc\s*)?voltage\s*(?:of|=|:|is|at|to|between|from|in)?\s*(\d+(?:\.\d+)?){_RANGE_SEP}(\d+(?:\.\d+)?)")),
    # Wire-feed rate (m/min).
    ("wire_feed_rate",  re.compile(
        rf"(\d+(?:\.\d+)?){_RANGE_SEP}(\d+(?:\.\d+)?)\s*m\s*/\s*min")),
]


# Phrases that introduce the value being *replaced*, not a new value to apply,
# e.g. "use 1.5 mm wire instead of 1.2 mm". Masked first so the replaced number is
# never read as a plate thickness or a second override.
_REPLACED_VALUE_RE = re.compile(
    r"\b(?:instead\s+of|rather\s+than|in\s+place\s+of|not|vs\.?|versus)\s+(\d+(?:\.\d+)?)\s*(?:mm)?",
)

# Comparator words that turn a pinned value into a one-sided constraint
# ("more than 21 V" -> voltage >= 21, "under 25 V" -> voltage <= 25). The grid
# search then honours the bound instead of pinning the value exactly. Each regex
# must match at the END of the text preceding the number (only whitespace between),
# and word-bounded, so "recover"/"aluminum" never trip "over"/"min".
_MIN_RE = re.compile(
    r"(?:\b(?:more than|greater than|at least|no less than|not less than|"
    r"above|over|minimum|min)\b|>=|=>|>|≥)\s*$"
)
_MAX_RE = re.compile(
    r"(?:\b(?:less than|no more than|not more than|at most|up to|"
    r"below|under|maximum|max)\b|<=|=<|<|≤)\s*$"
)


def _detect_comparator(text_before: str) -> str | None:
    """Return 'min' / 'max' if a comparator immediately precedes a value, else None."""
    tail = text_before[-30:]
    if _MIN_RE.search(tail):
        return "min"
    if _MAX_RE.search(tail):
        return "max"
    return None


def _extract_overrides(q: str) -> tuple[dict, dict, str, dict]:
    """
    Pull user-pinned parameter values out of a (lowercased) question.

    Returns (overrides, bounds, masked_q, override_ranges):
      - overrides : {field: value} pinned to an exact value. A ranged request
                    ("400-500 speed") is pinned to the band MIDPOINT here.
      - bounds    : {field: {"op": "min"|"max", "value": x}} one-sided constraints
                    parsed from comparator words ("more than 21 voltage").
      - masked_q  : the question with every matched span (and any "instead of X"
                    replaced-value) blanked, so the caller can run thickness/material
                    detection without an override's units being mis-read as thickness.
      - override_ranges : {field: (low, high)} the original band for any axis given
                    as a range — display-only, so the answer can say "400–500 (≈450)".
    """
    # Blank out replaced values ("instead of 1.2 mm") up front so they neither
    # become a spurious thickness nor a second override.
    masked = _REPLACED_VALUE_RE.sub(lambda m: " " * (m.end() - m.start()), q)

    overrides: dict[str, float] = {}
    bounds: dict[str, dict] = {}
    override_ranges: dict[str, tuple] = {}

    # --- Range pre-pass: "400-500 speed" / "200 to 250 A" -> pin midpoint, keep band ---
    for field, pat in _RANGE_PATTERNS:
        if field in overrides or field in bounds:
            continue
        m = pat.search(masked)
        if not m:
            continue
        try:
            a, b = float(m.group(1)), float(m.group(2))
        except (TypeError, ValueError):
            continue
        lo, hi = (a, b) if a <= b else (b, a)
        overrides[field] = round((lo + hi) / 2.0, 4)
        override_ranges[field] = (lo, hi)
        masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]

    # --- Single-value pass (exact pins + comparator bounds) ---
    for field, pat in _OVERRIDE_PATTERNS:
        if field in overrides or field in bounds:
            continue
        m = pat.search(masked)
        if not m:
            continue
        try:
            value = float(m.group(1))
        except (TypeError, ValueError):
            continue
        op = _detect_comparator(masked[:m.start()])
        if op is not None:
            bounds[field] = {"op": op, "value": value}
        else:
            overrides[field] = value
        masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
    return overrides, bounds, masked, override_ranges


def parse_param_overrides(question: str) -> dict:
    """Public helper: return the dict of user-pinned parameter values (may be empty)."""
    return _extract_overrides(question.lower())[0]


def parse_param_bounds(question: str) -> dict:
    """Public helper: return the dict of one-sided constraints (may be empty)."""
    return _extract_overrides(question.lower())[1]


def parse_param_ranges(question: str) -> dict:
    """Public helper: return the dict of axis bands given as a range (may be empty)."""
    return _extract_overrides(question.lower())[3]


# ── NL query parsing ──────────────────────────────────────────────────────────

# Parameter-adjustment follow-up phrases ("what is the change in current and
# voltage", "if I weld faster"). Kept parameter-specific so a non-welding sentence
# doesn't bypass the intent guard via the param route. Unlike the generic triggers,
# one of these alone is enough to qualify a query as a parameter request even with
# no material/thickness named — a follow-up inherits those from conversation context.
_ADJUSTMENT_TRIGGERS = (
    "change in current", "change in voltage", "change in speed",
    "change the current", "change the voltage", "change the speed",
    "change in wire", "change in heat input",
    "faster speed", "slower speed", "higher speed", "lower speed",
    "increase speed", "decrease speed", "increase the speed", "decrease the speed",
)


def parse_param_query(question: str) -> dict | None:
    """
    Extract material, thickness, and process from a natural-language parameter query.
    Returns None if the question doesn't look like a parameter request.

    Examples:
      "best settings for 5mm mild steel using MIG" ->
          {material: "mild_steel", thickness_mm: 5.0, process: "MIG"}
      "what speed for 3mm aluminum with TIG"        ->
          {material: "aluminum", thickness_mm: 3.0, process: "TIG"}
      "optimal current for 10mm stainless"          ->
          {material: "stainless_steel", thickness_mm: 10.0, process: None}
    """
    q = question.lower()

    # Pull any pinned parameter values first. Their presence both (a) qualifies
    # the question as a parameter request even with no explicit material/thickness
    # ("if I used 1.5 wire diameter, give me the other values") and (b) lets us run
    # thickness/material detection on a masked string so an override's units aren't
    # mistaken for a plate thickness.
    overrides, bounds, masked_q, override_ranges = _extract_overrides(q)

    param_triggers = [
        "best setting", "optimal setting", "recommend setting",
        "what speed", "what current", "what voltage", "what parameter",
        "best parameter", "optimal parameter", "best speed",
        "best current", "best voltage", "best wire", "best gas",
        "optimal current", "optimal speed", "optimal voltage",
        "optimal feed", "optimal gas",
        "how fast", "what feed", "optimal weld",
        "travel speed", "wire feed speed", "feed speed",
        "wire diameter", "electrode diameter", "settings for",
        # "if I change/use X, what are the other values" style requests
        "other value", "other parameter", "other setting", "remaining",
        "rest of", "adjust", "recalculate", "recompute",
    ] + list(_ADJUSTMENT_TRIGGERS)
    has_adjustment = any(t in q for t in _ADJUSTMENT_TRIGGERS)
    if not overrides and not bounds and not any(t in q for t in param_triggers):
        return None

    # Extract thickness (number followed by mm) — from the masked string so a
    # pinned "300 mm/min" speed or "1.5 mm wire" isn't read as the plate thickness.
    thickness_match = re.search(r"(\d+(?:\.\d+)?)\s*mm", masked_q)
    if not thickness_match:
        thickness_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:millimetre|millimeter)", masked_q)
    thickness_mm = float(thickness_match.group(1)) if thickness_match else None

    # Extract material. Short aliases (e.g. "al", "ss", "ms") use word boundaries
    # so they don't match inside unrelated words (e.g. "al" inside "optimal").
    material = None
    for alias in sorted(MATERIAL_ALIASES.keys(), key=len, reverse=True):
        if len(alias) <= 3:
            if re.search(rf"\b{re.escape(alias)}\b", masked_q):
                material = MATERIAL_ALIASES[alias]
                break
        elif alias in masked_q:
            material = MATERIAL_ALIASES[alias]
            break

    # If the user explicitly names a metal we DON'T support, pass that token
    # through so recommend_parameters() returns an honest "not supported" error
    # instead of silently defaulting to mild steel and fabricating numbers.
    if material is None:
        for metal in UNSUPPORTED_MATERIALS:
            if re.search(rf"\b{re.escape(metal)}\b", masked_q):
                material = metal
                break

    # Nothing to anchor on (no material, no thickness, no pinned parameter)? Not a
    # parameter query — UNLESS it's a clear parameter-adjustment follow-up ("what is
    # the change in current and voltage"), which inherits material/thickness from the
    # conversation context downstream.
    if (material is None and thickness_mm is None
            and not overrides and not bounds and not has_adjustment):
        return None

    return {
        "material":     material or "mild_steel",
        "thickness_mm": thickness_mm or 5.0,
        "process":      _detect_process(masked_q),   # None -> recommend_parameters() uses the default (GMAW)
        "overrides":    overrides,                    # user-pinned exact parameter values (may be empty)
        "bounds":       bounds,                       # one-sided constraints, e.g. voltage >= 21 (may be empty)
        "override_ranges": override_ranges,           # axes given as a band, e.g. speed 400-500 (display-only)
        "defaulted":    {                             # which anchors we had to assume
            "material":  material is None,
            "thickness": thickness_mm is None,
        },
    }


def merge_param_context(parsed: dict, prior: dict | None) -> dict:
    """
    Carry conversation state forward into a follow-up parameter query.

    A follow-up like "now stay in 1.5 diameter and more than 21 voltage" rarely
    restates the material/thickness — those were established earlier. This fills
    any anchor the current turn left unspecified from `prior`, and accumulates
    pinned parameters/constraints across turns (so the wire diameter set two turns
    ago is still in force).

    `prior` shape (also what run_param_pipeline persists back as `param_context`):
        {material, thickness_mm, process, overrides, bounds}

    Returns a new parsed dict (material/thickness_mm/process/overrides/bounds/
    defaulted/carried). `carried` is a display-ready list of what was inherited.
    """
    if not prior:
        parsed.setdefault("bounds", {})
        parsed.setdefault("override_ranges", {})
        parsed["carried"] = []
        return parsed

    defaulted = parsed.get("defaulted", {})
    new_material = not defaulted.get("material", False)
    new_thickness = not defaulted.get("thickness", False)

    prior_material = prior.get("material")
    prior_thickness = prior.get("thickness_mm")

    # The operator naming a *different* material/thickness starts a fresh scenario,
    # so previously-pinned values should not silently bleed into it.
    scenario_changed = (
        (new_material and prior_material and parsed["material"] != prior_material)
        or (new_thickness and prior_thickness and parsed["thickness_mm"] != prior_thickness)
    )

    carried: list[str] = []
    material = parsed["material"]
    thickness = parsed["thickness_mm"]
    process = parsed.get("process")

    if not new_material and prior_material:
        material = prior_material
        carried.append(f"material: {prior_material.replace('_', ' ')}")
    if not new_thickness and prior_thickness:
        thickness = prior_thickness
        carried.append(f"thickness: {prior_thickness} mm")
    if process is None and prior.get("process"):
        process = prior["process"]
        carried.append(f"process: {process}")

    base_overrides = {} if scenario_changed else dict(prior.get("overrides", {}))
    base_bounds = {} if scenario_changed else dict(prior.get("bounds", {}))
    if not scenario_changed:
        for field, val in base_overrides.items():
            if field not in parsed.get("overrides", {}) and field not in parsed.get("bounds", {}):
                label, unit = _PARAM_LABELS.get(field, (field, ""))
                carried.append(f"{label.lower()}: {val} {unit}".strip())

    overrides = {**base_overrides, **parsed.get("overrides", {})}
    bounds = {**base_bounds, **parsed.get("bounds", {})}
    # A field newly given as an exact value drops any stale bound on it, and vice versa.
    for field in parsed.get("overrides", {}):
        bounds.pop(field, None)
    for field in parsed.get("bounds", {}):
        overrides.pop(field, None)

    # Carry the display band for any range-pinned axis; a field re-pinned this turn
    # as a plain value or a bound drops its stale band.
    base_ranges = {} if scenario_changed else dict(prior.get("override_ranges", {}))
    new_ranges = parsed.get("override_ranges", {})
    override_ranges = {**base_ranges, **new_ranges}
    for field in parsed.get("overrides", {}):
        if field not in new_ranges:
            override_ranges.pop(field, None)
    for field in parsed.get("bounds", {}):
        override_ranges.pop(field, None)
    # A band only makes sense while its midpoint is still the active override.
    override_ranges = {f: r for f, r in override_ranges.items() if f in overrides}

    return {
        "material":     material,
        "thickness_mm": thickness,
        "process":      process,
        "overrides":    overrides,
        "bounds":       bounds,
        "override_ranges": override_ranges,
        "defaulted": {
            "material":  defaulted.get("material", False) and not prior_material,
            "thickness": defaulted.get("thickness", False) and not prior_thickness,
        },
        "carried":      carried,
    }


def _detect_process(q: str) -> str | None:
    """Detect a welding-process mention in a (lowercased) question string."""
    for alias, family in sorted(_process_alias_map().items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return alias.upper() if len(alias) <= 5 else family
    return None


# ── User-pinned-parameter reporting ───────────────────────────────────────────

_PARAM_LABELS = {
    "welding_current": ("Current", "A"),
    "arc_voltage":     ("Voltage", "V"),
    "welding_speed":   ("Welding speed", "mm/min"),
    "wire_feed_rate":  ("Wire feed", "m/min"),
    "wire_diameter":   ("Wire diameter", "mm"),
}


def _fmt_num(x: float) -> str:
    """Compact number for display: drop a trailing .0 (450.0 -> '450', 1.5 -> '1.5')."""
    return str(int(x)) if float(x).is_integer() else str(round(x, 2))


def _user_fixed_list(p: dict, axis_overrides: dict, axis_bounds: dict,
                     diameter_override, axis_ranges: dict | None = None) -> list[dict]:
    """Display-ready list of the parameters the operator pinned or bounded (stable order)."""
    axis_ranges = axis_ranges or {}
    fixed = []
    for field in _AXIS_FIELDS:
        if field in axis_overrides:
            label, unit = _PARAM_LABELS[field]
            val = round(axis_overrides[field], 2)
            if field in axis_ranges:
                lo, hi = axis_ranges[field]
                disp = f"{_fmt_num(lo)}–{_fmt_num(hi)} (≈{_fmt_num(val)})"
                fixed.append({"field": field, "label": label, "unit": unit,
                              "value": val, "op": "range", "range": [lo, hi],
                              "display_value": disp})
            else:
                fixed.append({"field": field, "label": label, "unit": unit,
                              "value": val, "op": None, "display_value": f"{_fmt_num(val)}"})
        elif field in axis_bounds:
            label, unit = _PARAM_LABELS[field]
            b = axis_bounds[field]
            val = round(b["value"], 2)
            sym = "≥" if b["op"] == "min" else "≤"
            fixed.append({"field": field, "label": label, "unit": unit,
                          "value": val, "op": b["op"], "display_value": f"{sym} {val}"})
    if diameter_override is not None:
        label, unit = _PARAM_LABELS["wire_diameter"]
        val = round(diameter_override, 2)
        fixed.append({"field": "wire_diameter", "label": label, "unit": unit,
                      "value": val, "op": None, "display_value": f"{val}"})
    return fixed


def _consumable_change(p: dict, table_diameter, diameter_override, best,
                       deposition_efficiency, density) -> dict | None:
    """
    Quantify the effect of swapping wire diameter (e.g. 1.2 mm -> 1.5 mm) on a
    wire-fed process.

    The current / voltage / travel-speed windows are fixed by the joint and do not
    change with wire size. What changes is deposition per unit wire-feed, so this
    returns the wire-feed speed that holds the SAME deposit as the table-standard
    wire — a concrete, computed "this is the parameter that changes" answer.

    Returns None for non-wire-fed processes (TIG/SMAW) or when the diameter wasn't
    changed.
    """
    wfr = best.get("wire_feed_rate")
    if wfr is None or diameter_override is None or not table_diameter:
        return None
    if abs(diameter_override - table_diameter) < 1e-9:
        return None

    # deposition_rate in `best` was already computed with the NEW wire diameter.
    deposition_new_wire = best.get("deposition_rate")
    # Same wire-feed with the standard wire would deposit (d_std/d_new)^2 as much.
    ratio = (table_diameter / diameter_override) ** 2
    deposition_standard_wire = (deposition_new_wire * ratio
                                if deposition_new_wire is not None else None)
    # To hold the standard deposit with the new wire, scale wire-feed by the ratio.
    wfr_same_deposition = round(wfr * ratio, 1)

    return {
        "table_diameter":            table_diameter,
        "new_diameter":              diameter_override,
        "optimized_wire_feed":       round(wfr, 1),
        "deposition_at_optimized_wire_feed": _round_or_none(deposition_new_wire),
        "deposition_standard_wire":  _round_or_none(deposition_standard_wire),
        "wire_feed_for_same_deposition": wfr_same_deposition,
        "bigger_wire":               diameter_override > table_diameter,
    }


def _override_warnings(p: dict, axis_overrides: dict, axis_bounds: dict,
                       diameter_override, table_diameter,
                       axis_ranges: dict | None = None) -> list[str]:
    """Flag any pinned value or constraint that falls outside the standards window."""
    axis_ranges = axis_ranges or {}
    warnings = []
    for field in _AXIS_FIELDS:
        rng = p.get(field)
        label, unit = _PARAM_LABELS[field]
        if field in axis_overrides:
            if not rng:
                continue
            val = axis_overrides[field]
            if val < rng[0] or val > rng[1]:
                if field in axis_ranges:
                    lo, hi = axis_ranges[field]
                    side = "above" if val > rng[1] else "below"
                    warnings.append(
                        f"{label} {_fmt_num(lo)}–{_fmt_num(hi)} {unit} is {side} the standards "
                        f"window ({rng[0]}-{rng[1]} {unit}) for this material/thickness/process — "
                        f"the other values were computed at the band midpoint (~{_fmt_num(val)} "
                        f"{unit}), but verify against your WPS."
                    )
                else:
                    warnings.append(
                        f"{label} {round(val, 2)} {unit} is outside the standards window "
                        f"({rng[0]}-{rng[1]} {unit}) for this material/thickness/process — "
                        f"the other values were still computed around it, but verify against your WPS."
                    )
        elif field in axis_bounds and rng:
            b = axis_bounds[field]
            val = round(b["value"], 2)
            if b["op"] == "min" and val > rng[1]:
                warnings.append(
                    f"{label} ≥ {val} {unit} forces the value above the standards window "
                    f"({rng[0]}-{rng[1]} {unit}) for this combination — the result was "
                    f"clamped to your limit, but verify against your WPS."
                )
            elif b["op"] == "max" and val < rng[0]:
                warnings.append(
                    f"{label} ≤ {val} {unit} forces the value below the standards window "
                    f"({rng[0]}-{rng[1]} {unit}) for this combination — the result was "
                    f"clamped to your limit, but verify against your WPS."
                )
    if diameter_override is not None and table_diameter and diameter_override != table_diameter:
        warnings.append(
            f"Wire diameter {round(diameter_override, 2)} mm differs from the "
            f"table-typical {table_diameter} mm for this combination — deposition rate "
            f"was recomputed for your wire size."
        )
    return warnings


# ── Summary text (deterministic non-LLM fallback) ─────────────────────────────

def _build_summary(mat, thickness_mm, band, process_label, p, optimized,
                   computed_metrics, consumable_field, consumable_size,
                   user_fixed=None, override_warnings=None,
                   consumable_change=None) -> str:
    mat_label = mat.replace("_", " ").title()
    lines = []
    if user_fixed:
        fixed_str = ", ".join(
            f"{f['label']} = {f.get('display_value', f['value'])} {f['unit']}" for f in user_fixed
        )
        lines.append(f"You fixed: {fixed_str}. Other values recomputed around it:")
    lines += [
        f"Optimized {process_label} parameters for {mat_label} ({thickness_mm} mm — {band}):",
        f"  Current:     {optimized['welding_current']} A   "
        f"(window {p['welding_current'][0]}-{p['welding_current'][1]} A)",
        f"  Voltage:     {optimized['arc_voltage']} V   "
        f"(window {p['arc_voltage'][0]}-{p['arc_voltage'][1]} V)",
        f"  Speed:       {optimized['welding_speed']} mm/min   "
        f"(window {p['welding_speed'][0]}-{p['welding_speed'][1]} mm/min)",
    ]
    if optimized["wire_feed_rate"] is not None:
        wfr_range = p.get("wire_feed_rate", [None, None])
        lines.append(
            f"  Wire feed:   {optimized['wire_feed_rate']} m/min   "
            f"(window {wfr_range[0]}-{wfr_range[1]} m/min)"
        )
    else:
        lines.append("  Wire feed:   N/A for this process")

    if p.get("shielding_gas_flow"):
        g = p["shielding_gas_flow"]
        lines.append(f"  Gas flow:    {g[0]}-{g[1]} L/min  ({p.get('gas_mix','-')})")
    else:
        lines.append("  Shielding gas: N/A for this process")

    if consumable_field and consumable_size:
        label = {"wire_diameter": "Wire diameter", "tungsten_diameter": "Tungsten diameter",
                 "electrode_diameter": "Electrode diameter"}.get(consumable_field, "Consumable size")
        extra = f" ({p['electrode_type']})" if consumable_field == "electrode_diameter" and p.get("electrode_type") else ""
        lines.append(f"  {label}: {consumable_size} mm{extra}")

    lines += [
        f"  Heat input:  {computed_metrics['heat_input_kj_per_mm']} kJ/mm   "
        f"(standards window {computed_metrics['heat_input_range'][0]}-{computed_metrics['heat_input_range'][1]} kJ/mm)",
    ]
    if computed_metrics["deposition_rate_g_per_min"] is not None:
        lines.append(f"  Deposition:  ~{computed_metrics['deposition_rate_g_per_min']} g/min")
    lines += [
        f"  Efficiency score: {p.get('efficiency_score', 7)}/10",
        f"  Note: {p.get('notes', '')}",
    ]
    if consumable_change:
        cc = consumable_change
        verb = "bigger" if cc["bigger_wire"] else "smaller"
        lines += [
            "",
            f"Switching {cc['table_diameter']} mm -> {cc['new_diameter']} mm wire "
            f"({verb} wire):",
            f"  - Current / voltage / travel-speed windows are unchanged — they are set "
            f"by the {thickness_mm} mm joint, not the wire size.",
            f"  - At the same {cc['optimized_wire_feed']} m/min wire feed you now deposit "
            f"~{cc['deposition_at_optimized_wire_feed']} g/min "
            f"(the {cc['table_diameter']} mm wire gives ~{cc['deposition_standard_wire']} g/min).",
            f"  - To hold the same deposit with the {cc['new_diameter']} mm wire, set wire "
            f"feed to ~{cc['wire_feed_for_same_deposition']} m/min. Confirm arc stability "
            f"against your WPS.",
        ]
    for w in (override_warnings or []):
        lines.append(f"  ⚠ {w}")
    return "\n".join(lines)
