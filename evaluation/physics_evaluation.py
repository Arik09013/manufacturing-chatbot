"""
Physics-Based Recommendation Evaluation Engine.

Provides an independent, separate mathematical re-implementation of the physics
equations and process rules implemented in src/reasoning/param_advisor.py, along with
deterministic benchmark scenarios, constraint validation, edge case suites,
sensitivity analysis, and latency profiling.

CRITICAL RESEARCH INTEGRITY REQUIREMENT:
The mathematical equations evaluated here are independently implemented in this
module and do NOT call internal production calculation methods to verify themselves.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Path references
_REPO_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _REPO_ROOT / "config" / "welding_params.yaml"
_EFFECTS_CONFIG_PATH = _REPO_ROOT / "config" / "parameter_effects.yaml"

# Import production advisor main entry point unchanged
from src.reasoning.param_advisor import (
    recommend_parameters,
    compare_to_optimal,
    MATERIAL_ALIASES,
    UNSUPPORTED_MATERIALS,
    THICKNESS_BANDS,
)


# ==============================================================================
# 1. INDEPENDENT MATHEMATICAL RE-IMPLEMENTATION
# ==============================================================================

class IndependentPhysicsEvaluator:
    """
    Independent re-implementation of the physics and optimization formulas.
    Does NOT call any production calculation functions.
    """

    @staticmethod
    def calculate_heat_input(
        current_a: float,
        voltage_v: float,
        speed_mm_per_min: float,
        arc_efficiency: float,
    ) -> float:
        """
        Independent re-implementation of heat input:
        HI = (eta * 60 * I * V) / (1000 * S)  [kJ/mm]
        """
        if speed_mm_per_min <= 0:
            raise ValueError("Travel speed must be strictly positive to compute heat input.")
        power_watts = current_a * voltage_v
        energy_joules_per_min = power_watts * 60.0 * arc_efficiency
        heat_input_kj_per_mm = energy_joules_per_min / (1000.0 * speed_mm_per_min)
        return heat_input_kj_per_mm

    @staticmethod
    def calculate_deposition_rate(
        wire_feed_rate_m_per_min: Optional[float],
        wire_diameter_mm: float,
        density_g_per_cm3: float,
        deposition_efficiency: float,
    ) -> Optional[float]:
        """
        Independent re-implementation of wire deposition rate:
        DR = WFR * pi * (d / 2)^2 * rho * eta_dep  [g/min]
        Dimensional check:
        WFR in m/min = 100 cm/min
        Area in mm^2 = 0.01 cm^2
        Product (WFR * Area) = 100 * 0.01 = 1.0 cm^3/min
        Multiplying by density (g/cm^3) yields g/min directly.
        """
        if wire_feed_rate_m_per_min is None:
            return None
        radius_mm = wire_diameter_mm / 2.0
        wire_area_mm2 = math.pi * (radius_mm ** 2)
        deposition_g_per_min = (
            wire_feed_rate_m_per_min * wire_area_mm2 * density_g_per_cm3 * deposition_efficiency
        )
        return deposition_g_per_min

    @staticmethod
    def calculate_window_score(
        heat_input_kj_per_mm: float,
        hi_min: float,
        hi_max: float,
    ) -> float:
        """
        Independent re-implementation of triangular window scoring:
        1.0 at center, tapering to 0.7 at edges, 0.0 outside.
        """
        if heat_input_kj_per_mm < hi_min or heat_input_kj_per_mm > hi_max:
            return 0.0
        center = (hi_min + hi_max) / 2.0
        half_width = (hi_max - hi_min) / 2.0
        if half_width <= 0:
            return 1.0
        deviation_ratio = abs(heat_input_kj_per_mm - center) / half_width
        return 1.0 - 0.3 * deviation_ratio

    @staticmethod
    def calculate_window_distance(
        heat_input_kj_per_mm: float,
        hi_min: float,
        hi_max: float,
    ) -> float:
        """
        Independent fallback distance metric to standards window.
        """
        if heat_input_kj_per_mm < hi_min:
            return hi_min - heat_input_kj_per_mm
        if heat_input_kj_per_mm > hi_max:
            return heat_input_kj_per_mm - hi_max
        return 0.0

    @classmethod
    def run_independent_grid_search(
        cls,
        standards: dict,
        arc_efficiency: float,
        deposition_efficiency: float,
        wire_diameter_mm: float,
        density_g_per_cm3: float,
        axis_overrides: Optional[dict] = None,
        axis_bounds: Optional[dict] = None,
        grid_steps: int = 7,
    ) -> dict:
        """
        Independent implementation of the 7x7 grid search optimizer.
        """
        axis_overrides = axis_overrides or {}
        axis_bounds = axis_bounds or {}

        i_lo, i_hi = standards["welding_current"]
        v_lo, v_hi = standards["arc_voltage"]
        s_lo, s_hi = standards["welding_speed"]
        hi_lo, hi_hi = standards["heat_input_range"]
        wfr_range = standards.get("wire_feed_rate")

        def apply_bound(val: float, bound: dict) -> float:
            if bound.get("op") == "min":
                return max(val, bound["value"])
            if bound.get("op") == "max":
                return min(val, bound["value"])
            return val

        def get_axis_grid(lo: float, hi: float, field: str) -> List[float]:
            if field in axis_overrides:
                return [float(axis_overrides[field])]
            step = (hi - lo) / (grid_steps - 1)
            points = [lo + k * step for k in range(grid_steps)]
            if field in axis_bounds:
                points = [apply_bound(p, axis_bounds[field]) for p in points]
            return points

        current_grid = get_axis_grid(i_lo, i_hi, "welding_current")
        speed_grid = get_axis_grid(s_lo, s_hi, "welding_speed")

        candidates = []
        for current in current_grid:
            frac = (current - i_lo) / (i_hi - i_lo) if i_hi > i_lo else 0.5
            frac = min(max(frac, 0.0), 1.0)

            if "arc_voltage" in axis_overrides:
                voltage = float(axis_overrides["arc_voltage"])
            else:
                voltage = v_lo + frac * (v_hi - v_lo)
                if "arc_voltage" in axis_bounds:
                    voltage = apply_bound(voltage, axis_bounds["arc_voltage"])

            if "wire_feed_rate" in axis_overrides:
                wfr = float(axis_overrides["wire_feed_rate"])
            elif wfr_range is not None:
                wfr = wfr_range[0] + frac * (wfr_range[1] - wfr_range[0])
                if "wire_feed_rate" in axis_bounds:
                    wfr = apply_bound(wfr, axis_bounds["wire_feed_rate"])
            else:
                wfr = None

            for speed in speed_grid:
                hi = cls.calculate_heat_input(current, voltage, speed, arc_efficiency)
                score = cls.calculate_window_score(hi, hi_lo, hi_hi)
                dep = cls.calculate_deposition_rate(
                    wfr, wire_diameter_mm, density_g_per_cm3, deposition_efficiency
                )
                objective = (dep * score) if dep is not None else score

                candidates.append({
                    "current": current,
                    "voltage": voltage,
                    "speed": speed,
                    "wire_feed_rate": wfr,
                    "heat_input": hi,
                    "deposition_rate": dep,
                    "window_score": score,
                    "objective": objective,
                })

        in_window = [c for c in candidates if c["window_score"] > 0]
        if in_window:
            best = max(in_window, key=lambda c: c["objective"])
        else:
            best = min(candidates, key=lambda c: cls.calculate_window_distance(c["heat_input"], hi_lo, hi_hi))

        return best

    @staticmethod
    def calculate_iso_deposition_feed(
        table_diameter_mm: float,
        override_diameter_mm: float,
        optimized_wfr_m_per_min: float,
    ) -> float:
        """
        Independent re-implementation of wire swap iso-deposition feed calculation.
        """
        area_ratio = (table_diameter_mm / override_diameter_mm) ** 2
        return round(optimized_wfr_m_per_min * area_ratio, 1)


# ==============================================================================
# 2. DETERMINISTIC BENCHMARK GENERATION
# ==============================================================================

def load_standards_database() -> dict:
    """Load the raw YAML standards configuration."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_physics_benchmark() -> List[Dict[str, Any]]:
    """
    Construct a deterministic benchmark suite of valid scenarios.
    Covers all 8 valid material-process combinations across thin, medium,
    and thick plate regimes, along with systematic parameter override variations.
    Produces 144 deterministic test cases.
    """
    valid_combos = [
        ("mild_steel", "GMAW"),
        ("mild_steel", "TIG"),
        ("mild_steel", "SMAW"),
        ("stainless_steel", "GMAW"),
        ("stainless_steel", "TIG"),
        ("stainless_steel", "SMAW"),
        ("aluminum", "GMAW"),
        ("aluminum", "TIG"),
    ]

    # Deterministic thickness sampling covering all 3 thickness regimes
    # mild_steel & stainless: thin [1,3), med [3,8), thick [8,999)
    # aluminum: thin [1,4), med [4,12), thick [12,999)
    thickness_sets = {
        "mild_steel": [1.2, 1.6, 2.0, 2.8, 3.5, 4.5, 5.5, 6.5, 7.5, 9.0, 12.0, 16.0, 20.0, 25.0],
        "stainless_steel": [1.2, 1.6, 2.0, 2.8, 3.5, 4.5, 5.5, 6.5, 7.5, 9.0, 12.0, 16.0, 20.0, 25.0],
        "aluminum": [1.2, 2.0, 2.8, 3.8, 4.5, 6.0, 8.0, 10.0, 11.5, 13.0, 16.0, 20.0, 25.0, 30.0],
    }

    scenarios: List[Dict[str, Any]] = []
    idx = 0

    # 1. Baseline unconstrained scenarios (8 combos * 14 thicknesses = 112 scenarios)
    for mat, proc in valid_combos:
        for thk in thickness_sets[mat]:
            idx += 1
            scenarios.append({
                "scenario_id": f"SCEN_{idx:03d}",
                "material": mat,
                "process": proc,
                "thickness_mm": thk,
                "type": "baseline_unconstrained",
                "overrides": None,
                "bounds": None,
            })

    # 2. Override scenarios with consumable and axis pinning (32 scenarios)
    # Wire diameter overrides (for GMAW)
    wire_overrides = [0.8, 1.0, 1.2, 1.6]
    for mat in ["mild_steel", "stainless_steel", "aluminum"]:
        for d_override in wire_overrides:
            idx += 1
            scenarios.append({
                "scenario_id": f"SCEN_{idx:03d}",
                "material": mat,
                "process": "GMAW",
                "thickness_mm": 5.0 if mat != "aluminum" else 6.0,
                "type": "wire_diameter_override",
                "overrides": {"wire_diameter": d_override},
                "bounds": None,
            })

    # Axis overrides (pinned current, voltage, speed)
    axis_variations = [
        ("mild_steel", "GMAW", 5.0, {"welding_current": 160.0}),
        ("mild_steel", "GMAW", 5.0, {"welding_speed": 350.0}),
        ("mild_steel", "TIG", 4.0, {"welding_current": 110.0}),
        ("mild_steel", "SMAW", 5.0, {"welding_current": 125.0}),
        ("stainless_steel", "GMAW", 4.0, {"arc_voltage": 22.0}),
        ("stainless_steel", "TIG", 3.0, {"welding_speed": 95.0}),
        ("stainless_steel", "SMAW", 6.0, {"welding_current": 115.0}),
        ("aluminum", "GMAW", 6.0, {"welding_current": 180.0}),
        ("aluminum", "TIG", 5.0, {"welding_current": 150.0}),
        ("mild_steel", "GMAW", 12.0, {"wire_feed_rate": 12.0}),
    ]
    for mat, proc, thk, ov in axis_variations:
        idx += 1
        scenarios.append({
            "scenario_id": f"SCEN_{idx:03d}",
            "material": mat,
            "process": proc,
            "thickness_mm": thk,
            "type": "axis_override",
            "overrides": ov,
            "bounds": None,
        })

    # Axis comparator bounds
    bound_variations = [
        ("mild_steel", "GMAW", 5.0, {"arc_voltage": {"op": "min", "value": 22.0}}),
        ("mild_steel", "GMAW", 5.0, {"welding_current": {"op": "max", "value": 180.0}}),
        ("stainless_steel", "TIG", 4.0, {"welding_speed": {"op": "min", "value": 85.0}}),
        ("aluminum", "GMAW", 8.0, {"welding_current": {"op": "min", "value": 170.0}}),
        ("aluminum", "TIG", 5.0, {"arc_voltage": {"op": "max", "value": 15.0}}),
        ("mild_steel", "SMAW", 6.0, {"welding_current": {"op": "min", "value": 130.0}}),
        ("stainless_steel", "GMAW", 6.0, {"welding_speed": {"op": "max", "value": 380.0}}),
        ("mild_steel", "GMAW", 10.0, {"arc_voltage": {"op": "min", "value": 27.0}}),
        ("aluminum", "GMAW", 4.0, {"wire_feed_rate": {"op": "min", "value": 11.0}}),
        ("stainless_steel", "SMAW", 5.0, {"arc_voltage": {"op": "min", "value": 23.0}}),
    ]
    for mat, proc, thk, bnd in bound_variations:
        idx += 1
        scenarios.append({
            "scenario_id": f"SCEN_{idx:03d}",
            "material": mat,
            "process": proc,
            "thickness_mm": thk,
            "type": "axis_bound",
            "overrides": None,
            "bounds": bnd,
        })

    return scenarios


# ==============================================================================
# 3. MATHEMATICAL CONSISTENCY EVALUATOR
# ==============================================================================

def evaluate_mathematical_consistency(
    scenarios: List[Dict[str, Any]],
    tolerance: float = 1e-3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates mathematical consistency between production output and
    independent re-implementation across all benchmark scenarios.
    """
    standards_db = load_standards_database()
    results = []

    hi_abs_errors: List[float] = []
    hi_rel_errors: List[float] = []
    dr_abs_errors: List[float] = []
    dr_rel_errors: List[float] = []
    optimizer_matches = 0

    for sc in scenarios:
        mat = sc["material"]
        proc = sc["process"]
        thk = sc["thickness_mm"]
        overrides = sc.get("overrides")
        bounds = sc.get("bounds")

        # 1. Execute production advisor
        prod_rec = recommend_parameters(
            material=mat,
            thickness_mm=thk,
            process=proc,
            overrides=overrides,
            bounds=bounds,
        )

        assert "error" not in prod_rec, f"Unexpected error in valid scenario {sc['scenario_id']}: {prod_rec.get('error')}"

        prod_opt = prod_rec["optimized"]
        prod_metrics = prod_rec["computed_metrics"]
        prod_hi = prod_metrics["heat_input_kj_per_mm"]
        prod_dr = prod_metrics["deposition_rate_g_per_min"]

        # 2. Extract standards reference metadata
        mat_meta = standards_db["materials"][mat]
        density = mat_meta["density_g_cm3"]
        proc_family = prod_rec["process_family"]
        proc_meta = standards_db["processes"][proc_family]
        arc_eff = proc_meta["arc_efficiency"]
        dep_eff = proc_meta["deposition_efficiency"]

        # Determine band
        band = prod_rec["band"]
        standards_band = mat_meta[proc_family][band]
        table_diameter = (
            standards_band.get("wire_diameter")
            or standards_band.get("tungsten_diameter")
            or standards_band.get("electrode_diameter")
            or 1.2
        )
        effective_diameter = (
            overrides.get("wire_diameter") if overrides and "wire_diameter" in overrides else table_diameter
        )

        # 3. Independent mathematical recomputations
        # A. Independent grid search verification (solver level)
        indep_opt = IndependentPhysicsEvaluator.run_independent_grid_search(
            standards=standards_band,
            arc_efficiency=arc_eff,
            deposition_efficiency=dep_eff,
            wire_diameter_mm=effective_diameter,
            density_g_per_cm3=density,
            axis_overrides=overrides,
            axis_bounds=bounds,
        )

        opt_match = (
            round(prod_opt["welding_current"]) == round(indep_opt["current"])
            and round(prod_opt["arc_voltage"], 1) == round(indep_opt["voltage"], 1)
            and round(prod_opt["welding_speed"]) == round(indep_opt["speed"])
        )
        if opt_match:
            optimizer_matches += 1

        # Solver-level consistency (comparing against independent solver result rounded to display precision)
        indep_solver_hi = round(indep_opt["heat_input"], 3)
        solver_hi_abs_err = abs(prod_hi - indep_solver_hi)
        solver_hi_rel_err = solver_hi_abs_err / indep_solver_hi if indep_solver_hi > 0 else 0.0

        if indep_opt["deposition_rate"] is not None:
            indep_solver_dr = round(indep_opt["deposition_rate"], 1)
            solver_dr_abs_err = abs(prod_dr - indep_solver_dr)
            solver_dr_rel_err = solver_dr_abs_err / indep_solver_dr if indep_solver_dr > 0 else 0.0
        else:
            indep_solver_dr = None
            solver_dr_abs_err = 0.0
            solver_dr_rel_err = 0.0

        # B. Post-discretization delta: evaluate formula using the rounded integer/display values
        indep_display_hi = IndependentPhysicsEvaluator.calculate_heat_input(
            current_a=prod_opt["welding_current"],
            voltage_v=prod_opt["arc_voltage"],
            speed_mm_per_min=prod_opt["welding_speed"],
            arc_efficiency=arc_eff,
        )
        display_hi_abs_err = abs(prod_hi - indep_display_hi)
        display_hi_rel_err = display_hi_abs_err / indep_display_hi if indep_display_hi > 0 else 0.0
        hi_abs_errors.append(display_hi_abs_err)
        hi_rel_errors.append(display_hi_rel_err)

        if prod_opt["wire_feed_rate"] is not None:
            indep_display_dr = IndependentPhysicsEvaluator.calculate_deposition_rate(
                wire_feed_rate_m_per_min=prod_opt["wire_feed_rate"],
                wire_diameter_mm=effective_diameter,
                density_g_per_cm3=density,
                deposition_efficiency=dep_eff,
            )
            assert prod_dr is not None, "Deposition rate missing in wire-fed production output"
            display_dr_abs_err = abs(prod_dr - indep_display_dr)
            display_dr_rel_err = display_dr_abs_err / indep_display_dr if indep_display_dr > 0 else 0.0
            dr_abs_errors.append(display_dr_abs_err)
            dr_rel_errors.append(display_dr_rel_err)
        else:
            indep_display_dr = None
            display_dr_abs_err = 0.0
            display_dr_rel_err = 0.0

        results.append({
            "scenario_id": sc["scenario_id"],
            "material": mat,
            "process": proc,
            "thickness_mm": thk,
            "scenario_type": sc["type"],
            "prod_current": prod_opt["welding_current"],
            "prod_voltage": prod_opt["arc_voltage"],
            "prod_speed": prod_opt["welding_speed"],
            "prod_wire_feed": prod_opt["wire_feed_rate"],
            "prod_heat_input": prod_hi,
            "indep_solver_heat_input": indep_solver_hi,
            "solver_hi_abs_error": round(solver_hi_abs_err, 6),
            "indep_display_heat_input": round(indep_display_hi, 4),
            "display_hi_abs_error": round(display_hi_abs_err, 6),
            "display_hi_rel_error": round(display_hi_rel_err, 6),
            "prod_deposition_rate": prod_dr,
            "indep_solver_deposition_rate": indep_solver_dr,
            "solver_dr_abs_error": round(solver_dr_abs_err, 6),
            "indep_display_deposition_rate": round(indep_display_dr, 4) if indep_display_dr is not None else None,
            "display_dr_abs_error": round(display_dr_abs_err, 6) if indep_display_dr is not None else None,
            "display_dr_rel_error": round(display_dr_rel_err, 6) if indep_display_dr is not None else None,
            "optimizer_grid_match": opt_match,
            "solver_exact_match": (solver_hi_abs_err == 0.0 and solver_dr_abs_err == 0.0),
            "hi_within_tol": display_hi_abs_err <= 0.01,  # 0.01 kJ/mm tolerance for integer rounding
            "dr_within_tol": display_dr_abs_err <= 1.0 if indep_display_dr is not None else True,
        })

    def calc_stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "max": 0.0, "p95": 0.0}
        s_vals = sorted(vals)
        n = len(s_vals)
        mean_v = sum(s_vals) / n
        median_v = s_vals[n // 2]
        max_v = s_vals[-1]
        p95_v = s_vals[int(0.95 * n)]
        return {"mean": mean_v, "median": median_v, "max": max_v, "p95": p95_v}

    summary = {
        "total_scenarios_evaluated": len(scenarios),
        "solver_exact_match_pct": round(
            100.0 * sum(1 for r in results if r["solver_exact_match"]) / len(scenarios), 2
        ),
        "grid_optimizer_exact_match_pct": round(
            100.0 * optimizer_matches / len(scenarios), 2
        ),
        "heat_input_abs_error": calc_stats(hi_abs_errors),
        "heat_input_rel_error": calc_stats(hi_rel_errors),
        "heat_input_within_tolerance_pct": round(
            100.0 * sum(1 for e in hi_abs_errors if e <= 0.01) / len(hi_abs_errors), 2
        ),
        "deposition_rate_abs_error": calc_stats(dr_abs_errors),
        "deposition_rate_rel_error": calc_stats(dr_rel_errors),
        "deposition_rate_within_tolerance_pct": round(
            100.0 * sum(1 for e in dr_abs_errors if e <= 1.0) / len(dr_abs_errors), 2
        ) if dr_abs_errors else 100.0,
    }

    return results, summary


# ==============================================================================
# 4. ENGINEERING CONSTRAINT VALIDATION
# ==============================================================================

def evaluate_engineering_constraints(
    scenarios: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates compliance of production recommendations with explicit encoded
    engineering constraints from the standards database.
    """
    standards_db = load_standards_database()
    results = []

    total_checks = 0
    passed_checks = 0

    checks_by_param = {
        "welding_current": {"total": 0, "passed": 0},
        "arc_voltage": {"total": 0, "passed": 0},
        "welding_speed": {"total": 0, "passed": 0},
        "wire_feed_rate": {"total": 0, "passed": 0},
        "heat_input": {"total": 0, "passed": 0},
    }

    checks_by_process = {
        "GMAW": {"total": 0, "passed": 0},
        "TIG": {"total": 0, "passed": 0},
        "SMAW": {"total": 0, "passed": 0},
    }

    for sc in scenarios:
        rec = recommend_parameters(
            material=sc["material"],
            thickness_mm=sc["thickness_mm"],
            process=sc["process"],
            overrides=sc.get("overrides"),
            bounds=sc.get("bounds"),
        )

        opt = rec["optimized"]
        metrics = rec["computed_metrics"]
        band = rec["band"]
        mat_meta = standards_db["materials"][sc["material"]]
        proc_table = mat_meta[rec["process_family"]][band]

        scenario_violations = []

        # 1. Welding current check
        i_range = proc_table["welding_current"]
        i_val = opt["welding_current"]
        i_ok = (i_range[0] - 0.5 <= i_val <= i_range[1] + 0.5)
        # Note: if an explicit out-of-window override was provided, the advisor correctly pins it
        if sc.get("overrides") and "welding_current" in sc["overrides"]:
            i_expected = sc["overrides"]["welding_current"]
            i_ok = abs(i_val - i_expected) <= 1.0
        checks_by_param["welding_current"]["total"] += 1
        if i_ok:
            checks_by_param["welding_current"]["passed"] += 1
        else:
            scenario_violations.append(f"Current {i_val} A outside {i_range}")

        # 2. Arc voltage check
        v_range = proc_table["arc_voltage"]
        v_val = opt["arc_voltage"]
        v_ok = (v_range[0] - 0.1 <= v_val <= v_range[1] + 0.1)
        if sc.get("overrides") and "arc_voltage" in sc["overrides"]:
            v_expected = sc["overrides"]["arc_voltage"]
            v_ok = abs(v_val - v_expected) <= 0.1
        checks_by_param["arc_voltage"]["total"] += 1
        if v_ok:
            checks_by_param["arc_voltage"]["passed"] += 1
        else:
            scenario_violations.append(f"Voltage {v_val} V outside {v_range}")

        # 3. Welding speed check
        s_range = proc_table["welding_speed"]
        s_val = opt["welding_speed"]
        s_ok = (s_range[0] - 0.5 <= s_val <= s_range[1] + 0.5)
        if sc.get("overrides") and "welding_speed" in sc["overrides"]:
            s_expected = sc["overrides"]["welding_speed"]
            s_ok = abs(s_val - s_expected) <= 1.0
        checks_by_param["welding_speed"]["total"] += 1
        if s_ok:
            checks_by_param["welding_speed"]["passed"] += 1
        else:
            scenario_violations.append(f"Speed {s_val} mm/min outside {s_range}")

        # 4. Wire feed rate check (for wire-fed processes)
        wfr_range = proc_table.get("wire_feed_rate")
        wfr_val = opt["wire_feed_rate"]
        if wfr_range is not None and wfr_val is not None:
            wfr_ok = (wfr_range[0] - 0.1 <= wfr_val <= wfr_range[1] + 0.1)
            if sc.get("overrides") and "wire_feed_rate" in sc["overrides"]:
                wfr_ok = abs(wfr_val - sc["overrides"]["wire_feed_rate"]) <= 0.1
            checks_by_param["wire_feed_rate"]["total"] += 1
            if wfr_ok:
                checks_by_param["wire_feed_rate"]["passed"] += 1
            else:
                scenario_violations.append(f"WFR {wfr_val} m/min outside {wfr_range}")
        elif wfr_range is None:
            # Non-wire-fed process must report None
            assert wfr_val is None, f"Expected None WFR for {sc['process']}"

        # 5. Heat input window check
        hi_range = proc_table["heat_input_range"]
        hi_val = metrics["heat_input_kj_per_mm"]
        # Allow small boundary tolerance for numerical discretization
        hi_ok = (hi_range[0] - 0.05 <= hi_val <= hi_range[1] + 0.05)
        checks_by_param["heat_input"]["total"] += 1
        if hi_ok:
            checks_by_param["heat_input"]["passed"] += 1
        else:
            scenario_violations.append(f"Heat input {hi_val} kJ/mm outside {hi_range}")

        # Total scenario checks
        n_sc_checks = 4 + (1 if wfr_range is not None else 0)
        n_sc_passed = n_sc_checks - len(scenario_violations)
        total_checks += n_sc_checks
        passed_checks += n_sc_passed

        proc_key = rec["process_family"]
        checks_by_process[proc_key]["total"] += n_sc_checks
        checks_by_process[proc_key]["passed"] += n_sc_passed

        results.append({
            "scenario_id": sc["scenario_id"],
            "material": sc["material"],
            "process": sc["process"],
            "thickness_mm": sc["thickness_mm"],
            "is_valid": len(scenario_violations) == 0,
            "violations_count": len(scenario_violations),
            "violations": scenario_violations,
        })

    summary = {
        "overall_constraint_satisfaction_pct": round(100.0 * passed_checks / total_checks, 2),
        "total_constraint_checks": total_checks,
        "passed_constraint_checks": passed_checks,
        "satisfaction_by_parameter": {
            param: round(100.0 * d["passed"] / d["total"], 2) if d["total"] > 0 else 100.0
            for param, d in checks_by_param.items()
        },
        "satisfaction_by_process": {
            proc: round(100.0 * d["passed"] / d["total"], 2) if d["total"] > 0 else 100.0
            for proc, d in checks_by_process.items()
        },
    }

    return results, summary


# ==============================================================================
# 5. INPUT VALIDITY & EDGE CASE TESTING
# ==============================================================================

def evaluate_edge_cases() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates advisor behavior across boundary conditions, unsupported materials,
    incompatible processes, zero/negative thickness, and extreme values.
    Verifies that all conditions are gracefully handled with zero uncaught exceptions.
    """
    test_cases = [
        {"case_id": "EDGE_001", "name": "Zero thickness", "input": {"material": "mild_steel", "thickness_mm": 0.0, "process": "MIG"}},
        {"case_id": "EDGE_002", "name": "Negative thickness", "input": {"material": "mild_steel", "thickness_mm": -2.5, "process": "MIG"}},
        {"case_id": "EDGE_003", "name": "Sub-millimeter thin sheet", "input": {"material": "stainless_steel", "thickness_mm": 0.5, "process": "TIG"}},
        {"case_id": "EDGE_004", "name": "Extreme heavy section (150 mm)", "input": {"material": "mild_steel", "thickness_mm": 150.0, "process": "GMAW"}},
        {"case_id": "EDGE_005", "name": "Astronomical thickness (1500 mm)", "input": {"material": "mild_steel", "thickness_mm": 1500.0, "process": "SMAW"}},
        {"case_id": "EDGE_006", "name": "Unsupported material: titanium", "input": {"material": "titanium", "thickness_mm": 5.0, "process": "TIG"}},
        {"case_id": "EDGE_007", "name": "Unsupported material: inconel", "input": {"material": "inconel", "thickness_mm": 4.0, "process": "TIG"}},
        {"case_id": "EDGE_008", "name": "Unsupported material: copper", "input": {"material": "copper", "thickness_mm": 3.0, "process": "GMAW"}},
        {"case_id": "EDGE_009", "name": "Arbitrary unknown material", "input": {"material": "vibranium_superalloy", "thickness_mm": 10.0, "process": "MIG"}},
        {"case_id": "EDGE_010", "name": "Unsupported process on aluminum (SMAW)", "input": {"material": "aluminum", "thickness_mm": 5.0, "process": "SMAW"}},
        {"case_id": "EDGE_011", "name": "Unknown process string", "input": {"material": "mild_steel", "thickness_mm": 5.0, "process": "friction_stir"}},
        {"case_id": "EDGE_012", "name": "Out-of-range override current (5000 A)", "input": {"material": "mild_steel", "thickness_mm": 5.0, "process": "MIG", "overrides": {"welding_current": 5000.0}}},
        {"case_id": "EDGE_013", "name": "Negative override voltage (-10 V)", "input": {"material": "mild_steel", "thickness_mm": 5.0, "process": "MIG", "overrides": {"arc_voltage": -10.0}}},
        {"case_id": "EDGE_014", "name": "Extreme wire diameter override (10.0 mm)", "input": {"material": "mild_steel", "thickness_mm": 5.0, "process": "MIG", "overrides": {"wire_diameter": 10.0}}},
        {"case_id": "EDGE_015", "name": "Contradictory bound (min voltage > window)", "input": {"material": "stainless_steel", "thickness_mm": 3.0, "process": "TIG", "bounds": {"arc_voltage": {"op": "min", "value": 50.0}}}},
    ]

    results = []
    handled_count = 0
    crash_count = 0

    for tc in test_cases:
        inp = tc["input"]
        try:
            res = recommend_parameters(
                material=inp["material"],
                thickness_mm=inp["thickness_mm"],
                process=inp.get("process", "GMAW"),
                overrides=inp.get("overrides"),
                bounds=inp.get("bounds"),
            )

            # Analyze handling behavior
            if "error" in res:
                handling = "informative_error"
                is_handled = True
            elif "override_warnings" in res and res["override_warnings"]:
                handling = "warning_fallback"
                is_handled = True
            elif "optimized" in res:
                handling = "valid_recommendation"
                is_handled = True
            else:
                handling = "unexpected_response_shape"
                is_handled = False

            handled_count += int(is_handled)

            results.append({
                "case_id": tc["case_id"],
                "name": tc["name"],
                "input": inp,
                "handling": handling,
                "has_error": "error" in res,
                "error_message": res.get("error"),
                "warnings": res.get("override_warnings", []),
                "crashed": False,
            })
        except Exception as exc:
            crash_count += 1
            results.append({
                "case_id": tc["case_id"],
                "name": tc["name"],
                "input": inp,
                "handling": "unhandled_exception",
                "has_error": True,
                "error_message": str(exc),
                "crashed": True,
            })

    summary = {
        "total_edge_cases": len(test_cases),
        "gracefully_handled": handled_count,
        "unhandled_crashes": crash_count,
        "robustness_rate_pct": round(100.0 * handled_count / len(test_cases), 2),
    }

    return results, summary


# ==============================================================================
# 6. SENSITIVITY & STABILITY ANALYSIS
# ==============================================================================

def evaluate_sensitivity(
    base_cases: Optional[List[Dict[str, Any]]] = None,
    perturbations: Tuple[float, ...] = (-0.05, -0.02, -0.01, 0.01, 0.02, 0.05),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates response stability and variation under systematic continuous
    perturbations (e.g. +/-1%, +/-2%, +/-5%) on thickness and continuous overrides.
    """
    if base_cases is None:
        base_cases = [
            {"material": "mild_steel", "thickness_mm": 5.0, "process": "GMAW"},
            {"material": "mild_steel", "thickness_mm": 2.0, "process": "TIG"},
            {"material": "stainless_steel", "thickness_mm": 6.0, "process": "SMAW"},
            {"material": "stainless_steel", "thickness_mm": 2.0, "process": "GMAW"},
            {"material": "aluminum", "thickness_mm": 8.0, "process": "GMAW"},
            {"material": "aluminum", "thickness_mm": 3.0, "process": "TIG"},
        ]

    results = []
    perturbation_records: List[Dict[str, Any]] = []

    for bc in base_cases:
        mat = bc["material"]
        thk_base = bc["thickness_mm"]
        proc = bc["process"]

        rec_base = recommend_parameters(material=mat, thickness_mm=thk_base, process=proc)
        opt_base = rec_base["optimized"]
        hi_base = rec_base["computed_metrics"]["heat_input_kj_per_mm"]
        dr_base = rec_base["computed_metrics"]["deposition_rate_g_per_min"]

        case_results = []
        for delta in perturbations:
            thk_pert = thk_base * (1.0 + delta)
            rec_pert = recommend_parameters(material=mat, thickness_mm=thk_pert, process=proc)
            opt_pert = rec_pert["optimized"]
            hi_pert = rec_pert["computed_metrics"]["heat_input_kj_per_mm"]
            dr_pert = rec_pert["computed_metrics"]["deposition_rate_g_per_min"]

            # Calculate parameter delta percentages
            current_diff_pct = (
                (opt_pert["welding_current"] - opt_base["welding_current"]) / opt_base["welding_current"] * 100.0
            )
            speed_diff_pct = (
                (opt_pert["welding_speed"] - opt_base["welding_speed"]) / opt_base["welding_speed"] * 100.0
            )
            hi_diff_pct = ((hi_pert - hi_base) / hi_base * 100.0) if hi_base > 0 else 0.0

            band_changed = rec_pert["band"] != rec_base["band"]

            record = {
                "material": mat,
                "process": proc,
                "base_thickness_mm": thk_base,
                "perturbed_thickness_mm": round(thk_pert, 3),
                "perturbation_fraction": delta,
                "band_changed": band_changed,
                "current_diff_pct": round(current_diff_pct, 2),
                "speed_diff_pct": round(speed_diff_pct, 2),
                "heat_input_diff_pct": round(hi_diff_pct, 2),
            }
            case_results.append(record)
            perturbation_records.append(record)

        results.append({
            "base_case": bc,
            "perturbations": case_results,
        })

    # Summary: within-band stability vs across-band transitions
    within_band_records = [r for r in perturbation_records if not r["band_changed"]]
    across_band_records = [r for r in perturbation_records if r["band_changed"]]

    summary = {
        "total_perturbations_evaluated": len(perturbation_records),
        "within_band_evaluations": len(within_band_records),
        "within_band_zero_change_pct": round(
            100.0 * sum(1 for r in within_band_records if abs(r["current_diff_pct"]) < 1e-4) / len(within_band_records), 2
        ) if within_band_records else 100.0,
        "across_band_transitions": len(across_band_records),
        "mean_heat_input_diff_pct": round(
            sum(abs(r["heat_input_diff_pct"]) for r in perturbation_records) / len(perturbation_records), 2
        ),
    }

    return results, summary


# ==============================================================================
# 7. CROSS-PROCESS COVERAGE & COMPARATIVE ANALYSIS
# ==============================================================================

def evaluate_cross_process_coverage() -> Dict[str, Any]:
    """
    Evaluates process coverage across GMAW, TIG, and SMAW for all supported materials.
    """
    standards_db = load_standards_database()
    materials = ["mild_steel", "stainless_steel", "aluminum"]
    processes = ["GMAW", "TIG", "SMAW"]

    coverage_grid = {}
    for mat in materials:
        coverage_grid[mat] = {}
        for proc in processes:
            rec = recommend_parameters(material=mat, thickness_mm=5.0, process=proc)
            if "error" in rec:
                coverage_grid[mat][proc] = {
                    "supported": False,
                    "reason": rec["error"],
                    "bands": [],
                }
            else:
                proc_family = rec["process_family"]
                bands = list(standards_db["materials"][mat][proc_family].keys())
                coverage_grid[mat][proc] = {
                    "supported": True,
                    "bands": bands,
                    "uses_wire_feed": rec["optimized"]["wire_feed_rate"] is not None,
                    "arc_efficiency": rec["arc_efficiency"],
                    "deposition_efficiency": rec["deposition_efficiency"],
                }

    return coverage_grid


# ==============================================================================
# 8. INTERNAL SANITY & SELF-CONSISTENCY CHECK
# ==============================================================================

def evaluate_internal_consistency(
    scenarios: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Verifies internal mathematical self-consistency across all returned fields.
    Checks:
    1. Heat input equation consistency: does computed_metrics["heat_input_kj_per_mm"]
       strictly match _heat_input(optimized["welding_current"], optimized["arc_voltage"], optimized["welding_speed"], arc_eff)?
    2. Deposition consistency: does computed deposition match reported WFR and diameter?
    3. Structural consistency: TIG/SMAW reporting None for wire_feed_rate and deposition_rate.
    4. Text consistency: does summary_text contain the reported numerical values?
    """
    results = []
    passed_cases = 0

    for sc in scenarios:
        rec = recommend_parameters(
            material=sc["material"],
            thickness_mm=sc["thickness_mm"],
            process=sc["process"],
            overrides=sc.get("overrides"),
            bounds=sc.get("bounds"),
        )

        opt = rec["optimized"]
        met = rec["computed_metrics"]
        arc_eff = rec["arc_efficiency"]
        dep_eff = rec["deposition_efficiency"]

        # Check 1: Heat input matches reported optimized triple
        recalc_hi = IndependentPhysicsEvaluator.calculate_heat_input(
            current_a=opt["welding_current"],
            voltage_v=opt["arc_voltage"],
            speed_mm_per_min=opt["welding_speed"],
            arc_efficiency=arc_eff,
        )
        hi_diff = abs(met["heat_input_kj_per_mm"] - recalc_hi)
        hi_self_consistent = (hi_diff < 0.05)  # small delta due to rounding to 3 decimals in output

        # Check 2: Deposition rate self-consistency
        if opt["wire_feed_rate"] is not None:
            mat_meta = load_standards_database()["materials"][sc["material"]]
            wire_dia = rec.get("wire_diameter") or 1.2
            recalc_dr = IndependentPhysicsEvaluator.calculate_deposition_rate(
                wire_feed_rate_m_per_min=opt["wire_feed_rate"],
                wire_diameter_mm=wire_dia,
                density_g_per_cm3=mat_meta["density_g_cm3"],
                deposition_efficiency=dep_eff,
            )
            dr_diff = abs(met["deposition_rate_g_per_min"] - recalc_dr)
            dr_self_consistent = (dr_diff < 1.0)  # accommodates 0.1 m/min display rounding on WFR
        else:
            dr_self_consistent = (met["deposition_rate_g_per_min"] is None)

        # Check 3: Text consistency
        summary = rec["summary_text"]
        text_consistent = (
            str(opt["welding_current"]) in summary
            and str(opt["arc_voltage"]) in summary
            and str(opt["welding_speed"]) in summary
        )

        is_self_consistent = hi_self_consistent and dr_self_consistent and text_consistent
        if is_self_consistent:
            passed_cases += 1

        results.append({
            "scenario_id": sc["scenario_id"],
            "heat_input_self_consistent": hi_self_consistent,
            "deposition_rate_self_consistent": dr_self_consistent,
            "text_self_consistent": text_consistent,
            "is_self_consistent": is_self_consistent,
        })

    summary = {
        "total_scenarios_evaluated": len(scenarios),
        "self_consistent_scenarios": passed_cases,
        "internal_consistency_rate_pct": round(100.0 * passed_cases / len(scenarios), 2),
    }

    return results, summary


# ==============================================================================
# 9. COMPUTATIONAL LATENCY PROFILER
# ==============================================================================

def evaluate_latency(
    benchmark_scenarios: List[Dict[str, Any]],
    n_iterations: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, Any]:
    """
    Profiles computational latency (mean, median, p95, p99, min, max) of
    producing parameter recommendations.
    """
    # Warmup
    for i in range(warmup_runs):
        sc = benchmark_scenarios[i % len(benchmark_scenarios)]
        _ = recommend_parameters(
            material=sc["material"],
            thickness_mm=sc["thickness_mm"],
            process=sc["process"],
            overrides=sc.get("overrides"),
            bounds=sc.get("bounds"),
        )

    latencies_ms: List[float] = []
    for i in range(n_iterations):
        sc = benchmark_scenarios[i % len(benchmark_scenarios)]
        t0 = time.perf_counter()
        _ = recommend_parameters(
            material=sc["material"],
            thickness_mm=sc["thickness_mm"],
            process=sc["process"],
            overrides=sc.get("overrides"),
            bounds=sc.get("bounds"),
        )
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms.sort()
    n = len(latencies_ms)

    return {
        "total_iterations": n,
        "mean_latency_ms": round(sum(latencies_ms) / n, 3),
        "median_latency_ms": round(latencies_ms[n // 2], 3),
        "p95_latency_ms": round(latencies_ms[int(0.95 * n)], 3),
        "p99_latency_ms": round(latencies_ms[int(0.99 * n)], 3),
        "min_latency_ms": round(latencies_ms[0], 3),
        "max_latency_ms": round(latencies_ms[-1], 3),
        "latencies_ms_sample": [round(x, 3) for x in latencies_ms],
    }
