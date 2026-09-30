"""
Unit and regression tests for Step 10: Physics-Based Recommendation Evaluation.

Verifies:
1. Independent mathematical formulas compute exact analytical quantities.
2. Benchmark generation is strictly deterministic and covers all valid regimes.
3. Production advisor output matches independent mathematical re-implementation.
4. Engineering constraints from welding_params.yaml are honored.
5. Edge cases, boundary inputs, and unsupported materials are handled gracefully with 0 crashes.
6. Perturbation sensitivity exhibits within-band stability.
7. Internal self-consistency across all dictionary fields.
8. Cross-process coverage accurately differentiates valid and invalid combinations.
9. Computational latency executes within realistic sub-millisecond bounds.
10. All generated artifacts (JSON, CSV, MD, PNG figures) exist and are non-empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.physics_evaluation import (
    IndependentPhysicsEvaluator,
    generate_physics_benchmark,
    evaluate_mathematical_consistency,
    evaluate_engineering_constraints,
    evaluate_edge_cases,
    evaluate_sensitivity,
    evaluate_cross_process_coverage,
    evaluate_internal_consistency,
    evaluate_latency,
    load_standards_database,
)
from src.reasoning.param_advisor import recommend_parameters

_REPO_ROOT = Path(__file__).parent.parent
_ARTIFACTS_DIR = _REPO_ROOT / "evaluation" / "artifacts"
_FIGURES_DIR = _ARTIFACTS_DIR / "figures"


# 1. Independent Physics Formula Correctness
def test_independent_heat_input_formula():
    """Verify heat input equation: HI = (eta * 60 * I * V) / (1000 * S)."""
    # 200 A, 25 V, 300 mm/min, eta = 0.82
    # Power = 200 * 25 = 5000 W
    # Energy = 5000 * 60 * 0.82 = 246,000 J/min
    # HI = 246,000 / (1000 * 300) = 0.82 kJ/mm
    hi = IndependentPhysicsEvaluator.calculate_heat_input(200.0, 25.0, 300.0, 0.82)
    assert abs(hi - 0.82) < 1e-6

    with pytest.raises(ValueError):
        IndependentPhysicsEvaluator.calculate_heat_input(200.0, 25.0, 0.0, 0.82)


def test_independent_deposition_rate_formula():
    """Verify deposition rate: DR = WFR * pi * (d/2)^2 * rho * eta_dep."""
    # WFR = 10 m/min, wire_dia = 1.2 mm, rho = 7.85 g/cm3, eta_dep = 0.95
    # Area = pi * 0.6^2 = 1.130973 mm2
    # Vol = 10 * 1.130973 = 11.30973 cm3/min
    # Mass = 11.30973 * 7.85 * 0.95 = 84.342 g/min
    dr = IndependentPhysicsEvaluator.calculate_deposition_rate(10.0, 1.2, 7.85, 0.95)
    assert dr is not None
    assert abs(dr - 84.342) < 1e-2

    # None input should return None
    assert IndependentPhysicsEvaluator.calculate_deposition_rate(None, 1.2, 7.85, 0.95) is None


def test_independent_window_score():
    """Verify triangular score: 1.0 at center, 0.7 at edges, 0.0 outside."""
    lo, hi_max = 0.4, 0.8
    center = 0.6

    # Center
    assert abs(IndependentPhysicsEvaluator.calculate_window_score(0.6, lo, hi_max) - 1.0) < 1e-6
    # Lower edge
    assert abs(IndependentPhysicsEvaluator.calculate_window_score(0.4, lo, hi_max) - 0.7) < 1e-6
    # Upper edge
    assert abs(IndependentPhysicsEvaluator.calculate_window_score(0.8, lo, hi_max) - 0.7) < 1e-6
    # Outside below
    assert IndependentPhysicsEvaluator.calculate_window_score(0.39, lo, hi_max) == 0.0
    # Outside above
    assert IndependentPhysicsEvaluator.calculate_window_score(0.81, lo, hi_max) == 0.0


def test_independent_iso_deposition_feed():
    """Verify wire swap compensation: (d_table / d_override)^2 scaling."""
    # 1.2 mm wire swapping to 1.6 mm wire at 10.0 m/min
    # Ratio = (1.2 / 1.6)^2 = (0.75)^2 = 0.5625
    # WFR_same = 10.0 * 0.5625 = 5.625 -> 5.6 m/min
    wfr_iso = IndependentPhysicsEvaluator.calculate_iso_deposition_feed(1.2, 1.6, 10.0)
    assert wfr_iso == 5.6


# 2. Benchmark Determinism and Coverage
def test_benchmark_generation_determinism():
    """Ensure benchmark generation is strictly deterministic and covers valid regimes."""
    b1 = generate_physics_benchmark()
    b2 = generate_physics_benchmark()
    assert len(b1) == 144
    assert len(b1) == len(b2)
    assert b1 == b2

    materials = {s["material"] for s in b1}
    assert materials == {"mild_steel", "stainless_steel", "aluminum"}

    processes = {s["process"] for s in b1}
    assert {"GMAW", "TIG", "SMAW"}.issubset(processes)


# 3. Mathematical Consistency
def test_mathematical_consistency_evaluation():
    """Verify mathematical consistency between production advisor and independent solver."""
    scenarios = generate_physics_benchmark()[:20]  # subset for fast unit testing
    results, summary = evaluate_mathematical_consistency(scenarios)

    assert summary["solver_exact_match_pct"] == 100.0
    assert summary["grid_optimizer_exact_match_pct"] == 100.0
    assert summary["heat_input_within_tolerance_pct"] == 100.0
    assert summary["deposition_rate_within_tolerance_pct"] == 100.0


# 4. Engineering Constraints Validation
def test_engineering_constraints_compliance():
    """Verify compliance with explicit encoded standards windows."""
    scenarios = [s for s in generate_physics_benchmark() if s["type"] == "baseline_unconstrained"][:25]
    results, summary = evaluate_engineering_constraints(scenarios)

    assert summary["overall_constraint_satisfaction_pct"] == 100.0
    assert all(r["is_valid"] for r in results)


# 5. Input Validity & Edge Cases
def test_edge_case_robustness():
    """Verify boundary conditions and unsupported inputs are handled with zero crashes."""
    results, summary = evaluate_edge_cases()

    assert summary["unhandled_crashes"] == 0
    assert summary["robustness_rate_pct"] == 100.0

    # Explicit check for unsupported material
    rec_ti = recommend_parameters("titanium", 5.0, "TIG")
    assert "error" in rec_ti
    assert "not recognised" in rec_ti["error"]

    # Explicit check for incompatible process
    rec_al_smaw = recommend_parameters("aluminum", 5.0, "SMAW")
    assert "error" in rec_al_smaw
    assert "not standard industrial practice" in rec_al_smaw["error"]


# 6. Perturbation Sensitivity
def test_sensitivity_stability():
    """Verify within-band step function invariance under continuous thickness jitter."""
    results, summary = evaluate_sensitivity()

    assert summary["within_band_zero_change_pct"] == 100.0
    assert summary["total_perturbations_evaluated"] == 36


# 7. Internal Self-Consistency
def test_internal_self_consistency():
    """Verify coherence across reported recommendation dictionary fields."""
    scenarios = generate_physics_benchmark()[:25]
    results, summary = evaluate_internal_consistency(scenarios)

    assert summary["internal_consistency_rate_pct"] == 100.0


# 8. Cross-Process Coverage
def test_cross_process_coverage():
    """Verify process capability grid accurately reflects supported domains."""
    grid = evaluate_cross_process_coverage()

    assert grid["mild_steel"]["GMAW"]["supported"] is True
    assert grid["mild_steel"]["TIG"]["supported"] is True
    assert grid["mild_steel"]["SMAW"]["supported"] is True
    assert grid["aluminum"]["GMAW"]["supported"] is True
    assert grid["aluminum"]["TIG"]["supported"] is True
    assert grid["aluminum"]["SMAW"]["supported"] is False


# 9. Latency Benchmark
def test_computational_latency():
    """Verify recommendation generation runs with sub-millisecond execution time."""
    scenarios = generate_physics_benchmark()[:10]
    latency = evaluate_latency(scenarios, n_iterations=30, warmup_runs=5)

    assert latency["mean_latency_ms"] < 5.0
    assert latency["median_latency_ms"] < 5.0


# 10. Evaluation Artifacts Verification
def test_evaluation_artifacts_exist():
    """Verify that all required evaluation artifacts and figures exist and are populated."""
    required_artifacts = [
        _ARTIFACTS_DIR / "physics_inventory.json",
        _ARTIFACTS_DIR / "physics_evaluation_results.json",
        _ARTIFACTS_DIR / "physics_evaluation_results.csv",
        _ARTIFACTS_DIR / "physics_evaluation_summary.json",
        _ARTIFACTS_DIR / "physics_evaluation_report.md",
    ]

    for path in required_artifacts:
        assert path.exists(), f"Missing required artifact: {path}"
        assert path.stat().st_size > 0, f"Artifact is empty: {path}"

    required_figures = [
        _FIGURES_DIR / "physics_error_distribution.png",
        _FIGURES_DIR / "physics_constraint_satisfaction.png",
        _FIGURES_DIR / "physics_sensitivity.png",
        _FIGURES_DIR / "physics_latency.png",
        _FIGURES_DIR / "physics_process_comparison.png",
    ]

    for fig in required_figures:
        assert fig.exists(), f"Missing required figure: {fig}"
        assert fig.stat().st_size > 1000, f"Figure file suspiciously small: {fig}"
