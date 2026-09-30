"""
Plotting script for Physics-Based Recommendation Evaluation.

Generates 5 publication-ready diagnostic visualizations:
1. physics_error_distribution.png: Mathematical error distribution (absolute and relative errors)
2. physics_constraint_satisfaction.png: Constraint satisfaction breakdown by parameter and process
3. physics_sensitivity.png: Response variation under input perturbations (+/-1%, +/-2%, +/-5%)
4. physics_latency.png: Latency distribution of recommendation generation
5. physics_process_comparison.png: Cross-process comparison across welding regimes
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

_OUTPUT_DIR = Path(__file__).parent / "artifacts" / "figures"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_error_distribution(
    math_results: List[Dict[str, Any]],
    output_path: Path = _OUTPUT_DIR / "physics_error_distribution.png",
) -> None:
    """Plot distribution of absolute and relative errors against independent equations."""
    hi_abs_errs = [r.get("display_hi_abs_error", r.get("hi_abs_error", 0.0)) for r in math_results]
    dr_abs_errs = [r.get("display_dr_abs_error", r.get("dr_abs_error")) for r in math_results if r.get("display_dr_abs_error") is not None or r.get("dr_abs_error") is not None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heat input error
    ax1 = axes[0]
    ax1.hist(hi_abs_errs, bins=25, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax1.set_title("Heat Input Absolute Error (|Prod - Indep|)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Absolute Error (kJ/mm)", fontsize=10)
    ax1.set_ylabel("Scenario Count", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    mean_hi = np.mean(hi_abs_errs)
    ax1.axvline(mean_hi, color="red", linestyle="--", label=f"Mean: {mean_hi:.2e} kJ/mm")
    ax1.legend(loc="upper right", fontsize=9)

    # Deposition rate error
    ax2 = axes[1]
    if dr_abs_errs:
        ax2.hist(dr_abs_errs, bins=25, color="#2ca02c", edgecolor="black", alpha=0.75)
        mean_dr = np.mean(dr_abs_errs)
        ax2.axvline(mean_dr, color="red", linestyle="--", label=f"Mean: {mean_dr:.2e} g/min")
        ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title("Deposition Rate Absolute Error (|Prod - Indep|)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Absolute Error (g/min)", fontsize=10)
    ax2.set_ylabel("Scenario Count", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_constraint_satisfaction(
    constraint_summary: Dict[str, Any],
    output_path: Path = _OUTPUT_DIR / "physics_constraint_satisfaction.png",
) -> None:
    """Plot constraint satisfaction percentages across parameters and processes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # By parameter
    ax1 = axes[0]
    params = list(constraint_summary["satisfaction_by_parameter"].keys())
    param_labels = [p.replace("_", " ").title() for p in params]
    param_vals = [constraint_summary["satisfaction_by_parameter"][p] for p in params]
    bars1 = ax1.bar(param_labels, param_vals, color="#3b528b", edgecolor="black", width=0.6)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Satisfaction Rate (%)", fontsize=10)
    ax1.set_title("Engineering Constraint Compliance by Parameter", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax1.tick_params(axis="x", rotation=25)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9)

    # By process
    ax2 = axes[1]
    procs = list(constraint_summary["satisfaction_by_process"].keys())
    proc_vals = [constraint_summary["satisfaction_by_process"][p] for p in procs]
    bars2 = ax2.bar(procs, proc_vals, color="#5ec962", edgecolor="black", width=0.5)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Satisfaction Rate (%)", fontsize=10)
    ax2.set_title("Constraint Compliance by Process Family", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_sensitivity(
    sensitivity_results: List[Dict[str, Any]],
    output_path: Path = _OUTPUT_DIR / "physics_sensitivity.png",
) -> None:
    """Plot parameter variation across continuous input perturbations."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1 = axes[0]
    ax2 = axes[1]

    # Collect perturbations for each base scenario
    deltas = [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]
    delta_pcts = [d * 100 for d in deltas]

    for item in sensitivity_results:
        bc = item["base_case"]
        lbl = f"{bc['material'][:4]}_{bc['process']}_{bc['thickness_mm']}mm"
        pert_dict = {p["perturbation_fraction"]: p for p in item["perturbations"]}

        curr_diffs = [pert_dict[d]["current_diff_pct"] for d in deltas]
        hi_diffs = [pert_dict[d]["heat_input_diff_pct"] for d in deltas]

        ax1.plot(delta_pcts, curr_diffs, marker="o", label=lbl, alpha=0.85)
        ax2.plot(delta_pcts, hi_diffs, marker="s", label=lbl, alpha=0.85)

    ax1.set_title("Welding Current Stability under Input Jitter", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Thickness Perturbation (%)", fontsize=10)
    ax1.set_ylabel("Welding Current Delta (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8)

    ax2.set_title("Heat Input Stability under Input Jitter", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Thickness Perturbation (%)", fontsize=10)
    ax2.set_ylabel("Heat Input Delta (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_latency(
    latency_summary: Dict[str, Any],
    output_path: Path = _OUTPUT_DIR / "physics_latency.png",
) -> None:
    """Plot distribution and summary statistics of recommendation generation latency."""
    latencies = latency_summary["latencies_ms_sample"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(latencies, bins=20, color="#440154", edgecolor="black", alpha=0.75)
    ax1.set_title("Recommendation Latency Distribution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Latency (ms)", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    mean_lat = latency_summary["mean_latency_ms"]
    p95_lat = latency_summary["p95_latency_ms"]
    ax1.axvline(mean_lat, color="cyan", linestyle="--", label=f"Mean: {mean_lat:.2f} ms")
    ax1.axvline(p95_lat, color="orange", linestyle="--", label=f"P95: {p95_lat:.2f} ms")
    ax1.legend(loc="upper right", fontsize=9)

    # Boxplot
    ax2 = axes[1]
    box = ax2.boxplot(latencies, vert=True, patch_artist=True, boxprops=dict(facecolor="#21908d", alpha=0.7))
    ax2.set_title("Recommendation Latency Boxplot", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Latency (ms)", fontsize=10)
    ax2.set_xticks([1])
    ax2.set_xticklabels(["Param Advisor Grid Search"])
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_process_comparison(
    benchmark_scenarios: List[Dict[str, Any]],
    output_path: Path = _OUTPUT_DIR / "physics_process_comparison.png",
) -> None:
    """Plot cross-process comparison of recommended current, heat input, and deposition."""
    from src.reasoning.param_advisor import recommend_parameters

    # Compare 5.0 mm plate for mild steel and stainless across processes
    targets = [
        ("mild_steel", "GMAW", 5.0),
        ("mild_steel", "TIG", 5.0),
        ("mild_steel", "SMAW", 5.0),
        ("stainless_steel", "GMAW", 5.0),
        ("stainless_steel", "TIG", 5.0),
        ("stainless_steel", "SMAW", 5.0),
        ("aluminum", "GMAW", 6.0),
        ("aluminum", "TIG", 6.0),
    ]

    labels = []
    currents = []
    heat_inputs = []
    depositions = []

    for mat, proc, thk in targets:
        rec = recommend_parameters(material=mat, thickness_mm=thk, process=proc)
        mat_short = "MS" if "mild" in mat else ("SS" if "stainless" in mat else "Al")
        labels.append(f"{mat_short}\n{proc}")
        currents.append(rec["optimized"]["welding_current"])
        heat_inputs.append(rec["computed_metrics"]["heat_input_kj_per_mm"])
        dep = rec["computed_metrics"]["deposition_rate_g_per_min"]
        depositions.append(dep if dep is not None else 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Currents
    axes[0].bar(labels, currents, color="#2b83ba", edgecolor="black", width=0.6)
    axes[0].set_title("Recommended Current by Regime", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Current (A)", fontsize=10)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Heat input
    axes[1].bar(labels, heat_inputs, color="#fdae61", edgecolor="black", width=0.6)
    axes[1].set_title("Recommended Heat Input", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Heat Input (kJ/mm)", fontsize=10)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Deposition rate
    axes[2].bar(labels, depositions, color="#abdda4", edgecolor="black", width=0.6)
    axes[2].set_title("Deposition Throughput (g/min)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Deposition Rate (g/min)", fontsize=10)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
