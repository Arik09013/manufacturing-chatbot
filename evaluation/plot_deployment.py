"""
Diagnostic plotting script for Deployment and Computational Efficiency Evaluation.

Generates 5 publication-ready figures:
1. deployment_latency.png: Warm latency boxplot and distribution by category.
2. deployment_component_breakdown.png: Breakdown of execution time across pipeline stages.
3. deployment_throughput.png: Query processing throughput across routes.
4. deployment_memory.png: Memory (RSS) progression across lifecycle phases.
5. deployment_cold_warm.png: Cold-start vs warm-start latency across routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

_OUTPUT_DIR = Path(__file__).parent / "artifacts" / "figures"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_deployment_latency(
    benchmark_results: List[Dict[str, Any]],
    output_path: Path = _OUTPUT_DIR / "deployment_latency.png",
) -> None:
    """Plot latency distribution across query categories (log scale for dynamic range)."""
    categories = ["param", "general", "knowledge", "out_of_scope", "anomaly"]
    cat_labels = ["Param\nAdvisor", "General\nMfg", "Knowledge\n(RAG)", "Out of\nScope", "Anomaly\n(ML+XAI)"]

    cat_data = []
    for cat in categories:
        vals = [
            r["warm_mean_latency_ms"]
            for r in benchmark_results
            if r["category"] == cat
        ]
        cat_data.append(vals if vals else [0.1])

    fig, ax = plt.subplots(figsize=(10, 5))
    boxes = ax.boxplot(
        cat_data,
        tick_labels=cat_labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="o", markerfacecolor="red", markeredgecolor="black"),
    )

    colors = ["#2b83ba", "#abdda4", "#fdae61", "#999999", "#d7191c"]
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yscale("log")
    ax.set_ylabel("Warm Latency (ms) — Log Scale", fontsize=10)
    ax.set_title("End-to-End Local Pipeline Latency by Route (Steady State)", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)

    # Annotate medians
    for idx, (cat, vals) in enumerate(zip(categories, cat_data)):
        med = np.median(vals)
        ax.text(idx + 1, med * 1.35, f"{med:.2f} ms", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_component_breakdown(
    component_summary: Dict[str, Any],
    output_path: Path = _OUTPUT_DIR / "deployment_component_breakdown.png",
) -> None:
    """Plot component-wise latency breakdown highlighting dominant bottlenecks."""
    # Exclude LLM synthesis from on-device breakdown to show internal module resolution clearly
    stages = [
        ("Intent Guard", component_summary["intent_classification"]["median_ms"]),
        ("Sensor Preproc & Fusion", component_summary["sensor_preprocessing"]["median_ms"]),
        ("Tabular RF Inference", component_summary["tabular_model_inference"]["median_ms"]),
        ("TreeSHAP XAI", component_summary["shap_explanation"]["median_ms"]),
        ("LIME Surrogate XAI", component_summary["lime_explanation"]["median_ms"]),
        ("DistilBERT Attention XAI", component_summary["attention_explanation"]["median_ms"]),
        ("Physics Optimization", component_summary["physics_optimization"]["median_ms"]),
        ("RAG Hybrid Retrieval", component_summary["rag_retrieval"]["median_ms"]),
    ]

    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    ax1 = axes[0]
    bars = ax1.barh(labels, values, color="#3182bd", edgecolor="black", alpha=0.8)
    ax1.set_xlabel("Median Latency (ms)", fontsize=10)
    ax1.set_title("On-Device Pipeline Stage Latencies", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 10, bar.get_y() + bar.get_height() / 2, f"{w:.2f} ms", va="center", fontsize=8)

    # Pie chart of anomaly pipeline stages
    ax2 = axes[1]
    anom_stages = [
        ("Sensor Preproc", component_summary["sensor_preprocessing"]["median_ms"]),
        ("RF Model", component_summary["tabular_model_inference"]["median_ms"]),
        ("TreeSHAP", component_summary["shap_explanation"]["median_ms"]),
        ("LIME Explainer", component_summary["lime_explanation"]["median_ms"]),
        ("DistilBERT", component_summary["attention_explanation"]["median_ms"]),
    ]
    anom_labels = [s[0] for s in anom_stages]
    anom_vals = [s[1] for s in anom_stages]
    pie_colors = ["#6baed6", "#9ecae1", "#fdae6b", "#e6550d", "#756bb1"]

    ax2.pie(
        anom_vals,
        labels=anom_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=pie_colors,
        wedgeprops=dict(edgecolor="black", alpha=0.85),
    )
    ax2.set_title("Anomaly Route Latency Share\n(Dominant Bottleneck: LIME & DistilBERT)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_throughput(
    throughput_data: Dict[str, Any],
    output_path: Path = _OUTPUT_DIR / "deployment_throughput.png",
) -> None:
    """Plot sequential query throughput (queries per second) across categories."""
    by_cat = throughput_data["by_category_throughput"]
    categories = list(by_cat.keys())
    qps = [by_cat[c]["queries_per_sec"] for c in categories]
    cat_labels = [c.replace("_", " ").title() for c in categories]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(cat_labels, qps, color="#31a354", edgecolor="black", width=0.55, alpha=0.85)

    ax.set_ylabel("Sequential Throughput (Queries / Second)", fontsize=10)
    ax.set_title("Sequential Throughput by Pipeline Category", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h * 1.15, f"{h:.1f} qps", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_memory_progression(
    memory_phases: Dict[str, float],
    output_path: Path = _OUTPUT_DIR / "deployment_memory.png",
) -> None:
    """Plot memory (RSS) progression across lifecycle phases."""
    phases = list(memory_phases.keys())
    values = list(memory_phases.values())
    labels = [p.replace("_", " ").title() for p in phases]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color="#756bb1", edgecolor="black", width=0.5, alpha=0.8)

    ax.set_ylabel("Process Resident Set Size (MB)", fontsize=10)
    ax.set_title("Memory Footprint Progression Across Lifecycle Phases", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 15, f"{h:.1f} MB", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, max(values) * 1.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cold_vs_warm(
    cold_data: Dict[str, float],
    warm_data: Dict[str, Dict[str, float]],
    output_path: Path = _OUTPUT_DIR / "deployment_cold_warm.png",
) -> None:
    """Plot cold-start vs steady-state warm latency across routes."""
    routes = ["param", "general", "knowledge", "anomaly"]
    labels = ["Param Advisor", "General Mfg", "Knowledge (RAG)", "Anomaly (ML+XAI)"]

    cold_vals = [cold_data.get(r, 0.0) for r in routes]
    warm_vals = [warm_data[r]["median"] for r in routes]

    x = np.arange(len(routes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, cold_vals, width, label="Cold Start (First Call)", color="#e6550d", edgecolor="black", alpha=0.85)
    b2 = ax.bar(x + width / 2, warm_vals, width, label="Warm Steady State (Median)", color="#3182bd", edgecolor="black", alpha=0.85)

    ax.set_ylabel("Latency (ms) — Log Scale", fontsize=10)
    ax.set_yscale("log")
    ax.set_title("Cold-Start vs Steady-State Warm Latency", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
