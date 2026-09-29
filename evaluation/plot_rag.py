"""
Publication-Quality Figures for Formal RAG Retrieval Evaluation (Step 7).

Generates:
  1. rag_recall_at_k.png: Recall@K progression across retrieval conditions
  2. rag_mrr_comparison.png: MRR and nDCG@5 comparison across ablation conditions
  3. rag_ablation.png: Comprehensive multi-metric retrieval component ablation
  4. rag_threshold_sensitivity.png: Score threshold tau sensitivity curve & operating window
  5. rag_latency_breakdown.png: Component-level latency breakdown (encoding vs search/scoring)

Outputs are saved in:
  - evaluation/results/figures/
  - evaluation/results/rag/figures/
  - artifacts/figures/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = _ROOT / "evaluation" / "results" / "rag" / "rag_evaluation_results.json"

FIGURE_DIRS = [
    _ROOT / "evaluation" / "results" / "figures",
    _ROOT / "evaluation" / "results" / "rag" / "figures",
    _ROOT / "artifacts" / "figures",
]


def setup_style():
    """Configure matplotlib parameters for publication-grade formatting."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def save_figure(fig: plt.Figure, filename: str):
    """Save the figure into all target figure directories."""
    for fdir in FIGURE_DIRS:
        fdir.mkdir(parents=True, exist_ok=True)
        out_path = fdir / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {filename}")


# =====================================================================
# Figure 1: Recall@K Comparison Across Conditions
# =====================================================================

def plot_recall_at_k(data: Dict[str, Any]):
    setup_style()
    ablation = data["ablation_metrics"]
    conditions = list(ablation.keys())
    readable_labels = [c.replace("WITHOUT_", "W/O ").replace("_", " ") for c in conditions]

    k_levels = [1, 3, 4, 5]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    x = np.arange(len(conditions))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for idx, k in enumerate(k_levels):
        vals = [ablation[c][f"Recall@{k}"] for c in conditions]
        offset = (idx - 1.5) * width
        rects = ax.bar(x + offset, vals, width, label=f"Recall@{k}", color=colors[idx], alpha=0.9, edgecolor="black", linewidth=0.5)
        for r in rects:
            height = r.get_height()
            if height > 0.05:
                ax.annotate(f"{height:.2f}",
                            xy=(r.get_x() + r.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax.set_ylabel("Recall (Hit Rate)")
    ax.set_title("Recall@K Progression Across RAG Retrieval Ablation Conditions (N=40 Queries)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(readable_labels, fontweight="semibold")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    ax.legend(loc="lower right", framealpha=0.9)

    save_figure(fig, "rag_recall_at_k.png")


# =====================================================================
# Figure 2: MRR & nDCG@5 Comparison Across Conditions
# =====================================================================

def plot_mrr_comparison(data: Dict[str, Any]):
    setup_style()
    ablation = data["ablation_metrics"]
    conditions = list(ablation.keys())
    readable_labels = [c.replace("WITHOUT_", "W/O ").replace("_", " ") for c in conditions]

    mrrs = [ablation[c]["MRR"] for c in conditions]
    ndcgs = [ablation[c]["nDCG@5"] for c in conditions]

    y = np.arange(len(conditions))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))

    rects1 = ax.barh(y + height/2, mrrs, height, label="Mean Reciprocal Rank (MRR)", color="#2b5c8f", edgecolor="black", linewidth=0.5)
    rects2 = ax.barh(y - height/2, ndcgs, height, label="nDCG@5", color="#e67e22", edgecolor="black", linewidth=0.5)

    for r in rects1:
        w = r.get_width()
        ax.annotate(f"{w:.4f}", xy=(w, r.get_y() + r.get_height()/2), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.5, fontweight="bold")

    for r in rects2:
        w = r.get_width()
        ax.annotate(f"{w:.4f}", xy=(w, r.get_y() + r.get_height()/2), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.5)

    ax.set_xlabel("Metric Score")
    ax.set_title("Ranking Quality: MRR and nDCG@5 Across RAG Retrieval Conditions", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(readable_labels, fontweight="semibold")
    ax.set_xlim(0, 1.12)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.invert_yaxis()

    save_figure(fig, "rag_mrr_comparison.png")


# =====================================================================
# Figure 3: Full Multi-Metric Ablation Overview
# =====================================================================

def plot_ablation_overview(data: Dict[str, Any]):
    setup_style()
    ablation = data["ablation_metrics"]
    conditions = list(ablation.keys())
    readable_labels = [c.replace("WITHOUT_", "W/O ").replace("_", " ") for c in conditions]

    metrics = ["Recall@1", "Recall@4", "MRR", "nDCG@5"]
    metric_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#e67e22"]

    x = np.arange(len(conditions))
    width = 0.19

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, m in enumerate(metrics):
        vals = [ablation[c][m] for c in conditions]
        offset = (idx - 1.5) * width
        rects = ax.bar(x + offset, vals, width, label=m, color=metric_colors[idx], edgecolor="black", linewidth=0.5)
        for r in rects:
            height = r.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(r.get_x() + r.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5)

    ax.set_ylabel("Score")
    ax.set_title("RAG Retrieval Component Ablation Study (N=40 Welding Queries)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(readable_labels, fontweight="semibold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", framealpha=0.9, ncol=2)

    save_figure(fig, "rag_ablation.png")


# =====================================================================
# Figure 4: Threshold Sensitivity Curve & Operating Regime
# =====================================================================

def plot_threshold_sensitivity(data: Dict[str, Any]):
    setup_style()
    sweep = data["threshold_sweep"]

    thresholds = [s["threshold"] for s in sweep]
    r1 = [s["Recall@1"] for s in sweep]
    r4 = [s["Recall@4"] for s in sweep]
    mrr = [s["MRR"] for s in sweep]
    p4 = [s["P@4"] for s in sweep]
    avg_docs = [s["avg_retrieved"] for s in sweep]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    # Left axis: Retrieval quality metrics
    l1, = ax1.plot(thresholds, r4, marker="o", color="#2ca02c", linewidth=2.2, label="Recall@4")
    l2, = ax1.plot(thresholds, mrr, marker="s", color="#1f77b4", linewidth=2.0, label="MRR")
    l3, = ax1.plot(thresholds, r1, marker="^", color="#ff7f0e", linewidth=1.8, linestyle="--", label="Recall@1")
    l4, = ax1.plot(thresholds, p4, marker="d", color="#9467bd", linewidth=1.8, linestyle="-.", label="Precision@4")

    ax1.set_xlabel("Minimum Score Threshold (tau)")
    ax1.set_ylabel("Retrieval Metric Score")
    ax1.set_ylim(0.15, 1.08)

    # Optimal operating window shading
    ax1.axvspan(0.15, 0.25, color="#2ca02c", alpha=0.12, label="Optimal Window [0.15, 0.25]")
    ax1.axvline(0.15, color="#d62728", linestyle=":", linewidth=1.5, label="Default Cutoff (tau=0.15)")

    # Right axis: Average passages retained
    ax2 = ax1.twinx()
    l5, = ax2.plot(thresholds, avg_docs, marker="x", color="#7f7f7f", linewidth=1.8, linestyle=":", label="Avg Passages Retained")
    ax2.set_ylabel("Avg Retained Passages (out of 4)", color="#555555")
    ax2.set_ylim(0, 4.5)
    ax2.tick_params(axis="y", labelcolor="#555555")
    ax2.grid(False)

    lines = [l1, l2, l3, l4, l5]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower left", framealpha=0.9, fontsize=8.5)

    ax1.set_title("Minimum Score Filtering Threshold (tau) Sensitivity on Hybrid Retrieval", fontweight="bold")

    save_figure(fig, "rag_threshold_sensitivity.png")


# =====================================================================
# Figure 5: Component-Level Latency Breakdown
# =====================================================================

def plot_latency_breakdown(data: Dict[str, Any]):
    setup_style()
    lat = data["latency_profile"]

    components = ["Query Encoding\n(MiniLM-L6-v2)", "Vector Search\n& Reranking", "Total End-to-End\nRetrieval"]
    means = [
        lat["encoding_latency"]["mean_ms"],
        lat["scoring_latency"]["mean_ms"],
        lat["total_latency"]["mean_ms"],
    ]
    medians = [
        lat["encoding_latency"]["median_ms"],
        lat["scoring_latency"]["median_ms"],
        lat["total_latency"]["median_ms"],
    ]
    p95s = [
        lat["encoding_latency"]["p95_ms"],
        lat["scoring_latency"]["p95_ms"],
        lat["total_latency"]["p95_ms"],
    ]

    x = np.arange(len(components))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.5, 5))

    r1 = ax.bar(x - width, means, width, label="Mean (ms)", color="#2b5c8f", edgecolor="black", linewidth=0.5)
    r2 = ax.bar(x, medians, width, label="Median (ms)", color="#2ca02c", edgecolor="black", linewidth=0.5)
    r3 = ax.bar(x + width, p95s, width, label="P95 (ms)", color="#e67e22", edgecolor="black", linewidth=0.5)

    for group in [r1, r2, r3]:
        for r in group:
            h = r.get_height()
            label_text = f"{h:.2f} ms" if h >= 1.0 else f"{h:.3f} ms"
            ax.annotate(label_text,
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Latency (milliseconds)")
    ax.set_title("RAG Retrieval Subsystem Latency Decomposition (N=120 Benchmarked Queries)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontweight="semibold")
    ax.set_ylim(0, max(p95s) * 1.25)
    ax.legend(loc="upper left", framealpha=0.9)

    save_figure(fig, "rag_latency_breakdown.png")


def generate_all_figures():
    """Load evaluation results and generate all 5 publication-ready figures."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_FILE}. Run eval_rag.py first.")

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    plot_recall_at_k(data)
    plot_mrr_comparison(data)
    plot_ablation_overview(data)
    plot_threshold_sensitivity(data)
    plot_latency_breakdown(data)
    print("All 5 RAG retrieval evaluation figures generated successfully.")


if __name__ == "__main__":
    generate_all_figures()
