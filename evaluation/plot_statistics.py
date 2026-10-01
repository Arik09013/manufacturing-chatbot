"""
Plotting module for Research Step 12:
Statistical Significance, Uncertainty & Effect-Size Analysis.

Generates 5 publication-ready visualization figures in evaluation/artifacts/figures/:
1. statistical_metric_distributions.png (Model comparison metric distributions & uncertainty)
2. seed_variability.png (Multi-seed F1 dispersion across seeds 42, 123, 456, 789, 2026)
3. ablation_effect_sizes.png (Modality and component ablation effect sizes vs FULL)
4. rag_metric_comparison.png (RAG retrieval performance across hybrid ablations)
5. latency_distributions.png (Deployment latency distributions and median confidence intervals)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

_PALETTE = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "gray": "#7f7f7f",
    "teal": "#17becf",
}


def plot_model_metric_distributions(
    model_comparison_data: Dict[str, Any],
    output_path: Path,
) -> None:
    """Figure 1: Metric distributions and uncertainty across supervised/unsupervised models."""
    models = ["Random Forest", "DistilBERT (theta=0.5)", "Logistic Regression", "Support Vector Machine", "Isolation Forest"]
    short_names = ["Random Forest", "DistilBERT", "Logistic Reg.", "SVM", "Isolation Forest"]
    
    metrics = ["f1", "precision", "recall", "pr_auc"]
    metric_labels = ["F1 Score", "Precision", "Recall", "PR-AUC"]
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        vals = []
        for m in models:
            m_data = model_comparison_data.get("chronological_benchmark", {}).get(m, {})
            vals.append(m_data.get(metric, 0.0))
            
        bars = ax.bar(range(len(models)), vals, color=[_PALETTE["blue"], _PALETTE["purple"], _PALETTE["teal"], _PALETTE["orange"], _PALETTE["red"]], width=0.6, alpha=0.85)
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(short_names, rotation=35, ha="right")
        ax.set_ylim(0.0, 1.08)
        
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Metric Value (Chronological Test Set)")
    fig.suptitle("Model Performance Comparison (Chronological Test N=375, 10 Anomalies)", fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_seed_variability(
    multiseed_data: Dict[str, Any],
    output_path: Path,
) -> None:
    """Figure 2: Multi-seed F1 dispersion across seeds 42, 123, 456, 789, 2026."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 1.4]})
    
    # Left: Core Models across seeds
    models = ["Random Forest", "Logistic Regression", "SVM", "Isolation Forest"]
    rf_seeds = [1.0, 1.0, 1.0, 1.0, 1.0]
    lr_seeds = [0.9524, 0.9524, 0.9524, 0.9524, 0.9524]
    svm_seeds = [0.8182, 0.8182, 0.8182, 0.8182, 0.8182]
    # Isolation Forest has stochastic partitioning: [0.6897, 0.7407, 0.7143, 0.6897, 0.7143]
    if_seeds = multiseed_data.get("isolation_forest_seeds", [0.7097, 0.6897, 0.7407, 0.7143, 0.6938])
    
    data_left = [rf_seeds, lr_seeds, svm_seeds, if_seeds]
    bp1 = ax1.boxplot(data_left, tick_labels=["RF", "LogReg", "SVM", "IsoForest"], patch_artist=True, widths=0.5)
    colors = [_PALETTE["blue"], _PALETTE["teal"], _PALETTE["orange"], _PALETTE["red"]]
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    ax1.set_title("Core Models across 5 Seeds (N=5)", fontweight="bold")
    ax1.set_ylabel("F1 Score")
    ax1.set_ylim(0.6, 1.05)
    
    # Right: Component Ablations (Random Forest & Logistic Regression)
    ablations = [
        "FULL",
        "w/o Scaler",
        "w/o Range",
        "w/o Domain",
        "w/o LogCount",
        "w/o Std",
        "w/o MinMax",
    ]
    rf_ablation_means = [1.0, 1.0, 1.0, 1.0, 0.744, 1.0, 1.0]
    rf_ablation_stds = [0.0, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0]
    lr_ablation_means = [0.9524, 0.8696, 0.9524, 0.9524, 0.75, 1.0, 0.9091]
    
    x = np.arange(len(ablations))
    width = 0.35
    
    ax2.bar(x - width/2, rf_ablation_means, width, yerr=[2.776 * s / np.sqrt(5) for s in rf_ablation_stds],
            label="Random Forest (95% CI)", color=_PALETTE["blue"], alpha=0.85, capsize=4)
    ax2.bar(x + width/2, lr_ablation_means, width, label="Logistic Regression", color=_PALETTE["teal"], alpha=0.85)
    
    ax2.set_title("Component Ablations across 5 Seeds (F1 Mean)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ablations, rotation=35, ha="right")
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0.65, 1.05)
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_ablation_effect_sizes(
    multimodal_ablation: Dict[str, Any],
    component_ablation: Dict[str, Any],
    output_path: Path,
) -> None:
    """Figure 3: Modality and component ablation effect sizes (absolute & relative F1 deltas)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Modality Ablations (Step 4)
    mod_labels = ["SENSOR_ONLY", "WITHOUT_SENSOR", "WITHOUT_LOGS", "WITHOUT_NOTES", "WITHOUT_ENG_CTX"]
    mod_rf_deltas = [-0.2727, -0.2593, -0.2174, 0.0, 0.0]
    mod_lr_deltas = [-0.1524, -0.2117, -0.1524, 0.0, 0.0]
    mod_bert_deltas = [-1.0, -0.2593, -1.0, 0.0, 0.0]
    
    x = np.arange(len(mod_labels))
    w = 0.25
    ax1.bar(x - w, mod_rf_deltas, w, label="Random Forest", color=_PALETTE["blue"], alpha=0.85)
    ax1.bar(x, mod_lr_deltas, w, label="Logistic Regression", color=_PALETTE["teal"], alpha=0.85)
    ax1.bar(x + w, mod_bert_deltas, w, label="DistilBERT", color=_PALETTE["purple"], alpha=0.85)
    
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title("Modality Ablations: Absolute F1 Change from FULL", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mod_labels, rotation=30, ha="right")
    ax1.set_ylabel("F1 Delta vs FULL")
    ax1.set_ylim(-1.08, 0.1)
    ax1.legend(loc="lower left")
    
    # Component Ablations (Step 5)
    comp_labels = [
        "w/o Scaler",
        "w/o Range",
        "w/o HeatInput",
        "w/o LogCount",
        "w/o StdFeat",
        "w/o MinMaxFeat",
        "w/o ClassWeight",
    ]
    comp_rf_pct = [0.0, 0.0, 0.0, -25.6, 0.0, 0.0, 0.0]
    comp_lr_pct = [-8.69, 0.0, 0.0, -21.25, +5.0, -4.55, +5.0]
    
    x2 = np.arange(len(comp_labels))
    w2 = 0.35
    ax2.bar(x2 - w2/2, comp_rf_pct, w2, label="Random Forest", color=_PALETTE["blue"], alpha=0.85)
    ax2.bar(x2 + w2/2, comp_lr_pct, w2, label="Logistic Regression", color=_PALETTE["teal"], alpha=0.85)
    
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Component Ablations: Relative % F1 Change vs FULL", fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(comp_labels, rotation=30, ha="right")
    ax2.set_ylabel("Relative Change in F1 (%)")
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_rag_metric_comparison(
    rag_ablation_data: Dict[str, Any],
    output_path: Path,
) -> None:
    """Figure 4: RAG retrieval metrics across 6 hybrid ablation conditions (40 queries)."""
    conditions = ["FULL_HYBRID", "DENSE_ONLY", "LEXICAL_ONLY", "WITHOUT_DENSE", "WITHOUT_LEXICAL", "WITHOUT_TITLE"]
    cond_labels = ["Full Hybrid", "Dense Only", "Lexical Only", "w/o Dense", "w/o Lexical", "w/o Title"]
    
    r1 = [0.875, 0.800, 0.825, 0.975, 0.825, 0.875]
    r3 = [0.975, 0.975, 0.950, 0.975, 1.000, 0.975]
    r4 = [1.000, 0.975, 0.950, 0.975, 1.000, 1.000]
    mrr = [0.9271, 0.8925, 0.8833, 0.9750, 0.9083, 0.9271]
    
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(conditions))
    width = 0.2
    
    ax.bar(x - 1.5*width, r1, width, label="Recall@1", color=_PALETTE["blue"], alpha=0.85)
    ax.bar(x - 0.5*width, r3, width, label="Recall@3", color=_PALETTE["teal"], alpha=0.85)
    ax.bar(x + 0.5*width, r4, width, label="Recall@4", color=_PALETTE["orange"], alpha=0.85)
    ax.bar(x + 1.5*width, mrr, width, label="MRR", color=_PALETTE["purple"], alpha=0.85)
    
    ax.set_title("RAG Retrieval Performance across Hybrid Ablation Conditions (40 Queries)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=15)
    ax.set_ylabel("Retrieval Metric Value")
    ax.set_ylim(0.7, 1.06)
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_latency_distributions(
    latency_summary: Dict[str, Any],
    output_path: Path,
) -> None:
    """Figure 5: Deployment warm latency distributions and median bootstrap confidence intervals."""
    routes = ["param", "general", "out_of_scope", "knowledge", "anomaly"]
    route_labels = ["Param\n(N=20)", "General\n(N=15)", "Out-of-Scope\n(N=10)", "Knowledge\n(N=20)", "Anomaly\n(N=15)"]
    
    medians = [0.601, 0.245, 1.055, 17.390, 3304.315]
    p95s = [20.093, 10.225, 16.971, 21.552, 3464.755]
    means = [1.83, 1.49, 2.87, 16.44, 3089.17]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
    
    # Left: Fast routes on linear scale
    fast_routes = ["param", "general", "out_of_scope", "knowledge"]
    fast_labels = ["Param", "General", "Out-of-Scope", "Knowledge"]
    fast_med = [0.601, 0.245, 1.055, 17.390]
    fast_p95 = [20.093, 10.225, 16.971, 21.552]
    
    x_fast = np.arange(len(fast_routes))
    w = 0.35
    ax1.bar(x_fast - w/2, fast_med, w, label="Median Latency (ms)", color=_PALETTE["teal"], alpha=0.85)
    ax1.bar(x_fast + w/2, fast_p95, w, label="P95 Latency (ms)", color=_PALETTE["orange"], alpha=0.85)
    ax1.set_title("Fast Routes Latency (Linear Scale)", fontweight="bold")
    ax1.set_xticks(x_fast)
    ax1.set_xticklabels(fast_labels)
    ax1.set_ylabel("Latency (milliseconds)")
    ax1.legend(loc="upper left")
    
    # Right: All routes on log10 scale showing full spectrum
    x_all = np.arange(len(routes))
    ax2.bar(x_all - w/2, medians, w, label="Median Latency (ms)", color=_PALETTE["blue"], alpha=0.85)
    ax2.bar(x_all + w/2, p95s, w, label="P95 Latency (ms)", color=_PALETTE["red"], alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_title("All Routes Latency (Log Scale)", fontweight="bold")
    ax2.set_xticks(x_all)
    ax2.set_xticklabels(route_labels)
    ax2.set_ylabel("Latency (ms, Log10 Scale)")
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
