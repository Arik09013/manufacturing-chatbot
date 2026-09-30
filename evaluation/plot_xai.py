"""
Visualization Suite for Explainability (XAI) Evaluation (Step 9).

Generates publication-quality figures saved to evaluation/artifacts/figures/:
  1. xai_deletion_curves.png   - Progressive feature deletion fidelity curves
  2. xai_stability.png         - Explanation stability under small input perturbations
  3. xai_method_agreement.png  - SHAP vs LIME agreement (top-k Jaccard & Spearman correlation)
  4. xai_normal_vs_anomaly.png - Structural differences between normal and anomaly explanations
  5. xai_latency.png           - Latency benchmarks across explainers (mean, median, p95)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_FIG_DIR = Path(__file__).resolve().parent / "artifacts" / "figures"


def plot_deletion_curves(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot progressive feature deletion curves (probability drop vs k masked features)."""
    fig_path = output_path or (_DEFAULT_FIG_DIR / "xai_deletion_curves.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    shap_del = results["fidelity"]["shap"]
    lime_del = results["fidelity"]["lime"]
    k_levels = shap_del["k_levels"]

    shap_means = [shap_del["mean_delta_by_k"][k] for k in k_levels]
    lime_means = [lime_del["mean_delta_by_k"][k] for k in k_levels]
    rand_means = [shap_del["random_baseline"]["mean_delta_by_k"][k] for k in k_levels]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    ax.plot(k_levels, shap_means, marker="o", linewidth=2.2, color="#1f77b4", label="SHAP Deletion Sensitivity")
    ax.plot(k_levels, lime_means, marker="s", linewidth=2.2, color="#ff7f0e", linestyle="--", label="LIME Deletion Sensitivity")
    ax.plot(k_levels, rand_means, marker="^", linewidth=1.8, color="#7f7f7f", linestyle=":", label="Random Feature Deletion (Baseline)")

    ax.set_title("Explanation Fidelity: Progressive Feature Deletion", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Top Ranked Features Masked (k)", fontsize=11)
    ax.set_ylabel("Mean Probability Drop (Δp = p0 - pk)", fontsize=11)
    ax.set_xticks(k_levels)
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    # Annotate AUC values
    auc_text = (
        f"AUC Metrics (Normalized):\n"
        f"  SHAP AUC: {shap_del['auc_mean']:.3f}\n"
        f"  LIME AUC: {lime_del['auc_mean']:.3f}\n"
        f"  Random AUC: {shap_del['random_baseline']['auc_mean']:.3f}"
    )
    ax.text(
        0.04, 0.55, auc_text, transform=ax.transAxes,
        fontsize=9.5, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da")
    )

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    logger.info("Saved deletion curves figure to %s", fig_path)
    return fig_path


def plot_stability(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot explanation stability under input perturbation (Jaccard similarity and rank correlation)."""
    fig_path = output_path or (_DEFAULT_FIG_DIR / "xai_stability.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    shap_stab = results["stability"]["shap"]
    lime_stab = results["stability"]["lime"]
    att_stab = results["stability"]["attention"]

    k_levels = [1, 3, 5, 10]
    shap_jacc = [shap_stab["jaccard_mean_by_k"][k] for k in k_levels]
    lime_jacc = [lime_stab["jaccard_mean_by_k"][k] for k in k_levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # Left: Jaccard similarity across k
    x = np.arange(len(k_levels))
    width = 0.35
    ax1.bar(x - width/2, shap_jacc, width, label="SHAP", color="#1f77b4", edgecolor="#135380")
    ax1.bar(x + width/2, lime_jacc, width, label="LIME", color="#ff7f0e", edgecolor="#b85805")

    ax1.set_title("Top-k Jaccard Stability under Perturbation", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Top-k Features", fontsize=10.5)
    ax1.set_ylabel("Mean Jaccard Similarity", fontsize=10.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"k={k}" for k in k_levels])
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(frameon=True, fontsize=10)

    # Right: Spearman Rank Correlation Comparison
    models = ["SHAP (RF)", "LIME (RF)"]
    corrs = [shap_stab["spearman_mean"], lime_stab["spearman_mean"]]
    errors = [shap_stab["spearman_std"], lime_stab["spearman_std"]]
    colors = ["#1f77b4", "#ff7f0e"]

    bars = ax2.bar(models, corrs, yerr=errors, capsize=6, color=colors, width=0.5, edgecolor="#333333")
    ax2.set_title("Feature-Ranking Spearman Correlation", fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("Mean Spearman Rank Correlation (ρ)", fontsize=10.5)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Add text labels on bars
    for bar, val in zip(bars, corrs):
        ax2.text(bar.get_x() + bar.get_width()/2.0, val / 2.0, f"ρ={val:.3f}", ha="center", va="center", color="white", fontweight="bold")

    # Note attention result in footnote
    if "jaccard_mean_by_k" in att_stab:
        att_note = f"DistilBERT Attention Token Jaccard: k=3: {att_stab['jaccard_mean_by_k'][3]:.2f}, k=5: {att_stab['jaccard_mean_by_k'][5]:.2f}"
    else:
        att_note = "DistilBERT Attention: Evaluated on text notes"
    fig.text(0.5, 0.01, att_note, ha="center", fontsize=9, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(fig_path)
    plt.close()
    logger.info("Saved stability figure to %s", fig_path)
    return fig_path


def plot_method_agreement(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot agreement between SHAP and LIME across evaluation samples."""
    fig_path = output_path or (_DEFAULT_FIG_DIR / "xai_method_agreement.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    agreement = results["agreement"]["shap_vs_lime"]
    k_levels = [1, 3, 5, 10]
    jacc_means = [agreement["jaccard_mean_by_k"][k] for k in k_levels]
    overlap_means = [agreement["overlap_mean_by_k"][k] for k in k_levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # Left: Top-k overlap and Jaccard
    x = np.arange(len(k_levels))
    ax1.plot(x, jacc_means, marker="o", linewidth=2.2, color="#2ca02c", label="Top-k Jaccard Similarity")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"k={k}" for k in k_levels])
    ax1.set_title("SHAP vs LIME: Top-k Feature Overlap", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Top-k Features", fontsize=10.5)
    ax1.set_ylabel("Jaccard Similarity", fontsize=10.5)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.5)

    for i, (k, jm, om) in enumerate(zip(k_levels, jacc_means, overlap_means)):
        ax1.annotate(f"J={jm:.2f}\n({om:.1f}/{k})", (x[i], jm), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    # Right: Distribution of Spearman Rank Correlations
    spearman_vals = agreement["per_sample_spearman"]
    ax2.hist(spearman_vals, bins=12, color="#17becf", edgecolor="#0e6872", alpha=0.85)
    ax2.axvline(np.mean(spearman_vals), color="#d62728", linestyle="--", linewidth=2, label=f"Mean ρ = {np.mean(spearman_vals):.3f}")
    ax2.axvline(np.median(spearman_vals), color="#9467bd", linestyle=":", linewidth=2, label=f"Median ρ = {np.median(spearman_vals):.3f}")

    ax2.set_title("SHAP vs LIME: Full Rank Correlation (N=60)", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Spearman Rank Correlation (ρ)", fontsize=10.5)
    ax2.set_ylabel("Sample Count", fontsize=10.5)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(frameon=True, fontsize=9.5)

    note = "Attention comparison marked N/A (modal mismatch: text tokens vs tabular sensor/log features)"
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(fig_path)
    plt.close()
    logger.info("Saved agreement figure to %s", fig_path)
    return fig_path


def plot_normal_vs_anomaly(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot structural differences between normal and anomalous explanations."""
    fig_path = output_path or (_DEFAULT_FIG_DIR / "xai_normal_vs_anomaly.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    norm_stats = results["normal_vs_anomaly"]["normal"]
    ano_stats = results["normal_vs_anomaly"]["anomalous"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), dpi=300)

    # Left: Structural Concentration & Entropy
    metrics = ["Top-1 Conc.", "Top-3 Conc.", "Top-5 Conc.", "k50 Count"]
    norm_vals = [
        norm_stats["top1_concentration_mean"],
        norm_stats["top3_concentration_mean"],
        norm_stats["top5_concentration_mean"],
        norm_stats["k50_mean"] / 40.0,  # normalized by total features
    ]
    ano_vals = [
        ano_stats["top1_concentration_mean"],
        ano_stats["top3_concentration_mean"],
        ano_stats["top5_concentration_mean"],
        ano_stats["k50_mean"] / 40.0,
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width/2, norm_vals, width, label="Normal Samples", color="#2ca02c", edgecolor="#1b631b")
    ax1.bar(x + width/2, ano_vals, width, label="Anomalous Samples", color="#d62728", edgecolor="#821415")

    ax1.set_title("Explanation Concentration & Parsimony", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Ratio / Normalized Score", fontsize=10.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(frameon=True, fontsize=10)

    # Right: Domain Feature Group Shares
    groups = ["current", "voltage", "speed", "wire_feed", "gas", "heat", "logs", "notes"]
    norm_groups = [norm_stats["group_shares_mean"].get(g, 0.0) * 100 for g in groups]
    ano_groups = [ano_stats["group_shares_mean"].get(g, 0.0) * 100 for g in groups]

    y = np.arange(len(groups))
    height = 0.35
    ax2.barh(y - height/2, norm_groups, height, label="Normal", color="#2ca02c", edgecolor="#1b631b")
    ax2.barh(y + height/2, ano_groups, height, label="Anomalous", color="#d62728", edgecolor="#821415")

    ax2.set_title("Dominant Feature Group Share (%)", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Mean Importance Share (%)", fontsize=10.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(groups, fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="x")
    ax2.legend(frameon=True, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    logger.info("Saved normal vs anomaly figure to %s", fig_path)
    return fig_path


def plot_latency(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot computational latency benchmarks across explainers."""
    fig_path = output_path or (_DEFAULT_FIG_DIR / "xai_latency.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    lat_data = results["computational_cost"]
    explainers = ["SHAP (RF)", "LIME (RF)", "Attention (DistilBERT)"]
    keys = ["shap", "lime", "attention"]

    means = [lat_data[k]["mean_latency_ms"] for k in keys]
    medians = [lat_data[k]["median_latency_ms"] for k in keys]
    p95s = [lat_data[k]["p95_latency_ms"] for k in keys]

    x = np.arange(len(explainers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)

    ax.bar(x - width, means, width, label="Mean Latency (ms)", color="#3470a3", edgecolor="#1e4463")
    ax.bar(x, medians, width, label="Median Latency (ms)", color="#48a868", edgecolor="#2d6e42")
    ax.bar(x + width, p95s, width, label="p95 Latency (ms)", color="#e07a38", edgecolor="#8c471c")

    ax.set_title("Explanation Latency Benchmark (Log Scale)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Latency per Explanation (milliseconds, log scale)", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(explainers, fontsize=10.5)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5, which="both", axis="y")
    ax.legend(frameon=True, fontsize=10)

    # Annotate absolute values
    for i in range(len(explainers)):
        ax.text(x[i] - width, means[i] * 1.15, f"{means[i]:.1f}", ha="center", fontsize=8.5, rotation=45)
        ax.text(x[i], medians[i] * 1.15, f"{medians[i]:.1f}", ha="center", fontsize=8.5, rotation=45)
        ax.text(x[i] + width, p95s[i] * 1.15, f"{p95s[i]:.1f}", ha="center", fontsize=8.5, rotation=45)

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    logger.info("Saved latency figure to %s", fig_path)
    return fig_path


def generate_all_xai_plots(
    results: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate and save all 5 XAI evaluation figures."""
    out_dir = output_dir or _DEFAULT_FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "xai_deletion_curves": plot_deletion_curves(results, out_dir / "xai_deletion_curves.png"),
        "xai_stability": plot_stability(results, out_dir / "xai_stability.png"),
        "xai_method_agreement": plot_method_agreement(results, out_dir / "xai_method_agreement.png"),
        "xai_normal_vs_anomaly": plot_normal_vs_anomaly(results, out_dir / "xai_normal_vs_anomaly.png"),
        "xai_latency": plot_latency(results, out_dir / "xai_latency.png"),
    }
