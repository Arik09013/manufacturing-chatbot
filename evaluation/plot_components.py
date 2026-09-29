"""
Publication-Quality Figures for Component Ablation Study.

Generates:
  1. Chronological F1 score comparison across component ablations.
  2. Chronological PR-AUC comparison across component ablations.
  3. Leave-One-Machine-Out (LOMO) Macro F1 comparison.
  4. Multi-seed F1 mean +/- std across independent seeds.

Outputs saved to evaluation/results/components/figures/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).parent.parent
RESULTS_DIR = _ROOT / "evaluation" / "results" / "components"
FIGURES_DIR = RESULTS_DIR / "figures"


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_chronological_f1(summary: dict):
    setup_style()
    chrono = summary["chronological_summary"]
    conditions = list(chrono.keys())

    # Filter readable labels
    labels = [c.replace("WITHOUT_", "W/O ").replace("_", " ") for c in conditions]

    rf_f1 = [chrono[c].get("Random Forest", {}).get("f1", 0.0) for c in conditions]
    lr_f1 = [chrono[c].get("Logistic Regression", {}).get("f1", 0.0) for c in conditions]
    bert_f1 = [chrono[c].get("DistilBERT", {}).get("f1", 0.0) if "DistilBERT" in chrono[c] else np.nan for c in conditions]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width, rf_f1, width, label="Random Forest", color="#1f77b4")
    rects2 = ax.bar(x, lr_f1, width, label="Logistic Regression", color="#2ca02c")
    rects3 = ax.bar(x + width, bert_f1, width, label="DistilBERT", color="#ff7f0e")

    ax.set_ylabel("F1 Score")
    ax.set_title("Chronological Holdout F1 Score across Pipeline Component Ablations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    # Annotate bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            if not np.isnan(h) and h > 0.01:
                ax.annotate(
                    f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    out = FIGURES_DIR / "f1_by_component_chronological.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved -> {out}")


def plot_chronological_prauc(summary: dict):
    setup_style()
    chrono = summary["chronological_summary"]
    conditions = list(chrono.keys())
    labels = [c.replace("WITHOUT_", "W/O ").replace("_", " ") for c in conditions]

    rf_pr = [chrono[c].get("Random Forest", {}).get("pr_auc", 0.0) or 0.0 for c in conditions]
    lr_pr = [chrono[c].get("Logistic Regression", {}).get("pr_auc", 0.0) or 0.0 for c in conditions]
    bert_pr = [chrono[c].get("DistilBERT", {}).get("pr_auc", 0.0) if "DistilBERT" in chrono[c] else np.nan for c in conditions]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, rf_pr, width, label="Random Forest", color="#1f77b4")
    ax.bar(x, lr_pr, width, label="Logistic Regression", color="#2ca02c")
    ax.bar(x + width, bert_pr, width, label="DistilBERT", color="#ff7f0e")

    ax.set_ylabel("PR-AUC (Average Precision)")
    ax.set_title("Chronological Holdout PR-AUC across Pipeline Component Ablations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = FIGURES_DIR / "prauc_by_component_chronological.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved -> {out}")


def plot_lomo_macro_f1(summary: dict):
    setup_style()
    lomo = summary["lomo_summary"]
    conditions = list(lomo.keys())
    labels = [c.replace("WITHOUT_", "W/O ").replace("_", " ") for c in conditions]

    rf_f1 = [lomo[c].get("Random Forest", {}).get("macro_f1", 0.0) for c in conditions]
    lr_f1 = [lomo[c].get("Logistic Regression", {}).get("macro_f1", 0.0) for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))

    rects1 = ax.bar(x - width / 2, rf_f1, width, label="Random Forest", color="#1f77b4")
    rects2 = ax.bar(x + width / 2, lr_f1, width, label="Logistic Regression", color="#2ca02c")

    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Leave-One-Machine-Out (LOMO) Cross-Validation Macro F1 across Component Ablations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            if not np.isnan(h) and h > 0.01:
                ax.annotate(
                    f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    out = FIGURES_DIR / "f1_by_component_lomo.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved -> {out}")


def plot_multiseed_robustness(summary: dict):
    setup_style()
    mseed = summary.get("multiseed_summary", {})
    if not mseed:
        return

    conditions = list(mseed.keys())
    labels = [c.replace("WITHOUT_", "W/O ").replace("_", " ") for c in conditions]

    rf_means = [mseed[c]["Random Forest"]["f1_mean"] for c in conditions]
    rf_stds = [mseed[c]["Random Forest"]["f1_std"] for c in conditions]

    lr_means = [mseed[c]["Logistic Regression"]["f1_mean"] for c in conditions]
    lr_stds = [mseed[c]["Logistic Regression"]["f1_std"] for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.bar(
        x - width / 2,
        rf_means,
        width,
        yerr=rf_stds,
        capsize=4,
        label="Random Forest (5 Seeds)",
        color="#1f77b4",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        lr_means,
        width,
        yerr=lr_stds,
        capsize=4,
        label="Logistic Regression (5 Seeds)",
        color="#2ca02c",
        alpha=0.85,
    )

    ax.set_ylabel("F1 Score (Mean ± Std)")
    ax.set_title("Multi-Seed Stability (Seeds: 42, 123, 456, 789, 2026)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = FIGURES_DIR / "f1_multiseed_by_component.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved -> {out}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "component_ablation_summary.json"
    if not summary_path.exists():
        print(f"Error: {summary_path} not found. Run evaluation/eval_components.py first.")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    plot_chronological_f1(summary)
    plot_chronological_prauc(summary)
    plot_lomo_macro_f1(summary)
    plot_multiseed_robustness(summary)
    print(f"All component figures generated successfully in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
