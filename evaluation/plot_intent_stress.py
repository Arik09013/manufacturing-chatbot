"""
Publication-Quality Figures for Intent Classification Stress Evaluation (Step 8).

Generates:
  1. intent_confusion_matrix.png: 5x5 annotated confusion matrix heatmap
  2. intent_f1_by_category.png: Precision, Recall, and F1 across the 5 intent classes
  3. intent_perturbation_robustness.png: Accuracy and degradation across 12 perturbation types

Outputs are saved in:
  - evaluation/artifacts/figures/
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
RESULTS_FILE = _ROOT / "evaluation" / "artifacts" / "intent_stress_results.json"

FIGURE_DIRS = [
    _ROOT / "evaluation" / "artifacts" / "figures",
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
# Figure 1: 5x5 Confusion Matrix
# =====================================================================

def plot_confusion_matrix(data: Dict[str, Any]):
    setup_style()
    cm_data = data["confusion_matrix_5x5"]
    labels = cm_data["labels"]
    matrix = np.array(cm_data["matrix"], dtype=int)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Number of Queries", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([f"`{l}`" for l in labels], rotation=30, ha="right", fontweight="semibold")
    ax.set_yticklabels([f"`{l}`" for l in labels], fontweight="semibold")

    ax.set_xlabel("Predicted Intent Class", fontweight="bold")
    ax.set_ylabel("Expected (True) Intent Class", fontweight="bold")
    ax.set_title("5x5 Confusion Matrix: Intent Classification Stress Benchmark (N=100)", fontweight="bold")

    # Threshold for text color contrast
    thresh = matrix.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            color = "white" if val > thresh else "black"
            fontweight = "bold" if val > 0 else "normal"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight=fontweight, fontsize=10)

    ax.grid(False)
    save_figure(fig, "intent_confusion_matrix.png")


# =====================================================================
# Figure 2: Precision, Recall, and F1 by Intent Category
# =====================================================================

def plot_f1_by_intent(data: Dict[str, Any]):
    setup_style()
    multi = data["multi_class_intent_metrics"]
    labels = list(multi["per_class"].keys())

    precisions = [multi["per_class"][l]["precision"] for l in labels]
    recalls = [multi["per_class"][l]["recall"] for l in labels]
    f1s = [multi["per_class"][l]["f1"] for l in labels]

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5.5))

    r1 = ax.bar(x - width, precisions, width, label="Precision", color="#1f77b4", edgecolor="black", linewidth=0.5)
    r2 = ax.bar(x, recalls, width, label="Recall", color="#2ca02c", edgecolor="black", linewidth=0.5)
    r3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#e67e22", edgecolor="black", linewidth=0.5)

    for group in [r1, r2, r3]:
        for r in group:
            h = r.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Metric Score")
    ax.set_title(f"Classification Metrics by Intent Category (Macro F1 = {multi['macro_f1']:.4f}, Acc = {multi['accuracy']:.4f})", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"`{l}`" for l in labels], fontweight="semibold")
    ax.set_ylim(0, 1.18)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)

    save_figure(fig, "intent_f1_by_category.png")


# =====================================================================
# Figure 3: Performance & Degradation by Perturbation Type
# =====================================================================

def plot_perturbation_robustness(data: Dict[str, Any]):
    setup_style()
    perts = data["perturbation_breakdown"]
    types = list(perts.keys())
    readable_types = [t.replace("_", " ").title() for t in types]

    accuracies = [perts[t]["accuracy"] * 100 for t in types]
    degradations = [perts[t]["degradation_vs_clean"] * 100 for t in types]

    y = np.arange(len(types))
    height = 0.6

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Color bars based on degradation
    bar_colors = ["#2ca02c" if d <= 0 else ("#e67e22" if d < 20 else "#d62728") for d in degradations]
    rects = ax.barh(y, accuracies, height, color=bar_colors, edgecolor="black", linewidth=0.5)

    for i, r in enumerate(rects):
        w = r.get_width()
        deg = degradations[i]
        deg_label = f"({-deg:+.1f}%)" if deg != 0 else "(Ref)"
        ax.annotate(f"{w:.1f}% {deg_label}",
                    xy=(w, r.get_y() + r.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left", va="center", fontsize=8.5)

    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Intent Classification Robustness Across Operator Linguistic Perturbations", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(readable_types, fontweight="semibold")
    ax.set_xlim(0, 115)
    ax.axvline(100.0, color="gray", linestyle=":", alpha=0.6)
    ax.invert_yaxis()

    save_figure(fig, "intent_perturbation_robustness.png")


def generate_all_figures():
    """Load results and generate all 3 figures."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_FILE}. Run eval_intent_stress.py first.")

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    plot_confusion_matrix(data)
    plot_f1_by_intent(data)
    plot_perturbation_robustness(data)
    print("All 3 intent classification stress figures generated successfully.")


if __name__ == "__main__":
    generate_all_figures()
