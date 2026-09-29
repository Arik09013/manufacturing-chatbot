"""
Publication-Quality Figures for Step 6: DistilBERT vs Conventional Models.

Generates:
  1. model_comparison_f1.png - Chronological vs LOMO Macro F1.
  2. model_comparison_prauc.png - Chronological vs LOMO Macro PR-AUC.
  3. model_comparison_rocauc.png - ROC Curves on Chronological Test Set.
  4. model_comparison_multiseed.png - Multi-seed F1 (mean +/- std).

Outputs saved to evaluation/results/comparison/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
RESULTS_DIR = _ROOT / "evaluation" / "results" / "comparison"
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


def plot_f1_comparison(summary: dict):
    setup_style()
    chrono = summary["chronological_summary"]
    lomo = summary["lomo_summary"]

    models = [
        "Random Forest",
        "Logistic Regression",
        "Support Vector Machine",
        "DistilBERT",
        "Isolation Forest",
    ]

    chrono_f1 = [
        chrono.get(m, {}).get("f1")
        if m in chrono
        else chrono.get("DistilBERT (theta=0.5)", {}).get("f1", 0.0)
        for m in models
    ]
    lomo_f1 = [lomo.get(m, {}).get("macro_f1", 0.0) for m in models]

    labels = [
        "Random Forest\n(Supervised)",
        "Logistic Reg\n(Supervised)",
        "SVM (RBF)\n(Supervised)",
        "DistilBERT\n(Text LLM)",
        "Isolation Forest\n(Unsupervised)",
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects1 = ax.bar(x - width / 2, chrono_f1, width, label="Chronological Holdout (80/20)", color="#1f77b4")
    rects2 = ax.bar(x + width / 2, lomo_f1, width, label="Leave-One-Machine-Out (LOMO)", color="#2ca02c")

    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison: DistilBERT vs Conventional Tabular Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            if not np.isnan(h) and h > 0.01:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    out = FIGURES_DIR / "model_comparison_f1.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved -> {out}")


def plot_prauc_comparison(summary: dict):
    setup_style()
    chrono = summary["chronological_summary"]
    lomo = summary["lomo_summary"]

    models = [
        "Random Forest",
        "Logistic Regression",
        "Support Vector Machine",
        "DistilBERT",
        "Isolation Forest",
    ]

    chrono_pr = [
        chrono.get(m, {}).get("pr_auc")
        if m in chrono
        else chrono.get("DistilBERT (theta=0.5)", {}).get("pr_auc", 0.0)
        for m in models
    ]
    lomo_pr = [lomo.get(m, {}).get("macro_pr_auc", 0.0) for m in models]

    labels = [
        "Random Forest\n(Supervised)",
        "Logistic Reg\n(Supervised)",
        "SVM (RBF)\n(Supervised)",
        "DistilBERT\n(Text LLM)",
        "Isolation Forest\n(Unsupervised)",
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects1 = ax.bar(x - width / 2, chrono_pr, width, label="Chronological Holdout (80/20)", color="#1f77b4")
    rects2 = ax.bar(x + width / 2, lomo_pr, width, label="Leave-One-Machine-Out (LOMO)", color="#2ca02c")

    ax.set_ylabel("PR-AUC (Average Precision)")
    ax.set_title("PR-AUC Comparison: DistilBERT vs Conventional Tabular Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            if not np.isnan(h) and h > 0.01:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    out = FIGURES_DIR / "model_comparison_prauc.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved -> {out}")


def plot_multiseed_robustness(summary: dict):
    setup_style()
    mseed = summary.get("multiseed_summary", {})
    if not mseed:
        return

    models = [
        "Random Forest",
        "Logistic Regression",
        "Support Vector Machine",
        "DistilBERT",
        "Isolation Forest",
    ]

    f1_means = [mseed.get(m, {}).get("f1_mean", 0.0) for m in models]
    f1_stds = [mseed.get(m, {}).get("f1_std", 0.0) for m in models]
    prauc_means = [mseed.get(m, {}).get("prauc_mean", 0.0) or 0.0 for m in models]

    labels = [
        "Random Forest",
        "Logistic Reg",
        "SVM (RBF)",
        "DistilBERT",
        "Isolation Forest",
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.bar(
        x - width / 2,
        f1_means,
        width,
        yerr=f1_stds,
        capsize=4,
        label="F1 Score (Mean ± Std)",
        color="#1f77b4",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        prauc_means,
        width,
        label="PR-AUC (Mean)",
        color="#ff7f0e",
        alpha=0.85,
    )

    ax.set_ylabel("Metric Value")
    ax.set_title("Multi-Seed Robustness across Seeds [42, 123, 456, 789, 2026]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = FIGURES_DIR / "model_comparison_multiseed.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved -> {out}")


def plot_roc_curves():
    """Plot ROC curves on chronological test data."""
    setup_style()
    from src.data.splits import prepare_tabular_data, split_chronological
    from src.fusion.fuse import load_fused
    from src.model.baselines import get_baseline_registry
    from src.model.bert_detector import BertFaultDetector
    from sklearn.metrics import roc_curve, auc

    df = load_fused()
    train_df, test_df, _ = split_chronological(df, train_ratio=0.8, embargo_minutes=30)
    X_tr, y_tr, X_te, y_te, _, _ = prepare_tabular_data(train_df, test_df, scale=True)

    registry = get_baseline_registry()
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "Random Forest": "#1f77b4",
        "Logistic Regression": "#2ca02c",
        "Support Vector Machine": "#9467bd",
        "Isolation Forest": "#7f7f7f",
        "DistilBERT": "#d62728",
    }

    for name in ["Random Forest", "Logistic Regression", "Support Vector Machine", "Isolation Forest"]:
        spec = registry[name]
        clf = spec.instantiate()
        clf.fit(X_tr, y_tr) if spec.is_supervised else clf.fit(X_tr)
        if spec.is_supervised:
            prob = clf.predict_proba(X_te)[:, 1]
        else:
            scores = clf.score_samples(X_te)
            prob = 1 / (1 + np.exp(scores * 5))
        fpr, tpr, _ = roc_curve(y_te, prob)
        roc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.4f})", color=colors[name], lw=1.8)

    # DistilBERT
    det = BertFaultDetector()
    bert_prob = det.predict_proba(test_df)
    fpr_b, tpr_b, _ = roc_curve(y_te, bert_prob)
    roc_b = auc(fpr_b, tpr_b)
    ax.plot(fpr_b, tpr_b, label=f"DistilBERT (AUC = {roc_b:.4f})", color=colors["DistilBERT"], lw=1.8, linestyle="--")

    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.7, label="Chance Level")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("ROC Curves: DistilBERT vs Conventional Models (Chronological Test)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = FIGURES_DIR / "model_comparison_rocauc.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved -> {out}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "model_comparison_summary.json"
    if not summary_path.exists():
        print(f"Error: {summary_path} not found. Run evaluation/eval_model_comparison.py first.")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    plot_f1_comparison(summary)
    plot_prauc_comparison(summary)
    plot_multiseed_robustness(summary)
    plot_roc_curves()
    print(f"All comparison figures generated successfully in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
