"""
Visualization generator for multimodal ablation experiments.

Generates:
  1. f1_by_ablation_chronological.png: Grouped comparison of F1 across conditions
  2. prauc_by_ablation_chronological.png: Grouped comparison of PR-AUC across conditions
  3. f1_by_ablation_lomo.png: LOMO macro F1 comparison across conditions
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results" / "ablation"
FIG_DIR = RESULTS_DIR / "figures"


def load_data():
    json_path = RESULTS_DIR / "multimodal_ablation_results.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chronological"], data["lomo"]


def plot_chronological_f1(chrono_data):
    conditions = ["FULL", "SENSOR_ONLY", "WITHOUT_SENSOR", "WITHOUT_LOGS", "WITHOUT_OPERATOR_NOTES"]
    labels = ["Full (All)", "Sensor Only", "w/o Sensor", "w/o Logs", "w/o Notes"]

    models = ["Random Forest", "Logistic Regression", "DistilBERT"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    plt.figure(figsize=(9, 5), dpi=300)
    x = np.arange(len(conditions))
    width = 0.25

    for i, (m_name, color) in enumerate(zip(models, colors)):
        vals = []
        for c in conditions:
            m_res = chrono_data["conditions"][c]["models"].get(m_name)
            vals.append(m_res["f1"] if m_res else 0.0)
        plt.bar(x + (i - 1) * width, vals, width, label=m_name, color=color, alpha=0.88)

    plt.title("Chronological F1 Score by Modality Ablation Condition", fontsize=12, fontweight="bold")
    plt.xticks(x, labels, fontsize=10)
    plt.ylabel("F1 Score", fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    out = FIG_DIR / "f1_by_ablation_chronological.png"
    plt.savefig(out)
    plt.close()
    print("Saved ->", out)


def plot_chronological_prauc(chrono_data):
    conditions = ["FULL", "SENSOR_ONLY", "WITHOUT_SENSOR", "WITHOUT_LOGS", "WITHOUT_OPERATOR_NOTES"]
    labels = ["Full (All)", "Sensor Only", "w/o Sensor", "w/o Logs", "w/o Notes"]

    models = ["Random Forest", "Logistic Regression", "DistilBERT"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    plt.figure(figsize=(9, 5), dpi=300)
    x = np.arange(len(conditions))
    width = 0.25

    for i, (m_name, color) in enumerate(zip(models, colors)):
        vals = []
        for c in conditions:
            m_res = chrono_data["conditions"][c]["models"].get(m_name)
            vals.append(m_res["pr_auc"] if m_res and m_res["pr_auc"] is not None else 0.0)
        plt.bar(x + (i - 1) * width, vals, width, label=m_name, color=color, alpha=0.88)

    plt.title("Chronological PR-AUC by Modality Ablation Condition", fontsize=12, fontweight="bold")
    plt.xticks(x, labels, fontsize=10)
    plt.ylabel("PR-AUC Score", fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    out = FIG_DIR / "prauc_by_ablation_chronological.png"
    plt.savefig(out)
    plt.close()
    print("Saved ->", out)


def plot_lomo_f1(lomo_data):
    conditions = ["FULL", "SENSOR_ONLY", "WITHOUT_SENSOR", "WITHOUT_LOGS", "WITHOUT_OPERATOR_NOTES"]
    labels = ["Full (All)", "Sensor Only", "w/o Sensor", "w/o Logs", "w/o Notes"]

    models = ["Random Forest", "Logistic Regression"]
    colors = ["#1f77b4", "#ff7f0e"]

    plt.figure(figsize=(8, 4.8), dpi=300)
    x = np.arange(len(conditions))
    width = 0.35

    for i, (m_name, color) in enumerate(zip(models, colors)):
        vals = []
        for c in conditions:
            m_res = lomo_data["conditions"][c]["models"].get(m_name)
            vals.append(m_res["macro_average"]["f1"] if m_res else 0.0)
        plt.bar(x + (i - 0.5) * width, vals, width, label=m_name, color=color, alpha=0.88)

    plt.title("Leave-One-Machine-Out (LOMO) Macro F1 by Modality Ablation", fontsize=12, fontweight="bold")
    plt.xticks(x, labels, fontsize=10)
    plt.ylabel("Macro F1 Score", fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    out = FIG_DIR / "f1_by_ablation_lomo.png"
    plt.savefig(out)
    plt.close()
    print("Saved ->", out)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    chrono_data, lomo_data = load_data()

    plot_chronological_f1(chrono_data)
    plot_chronological_prauc(chrono_data)
    plot_lomo_f1(lomo_data)
    print("All ablation figures generated successfully in", FIG_DIR)


if __name__ == "__main__":
    main()
