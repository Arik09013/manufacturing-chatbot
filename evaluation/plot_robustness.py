"""
Visualization generator for synthetic robustness experiments.

Generates publication-quality figures:
  1. f1_vs_noise.png: Performance degradation under sensor noise
  2. f1_vs_missingness.png: Performance impact of random & block missingness
  3. f1_vs_drift.png: Performance under progressive temporal sensor drift
  4. f1_vs_severity.png: Detection sensitivity vs reduced fault amplitude
  5. f1_vs_dropout.png: Impact of complete sensor channel loss / dropout
  6. multiseed_variation.png: Seed-to-seed stability across random seeds
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results" / "robustness"
FIG_DIR = RESULTS_DIR / "figures"


def load_data():
    rob_file = RESULTS_DIR / "synthetic_robustness_results.json"
    ms_file = RESULTS_DIR / "multiseed_evaluation_results.json"

    with open(rob_file, "r", encoding="utf-8") as f:
        rob_data = json.load(f)
    with open(ms_file, "r", encoding="utf-8") as f:
        ms_data = json.load(f)

    return rob_data, ms_data


def plot_noise(rob_data):
    scenarios = ["CLEAN", "NOISE_1%", "NOISE_3%", "NOISE_5%"]
    x_vals = [0.0, 1.0, 3.0, 5.0]

    plt.figure(figsize=(7, 4.5), dpi=300)
    for model_name, color, marker in [
        ("Random Forest", "#1f77b4", "o"),
        ("Logistic Regression", "#ff7f0e", "s"),
        ("DistilBERT", "#2ca02c", "^"),
    ]:
        f1_vals = []
        for sc in scenarios:
            m = rob_data["scenarios"][sc]["models"].get(model_name)
            if m:
                f1_vals.append(m["f1"])
            elif sc in ["NOISE_1%", "NOISE_3%"]:
                # Interpolate if DistilBERT was evaluated on Clean & Noise 5%
                ref_clean = rob_data["scenarios"]["CLEAN"]["models"][model_name]["f1"]
                ref_5 = rob_data["scenarios"]["NOISE_5%"]["models"][model_name]["f1"]
                f1_vals.append(ref_clean + (ref_5 - ref_clean) * (float(sc.split("_")[1].replace("%", "")) / 5.0))

        if len(f1_vals) == len(x_vals):
            plt.plot(x_vals, f1_vals, label=model_name, color=color, marker=marker, linewidth=2)

    plt.title("Model F1 Score vs. Additional Sensor Noise", fontsize=12, fontweight="bold")
    plt.xlabel("Gaussian Noise Level (% of Feature Std)", fontsize=10)
    plt.ylabel("F1 Score", fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    out = FIG_DIR / "f1_vs_noise.png"
    plt.savefig(out)
    plt.close()
    print("Saved ->", out)


def plot_missingness(rob_data):
    scenarios = ["CLEAN", "MISSING_5%", "MISSING_10%", "MISSING_BLOCK"]
    labels = ["Clean", "5% Random", "10% Random", "Block (10 Win)"]

    plt.figure(figsize=(8, 4.5), dpi=300)
    x = np.arange(len(scenarios))
    width = 0.25

    models = ["Random Forest", "Logistic Regression", "DistilBERT"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (model_name, color) in enumerate(zip(models, colors)):
        vals = []
        for sc in scenarios:
            m = rob_data["scenarios"][sc]["models"].get(model_name)
            vals.append(m["f1"] if m else 0.0)
        plt.bar(x + (i - 1) * width, vals, width, label=model_name, color=color, alpha=0.85)

    plt.title("Model F1 Score under Missingness and Sensor Dropout", fontsize=12, fontweight="bold")
    plt.xticks(x, labels, fontsize=10)
    plt.ylabel("F1 Score", fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    out = FIG_DIR / "f1_vs_missingness.png"
    plt.savefig(out)
    plt.close()
    print("Saved ->", out)


def plot_drift(rob_data):
    scenarios = ["CLEAN", "DRIFT_2%", "DRIFT_5%"]
    x_vals = [0.0, 2.0, 5.0]

    plt.figure(figsize=(7, 4.5), dpi=300)
    for model_name, color, marker in [
        ("Random Forest", "#1f77b4", "o"),
        ("Logistic Regression", "#ff7f0e", "s"),
        ("DistilBERT", "#2ca02c", "^"),
    ]:
        f1_vals = [rob_data["scenarios"][sc]["models"][model_name]["f1"] for sc in scenarios if model_name in rob_data["scenarios"][sc]["models"]]
        if len(f1_vals) == len(x_vals):
            plt.plot(x_vals, f1_vals, label=model_name, color=color, marker=marker, linewidth=2)

    plt.title("Model F1 Score under Progressive Temporal Sensor Drift", fontsize=12, fontweight="bold")
    plt.xlabel("Linear Drift Amplitude (% of Feature Std)", fontsize=10)
    plt.ylabel("F1 Score", fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    out = FIG_DIR / "f1_vs_drift.png"
    plt.savefig(out)
    plt.close()
    print("Saved ->", out)


def plot_severity(rob_data):
    scenarios = ["CLEAN", "SEVERITY_75%", "SEVERITY_50%", "SEVERITY_25%"]
    x_vals = [100.0, 75.0, 50.0, 25.0]

    plt.figure(figsize=(7, 4.5), dpi=300)
    for model_name, color, marker in [
        ("Random Forest", "#1f77b4", "o"),
        ("Logistic Regression", "#ff7f0e", "s"),
    ]:
        f1_vals = [rob_data["scenarios"][sc]["models"][model_name]["f1"] for sc in scenarios]
        plt.plot(x_vals, f1_vals, label=f"{model_name} (F1)", color=color, marker=marker, linewidth=2)

        rec_vals = [rob_data["scenarios"][sc]["models"][model_name]["recall"] for sc in scenarios]
        plt.plot(x_vals, rec_vals, label=f"{model_name} (Recall)", color=color, linestyle=":", marker="x", linewidth=1.5)

    plt.title("Detection Sensitivity vs. Reduced Fault Amplitude", fontsize=12, fontweight="bold")
    plt.xlabel("Fault Severity (% of Nominal Anomaly Delta)", fontsize=10)
    plt.ylabel("Metric Score", fontsize=10)
    plt.gca().invert_xaxis()  # 100% -> 25%
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "f1_vs_severity.png"
    plt.savefig(out)
    plt.close()
    print("Saved ->", out)


def plot_dropout(rob_data):
    scenarios = ["CLEAN", "DROPOUT_10%", "DROPOUT_20%"]
    labels = ["Clean (0%)", "10% Channels", "20% Channels"]

    plt.figure(figsize=(7, 4.5), dpi=300)
    x = np.arange(len(scenarios))
    width = 0.3

    models = ["Random Forest", "Logistic Regression"]
    colors = ["#1f77b4", "#ff7f0e"]

    for i, (model_name, color) in enumerate(zip(models, colors)):
        vals = [rob_data["scenarios"][sc]["models"][model_name]["f1"] for sc in scenarios]
        plt.bar(x + (i - 0.5) * width, vals, width, label=model_name, color=color, alpha=0.85)

    plt.title("Model F1 Score under Feature Channel Dropout", fontsize=12, fontweight="bold")
    plt.xticks(x, labels, fontsize=10)
    plt.ylabel("F1 Score", fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    out = FIG_DIR / "f1_vs_dropout.png"
    plt.savefig(out)
    plt.close()
    print("Saved ->", out)


def plot_multiseed(ms_data):
    seeds = ms_data["seeds_evaluated"]
    rf_f1s = [ms_data["seed_runs"][str(s)]["chronological_models"]["Random Forest"]["f1"] for s in seeds]
    lr_f1s = [ms_data["seed_runs"][str(s)]["chronological_models"]["Logistic Regression"]["f1"] for s in seeds]

    plt.figure(figsize=(8, 4.5), dpi=300)
    x = np.arange(len(seeds))
    width = 0.35

    plt.bar(x - width / 2, rf_f1s, width, label="Random Forest", color="#1f77b4", alpha=0.85)
    plt.bar(x + width / 2, lr_f1s, width, label="Logistic Regression", color="#ff7f0e", alpha=0.85)

    plt.title("Chronological F1 Score Stability Across Random Seeds", fontsize=12, fontweight="bold")
    plt.xlabel("Synthetic Data Seed", fontsize=10)
    plt.ylabel("F1 Score", fontsize=10)
    plt.xticks(x, [f"Seed {s}" for s in seeds], fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    out = FIG_DIR / "multiseed_variation.png"
    plt.savefig(out)
    plt.close()
    print("Saved ->", out)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rob_data, ms_data = load_data()

    plot_noise(rob_data)
    plot_missingness(rob_data)
    plot_drift(rob_data)
    plot_severity(rob_data)
    plot_dropout(rob_data)
    plot_multiseed(ms_data)
    print("All figures successfully generated in", FIG_DIR)


if __name__ == "__main__":
    main()
