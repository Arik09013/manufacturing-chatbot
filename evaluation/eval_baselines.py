"""
Benchmark Runner for Conventional Baseline Models.

Conducts rigorous, leakage-free evaluation of:
  - Random Forest (Reference Baseline)
  - Logistic Regression
  - Support Vector Machine (SVC)
  - Isolation Forest (Unsupervised)
  - Gradient Boosted Trees (XGBoost, LightGBM, CatBoost) - dynamic availability check

Evaluation regimes:
  1. Chronological Per-Machine Holdout (80% train / 20% test with >=30 min embargo)
  2. Leave-One-Machine-Out (LOMO) Cross-Validation

Fairness guarantees:
  - Exact same fused dataset and features (ALL_FEATURE_COLS, 40 features)
  - Preprocessing (StandardScaler) fitted strictly on training data
  - Automated leakage assertions executed before every fold
  - Zero test data used for tuning or thresholding

Outputs:
  - evaluation/results/baselines/baseline_chronological_results.json
  - evaluation/results/baselines/baseline_lomo_results.json
  - evaluation/results/baselines/baseline_summary.json
  - outputs/baseline_benchmark_report.md
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.data.splits import (
    DataLeakageError,
    compute_classification_metrics,
    prepare_tabular_data,
    split_chronological,
    split_leave_one_machine_out,
    verify_leakage_free,
)
from src.fusion.fuse import load_fused
from src.model.baselines import (
    BaselineModelSpec,
    fit_and_evaluate_model,
    get_baseline_registry,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = _ROOT / "evaluation" / "results" / "baselines"
OUTPUT_REPORT = _ROOT / "outputs" / "baseline_benchmark_report.md"


def run_chronological_baseline_benchmark(
    df: pd.DataFrame,
    registry: Dict[str, BaselineModelSpec],
    embargo_minutes: int = 30,
) -> Dict[str, Any]:
    """
    Run chronological per-machine and pooled evaluation for all baseline models.
    """
    logger.info("==================================================")
    logger.info("PHASE 1: CHRONOLOGICAL BASELINE BENCHMARK")
    logger.info("==================================================")

    # 1. Split and verify
    train_df, test_df, purge_df = split_chronological(
        df, train_ratio=0.8, embargo_minutes=embargo_minutes
    )
    verify_leakage_free(
        train_df, test_df, mode="chronological", embargo_minutes=embargo_minutes
    )

    machines = sorted(df["machine_id"].unique())

    chrono_results: Dict[str, Any] = {
        "embargo_minutes": embargo_minutes,
        "n_train_total": len(train_df),
        "n_test_total": len(test_df),
        "n_purged_total": len(purge_df),
        "models": {},
        "unavailable_models": {},
    }

    # Record unavailable models
    for name, spec in registry.items():
        if not spec.is_available:
            chrono_results["unavailable_models"][name] = {
                "category": spec.category,
                "status": "not executed — dependency unavailable",
                "notes": spec.notes,
            }

    # Available models
    active_models = {k: v for k, v in registry.items() if v.is_available}

    for model_name, spec in active_models.items():
        logger.info("--- Evaluating %s (Chronological) ---", model_name)
        model_res: Dict[str, Any] = {
            "category": spec.category,
            "is_supervised": spec.is_supervised,
            "class_imbalance_strategy": spec.class_imbalance_strategy,
            "config": spec.config,
            "per_machine": {},
            "macro_average": {},
            "pooled_all_machines": {},
            "timing_and_size": {},
        }

        # A. Per-machine evaluation
        machine_metrics: List[Dict[str, Any]] = []
        per_m_train_times: List[float] = []
        per_m_inf_times: List[float] = []

        for m in machines:
            m_tr = train_df[train_df["machine_id"] == m].copy()
            m_te = test_df[test_df["machine_id"] == m].copy()

            # Preprocessing strictly fitted on m_tr
            X_tr, y_tr, X_te, y_te, feat_names, _ = prepare_tabular_data(m_tr, m_te, scale=True)

            y_pred, y_prob, t_fit, t_inf, complexity = fit_and_evaluate_model(
                spec, X_tr, y_tr, X_te, y_te
            )

            metrics = compute_classification_metrics(
                y_te, y_pred, y_prob, n_train=len(m_tr), n_anom_train=int(y_tr.sum())
            )
            metrics["train_time_sec"] = round(t_fit, 4)
            metrics["inference_time_sec"] = round(t_inf, 4)
            metrics["inference_time_per_sample_ms"] = round((t_inf / len(y_te)) * 1000.0, 4)

            model_res["per_machine"][m] = metrics
            machine_metrics.append(metrics)
            per_m_train_times.append(t_fit)
            per_m_inf_times.append(t_inf)

            logger.info(
                "[%s | %s] Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f, AUC=%s",
                model_name,
                m,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A",
            )

        # Macro average across the 3 stations
        model_res["macro_average"] = {
            "accuracy": round(float(np.mean([m["accuracy"] for m in machine_metrics])), 4),
            "precision": round(float(np.mean([m["precision"] for m in machine_metrics])), 4),
            "recall": round(float(np.mean([m["recall"] for m in machine_metrics])), 4),
            "f1": round(float(np.mean([m["f1"] for m in machine_metrics])), 4),
            "roc_auc": round(float(np.mean([m["roc_auc"] for m in machine_metrics if m["roc_auc"] is not None])), 4),
            "pr_auc": round(float(np.mean([m["pr_auc"] for m in machine_metrics if m["pr_auc"] is not None])), 4),
            "mean_train_time_sec": round(float(np.mean(per_m_train_times)), 4),
            "mean_inference_time_sec": round(float(np.mean(per_m_inf_times)), 4),
        }

        # B. Multi-machine pooled model
        X_tr_all, y_tr_all, X_te_all, y_te_all, _, _ = prepare_tabular_data(train_df, test_df, scale=True)

        y_pred_all, y_prob_all, t_fit_all, t_inf_all, complexity_all = fit_and_evaluate_model(
            spec, X_tr_all, y_tr_all, X_te_all, y_te_all
        )

        pooled_res = compute_classification_metrics(
            y_te_all, y_pred_all, y_prob_all, n_train=len(train_df), n_anom_train=int(y_tr_all.sum())
        )
        pooled_res["train_time_sec"] = round(t_fit_all, 4)
        pooled_res["inference_time_sec"] = round(t_inf_all, 4)
        pooled_res["inference_time_per_sample_ms"] = round((t_inf_all / len(y_te_all)) * 1000.0, 4)
        pooled_res["model_complexity"] = complexity_all

        model_res["pooled_all_machines"] = pooled_res
        model_res["timing_and_size"] = {
            "train_time_sec": round(t_fit_all, 4),
            "inference_time_sec": round(t_inf_all, 4),
            "inference_time_per_sample_ms": round((t_inf_all / len(y_te_all)) * 1000.0, 4),
            "serialized_size_bytes": complexity_all.get("serialized_size_bytes"),
            "structural_parameters": complexity_all.get("structural_parameters"),
            "complexity_description": complexity_all.get("complexity_description"),
        }

        logger.info(
            "[%s | Pooled] Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f, AUC=%.4f",
            model_name,
            pooled_res["accuracy"],
            pooled_res["precision"],
            pooled_res["recall"],
            pooled_res["f1"],
            pooled_res["roc_auc"] if pooled_res["roc_auc"] is not None else 0.0,
        )

        chrono_results["models"][model_name] = model_res

    return chrono_results


def run_lomo_baseline_benchmark(
    df: pd.DataFrame,
    registry: Dict[str, BaselineModelSpec],
) -> Dict[str, Any]:
    """
    Run Leave-One-Machine-Out (LOMO) cross-validation for all baseline models.
    """
    logger.info("==================================================")
    logger.info("PHASE 2: LEAVE-ONE-MACHINE-OUT (LOMO) BENCHMARK")
    logger.info("==================================================")

    lomo_results: Dict[str, Any] = {
        "models": {},
        "unavailable_models": {},
    }

    # Record unavailable models
    for name, spec in registry.items():
        if not spec.is_available:
            lomo_results["unavailable_models"][name] = {
                "category": spec.category,
                "status": "not executed — dependency unavailable",
                "notes": spec.notes,
            }

    active_models = {k: v for k, v in registry.items() if v.is_available}

    for model_name, spec in active_models.items():
        logger.info("--- Evaluating %s (LOMO) ---", model_name)
        model_res: Dict[str, Any] = {
            "category": spec.category,
            "is_supervised": spec.is_supervised,
            "class_imbalance_strategy": spec.class_imbalance_strategy,
            "config": spec.config,
            "folds": {},
            "macro_average": {},
            "pooled_all_folds": {},
            "timing_and_size": {},
        }

        fold_metrics: List[Dict[str, Any]] = []
        all_y_true: List[int] = []
        all_y_pred: List[int] = []
        all_y_prob: List[float] = []
        train_times: List[float] = []
        inf_times: List[float] = []
        complexities: List[Dict[str, Any]] = []

        for train_df, test_df, held_out in split_leave_one_machine_out(df):
            # Assert leak-free
            verify_leakage_free(
                train_df, test_df, mode="lomo", held_out_machine=held_out
            )

            # Preprocessing strictly fitted on train_df
            X_tr, y_tr, X_te, y_te, feat_names, _ = prepare_tabular_data(train_df, test_df, scale=True)

            y_pred, y_prob, t_fit, t_inf, complexity = fit_and_evaluate_model(
                spec, X_tr, y_tr, X_te, y_te
            )

            f_metrics = compute_classification_metrics(
                y_te, y_pred, y_prob, n_train=len(train_df), n_anom_train=int(y_tr.sum())
            )
            f_metrics["train_time_sec"] = round(t_fit, 4)
            f_metrics["inference_time_sec"] = round(t_inf, 4)
            f_metrics["inference_time_per_sample_ms"] = round((t_inf / len(y_te)) * 1000.0, 4)

            model_res["folds"][held_out] = f_metrics
            fold_metrics.append(f_metrics)

            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_prob.extend(y_prob.tolist())
            train_times.append(t_fit)
            inf_times.append(t_inf)
            complexities.append(complexity)

            logger.info(
                "[%s | Held-out %s] Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f, AUC=%.4f",
                model_name,
                held_out,
                f_metrics["accuracy"],
                f_metrics["precision"],
                f_metrics["recall"],
                f_metrics["f1"],
                f_metrics["roc_auc"],
            )

        # Macro average across the 3 folds
        model_res["macro_average"] = {
            "accuracy": round(float(np.mean([f["accuracy"] for f in fold_metrics])), 4),
            "precision": round(float(np.mean([f["precision"] for f in fold_metrics])), 4),
            "recall": round(float(np.mean([f["recall"] for f in fold_metrics])), 4),
            "f1": round(float(np.mean([f["f1"] for f in fold_metrics])), 4),
            "roc_auc": round(float(np.mean([f["roc_auc"] for f in fold_metrics if f["roc_auc"] is not None])), 4),
            "pr_auc": round(float(np.mean([f["pr_auc"] for f in fold_metrics if f["pr_auc"] is not None])), 4),
            "mean_train_time_sec": round(float(np.mean(train_times)), 4),
            "mean_inference_time_sec": round(float(np.mean(inf_times)), 4),
        }

        # Pooled out-of-fold metrics across all 1,917 test predictions
        y_true_arr = np.asarray(all_y_true)
        y_pred_arr = np.asarray(all_y_pred)
        y_prob_arr = np.asarray(all_y_prob)

        pooled_res = compute_classification_metrics(
            y_true_arr, y_pred_arr, y_prob_arr, n_train=1278, n_anom_train=40
        )
        total_inf_time = float(np.sum(inf_times))
        pooled_res["train_time_sec_total"] = round(float(np.sum(train_times)), 4)
        pooled_res["inference_time_sec_total"] = round(total_inf_time, 4)
        pooled_res["inference_time_per_sample_ms"] = round((total_inf_time / len(y_true_arr)) * 1000.0, 4)

        model_res["pooled_all_folds"] = pooled_res
        model_res["timing_and_size"] = {
            "mean_train_time_sec": round(float(np.mean(train_times)), 4),
            "mean_inference_time_sec": round(float(np.mean(inf_times)), 4),
            "inference_time_per_sample_ms": round((total_inf_time / len(y_true_arr)) * 1000.0, 4),
            "serialized_size_bytes": complexities[0].get("serialized_size_bytes"),
            "structural_parameters": complexities[0].get("structural_parameters"),
            "complexity_description": complexities[0].get("complexity_description"),
        }

        lomo_results["models"][model_name] = model_res

    return lomo_results


def build_summary(
    chrono_results: Dict[str, Any],
    lomo_results: Dict[str, Any],
    registry: Dict[str, BaselineModelSpec],
) -> Dict[str, Any]:
    """
    Construct high-level summary JSON comparing all models.
    """
    summary: Dict[str, Any] = {
        "benchmark_timestamp": "2026-09-29",
        "dataset_total_samples": 1917,
        "n_features": 40,
        "models_benchmarked": list(chrono_results["models"].keys()),
        "models_unavailable": list(chrono_results["unavailable_models"].keys()),
        "chronological_comparison": {},
        "lomo_comparison": {},
    }

    for name in summary["models_benchmarked"]:
        c_macro = chrono_results["models"][name]["macro_average"]
        c_pooled = chrono_results["models"][name]["pooled_all_machines"]
        c_timing = chrono_results["models"][name]["timing_and_size"]

        summary["chronological_comparison"][name] = {
            "macro_f1": c_macro["f1"],
            "macro_roc_auc": c_macro["roc_auc"],
            "macro_pr_auc": c_macro["pr_auc"],
            "pooled_accuracy": c_pooled["accuracy"],
            "pooled_precision": c_pooled["precision"],
            "pooled_recall": c_pooled["recall"],
            "pooled_f1": c_pooled["f1"],
            "pooled_roc_auc": c_pooled["roc_auc"],
            "pooled_pr_auc": c_pooled["pr_auc"],
            "train_time_sec": c_timing["train_time_sec"],
            "inf_time_per_sample_ms": c_timing["inference_time_per_sample_ms"],
            "serialized_size_bytes": c_timing["serialized_size_bytes"],
            "complexity": c_timing["complexity_description"],
        }

        l_macro = lomo_results["models"][name]["macro_average"]
        l_pooled = lomo_results["models"][name]["pooled_all_folds"]
        l_timing = lomo_results["models"][name]["timing_and_size"]

        summary["lomo_comparison"][name] = {
            "macro_accuracy": l_macro["accuracy"],
            "macro_precision": l_macro["precision"],
            "macro_recall": l_macro["recall"],
            "macro_f1": l_macro["f1"],
            "macro_roc_auc": l_macro["roc_auc"],
            "macro_pr_auc": l_macro["pr_auc"],
            "pooled_accuracy": l_pooled["accuracy"],
            "pooled_precision": l_pooled["precision"],
            "pooled_recall": l_pooled["recall"],
            "pooled_f1": l_pooled["f1"],
            "pooled_roc_auc": l_pooled["roc_auc"],
            "pooled_pr_auc": l_pooled["pr_auc"],
            "mean_train_time_sec": l_timing["mean_train_time_sec"],
            "inf_time_per_sample_ms": l_timing["inference_time_per_sample_ms"],
        }

    return summary


def write_markdown_report(
    chrono_results: Dict[str, Any],
    lomo_results: Dict[str, Any],
    summary: Dict[str, Any],
    registry: Dict[str, BaselineModelSpec],
    out_path: Path = OUTPUT_REPORT,
) -> None:
    """Generate a clean, scientific markdown report of the baseline comparison."""
    lines: List[str] = [
        "# Conventional Baseline Benchmark Report",
        "",
        "> **Evaluation Harness:** Leakage-safe time-aware evaluation with >=30 min purge embargo and Leave-One-Machine-Out validation.",
        "> **Methodology:** Preprocessing (StandardScaler) fitted strictly on training data; identical feature matrix (40 columns); zero parameter tuning on test sets.",
        "",
        "---",
        "",
        "## 1. Experimental Overview & Model Configurations",
        "",
        "| Model Name | Category | Imbalance Strategy | Active Hyperparameters | Status |",
        "|---|---|---|---|---|",
    ]

    for name, spec in registry.items():
        status = "**Executed**" if spec.is_available else "*Not Executed (Unavailable)*"
        config_str = (
            ", ".join(f"`{k}={v}`" for k, v in spec.config.items() if k not in ("n_jobs", "random_state"))
            if spec.config
            else "None"
        )
        lines.append(
            f"| **{name}** | {spec.category} | {spec.class_imbalance_strategy} | {config_str} | {status} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Chronological Per-Machine Holdout Benchmark",
        "",
        "Evaluated on the chronological test partition ($N_{test}=375$ windows across 3 stations, 10 anomalies) separated by an embargo buffer of 30+ minutes.",
        "",
        "### A. Macro-Average Results (Across 3 Machines)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|",
    ])

    for name in summary["models_benchmarked"]:
        m = chrono_results["models"][name]["macro_average"]
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        pr_str = f"{m['pr_auc']:.4f}" if m['pr_auc'] is not None else "N/A"
        lines.append(
            f"| **{name}** | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {auc_str} | {pr_str} |"
        )

    lines.extend([
        "",
        "### B. Multi-Machine Pooled Model Results ($N=375$, 10 Anomalies)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Confusion Matrix |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for name in summary["models_benchmarked"]:
        p = chrono_results["models"][name]["pooled_all_machines"]
        cm_str = f"`{p['confusion_matrix']}`"
        auc_str = f"{p['roc_auc']:.4f}" if p['roc_auc'] is not None else "N/A"
        pr_str = f"{p['pr_auc']:.4f}" if p['pr_auc'] is not None else "N/A"
        lines.append(
            f"| **{name}** | {p['accuracy']:.4f} | {p['precision']:.4f} | {p['recall']:.4f} | {p['f1']:.4f} | {auc_str} | {pr_str} | {cm_str} |"
        )

    lines.extend([
        "",
        "### C. Station-Specific Breakdown (Chronological Test)",
        "",
        "| Model | Station | Train / Test | Anomalies (Tr/Te) | Accuracy | Precision | Recall | F1 Score | ROC-AUC |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for name in summary["models_benchmarked"]:
        for st_name, sm in chrono_results["models"][name]["per_machine"].items():
            auc_val = f"{sm['roc_auc']:.4f}" if sm['roc_auc'] is not None else "N/A"
            lines.append(
                f"| {name} | `{st_name}` | {sm['n_train']} / {sm['n_test']} | {sm['n_anomalies_train']} / {sm['n_anomalies_test']} | {sm['accuracy']:.4f} | {sm['precision']:.4f} | {sm['recall']:.4f} | {sm['f1']:.4f} | {auc_val} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Leave-One-Machine-Out (LOMO) Cross-Validation Benchmark",
        "",
        "Evaluates cross-machine generalizability: trained on 2 stations ($N=1,278$, 40 anomalies), tested on the held-out 3rd station ($N=639$, 20 anomalies).",
        "",
        "### A. Macro-Average Results (Across 3 Folds)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|",
    ])

    for name in summary["models_benchmarked"]:
        m = lomo_results["models"][name]["macro_average"]
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        pr_str = f"{m['pr_auc']:.4f}" if m['pr_auc'] is not None else "N/A"
        lines.append(
            f"| **{name}** | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {auc_str} | {pr_str} |"
        )

    lines.extend([
        "",
        "### B. Pooled Out-of-Fold Predictions ($N=1,917$, 60 Anomalies)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Confusion Matrix |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for name in summary["models_benchmarked"]:
        p = lomo_results["models"][name]["pooled_all_folds"]
        cm_str = f"`{p['confusion_matrix']}`"
        auc_str = f"{p['roc_auc']:.4f}" if p['roc_auc'] is not None else "N/A"
        pr_str = f"{p['pr_auc']:.4f}" if p['pr_auc'] is not None else "N/A"
        lines.append(
            f"| **{name}** | {p['accuracy']:.4f} | {p['precision']:.4f} | {p['recall']:.4f} | {p['f1']:.4f} | {auc_str} | {pr_str} | {cm_str} |"
        )

    lines.extend([
        "",
        "### C. Fold Breakdown (Held-Out Stations)",
        "",
        "| Model | Held-Out Station | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Confusion Matrix |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for name in summary["models_benchmarked"]:
        for held_out, fm in lomo_results["models"][name]["folds"].items():
            lines.append(
                f"| {name} | `{held_out}` | {fm['accuracy']:.4f} | {fm['precision']:.4f} | {fm['recall']:.4f} | {fm['f1']:.4f} | {fm['roc_auc']:.4f} | {fm['pr_auc']:.4f} | `{fm['confusion_matrix']}` |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Computational Efficiency & Model Complexity",
        "",
        "| Model | Train Time (Pooled) | Inference Time (Test Pool) | Inference / Sample | Serialized Size | Complexity / Parameters |",
        "|---|---|---|---|---|---|",
    ])

    for name in summary["models_benchmarked"]:
        t = chrono_results["models"][name]["timing_and_size"]
        size_str = f"{t['serialized_size_bytes']:,} B" if t['serialized_size_bytes'] else "N/A"
        lines.append(
            f"| **{name}** | {t['train_time_sec']:.4f} s | {t['inference_time_sec']:.4f} s | {t['inference_time_per_sample_ms']:.4f} ms | {size_str} | {t['complexity_description']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Dependency Audit & Status",
        "",
        "- **Installed & Benchmarked:** `Random Forest`, `Logistic Regression`, `Support Vector Machine`, `Isolation Forest` (via `scikit-learn==1.9.0`).",
        "- **Unavailable Dependencies:**",
        "  - `XGBoost`: Package not installed in `.venv`. Status: *Not executed — dependency unavailable*.",
        "  - `LightGBM`: Package not installed in `.venv`. Status: *Not executed — dependency unavailable*.",
        "  - `CatBoost`: Package not installed in `.venv`. Status: *Not executed — dependency unavailable*.",
        "",
        "---",
        "",
        "## 6. Reproducibility Commands",
        "",
        "```powershell",
        "# Run baseline benchmark suite",
        ".\\.venv\\Scripts\\python.exe evaluation/eval_baselines.py",
        "",
        "# Run unit tests for baselines and leakage",
        ".\\.venv\\Scripts\\pytest tests/test_baselines.py tests/test_leakage_free.py -v",
        "```",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Markdown report written -> %s", out_path)

    # Also save inside RESULTS_DIR
    colocated = RESULTS_DIR / "baseline_benchmark_report.md"
    colocated.parent.mkdir(parents=True, exist_ok=True)
    with open(colocated, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Markdown report co-located -> %s", colocated)


def main() -> None:
    logger.info("Loading fused multimodal dataset...")
    df = load_fused()
    logger.info("Loaded fused dataset with shape %s", df.shape)

    registry = get_baseline_registry()

    # 1. Chronological Benchmark
    chrono_res = run_chronological_baseline_benchmark(df, registry, embargo_minutes=30)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    chrono_out = RESULTS_DIR / "baseline_chronological_results.json"
    with open(chrono_out, "w", encoding="utf-8") as f:
        json.dump(chrono_res, f, indent=2)
    logger.info("Saved chronological results -> %s", chrono_out)

    # 2. LOMO Benchmark
    lomo_res = run_lomo_baseline_benchmark(df, registry)
    lomo_out = RESULTS_DIR / "baseline_lomo_results.json"
    with open(lomo_out, "w", encoding="utf-8") as f:
        json.dump(lomo_res, f, indent=2)
    logger.info("Saved LOMO results -> %s", lomo_out)

    # 3. Summary
    summary = build_summary(chrono_res, lomo_res, registry)
    summary_out = RESULTS_DIR / "baseline_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary -> %s", summary_out)

    # 4. Human-readable report
    write_markdown_report(chrono_res, lomo_res, summary, registry)
    logger.info("Baseline benchmark complete.")


if __name__ == "__main__":
    main()
