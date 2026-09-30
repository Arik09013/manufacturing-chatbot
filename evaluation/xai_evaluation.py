"""
Core Evaluation Framework for Explainability (XAI) Methods (Step 9).

Provides rigorous, reproducible, leakage-free evaluation protocols for:
  1. Explanation Fidelity / Faithfulness (Progressive feature deletion, neutral replacement, AUC)
  2. Explanation Stability / Robustness (Small feature jitter, class preservation, Jaccard & rank correlation)
  3. Feature-Ranking Consistency / Cross-Method Agreement (SHAP vs LIME vs Attention)
  4. Normal vs Anomalous Explanation Structural Distributions (Concentration, entropy, feature groups)
  5. Computational Cost & Latency Benchmarks
  6. Failure Mode & Edge-Case Diagnostics
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from src.data.splits import split_chronological, verify_leakage_free
from src.explain.attention_explainer import explain_attention
from src.explain.lime_explainer import LimeAnomalyExplainer, _background_sample
from src.explain.shap_explainer import AnomalyExplainer
from src.fusion.fuse import load_fused
from src.model.anomaly import AnomalyDetector, get_feature_matrix

logger = logging.getLogger(__name__)

# Feature taxonomy grouped by domain / physical channel
FEATURE_GROUPS: Dict[str, List[str]] = {
    "current": [
        "welding_current_mean", "welding_current_std", "welding_current_min",
        "welding_current_max", "welding_current_range"
    ],
    "voltage": [
        "arc_voltage_mean", "arc_voltage_std", "arc_voltage_min",
        "arc_voltage_max", "arc_voltage_range"
    ],
    "speed": [
        "welding_speed_mean", "welding_speed_std", "welding_speed_min",
        "welding_speed_max", "welding_speed_range"
    ],
    "wire_feed": [
        "wire_feed_rate_mean", "wire_feed_rate_std", "wire_feed_rate_min",
        "wire_feed_rate_max", "wire_feed_rate_range"
    ],
    "gas": [
        "shielding_gas_flow_mean", "shielding_gas_flow_std", "shielding_gas_flow_min",
        "shielding_gas_flow_max", "shielding_gas_flow_range"
    ],
    "heat": [
        "heat_input_mean", "heat_input_std", "heat_input_min",
        "heat_input_max", "heat_input_range"
    ],
    "logs": [
        "n_events", "n_alarm", "n_warning", "n_maintenance",
        "n_production", "n_diagnostic", "n_operational", "has_alarm", "has_warning"
    ],
    "notes": [
        "has_note"
    ]
}


@dataclass
class XaiDatasetSplit:
    """Encapsulates frozen chronological split and deterministic evaluation subset."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    eval_df: pd.DataFrame
    feature_names: List[str]
    train_medians: np.ndarray
    train_stds: np.ndarray
    anomalous_indices: List[int]
    normal_indices: List[int]
    total_samples: int
    n_anomalous: int
    n_normal: int


def prepare_xai_evaluation_data(
    df: Optional[pd.DataFrame] = None,
    train_ratio: float = 0.8,
    embargo_minutes: int = 30,
    n_normal: int = 50,
    random_seed: int = 42,
) -> XaiDatasetSplit:
    """
    Construct leakage-free chronological test split and deterministic evaluation subset.

    Selection protocol:
      - Uses chronological holdout with temporal embargo (zero leakage).
      - Selects ALL anomalous test samples (rare events).
      - Selects exactly n_normal normal samples using a deterministic random seed.
      - Fits medians and standard deviations strictly on the training set.
    """
    if df is None:
        df = load_fused()

    train_df, test_df, _ = split_chronological(
        df, train_ratio=train_ratio, embargo_minutes=embargo_minutes
    )
    verify_leakage_free(train_df, test_df, mode="chronological", embargo_minutes=embargo_minutes)

    X_train, feature_names = get_feature_matrix(train_df)
    train_medians = np.median(X_train, axis=0)
    train_stds = np.std(X_train, axis=0)
    # Avoid zero division in jitter
    train_stds = np.where(train_stds < 1e-6, 1.0, train_stds)

    is_ano = test_df["is_anomaly"].values.astype(bool)
    test_indices = np.arange(len(test_df))
    ano_idx = test_indices[is_ano].tolist()
    norm_pool = test_indices[~is_ano]

    rng = np.random.RandomState(random_seed)
    if len(norm_pool) > n_normal:
        norm_idx = sorted(rng.choice(norm_pool, size=n_normal, replace=False).tolist())
    else:
        norm_idx = norm_pool.tolist()

    eval_idx = sorted(ano_idx + norm_idx)
    eval_df = test_df.iloc[eval_idx].copy().reset_index(drop=True)

    # Recompute indices relative to eval_df
    eval_is_ano = eval_df["is_anomaly"].values.astype(bool)
    eval_ano_idx = np.where(eval_is_ano)[0].tolist()
    eval_norm_idx = np.where(~eval_is_ano)[0].tolist()

    return XaiDatasetSplit(
        train_df=train_df,
        test_df=test_df,
        eval_df=eval_df,
        feature_names=feature_names,
        train_medians=train_medians,
        train_stds=train_stds,
        anomalous_indices=eval_ano_idx,
        normal_indices=eval_norm_idx,
        total_samples=len(eval_df),
        n_anomalous=len(eval_ano_idx),
        n_normal=len(eval_norm_idx),
    )


# =====================================================================
# Explanation Extractors
# =====================================================================

def extract_shap_explanation(
    explainer: AnomalyExplainer,
    row: pd.DataFrame,
) -> Dict[str, Any]:
    """Extract full 40-feature SHAP explanation with magnitude rankings."""
    res = explainer.explain_row(row)
    raw_shap: np.ndarray = res["raw_shap"]
    feature_names = explainer._feature_names

    abs_shap = np.abs(raw_shap)
    ranked_order = np.argsort(abs_shap)[::-1]

    ranked_features = [feature_names[i] for i in ranked_order]
    ranked_importances = [float(abs_shap[i]) for i in ranked_order]
    ranked_signed = [float(raw_shap[i]) for i in ranked_order]

    return {
        "anomaly_prob": float(res["anomaly_prob"]),
        "base_value": float(res["base_value"]),
        "raw_shap": raw_shap,
        "ranked_features": ranked_features,
        "ranked_importances": ranked_importances,
        "ranked_signed": ranked_signed,
        "top_drivers": res["shap_drivers"],
        "feature_to_importance": {feat: float(abs_shap[i]) for i, feat in enumerate(feature_names)},
        "feature_to_signed": {feat: float(raw_shap[i]) for i, feat in enumerate(feature_names)},
    }


def extract_lime_explanation(
    explainer: LimeAnomalyExplainer,
    row: pd.DataFrame,
) -> Dict[str, Any]:
    """Extract full 40-feature LIME explanation with magnitude rankings."""
    # Ensure explainer extracts all features for ranking
    res = explainer.explain_row(row)
    drivers = res["lime_drivers"]
    feature_names = explainer.feature_names

    feature_to_weight: Dict[str, float] = {}
    for d in drivers:
        feature_to_weight[d["feature"]] = float(d["weight"])

    # Fallback for any unreturned feature (assign 0.0)
    for feat in feature_names:
        if feat not in feature_to_weight:
            feature_to_weight[feat] = 0.0

    abs_weights = {f: abs(w) for f, w in feature_to_weight.items()}
    ranked_features = sorted(feature_names, key=lambda f: abs_weights[f], reverse=True)
    ranked_importances = [abs_weights[f] for f in ranked_features]
    ranked_signed = [feature_to_weight[f] for f in ranked_features]

    return {
        "anomaly_prob": float(res["anomaly_prob"]),
        "ranked_features": ranked_features,
        "ranked_importances": ranked_importances,
        "ranked_signed": ranked_signed,
        "top_drivers": drivers[:explainer.top_n],
        "feature_to_importance": abs_weights,
        "feature_to_signed": feature_to_weight,
    }


# =====================================================================
# A. Fidelity / Faithfulness Evaluation
# =====================================================================

def evaluate_deletion_fidelity(
    detector: AnomalyDetector,
    eval_df: pd.DataFrame,
    feature_names: List[str],
    train_medians: np.ndarray,
    ranked_features_list: List[List[str]],
    mask_k_levels: Tuple[int, ...] = (0, 1, 3, 5, 10),
    random_baseline: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform progressive feature deletion/masking using training-set medians.

    For each sample:
      - At k=0: baseline prediction probability p0 and class y0.
      - At each k in mask_k_levels: mask top-k features with training-set median.
      - Measure probability drop: Delta p_k = p0 - p_k.
      - Measure prediction flip: (p_k >= 0.5) != (p0 >= 0.5).
      - Compute Area Under Deletion Curve (AUC) via trapezoidal integration.
    """
    X_orig, _ = get_feature_matrix(eval_df)
    n_samples = len(eval_df)
    feat_to_idx = {name: i for i, name in enumerate(feature_names)}

    prob_curves: Dict[int, List[float]] = {k: [] for k in mask_k_levels}
    delta_curves: Dict[int, List[float]] = {k: [] for k in mask_k_levels}
    flips: Dict[int, List[bool]] = {k: [] for k in mask_k_levels}
    auc_values: List[float] = []

    # Baseline probabilities
    p0_all = detector.model.predict_proba(X_orig)[:, 1]
    y0_all = (p0_all >= 0.5).astype(int)

    for i in range(n_samples):
        p0 = float(p0_all[i])
        y0 = int(y0_all[i])
        ranked_feats = ranked_features_list[i]

        sample_deltas = []
        for k in mask_k_levels:
            if k == 0:
                prob_curves[0].append(p0)
                delta_curves[0].append(0.0)
                flips[0].append(False)
                sample_deltas.append(0.0)
            else:
                top_k = ranked_feats[:k]
                col_indices = [feat_to_idx[f] for f in top_k if f in feat_to_idx]

                X_masked = X_orig[i:i+1].copy()
                X_masked[0, col_indices] = train_medians[col_indices]

                pk = float(detector.model.predict_proba(X_masked)[0, 1])
                yk = int(pk >= 0.5)

                prob_curves[k].append(pk)
                delta = p0 - pk
                delta_curves[k].append(delta)
                flips[k].append(yk != y0)
                sample_deltas.append(delta)

        # Trapezoidal AUC over mask_k_levels normalized by max(k)
        total_trapz = sum(
            (mask_k_levels[idx + 1] - mask_k_levels[idx]) * (sample_deltas[idx + 1] + sample_deltas[idx]) / 2.0
            for idx in range(len(mask_k_levels) - 1)
        )
        auc = float(total_trapz) / float(max(mask_k_levels))
        auc_values.append(auc)

    # Summary statistics
    summary = {
        "k_levels": list(mask_k_levels),
        "mean_prob_by_k": {k: float(np.mean(prob_curves[k])) for k in mask_k_levels},
        "mean_delta_by_k": {k: float(np.mean(delta_curves[k])) for k in mask_k_levels},
        "median_delta_by_k": {k: float(np.median(delta_curves[k])) for k in mask_k_levels},
        "std_delta_by_k": {k: float(np.std(delta_curves[k])) for k in mask_k_levels},
        "flip_rate_by_k": {k: float(np.mean(flips[k])) for k in mask_k_levels},
        "auc_mean": float(np.mean(auc_values)),
        "auc_median": float(np.median(auc_values)),
        "auc_std": float(np.std(auc_values)),
        "per_sample_auc": auc_values,
        "per_sample_prob_curves": prob_curves,
        "per_sample_delta_curves": delta_curves,
        "per_sample_flips": flips,
    }

    # Optional random deletion baseline
    if random_baseline:
        rng = np.random.RandomState(seed)
        rand_delta_curves: Dict[int, List[float]] = {k: [] for k in mask_k_levels}
        rand_auc_values: List[float] = []

        for i in range(n_samples):
            p0 = float(p0_all[i])
            sample_rand_deltas = []
            for k in mask_k_levels:
                if k == 0:
                    rand_delta_curves[0].append(0.0)
                    sample_rand_deltas.append(0.0)
                else:
                    rand_indices = rng.choice(len(feature_names), size=k, replace=False)
                    X_rand = X_orig[i:i+1].copy()
                    X_rand[0, rand_indices] = train_medians[rand_indices]
                    pk_rand = float(detector.model.predict_proba(X_rand)[0, 1])
                    d_rand = p0 - pk_rand
                    rand_delta_curves[k].append(d_rand)
                    sample_rand_deltas.append(d_rand)

            total_trapz_rand = sum(
                (mask_k_levels[idx + 1] - mask_k_levels[idx]) * (sample_rand_deltas[idx + 1] + sample_rand_deltas[idx]) / 2.0
                for idx in range(len(mask_k_levels) - 1)
            )
            auc_rand = float(total_trapz_rand) / float(max(mask_k_levels))
            rand_auc_values.append(auc_rand)

        summary["random_baseline"] = {
            "mean_delta_by_k": {k: float(np.mean(rand_delta_curves[k])) for k in mask_k_levels},
            "auc_mean": float(np.mean(rand_auc_values)),
            "auc_median": float(np.median(rand_auc_values)),
        }

    return summary


# =====================================================================
# B. Stability / Robustness Evaluation
# =====================================================================

def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def compute_spearman_correlation(
    feature_names: List[str],
    weights_a: Dict[str, float],
    weights_b: Dict[str, float],
) -> float:
    """Compute Spearman rank correlation across all features between two importance vectors."""
    vals_a = [weights_a.get(f, 0.0) for f in feature_names]
    vals_b = [weights_b.get(f, 0.0) for f in feature_names]
    if np.all(vals_a == vals_a[0]) or np.all(vals_b == vals_b[0]):
        return 0.0
    corr, _ = stats.spearmanr(vals_a, vals_b)
    return float(corr) if not math.isnan(corr) else 0.0


def perturb_tabular_samples(
    eval_df: pd.DataFrame,
    train_stds: np.ndarray,
    feature_names: List[str],
    noise_level: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Apply deterministic small Gaussian perturbation to continuous sensor features.

    Preserves integer log counts and boolean flags intact so schema validity is maintained.
    """
    rng = np.random.RandomState(seed)
    perturbed_df = eval_df.copy()

    # Identify continuous sensor feature column indices
    sensor_cols = set()
    for grp in ["current", "voltage", "speed", "wire_feed", "gas", "heat"]:
        sensor_cols.update(FEATURE_GROUPS[grp])

    for col in eval_df.columns:
        if col in sensor_cols and col in feature_names:
            idx = feature_names.index(col)
            sigma = float(train_stds[idx])
            noise = rng.normal(loc=0.0, scale=noise_level * sigma, size=len(eval_df))
            perturbed_df[col] = eval_df[col].astype(float) + noise

    return perturbed_df


def evaluate_tabular_stability(
    feature_names: List[str],
    orig_explanations: List[Dict[str, Any]],
    pert_explanations: List[Dict[str, Any]],
    k_levels: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[str, Any]:
    """
    Evaluate explanation stability under input perturbation.

    Measures:
      - Top-k Jaccard similarity between original and perturbed explanations.
      - Spearman rank correlation between original and perturbed feature attributions.
    """
    n_samples = len(orig_explanations)
    jaccard_by_k: Dict[int, List[float]] = {k: [] for k in k_levels}
    spearman_corrs: List[float] = []

    for i in range(n_samples):
        orig = orig_explanations[i]
        pert = pert_explanations[i]

        for k in k_levels:
            top_orig = set(orig["ranked_features"][:k])
            top_pert = set(pert["ranked_features"][:k])
            jaccard_by_k[k].append(compute_jaccard_similarity(top_orig, top_pert))

        corr = compute_spearman_correlation(
            feature_names,
            orig["feature_to_importance"],
            pert["feature_to_importance"],
        )
        spearman_corrs.append(corr)

    return {
        "jaccard_mean_by_k": {k: float(np.mean(jaccard_by_k[k])) for k in k_levels},
        "jaccard_median_by_k": {k: float(np.median(jaccard_by_k[k])) for k in k_levels},
        "jaccard_std_by_k": {k: float(np.std(jaccard_by_k[k])) for k in k_levels},
        "spearman_mean": float(np.mean(spearman_corrs)),
        "spearman_median": float(np.median(spearman_corrs)),
        "spearman_std": float(np.std(spearman_corrs)),
        "per_sample_jaccard": jaccard_by_k,
        "per_sample_spearman": spearman_corrs,
    }


def evaluate_attention_stability(
    notes_df: pd.DataFrame,
    k_levels: Tuple[int, ...] = (3, 5),
) -> Dict[str, Any]:
    """
    Evaluate DistilBERT self-attention explanation stability on operator notes.

    Perturbation: Append harmless trailing whitespace and punctuation.
    """
    jaccard_by_k: Dict[int, List[float]] = {k: [] for k in k_levels}
    evaluated_count = 0

    for _, row in notes_df.iterrows():
        text = str(row.get("note_text", "")).strip()
        if not text:
            continue

        exp_orig = explain_attention(text)
        # Minor semantics-preserving perturbation: trailing period and whitespace
        pert_text = text + " ."
        exp_pert = explain_attention(pert_text)

        if not exp_orig or not exp_pert:
            continue

        evaluated_count += 1
        for k in k_levels:
            orig_tokens = set(exp_orig["top_tokens"][:k])
            pert_tokens = set(exp_pert["top_tokens"][:k])
            jaccard_by_k[k].append(compute_jaccard_similarity(orig_tokens, pert_tokens))

    if evaluated_count == 0:
        return {"status": "N/A", "reason": "No valid operator notes available."}

    return {
        "evaluated_samples": evaluated_count,
        "jaccard_mean_by_k": {k: float(np.mean(jaccard_by_k[k])) for k in k_levels},
        "jaccard_median_by_k": {k: float(np.median(jaccard_by_k[k])) for k in k_levels},
        "jaccard_std_by_k": {k: float(np.std(jaccard_by_k[k])) for k in k_levels},
    }


# =====================================================================
# C. SHAP vs LIME vs Attention Agreement
# =====================================================================

def evaluate_shap_lime_agreement(
    feature_names: List[str],
    shap_explanations: List[Dict[str, Any]],
    lime_explanations: List[Dict[str, Any]],
    k_levels: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[str, Any]:
    """
    Evaluate feature-ranking agreement between SHAP and LIME on identical samples.

    Measures:
      - Top-k overlap count and top-k Jaccard similarity.
      - Spearman rank correlation between absolute feature importances.
    """
    n_samples = len(shap_explanations)
    overlap_by_k: Dict[int, List[int]] = {k: [] for k in k_levels}
    jaccard_by_k: Dict[int, List[float]] = {k: [] for k in k_levels}
    spearman_corrs: List[float] = []

    for i in range(n_samples):
        s_exp = shap_explanations[i]
        l_exp = lime_explanations[i]

        for k in k_levels:
            s_top = set(s_exp["ranked_features"][:k])
            l_top = set(l_exp["ranked_features"][:k])
            overlap = len(s_top & l_top)
            jacc = compute_jaccard_similarity(s_top, l_top)
            overlap_by_k[k].append(overlap)
            jaccard_by_k[k].append(jacc)

        corr = compute_spearman_correlation(
            feature_names,
            s_exp["feature_to_importance"],
            l_exp["feature_to_importance"],
        )
        spearman_corrs.append(corr)

    return {
        "k_levels": list(k_levels),
        "overlap_mean_by_k": {k: float(np.mean(overlap_by_k[k])) for k in k_levels},
        "jaccard_mean_by_k": {k: float(np.mean(jaccard_by_k[k])) for k in k_levels},
        "jaccard_median_by_k": {k: float(np.median(jaccard_by_k[k])) for k in k_levels},
        "jaccard_std_by_k": {k: float(np.std(jaccard_by_k[k])) for k in k_levels},
        "spearman_mean": float(np.mean(spearman_corrs)),
        "spearman_median": float(np.median(spearman_corrs)),
        "spearman_std": float(np.std(spearman_corrs)),
        "per_sample_jaccard": jaccard_by_k,
        "per_sample_overlap": overlap_by_k,
        "per_sample_spearman": spearman_corrs,
    }


# =====================================================================
# D. Normal vs Anomalous Explanation Structural Distributions
# =====================================================================

def compute_structural_distribution_metrics(
    explanations: List[Dict[str, Any]],
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Compute distribution concentration, Shannon entropy, and feature group shares.

    Metrics:
      - Top-1, Top-3, Top-5 concentration ratios.
      - Shannon entropy of normalized absolute importance: H = -sum(p * log2(p)).
      - k50, k80: Minimum number of features required to reach 50% and 80% cumulative importance.
      - Feature group percentage share.
    """
    top1_conc, top3_conc, top5_conc = [], [], []
    entropies = []
    k50_list, k80_list = [], []
    group_shares: Dict[str, List[float]] = {grp: [] for grp in FEATURE_GROUPS}

    for exp in explanations:
        importances = exp["ranked_importances"]
        total_imp = sum(importances)
        if total_imp <= 1e-9:
            top1_conc.append(0.0)
            top3_conc.append(0.0)
            top5_conc.append(0.0)
            entropies.append(0.0)
            k50_list.append(len(feature_names))
            k80_list.append(len(feature_names))
            for grp in FEATURE_GROUPS:
                group_shares[grp].append(0.0)
            continue

        top1_conc.append(float(importances[0]) / total_imp)
        top3_conc.append(float(sum(importances[:3])) / total_imp)
        top5_conc.append(float(sum(importances[:5])) / total_imp)

        # Shannon entropy
        probs = np.array(importances, dtype=float) / total_imp
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log2(probs)))
        entropies.append(entropy)

        # k50, k80
        cum = np.cumsum(importances) / total_imp
        k50 = int(np.searchsorted(cum, 0.50) + 1)
        k80 = int(np.searchsorted(cum, 0.80) + 1)
        k50_list.append(k50)
        k80_list.append(k80)

        # Feature group shares
        feat_map = exp["feature_to_importance"]
        for grp, cols in FEATURE_GROUPS.items():
            grp_sum = sum(feat_map.get(c, 0.0) for c in cols)
            group_shares[grp].append(float(grp_sum) / total_imp)

    return {
        "top1_concentration_mean": float(np.mean(top1_conc)),
        "top3_concentration_mean": float(np.mean(top3_conc)),
        "top5_concentration_mean": float(np.mean(top5_conc)),
        "entropy_mean": float(np.mean(entropies)),
        "entropy_median": float(np.median(entropies)),
        "entropy_std": float(np.std(entropies)),
        "k50_mean": float(np.mean(k50_list)),
        "k80_mean": float(np.mean(k80_list)),
        "group_shares_mean": {grp: float(np.mean(group_shares[grp])) for grp in FEATURE_GROUPS},
        "per_sample_top1": top1_conc,
        "per_sample_entropy": entropies,
        "per_sample_k50": k50_list,
        "per_sample_k80": k80_list,
    }


def compare_normal_vs_anomalous_distributions(
    shap_explanations: List[Dict[str, Any]],
    eval_df: pd.DataFrame,
    anomalous_indices: List[int],
    normal_indices: List[int],
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Compare structural explanation properties between normal and anomalous samples.

    Includes non-parametric Mann-Whitney U statistical comparison.
    """
    ano_exps = [shap_explanations[i] for i in anomalous_indices]
    norm_exps = [shap_explanations[i] for i in normal_indices]

    ano_stats = compute_structural_distribution_metrics(ano_exps, feature_names)
    norm_stats = compute_structural_distribution_metrics(norm_exps, feature_names)

    # Non-parametric Mann-Whitney U tests
    mwu_entropy = stats.mannwhitneyu(
        ano_stats["per_sample_entropy"], norm_stats["per_sample_entropy"], alternative="two-sided"
    )
    mwu_top1 = stats.mannwhitneyu(
        ano_stats["per_sample_top1"], norm_stats["per_sample_top1"], alternative="two-sided"
    )
    mwu_k50 = stats.mannwhitneyu(
        ano_stats["per_sample_k50"], norm_stats["per_sample_k50"], alternative="two-sided"
    )

    # Stratify by fault type where available
    fault_type_breakdown = {}
    ano_df = eval_df.iloc[anomalous_indices]
    for ftype, group in ano_df.groupby("anomaly_type"):
        if not ftype:
            continue
        sub_indices = [int(idx) for idx in group.index.tolist()]
        sub_exps = [shap_explanations[i] for i in sub_indices if i < len(shap_explanations)]
        if sub_exps:
            sub_stats = compute_structural_distribution_metrics(sub_exps, feature_names)
            fault_type_breakdown[ftype] = {
                "count": len(sub_exps),
                "top1_concentration": sub_stats["top1_concentration_mean"],
                "entropy": sub_stats["entropy_mean"],
                "k50": sub_stats["k50_mean"],
                "dominant_group": max(sub_stats["group_shares_mean"], key=sub_stats["group_shares_mean"].get),
            }

    return {
        "anomalous": ano_stats,
        "normal": norm_stats,
        "statistical_tests": {
            "entropy_mwu_stat": float(mwu_entropy.statistic),
            "entropy_mwu_pvalue": float(mwu_entropy.pvalue),
            "top1_mwu_stat": float(mwu_top1.statistic),
            "top1_mwu_pvalue": float(mwu_top1.pvalue),
            "k50_mwu_stat": float(mwu_k50.statistic),
            "k50_mwu_pvalue": float(mwu_k50.pvalue),
        },
        "fault_type_breakdown": fault_type_breakdown,
    }


# =====================================================================
# E. Computational Cost & Latency Benchmark
# =====================================================================

def measure_explainer_latency(
    explainer_type: str,
    explainer_obj: Any,
    eval_df: pd.DataFrame,
    notes_df: Optional[pd.DataFrame] = None,
    n_warmup: int = 1,
    max_eval: int = 60,
) -> Dict[str, Any]:
    """
    Measure execution latency for each available explainer under identical timing methodology.

    Uses time.perf_counter() with warm-up passes.
    """
    latencies_ms: List[float] = []

    if explainer_type == "shap":
        # Warmup
        for _ in range(n_warmup):
            _ = explainer_obj.explain_row(eval_df.iloc[[0]])
        # Timing
        for i in range(min(max_eval, len(eval_df))):
            row = eval_df.iloc[[i]]
            t0 = time.perf_counter()
            _ = explainer_obj.explain_row(row)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        model_name = "RandomForestClassifier (via AnomalyDetector)"

    elif explainer_type == "lime":
        # Warmup
        for _ in range(n_warmup):
            _ = explainer_obj.explain_row(eval_df.iloc[[0]])
        # Timing
        for i in range(min(max_eval, len(eval_df))):
            row = eval_df.iloc[[i]]
            t0 = time.perf_counter()
            _ = explainer_obj.explain_row(row)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        model_name = "RandomForestClassifier (via AnomalyDetector)"

    elif explainer_type == "attention":
        if notes_df is None or notes_df.empty:
            sample_texts = ["station_1 welding parameters within spec, no issues noted."] * 10
        else:
            sample_texts = [str(r).strip() for r in notes_df["note_text"] if str(r).strip()][:max_eval]

        # Warmup
        for _ in range(n_warmup):
            _ = explain_attention(sample_texts[0])
        # Timing
        for text in sample_texts:
            t0 = time.perf_counter()
            _ = explain_attention(text)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        model_name = "distilbert-base-uncased (AutoModel encoder)"

    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")

    return {
        "explainer": explainer_type,
        "model_used": model_name,
        "sample_count": len(latencies_ms),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "median_latency_ms": float(np.median(latencies_ms)),
        "std_latency_ms": float(np.std(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if len(latencies_ms) >= 20 else float(np.max(latencies_ms)),
        "min_latency_ms": float(np.min(latencies_ms)),
        "max_latency_ms": float(np.max(latencies_ms)),
        "latencies_ms": latencies_ms,
    }
