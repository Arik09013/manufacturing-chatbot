"""
Statistical analysis engine for Research Step 12:
Statistical Significance, Uncertainty & Effect-Size Analysis.

Provides pure mathematical and statistical methods operating strictly on empirical observations:
- Exact McNemar's test for paired classification disagreement
- Student's t confidence intervals for small-sample multi-seed runs (n=5)
- Exact Clopper-Pearson binomial confidence intervals for deterministic proportions
- Non-parametric percentile bootstrap confidence intervals
- Paired Wilcoxon signed-rank and independent Mann-Whitney U tests
- Cohen's d effect sizes (independent and paired)
- Holm-Bonferroni step-down family-wise error rate (FWER) correction
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats


# ==============================================================================
# 1. PAIRED BINARY CLASSIFICATION DISAGREEMENT (MCNEMAR'S EXACT TEST)
# ==============================================================================

def exact_mcnemar_test(
    b: int,
    c: int,
    total_n: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes McNemar's exact test for paired binary classification disagreements.
    
    Parameters
    ----------
    b : int
        Count of samples where Model 1 is correct and Model 2 is incorrect.
    c : int
        Count of samples where Model 1 is incorrect and Model 2 is correct.
    total_n : Optional[int]
        Total paired evaluation sample size.

    Returns
    -------
    Dict[str, Any] with discordant pairs, exact p-value, odds ratio, and risk difference.
    """
    discordant = b + c
    if discordant == 0:
        p_val = 1.0
        odds_ratio = 1.0
    else:
        # Two-sided exact binomial test with p0 = 0.5
        k_min = min(b, c)
        p_val = float(stats.binomtest(k_min, discordant, p=0.5, alternative="two-sided").pvalue)
        # Odds ratio b / c (with continuity handling if c == 0 or b == 0)
        if c == 0:
            odds_ratio = float("inf") if b > 0 else 1.0
        else:
            odds_ratio = float(b / c)

    risk_diff = float((b - c) / total_n) if total_n and total_n > 0 else None

    return {
        "b_model1_correct_model2_wrong": int(b),
        "c_model1_wrong_model2_correct": int(c),
        "discordant_pairs": int(discordant),
        "exact_p_value": round(p_val, 6),
        "odds_ratio": round(odds_ratio, 4) if odds_ratio != float("inf") else "inf",
        "risk_difference": round(risk_diff, 6) if risk_diff is not None else None,
        "is_significant_alpha_05": bool(p_val < 0.05),
    }


# ==============================================================================
# 2. MULTI-SEED & SAMPLE UNCERTAINTY (STUDENT'S T & BOOTSTRAP)
# ==============================================================================

def calculate_sample_uncertainty(
    values: Sequence[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Computes descriptive statistics and Student's t confidence interval for small-sample runs (e.g. n=5 seeds).
    
    Parameters
    ----------
    values : Sequence[float]
        Empirical measurements across seeds or repeated trials.
    confidence : float
        Nominal confidence level (default 0.95).

    Returns
    -------
    Dict[str, Any] with n, mean, std, se, cv, median, iqr, min, max, and CI.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "n": 0, "mean": 0.0, "std": 0.0, "se": 0.0, "cv_percent": 0.0,
            "median": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
        }

    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr_val = q75 - q25

    if n > 1:
        std_val = float(np.std(arr, ddof=1))
        se_val = std_val / math.sqrt(n)
        cv_val = (std_val / mean_val * 100.0) if mean_val != 0 else 0.0
        # Student's t critical value
        t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
        ci_lower = mean_val - t_crit * se_val
        ci_upper = mean_val + t_crit * se_val
    else:
        std_val = 0.0
        se_val = 0.0
        cv_val = 0.0
        ci_lower = mean_val
        ci_upper = mean_val

    return {
        "n": n,
        "mean": round(mean_val, 5),
        "std": round(std_val, 5),
        "se": round(se_val, 5),
        "cv_percent": round(cv_val, 3),
        "median": round(median_val, 5),
        "iqr": round(iqr_val, 5),
        "min": round(min_val, 5),
        "max": round(max_val, 5),
        "ci_lower": round(ci_lower, 5),
        "ci_upper": round(ci_upper, 5),
        "confidence_level": confidence,
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_bootstraps: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes a non-parametric percentile bootstrap confidence interval.
    
    Returns
    -------
    Tuple[point_estimate, ci_lower, ci_upper]
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    point_est = float(stat_fn(data))
    if n <= 1:
        return point_est, point_est, point_est

    bootstrap_stats = np.empty(n_bootstraps, dtype=float)
    for i in range(n_bootstraps):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = stat_fn(sample)

    alpha = 1.0 - ci
    lower = float(np.percentile(bootstrap_stats, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(bootstrap_stats, 100.0 * (1.0 - alpha / 2.0)))
    return round(point_est, 5), round(lower, 5), round(upper, 5)


# ==============================================================================
# 3. EXACT CLOPPER-PEARSON BINOMIAL CONFIDENCE INTERVALS
# ==============================================================================

def exact_clopper_pearson_ci(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Computes the exact Clopper-Pearson confidence interval for a binomial proportion k/n
    using the Beta distribution.
    
    Parameters
    ----------
    k : int
        Number of successes / matches.
    n : int
        Total number of trials.
    confidence : float
        Nominal coverage probability (default 0.95).

    Returns
    -------
    Tuple[observed_proportion, ci_lower, ci_upper]
    """
    if n == 0:
        return 0.0, 0.0, 0.0

    p_hat = k / n
    alpha = 1.0 - confidence

    # Lower bound
    if k == 0:
        ci_lower = 0.0
    else:
        ci_lower = float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))

    # Upper bound
    if k == n:
        ci_upper = 1.0
    else:
        ci_upper = float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))

    return round(p_hat, 5), round(ci_lower, 5), round(ci_upper, 5)


# ==============================================================================
# 4. EFFECT SIZE COMPUTATION
# ==============================================================================

def calculate_cohens_d(
    group1: Sequence[float],
    group2: Sequence[float],
    paired: bool = False,
) -> Dict[str, Any]:
    """
    Computes Cohen's d effect size for two groups (independent or paired).
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    if paired:
        if len(g1) != len(g2):
            raise ValueError("Paired Cohen's d requires equal sample sizes.")
        diffs = g1 - g2
        n = len(diffs)
        mean_diff = float(np.mean(diffs))
        s_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        if s_d == 0.0:
            d_val = float("inf") if mean_diff != 0 else 0.0
        else:
            d_val = mean_diff / s_d
        return {
            "type": "paired_cohens_dz",
            "mean_difference": round(mean_diff, 5),
            "std_difference": round(s_d, 5),
            "effect_size_d": round(d_val, 4) if d_val != float("inf") else "inf",
        }
    else:
        n1, n2 = len(g1), len(g2)
        m1, m2 = float(np.mean(g1)), float(np.mean(g2))
        s1 = float(np.std(g1, ddof=1)) if n1 > 1 else 0.0
        s2 = float(np.std(g2, ddof=1)) if n2 > 1 else 0.0
        # Pooled standard deviation
        df = (n1 - 1) + (n2 - 1)
        if df > 0:
            s_pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df)
        else:
            s_pooled = 0.0

        if s_pooled == 0.0:
            d_val = float("inf") if (m1 - m2) != 0 else 0.0
        else:
            d_val = (m1 - m2) / s_pooled

        return {
            "type": "independent_cohens_d",
            "mean_1": round(m1, 5),
            "mean_2": round(m2, 5),
            "mean_difference": round(m1 - m2, 5),
            "pooled_std": round(s_pooled, 5),
            "effect_size_d": round(d_val, 4) if d_val != float("inf") else "inf",
        }


# ==============================================================================
# 5. PAIRED NON-PARAMETRIC TESTS (WILCOXON & MANN-WHITNEY)
# ==============================================================================

def paired_wilcoxon_test(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """
    Performs Wilcoxon signed-rank test on paired continuous measurements.
    Handles zero differences properly.
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    if len(arr_x) != len(arr_y):
        raise ValueError("Paired Wilcoxon requires equal array lengths.")

    diffs = arr_x - arr_y
    non_zero = diffs[diffs != 0]
    n_total = len(diffs)
    n_nonzero = len(non_zero)

    if n_nonzero == 0:
        return {
            "n_total": n_total,
            "n_nonzero": 0,
            "median_difference": 0.0,
            "mean_difference": 0.0,
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant_alpha_05": False,
        }

    try:
        res = stats.wilcoxon(arr_x, arr_y, alternative=alternative, zero_method="wilcox")
        stat_val = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        stat_val = 0.0
        p_val = 1.0

    return {
        "n_total": n_total,
        "n_nonzero": n_nonzero,
        "median_difference": round(float(np.median(diffs)), 5),
        "mean_difference": round(float(np.mean(diffs)), 5),
        "statistic": round(stat_val, 3),
        "p_value": round(p_val, 6),
        "is_significant_alpha_05": bool(p_val < 0.05),
    }


def independent_mann_whitney_test(
    group1: Sequence[float],
    group2: Sequence[float],
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """
    Performs Mann-Whitney U test on two independent groups.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    if len(g1) == 0 or len(g2) == 0:
        return {"n1": len(g1), "n2": len(g2), "p_value": 1.0, "is_significant_alpha_05": False}

    res = stats.mannwhitneyu(g1, g2, alternative=alternative)
    p_val = float(res.pvalue)
    return {
        "n1": len(g1),
        "n2": len(g2),
        "median_1": round(float(np.median(g1)), 5),
        "median_2": round(float(np.median(g2)), 5),
        "statistic": float(res.statistic),
        "p_value": round(p_val, 6) if p_val >= 1e-6 else p_val,
        "is_significant_alpha_05": bool(p_val < 0.05),
    }


# ==============================================================================
# 6. MULTIPLE COMPARISON CONTROL (HOLM-BONFERRONI)
# ==============================================================================

def apply_holm_bonferroni(
    tests: List[Dict[str, Any]],
    p_key: str = "raw_p_value",
) -> List[Dict[str, Any]]:
    """
    Applies the step-down Holm-Bonferroni correction to a list of hypothesis tests.
    Ensures strong control of the family-wise error rate (FWER) at level alpha.
    
    Formula:
    For sorted p-values p_(1) <= p_(2) <= ... <= p_(m):
    p_adj_(i) = min(1.0, max_{j <= i} ((m - j + 1) * p_(j)))
    """
    m = len(tests)
    if m == 0:
        return []

    # Sort indices by raw p-value
    sorted_indices = sorted(range(m), key=lambda idx: tests[idx].get(p_key, 1.0))
    
    adj_pvalues = [1.0] * m
    running_max = 0.0

    for rank, orig_idx in enumerate(sorted_indices):
        raw_p = tests[orig_idx].get(p_key, 1.0)
        multiplier = m - rank
        current_adj = raw_p * multiplier
        running_max = max(running_max, current_adj)
        adj_pvalues[orig_idx] = min(1.0, running_max)

    # Attach adjusted p-values and significance decisions
    updated_tests = []
    for idx, test_dict in enumerate(tests):
        item = dict(test_dict)
        adj_p = round(adj_pvalues[idx], 6) if adj_pvalues[idx] >= 1e-6 else adj_pvalues[idx]
        item["adjusted_p_value"] = adj_p
        item["correction_method"] = "Holm-Bonferroni"
        item["is_significant_adjusted_05"] = bool(adj_p < 0.05)
        updated_tests.append(item)

    return updated_tests
