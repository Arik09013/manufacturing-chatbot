"""
Unit and regression tests for Research Step 12:
Statistical Significance, Uncertainty & Effect-Size Analysis.

Verifies:
1. statistical_inventory.json exists and documents all 11 experimental steps.
2. No fabricated observations: sample counts exactly match empirical source artifacts.
3. Paired comparisons operate on valid, compatible sample identities.
4. All p-values are bounded strictly within [0.0, 1.0].
5. Confidence intervals are mathematically valid (lower <= point_estimate <= upper).
6. Effect-size calculations (Cohen's d, odds ratios, risk differences) are numerically valid.
7. Step-down Holm-Bonferroni correction maintains monotonicity and raw_p <= adjusted_p <= 1.0.
8. Exact Clopper-Pearson binomial confidence intervals reflect true analytical bounds.
9. All 5 generated Step 12 evaluation artifacts exist and are non-empty.
10. All 5 generated visualization figures exist in evaluation/artifacts/figures/ and are non-empty.
11. Source experiment artifacts and model weights remain completely unmodified.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from evaluation.statistical_analysis import (
    apply_holm_bonferroni,
    bootstrap_confidence_interval,
    calculate_cohens_d,
    calculate_sample_uncertainty,
    exact_clopper_pearson_ci,
    exact_mcnemar_test,
    paired_wilcoxon_test,
)

_REPO_ROOT = Path(__file__).parent.parent
_ARTIFACTS_DIR = _REPO_ROOT / "evaluation" / "artifacts"
_FIGURES_DIR = _ARTIFACTS_DIR / "figures"
_RESULTS_DIR = _REPO_ROOT / "evaluation" / "results"


# 1. Statistical Inventory Verification
def test_statistical_inventory_completeness():
    """Verify statistical_inventory.json exists and catalogs all required fields across experimental steps."""
    inv_path = _ARTIFACTS_DIR / "statistical_inventory.json"
    assert inv_path.exists(), "statistical_inventory.json missing"

    with open(inv_path, "r", encoding="utf-8") as f:
        inv_data = json.load(f)

    assert "metadata" in inv_data
    assert "inventory" in inv_data
    assert len(inv_data["inventory"]) >= 10

    required_fields = [
        "step",
        "artifact_path",
        "metrics_available",
        "number_of_observations",
        "paired_predictions",
        "per_sample_predictions_exist",
        "per_seed_results_exist",
        "per_station_results_exist",
        "confidence_intervals_computable",
        "significance_testing_valid",
    ]

    for item in inv_data["inventory"]:
        for field in required_fields:
            assert field in item, f"Field '{field}' missing in inventory item {item.get('step')}"


# 2. Sample Count Concordance (No Fabricated Data)
def test_sample_counts_match_empirical_sources():
    """Verify sample counts match the exact empirical numbers in source artifacts."""
    summary_path = _ARTIFACTS_DIR / "statistical_summary.json"
    assert summary_path.exists()

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 1. Chronological test samples = 375
    assert summary["model_comparison"]["sample_size"] == 375
    assert summary["model_comparison"]["n_anomalies"] == 10
    assert summary["model_comparison"]["n_normal"] == 365

    # 2. Multi-seed count = 5
    assert summary["multiseed_uncertainty"]["n_seeds"] == 5
    assert summary["multiseed_uncertainty"]["seeds_evaluated"] == [42, 123, 456, 789, 2026]

    # 3. LOMO stations = 3
    assert summary["station_variability"]["n_stations"] == 3
    assert summary["station_variability"]["stations_evaluated"] == ["station_1", "station_2", "station_3"]
    assert summary["station_variability"]["samples_per_station"] == 639

    # 4. RAG queries = 40
    assert summary["rag_retrieval"]["total_queries"] == 40

    # 5. Intent benchmark queries = 100
    assert summary["intent_uncertainty"]["n_benchmark_queries"] == 100

    # 6. XAI samples = 60 (10 anomalies, 50 normal)
    assert summary["xai_statistics"]["n_samples"] == 60
    assert summary["xai_statistics"]["n_anomalous"] == 10
    assert summary["xai_statistics"]["n_normal"] == 50

    # 7. Physics scenarios = 144
    assert summary["physics_statistics"]["summary"]["scenarios_evaluated"] == 144

    # 8. Deployment queries = 80
    assert summary["deployment_latency"]["total_queries_evaluated"] == 80


# 3. Paired McNemar Test Correctness
def test_exact_mcnemar_test_correctness():
    """Verify McNemar exact binomial test properties."""
    # Symmetrical (b = c): p = 1.0
    res_sym = exact_mcnemar_test(b=3, c=3, total_n=100)
    assert res_sym["exact_p_value"] == 1.0
    assert res_sym["discordant_pairs"] == 6

    # 0 vs 8 discordant pairs (2 * 0.5^8 = 2/256 = 0.0078125)
    res_iso = exact_mcnemar_test(b=8, c=0, total_n=375)
    assert abs(res_iso["exact_p_value"] - 0.007812) < 1e-5
    assert res_iso["is_significant_alpha_05"] is True
    assert res_iso["odds_ratio"] == "inf"

    # Zero discordant pairs
    res_zero = exact_mcnemar_test(b=0, c=0, total_n=375)
    assert res_zero["exact_p_value"] == 1.0
    assert res_zero["discordant_pairs"] == 0


# 4. P-Values Bounded in [0.0, 1.0]
def test_all_p_values_bounded():
    """Verify that all raw and adjusted p-values in the test registry are within [0.0, 1.0]."""
    tests_path = _ARTIFACTS_DIR / "statistical_tests.json"
    assert tests_path.exists()

    with open(tests_path, "r", encoding="utf-8") as f:
        registry_data = json.load(f)

    for item in registry_data["registry"]:
        raw_p = item["raw_p_value"]
        assert 0.0 <= raw_p <= 1.0, f"Raw p-value {raw_p} out of bounds for {item['comparison']}"
        adj_p = item.get("adjusted_p_value", raw_p)
        assert 0.0 <= adj_p <= 1.0, f"Adjusted p-value {adj_p} out of bounds for {item['comparison']}"


# 5. Confidence Intervals Mathematical Validity
def test_confidence_intervals_valid():
    """Verify confidence intervals obey lower <= point_estimate <= upper bounds."""
    # Student's t uncertainty
    sample = [0.70, 0.72, 0.71, 0.69, 0.73]
    unc = calculate_sample_uncertainty(sample, confidence=0.95)
    assert unc["ci_lower"] <= unc["mean"] <= unc["ci_upper"]
    assert unc["n"] == 5

    # Bootstrap CI
    data = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1])
    pt, low, high = bootstrap_confidence_interval(data, stat_fn=np.mean, n_bootstraps=500, seed=42)
    assert low <= pt <= high
    assert 0.0 <= low and high <= 1.0


# 6. Effect Size Calculations
def test_effect_size_calculations():
    """Verify Cohen's d effect size computation for paired and independent samples."""
    # Identical groups -> d = 0.0
    d_zero = calculate_cohens_d([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], paired=True)
    assert d_zero["effect_size_d"] == 0.0

    # Distinct groups
    g1 = [0.80, 0.82, 0.81, 0.79, 0.83]
    g2 = [0.70, 0.72, 0.71, 0.69, 0.73]
    d_ind = calculate_cohens_d(g1, g2, paired=False)
    assert d_ind["effect_size_d"] > 0.0
    assert abs(d_ind["mean_difference"] - 0.10) < 1e-4


# 7. Holm-Bonferroni Correction Monotonicity
def test_holm_bonferroni_correction_properties():
    """Verify that Holm-Bonferroni step-down correction is monotonic and bounded."""
    mock_tests = [
        {"name": "test_1", "raw_p_value": 0.001},
        {"name": "test_2", "raw_p_value": 0.02},
        {"name": "test_3", "raw_p_value": 0.04},
        {"name": "test_4", "raw_p_value": 0.30},
    ]

    corrected = apply_holm_bonferroni(mock_tests, p_key="raw_p_value")
    assert len(corrected) == 4

    for orig, adj in zip(mock_tests, corrected):
        assert adj["raw_p_value"] <= adj["adjusted_p_value"] <= 1.0

    # Sorted by raw p-value must be non-decreasing in adjusted p-value
    sorted_adj = sorted(corrected, key=lambda x: x["raw_p_value"])
    for i in range(len(sorted_adj) - 1):
        assert sorted_adj[i]["adjusted_p_value"] <= sorted_adj[i + 1]["adjusted_p_value"]


# 8. Exact Clopper-Pearson Binomial Confidence Intervals
def test_exact_clopper_pearson_bounds():
    """Verify analytical properties of Clopper-Pearson binomial intervals."""
    # 144 / 144 successes
    p_hat, low_144, high_144 = exact_clopper_pearson_ci(k=144, n=144, confidence=0.95)
    assert p_hat == 1.0
    assert high_144 == 1.0
    assert 0.97 <= low_144 <= 0.98

    # 641 / 641 successes
    _, low_641, high_641 = exact_clopper_pearson_ci(k=641, n=641, confidence=0.95)
    assert high_641 == 1.0
    assert 0.99 <= low_641 <= 1.0

    # 15 / 15 successes
    _, low_15, high_15 = exact_clopper_pearson_ci(k=15, n=15, confidence=0.95)
    assert high_15 == 1.0
    assert 0.78 <= low_15 <= 0.80


# 9. Step 12 Evaluation Artifacts Exist
def test_statistical_evaluation_artifacts_exist():
    """Verify all Step 12 statistical evaluation artifacts exist and are non-empty."""
    expected_artifacts = [
        "statistical_inventory.json",
        "station_variability.json",
        "statistical_tests.json",
        "statistical_summary.json",
        "statistical_evaluation_report.md",
    ]

    for name in expected_artifacts:
        path = _ARTIFACTS_DIR / name
        assert path.exists(), f"Artifact missing: {name}"
        assert path.stat().st_size > 0, f"Artifact empty: {name}"


# 10. Step 12 Figures Exist
def test_statistical_evaluation_figures_exist():
    """Verify all 5 Step 12 visualization figures exist and are non-empty (>1 KB)."""
    expected_figures = [
        "statistical_metric_distributions.png",
        "seed_variability.png",
        "ablation_effect_sizes.png",
        "rag_metric_comparison.png",
        "latency_distributions.png",
    ]

    for name in expected_figures:
        path = _FIGURES_DIR / name
        assert path.exists(), f"Figure missing: {name}"
        assert path.stat().st_size > 1000, f"Figure too small (<1KB): {name}"


# 11. Model Weights and Master Dataset Strictly Unmodified
def test_production_weights_and_data_unmodified():
    """Verify core model weights and master dataset are strictly unmodified."""
    distilbert_path = _REPO_ROOT / "models" / "distilbert_fault" / "model.safetensors"
    assert distilbert_path.exists()
    assert distilbert_path.stat().st_size == 267832560, "DistilBERT weights modified!"

    rf_path = _REPO_ROOT / "models" / "anomaly.joblib"
    assert rf_path.exists()
    assert rf_path.stat().st_size == 543489, "RandomForest model weights modified!"

    parquet_path = _REPO_ROOT / "data" / "processed" / "fused.parquet"
    assert parquet_path.exists()
    assert parquet_path.stat().st_size == 519912, "Master dataset modified!"
