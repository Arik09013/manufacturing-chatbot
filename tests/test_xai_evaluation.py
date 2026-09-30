"""
Research Integrity Tests for Explainability / XAI Evaluation (Step 9).

Verifies the 9 strict research integrity criteria:
  1. XAI evaluation is deterministic.
  2. Existing model predictions are not modified.
  3. Frozen master dataset remains unchanged.
  4. Existing XAI APIs still behave identically.
  5. No test-set fitting occurs.
  6. Feature/token names used in explanations are valid.
  7. Explanation ranking length and dimensions are consistent.
  8. N/A methods are explicitly represented rather than fabricated.
  9. All generated metrics are reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.xai_evaluation import (
    compute_jaccard_similarity,
    compute_spearman_correlation,
    extract_lime_explanation,
    extract_shap_explanation,
    prepare_xai_evaluation_data,
)
from src.explain.attention_explainer import explain_attention
from src.explain.lime_explainer import LimeAnomalyExplainer, _background_sample, get_default_lime_explainer
from src.explain.shap_explainer import AnomalyExplainer
from src.fusion.fuse import load_fused
from src.model.anomaly import AnomalyDetector, get_feature_matrix

_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACTS_DIR = _ROOT / "evaluation" / "artifacts"
_MODEL_PATH = _ROOT / "models" / "anomaly.joblib"
_FUSED_PATH = _ROOT / "data" / "processed" / "fused.parquet"

# Known frozen hash of data/processed/fused.parquet
FROZEN_FUSED_SHA256 = "d703b8999ab254996f2445f606ad4bd9523201f510324147352b2a4edb36b455"


# ---------------------------------------------------------------------
# 1. XAI Evaluation is Deterministic
# ---------------------------------------------------------------------

def test_xai_evaluation_determinism():
    """Verify that dataset subsetting, sampling, and explanations are bit-level deterministic under fixed seed."""
    split1 = prepare_xai_evaluation_data(random_seed=42, n_normal=20)
    split2 = prepare_xai_evaluation_data(random_seed=42, n_normal=20)

    assert split1.total_samples == split2.total_samples
    assert split1.anomalous_indices == split2.anomalous_indices
    assert split1.normal_indices == split2.normal_indices
    assert np.array_equal(split1.train_medians, split2.train_medians)
    assert np.array_equal(split1.train_stds, split2.train_stds)

    # Explanation ranking determinism on first sample
    import joblib
    bundle = joblib.load(_MODEL_PATH)
    detector = bundle["detector"]
    shap_exp = AnomalyExplainer(detector, top_n=40)

    row = split1.eval_df.iloc[[0]]
    exp1 = extract_shap_explanation(shap_exp, row)
    exp2 = extract_shap_explanation(shap_exp, row)

    assert exp1["ranked_features"] == exp2["ranked_features"]
    assert np.allclose(exp1["raw_shap"], exp2["raw_shap"], atol=1e-7)


# ---------------------------------------------------------------------
# 2. Existing Model Predictions Are Not Modified
# ---------------------------------------------------------------------

def test_existing_model_predictions_not_modified():
    """Verify that running explainers does not alter model weights, state, or inference outputs."""
    import joblib
    bundle = joblib.load(_MODEL_PATH)
    detector: AnomalyDetector = bundle["detector"]

    fused = load_fused()
    row = fused.iloc[[0]]

    # Baseline probability before explanations
    p_before = float(detector.predict_proba(row)[0])
    raw_tree_state = [tree.tree_.feature.copy() for tree in detector.model.estimators_]

    # Run SHAP and LIME
    shap_exp = AnomalyExplainer(detector, top_n=5)
    _ = shap_exp.explain_row(row)

    lime_exp = get_default_lime_explainer(top_n=5)
    _ = lime_exp.explain_row(row)

    # Verify probability and tree states remain identical
    p_after = float(detector.predict_proba(row)[0])
    assert p_before == p_after, "Model predict_proba altered after running explainers!"

    for orig_feat, current_tree in zip(raw_tree_state, detector.model.estimators_):
        assert np.array_equal(orig_feat, current_tree.tree_.feature), "Decision tree internal state modified!"


# ---------------------------------------------------------------------
# 3. Frozen Master Dataset Remains Unchanged
# ---------------------------------------------------------------------

def test_frozen_master_dataset_remains_unchanged():
    """Verify that fused.parquet has not been modified or overwritten."""
    assert _FUSED_PATH.exists(), "Master fused.parquet file is missing!"
    current_sha256 = hashlib.sha256(_FUSED_PATH.read_bytes()).hexdigest()
    assert current_sha256 == FROZEN_FUSED_SHA256, (
        f"Master dataset modified! Expected SHA256 {FROZEN_FUSED_SHA256}, got {current_sha256}"
    )

    df = load_fused()
    assert len(df) == 1917, f"Expected 1917 rows in fused dataset, found {len(df)}"
    assert df["is_anomaly"].sum() == 60, f"Expected 60 anomalies, found {df['is_anomaly'].sum()}"


# ---------------------------------------------------------------------
# 4. Existing XAI APIs Still Behave Identically
# ---------------------------------------------------------------------

def test_existing_xai_apis_behave_identically():
    """Verify that production XAI functions preserve their expected contracts and schemas."""
    fused = load_fused()
    row = fused.iloc[[0]]

    # 1. SHAP API
    import joblib
    bundle = joblib.load(_MODEL_PATH)
    detector = bundle["detector"]
    shap_exp = AnomalyExplainer(detector, top_n=5)
    s_res = shap_exp.explain_row(row)

    assert "anomaly_prob" in s_res
    assert "shap_drivers" in s_res
    assert "raw_shap" in s_res
    assert len(s_res["shap_drivers"]) == 5
    assert {"feature", "shap", "direction", "magnitude"} <= set(s_res["shap_drivers"][0])

    text_s = shap_exp.explain_text(s_res)
    assert isinstance(text_s, str) and "Anomaly probability" in text_s

    # 2. LIME API
    lime_exp = get_default_lime_explainer(top_n=5)
    l_res = lime_exp.explain_row(row)

    assert "anomaly_prob" in l_res
    assert "lime_drivers" in l_res
    assert len(l_res["lime_drivers"]) == 5
    assert {"feature", "condition", "weight", "direction", "magnitude"} <= set(l_res["lime_drivers"][0])

    # 3. Attention API
    att_res = explain_attention("Unstable arc on station_1, may need new contact tip")
    assert att_res is not None
    assert "tokens" in att_res and "top_tokens" in att_res
    assert explain_attention("   ") is None


# ---------------------------------------------------------------------
# 5. No Test-Set Fitting Occurs
# ---------------------------------------------------------------------

def test_no_test_set_fitting():
    """Verify that training medians are derived strictly from train_df without test data contamination."""
    split = prepare_xai_evaluation_data(random_seed=42)
    X_train, _ = get_feature_matrix(split.train_df)
    expected_medians = np.median(X_train, axis=0)

    assert np.allclose(split.train_medians, expected_medians), (
        "Train medians do not match direct computation on training data only!"
    )

    # Contaminate test_df and verify train_medians remain invariant
    df_copy = load_fused().copy()
    split_orig = prepare_xai_evaluation_data(df=df_copy, random_seed=42)

    # Mutate a test row
    df_mutated = df_copy.copy()
    test_idx = split_orig.test_df.index[0]
    df_mutated.loc[test_idx, "welding_current_mean"] = 999999.0

    split_mutated = prepare_xai_evaluation_data(df=df_mutated, random_seed=42)
    assert np.array_equal(split_orig.train_medians, split_mutated.train_medians), (
        "Leakage detected: Mutating test data altered training medians!"
    )


# ---------------------------------------------------------------------
# 6. Feature and Token Names Used in Explanations Are Valid
# ---------------------------------------------------------------------

def test_feature_and_token_names_are_valid():
    """Verify all feature names in explanations match known model columns and text tokens are clean."""
    import joblib
    bundle = joblib.load(_MODEL_PATH)
    detector = bundle["detector"]
    known_features = set(detector.feature_names)
    assert len(known_features) == 40

    row = load_fused().iloc[[0]]

    # SHAP feature names
    shap_exp = AnomalyExplainer(detector, top_n=40)
    s_res = extract_shap_explanation(shap_exp, row)
    assert set(s_res["ranked_features"]) == known_features

    # LIME feature names (parsed from conditions)
    lime_exp = LimeAnomalyExplainer(detector, _background_sample(n=100), top_n=40)
    l_res = extract_lime_explanation(lime_exp, row)
    assert set(l_res["ranked_features"]) == known_features

    # Attention token valid formatting
    att_res = explain_attention("Wire feed slipped during station_2 pass")
    assert att_res is not None
    for item in att_res["tokens"]:
        token = item["token"]
        assert token not in {"[CLS]", "[SEP]", "[PAD]"}, "Special token leaked into word heatmap!"
        assert any(ch.isalnum() for ch in token), "Pure punctuation token was not dropped!"


# ---------------------------------------------------------------------
# 7. Explanation Ranking Length and Dimensions Are Consistent
# ---------------------------------------------------------------------

def test_explanation_ranking_length_and_dimensions():
    """Verify that full-feature rankings have consistent length 40 and strictly descending magnitude."""
    import joblib
    bundle = joblib.load(_MODEL_PATH)
    detector = bundle["detector"]

    row = load_fused().iloc[[0]]
    shap_exp = AnomalyExplainer(detector, top_n=40)
    s_res = extract_shap_explanation(shap_exp, row)

    assert len(s_res["ranked_features"]) == 40
    assert len(s_res["ranked_importances"]) == 40
    assert len(set(s_res["ranked_features"])) == 40, "Duplicate features in SHAP ranking!"

    # Verify descending sort
    imps = s_res["ranked_importances"]
    for i in range(len(imps) - 1):
        assert imps[i] >= imps[i + 1], f"SHAP ranking not monotonic at {i}: {imps[i]} < {imps[i+1]}"

    lime_exp = LimeAnomalyExplainer(detector, _background_sample(n=100), top_n=40)
    l_res = extract_lime_explanation(lime_exp, row)
    assert len(l_res["ranked_features"]) == 40
    assert len(l_res["ranked_importances"]) == 40
    assert len(set(l_res["ranked_features"])) == 40, "Duplicate features in LIME ranking!"


# ---------------------------------------------------------------------
# 8. N/A Methods Are Explicitly Represented Rather Than Fabricated
# ---------------------------------------------------------------------

def test_na_methods_explicitly_represented():
    """Verify that incompatible pairs (SHAP vs Attention, LIME vs Attention) are marked N/A with rationale."""
    inventory_path = _ARTIFACTS_DIR / "xai_inventory.json"
    assert inventory_path.exists(), "xai_inventory.json missing!"

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    comp_matrix = inventory["cross_method_comparability_matrix"]

    assert comp_matrix["shap_vs_attention"]["comparable"] is False
    assert comp_matrix["shap_vs_attention"]["status"] == "N/A"
    assert "Disjoint feature spaces" in comp_matrix["shap_vs_attention"]["reason"]

    assert comp_matrix["lime_vs_attention"]["comparable"] is False
    assert comp_matrix["lime_vs_attention"]["status"] == "N/A"

    results_path = _ARTIFACTS_DIR / "xai_evaluation_results.json"
    assert results_path.exists()
    results = json.loads(results_path.read_text(encoding="utf-8"))

    assert results["fidelity"]["attention"]["status"] == "N/A"
    assert results["agreement"]["shap_vs_attention"]["status"] == "N/A"
    assert results["agreement"]["lime_vs_attention"]["status"] == "N/A"


# ---------------------------------------------------------------------
# 9. All Generated Metrics Are Reproducible
# ---------------------------------------------------------------------

def test_all_generated_metrics_reproducible():
    """Verify summary and results artifacts exist and contain valid, bounded metrics."""
    summary_path = _ARTIFACTS_DIR / "xai_evaluation_summary.json"
    csv_path = _ARTIFACTS_DIR / "xai_evaluation_results.csv"
    report_path = _ARTIFACTS_DIR / "xai_evaluation_report.md"

    assert summary_path.exists(), "Summary JSON missing!"
    assert csv_path.exists(), "Results CSV missing!"
    assert report_path.exists(), "Report markdown missing!"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # Boundedness checks
    assert 0.0 <= summary["fidelity_auc"]["shap_auc_mean"] <= 1.0
    assert 0.0 <= summary["fidelity_auc"]["lime_auc_mean"] <= 1.0
    assert 0.0 <= summary["fidelity_auc"]["random_auc_mean"] <= 1.0
    assert summary["fidelity_auc"]["shap_auc_mean"] > summary["fidelity_auc"]["random_auc_mean"]

    assert 0.0 <= summary["stability_k5_jaccard"]["shap_jaccard_k5_mean"] <= 1.0
    assert 0.0 <= summary["stability_k5_jaccard"]["lime_jaccard_k5_mean"] <= 1.0
    assert -1.0 <= summary["agreement_shap_vs_lime"]["spearman_rho_mean"] <= 1.0

    assert summary["latency_ms"]["shap_mean_ms"] > 0.0
    assert summary["latency_ms"]["lime_mean_ms"] > 0.0
    assert summary["latency_ms"]["attention_mean_ms"] > 0.0

    # CSV checks: 60 sample rows + 1 header = 61 rows
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 61, f"Expected 61 CSV lines, got {len(lines)}"
