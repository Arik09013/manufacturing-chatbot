"""
Unit and Research Integrity Tests for Step 8: Intent Classification Stress Evaluation.

Tests:
  1. Benchmark taxonomy membership and single-label validity
  2. Prediction interface contract and confidence bounds
  3. Evaluation determinism (reproducible zero-stochasticity)
  4. Non-mutation of underlying classifier state
  5. Mathematical exactness of classification metrics and confusion matrix
  6. Out-of-scope abstention / false acceptance bounds
  7. Artifact generation and schema conformity
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evaluation.eval_intent_stress import (
    compute_classification_stats,
    compute_confusion_matrix,
    predict_intent,
    run_intent_stress_evaluation,
)
from evaluation.intent_stress import (
    BENCHMARK_QUERIES,
    DIFFICULTY_LEVELS,
    INTENT_TAXONOMY,
    PERTURBATION_TYPES,
    get_benchmark_queries,
    validate_benchmark_schema,
)
from src.chat.intent import IntentResult, classify_intent


# =====================================================================
# 1. Benchmark Schema and Taxonomy Integrity
# =====================================================================

def test_benchmark_taxonomy_membership():
    """Verify all benchmark queries belong strictly to the existing intent taxonomy."""
    queries = get_benchmark_queries()
    assert len(queries) == 100, f"Expected 100 queries, got {len(queries)}"

    is_valid, errors = validate_benchmark_schema(queries)
    assert is_valid, f"Benchmark schema invalid: {errors}"
    assert len(errors) == 0

    intent_counts = {intent: 0 for intent in INTENT_TAXONOMY}
    for q in queries:
        assert q["expected_intent"] in INTENT_TAXONOMY
        intent_counts[q["expected_intent"]] += 1

    # Exactly 20 per intent
    for intent, count in intent_counts.items():
        assert count == 20, f"Expected 20 queries for intent '{intent}', got {count}"


def test_benchmark_perturbation_and_difficulty_coverage():
    """Verify all 12 perturbation types and 3 difficulty tiers are populated."""
    queries = get_benchmark_queries()
    seen_perts = {q["perturbation_type"] for q in queries}
    seen_diffs = {q["difficulty"] for q in queries}

    assert seen_perts == set(PERTURBATION_TYPES)
    assert seen_diffs == set(DIFFICULTY_LEVELS)


# =====================================================================
# 2. Prediction Interface Contract
# =====================================================================

def test_predict_intent_contract():
    """Verify predict_intent outputs expected keys and valid ranges."""
    sample = "Why did station_1 stop at 14:00?"
    out = predict_intent(sample)

    assert "predicted_intent" in out
    assert out["predicted_intent"] in INTENT_TAXONOMY
    assert "predicted_scope" in out
    assert out["predicted_scope"] in ("in_scope", "out_of_scope")
    assert isinstance(out["in_scope"], bool)
    assert out["confidence"] in (0.8, 1.0)
    assert isinstance(out["matched_term"], str)
    assert isinstance(out["reason"], str)


def test_out_of_scope_abstention():
    """Verify unsupported query is mapped to out_of_scope with in_scope=False."""
    sample = "Who won the World Cup?"
    out = predict_intent(sample)

    assert out["predicted_intent"] == "out_of_scope"
    assert out["predicted_scope"] == "out_of_scope"
    assert out["in_scope"] is False
    assert out["confidence"] == 1.0


# =====================================================================
# 3. Determinism and Non-Mutation Checks
# =====================================================================

def test_evaluation_determinism():
    """Verify two sequential evaluation passes yield bit-for-bit identical outputs."""
    queries = get_benchmark_queries()[:20]

    pass_1 = [predict_intent(q["text"]) for q in queries]
    pass_2 = [predict_intent(q["text"]) for q in queries]

    for r1, r2 in zip(pass_1, pass_2):
        assert r1["predicted_intent"] == r2["predicted_intent"]
        assert r1["predicted_scope"] == r2["predicted_scope"]
        assert r1["confidence"] == r2["confidence"]
        assert r1["matched_term"] == r2["matched_term"]


def test_evaluation_does_not_modify_classifier_state():
    """Verify production classify_intent behavior remains identical on reference inputs."""
    ref_query = "machine_1 is running hot"

    before = classify_intent(ref_query)
    _ = run_intent_stress_evaluation()
    after = classify_intent(ref_query)

    assert before.in_scope == after.in_scope
    assert before.confidence == after.confidence
    assert before.matched_term == after.matched_term
    assert before.reason == after.reason


# =====================================================================
# 4. Metric Computation Math
# =====================================================================

def test_metrics_calculation_exactness():
    """Verify classification metrics on synthetic known cases."""
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "A"]
    labels = ["A", "B"]

    stats = compute_classification_stats(y_true, y_pred, labels)
    assert stats["accuracy"] == 0.5000
    assert stats["macro_f1"] == 0.5000
    assert stats["per_class"]["A"]["precision"] == 0.5000
    assert stats["per_class"]["A"]["recall"] == 0.5000

    cm = compute_confusion_matrix(y_true, y_pred, labels)
    assert cm == [[1, 1], [1, 1]]


# =====================================================================
# 5. Artifact Verification
# =====================================================================

def test_artifacts_exist_and_conform():
    """Verify evaluation and inventory artifacts exist and contain valid fields."""
    inv_path = Path("evaluation/artifacts/intent_inventory.json")
    results_path = Path("evaluation/artifacts/intent_stress_results.json")
    summary_path = Path("evaluation/artifacts/intent_stress_summary.json")
    report_path = Path("evaluation/artifacts/intent_stress_report.md")

    assert inv_path.exists()
    assert results_path.exists()
    assert summary_path.exists()
    assert report_path.exists()

    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    assert res["multi_class_intent_metrics"]["accuracy"] >= 0.80
    assert res["binary_scope_metrics"]["accuracy"] >= 0.95
    assert res["abstention_analysis"]["correct_abstention_rate"] >= 0.95
    assert res["abstention_analysis"]["false_acceptance_rate"] == 0.0
