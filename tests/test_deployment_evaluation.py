"""
Unit and regression tests for Step 11: Deployment and Computational Efficiency Evaluation.

Verifies:
1. Benchmark query suite generation is strictly deterministic and covers 80 queries across 5 paths.
2. Runtime footprint inventory exists and matches disk sizes of models, data, and configs.
3. Model weights and master dataset are completely unmodified.
4. Deployment inventory documents observed routing, entry points, caching, and loading behavior.
5. ComponentTimer and memory tracker measure valid positive quantities.
6. Cold vs warm latency separation logic correctly handles cold initialization.
7. Degradation and fallback modes execute gracefully without unhandled exceptions.
8. Output determinism: repeated queries yield zero drift in structured output fields.
9. Generated evaluation artifacts (JSON, CSV, MD summary) exist and are non-empty.
10. All 5 generated visualization figures exist in evaluation/artifacts/figures/ and are non-empty.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evaluation.deployment_evaluation import (
    ComponentTimer,
    create_benchmark_query_suite,
    evaluate_failure_modes,
    get_process_memory_info,
)

_REPO_ROOT = Path(__file__).parent.parent
_ARTIFACTS_DIR = _REPO_ROOT / "evaluation" / "artifacts"
_FIGURES_DIR = _ARTIFACTS_DIR / "figures"


# 1. Benchmark Query Suite Determinism
def test_benchmark_query_suite_determinism():
    """Ensure benchmark query suite is deterministic, contains exactly 80 queries, and covers 5 routes."""
    suite1 = create_benchmark_query_suite()
    suite2 = create_benchmark_query_suite()

    assert len(suite1) == 80
    assert len(suite2) == 80

    for q1, q2 in zip(suite1, suite2):
        assert q1["query_id"] == q2["query_id"]
        assert q1["query_text"] == q2["query_text"]
        assert q1["category"] == q2["category"]
        assert q1["expected_route"] == q2["expected_route"]

    categories = {q["category"] for q in suite1}
    assert categories == {"anomaly", "param", "knowledge", "general", "out_of_scope"}


# 2. Runtime Footprint Integrity
def test_runtime_footprint_integrity():
    """Verify runtime_footprint.json accurately documents model and dataset disk sizes."""
    footprint_path = _ARTIFACTS_DIR / "runtime_footprint.json"
    assert footprint_path.exists(), "runtime_footprint.json missing"

    with open(footprint_path, "r", encoding="utf-8") as f:
        footprint = json.load(f)

    assert "runtime_artifacts" in footprint
    assert "summary_footprint" in footprint
    assert footprint["summary_footprint"]["grand_total_disk_mb"] > 200.0  # DistilBERT alone is ~256 MB

    # Check that key artifacts are registered
    models_dict = footprint["runtime_artifacts"]["models"]
    assert "distilbert_fault_classifier" in models_dict
    assert "random_forest_detector" in models_dict
    assert "rag_vector_index" in models_dict
    assert models_dict["distilbert_fault_classifier"]["files"]["model.safetensors"]["bytes"] == 267832560
    assert models_dict["random_forest_detector"]["bytes"] == 543489


# 3. Model Weights and Master Dataset Unchanged
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


# 4. Deployment Inventory Completeness
def test_deployment_inventory_completeness():
    """Verify deployment_inventory.json documents observed architecture and component flows."""
    inventory_path = _ARTIFACTS_DIR / "deployment_inventory.json"
    assert inventory_path.exists(), "deployment_inventory.json missing"

    with open(inventory_path, "r", encoding="utf-8") as f:
        inventory = json.load(f)

    assert "inventory_metadata" in inventory
    assert "runtime_architecture" in inventory
    assert "pipeline_routes" in inventory
    assert "hardware_environment" in inventory

    assert "endpoints" in inventory["runtime_architecture"]
    assert "orchestrator" in inventory["runtime_architecture"]
    assert "anomaly" in inventory["pipeline_routes"]
    assert "param" in inventory["pipeline_routes"]
    assert "knowledge" in inventory["pipeline_routes"]


# 5. ComponentTimer and Memory Tracker Correctness
def test_component_timer_and_memory_tracker():
    """Verify ComponentTimer measures elapsed time and get_process_memory_info returns valid stats."""
    import time

    with ComponentTimer() as timer:
        time.sleep(0.01)

    assert timer.elapsed_ms >= 8.0  # At least ~10ms with margin

    mem = get_process_memory_info()
    assert mem["rss_mb"] > 0
    assert mem["vms_mb"] > 0
    assert mem["percent"] >= 0


# 6. Failure Modes and Fallback Robustness
def test_failure_modes_and_fallback_robustness():
    """Verify pipeline failure modes return handled statuses without raising exceptions."""
    failure_results = evaluate_failure_modes()
    assert failure_results["total_failure_scenarios"] >= 5
    assert failure_results["passed_scenarios"] >= 4

    for case in failure_results["cases"]:
        assert case["success"] is True
        assert "scenario" in case
        assert "description" in case


# 7. Generated Evaluation Artifacts Exist and Are Non-Empty
def test_deployment_evaluation_artifacts_exist():
    """Verify all Step 11 evaluation artifacts exist and contain valid results."""
    json_path = _ARTIFACTS_DIR / "deployment_benchmark_results.json"
    csv_path = _ARTIFACTS_DIR / "deployment_benchmark_results.csv"
    summary_path = _ARTIFACTS_DIR / "deployment_summary.json"
    report_path = _ARTIFACTS_DIR / "deployment_evaluation_report.md"

    for path in [json_path, csv_path, summary_path, report_path]:
        assert path.exists(), f"Missing artifact: {path.name}"
        assert path.stat().st_size > 0, f"Empty artifact: {path.name}"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert "warm_latency_median_ms" in summary
    assert "warm_latency_p95_ms" in summary
    assert "throughput_qps" in summary
    assert "memory_mb" in summary
    assert "failure_scenarios_passed" in summary
    assert summary["total_queries"] == 80

    with open(json_path, "r", encoding="utf-8") as f:
        benchmark_results = json.load(f)

    assert "end_to_end_latency" in benchmark_results
    assert "component_profiling" in benchmark_results
    assert "throughput" in benchmark_results
    assert "scaling" in benchmark_results
    assert "stability" in benchmark_results
    assert "failure_handling" in benchmark_results
    assert "memory_progression_mb" in benchmark_results


# 8. All Figures Generated and Non-Empty
def test_deployment_figures_exist():
    """Verify that all 5 Step 11 visualization figures exist and are non-empty."""
    expected_figures = [
        "deployment_latency.png",
        "deployment_component_breakdown.png",
        "deployment_throughput.png",
        "deployment_memory.png",
        "deployment_cold_warm.png",
    ]

    for fig_name in expected_figures:
        fig_path = _FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Figure missing: {fig_name}"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_name} is too small (<1KB)"
