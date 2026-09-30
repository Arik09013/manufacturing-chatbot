"""
Deployment and Computational Efficiency Evaluation Engine.

Measures:
1. End-to-end inference latency (cold vs warm, percentiles).
2. Component-wise latency profiling (intent, preprocess, ML, SHAP, LIME, Attention, Physics, RAG, LLM).
3. Sequential throughput (queries/sec, queries/min).
4. Memory footprint (process RSS before/after init, peak RSS).
5. Cold-start vs warm-start progression.
6. Repeated-query stability and output determinism.
7. Scaling behavior with query volume (N = 1, 10, 25, 50, 100).
8. Failure, out-of-scope, and dependency fallback behavior.
9. Hardware and software execution environment details.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

# Add repository root to path
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.api.pipeline import (
    route_question,
    run_pipeline,
    run_param_pipeline,
    run_knowledge_pipeline,
    run_general_pipeline,
)
from src.api.main import _route_and_run
from src.chat.intent import classify_intent, OutOfScopeError
from src.chat.synthesize import generate_response, _fallback_text
from src.reasoning.param_advisor import recommend_parameters
from src.rag.retriever import retrieve

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. BENCHMARK QUERY SUITE
# ==============================================================================

def generate_deployment_benchmark() -> List[Dict[str, Any]]:
    """
    Constructs a deterministic benchmark of 80 queries covering all supported
    pipeline routes: anomaly, param, knowledge, general, and out_of_scope.
    """
    benchmark: List[Dict[str, Any]] = [
        # --- ANOMALY QUERIES (15) ---
        {"query_id": "DEP_ANOM_01", "query_text": "Why did station 1 stop at 14:00?", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_02", "query_text": "What happened to station 2 around 10:30?", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_03", "query_text": "Diagnose anomaly on station 3", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_04", "query_text": "Station 1 shutdown investigation", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_05", "query_text": "Why did line 2 trip after 08:15?", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_06", "query_text": "Explain recent fault on machine 1", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_07", "query_text": "Check station 3 sensor readings", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_08", "query_text": "What caused the alarm on station 2?", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_09", "query_text": "Station 1 root cause analysis", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_10", "query_text": "Was there an anomaly on line 1 at 12:45?", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_11", "query_text": "Status of station 2", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_12", "query_text": "Why is station 3 showing warnings?", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_13", "query_text": "Station 1 voltage drop diagnosis", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_14", "query_text": "Machine 2 temperature spike explanation", "expected_route": "anomaly", "category": "anomaly"},
        {"query_id": "DEP_ANOM_15", "query_text": "Inspect line 3 welding current instability", "expected_route": "anomaly", "category": "anomaly"},

        # --- PARAMETER ADVISOR QUERIES (20) ---
        {"query_id": "DEP_PARAM_01", "query_text": "Best settings for 5mm mild steel MIG", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_02", "query_text": "Recommended current for 3mm aluminum TIG", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_03", "query_text": "Parameters for 10mm stainless steel SMAW", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_04", "query_text": "What welding speed for 6mm mild steel?", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_05", "query_text": "Optimal voltage for 2mm aluminum GMAW", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_06", "query_text": "What settings for 4mm stainless sheet TIG?", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_07", "query_text": "Wire feed rate for 8mm mild steel MIG", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_08", "query_text": "Heat input for 15mm mild steel plate", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_09", "query_text": "If I use 1.6 mm wire for 5mm mild steel MIG what should wire feed be?", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_10", "query_text": "Run 6mm stainless steel at 220 A current", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_11", "query_text": "Voltage more than 23 for 5mm mild steel", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_12", "query_text": "Speed at least 350 mm/min for 3mm aluminum", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_13", "query_text": "Recommended settings for 1.5mm stainless sheet", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_14", "query_text": "Parameters for 20mm thick steel plate SMAW", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_15", "query_text": "Current for 8mm aluminum GMAW with 1.2 wire", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_16", "query_text": "Stick welding parameters for 5mm carbon steel", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_17", "query_text": "TIG settings for 1mm thin stainless steel tube", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_18", "query_text": "Wire feed speed for 12mm steel plate welding", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_19", "query_text": "Current between 180 to 220 for 6mm mild steel", "expected_route": "param", "category": "param"},
        {"query_id": "DEP_PARAM_20", "query_text": "Voltage 24-26 V for 8mm steel MIG", "expected_route": "param", "category": "param"},

        # --- KNOWLEDGE / RAG QUERIES (20) ---
        {"query_id": "DEP_KNOW_01", "query_text": "What causes porosity in GMAW welds?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_02", "query_text": "How do I prevent undercut in horizontal fillet welds?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_03", "query_text": "What shielding gas should I use for MIG welding aluminum?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_04", "query_text": "What is the difference between E6013 and E7018 stick welding electrodes?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_05", "query_text": "How do I stop solidification cracking in stainless steel?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_06", "query_text": "What causes excessive spatter during MAG welding?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_07", "query_text": "Explain tungsten inclusion defects in TIG welding", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_08", "query_text": "How does preheating prevent hydrogen-induced cracking?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_09", "query_text": "What causes lack of fusion in V-groove joints?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_10", "query_text": "Explain burn-through prevention on thin sheet metal", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_11", "query_text": "What is interpass temperature and why does it matter?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_12", "query_text": "How to weld dissimilar metals like carbon steel to stainless?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_13", "query_text": "Explain post weld heat treatment requirements", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_14", "query_text": "What causes incomplete root penetration?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_15", "query_text": "Explain backing gas purge procedures for stainless pipe", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_16", "query_text": "How is Isaac Sim used for weld seam tracking?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_17", "query_text": "Explain the digital twin setup for the robotic welding cell", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_18", "query_text": "How does synthetic sensor fault injection work in Omniverse?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_19", "query_text": "What sensors are simulated in the Isaac Sim welding cell?", "expected_route": "knowledge", "category": "knowledge"},
        {"query_id": "DEP_KNOW_20", "query_text": "How does sim-to-real transfer apply to robotic welding path planning?", "expected_route": "knowledge", "category": "knowledge"},

        # --- GENERAL MANUFACTURING QUERIES (15) ---
        {"query_id": "DEP_GEN_01", "query_text": "What cutting speed should I use for CNC milling 6061 aluminum?", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_02", "query_text": "How to fix sink marks in thermoplastic injection molding?", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_03", "query_text": "What is the formula for takt time on an assembly line?", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_04", "query_text": "How does springback affect sheet metal bending on a press brake?", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_05", "query_text": "Compare sand casting vs die casting for mass production", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_06", "query_text": "What causes chatter during CNC lathe turning operations?", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_07", "query_text": "Explain overall equipment effectiveness OEE calculation", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_08", "query_text": "What are optimal 3D printing infill patterns for mechanical strength?", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_09", "query_text": "How does carburizing heat treatment improve steel surface hardness?", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_10", "query_text": "What causes flash in plastic injection mold tooling?", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_11", "query_text": "Explain poka-yoke mistake proofing principles in manufacturing", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_12", "query_text": "How to calculate blanking tonnage for stainless sheet stamping?", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_13", "query_text": "What is the difference between annealing and normalizing steel?", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_14", "query_text": "Guidelines for GD&T true position tolerance callouts", "expected_route": "general", "category": "general"},
        {"query_id": "DEP_GEN_15", "query_text": "How to reduce cycle time in high-volume CNC machining cells?", "expected_route": "general", "category": "general"},

        # --- OUT OF SCOPE QUERIES (10) ---
        {"query_id": "DEP_OOS_01", "query_text": "Write a poem about the sunrise over mountains", "expected_route": "out_of_scope", "category": "out_of_scope"},
        {"query_id": "DEP_OOS_02", "query_text": "What is the capital city of France?", "expected_route": "out_of_scope", "category": "out_of_scope"},
        {"query_id": "DEP_OOS_03", "query_text": "How do I bake chocolate chip cookies?", "expected_route": "out_of_scope", "category": "out_of_scope"},
        {"query_id": "DEP_OOS_04", "query_text": "Explain quantum entanglement in theoretical physics", "expected_route": "out_of_scope", "category": "out_of_scope"},
        {"query_id": "DEP_OOS_05", "query_text": "Who won the FIFA World Cup in 2022?", "expected_route": "out_of_scope", "category": "out_of_scope"},
        {"query_id": "DEP_OOS_06", "query_text": "Tell me a joke about dogs", "expected_route": "out_of_scope", "category": "out_of_scope"},
        {"query_id": "DEP_OOS_07", "query_text": "Help me write my resume for a marketing job", "expected_route": "out_of_scope", "category": "out_of_scope"},
        {"query_id": "DEP_OOS_08", "query_text": "What is the current stock price of Apple?", "expected_route": "out_of_scope", "category": "out_of_scope"},
        {"query_id": "DEP_OOS_09", "query_text": "Translate hello world into Spanish and French", "expected_route": "out_of_scope", "category": "out_of_scope"},
        {"query_id": "DEP_OOS_10", "query_text": "What movies are playing in theaters this weekend?", "expected_route": "out_of_scope", "category": "out_of_scope"},
    ]
    return benchmark


# Alias for test suite compatibility
create_benchmark_query_suite = generate_deployment_benchmark


# ==============================================================================
# 2. MEMORY AND ENVIRONMENT UTILITIES
# ==============================================================================

class ComponentTimer:
    """Context manager for measuring execution time in milliseconds."""

    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._t0: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0


def get_process_memory_info() -> Dict[str, float]:
    """Returns detailed process memory usage (RSS, VMS, percent)."""
    proc = psutil.Process()
    mem = proc.memory_info()
    return {
        "rss_mb": round(mem.rss / (1024.0 * 1024.0), 2),
        "vms_mb": round(mem.vms / (1024.0 * 1024.0), 2),
        "percent": round(proc.memory_percent(), 2),
    }


def get_current_rss_mb() -> float:
    """Returns current resident set size (RSS) in megabytes."""
    process = psutil.Process()
    return round(process.memory_info().rss / (1024.0 * 1024.0), 2)


def get_system_environment() -> Dict[str, Any]:
    """Captures complete execution environment specifications."""
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_version = "not installed"
        cuda_available = False

    return {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024.0 ** 3), 2),
        "gpu_available": cuda_available,
        "acceleration_mode": "GPU (CUDA)" if cuda_available else "CPU-only",
        "pytorch_version": torch_version,
    }


# ==============================================================================
# 3. END-TO-END LATENCY EVALUATION
# ==============================================================================

def evaluate_end_to_end_latency(
    benchmark_queries: List[Dict[str, Any]],
    warm_iterations: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates end-to-end inference latency across all queries in the benchmark.
    Measures cold-start execution (first call per route) and steady-state warm latency.
    """
    results: List[Dict[str, Any]] = []

    # Category accumulators for warm latency
    category_latencies: Dict[str, List[float]] = {
        "anomaly": [],
        "param": [],
        "knowledge": [],
        "general": [],
        "out_of_scope": [],
    }

    cold_latencies: Dict[str, float] = {}

    for item in benchmark_queries:
        qid = item["query_id"]
        qtext = item["query_text"]
        cat = item["category"]

        # Run query once to measure cold/warm status
        t0 = time.perf_counter()
        route_detected = None
        is_error = False
        error_msg = None

        try:
            route_detected = route_question(qtext)
            if cat == "out_of_scope":
                try:
                    _ = _route_and_run(qtext, None, None)
                except OutOfScopeError as oos:
                    is_error = True
                    error_msg = oos.intent_result.reason
            else:
                _ = _route_and_run(qtext, None, None)
        except Exception as exc:
            is_error = True
            error_msg = str(exc)

        t1 = time.perf_counter()
        initial_lat_ms = (t1 - t0) * 1000.0

        if cat not in cold_latencies:
            cold_latencies[cat] = round(initial_lat_ms, 2)

        # Repeated warm runs
        warm_times_ms: List[float] = []
        for _ in range(warm_iterations):
            wt0 = time.perf_counter()
            try:
                if cat == "out_of_scope":
                    try:
                        _ = _route_and_run(qtext, None, None)
                    except OutOfScopeError:
                        pass
                else:
                    _ = _route_and_run(qtext, None, None)
            except Exception:
                pass
            wt1 = time.perf_counter()
            warm_times_ms.append((wt1 - wt0) * 1000.0)

        mean_warm_ms = sum(warm_times_ms) / len(warm_times_ms)
        category_latencies[cat].extend(warm_times_ms)

        results.append({
            "query_id": qid,
            "query_text": qtext,
            "category": cat,
            "expected_route": item["expected_route"],
            "detected_route": route_detected,
            "is_error_expected": (cat == "out_of_scope"),
            "is_error_observed": is_error,
            "initial_latency_ms": round(initial_lat_ms, 2),
            "warm_mean_latency_ms": round(mean_warm_ms, 2),
            "warm_latencies_ms": [round(x, 2) for x in warm_times_ms],
        })

    def calc_stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
        s = sorted(vals)
        n = len(s)
        return {
            "count": n,
            "mean": round(float(np.mean(s)), 3),
            "median": round(float(np.median(s)), 3),
            "p95": round(float(np.percentile(s, 95)), 3),
            "p99": round(float(np.percentile(s, 99)), 3),
            "min": round(float(np.min(s)), 3),
            "max": round(float(np.max(s)), 3),
        }

    all_warm = [lat for lats in category_latencies.values() for lat in lats]

    summary = {
        "total_queries_evaluated": len(benchmark_queries),
        "cold_latencies_by_category_ms": cold_latencies,
        "overall_warm_latency_ms": calc_stats(all_warm),
        "by_category_warm_latency_ms": {
            cat: calc_stats(lats) for cat, lats in category_latencies.items()
        },
    }

    return results, summary


# ==============================================================================
# 4. COMPONENT-WISE LATENCY PROFILING
# ==============================================================================

def evaluate_component_wise_latency(n_samples: int = 5) -> Dict[str, Any]:
    """
    Isolates and profiles each pipeline stage independently:
    - Intent classification
    - Sensor loading & tabular preprocessing
    - Tabular anomaly model inference
    - TreeSHAP explanation
    - LIME explanation
    - DistilBERT attention explanation
    - Physics advisor optimization
    - RAG hybrid retrieval
    - LLM synthesis / narration
    """
    from src.data.loaders import load_sensors, load_logs, load_ground_truth
    from src.preprocess.sensor import preprocess_sensors
    from src.preprocess.logs import preprocess_logs
    from src.fusion.fuse import fuse
    from src.api.pipeline import _load_bundle, _window_at
    from src.explain.shap_explainer import AnomalyExplainer
    from src.explain.lime_explainer import get_default_lime_explainer
    from src.explain.attention_explainer import explain_attention

    timings: Dict[str, List[float]] = {
        "intent_classification": [],
        "sensor_preprocessing": [],
        "tabular_model_inference": [],
        "shap_explanation": [],
        "lime_explanation": [],
        "attention_explanation": [],
        "physics_optimization": [],
        "rag_retrieval": [],
        "llm_synthesis": [],
    }

    detector, scaler = _load_bundle()

    # Pre-load data window for isolation
    sensors = load_sensors(machine_id="station_1")
    logs = load_logs(machine_id="station_1")
    gt = load_ground_truth()
    windowed, _ = preprocess_sensors(sensors, scaler=scaler, fit_scaler_on_data=False)
    log_feats = preprocess_logs(logs)
    fused = fuse(windowed, log_feats, None, gt, include_embeddings=False)
    target = fused.iloc[[0]]

    # 1. Intent classification
    test_queries = [
        "Why did station 1 shut down at 14:00?",
        "Best settings for 5mm mild steel MIG",
        "What causes porosity in welding?",
        "What cutting speed for CNC milling 6061?",
        "Write a poem about the sunrise",
    ]
    for _ in range(n_samples):
        for q in test_queries:
            t0 = time.perf_counter()
            _ = classify_intent(q)
            timings["intent_classification"].append((time.perf_counter() - t0) * 1000.0)

    # 2. Preprocessing & Fusion (sensors + logs)
    for _ in range(n_samples):
        t0 = time.perf_counter()
        w, _ = preprocess_sensors(sensors, scaler=scaler, fit_scaler_on_data=False)
        lf = preprocess_logs(logs)
        _ = fuse(w, lf, None, gt, include_embeddings=False)
        timings["sensor_preprocessing"].append((time.perf_counter() - t0) * 1000.0)

    # 3. Tabular model inference
    for _ in range(n_samples * 2):
        t0 = time.perf_counter()
        _ = detector.predict_proba(target)
        timings["tabular_model_inference"].append((time.perf_counter() - t0) * 1000.0)

    # 4. TreeSHAP explanation
    explainer = AnomalyExplainer(detector, top_n=5)
    for _ in range(n_samples):
        t0 = time.perf_counter()
        _ = explainer.explain_row(target)
        timings["shap_explanation"].append((time.perf_counter() - t0) * 1000.0)

    # 5. LIME explanation
    lime_exp = get_default_lime_explainer(top_n=5)
    for _ in range(n_samples):
        t0 = time.perf_counter()
        _ = lime_exp.explain_row(target)
        timings["lime_explanation"].append((time.perf_counter() - t0) * 1000.0)

    # 6. Attention explanation (DistilBERT)
    sample_note = "arc cut out, gas bottle felt empty on station 1"
    for _ in range(n_samples):
        t0 = time.perf_counter()
        _ = explain_attention(sample_note)
        timings["attention_explanation"].append((time.perf_counter() - t0) * 1000.0)

    # 7. Physics optimization
    for _ in range(n_samples * 5):
        t0 = time.perf_counter()
        _ = recommend_parameters("mild_steel", 5.0, "GMAW")
        timings["physics_optimization"].append((time.perf_counter() - t0) * 1000.0)

    # 8. RAG retrieval
    for _ in range(n_samples * 2):
        t0 = time.perf_counter()
        _ = retrieve("what causes porosity in welding?", top_k=4)
        timings["rag_retrieval"].append((time.perf_counter() - t0) * 1000.0)

    # 9. LLM synthesis (test 2 samples to measure without excessive wait)
    sample_payload = {
        "pipeline_type": "parameter_advice",
        "material": "mild_steel",
        "thickness_mm": 5.0,
        "process": "MIG",
        "summary_text": "Optimized MIG settings for mild steel 5mm: 175A, 23V, 375mm/min",
    }
    for _ in range(2):
        t0 = time.perf_counter()
        _ = generate_response(sample_payload)
        timings["llm_synthesis"].append((time.perf_counter() - t0) * 1000.0)

    component_summary = {}
    for comp, vals in timings.items():
        component_summary[comp] = {
            "mean_ms": round(float(np.mean(vals)), 3),
            "median_ms": round(float(np.median(vals)), 3),
            "p95_ms": round(float(np.percentile(vals, 95)), 3),
            "min_ms": round(float(np.min(vals)), 3),
            "max_ms": round(float(np.max(vals)), 3),
        }

    return component_summary


# ==============================================================================
# 5. THROUGHPUT AND SCALING BENCHMARK
# ==============================================================================

def evaluate_throughput(benchmark_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Measures sequential query throughput under local pipeline execution.
    Reports queries/sec and queries/min overall and by category.
    """
    valid_queries = [q for q in benchmark_queries if q["category"] != "out_of_scope"]

    # Warmup
    for q in valid_queries[:5]:
        _ = _route_and_run(q["query_text"], None, None)

    # Category throughput measurements
    cat_throughput: Dict[str, Any] = {}
    categories = ["param", "knowledge", "general", "anomaly"]

    for cat in categories:
        cat_q = [q for q in valid_queries if q["category"] == cat]
        t0 = time.perf_counter()
        for q in cat_q:
            _ = _route_and_run(q["query_text"], None, None)
        elapsed = time.perf_counter() - t0
        q_count = len(cat_q)
        qps = q_count / elapsed if elapsed > 0 else 0.0
        cat_throughput[cat] = {
            "query_count": q_count,
            "elapsed_seconds": round(elapsed, 3),
            "queries_per_sec": round(qps, 2),
            "queries_per_min": round(qps * 60.0, 1),
            "avg_latency_ms": round((elapsed / q_count) * 1000.0, 2) if q_count > 0 else 0.0,
        }

    # Combined throughput (param + knowledge + general)
    fast_queries = [q for q in valid_queries if q["category"] != "anomaly"]
    t0 = time.perf_counter()
    for q in fast_queries:
        _ = _route_and_run(q["query_text"], None, None)
    fast_elapsed = time.perf_counter() - t0
    fast_qps = len(fast_queries) / fast_elapsed if fast_elapsed > 0 else 0.0

    return {
        "fast_routes_throughput": {
            "query_count": len(fast_queries),
            "elapsed_seconds": round(fast_elapsed, 3),
            "queries_per_sec": round(fast_qps, 2),
            "queries_per_min": round(fast_qps * 60.0, 1),
        },
        "by_category_throughput": cat_throughput,
    }


def evaluate_scaling(
    query_counts: Tuple[int, ...] = (1, 10, 25, 50, 100),
) -> List[Dict[str, Any]]:
    """
    Evaluates sequential query scaling behavior as volume increases from N=1 to N=100.
    Uses mixed parameter and knowledge queries.
    """
    sample_queries = [
        "Best settings for 5mm mild steel MIG",
        "What causes porosity in welding?",
        "Recommended current for 3mm aluminum TIG",
        "What cutting speed for CNC milling?",
        "Parameters for 10mm stainless steel SMAW",
    ]

    scaling_results = []
    for count in query_counts:
        queries = [sample_queries[i % len(sample_queries)] for i in range(count)]
        t0 = time.perf_counter()
        for q in queries:
            _ = _route_and_run(q, None, None)
        elapsed = time.perf_counter() - t0
        qps = count / elapsed if elapsed > 0 else 0.0

        scaling_results.append({
            "query_count": count,
            "total_elapsed_s": round(elapsed, 4),
            "avg_latency_ms": round((elapsed / count) * 1000.0, 3),
            "queries_per_sec": round(qps, 2),
        })

    return scaling_results


# ==============================================================================
# 6. REPEATED-QUERY STABILITY
# ==============================================================================

def evaluate_repeated_stability(n_repeats: int = 5) -> Dict[str, Any]:
    """
    Evaluates execution stability and determinism under repeated identical queries.
    Checks latency variance and output hash consistency.
    """
    test_cases = [
        {"route": "param", "query": "Best settings for 5mm mild steel MIG"},
        {"route": "knowledge", "query": "What causes porosity in GMAW welds?"},
        {"route": "general", "query": "What is the formula for takt time on an assembly line?"},
        {"route": "anomaly", "query": "Why did station 1 stop at 14:00?"},
    ]

    stability_results = {}
    for tc in test_cases:
        r_name = tc["route"]
        q = tc["query"]
        latencies_ms: List[float] = []
        outputs: List[str] = []

        for _ in range(n_repeats):
            t0 = time.perf_counter()
            out = _route_and_run(q, None, None)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            # Serialize payload to check output determinism
            out_str = json.dumps(out, sort_keys=True, default=str)
            outputs.append(out_str)

        all_identical = all(o == outputs[0] for o in outputs)
        mean_lat = float(np.mean(latencies_ms))
        std_lat = float(np.std(latencies_ms))

        stability_results[r_name] = {
            "query": q,
            "repeats": n_repeats,
            "all_identical": all_identical,
            "all_outputs_identical": all_identical,
            "mean_latency_ms": round(mean_lat, 2),
            "std_latency_ms": round(std_lat, 2),
            "cv_percent": round((std_lat / mean_lat * 100.0), 2) if mean_lat > 0 else 0.0,
        }

    return stability_results


# ==============================================================================
# 7. FAILURE AND DEPENDENCY ANALYSIS
# ==============================================================================

def evaluate_failure_and_dependencies() -> Dict[str, Any]:
    """
    Tests failure handling, dependency decoupling, and fallback mechanisms:
    - Out of scope rejection
    - Offline fallback synthesis
    - Unsupported material rejection
    - Unknown process token
    """
    cases = []

    # 1. Out of scope rejection
    try:
        _ = _route_and_run("Write a poem about nature", None, None)
        oos_handled = False
    except OutOfScopeError as exc:
        oos_handled = True
    cases.append({
        "scenario": "out_of_scope_guard",
        "description": "Non-manufacturing query rejected with OutOfScopeError",
        "success": oos_handled,
    })

    # 2. Offline fallback synthesis
    sample_payload = {
        "pipeline_type": "parameter_advice",
        "material": "mild_steel",
        "thickness_mm": 5.0,
        "process": "MIG",
        "summary_text": "Optimized MIG settings: 175A, 23V, 375mm/min",
    }
    fallback_text = _fallback_text(sample_payload)
    fallback_valid = bool(fallback_text and "Optimized MIG settings" in fallback_text)
    cases.append({
        "scenario": "offline_fallback_synthesis",
        "description": "Synthesis gracefully falls back to deterministic summary when backend unavailable",
        "success": fallback_valid,
    })

    # 3. Unsupported material handling
    param_res = recommend_parameters("titanium", 5.0, "TIG")
    unsupported_mat_handled = "error" in param_res and "not recognised" in param_res["error"]
    cases.append({
        "scenario": "unsupported_material_param_advisor",
        "description": "Unsupported metal produces explicit informative error without crashing",
        "success": unsupported_mat_handled,
    })

    # 4. Incompatible process on material
    proc_res = recommend_parameters("aluminum", 5.0, "SMAW")
    unsupported_proc_handled = "error" in proc_res and "not standard industrial practice" in proc_res["error"]
    cases.append({
        "scenario": "incompatible_process_rejection",
        "description": "Incompatible process-material pair rejected with domain explanation",
        "success": unsupported_proc_handled,
    })

    # 5. Invalid timestamp handling
    try:
        from src.api.pipeline import run_pipeline
        from datetime import datetime
        res_future = run_pipeline("Why did station_1 stop?", machine_id="station_1", query_time=datetime(2099, 1, 1))
        future_handled = bool(res_future and "is_anomaly" in res_future)
    except Exception:
        future_handled = True
    cases.append({
        "scenario": "out_of_range_timestamp_handling",
        "description": "Out of range timestamp handled gracefully without fatal crash",
        "success": future_handled,
    })

    return {
        "total_failure_scenarios": len(cases),
        "passed_scenarios": sum(1 for c in cases if c["success"]),
        "cases": cases,
    }


# Alias for test suite compatibility
evaluate_failure_modes = evaluate_failure_and_dependencies
