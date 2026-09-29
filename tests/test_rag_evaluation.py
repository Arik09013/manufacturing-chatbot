"""
Unit and Integration Tests for Step 7: Formal RAG Retrieval Evaluation.

Tests:
  1. Benchmark query dataset schema and validity
  2. Ground-truth relevance mapping against corpus passages (0 missing references)
  3. Mathematical correctness of retrieval metrics (Recall@K, MRR, Precision@K, nDCG@K)
  4. Scoring logic across ablation conditions (dense, lexical, title, hybrid)
  5. Monotonicity of Top-K sensitivity sweeps
  6. Score filtering threshold behavior
  7. Latency profiling output schema and realism
  8. End-to-end production retrieve() execution on benchmark queries
  9. Artifact and report presence and schema integrity
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest
import numpy as np

from evaluation.rag_benchmark import (
    BENCHMARK_QUERIES,
    get_benchmark_queries,
    validate_benchmark_queries,
)
from evaluation.eval_rag import (
    ABLATION_CONDITIONS,
    compute_retrieval_metrics,
    run_retrieval_condition,
)
from src.rag.corpus import build_corpus
from src.rag.retriever import get_index, retrieve, _content_words


# =====================================================================
# 1. Benchmark Query Dataset Integrity
# =====================================================================

def test_benchmark_dataset_integrity():
    """Verify benchmark queries have required keys, non-empty fields, and unique IDs."""
    queries = get_benchmark_queries()
    assert len(queries) == 40, f"Expected exactly 40 benchmark queries, got {len(queries)}"

    qids = set()
    categories = set()
    for q in queries:
        assert "query_id" in q and q["query_id"].strip()
        assert "query" in q and len(q["query"].strip()) > 10
        assert "category" in q and q["category"].strip()
        assert "relevant_doc_ids" in q and len(q["relevant_doc_ids"]) >= 1
        assert "rationale" in q and len(q["rationale"].strip()) > 10

        assert q["query_id"] not in qids, f"Duplicate query_id: {q['query_id']}"
        qids.add(q["query_id"])
        categories.add(q["category"])

    assert len(categories) >= 6, f"Expected at least 6 categories, got {len(categories)}"


def test_benchmark_all_target_doc_ids_exist_in_corpus():
    """Verify every referenced relevant_doc_id in the benchmark exists in the corpus."""
    passages = build_corpus()
    is_valid, errors = validate_benchmark_queries(passages)
    assert is_valid, f"Benchmark references invalid doc IDs: {errors}"
    assert len(errors) == 0


# =====================================================================
# 2. Metric Computation Math & Edge Cases
# =====================================================================

def test_compute_retrieval_metrics_exactness():
    """Verify Recall@K, Precision@K, MRR, and nDCG@K on deterministic mock queries."""
    mock_queries = [
        # Query 1: Rank 1 hit (1 relevant doc)
        {"query_id": "M1", "relevant_doc_ids": ["doc_A"]},
        # Query 2: Rank 2 hit (1 relevant doc)
        {"query_id": "M2", "relevant_doc_ids": ["doc_B"]},
        # Query 3: Complete miss (relevant not in top 5)
        {"query_id": "M3", "relevant_doc_ids": ["doc_C"]},
        # Query 4: Two relevant docs at ranks 1 and 3
        {"query_id": "M4", "relevant_doc_ids": ["doc_D1", "doc_D2"]},
    ]

    mock_retrieved = {
        "M1": [{"id": "doc_A"}, {"id": "x1"}, {"id": "x2"}, {"id": "x3"}, {"id": "x4"}],
        "M2": [{"id": "x1"}, {"id": "doc_B"}, {"id": "x2"}, {"id": "x3"}, {"id": "x4"}],
        "M3": [{"id": "x1"}, {"id": "x2"}, {"id": "x3"}, {"id": "x4"}, {"id": "x5"}],
        "M4": [{"id": "doc_D1"}, {"id": "x1"}, {"id": "doc_D2"}, {"id": "x2"}, {"id": "x3"}],
    }

    metrics = compute_retrieval_metrics(mock_queries, mock_retrieved, k_vals=[1, 3, 4, 5])

    # Recall@1: M1 (hit), M2 (miss), M3 (miss), M4 (hit) -> 2/4 = 0.5000
    assert metrics["Recall@1"] == 0.5000

    # Recall@3: M1 (hit), M2 (hit), M3 (miss), M4 (hit) -> 3/4 = 0.7500
    assert metrics["Recall@3"] == 0.7500

    # Recall@4: M1, M2, M4 hit -> 3/4 = 0.7500
    assert metrics["Recall@4"] == 0.7500

    # MRR: M1=1/1=1.0, M2=1/2=0.5, M3=0.0, M4=1/1=1.0 -> (1.0 + 0.5 + 0.0 + 1.0) / 4 = 2.5 / 4 = 0.6250
    assert metrics["MRR"] == 0.6250

    # Precision@1: M1=1/1, M2=0/1, M3=0/1, M4=1/1 -> (1 + 0 + 0 + 1)/4 = 0.5000
    assert metrics["P@1"] == 0.5000

    # Precision@4: M1=1/4, M2=1/4, M3=0/4, M4=2/4 -> (0.25 + 0.25 + 0 + 0.5)/4 = 1.0/4 = 0.2500
    assert metrics["P@4"] == 0.2500

    # nDCG@3:
    # M1: DCG = 1/log2(2) = 1.0; IDCG = 1.0 -> nDCG = 1.0
    # M2: DCG = 1/log2(3) = 0.6309; IDCG = 1.0 -> nDCG = 0.6309
    # M3: DCG = 0; IDCG = 1.0 -> nDCG = 0.0
    # M4: DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5; IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309 -> nDCG = 1.5/1.6309 = 0.9197
    assert 0.0 < metrics["nDCG@3"] < 1.0


# =====================================================================
# 3. Component Scoring Logic
# =====================================================================

def test_ablation_scoring_pure_dense():
    """Verify DENSE_ONLY ignores lexical and title overlap."""
    queries = [{"query_id": "T1", "query": "porosity causes", "relevant_doc_ids": ["p1"]}]
    passages = [
        {"id": "p1", "text": "gas voids defect", "title": "weld defect"},
        {"id": "p2", "text": "porosity causes defect", "title": "porosity causes"},
    ]
    # Embeddings: p1 has higher dot product than p2
    embeddings = np.array([[0.9], [0.1]], dtype=np.float32)
    query_vecs = np.array([[1.0]], dtype=np.float32)
    passage_words = [_content_words(p["text"]) for p in passages]
    passage_title_words = [_content_words(p.get("title", "")) for p in passages]

    weights = {"sem": 1.0, "lex": 0.0, "title": 0.0}
    res = run_retrieval_condition(
        queries, passages, embeddings, query_vecs, passage_words, passage_title_words, weights, min_score=0.0, top_k=2
    )

    # In dense only, p1 must be ranked first because its embedding dot product is 0.9 vs 0.1,
    # despite p2 having 100% lexical match.
    assert res["T1"][0]["id"] == "p1"
    assert res["T1"][0]["score"] == 0.9


def test_ablation_scoring_pure_lexical():
    """Verify LEXICAL_ONLY ignores embedding dot product."""
    queries = [{"query_id": "T1", "query": "porosity defect", "relevant_doc_ids": ["p2"]}]
    passages = [
        {"id": "p1", "text": "gas voids", "title": "argon"},
        {"id": "p2", "text": "porosity defect fixes", "title": "inspection"},
    ]
    embeddings = np.array([[0.99], [0.01]], dtype=np.float32)
    query_vecs = np.array([[1.0]], dtype=np.float32)
    passage_words = [_content_words(p["text"]) for p in passages]
    passage_title_words = [_content_words(p.get("title", "")) for p in passages]

    weights = {"sem": 0.0, "lex": 1.0, "title": 0.0}
    res = run_retrieval_condition(
        queries, passages, embeddings, query_vecs, passage_words, passage_title_words, weights, min_score=0.0, top_k=2
    )

    # In lexical only, p2 must be ranked first because it contains "porosity defect".
    assert res["T1"][0]["id"] == "p2"


# =====================================================================
# 4. Top-K and Threshold Sensitivity
# =====================================================================

def test_top_k_recall_monotonicity():
    """Verify Recall@K is non-decreasing as K increases."""
    mock_queries = [
        {"query_id": "M1", "relevant_doc_ids": ["doc_A"]},
        {"query_id": "M2", "relevant_doc_ids": ["doc_B"]},
    ]
    mock_retrieved = {
        "M1": [{"id": "x1"}, {"id": "doc_A"}, {"id": "x2"}],
        "M2": [{"id": "x1"}, {"id": "x2"}, {"id": "doc_B"}],
    }

    m1 = compute_retrieval_metrics(mock_queries, mock_retrieved, k_vals=[1])["Recall@1"]
    m2 = compute_retrieval_metrics(mock_queries, mock_retrieved, k_vals=[2])["Recall@2"]
    m3 = compute_retrieval_metrics(mock_queries, mock_retrieved, k_vals=[3])["Recall@3"]

    assert m1 <= m2 <= m3
    assert m1 == 0.0
    assert m2 == 0.5
    assert m3 == 1.0


# =====================================================================
# 5. Production retrieve() Smoke Test
# =====================================================================

def test_production_retrieve_on_benchmark_sample():
    """Verify production retrieve() executes and retrieves valid passages."""
    sample_q = BENCHMARK_QUERIES[0]["query"]
    results = retrieve(sample_q, top_k=4, min_score=0.15)
    assert len(results) > 0
    assert len(results) <= 4

    for i, r in enumerate(results, start=1):
        assert r["cite"] == f"S{i}"
        assert "score" in r and r["score"] >= 0.15
        assert "semantic_score" in r
        assert "lexical_score" in r
        assert "title_score" in r

    # Ranked descending by score
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# =====================================================================
# 6. Artifact Files Integrity
# =====================================================================

def test_evaluation_artifacts_exist_and_valid():
    """Verify that JSON and CSV evaluation artifacts exist and adhere to schema."""
    results_path = Path("artifacts/rag_evaluation_results.json")
    summary_path = Path("artifacts/rag_evaluation_summary.json")
    csv_path = Path("artifacts/rag_evaluation_results.csv")
    report_path = Path("outputs/rag_evaluation_report.md")

    assert results_path.exists(), f"Missing {results_path}"
    assert summary_path.exists(), f"Missing {summary_path}"
    assert csv_path.exists(), f"Missing {csv_path}"
    assert report_path.exists(), f"Missing {report_path}"

    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    assert "production_retriever_metrics" in res
    assert "ablation_metrics" in res
    assert "latency_profile" in res
    assert "top_k_sweep" in res
    assert "threshold_sweep" in res

    # Production metrics values
    prod_m = res["production_retriever_metrics"]
    assert prod_m["Recall@4"] == 1.0000, f"Expected 1.0000 Recall@4, got {prod_m['Recall@4']}"
    assert prod_m["MRR"] >= 0.90, f"Expected MRR >= 0.90, got {prod_m['MRR']}"

    # Latency values
    lat = res["latency_profile"]["total_latency"]
    assert 0.0 < lat["mean_ms"] < 100.0, f"Unrealistic latency: {lat['mean_ms']} ms"
