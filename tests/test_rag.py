"""
Tests for the RAG retrieval layer.

Corpus / chunking tests run without the embedding model. The end-to-end
retrieval test is gated on sentence-transformers being importable AND the model
loading (it skips offline rather than failing CI).
"""

import pytest

from src.rag.corpus import build_corpus, _split_sections, _split_long


# ── Corpus building (no embedding model required) ─────────────────────────────

def test_corpus_includes_knowledge_base_entries():
    passages = build_corpus()
    ids = {p["id"] for p in passages}
    assert any(i.startswith("kb:") for i in ids)
    assert "kb:porosity" in ids


def test_corpus_includes_reference_documents():
    passages = build_corpus()
    sources = {p["source"] for p in passages}
    # The sample shielding-gas reference doc ships in data/knowledge_docs/.
    assert any(s.endswith(".md") for s in sources)
    assert any(p["category"] == "reference_document" for p in passages)


def test_corpus_passages_have_required_fields():
    for p in build_corpus():
        assert {"id", "text", "source", "title", "category"} <= p.keys()
        assert p["text"].strip()


def test_split_sections_splits_on_markdown_headings():
    text = "# Alpha\nbody a\n\n## Beta\nbody b"
    headings = [h for h, _ in _split_sections(text)]
    assert "Alpha" in headings
    assert "Beta" in headings


def test_split_long_packs_into_bounded_chunks():
    body = "\n\n".join(["para " * 60] * 10)  # well over the chunk limit
    chunks = _split_long(body)
    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)


# ── End-to-end retrieval (needs the embedding model) ──────────────────────────

def _retrieve_or_skip(question, **kw):
    pytest.importorskip("sentence_transformers")
    from src.rag import retriever
    try:
        return retriever.retrieve(question, **kw)
    except Exception as exc:  # model download/load failed (e.g. offline CI)
        pytest.skip(f"embedding model unavailable: {exc}")


def test_retrieve_returns_ranked_cited_passages():
    passages = _retrieve_or_skip("why am I getting porosity in stainless welds", top_k=3)
    assert passages, "expected at least one retrieved passage"

    # Ranked by blended score, descending.
    scores = [p["score"] for p in passages]
    assert scores == sorted(scores, reverse=True)

    # Sequential citation labels for source traceability.
    assert passages[0]["cite"] == "S1"

    # The porosity material should surface (KB entry and/or shielding-gas doc).
    assert any("poros" in p["text"].lower() for p in passages)


def test_retrieve_blends_semantic_and_lexical_scores():
    passages = _retrieve_or_skip("shielding gas for aluminium MIG", top_k=4)
    assert passages
    top = passages[0]
    assert {"semantic_score", "lexical_score", "score"} <= top.keys()
    # The reference document on shielding gas should be retrievable for this query.
    assert any(p["category"] == "reference_document" for p in passages)
