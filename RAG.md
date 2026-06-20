# RAG — Retrieval-Augmented Generation for the welding-knowledge route

This document describes the RAG layer added to the welding chatbot: what it is,
how it works, and how to run, extend, and evaluate it. It is written to be
defensible in a thesis review.

## 1. Why RAG here

The chatbot has three answer routes. Two are numeric and deterministic
(**parameter optimization** via physics + grid search; **fault diagnosis** via a
RandomForest + SHAP) — these must never use an LLM to invent numbers, so they
stay RAG-free by design.

The third route — **welding knowledge** (defects, troubleshooting, quality,
cost, process comparisons) — previously used brittle keyword/substring matching
against a YAML knowledge base. RAG upgrades this route to **semantic retrieval +
grounded generation with source citations**: the model answers strictly from
retrieved passages and cites them, which keeps answers traceable and
hallucination-resistant.

## 2. Architecture

```
question
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│  CORPUS  (src/rag/corpus.py)                                   │
│   • config/welding_knowledge.yaml  → 1 passage / topic         │
│   • data/knowledge_docs/*.md|*.txt → chunked by section        │
└──────────────────────────────────────────────────────────────┘
   │ build once, cached + persisted (rebuild on corpus change)
   ▼
┌──────────────────────────────────────────────────────────────┐
│  INDEX  (src/rag/index.py)                                     │
│   • local sentence-transformers encoder (all-MiniLM-L6-v2)     │
│   • L2-normalised embedding matrix in models/rag_index/        │
│   • cosine search (dot product on normalised vectors)          │
└──────────────────────────────────────────────────────────────┘
   │ top candidates
   ▼
┌──────────────────────────────────────────────────────────────┐
│  RETRIEVER  (src/rag/retriever.py)  — HYBRID RANKING           │
│   score = 0.7 · semantic(cosine) + 0.3 · lexical(word overlap) │
│   → top-k passages, each tagged [S1], [S2], … for citation     │
└──────────────────────────────────────────────────────────────┘
   │ rag_passages
   ▼
┌──────────────────────────────────────────────────────────────┐
│  GENERATION  (src/chat/prompts.py + synthesize.py)            │
│   LLM answers grounded ONLY in retrieved passages, citing [S#] │
│   (Anthropic / Groq / Ollama — phrasing only)                 │
└──────────────────────────────────────────────────────────────┘
```

Wired into the pipeline at `run_knowledge_pipeline()` in
[src/api/pipeline.py](src/api/pipeline.py); the payload gains `rag_passages` and
`retrieval` fields. The Streamlit UI renders the retrieved sources and their
scores ([app/streamlit_app.py](app/streamlit_app.py) → `_render_rag_sources`).

## 3. Design choices (and why)

| Choice | Rationale |
|---|---|
| **Local embeddings** (sentence-transformers `all-MiniLM-L6-v2`) | Anthropic has no embeddings API; a local encoder is free, fast, and **offline-capable** — matches the project's offline/Windows constraints. No data leaves the machine. |
| **Hybrid retrieval** (semantic + lexical) | Pure semantic search can miss exact terms (defect/parameter names); pure keyword search misses paraphrases. Blending keeps both strengths — a common, defensible RAG-quality technique. |
| **Inline `[S#]` citations** | Traceability: every claim points back to a retrieved passage. Backend-agnostic (works on Anthropic/Groq/Ollama), unlike a provider-specific citation API. |
| **Persisted index + content hash** | The index rebuilds only when the corpus changes; otherwise it loads from `models/rag_index/`. Fast startup, reproducible. |
| **Keyword fallback** | If the embedding model can't load (e.g. first run, no network), the route degrades gracefully to the original keyword grounding instead of failing. |

## 4. How to run

```powershell
# build / rebuild the index and run a test query
.venv\Scripts\python.exe -m src.rag.build_index "porosity in stainless welds"

# the app builds the index automatically on first use
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

First run downloads the embedding model (~80 MB) once; afterwards it is cached.

## 5. Extending the corpus

Drop `.md` / `.txt` files into `data/knowledge_docs/` (handbook excerpts,
standards notes, shop SOPs). They are chunked by heading and ingested on the next
build. See [data/knowledge_docs/README.md](data/knowledge_docs/README.md).

## 6. Evaluating retrieval (for the thesis)

Suggested metrics over a small labelled query→relevant-passage set:

- **Recall@k / Precision@k** — does the relevant passage appear in the top-k?
- **MRR** (mean reciprocal rank) — how high does the first relevant passage rank?
- **Hybrid vs semantic-only vs keyword-only** ablation — vary the
  `_SEMANTIC_WEIGHT` / `_LEXICAL_WEIGHT` blend in
  [src/rag/retriever.py](src/rag/retriever.py) and report the trade-off.
- **Citation faithfulness** — sample answers and check every `[S#]` claim is
  actually supported by the cited passage (mirrors the existing "fabricated
  numbers = 0" check for the numeric routes).

The `retrieve()` return values already expose `semantic_score`,
`lexical_score`, and the blended `score` per passage, so an evaluation harness
can log them directly.

## 7. Tests

[tests/test_rag.py](tests/test_rag.py) covers corpus building and chunking
(no model needed) plus end-to-end retrieval (gated on the embedding model being
available, so CI without network skips rather than fails).
