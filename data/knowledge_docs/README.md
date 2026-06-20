# Knowledge documents (RAG corpus)

Drop welding reference material here as `.md` or `.txt` files. Everything in this
folder is ingested by the RAG retriever ([src/rag/corpus.py](../../src/rag/corpus.py))
**in addition to** the curated knowledge base (`config/welding_knowledge.yaml`).

## How ingestion works

- Each file is split into passages at markdown headings (`#`, `##`, …).
- Long sections are further split by paragraph so each passage stays focused.
- Each passage is embedded locally (sentence-transformers, no API key) and
  retrieved by semantic + keyword hybrid search.

## Adding a document

1. Save a `.md`/`.txt` file in this folder (handbook excerpts, standards notes,
   shop SOPs, WPS commentary — anything text).
2. Rebuild the index:
   ```
   python -m src.rag.build_index "your test query"
   ```
   (The index also rebuilds automatically the next time the app runs, because the
   corpus content hash changed.)

Keep passages factual and self-contained — the LLM answers are grounded in these
texts and cite them by `[S#]` label, so a clean source means a clean, traceable
answer.
