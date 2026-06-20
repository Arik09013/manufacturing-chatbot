"""
Retrieval-Augmented Generation (RAG) for the welding knowledge route.

A small, self-contained retrieval layer:
  corpus.py    — assemble the retrieval corpus (curated KB + reference docs)
  index.py     — local sentence-embedding vector store + cosine search
  retriever.py — public hybrid (semantic + lexical) retrieval API with citations

The public entry point is ``src.rag.retriever.retrieve``.
"""

from src.rag.retriever import retrieve  # noqa: F401
