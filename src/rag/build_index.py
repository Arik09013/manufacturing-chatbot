"""
CLI to (re)build and inspect the RAG vector index.

    # rebuild the index and print corpus stats
    python -m src.rag.build_index

    # rebuild, then run a test query against it
    python -m src.rag.build_index "porosity in stainless welds"

The first run downloads the sentence-transformers embedding model (~80 MB);
afterwards it is cached and the system works fully offline.
"""

from __future__ import annotations

import sys

from src.rag.corpus import build_corpus
from src.rag.index import INDEX_DIR
from src.rag.retriever import get_index, retrieve


def main(argv: list[str]) -> None:
    passages = build_corpus()
    print(f"Corpus: {len(passages)} passages")
    by_source: dict[str, int] = {}
    for p in passages:
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
    for source, n in sorted(by_source.items()):
        print(f"  {source}: {n}")

    print("\nBuilding index (downloads the embedding model on first run)…")
    index = get_index(rebuild=True)
    print(f"Index built: {len(index.passages)} vectors, model={index.model_name}")
    print(f"Saved to {INDEX_DIR}")

    if len(argv) > 1:
        query = " ".join(argv[1:])
        print(f"\nQuery: {query!r}")
        for p in retrieve(query):
            print(
                f"  [{p['cite']}] score={p['score']:.3f} "
                f"(sem={p['semantic_score']:.3f}, lex={p['lexical_score']:.3f})  "
                f"{p['title']} ({p['source']})"
            )


if __name__ == "__main__":
    main(sys.argv)
