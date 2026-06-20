"""
Vector index for RAG: local sentence-embedding store + cosine-similarity search.

Embeddings are produced locally by a sentence-transformers model (default
``all-MiniLM-L6-v2``) — no API key, fully offline once the model is cached.
Anthropic has no embeddings endpoint, so a local encoder is the right choice
here; generation still goes through the existing LLM synthesis layer.

The index (embedding matrix + passages + metadata) persists under
``models/rag_index/`` and is reused across runs. It is rebuilt automatically when
the corpus changes, detected via a content hash stored in ``meta.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
INDEX_DIR = _ROOT / "models" / "rag_index"
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _model_name() -> str:
    return os.getenv("RAG_EMBED_MODEL", _DEFAULT_MODEL)


@lru_cache(maxsize=2)
def _load_model(name: str):
    """Load (and cache) the sentence-transformers encoder. Downloads on first use."""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading RAG embedding model %s", name)
    return SentenceTransformer(name)


def embed(texts, model_name: str | None = None) -> np.ndarray:
    """Encode texts to L2-normalised float32 vectors (so dot product == cosine)."""
    name = model_name or _model_name()
    model = _load_model(name)
    vecs = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32)


def corpus_hash(passages: list[dict]) -> str:
    """Content hash over passage ids + text — changes whenever the corpus changes."""
    h = hashlib.sha256()
    for p in passages:
        h.update(p["id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(p["text"].encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


class VectorIndex:
    """An in-memory embedding matrix + the passages it was built from."""

    def __init__(self, passages: list[dict], embeddings: np.ndarray,
                 model_name: str, chash: str):
        self.passages = passages
        self.embeddings = embeddings
        self.model_name = model_name
        self.corpus_hash = chash

    @classmethod
    def build(cls, passages: list[dict]) -> "VectorIndex":
        name = _model_name()
        if passages:
            embeddings = embed([p["text"] for p in passages], model_name=name)
        else:
            embeddings = np.zeros((0, 384), dtype=np.float32)
        return cls(passages, embeddings, name, corpus_hash(passages))

    def save(self, directory: Path = INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "passages.json").write_text(
            json.dumps(self.passages, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (directory / "meta.json").write_text(
            json.dumps({"model_name": self.model_name, "corpus_hash": self.corpus_hash}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path = INDEX_DIR) -> "VectorIndex":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        passages = json.loads((directory / "passages.json").read_text(encoding="utf-8"))
        embeddings = np.load(directory / "embeddings.npy")
        return cls(passages, embeddings, meta["model_name"], meta["corpus_hash"])

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        """Return the ``top_k`` passages by cosine similarity to ``query``."""
        if not self.passages:
            return []
        q = embed([query], model_name=self.model_name)[0]
        scores = self.embeddings @ q
        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in order:
            i = int(idx)
            p = dict(self.passages[i])
            p["semantic_score"] = float(scores[i])
            results.append(p)
        return results
