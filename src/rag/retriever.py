"""
Public RAG retrieval API used by the welding-knowledge route.

``retrieve(question)`` returns the passages most relevant to the question, ranked
by a HYBRID score that blends:

  * dense **semantic** similarity (sentence-embedding cosine),
  * sparse **lexical** overlap on the passage body (shared content words), and
  * a **title/field boost** when query terms match the passage title (so an
    on-topic entry like "Porosity" out-ranks a merely-related one for a
    porosity question — standard field-weighted retrieval).

The hybrid blend keeps exact-term matches (defect names, parameter names)
competitive while still surfacing semantically-related passages that share no
keywords. Each returned passage carries a ``score`` and a citation label
(``cite`` = "S1", "S2", …) so the generated answer can be traced back to its
source — important for a hallucination-resistant, defensible system.

The index is built lazily on first use and cached for the process; it is rebuilt
automatically when the corpus on disk no longer matches the live corpus.
"""

from __future__ import annotations

import logging
import re

from src.rag.corpus import build_corpus
from src.rag.index import INDEX_DIR, VectorIndex, corpus_hash

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "and", "for", "with", "that", "this", "are", "was", "what", "why",
    "how", "can", "should", "would", "into", "from", "your", "you", "but",
    "not", "get", "getting", "have", "has", "does", "did", "when", "where",
    "which", "out", "use", "using", "any", "some", "all", "due", "off", "too",
}

_SEMANTIC_WEIGHT = 0.6
_LEXICAL_WEIGHT = 0.25
_TITLE_WEIGHT = 0.15

# Process-cached index.
_INDEX: VectorIndex | None = None


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def get_index(rebuild: bool = False) -> VectorIndex:
    """Return the process-cached vector index, (re)building/loading as needed."""
    global _INDEX
    if _INDEX is not None and not rebuild:
        return _INDEX

    passages = build_corpus()
    chash = corpus_hash(passages)

    if not rebuild and (INDEX_DIR / "meta.json").exists():
        try:
            disk = VectorIndex.load(INDEX_DIR)
            if disk.corpus_hash == chash:
                _INDEX = disk
                return _INDEX
            logger.info("RAG corpus changed since last build; rebuilding index")
        except Exception:
            logger.warning("Failed to load RAG index from disk; rebuilding", exc_info=True)

    _INDEX = VectorIndex.build(passages)
    try:
        _INDEX.save(INDEX_DIR)
    except Exception:
        logger.warning("Could not persist RAG index", exc_info=True)
    return _INDEX


def retrieve(question: str, top_k: int = 4, min_score: float = 0.15) -> list[dict]:
    """Return up to ``top_k`` passages relevant to ``question``, hybrid-ranked.

    Each passage dict carries ``semantic_score``, ``lexical_score``, the blended
    ``score``, and a ``cite`` label ("S1", "S2", …). Passages below
    ``min_score`` are dropped. Returns ``[]`` if nothing is relevant.
    """
    index = get_index()
    candidates = index.search(question, top_k=max(top_k * 3, 10))
    if not candidates:
        return []

    q_words = _content_words(question)
    n = len(q_words) or 1
    for p in candidates:
        body_overlap = len(q_words & _content_words(p["text"])) / n
        title_overlap = len(q_words & _content_words(p.get("title", ""))) / n
        # Cosine is in [-1, 1]; clamp negatives to 0 before blending.
        sem = max(0.0, p.get("semantic_score", 0.0))
        p["lexical_score"] = round(body_overlap, 4)
        p["title_score"] = round(title_overlap, 4)
        p["score"] = round(
            _SEMANTIC_WEIGHT * sem
            + _LEXICAL_WEIGHT * body_overlap
            + _TITLE_WEIGHT * title_overlap,
            4,
        )

    candidates.sort(key=lambda p: p["score"], reverse=True)
    ranked = [p for p in candidates if p["score"] >= min_score][:top_k]

    for i, p in enumerate(ranked, start=1):
        p["cite"] = f"S{i}"
    return ranked
