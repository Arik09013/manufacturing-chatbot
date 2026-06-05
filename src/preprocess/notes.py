"""
Operator-notes preprocessing.

Steps:
  1. Text cleaning: lowercase, strip punctuation/symbols, collapse whitespace
  2. Embed each note using a lightweight sentence encoder
     (all-MiniLM-L6-v2, 384-dim)
  3. Return a DataFrame with original metadata + embedding columns

The encoder is lazy-loaded so the import does not block the rest of the
pipeline if sentence-transformers is not yet installed.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBED_DIM = 384


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=1)
def _get_encoder():
    """Lazy-load the sentence encoder (downloads model on first call)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Return (N, 384) float32 embedding matrix."""
    encoder = _get_encoder()
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def preprocess_notes(
    df: pd.DataFrame,
    embed: bool = True,
) -> pd.DataFrame:
    """
    Clean note text and (optionally) compute sentence embeddings.

    Returns the original DataFrame plus:
      - clean_text: cleaned note string
      - emb_0 … emb_383: embedding dimensions (if embed=True)
    """
    df = df.copy()
    df["clean_text"] = df["note_text"].astype(str).map(clean_text)

    if embed:
        vectors = embed_texts(df["clean_text"].tolist())
        emb_cols = pd.DataFrame(
            vectors,
            columns=[f"emb_{i}" for i in range(vectors.shape[1])],
            index=df.index,
        )
        df = pd.concat([df, emb_cols], axis=1)

    return df
