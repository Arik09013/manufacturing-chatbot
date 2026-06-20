"""
Attention-visualization XAI for operator notes (the third XAI method, alongside
SHAP and LIME).

Runs a free-form operator note through a pretrained DistilBERT transformer and
extracts its self-attention weights, exposing *which words the model focused on*
when encoding the note. This is the text-modality counterpart to SHAP/LIME on
the sensor features — it makes the note-understanding step interpretable.

The weights are descriptive of the transformer's encoding, not of the anomaly
label directly (the detector is a RandomForest). We surface them as a
human-readable heatmap so an operator can see, e.g., that "gas", "empty" drew
the most attention in "arc cut out, gas bottle felt empty".

First call downloads `distilbert-base-uncased` (~265 MB) and caches it; later
calls are fast. Everything here is best-effort and guarded by the caller.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "distilbert-base-uncased"
_SPECIAL = {"[CLS]", "[SEP]", "[PAD]"}


@lru_cache(maxsize=1)
def _load_model():
    """Load tokenizer + DistilBERT (eager attention so attentions are returned)."""
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModel.from_pretrained(
        _MODEL_NAME,
        output_attentions=True,
        attn_implementation="eager",
    )
    model.eval()
    return tok, model


def _merge_wordpieces(tokens: list[str], weights: np.ndarray) -> list[dict]:
    """
    Merge '##' word-pieces back into whole words (max-pool their weights) and
    drop special tokens, so the heatmap is word-level and readable.
    """
    words: list[dict] = []
    for tok, w in zip(tokens, weights):
        if tok in _SPECIAL:
            continue
        if not any(ch.isalnum() for ch in tok):
            continue  # drop pure-punctuation tokens (BERT over-attends to them)
        if tok.startswith("##") and words:
            words[-1]["token"] += tok[2:]
            words[-1]["weight"] = max(words[-1]["weight"], float(w))
        else:
            words.append({"token": tok, "weight": float(w)})
    return words


def explain_attention(text: str, max_tokens: int = 64) -> dict | None:
    """
    Return a per-word attention heatmap for an operator note.

    Returns
    -------
    dict with:
      text        : the input note
      tokens      : list of {token, weight} where weight is 0..1 (normalised)
      top_tokens  : the most-attended words (token strings), highest first
    or None if the text is empty.
    """
    if not text or not str(text).strip():
        return None

    import torch

    tok, model = _load_model()
    enc = tok(str(text), return_tensors="pt", truncation=True, max_length=max_tokens)
    with torch.no_grad():
        out = model(**enc)

    # out.attentions: tuple(num_layers) of [batch, heads, seq, seq]
    atts = torch.stack(out.attentions)          # [layers, batch, heads, seq, seq]
    att = atts.mean(dim=(0, 2))[0]              # mean over layers & heads -> [seq, seq]
    received = att.mean(dim=0).cpu().numpy()    # attention each token receives -> [seq]

    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    words = _merge_wordpieces(tokens, received)
    if not words:
        return None

    # Normalise weights to 0..1 across the (content) words for display
    ws = np.array([w["weight"] for w in words], dtype=float)
    lo, hi = ws.min(), ws.max()
    span = (hi - lo) or 1.0
    for w in words:
        w["weight"] = round((w["weight"] - lo) / span, 3)

    top = sorted(words, key=lambda w: w["weight"], reverse=True)[:5]
    return {
        "text":       str(text),
        "tokens":     words,
        "top_tokens": [w["token"] for w in top],
    }


def explain_text(result: dict) -> str:
    """Plain-text rendering of the attention focus."""
    if not result:
        return "No operator note to analyse."
    return (
        f'Operator note: "{result["text"]}"\n'
        f"Model focused most on: {', '.join(result['top_tokens'])}"
    )
