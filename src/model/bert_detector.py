"""
DistilBERT fault detector — a fine-tuned transformer alternative to the
RandomForest anomaly detector.

This serialises each fused multimodal window (sensor aggregates + log event
counts + note presence) into a short text description and classifies it with a
fine-tuned DistilBERT sequence classifier. It is the "LLM fine-tuning" path from
the thesis, kept as an *optional* second detector so it can be compared head-to-
head with the RandomForest baseline (`anomaly.py`).

Training: see `src/model/finetune_distilbert.py`.
The fine-tuned weights are saved to `models/distilbert_fault/` (on the E drive).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "distilbert_fault"
BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 128

_SENSORS = [
    ("welding_current", "current", "A"),
    ("arc_voltage", "voltage", "V"),
    ("welding_speed", "speed", "mm/min"),
    ("wire_feed_rate", "wire feed", "m/min"),
    ("shielding_gas_flow", "gas flow", "L/min"),
    ("heat_input", "heat input", "kJ/mm"),
]


def window_to_text(row: pd.Series) -> str:
    """Serialise one fused window into a compact natural-language description."""
    parts = ["Welding station window."]
    for col, label, unit in _SENSORS:
        mean = row.get(f"{col}_mean")
        std = row.get(f"{col}_std")
        rng = row.get(f"{col}_range")
        if mean is None or pd.isna(mean):
            continue
        seg = f"{label} mean {float(mean):.1f}{unit}"
        if std is not None and not pd.isna(std):
            seg += f" std {float(std):.1f}"
        if rng is not None and not pd.isna(rng):
            seg += f" range {float(rng):.1f}"
        parts.append(seg + ".")

    n_alarm = int(row.get("n_alarm", 0) or 0)
    n_warn = int(row.get("n_warning", 0) or 0)
    n_diag = int(row.get("n_diagnostic", 0) or 0)
    parts.append(f"Log: {n_alarm} alarms, {n_warn} warnings, {n_diag} diagnostics.")

    codes = str(row.get("unique_event_codes", "") or "").strip()
    parts.append(f"Event codes: {codes if codes and codes != 'nan' else 'none'}.")

    has_note = bool(row.get("has_note", False))
    parts.append("Operator note present." if has_note else "No operator note.")
    return " ".join(parts)


def build_texts(df: pd.DataFrame) -> list[str]:
    """Serialise every row of a fused DataFrame to text."""
    return [window_to_text(r) for _, r in df.iterrows()]


class BertFaultDetector:
    """Inference wrapper around a fine-tuned DistilBERT fault classifier."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()

    @classmethod
    def is_available(cls) -> bool:
        return (MODEL_DIR / "config.json").exists()

    def predict_proba(self, df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        """Return anomaly probability (class 1) for each row."""
        torch = self._torch
        texts = build_texts(df)
        probs: list[float] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", truncation=True,
                                 padding=True, max_length=MAX_LEN)
            with torch.no_grad():
                logits = self.model(**enc).logits
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
        return np.asarray(probs)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)
