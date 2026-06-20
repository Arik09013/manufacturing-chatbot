"""Tests for the LIME explainability layer and its pipeline integration."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL = Path(__file__).parent.parent / "models" / "anomaly.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="model not trained")


def test_lime_explainer_returns_drivers():
    from src.explain.lime_explainer import get_default_lime_explainer
    from src.fusion.fuse import load_fused

    explainer = get_default_lime_explainer(top_n=5)
    row = load_fused().iloc[[0]]
    res = explainer.explain_row(row)

    assert "lime_drivers" in res and res["lime_drivers"]
    d = res["lime_drivers"][0]
    assert {"feature", "condition", "weight", "direction", "magnitude"} <= set(d)
    assert d["direction"] in ("toward_anomaly", "toward_normal")


def test_lime_feature_maps_to_known_name():
    from src.explain.lime_explainer import get_default_lime_explainer

    explainer = get_default_lime_explainer(top_n=5)
    feat = explainer._feature_from_condition("1.0 < welding_current_std <= 5.0")
    assert feat == "welding_current_std"


def test_pipeline_payload_includes_lime():
    from src.api.pipeline import run_pipeline

    payload = run_pipeline("Why did station_1 stop welding at 14:00?")
    assert "lime_drivers" in payload  # present even if empty on a normal window


# ── Attention visualization (DistilBERT) ──────────────────────────────────────

def test_attention_highlights_content_words():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from src.explain.attention_explainer import explain_attention

    res = explain_attention("Unstable arc on station_1, may need new contact tip")
    assert res and res["tokens"]
    # weights normalised to 0..1, punctuation dropped
    assert all(0.0 <= t["weight"] <= 1.0 for t in res["tokens"])
    assert all(any(c.isalnum() for c in t["token"]) for t in res["tokens"])
    assert "arc" in [t["token"] for t in res["tokens"]]
    assert len(res["top_tokens"]) <= 5


def test_attention_empty_text_returns_none():
    pytest.importorskip("transformers")
    from src.explain.attention_explainer import explain_attention
    assert explain_attention("   ") is None
