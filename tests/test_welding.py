"""Smoke tests for the welding domain — anomaly pipeline + parameter advisor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.pipeline import run_pipeline, run_param_pipeline
from src.reasoning.param_advisor import recommend_parameters, parse_param_query
from src.chat.intent import classify_intent


# ── Parameter advisor ─────────────────────────────────────────────────────────

def test_param_mild_steel_medium():
    r = recommend_parameters("mild_steel", 5.0)
    assert r["efficiency_score"] >= 1
    assert r["optimal_current"] > 0
    assert r["optimal_speed"] > 0
    assert "mild" in r["material"]

def test_param_aluminum_thin():
    r = recommend_parameters("aluminum", 2.0)
    assert r["band"] == "thin"
    assert r["optimal_current"] < 200

def test_param_stainless_thick():
    r = recommend_parameters("stainless steel", 12.0)
    assert r["band"] == "thick"

def test_param_aliases():
    assert recommend_parameters("steel", 5.0)["material"] == "mild_steel"
    assert recommend_parameters("aluminium", 3.0)["material"] == "aluminum"
    assert recommend_parameters("SS", 5.0)["material"] == "stainless_steel"

def test_parse_param_query():
    assert parse_param_query("best settings for 5mm mild steel") is not None
    assert parse_param_query("what speed for 3mm aluminum") is not None
    assert parse_param_query("optimal current for 10mm stainless") is not None
    assert parse_param_query("why did station_1 stop?") is None

def test_param_pipeline():
    r = run_param_pipeline("best settings for 6mm mild steel")
    assert r["pipeline_type"] == "parameter_advice"
    assert r["optimal_speed"] > 0


# ── Anomaly pipeline ──────────────────────────────────────────────────────────

def test_anomaly_pipeline_runs():
    r = run_pipeline("Why did station_1 stop welding?")
    assert "is_anomaly" in r
    assert "confidence" in r
    assert "shap_drivers" in r

def test_anomaly_pipeline_has_welding_features():
    r = run_pipeline("Arc fault on station_1?")
    drivers = r.get("shap_drivers", [])
    feature_names = [d["feature"] for d in drivers]
    welding_terms = ["welding_current", "arc_voltage", "welding_speed",
                     "wire_feed_rate", "shielding_gas_flow", "heat_input"]
    assert any(any(t in f for t in welding_terms) for f in feature_names)


# ── Intent ────────────────────────────────────────────────────────────────────

def test_welding_intent_in_scope():
    assert classify_intent("Why did station_1 stop?").in_scope
    assert classify_intent("Arc instability alarm on station_2").in_scope
    assert classify_intent("Best settings for 5mm mild steel").in_scope
    assert classify_intent("Optimal welding speed for aluminum").in_scope
    assert classify_intent("Wire feed fault root cause").in_scope
    assert classify_intent("Gas flow alarm on station_3").in_scope

def test_welding_intent_out_of_scope():
    assert not classify_intent("Who won the World Cup?").in_scope
    assert not classify_intent("Tell me a joke").in_scope


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
