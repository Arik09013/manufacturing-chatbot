"""Smoke tests for the welding domain — anomaly pipeline + parameter advisor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.pipeline import run_pipeline, run_param_pipeline, route_question
from src.reasoning.param_advisor import (
    recommend_parameters, parse_param_query, parse_param_overrides,
    parse_param_bounds, parse_param_ranges, merge_param_context,
)
from src.chat.intent import classify_intent


# ── Parameter advisor ─────────────────────────────────────────────────────────

def test_param_mild_steel_medium():
    r = recommend_parameters("mild_steel", 5.0)
    assert r["efficiency_score"] >= 1
    assert r["optimized"]["welding_current"] > 0
    assert r["optimized"]["welding_speed"] > 0
    assert "mild" in r["material"]
    assert r["computed_metrics"]["heat_input_kj_per_mm"] > 0
    assert r["computed_metrics"]["deposition_rate_g_per_min"] > 0

def test_param_aluminum_thin():
    r = recommend_parameters("aluminum", 2.0)
    assert r["band"] == "thin"
    assert r["optimized"]["welding_current"] < 200

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

def test_parse_param_query_detects_process():
    assert parse_param_query("best settings for 5mm mild steel using MIG")["process"] == "MIG"
    assert parse_param_query("what speed for 3mm aluminum with TIG")["process"] == "TIG"
    assert parse_param_query("optimal current for 10mm stainless")["process"] is None

def test_param_pipeline():
    r = run_param_pipeline("best settings for 6mm mild steel")
    assert r["pipeline_type"] == "parameter_advice"
    assert r["optimized"]["welding_speed"] > 0


# ── Parameter overrides ("if I fix X, recompute the rest") ─────────────────────

def test_override_query_is_recognized_without_material_or_thickness():
    """The reported bug: "if i used 1.5 wire diameter then give me other values"
    must route to the param advisor, not fall through to the knowledge base."""
    q = "if i used 1.5 wire diameter then give me other values"
    assert route_question(q) == "param"
    parsed = parse_param_query(q)
    assert parsed is not None
    assert parsed["overrides"] == {"wire_diameter": 1.5}
    assert parsed["defaulted"] == {"material": True, "thickness": True}

def test_parse_overrides_each_axis():
    assert parse_param_overrides("run it at 260 A") == {"welding_current": 260.0}
    assert parse_param_overrides("set voltage to 24") == {"arc_voltage": 24.0}
    assert parse_param_overrides("wire diameter 1.6") == {"wire_diameter": 1.6}
    assert parse_param_overrides("at 300 mm/min") == {"welding_speed": 300.0}
    assert parse_param_overrides("wire feed 9 m/min") == {"wire_feed_rate": 9.0}

def test_override_speed_not_mistaken_for_thickness():
    """A pinned "300 mm/min" speed must not be read as a 300 mm plate thickness."""
    parsed = parse_param_query("other settings for 8mm stainless at 300 mm/min")
    assert parsed["thickness_mm"] == 8.0
    assert parsed["overrides"] == {"welding_speed": 300.0}

def test_override_wire_diameter_recomputes_deposition():
    base = recommend_parameters("mild_steel", 5.0, process="MIG")
    fixed = recommend_parameters("mild_steel", 5.0, process="MIG",
                                 overrides={"wire_diameter": 1.6})
    # Bigger wire at the same feed → strictly higher deposition rate.
    assert fixed["wire_diameter"] == 1.6
    assert fixed["computed_metrics"]["deposition_rate_g_per_min"] > \
           base["computed_metrics"]["deposition_rate_g_per_min"]
    assert any(f["field"] == "wire_diameter" for f in fixed["user_fixed"])

def test_override_pins_axis_and_flags_out_of_window():
    r = recommend_parameters("mild_steel", 6.0, process="MIG",
                             overrides={"welding_current": 260.0})
    assert r["optimized"]["welding_current"] == 260            # held fixed
    assert r["override_warnings"]                              # 260 A is outside 130-220
    assert any("outside the standards window" in w for w in r["override_warnings"])

def test_no_override_leaves_recommendation_clean():
    r = recommend_parameters("mild_steel", 5.0, process="MIG")
    assert r["user_fixed"] == []
    assert r["overrides"] == {}
    assert r["override_warnings"] == []

def test_plain_param_query_has_no_spurious_overrides():
    for q in ("best settings for 5mm mild steel",
              "what speed for 3mm aluminum",
              "optimal current for 10mm stainless"):
        assert parse_param_overrides(q) == {}, q


# ── Bare "diameter" / comparator parsing (follow-up phrasing) ──────────────────

def test_bare_diameter_without_wire_word():
    """The reported bug: a follow-up says "stay in 1.5 diameter" with no "wire"
    word (it was established earlier) — it must still parse as wire diameter."""
    assert parse_param_overrides("stay in 1.5 diameter") == {"wire_diameter": 1.5}
    assert parse_param_overrides("1.5 mm dia") == {"wire_diameter": 1.5}
    assert parse_param_overrides("diameter of 1.6") == {"wire_diameter": 1.6}

def test_comparator_becomes_bound_not_exact_pin():
    """"more than 21 voltage" is a lower bound, not an exact pin."""
    assert parse_param_bounds("more than 21 voltage") == {
        "arc_voltage": {"op": "min", "value": 21.0}}
    assert parse_param_bounds("keep current above 200 A") == {
        "welding_current": {"op": "min", "value": 200.0}}
    assert parse_param_bounds("under 24 v") == {
        "arc_voltage": {"op": "max", "value": 24.0}}
    # No comparator -> it stays an exact pin, not a bound.
    assert parse_param_bounds("set voltage to 24") == {}
    assert parse_param_overrides("set voltage to 24") == {"arc_voltage": 24.0}

def test_comparator_not_tripped_by_substrings():
    """Comparator detection is word-bounded and anchored, so material/feed words
    that merely contain 'min'/'over' don't create phantom bounds."""
    # "aluminum ... 24 v" must not read "min" out of "aluminum".
    assert parse_param_bounds("3mm aluminum at 24 v") == {}
    assert parse_param_overrides("at 300 mm/min and 24 v") == {
        "welding_speed": 300.0, "arc_voltage": 24.0}

def test_instead_of_value_not_read_as_thickness():
    """The replaced value in "instead of 1.2mm" must NOT become a plate thickness."""
    parsed = parse_param_query("if i use 1.5mm wire diameter instead of 1.2mm what is the outcome?")
    assert parsed["overrides"] == {"wire_diameter": 1.5}
    assert parsed["defaulted"]["thickness"] is True   # 1.2 was not captured as thickness

def test_voltage_lower_bound_is_honored_by_optimizer():
    r = recommend_parameters("mild_steel", 5.0, process="MIG",
                             bounds={"arc_voltage": {"op": "min", "value": 21.0}})
    assert r["optimized"]["arc_voltage"] >= 21.0
    assert any(f["field"] == "arc_voltage" and f["op"] == "min" for f in r["user_fixed"])


# ── Ranged / value-before-keyword parameter parsing ("400-500 speed") ──────────

def test_value_before_keyword_speed():
    """A bare "450 speed" (number before the word, no compact unit) must parse."""
    assert parse_param_overrides("run it at 450 speed") == {"welding_speed": 450.0}
    assert parse_param_overrides("use 320 travel speed") == {"welding_speed": 320.0}

def test_ranged_speed_pins_midpoint_and_records_band():
    """"400-500 speed" pins the midpoint and remembers the band for display."""
    assert parse_param_overrides("use 400-500 speed") == {"welding_speed": 450.0}
    assert parse_param_ranges("use 400-500 speed") == {"welding_speed": (400.0, 500.0)}
    # "to" and keyword-first phrasings work too.
    assert parse_param_overrides("speed of 400 to 500") == {"welding_speed": 450.0}

def test_ranged_current_and_voltage():
    assert parse_param_overrides("keep current 200-260 A") == {"welding_current": 230.0}
    assert parse_param_ranges("keep current 200-260 A") == {"welding_current": (200.0, 260.0)}
    assert parse_param_overrides("voltage 24-28 v") == {"arc_voltage": 26.0}

def test_followup_speed_change_routes_to_param():
    """The reported bug: a follow-up that changes speed ("400-500 speed instead of
    200-300") must route to the param advisor, not fall through to the knowledge base."""
    q = "i want to use 400-500 speed instead of 200-300 what is the change in current and voltage?"
    assert route_question(q) == "param"
    parsed = parse_param_query(q)
    assert parsed is not None
    assert parsed["overrides"] == {"welding_speed": 450.0}
    assert parsed["override_ranges"] == {"welding_speed": (400.0, 500.0)}
    # The replaced "200-300" must NOT leak in as a second override or a thickness.
    assert "welding_current" not in parsed["overrides"]

def test_ranged_speed_does_not_eat_thickness():
    """A plate thickness range ("5-8mm steel") is not a travel-speed band."""
    assert parse_param_overrides("best settings for 5-8mm steel") == {}

def test_speed_change_followup_raises_current_and_voltage():
    """End-to-end: after 'best settings for 8mm mild steel', asking to run faster
    (400-500 vs the optimized ~380) recomputes a HIGHER current and voltage to hold
    heat input inside the standards window — the answer the operator asked for."""
    t1 = run_param_pipeline("Best settings for 8mm mild steel")
    base_i = t1["optimized"]["welding_current"]
    base_v = t1["optimized"]["arc_voltage"]

    t2 = run_param_pipeline(
        "i want to use 400-500 speed instead of 200-300 what is the change in current and voltage?",
        context=t1["param_context"],
    )
    assert t2["pipeline_type"] == "parameter_advice"
    assert t2["material"] == "mild_steel" and t2["thickness_mm"] == 8.0   # inherited
    assert t2["optimized"]["welding_speed"] == 450                        # band midpoint pinned
    assert t2["optimized"]["welding_current"] > base_i                    # faster -> more current
    assert t2["optimized"]["arc_voltage"] >= base_v
    # heat input stays inside the standards window despite the faster travel speed
    hi = t2["computed_metrics"]
    assert hi["heat_input_range"][0] <= hi["heat_input_kj_per_mm"] <= hi["heat_input_range"][1]
    # the operator-facing card shows the band they asked for
    assert any(f["field"] == "welding_speed" and f["op"] == "range" for f in t2["user_fixed"])
    assert t2["override_warnings"]                                        # 400-500 is above 200-380


# ── Conversation context carry-forward ─────────────────────────────────────────

def test_merge_carries_material_and_thickness():
    prior = {"material": "mild_steel", "thickness_mm": 5.0, "process": None,
             "overrides": {}, "bounds": {}}
    parsed = parse_param_query("if i use 1.5 wire diameter then give me other values")
    merged = merge_param_context(parsed, prior)
    assert merged["material"] == "mild_steel"
    assert merged["thickness_mm"] == 5.0
    assert merged["overrides"]["wire_diameter"] == 1.5
    assert merged["defaulted"] == {"material": False, "thickness": False}

def test_merge_accumulates_overrides_and_resets_on_new_scenario():
    prior = {"material": "mild_steel", "thickness_mm": 5.0, "process": None,
             "overrides": {"wire_diameter": 1.5}, "bounds": {}}
    # Follow-up adds a voltage bound but keeps the wire diameter from before.
    parsed = parse_param_query("more than 21 voltage")
    merged = merge_param_context(parsed, prior)
    assert merged["overrides"]["wire_diameter"] == 1.5
    assert merged["bounds"]["arc_voltage"] == {"op": "min", "value": 21.0}
    # Naming a different thickness starts a fresh scenario -> stale overrides drop.
    parsed2 = parse_param_query("best settings for 10mm stainless")
    merged2 = merge_param_context(parsed2, prior)
    assert merged2["material"] == "stainless_steel"
    assert merged2["thickness_mm"] == 10.0
    assert merged2["overrides"] == {}

def test_three_turn_conversation_keeps_wire_diameter():
    """End-to-end reproduction of the reported bug across three turns:
       1) best settings for 5mm mild steel
       2) if i use 1.5mm wire diameter instead of 1.2mm
       3) stay in 1.5 diameter and more than 21 voltage
    Turn 3 must stay at 1.5 mm wire (not fall back to 1.2) and 5mm mild steel."""
    t1 = run_param_pipeline("Best settings for 5mm mild steel")
    ctx = t1["param_context"]
    assert ctx["material"] == "mild_steel" and ctx["thickness_mm"] == 5.0

    t2 = run_param_pipeline(
        "if i use 1.5mm wire diameter instead of 1.2mm what is the outcome?",
        context=ctx,
    )
    assert t2["wire_diameter"] == 1.5
    assert t2["thickness_mm"] == 5.0      # not 1.2 — "instead of 1.2mm" is the old value
    assert t2["material"] == "mild_steel"
    ctx = t2["param_context"]

    t3 = run_param_pipeline(
        "actually i want to stay in 1.5 diameter and more than 21 voltage then what is the output",
        context=ctx,
    )
    assert t3["wire_diameter"] == 1.5     # the core fix — it no longer reverts to 1.2
    assert t3["thickness_mm"] == 5.0
    assert t3["band"] == "medium"
    assert t3["optimized"]["arc_voltage"] >= 21.0


# ── Process-aware optimization & null handling ────────────────────────────────

def test_param_gmaw_has_deposition_and_gas():
    r = recommend_parameters("mild_steel", 5.0, process="MIG")
    assert r["process_family"] == "GMAW"
    assert r["ranges"]["wire_feed_rate"] is not None
    assert r["ranges"]["shielding_gas_flow"] is not None
    assert r["computed_metrics"]["deposition_rate_g_per_min"] is not None
    assert r["consumable_field"] == "wire_diameter"

def test_param_tig_has_no_wire_feed():
    r = recommend_parameters("stainless_steel", 3.0, process="TIG")
    assert r["process_family"] == "TIG"
    assert r["optimized"]["wire_feed_rate"] is None
    assert r["ranges"]["wire_feed_rate"] is None
    assert r["computed_metrics"]["deposition_rate_g_per_min"] is None
    assert r["consumable_field"] == "tungsten_diameter"
    # arc is still struck through shielding gas
    assert r["ranges"]["shielding_gas_flow"] is not None

def test_param_smaw_has_no_wire_feed_or_gas():
    r = recommend_parameters("mild_steel", 10.0, process="SMAW")
    assert r["process_family"] == "SMAW"
    assert r["optimized"]["wire_feed_rate"] is None
    assert r["ranges"]["wire_feed_rate"] is None
    assert r["ranges"]["shielding_gas_flow"] is None
    assert r["computed_metrics"]["deposition_rate_g_per_min"] is None
    assert r["consumable_field"] == "electrode_diameter"

def test_param_aluminum_smaw_unsupported():
    r = recommend_parameters("aluminum", 5.0, process="SMAW")
    assert "error" in r
    assert "supported_processes" in r
    assert "SMAW" not in r["supported_processes"]


# ── Optimization & sensitivity structure ──────────────────────────────────────

def test_param_optimized_within_standards_window():
    r = recommend_parameters("mild_steel", 8.0, process="MIG")
    rng = r["ranges"]["welding_current"]
    assert rng[0] <= r["optimized"]["welding_current"] <= rng[1]
    hi_rng = r["computed_metrics"]["heat_input_range"]
    assert hi_rng[0] <= r["computed_metrics"]["heat_input_kj_per_mm"] <= hi_rng[1]

def test_param_sensitivity_structure():
    r = recommend_parameters("mild_steel", 5.0, process="MIG")
    sens = r["sensitivity"]
    assert len(sens) > 0
    row = next(s for s in sens if s["parameter"] == "welding_current")
    for side in ("up", "down"):
        entry = row[side]
        assert {"value", "heat_input", "deposition_rate", "heat_input_change", "heat_input_effects"} <= entry.keys()
    assert "increase" in row["effects"] and "decrease" in row["effects"]

def test_param_wear_effects_attached_to_current_not_speed():
    """Critical physical-correctness constraint: consumable/contact-tip/tungsten
    wear must be driven by current/heat input, never by travel speed."""
    r = recommend_parameters("mild_steel", 5.0, process="MIG")
    sens = r["sensitivity"]
    speed_row = next(s for s in sens if s["parameter"] == "welding_speed")
    current_row = next(s for s in sens if s["parameter"] == "welding_current")

    def _effect_names(effects):
        return {e["effect"] for e in effects}

    speed_effects = _effect_names(speed_row["effects"]["increase"]) | _effect_names(speed_row["effects"]["decrease"])
    current_effects = _effect_names(current_row["effects"]["increase"]) | _effect_names(current_row["effects"]["decrease"])
    wear_terms = {name for name in (speed_effects | current_effects) if "wear" in name}
    assert wear_terms, "expected at least one wear-related effect label in the data"
    assert wear_terms.isdisjoint(speed_effects)
    assert wear_terms <= current_effects


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
