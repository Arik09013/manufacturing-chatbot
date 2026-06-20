"""Tests for the welding knowledge base, matcher, and question router."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning.knowledge import match_knowledge, build_knowledge_summary
from src.api.pipeline import route_question, run_knowledge_pipeline
from src.chat.intent import OutOfScopeError


# ── Knowledge matcher ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected_topic", [
    ("Why am I getting porosity in my stainless steel welds?", "porosity"),
    ("How can I prevent undercut during MIG welding?", "undercut"),
    ("What causes lack of fusion in 5mm plate welding?", "lack_of_fusion"),
    ("How do I reduce spatter while maintaining penetration?", "spatter"),
    ("How can I avoid distortion in thin stainless steel sheets?", "distortion"),
    ("My weld is cracking after cooling. What could be causing it?", "cracking"),
    ("Why is my bead too narrow?", "narrow_bead"),
    ("Why does discoloration appear around the weld?", "discoloration"),
    ("How can I maximize weld tensile strength?", "tensile_strength"),
    ("How can I reduce shielding gas consumption?", "gas_consumption"),
    ("Should I use spray transfer or pulse MIG for this application?", "transfer_mode"),
    ("Which shielding gas gives the strongest weld on stainless steel?", "shielding_gas_choice"),
    ("How can I increase deposition rate without sacrificing quality?", "deposition_increase"),
    ("Why am I experiencing excessive warping?", "distortion"),
])
def test_match_knowledge_topic(question, expected_topic):
    topics = match_knowledge(question)
    assert topics, f"No KB match for: {question!r}"
    assert topics[0]["topic"] == expected_topic, (
        f"{question!r} -> {topics[0]['topic']} (expected {expected_topic})"
    )


def test_match_knowledge_returns_structured_entry():
    topics = match_knowledge("Why am I getting porosity?")
    t = topics[0]
    assert t["causes"] and t["remedies"]
    assert "shielding_gas_flow" in t["parameters"]


def test_match_knowledge_no_match():
    assert match_knowledge("zzz nonsense token qwerty") == []


def test_build_summary_nonempty():
    topics = match_knowledge("How can I prevent undercut?")
    summary = build_knowledge_summary("How can I prevent undercut?", topics)
    assert "Undercut" in summary
    assert "Corrective actions" in summary


# ── Router ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected_route", [
    ("Best settings for 5mm mild steel using MIG", "param"),
    ("What voltage and wire feed for 5mm stainless steel?", "param"),
    ("Why did station_1 stop welding at 14:00?", "anomaly"),
    ("Is station_3 running within normal parameters?", "anomaly"),
    ("Why am I getting porosity in my welds?", "knowledge"),
    ("How can I reduce manufacturing cost?", "knowledge"),
    ("Should I use spray transfer or pulse MIG?", "knowledge"),
])
def test_route_question(question, expected_route):
    assert route_question(question) == expected_route


# ── Knowledge pipeline ────────────────────────────────────────────────────────

def test_run_knowledge_pipeline_shape():
    payload = run_knowledge_pipeline("Why am I getting porosity in my stainless welds?")
    assert payload["pipeline_type"] == "knowledge_advice"
    assert payload["matched_topics"][0]["topic"] == "porosity"
    assert payload["summary_text"]


def test_run_knowledge_pipeline_rejects_out_of_scope():
    with pytest.raises(OutOfScopeError):
        run_knowledge_pipeline("Who won the World Cup?")
