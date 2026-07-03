"""
Tests for the automated benchmark metrics (evaluation/metrics.py). Pure
functions — fully deterministic and offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import metrics as M


def test_extract_numbers_normalizes():
    nums = M.extract_numbers("current 180 A, heat 1.20 kJ, feed 6.0")
    assert "180" in nums
    assert "1.2" in nums
    assert "6" in nums


def test_faithfulness_all_supported():
    src = "heat_input 1.2 kJ/mm; current 180 A; voltage 24 V"
    resp = "Set current 180 A and voltage 24 V for 1.2 kJ/mm."
    f = M.faithfulness(resp, src)
    assert f["faithfulness"] == 1.0
    assert f["n_invented"] == 0


def test_faithfulness_flags_invented_number():
    src = "current 180 A"
    resp = "Run at 999 A."  # 999 not in source, and not a trivial small int
    f = M.faithfulness(resp, src)
    assert f["n_invented"] == 1
    assert f["faithfulness"] == 0.0
    assert M.hallucination_rate(resp, src) == 1.0


def test_trivial_small_integers_not_flagged():
    src = "no numbers here"
    resp = "First do 1 thing, then 2 things, then 3."
    f = M.faithfulness(resp, src)
    assert f["n_numbers"] == 0  # 1,2,3 are trivial and ignored


def test_citation_score_rewards_preserved_and_penalizes_invented():
    src = "[S1] passage one [S2] passage two"
    good = "Do X [S1] and Y [S2]."
    bad = "Do X [S5]."  # invented label
    assert M.citation_score(good, src)["citation_score"] == 1.0
    assert M.citation_score(bad, src)["citation_score"] < 0.5
    # No labels available → N/A
    assert M.citation_score("anything", "no labels")["citation_score"] is None


def test_readability_and_conciseness_ranges():
    r = M.readability("This is a short, clear sentence for an operator.")
    assert 0 <= r["flesch"] <= 100
    assert 0 <= r["operator_score"] <= 1
    c = M.conciseness("word " * 50)
    assert c["word_count"] == 50
    assert 0 <= c["conciseness_score"] <= 1


def test_safety_refusal_expected_on_error_payload():
    payload = {"error": "unsupported material", "unsupported": True}
    refuse = M.safety_score("I cannot recommend settings; consult your WPS.", "", payload)
    assert refuse["expected_refusal"] is True
    assert refuse["safety_score"] == 1.0
    fabricate = M.safety_score("Use 200 A and 25 V.", "", payload)
    assert fabricate["safety_score"] == 0.0


def test_safety_ungrounded_route_not_penalized_for_numbers():
    # No numeric ground truth: stating a textbook number must not tank safety.
    s = M.safety_score("Typical spindle speed is 3000 rpm; verify against your spec.",
                       "", {}, numeric_grounded=False)
    assert s["safety_score"] >= 0.75


def test_composite_drops_na_metrics():
    grounded = M.score_response("Use 180 A [S1].", "180 A [S1]", {}, "ollama_qwen", 1.0,
                                numeric_grounded=True)
    ungrounded = M.score_response("A clear helpful answer.", "", {}, "ollama_qwen", 1.0,
                                  numeric_grounded=False)
    assert grounded["faithfulness"] is not None
    assert ungrounded["faithfulness"] is None       # N/A on ungrounded route
    assert 0 <= grounded["composite_quality"] <= 1
    assert 0 <= ungrounded["composite_quality"] <= 1


def test_determinism_identical_responses_score_high():
    d = M.determinism(["same answer here", "same answer here", "same answer here"])
    assert d["determinism_score"] == 1.0
    d2 = M.determinism(["short", "a much much much longer different response entirely"])
    assert d2["determinism_score"] < 1.0


def test_cost_local_is_zero_hosted_is_positive():
    assert M.estimate_cost("ollama_qwen", 500, 300) == 0.0
    assert M.estimate_cost("anthropic", 500, 300) > 0.0
