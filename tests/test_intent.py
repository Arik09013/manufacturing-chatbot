"""
Tests for the intent classification layer.

Covers:
  - Out-of-scope queries that must never reach the pipeline
  - In-scope queries that must pass through
  - Edge cases and partial matches
  - OutOfScopeError raised by run_pipeline
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.intent import classify_intent, OutOfScopeError, OUT_OF_SCOPE_MESSAGE


# ── Out-of-scope ──────────────────────────────────────────────────────────────

OUT_OF_SCOPE_QUERIES = [
    "Who won the World Cup?",
    "Tell me a joke.",
    "What is the capital of Bangladesh?",
    "What is 2 + 2?",
    "Write a poem about summer.",
    "Who is the president of France?",
    "What is the weather today?",
    "Recommend a good restaurant.",
    "Translate this to Spanish.",
    "What is the meaning of life?",
    "How do I bake a chocolate cake?",
    "What movies are playing tonight?",
]

@pytest.mark.parametrize("question", OUT_OF_SCOPE_QUERIES)
def test_out_of_scope(question):
    result = classify_intent(question)
    assert not result.in_scope, (
        f"Expected out-of-scope for: {question!r}\n"
        f"Got matched_term={result.matched_term!r}, reason={result.reason}"
    )


# ── In-scope ──────────────────────────────────────────────────────────────────

IN_SCOPE_QUERIES = [
    # Core demo queries
    "Why did line 3 slow down at 14:00?",
    "Why is machine_1 running hot at 15:30?",
    "Is machine_2 running normally?",
    "What caused the vibration alarm on machine_2?",
    # Equipment references
    "Is the pump running normally?",
    "Check the bearing on line 2.",
    "The motor tripped — what should I do?",
    "Valve is stuck on machine_3.",
    # Sensor references
    "What is the temperature reading on machine_1?",
    "Pressure dropped below threshold.",
    "High vibration detected.",
    "RPM is fluctuating on the conveyor.",
    # Anomaly / fault references
    "There is a fault on the production line.",
    "Alarm triggered on machine_2.",
    "Overload fault at 10:00 — root cause?",
    "Why did the overheating alarm trigger?",
    "What caused the coolant failure?",
    # Maintenance / operations
    "Schedule maintenance for the bearing.",
    "When did the last downtime occur?",
    "Operator noted a noisy gearbox.",
    "Shift report shows anomaly at 08:00.",
    # Natural language variants
    "machine 1 is running hot",
    "Why did line_2 stop?",
    "line 1 alarm",
]

@pytest.mark.parametrize("question", IN_SCOPE_QUERIES)
def test_in_scope(question):
    result = classify_intent(question)
    assert result.in_scope, (
        f"Expected in-scope for: {question!r}\n"
        f"Got reason={result.reason!r}"
    )


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_string():
    result = classify_intent("")
    assert not result.in_scope

def test_whitespace_only():
    result = classify_intent("   ")
    assert not result.in_scope

def test_case_insensitive_keyword():
    assert classify_intent("MACHINE_1 IS OVERHEATING").in_scope
    assert classify_intent("Vibration Alert on Line 2").in_scope

def test_machine_pattern_variants():
    assert classify_intent("machine_1 fault").in_scope
    assert classify_intent("machine 2 alarm").in_scope
    assert classify_intent("line 3 slow down").in_scope

def test_result_fields():
    r = classify_intent("vibration alarm on machine_1")
    assert r.in_scope is True
    assert 0.0 <= r.confidence <= 1.0
    assert isinstance(r.matched_term, str)
    assert isinstance(r.reason, str)

def test_out_of_scope_result_fields():
    r = classify_intent("Who won the World Cup?")
    assert r.in_scope is False
    assert r.matched_term == ""
    assert "manufacturing" in r.reason.lower() or "no" in r.reason.lower()


# ── OutOfScopeError ───────────────────────────────────────────────────────────

def test_out_of_scope_error_message():
    from src.chat.intent import classify_intent, OutOfScopeError
    result = classify_intent("Who won the World Cup?")
    exc = OutOfScopeError("Who won the World Cup?", result)
    msg = str(exc)
    assert "manufacturing" in msg.lower()
    assert "machine" in msg.lower()


def test_pipeline_raises_out_of_scope():
    """run_pipeline must raise OutOfScopeError for non-manufacturing questions."""
    from src.api.pipeline import run_pipeline
    with pytest.raises(OutOfScopeError):
        run_pipeline("Who won the World Cup?")

    with pytest.raises(OutOfScopeError):
        run_pipeline("Tell me a joke.")

    with pytest.raises(OutOfScopeError):
        run_pipeline("What is the capital of Bangladesh?")
