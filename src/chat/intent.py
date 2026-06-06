"""
Intent classification: decide whether a question is within the supported
manufacturing-operations scope before running the pipeline.

Strategy: keyword + regex matching against a curated vocabulary of
industrial/manufacturing terms.  Fast, deterministic, zero extra deps.

Returns an IntentResult with:
  in_scope : bool
  confidence : float (1.0 = strong keyword hit, 0.5 = pattern-only hit)
  reason : str  (human-readable explanation for logging / error messages)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ── Vocabulary ────────────────────────────────────────────────────────────────

# Any single match from this set → in-scope.
# Words are matched as whole tokens (word boundaries) after lowercasing.
MANUFACTURING_KEYWORDS: frozenset[str] = frozenset({
    # Machines & equipment
    "machine", "motor", "pump", "valve", "bearing", "spindle",
    "compressor", "actuator", "conveyor", "robot", "gearbox", "belt",
    "coolant", "fan", "heater", "boiler", "turbine",
    # Sensors & measurements
    "sensor", "temperature", "vibration", "pressure", "rpm",
    "torque", "flowrate", "flow rate", "current", "voltage",
    "power consumption",
    # Anomaly / fault vocabulary
    "anomaly", "anomalies", "fault", "failure", "alarm", "alert",
    "warning", "overload", "overheating", "overheat", "trip",
    "diagnostic",
    # Operations
    "production", "manufacturing", "factory", "plant",
    "maintenance", "downtime", "startup", "shutdown",
    "operator", "shift", "batch", "setpoint", "throughput",
    # Condition descriptions
    "slow down", "slowing down", "stopped", "noisy", "leaking",
    "hot", "vibrating", "stalled",
    # Pipeline concepts
    "root cause", "root-cause", "recommendation", "log event",
})

# Regex patterns that strongly signal manufacturing context even if the exact
# keyword isn't in the vocabulary (e.g. "line 3", "machine_2").
_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bmachine[_\s]\d\b"),       # machine_1, machine 2
    re.compile(r"\bline[_\s]\d\b"),          # line 1, line_3
    re.compile(r"\b(m1|m2|m3)\b"),
    re.compile(r"\bsensor\s+\w+\b"),         # sensor reading, sensor fault
    re.compile(r"\balarm\s+on\b"),           # alarm on machine
    re.compile(r"\bwhy\s+did\s+.{0,40}(stop|slow|trip|fail|alarm)\b"),
]

OUT_OF_SCOPE_MESSAGE = (
    "I'm a manufacturing operations assistant. I can only answer questions about "
    "machine anomalies, sensor readings, production logs, equipment faults, "
    "maintenance actions, and root-cause analysis.\n\n"
    "Examples of supported questions:\n"
    "  • Why is machine_1 running hot?\n"
    "  • What caused the vibration alarm on line 2?\n"
    "  • Is the pump on machine_3 running normally?\n"
    "  • What happened to line 1 at 15:30?"
)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    in_scope: bool
    confidence: float        # 0–1
    matched_term: str        # what triggered the decision (for logging)
    reason: str              # human-readable


# ── Classifier ────────────────────────────────────────────────────────────────

def classify_intent(question: str) -> IntentResult:
    """
    Classify whether a question is within manufacturing scope.

    Returns IntentResult.  Check `.in_scope` before running the pipeline.
    """
    q = question.lower().strip()

    # 1. Keyword check (word-boundary aware)
    for kw in MANUFACTURING_KEYWORDS:
        # Multi-word keywords: substring match is fine (already specific enough)
        if " " in kw:
            if kw in q:
                return IntentResult(
                    in_scope=True,
                    confidence=1.0,
                    matched_term=kw,
                    reason=f"Matched keyword phrase: '{kw}'",
                )
        else:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, q):
                return IntentResult(
                    in_scope=True,
                    confidence=1.0,
                    matched_term=kw,
                    reason=f"Matched keyword: '{kw}'",
                )

    # 2. Regex pattern check
    for pat in _PATTERNS:
        m = pat.search(q)
        if m:
            return IntentResult(
                in_scope=True,
                confidence=0.8,
                matched_term=m.group(0),
                reason=f"Matched pattern: '{pat.pattern}'",
            )

    # 3. No match → out of scope
    return IntentResult(
        in_scope=False,
        confidence=1.0,
        matched_term="",
        reason="No manufacturing-related terms detected",
    )


class OutOfScopeError(ValueError):
    """Raised when a question is outside the supported manufacturing scope."""

    def __init__(self, question: str, result: IntentResult):
        self.question = question
        self.result = result
        super().__init__(OUT_OF_SCOPE_MESSAGE)
