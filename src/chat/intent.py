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
    # ── Welding-specific ──
    "weld", "welding", "welder", "weldment",
    "arc", "arc voltage", "arc instability", "arc loss",
    "wire feed", "wire feed rate", "wire feeder", "wire spool",
    "contact tip", "liner", "drive roll", "torch",
    "shielding gas", "gas flow", "gas cylinder", "solenoid",
    "heat input", "penetration", "fusion", "burn-through", "porosity",
    "spatter", "undercut", "distortion", "bead", "weld bead",
    "mig", "mag", "tig", "gmaw", "gtaw", "fcaw",
    "deposition", "deposition rate",
    "welding speed", "travel speed", "welding current",
    "mild steel", "stainless", "aluminum", "aluminium",
    "weld quality", "weld procedure", "wps",
    "station", "station_1", "station_2", "station_3",
    # ── General manufacturing / equipment ──
    "machine", "motor", "pump", "valve", "bearing", "spindle",
    "compressor", "actuator", "conveyor", "robot", "gearbox",
    "sensor", "temperature", "vibration", "pressure", "rpm",
    "torque", "flow rate", "current", "voltage",
    # ── Anomaly / fault vocabulary ──
    "anomaly", "anomalies", "fault", "failure", "alarm", "alert",
    "warning", "overload", "overheating", "overheat", "trip",
    "diagnostic", "underheat",
    # ── Operations ──
    "production", "manufacturing", "factory", "plant",
    "maintenance", "downtime", "startup", "shutdown",
    "operator", "shift", "batch", "setpoint", "throughput",
    # ── Parameter queries ──
    "best setting", "optimal setting", "recommend setting",
    "best speed", "best current", "best parameter", "optimal parameter",
    "what speed", "what current", "what voltage", "how fast",
    "efficiency",
    # ── Pipeline concepts ──
    "root cause", "root-cause", "recommendation",
})

# Regex patterns that strongly signal manufacturing context even if the exact
# keyword isn't in the vocabulary (e.g. "line 3", "machine_2").
_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bmachine[_\s]\d\b"),
    re.compile(r"\bstation[_\s]\d\b"),       # station_1, station 2
    re.compile(r"\bline[_\s]\d\b"),
    re.compile(r"\b(m1|m2|m3|s1|s2|s3)\b"),
    re.compile(r"\bsensor\s+\w+\b"),
    re.compile(r"\balarm\s+on\b"),
    re.compile(r"\bwhy\s+did\s+.{0,40}(stop|slow|trip|fail|alarm)\b"),
    re.compile(r"\d+\s*mm\b"),              # "5mm", "10 mm" → thickness query
    re.compile(r"\bfor\s+\d+\s*mm\b"),      # "for 5mm steel"
]

OUT_OF_SCOPE_MESSAGE = (
    "I'm a welding operations AI assistant. I can only answer questions about "
    "welding anomalies, parameter optimisation, equipment faults, sensor readings, "
    "and root-cause analysis.\n\n"
    "Examples of supported questions:\n"
    "  • Why did station_1 stop welding at 14:00?\n"
    "  • What caused the arc instability alarm on station_2?\n"
    "  • Best settings for 5mm mild steel?\n"
    "  • What is the optimal welding speed for 3mm aluminum?\n"
    "  • Is station_3 running within normal parameters?"
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
