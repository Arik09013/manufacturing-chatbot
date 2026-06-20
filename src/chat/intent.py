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
    # ── Defects / troubleshooting / quality vocabulary ──
    "crack", "cracking", "cracked", "warping", "warp", "warped", "buckling",
    "lack of fusion", "incomplete fusion", "lack of penetration",
    "discoloration", "discolouration", "heat tint", "sugaring",
    "tensile", "tensile strength", "weld strength", "weld quality",
    "consumable", "consumables", "rework", "spray transfer", "spray", "pulse",
    "pulsed", "transfer mode", "shielding gas", "gas mix", "gas consumption",
    "cost", "productivity", "deposition rate", "duty cycle", "filler",
    "mig", "mag", "tig", "smaw", "gmaw", "gtaw", "fcaw", "mma",
    "stick welding", "stick electrode", "manual metal arc",
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
    # ── General manufacturing processes (non-welding) — answered by the
    #    general-manufacturing LLM advisor, not the welding pipeline ──
    "machining", "cnc", "milling", "mill", "lathe", "turning", "drilling",
    "boring", "grinding", "spindle speed", "feed rate", "cutting speed",
    "end mill", "tooling", "tool wear", "chatter", "swarf", "coolant",
    "injection molding", "injection moulding", "molding", "moulding",
    "mold", "mould", "sink mark", "sink marks", "short shot", "flash",
    "warpage", "shrinkage", "gate", "runner", "sprue", "clamp tonnage",
    "thermoplastic", "abs", "polypropylene", "polymer", "resin", "plastic",
    "casting", "die casting", "sand casting", "investment casting",
    "forging", "extrusion", "extrude", "rolling", "drawing",
    "sheet metal", "stamping", "press brake", "bending", "springback",
    "blanking", "punching", "deep drawing", "tonnage",
    "additive", "additive manufacturing", "3d print", "3d printing",
    "3d printed", "fdm", "fff", "sla", "sls", "dmls", "slicing", "infill",
    "machine tool", "fixture", "jig", "workholding", "clamp",
    "heat treatment", "annealing", "quenching", "tempering", "hardening",
    "case hardening", "carburizing", "normalizing",
    "surface finish", "roughness", "ra", "plating", "anodizing", "coating",
    "deburring", "polishing", "honing",
    "tolerance", "tolerances", "gd&t", "gdt", "metrology", "cmm",
    "inspection", "spc", "process capability", "cpk", "ppk",
    "six sigma", "lean", "kaizen", "5s", "kanban", "poka yoke",
    "oee", "overall equipment effectiveness", "takt", "takt time",
    "cycle time", "lead time", "bottleneck", "scrap", "yield",
    "assembly", "assembly line", "bom", "bill of materials",
    "cad", "cam", "cnc program", "g-code", "gcode",
    "material", "alloy", "carbon steel", "titanium", "copper", "brass",
    "composite", "ceramic", "tooling cost", "machining cost",
    # ── Parameter queries ──
    "best setting", "optimal setting", "recommend setting",
    "best speed", "best current", "best parameter", "optimal parameter",
    "what speed", "what current", "what voltage", "how fast",
    "efficiency",
    # parameter / consumable vocabulary (so "what's the parameter if I use a
    # 1.5 wire diameter instead of 1.2" is recognised as in-scope)
    "parameter", "parameters", "setting", "settings",
    "wire diameter", "wire dia", "wire size", "electrode diameter",
    "tungsten", "tungsten diameter", "filler", "filler wire", "diameter",
    # ── Pipeline concepts ──
    "root cause", "root-cause", "recommendation",
    # ── Simulation & robotics integration (Isaac Sim domain) ──
    "isaac sim", "isaac", "isaac lab", "omniverse", "nvidia isaac",
    "simulation", "simulate", "simulated", "simulator", "digital twin",
    "manipulator", "end effector", "end-effector", "articulation",
    "urdf", "usd", "rmpflow", "motion controller", "trajectory",
    "depth camera", "rgbd", "rgb-d", "rgb camera", "camera", "point cloud",
    "lidar", "lidar sensor", "force sensor", "torque sensor",
    "force torque", "force/torque", "f/t sensor", "6-axis", "wrist sensor",
    "teleop", "teleoperation", "tele-operation", "vision pro", "apple vision pro",
    "cloudxr", "cloud xr", "headset", "hand tracking", "retarget", "retargeting",
    "ros", "ros2", "ros 2", "omnigraph", "action graph", "topic", "bridge",
    "daq", "data acquisition", "ethercat", "ni cdaq", "ptp", "time sync",
    "realsense", "ouster", "ati", "robotiq", "wrist camera", "seam tracking",
    "scene", "workpiece", "fixture", "weld cell", "welding cell", "gr00t",
    # ── Kinematics / control / planning ──
    "kinematics", "inverse kinematics", "forward kinematics", "ik solver",
    "motion planning", "trajectory planning", "path planning", "moveit",
    "cumotion", "isaac manipulator", "lula", "collision avoidance",
    "standoff", "touch sensing", "seam finding", "force control",
    "waypoint", "waypoints",
    # ── Sim realism / faults / sim-to-real ──
    "weld pool", "weld puddle", "melt pool", "fault injection",
    "sim to real", "sim-to-real", "sim2real", "reality gap",
    "domain randomization", "domain randomisation",
    # ── In-sim UI / visualization ──
    "omni.ui", "viewport", "spatial ui", "overlay", "point cloud overlay",
    # ── Explainability / thesis framing (integration context) ──
    "explainability", "interpretability", "novel contribution", "novelty",
    "research contribution", "future work", "reality",
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
    "I'm a manufacturing AI assistant. I answer questions about manufacturing and "
    "industrial engineering — welding (my deepest specialism), plus machining, molding, "
    "casting, forming, additive manufacturing, quality, maintenance, and operations.\n\n"
    "Examples of supported questions:\n"
    "  • Why did station_1 stop welding at 14:00?\n"
    "  • Best settings for 5mm mild steel?\n"
    "  • What spindle speed and feed for CNC milling 6061 aluminium?\n"
    "  • How do I reduce sink marks in injection-moulded ABS parts?\n"
    "  • How do I calculate OEE for my production line?"
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
