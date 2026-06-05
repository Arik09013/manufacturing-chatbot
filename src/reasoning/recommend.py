"""
Recommendation module: given anomaly type and causes, return a
primary + secondary action with urgency level.
"""

from __future__ import annotations

from src.reasoning.root_cause import _load_map


def get_recommendation(anomaly_type: str) -> dict:
    """
    Return action recommendations for an anomaly type.

    Returns
    -------
    dict with keys: primary, secondary, urgency
    """
    config = _load_map()
    atype = anomaly_type if anomaly_type in config else "unknown"
    entry = config[atype]

    return {
        "primary":   entry["actions"]["primary"],
        "secondary": entry["actions"]["secondary"],
        "urgency":   entry.get("urgency", "medium"),
    }
