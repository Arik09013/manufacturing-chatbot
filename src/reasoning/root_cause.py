"""
Root-cause mapper: given an anomaly type and its SHAP drivers, return a
ranked list of likely causes with supporting evidence.

Uses the heuristic config/cause_action_map.yaml (T7 MVP approach).
Learned ranking (MRR) is deferred to a later phase.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "cause_action_map.yaml"


@lru_cache(maxsize=1)
def _load_map() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def identify_causes(
    anomaly_type: str,
    shap_drivers: list[dict],
    log_event_codes: str = "",
    top_n: int = 3,
) -> list[dict]:
    """
    Return a ranked list of likely root causes.

    Parameters
    ----------
    anomaly_type   : one of the five anomaly types (or "unknown")
    shap_drivers   : list of dicts from AnomalyExplainer.explain_row()
                     each has keys: feature, shap, direction, magnitude
    log_event_codes: comma-separated event codes in the anomaly window
    top_n          : max causes to return

    Returns
    -------
    List of cause dicts with keys: rank, cause, evidence, shap_support
    """
    config = _load_map()
    atype = anomaly_type if anomaly_type in config else "unknown"
    cause_map = config[atype].get("causes", [])

    # Positive SHAP drivers (toward anomaly), sorted by magnitude
    positive_drivers = sorted(
        [d for d in shap_drivers if d["direction"] == "toward_anomaly"],
        key=lambda x: x["magnitude"],
        reverse=True,
    )
    top_driver_features = [d["feature"] for d in positive_drivers]

    # Score each cause entry by checking if any positive driver matches the pattern
    scored = []
    for entry in cause_map:
        if isinstance(entry, dict):
            pattern, cause_text = list(entry.items())[0]
        else:
            pattern, cause_text = "", str(entry)

        # Find matching SHAP drivers
        matching = [f for f in top_driver_features if pattern in f]
        score = len(matching)

        # Bonus if there's a correlated log event
        if log_event_codes and pattern.upper()[:4] in log_event_codes:
            score += 1

        scored.append({
            "cause":       cause_text,
            "shap_support": matching[:2],
            "score":       score,
        })

    # Sort by score descending, then preserve config order
    scored.sort(key=lambda x: x["score"], reverse=True)

    causes = []
    for rank, item in enumerate(scored[:top_n], start=1):
        causes.append({
            "rank":        rank,
            "cause":       item["cause"],
            "shap_support": item["shap_support"],
            "evidence_strength": _band(item["score"]),
        })

    # If no cause matched, return a generic entry
    if not causes:
        causes = [{
            "rank": 1,
            "cause": "Unusual sensor pattern — no specific cause matched heuristic rules",
            "shap_support": top_driver_features[:2],
            "evidence_strength": "low",
        }]

    return causes


def _band(score: int) -> str:
    if score >= 2:
        return "high"
    if score == 1:
        return "medium"
    return "low"
