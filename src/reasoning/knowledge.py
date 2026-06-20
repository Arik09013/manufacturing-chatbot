"""
Welding knowledge retriever.

Matches a free-text welding question (defect, troubleshooting, quality, cost,
productivity, or process-comparison) against the curated knowledge base in
``config/welding_knowledge.yaml`` and returns the most relevant entries.

This is the grounding layer for the "general welding advice" route: the LLM
narrates ONLY the causes / remedies in the matched entries — it does not invent
new facts. If nothing matches, ``match_knowledge`` returns an empty list and the
caller falls back to a general advisory answer.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG = Path(__file__).parent.parent.parent / "config" / "welding_knowledge.yaml"

_WORD_RE = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=1)
def _load_kb() -> dict:
    with open(_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("topics", {})


def _score(question: str, phrases: list[str]) -> tuple[float, str]:
    """
    Score how strongly a (lower-cased) question matches an entry's trigger phrases.

    A multi-word phrase that appears as a substring scores higher than a single
    word, so "lack of fusion" beats an incidental "fusion". Returns the best
    (score, matched_phrase).
    """
    best_score = 0.0
    best_phrase = ""
    for phrase in phrases:
        p = phrase.lower()
        if p in question:
            # weight by number of words in the phrase → specificity
            weight = 1.0 + 0.5 * (p.count(" "))
            if weight > best_score:
                best_score = weight
                best_phrase = phrase
    return best_score, best_phrase


def match_knowledge(question: str, top_n: int = 2, min_score: float = 1.0) -> list[dict]:
    """
    Return up to ``top_n`` knowledge-base entries most relevant to ``question``.

    Each returned dict is the YAML entry plus ``topic`` and ``matched_phrase``
    keys. Empty list means no entry matched (caller should fall back to a
    general advisory answer).
    """
    q = question.lower()
    kb = _load_kb()

    scored = []
    for topic, entry in kb.items():
        score, phrase = _score(q, entry.get("match", []))
        if score >= min_score:
            scored.append((score, topic, phrase, entry))

    scored.sort(key=lambda t: t[0], reverse=True)

    results = []
    for score, topic, phrase, entry in scored[:top_n]:
        out = {
            "topic":          topic,
            "matched_phrase": phrase,
            "score":          score,
            "category":       entry.get("category", ""),
            "summary":        entry.get("summary", ""),
            "causes":         entry.get("causes", []) or [],
            "remedies":       entry.get("remedies", []) or [],
            "parameters":     entry.get("parameters", []) or [],
            "notes":          entry.get("notes", ""),
        }
        results.append(out)
    return results


def build_knowledge_summary(question: str, topics: list[dict]) -> str:
    """
    Deterministic plain-text answer assembled directly from matched entries.
    Used as the non-LLM fallback (no API key / backend error).
    """
    if not topics:
        return (
            "I don't have a curated knowledge-base entry for that exact question. "
            "Try rephrasing toward a specific welding defect, parameter, or process "
            "(e.g. porosity, undercut, shielding gas choice, spray vs pulse MIG), "
            "or ask the parameter advisor for settings on a material + thickness."
        )

    blocks = []
    for t in topics:
        lines = [f"### {t['topic'].replace('_', ' ').title()}", t["summary"]]
        if t["causes"]:
            lines.append("\n**Likely causes:**")
            lines += [f"- {c}" for c in t["causes"]]
        if t["remedies"]:
            lines.append("\n**Corrective actions:**")
            lines += [f"- {r}" for r in t["remedies"]]
        if t["notes"]:
            lines.append(f"\n_Note: {t['notes']}_")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
