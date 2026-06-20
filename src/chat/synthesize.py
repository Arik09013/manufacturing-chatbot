"""
LLM synthesis layer.

Takes the structured pipeline output payload and calls Claude to produce
a plain-language operator response.

To use a local model instead (Ollama), swap the _call_claude() function
for an equivalent _call_ollama() and set SYNTHESIZER_BACKEND=ollama in .env.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.chat.prompts import (
    SYSTEM_PROMPT,
    PARAM_SYSTEM_PROMPT,
    KNOWLEDGE_SYSTEM_PROMPT,
    INTEGRATION_SYSTEM_PROMPT,
    GENERAL_MANUFACTURING_SYSTEM_PROMPT,
    build_synthesis_prompt,
    build_param_prompt,
    build_knowledge_prompt,
    build_general_prompt,
)

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
# The general-manufacturing route has no deterministic grounding — the model IS
# the knowledge source — so it defaults to a more capable model. Overridable via
# ANTHROPIC_GENERAL_MODEL.
_DEFAULT_GENERAL_MODEL = "claude-opus-4-8"
_DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
_MAX_TOKENS = 500
_INTEGRATION_MAX_TOKENS = 4000  # headroom for long, comprehensive build-out answers (not a target — length follows the question)
_GENERAL_MAX_TOKENS = 900       # general advisory answers run a little longer than narration
_BACKEND = os.getenv("SYNTHESIZER_BACKEND", "anthropic").lower()


def _resolve_backend() -> str:
    """
    Pick the synthesis backend. Honours SYNTHESIZER_BACKEND, but if that backend
    has no usable credentials, fall back through the chain so a demo still runs.
    """
    if _BACKEND == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        if os.getenv("GROQ_API_KEY"):
            logger.info("ANTHROPIC_API_KEY missing; using Groq backend")
            return "groq"
    return _BACKEND


def synthesize(payload: dict) -> str:
    """
    Convert a structured pipeline result dict to a plain-language answer.

    Parameters
    ----------
    payload : dict
        Output from src.api.pipeline.run_pipeline(), containing:
        question, machine_id, is_anomaly, anomaly_type, anomaly_prob,
        confidence, causes, recommendation, shap_drivers

    Returns
    -------
    str — operator-facing explanation
    """
    pipeline_type = payload.get("pipeline_type")
    max_tokens = _MAX_TOKENS
    model = None  # None → backend default (ANTHROPIC_MODEL); set per-route below
    if pipeline_type == "parameter_advice":
        user_content = build_param_prompt(payload)
        system_prompt = PARAM_SYSTEM_PROMPT
    elif pipeline_type == "general_manufacturing":
        # No deterministic grounding — the LLM answers from general manufacturing
        # knowledge, so use a more capable model and give it a little more room.
        user_content = build_general_prompt(payload)
        system_prompt = GENERAL_MANUFACTURING_SYSTEM_PROMPT
        max_tokens = _GENERAL_MAX_TOKENS
        model = os.getenv("ANTHROPIC_GENERAL_MODEL", _DEFAULT_GENERAL_MODEL)
    elif pipeline_type == "knowledge_advice":
        user_content = build_knowledge_prompt(payload)
        # Same retrieval/narration path, but swap the persona for the
        # simulation & robotics-integration domain (Isaac Sim, sensors, teleop, DAQ).
        if payload.get("knowledge_domain") == "integration":
            system_prompt = INTEGRATION_SYSTEM_PROMPT
            max_tokens = _INTEGRATION_MAX_TOKENS  # phased answers run longer
        else:
            system_prompt = KNOWLEDGE_SYSTEM_PROMPT
    else:
        user_content = build_synthesis_prompt(payload)
        system_prompt = SYSTEM_PROMPT

    backend = _resolve_backend()
    if backend == "anthropic":
        return _call_claude(user_content, system_prompt, max_tokens, model)
    elif backend == "groq":
        return _call_groq(user_content, system_prompt, max_tokens)
    elif backend == "ollama":
        return _call_ollama(user_content, system_prompt, max_tokens)
    else:
        logger.warning("Unknown backend %r; returning structured summary", backend)
        return _fallback_text(payload)


def _call_claude(user_content: str, system_prompt: str = SYSTEM_PROMPT,
                 max_tokens: int = _MAX_TOKENS, model: str | None = None) -> str:
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed; falling back to text summary")
        return user_content

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; returning structured text")
        return user_content

    client = anthropic.Anthropic(api_key=api_key)
    # Per-route model override (e.g. the general-manufacturing route) wins;
    # otherwise honour ANTHROPIC_MODEL, then the narration default.
    model = model or os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODEL)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    return message.content[0].text


def _call_groq(user_content: str, system_prompt: str = SYSTEM_PROMPT,
               max_tokens: int = _MAX_TOKENS) -> str:
    """Call the Groq API (fast hosted Llama models, OpenAI-compatible chat)."""
    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed (pip install groq); returning text summary")
        return user_content

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set; returning structured text")
        return user_content

    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_MODEL", _DEFAULT_GROQ_MODEL)

    completion = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return completion.choices[0].message.content


def _call_ollama(user_content: str, system_prompt: str = SYSTEM_PROMPT,
                 max_tokens: int = _MAX_TOKENS) -> str:
    """Call a locally running Ollama instance."""
    import urllib.request
    import json

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")

    payload_json = json.dumps({
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }).encode()

    req = urllib.request.Request(
        ollama_url,
        data=payload_json,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        return data["message"]["content"]


def _rag_fallback_text(passages: list[dict]) -> str:
    """Deterministic answer assembled from RAG-retrieved passages (no LLM)."""
    blocks = ["Based on the retrieved welding references:"]
    for p in passages:
        blocks.append(
            f"**[{p.get('cite', '?')}] {p.get('title', '')}** "
            f"(source: {p.get('source', '')})\n{p.get('text', '')}"
        )
    return "\n\n".join(blocks)


def _fallback_text(payload: dict) -> str:
    """Plain-text fallback when no LLM backend is available."""
    if payload.get("pipeline_type") == "parameter_advice":
        return payload.get("summary_text") or payload.get("error", "No recommendation available.")

    if payload.get("pipeline_type") == "knowledge_advice":
        # Prefer the curated summary when keyword topics matched; otherwise build a
        # plain-text answer from the RAG-retrieved passages so the fallback still
        # carries (and cites) the grounding.
        if payload.get("matched_topics") and payload.get("summary_text"):
            return payload["summary_text"]
        passages = payload.get("rag_passages")
        if passages:
            return _rag_fallback_text(passages)
        return payload.get("summary_text") or "No knowledge-base entry matched that question."

    if payload.get("pipeline_type") == "general_manufacturing":
        return (
            "That's a general manufacturing question outside the welding pipeline. "
            "Set an LLM backend (ANTHROPIC_API_KEY) to get a full answer."
        )

    if not payload.get("is_anomaly"):
        return (
            f"Machine {payload.get('machine_id')} appears to be operating normally "
            f"(anomaly probability: {payload.get('anomaly_prob', 0):.0%})."
        )
    rec = payload.get("recommendation", {})
    conf = payload.get("confidence", {})
    causes = payload.get("causes", [])
    cause_str = causes[0]["cause"] if causes else "unknown"
    return (
        f"ANOMALY DETECTED on {payload.get('machine_id')} "
        f"[{payload.get('anomaly_type', 'unknown')}] — "
        f"confidence: {conf.get('band', 'unknown')} ({conf.get('score', 0):.0%}).\n"
        f"Likely cause: {cause_str}\n"
        f"Action: {rec.get('primary', 'Inspect machine')}\n"
        f"Urgency: {rec.get('urgency', 'medium')}"
    )
