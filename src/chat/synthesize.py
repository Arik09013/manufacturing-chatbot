"""
LLM synthesis layer.

Takes the structured pipeline output payload and calls the configured LLM
backend to produce a plain-language operator response.

The concrete backends (Claude / Groq / Ollama-llama / Ollama-qwen /
Ollama-mistral) live in src.chat.backends and all expose the same interface.
Pick one with SYNTHESIZER_BACKEND (env or config/llm.yaml). On any backend
failure this layer returns the deterministic grounded fallback so the operator
never sees a crash.
"""

from __future__ import annotations

import logging
import os
import time
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
from src.chat.backends import (
    BackendError,
    get_backend,
    resolve_backend_name,
    backend_model,
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


def _resolve_backend() -> str:
    """
    Pick the synthesis backend. Honours SYNTHESIZER_BACKEND (via config/llm.yaml
    resolution), but if the Anthropic backend has no usable credentials while a
    Groq key is present, fall back so a demo still runs.
    """
    name = resolve_backend_name()
    if name in ("anthropic", "claude") and not os.getenv("ANTHROPIC_API_KEY"):
        if os.getenv("GROQ_API_KEY"):
            logger.info("ANTHROPIC_API_KEY missing; using Groq backend")
            return "groq"
    return name


def _build_prompt(payload: dict) -> tuple[str, str, int, str | None]:
    """
    Turn a pipeline payload into (user_content, system_prompt, max_tokens, model).
    `model` is a per-route Anthropic override (or None → backend default).
    Shared by synthesize() and the benchmark harness so every backend sees the
    exact same prompt for a given payload.
    """
    pipeline_type = payload.get("pipeline_type")
    max_tokens = _MAX_TOKENS
    model = None  # None → backend default; set per-route below
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
    return user_content, system_prompt, max_tokens, model


def synthesize(payload: dict, backend: str | None = None) -> str:
    """
    Convert a structured pipeline result dict to a plain-language answer.

    Parameters
    ----------
    payload : dict
        Output from src.api.pipeline.run_pipeline(), containing:
        question, machine_id, is_anomaly, anomaly_type, anomaly_prob,
        confidence, causes, recommendation, shap_drivers
    backend : str | None
        Force a specific backend (e.g. "ollama_qwen"). Defaults to the
        configured SYNTHESIZER_BACKEND.

    Returns
    -------
    str — operator-facing explanation
    """
    return generate_response(payload, backend=backend)["text"]


def generate_response(payload: dict, backend: str | None = None) -> dict:
    """
    Backend-aware synthesis with metadata — used by both synthesize() and the
    benchmark harness. Never raises: on any backend failure it returns the
    deterministic grounded fallback and records what happened.

    Returns dict:
        text        — the answer string (LLM output or grounded fallback)
        backend     — resolved backend name
        model       — model id used (best-effort)
        latency_s   — wall-clock seconds for the backend call
        ok          — True if the LLM produced the text, False if we fell back
        fell_back   — True if the deterministic fallback was used
        error       — error string when fell_back, else None
    """
    user_content, system_prompt, max_tokens, model = _build_prompt(payload)
    backend_name = resolve_backend_name(backend) if backend else _resolve_backend()

    # The per-route `model` override (e.g. claude-opus-4-8 for the general route)
    # is Anthropic-specific — it must NOT leak into Groq/Ollama, which have their
    # own configured model and would 404 on a Claude model id.
    route_model = model if backend_name in ("anthropic", "claude") else None

    t0 = time.perf_counter()
    try:
        be = get_backend(backend_name)
        text = be.generate(user_content, system_prompt, max_tokens, route_model)
        latency = time.perf_counter() - t0
        if not text or not str(text).strip():
            raise BackendError("empty response")
        return {
            "text": text,
            "backend": backend_name,
            "model": route_model or be.model,
            "latency_s": round(latency, 3),
            "ok": True,
            "fell_back": False,
            "error": None,
        }
    except Exception as exc:  # BackendError or anything unexpected
        latency = time.perf_counter() - t0
        # A local backend (Ollama) may be stopped, slow, or mid-download; rather
        # than surfacing a 500 to the operator, fall back to the deterministic
        # grounded summary (which still carries the computed result / cited RAG
        # passages).
        logger.warning("LLM backend %r failed; returning deterministic fallback: %s",
                       backend_name, exc)
        return {
            "text": _fallback_text(payload),
            "backend": backend_name,
            "model": route_model or backend_model(backend_name),
            "latency_s": round(latency, 3),
            "ok": False,
            "fell_back": True,
            "error": str(exc),
        }


# ── Backward-compatible thin wrappers (kept for any external callers) ──────────

def _call_claude(user_content: str, system_prompt: str = SYSTEM_PROMPT,
                 max_tokens: int = _MAX_TOKENS, model: str | None = None) -> str:
    try:
        return get_backend("anthropic").generate(
            user_content, system_prompt, max_tokens, model)
    except BackendError:
        logger.warning("Anthropic backend unavailable; returning structured text")
        return user_content


def _call_groq(user_content: str, system_prompt: str = SYSTEM_PROMPT,
               max_tokens: int = _MAX_TOKENS) -> str:
    """Call the Groq API (fast hosted Llama models, OpenAI-compatible chat)."""
    try:
        return get_backend("groq").generate(user_content, system_prompt, max_tokens)
    except BackendError:
        logger.warning("Groq backend unavailable; returning structured text")
        return user_content


def _call_ollama(user_content: str, system_prompt: str = SYSTEM_PROMPT,
                 max_tokens: int = _MAX_TOKENS) -> str:
    """Call a locally running Ollama instance (generic model from OLLAMA_MODEL)."""
    return get_backend("ollama").generate(user_content, system_prompt, max_tokens)


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
