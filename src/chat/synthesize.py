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

from src.chat.prompts import SYSTEM_PROMPT, build_synthesis_prompt

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 400
_BACKEND = os.getenv("SYNTHESIZER_BACKEND", "anthropic").lower()


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
    user_content = build_synthesis_prompt(payload)

    if _BACKEND == "anthropic":
        return _call_claude(user_content)
    elif _BACKEND == "ollama":
        return _call_ollama(user_content)
    else:
        logger.warning("Unknown backend %r; returning structured summary", _BACKEND)
        return _fallback_text(payload)


def _call_claude(user_content: str) -> str:
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
    model = os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODEL)

    message = client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return message.content[0].text


def _call_ollama(user_content: str) -> str:
    """Call a locally running Ollama instance."""
    import urllib.request
    import json

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")

    payload_json = json.dumps({
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"num_predict": _MAX_TOKENS},
    }).encode()

    req = urllib.request.Request(
        ollama_url,
        data=payload_json,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        return data["message"]["content"]


def _fallback_text(payload: dict) -> str:
    """Plain-text fallback when no LLM backend is available."""
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
