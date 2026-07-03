"""
Tests for the pluggable LLM backend abstraction (src/chat/backends.py) and the
backend-aware synthesis layer. These stay offline: no live API/Ollama call is
made — we only exercise resolution, validation, and the graceful-fallback path.
"""

from __future__ import annotations

import pytest

from src.chat import backends as B
from src.chat.synthesize import synthesize, generate_response


def test_known_backends_include_qwen_and_mistral():
    assert "ollama_qwen" in B.KNOWN_BACKENDS
    assert "ollama_mistral" in B.KNOWN_BACKENDS
    assert "ollama_llama" in B.KNOWN_BACKENDS


def test_backend_model_resolution():
    assert B.backend_model("ollama_qwen") == "qwen2.5:3b"
    assert B.backend_model("ollama_mistral") == "mistral:7b"


def test_get_backend_builds_expected_provider():
    assert isinstance(B.get_backend("ollama_qwen"), B.OllamaBackend)
    assert isinstance(B.get_backend("groq"), B.GroqBackend)
    assert isinstance(B.get_backend("anthropic"), B.AnthropicBackend)


def test_named_ollama_variants_are_pinned(monkeypatch):
    # OLLAMA_MODEL must NOT rename the pinned comparison variants.
    monkeypatch.setenv("OLLAMA_MODEL", "some-other-model")
    qwen = B.get_backend("ollama_qwen")
    assert qwen._resolve_model(None) == "qwen2.5:3b"
    # ...but it DOES steer the generic 'ollama' backend.
    generic = B.get_backend("ollama")
    assert generic._resolve_model(None) == "some-other-model"


def test_unknown_backend_raises():
    with pytest.raises(B.BackendError):
        B.get_backend("totally_not_a_backend")


def test_resolve_backend_name_explicit_wins():
    assert B.resolve_backend_name("ollama_mistral") == "ollama_mistral"


def test_probe_returns_tuple():
    ok, reason = B.probe_backend("ollama_qwen")
    assert isinstance(ok, bool)
    assert isinstance(reason, str) and reason


def test_generate_response_falls_back_on_invalid_backend():
    payload = {"pipeline_type": "parameter_advice",
               "summary_text": "FALLBACK SUMMARY", "question": "q"}
    res = generate_response(payload, backend="totally_not_a_backend")
    assert res["fell_back"] is True
    assert res["ok"] is False
    assert res["error"]
    assert res["text"] == "FALLBACK SUMMARY"


def test_synthesize_preserves_string_return():
    payload = {"pipeline_type": "parameter_advice",
               "summary_text": "FALLBACK SUMMARY", "question": "q"}
    out = synthesize(payload, backend="totally_not_a_backend")
    assert isinstance(out, str)
    assert "FALLBACK SUMMARY" in out


def test_general_route_model_override_does_not_leak_to_other_backends(monkeypatch):
    # The general route sets an Anthropic-specific model (claude-opus-4-8). It must
    # NOT be handed to Groq/Ollama, which would 404 on a Claude model id.
    captured = {}

    def fake_generate(self, user_content, system_prompt, max_tokens, model=None):
        captured["model"] = model
        return "ok"

    monkeypatch.setattr(B.OllamaBackend, "generate", fake_generate, raising=True)
    payload = {"pipeline_type": "general_manufacturing", "question": "What is OEE?"}
    res = generate_response(payload, backend="ollama_qwen")
    assert captured["model"] is None            # override did not leak
    assert res["model"] == "qwen2.5:3b"          # reports the real model used


def test_config_yaml_loads():
    cfg = B.load_llm_config()
    assert "backends" in cfg
    assert cfg["backends"]["ollama_qwen"]["model"] == "qwen2.5:3b"
