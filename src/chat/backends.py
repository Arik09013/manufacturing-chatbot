"""
Pluggable LLM synthesis backends.

Every backend exposes the SAME interface so the synthesis layer (and the
benchmark harness) can treat them interchangeably:

    backend.generate(user_content, system_prompt, max_tokens, model=None) -> str
    backend.synthesize(...)   # alias of generate()
    backend.chat(...)         # alias of generate()
    backend.stream(...)       # yields chunks (falls back to a single chunk)

Supported SYNTHESIZER_BACKEND values
-------------------------------------
    anthropic / claude   -> Anthropic Claude (hosted API)
    groq                 -> Groq (hosted, OpenAI-compatible Llama)
    ollama               -> local Ollama, model from OLLAMA_MODEL
    ollama_llama         -> local Ollama, llama3.2 (baseline)
    ollama_qwen          -> local Ollama, qwen2.5:3b
    ollama_mistral       -> local Ollama, mistral:7b

Configuration precedence (highest first):
    1. explicit argument / environment variable (OLLAMA_MODEL, ANTHROPIC_MODEL, …)
    2. config/llm.yaml  (backends.<name>.model)
    3. built-in default baked in below

Design contract: a backend NEVER silently returns a wrong answer. On any
failure (package missing, model not pulled, Ollama down, timeout, context
overflow, bad backend name) it raises BackendError; the caller
(src.chat.synthesize) turns that into the deterministic grounded fallback so
the operator never sees a 500.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent
_LLM_CONFIG_PATH = _REPO_ROOT / "config" / "llm.yaml"

# Built-in defaults (used when neither env nor config/llm.yaml specify a model).
_DEFAULTS = {
    "anthropic":      {"provider": "anthropic", "model": "claude-haiku-4-5-20251001",
                       "general_model": "claude-opus-4-8"},
    "claude":         {"provider": "anthropic", "model": "claude-haiku-4-5-20251001",
                       "general_model": "claude-opus-4-8"},
    "groq":           {"provider": "groq",   "model": "llama-3.3-70b-versatile"},
    "ollama":         {"provider": "ollama", "model": "llama3.2"},
    "ollama_llama":   {"provider": "ollama", "model": "llama3.2"},
    "ollama_qwen":    {"provider": "ollama", "model": "qwen2.5:3b"},
    "ollama_mistral": {"provider": "ollama", "model": "mistral:7b"},
}

# Canonical set of backend names the system understands.
KNOWN_BACKENDS = tuple(_DEFAULTS.keys())

_DEFAULT_MAX_TOKENS = 500


class BackendError(RuntimeError):
    """Raised when a backend cannot produce a response (any reason)."""


# ── Config loading ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_llm_config() -> dict:
    """
    Load config/llm.yaml if present. Returns {} on any problem so the system
    still runs on the baked-in defaults. Shape:

        default_backend: anthropic
        backends:
          ollama_qwen: {provider: ollama, model: qwen2.5:3b}
          ...
    """
    if not _LLM_CONFIG_PATH.exists():
        return {}
    try:
        import yaml
        with open(_LLM_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("Could not parse %s; using built-in LLM defaults",
                       _LLM_CONFIG_PATH, exc_info=True)
        return {}


def _backend_config(name: str) -> dict:
    """Merge built-in defaults with any config/llm.yaml override for `name`."""
    cfg = dict(_DEFAULTS.get(name, {}))
    yaml_backends = load_llm_config().get("backends", {}) or {}
    if name in yaml_backends and isinstance(yaml_backends[name], dict):
        cfg.update(yaml_backends[name])
    return cfg


def resolve_backend_name(requested: str | None = None) -> str:
    """
    Resolve the active backend name from (in order):
      explicit arg -> SYNTHESIZER_BACKEND env -> llm.yaml default_backend -> anthropic.
    Unknown names are returned as-is so the caller can validate/fall back.
    """
    name = (requested
            or os.getenv("SYNTHESIZER_BACKEND")
            or load_llm_config().get("default_backend")
            or "anthropic")
    return str(name).strip().lower()


# ── Backend interface ─────────────────────────────────────────────────────────

class LLMBackend:
    """Common interface. Subclasses implement _generate()."""

    provider: str = "base"

    def __init__(self, name: str, config: dict | None = None):
        self.name = name
        self.config = config or _backend_config(name)
        self.model = self.config.get("model")

    # -- primary API --
    def generate(self, user_content: str, system_prompt: str,
                 max_tokens: int = _DEFAULT_MAX_TOKENS,
                 model: str | None = None) -> str:
        return self._generate(user_content, system_prompt, max_tokens,
                              model or self.model)

    # -- aliases (identical interface across backends) --
    def synthesize(self, user_content: str, system_prompt: str,
                   max_tokens: int = _DEFAULT_MAX_TOKENS,
                   model: str | None = None) -> str:
        return self.generate(user_content, system_prompt, max_tokens, model)

    def chat(self, user_content: str, system_prompt: str,
             max_tokens: int = _DEFAULT_MAX_TOKENS,
             model: str | None = None) -> str:
        return self.generate(user_content, system_prompt, max_tokens, model)

    def stream(self, user_content: str, system_prompt: str,
               max_tokens: int = _DEFAULT_MAX_TOKENS,
               model: str | None = None):
        """Default streaming: yield the whole answer as one chunk.

        Subclasses that support native token streaming override this.
        """
        yield self.generate(user_content, system_prompt, max_tokens, model)

    def _generate(self, user_content, system_prompt, max_tokens, model) -> str:
        raise NotImplementedError


class AnthropicBackend(LLMBackend):
    provider = "anthropic"

    def _generate(self, user_content, system_prompt, max_tokens, model) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise BackendError("anthropic package not installed") from exc

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise BackendError("ANTHROPIC_API_KEY not set")

        model = model or os.getenv("ANTHROPIC_MODEL") or self.model
        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            return message.content[0].text
        except Exception as exc:
            raise BackendError(f"Anthropic call failed: {exc}") from exc


class GroqBackend(LLMBackend):
    provider = "groq"

    def _generate(self, user_content, system_prompt, max_tokens, model) -> str:
        try:
            from groq import Groq
        except ImportError as exc:
            raise BackendError("groq package not installed") from exc

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise BackendError("GROQ_API_KEY not set")

        model = model or os.getenv("GROQ_MODEL") or self.model
        try:
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return completion.choices[0].message.content
        except Exception as exc:
            raise BackendError(f"Groq call failed: {exc}") from exc


class OllamaBackend(LLMBackend):
    """Local Ollama. Model resolution: explicit > OLLAMA_MODEL (only for the
    generic 'ollama' name) > per-variant config/default.

    The named variants (ollama_qwen/ollama_mistral/ollama_llama) are pinned to a
    specific model so a benchmark can compare them side by side even when
    OLLAMA_MODEL is set for interactive use.
    """
    provider = "ollama"

    def _resolve_model(self, model: str | None) -> str:
        if model:
            return model
        # OLLAMA_MODEL only overrides the generic 'ollama' backend, so it does
        # not silently rename the pinned comparison variants.
        if self.name == "ollama" and os.getenv("OLLAMA_MODEL"):
            return os.getenv("OLLAMA_MODEL")
        return self.model or "llama3.2"

    def _generate(self, user_content, system_prompt, max_tokens, model) -> str:
        import json
        import urllib.error
        import urllib.request

        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
        ollama_model = self._resolve_model(model)

        body = json.dumps({
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode()

        req = urllib.request.Request(
            ollama_url, data=body, headers={"Content-Type": "application/json"})
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "120"))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode(errors="ignore")
            except Exception:
                pass
            if exc.code == 404 or "not found" in detail.lower():
                raise BackendError(
                    f"Ollama model '{ollama_model}' not pulled "
                    f"(run: ollama pull {ollama_model})") from exc
            raise BackendError(f"Ollama HTTP {exc.code}: {detail[:200]}") from exc
        except urllib.error.URLError as exc:
            raise BackendError(
                f"Ollama unavailable at {ollama_url} ({exc.reason}); "
                f"is `ollama serve` running?") from exc
        except TimeoutError as exc:
            raise BackendError(
                f"Ollama timed out after {timeout}s for model "
                f"'{ollama_model}'") from exc
        except Exception as exc:
            raise BackendError(f"Ollama call failed: {exc}") from exc

        if data.get("error"):
            raise BackendError(f"Ollama error: {data['error']}")
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise BackendError(
                f"Malformed Ollama response: {str(data)[:200]}") from exc


_PROVIDER_CLASSES = {
    "anthropic": AnthropicBackend,
    "groq": GroqBackend,
    "ollama": OllamaBackend,
}


def get_backend(name: str | None = None) -> LLMBackend:
    """
    Build the backend object for `name` (or the resolved active backend).

    Raises BackendError for an unknown / unconfigured backend name so callers
    fail loudly rather than silently mis-routing.
    """
    resolved = resolve_backend_name(name)
    cfg = _backend_config(resolved)
    provider = cfg.get("provider")
    if provider not in _PROVIDER_CLASSES:
        raise BackendError(
            f"Unknown or unconfigured backend '{resolved}'. "
            f"Known backends: {', '.join(KNOWN_BACKENDS)}")
    return _PROVIDER_CLASSES[provider](resolved, cfg)


def backend_model(name: str) -> str:
    """Human-readable model id a backend name will use (for reports/labels)."""
    cfg = _backend_config(resolve_backend_name(name))
    return cfg.get("model", "?")


def _ollama_reachable(url: str, timeout: float = 1.5) -> bool:
    """
    Fast TCP probe of the Ollama daemon. Uses a raw socket to an explicit IP so a
    refused connection returns immediately (avoids the ~4 s IPv6→IPv4 fallback
    that `localhost` incurs on Windows).
    """
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    if host == "localhost":
        host = "127.0.0.1"
    port = parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_backend(name: str) -> tuple[bool, str]:
    """
    Cheap availability check used by the benchmark harness so it does not hammer
    a down backend hundreds of times. Returns (available, reason).

      * anthropic/groq → is the API key present?
      * ollama*        → is the daemon reachable, and is the model pulled?
    """
    resolved = resolve_backend_name(name)
    cfg = _backend_config(resolved)
    provider = cfg.get("provider")

    if provider == "anthropic":
        ok = bool(os.getenv("ANTHROPIC_API_KEY"))
        return ok, "ANTHROPIC_API_KEY set" if ok else "ANTHROPIC_API_KEY missing"
    if provider == "groq":
        ok = bool(os.getenv("GROQ_API_KEY"))
        return ok, "GROQ_API_KEY set" if ok else "GROQ_API_KEY missing"
    if provider == "ollama":
        url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
        if not _ollama_reachable(url):
            return False, "Ollama daemon not reachable (run `ollama serve`)"
        # Daemon is up — check the model is pulled via /api/tags.
        model = cfg.get("model", "")
        try:
            import json
            import urllib.request
            from urllib.parse import urlparse
            p = urlparse(url)
            tags_url = f"{p.scheme}://{p.hostname}:{p.port or 11434}/api/tags"
            with urllib.request.urlopen(tags_url, timeout=5) as resp:
                names = [m.get("name", "") for m in json.loads(resp.read()).get("models", [])]
            pulled = any(n == model or n.startswith(model.split(":")[0]) for n in names)
            if not pulled:
                return False, f"model '{model}' not pulled (run `ollama pull {model}`)"
            return True, f"Ollama up, model '{model}' available"
        except Exception:
            # Daemon answered the socket but /api/tags failed — let the real call try.
            return True, "Ollama reachable (tag check skipped)"
    return False, f"unknown backend '{resolved}'"
