"""
RAG corpus builder.

Assembles the retrieval corpus from two sources:

1. The curated welding knowledge base (``config/welding_knowledge.yaml``) — one
   passage per topic, combining its summary, causes, remedies, parameters and
   notes into a single retrievable blob.
2. Free-text reference documents under ``data/knowledge_docs/`` (``*.md`` /
   ``*.txt``) — chunked into section-sized passages so longer handbook excerpts
   stay retrievable at a useful granularity. Drop new ``.md``/``.txt`` files in
   that folder and they are ingested automatically on the next index build.

Each passage is a plain dict so it serialises straight to JSON alongside the
vector index::

    {
        "id":       stable unique id (e.g. "kb:porosity" or "doc:gas.md#2"),
        "text":     the retrievable text,
        "source":   where it came from (filename),
        "title":    short human label,
        "category": KB category / "reference_document",
    }
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent.parent
_KB_PATH = _ROOT / "config" / "welding_knowledge.yaml"
_DOCS_DIR = _ROOT / "data" / "knowledge_docs"

# Section chunks longer than this are split further (by paragraph) so a single
# retrieved passage stays focused.
_MAX_CHUNK_CHARS = 1100

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)")


def build_corpus() -> list[dict]:
    """Return the full retrieval corpus (KB entries + reference-doc chunks)."""
    passages: list[dict] = []
    passages.extend(_passages_from_knowledge_base())
    passages.extend(_passages_from_docs())
    return passages


def _passages_from_knowledge_base() -> list[dict]:
    """One passage per curated welding-knowledge topic."""
    if not _KB_PATH.exists():
        return []
    with open(_KB_PATH, "r", encoding="utf-8") as f:
        topics = (yaml.safe_load(f) or {}).get("topics", {}) or {}

    out: list[dict] = []
    for topic, entry in topics.items():
        parts: list[str] = []
        if entry.get("summary"):
            parts.append(entry["summary"])
        if entry.get("causes"):
            parts.append("Likely causes: " + "; ".join(entry["causes"]))
        if entry.get("remedies"):
            parts.append("Corrective actions: " + "; ".join(entry["remedies"]))
        if entry.get("parameters"):
            parts.append("Key parameters: " + ", ".join(entry["parameters"]))
        if entry.get("notes"):
            parts.append("Notes: " + entry["notes"])

        title = topic.replace("_", " ").title()
        out.append({
            "id":       f"kb:{topic}",
            "text":     f"{title}. " + " ".join(parts),
            "source":   "welding_knowledge.yaml",
            "title":    title,
            "category": entry.get("category", "welding"),
        })
    return out


def _passages_from_docs() -> list[dict]:
    """Section-sized passages from reference documents in data/knowledge_docs/."""
    if not _DOCS_DIR.exists():
        return []
    out: list[dict] = []
    for path in sorted(_DOCS_DIR.glob("*")):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        if path.stem.lower() == "readme":  # folder docs, not welding content
            continue
        text = path.read_text(encoding="utf-8")
        for i, (heading, body) in enumerate(_split_sections(text)):
            chunk = (f"{heading}. {body}" if heading else body).strip()
            if len(chunk) < 20:
                continue
            out.append({
                "id":       f"doc:{path.name}#{i}",
                "text":     chunk,
                "source":   path.name,
                "title":    heading or path.stem.replace("_", " ").title(),
                "category": "reference_document",
            })
    return out


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown/plain text into (heading, body) sections at ``#`` headings.

    Long section bodies are further split by paragraph so no single passage
    exceeds ``_MAX_CHUNK_CHARS``.
    """
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    lines: list[str] = []
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            sections.append((heading, lines))
            heading = m.group(1).strip()
            lines = []
        else:
            lines.append(line)
    sections.append((heading, lines))

    out: list[tuple[str, str]] = []
    for h, ls in sections:
        body = "\n".join(ls).strip()
        if not body and not h:
            continue
        for sub in _split_long(body):
            out.append((h, sub))
    return out


def _split_long(body: str) -> list[str]:
    """Pack paragraphs of ``body`` into chunks of at most ``_MAX_CHUNK_CHARS``."""
    if len(body) <= _MAX_CHUNK_CHARS:
        return [body]
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if buf and len(buf) + len(p) + 1 > _MAX_CHUNK_CHARS:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks or [body]
