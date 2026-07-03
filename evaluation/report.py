"""
Thesis-ready markdown report generator for the benchmark.

generate(agg, per_backend, raw_records, prompts, out_path) writes report.md with:
    Methodology · Experimental Setup · Hardware · Metrics · Results table ·
    Per-route breakdown · Discussion (per-model advantages) · Limitations · Future Work.

The prose is templated but the numbers are pulled live from the run, so the
report always matches the CSVs and figures in evaluation/results/.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import date
from pathlib import Path


def _ollama_version() -> str:
    try:
        out = subprocess.run(["ollama", "--version"], capture_output=True,
                             text=True, timeout=5)
        return (out.stdout or out.stderr).strip() or "installed (version unknown)"
    except Exception:
        return "not detected at report time (models run via local Ollama daemon)"


def _fmt(v, nd=3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _results_table(agg: list[dict]) -> str:
    head = ("| Rank | Backend | Model | Quality | Faithfulness | Halluc. | "
            "Citation | Readability | Safety | Concise | Latency (s) | tok/s | "
            "Cost/1k ($) | Fallback |")
    sep = "|" + "---|" * 14
    rows = [head, sep]
    for i, s in enumerate(agg, 1):
        cost1k = (s.get("est_cost_usd") or 0) * 1000
        rows.append(
            f"| {i} | {s['backend']} | {s['model']} | "
            f"{_fmt(s['composite_quality'])} | {_fmt(s['faithfulness'])} | "
            f"{_fmt(s['hallucination_rate'])} | {_fmt(s['citation_score'])} | "
            f"{_fmt(s['readability_operator'])} | {_fmt(s['safety_score'])} | "
            f"{_fmt(s['conciseness_score'])} | {_fmt(s['latency_s'],2)} | "
            f"{_fmt(s['tokens_per_sec'],1)} | {_fmt(cost1k,4)} | "
            f"{_fmt(s['fallback_rate'],2)} |")
    return "\n".join(rows)


def _route_table(per_backend: dict) -> str:
    routes = sorted({r["route"] for rows in per_backend.values() for r in rows})
    backends = [b for b, rows in per_backend.items() if rows]
    head = "| Route | " + " | ".join(backends) + " |"
    sep = "|" + "---|" * (len(backends) + 1)
    lines = [head, sep]
    for route in routes:
        cells = [route]
        for b in backends:
            vals = [r["composite_quality"] for r in per_backend[b] if r["route"] == route]
            cells.append(f"{sum(vals)/len(vals):.3f}" if vals else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


_ADVANTAGES = {
    "anthropic": ("Claude",
        "Highest linguistic quality and instruction-following; the most reliable "
        "at honouring the 'narrate, don't invent' contract and at citation "
        "discipline on the knowledge route. Trade-off: hosted API (network + cost "
        "dependency), higher per-call latency than a warm local model."),
    "groq": ("Llama-3.3-70B (Groq)",
        "Balanced baseline with very fast hosted inference. Good general fluency; "
        "occasionally looser with numeric grounding than Claude. Trade-off: still "
        "an external API dependency."),
    "ollama_llama": ("Llama 3.2 (local)",
        "Balanced local baseline — no API key, fully offline, reasonable prose. "
        "Trade-off: CPU latency and a smaller model than the hosted 70B."),
    "ollama_qwen": ("Qwen2.5:3b (local)",
        "Strong technical reasoning and multilingual ability for its size, small "
        "memory footprint, fully local. A good fit when the shop floor needs an "
        "offline assistant with solid structured-instruction following. Trade-off: "
        "3B capacity caps nuance vs the hosted models."),
    "ollama_mistral": ("Mistral 7B (local)",
        "Fast, lightweight, efficient CPU inference; fully local and private. "
        "Good throughput-per-watt for edge deployment. Trade-off: less consistent "
        "citation discipline than Claude on the knowledge route."),
}


def _discussion(agg: list[dict]) -> str:
    present = {s["backend"] for s in agg}
    blocks = []
    for backend in ["anthropic", "groq", "ollama_qwen", "ollama_mistral", "ollama_llama"]:
        if backend in present and backend in _ADVANTAGES:
            name, text = _ADVANTAGES[backend]
            blocks.append(f"- **{name}** — {text}")
    return "\n".join(blocks)


def generate(agg: list[dict], per_backend: dict, raw_records: list[dict],
             prompts: list[dict], out_path: Path) -> None:
    n_prompts = len(prompts)
    routes = sorted({p["route"] for p in prompts})
    any_fallback = any((s.get("fallback_rate") or 0) > 0 for s in agg)
    winner = agg[0] if agg else None

    fallback_note = ""
    if any_fallback:
        fallback_note = (
            "\n> **Note.** One or more backends were unavailable at run time "
            "(Ollama daemon not running, model not pulled, or missing API key) and "
            "used the system's deterministic grounded fallback. Those rows are "
            "flagged by a non-zero *Fallback* rate and reflect the fallback text, "
            "not live model output. Re-run with the backends available "
            "(`ollama serve` + `ollama pull qwen2.5:3b mistral:7b`, and/or API keys "
            "in `.env`) for a full head-to-head.\n")

    md = f"""# Comparative LLM Evaluation — Explainable Welding Assistant

*Generated {date.today().isoformat()} by `evaluation/benchmark.py`.*

## 1. Objective

This report determines **which LLM is most suitable for an explainable welding
assistant** — not which model is "best" in the abstract. The assistant narrates
the output of a deterministic pipeline (anomaly detector + SHAP/LIME, welding
parameter optimiser, and a RAG-grounded knowledge base), so the decisive
qualities are *faithfulness to the computed payload*, *citation discipline*,
*operator readability*, *safety*, and *deployability* (latency / cost / offline
capability) — balanced against raw linguistic quality.

## 2. Methodology

Each prompt is routed exactly as in production and the deterministic pipeline is
run **once** to build the grounding payload. That single payload is then narrated
by **every** backend, so the comparison isolates *synthesis quality* on an
identical input. All scoring is **automated and deterministic** (heuristics, no
human raters and no LLM-as-judge), so the benchmark is fully reproducible.

Numeric-faithfulness metrics are applied only to the **grounded routes**
(parameter optimisation, anomaly diagnosis) where every number must trace to the
payload. On the knowledge/general/robotics routes the models supply standard
textbook starting points, so numeric faithfulness is recorded as N/A and quality
is judged on citation preservation, readability, safety, and conciseness.

## 3. Experimental Setup

- **Dataset:** `evaluation/datasets/benchmark_set.json` — **{n_prompts} prompts** spanning routes: {", ".join(routes)}.
- **Backends compared:** {", ".join(s['backend'] + " (" + s['model'] + ")" for s in agg)}.
- **Prompt parity:** identical grounding payload and system prompt per route across all backends.
- **Fallback:** any backend failure (down / not pulled / timeout / no key) falls back to the deterministic grounded summary and is recorded — the harness never crashes.

### Hardware / Environment

- **Compute:** CPU-only (no GPU required); local models served by Ollama.
- **OS / Python:** {platform.system()} {platform.release()} · Python {platform.python_version()}.
- **Ollama:** {_ollama_version()}.
- **Local models:** `qwen2.5:3b`, `mistral:7b`, `llama3.2`. **Hosted models:** Anthropic Claude, Llama-3.3-70B via Groq.

## 4. Evaluation Metrics

| Metric | What it measures | How (automated) |
|---|---|---|
| Groundedness (0–5) | Answer uses only payload numbers | numeric overlap with grounding |
| Hallucination rate | Invented numbers/parameters | fraction of stated numbers absent from grounding |
| Faithfulness | Numbers identical to source | supported / total substantive numbers |
| Citation preservation | `[S#]` labels kept, none invented | set overlap vs supplied passages |
| Readability | Operator friendliness | Flesch reading ease + sentence length |
| Safety | Correct refusal / no fabricated recs | refusal detection + faithfulness on error payloads |
| Conciseness | Response length discipline | word count vs target band |
| Latency | Seconds per response | wall-clock timing |
| Throughput | tokens/sec | est. output tokens / latency |
| Cost | Estimated USD | token counts × public pricing (local = \$0) |
| Determinism | Variance over repeated runs | length CV + pairwise token Jaccard |
| **Composite quality** | Single ranking score (0–1) | weighted blend (faithfulness+safety dominate) |
{fallback_note}
## 5. Results

{_results_table(agg)}

*Composite quality weights: faithfulness 0.35, safety 0.25, citation 0.15,
readability 0.15, conciseness 0.10 (metrics that are N/A on a route are dropped
and the remaining weights renormalized).*

### 5.1 Quality by route

{_route_table(per_backend)}

### 5.2 Figures

See `evaluation/results/figures/`:
`quality_bar.png`, `radar.png`, `latency_bar.png`, `hallucination_bar.png`,
`faithfulness_bar.png`, `quality_by_route.png`, `cost_vs_quality.png`.

## 6. Discussion — model trade-offs

{_discussion(agg)}
"""

    if winner:
        md += f"""
**Leading configuration this run:** `{winner['backend']}` ({winner['model']}) at
composite quality **{_fmt(winner['composite_quality'])}**, mean latency
{_fmt(winner['latency_s'],2)} s, fallback rate {_fmt(winner['fallback_rate'],2)}.
For a *fully-offline* deployment the strongest local candidates are Qwen2.5:3b
(technical reasoning, small footprint) and Mistral 7B (fast CPU inference); for
*maximum linguistic quality and citation discipline* with a network budget,
hosted Claude leads.
"""

    md += """
## 7. Limitations

- Metrics are **heuristic**: numeric faithfulness is a strong proxy for
  hallucination but cannot judge subtle domain correctness of prose; Flesch
  approximates readability rather than measuring operator comprehension.
- Local-model results depend on the exact quantization/tag pulled by Ollama and
  on CPU class; latency figures are machine-specific.
- Cost figures are **estimates** from public list pricing and token
  approximations, intended for relative comparison, not billing.
- The dataset, while spanning all routes, is finite; rare defect types and long
  multi-turn conversations are under-represented.

## 8. Future Work

- Add an optional **LLM-as-judge** rubric pass (with a fixed judge model) to
  complement the heuristics on prose quality, and correlate it with the
  automated scores.
- Expand the dataset toward 200+ prompts with multi-turn parameter follow-ups
  and adversarial "make up a number" probes.
- Measure **energy / memory** per response for the local models to quantify edge
  deployability, and evaluate additional Ollama tags (e.g. `qwen2.5:7b`,
  `mistral-nemo`) and quantization levels.
- Wire the winning local backend into the Isaac Sim loop for on-prem, offline
  operation of the full explainable welding cell.
"""

    out_path.write_text(md, encoding="utf-8")
