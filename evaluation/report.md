# Comparative LLM Evaluation — Explainable Welding Assistant

*Generated 2026-07-04 by `evaluation/benchmark.py`.*

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

- **Dataset:** `evaluation/datasets/benchmark_set.json` — **64 prompts** spanning routes: anomaly, general, knowledge, param, robotics.
- **Backends compared:** anthropic (claude-haiku-4-5-20251001), ollama_mistral (mistral:7b), ollama_llama (llama3.2), groq (llama-3.3-70b-versatile), ollama_qwen (qwen2.5:3b).
- **Prompt parity:** identical grounding payload and system prompt per route across all backends.
- **Fallback:** any backend failure (down / not pulled / timeout / no key) falls back to the deterministic grounded summary and is recorded — the harness never crashes.

### Hardware / Environment

- **Compute:** CPU-only (no GPU required); local models served by Ollama.
- **OS / Python:** Windows 10 · Python 3.11.0.
- **Ollama:** ollama version is 0.30.11.
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

> **Note.** One or more backends were unavailable at run time (Ollama daemon not running, model not pulled, or missing API key) and used the system's deterministic grounded fallback. Those rows are flagged by a non-zero *Fallback* rate and reflect the fallback text, not live model output. Re-run with the backends available (`ollama serve` + `ollama pull qwen2.5:3b mistral:7b`, and/or API keys in `.env`) for a full head-to-head.

## 5. Results

| Rank | Backend | Model | Quality | Faithfulness | Halluc. | Citation | Readability | Safety | Concise | Latency (s) | tok/s | Cost/1k ($) | Fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | anthropic | claude-haiku-4-5-20251001 | 0.851 | 0.970 | 0.030 | 0.632 | 0.725 | 0.992 | 0.748 | 10.86 | 47.1 | 4.1000 | 0.00 |
| 2 | ollama_mistral | mistral:7b | 0.812 | 0.972 | 0.028 | 0.711 | 0.529 | 0.946 | 0.888 | 37.78 | 7.4 | 0 | 0.03 |
| 3 | ollama_llama | llama3.2 | 0.776 | 0.956 | 0.044 | 0.605 | 0.487 | 0.935 | 0.842 | 9.45 | 27.2 | 0 | 0.00 |
| 4 | groq | llama-3.3-70b-versatile | 0.769 | 1.000 | 0.000 | 0.500 | 0.499 | 0.930 | 0.765 | 0.91 | 748.0 | 1.1000 | 0.94 |
| 5 | ollama_qwen | qwen2.5:3b | 0.758 | 0.985 | 0.015 | 0.355 | 0.466 | 0.964 | 0.775 | 9.02 | 34.6 | 0 | 0.00 |

*Composite quality weights: faithfulness 0.35, safety 0.25, citation 0.15,
readability 0.15, conciseness 0.10 (metrics that are N/A on a route are dropped
and the remaining weights renormalized).*

### 5.1 Quality by route

| Route | anthropic | groq | ollama_llama | ollama_qwen | ollama_mistral |
|---|---|---|---|---|---|
| anomaly | 0.935 | 0.899 | 0.892 | 0.901 | 0.921 |
| general | 0.899 | 0.847 | 0.722 | 0.765 | 0.718 |
| knowledge | 0.775 | 0.631 | 0.713 | 0.653 | 0.783 |
| param | 0.953 | 0.996 | 0.931 | 0.945 | 0.906 |

### 5.2 Figures

See `evaluation/results/figures/`:
`quality_bar.png`, `radar.png`, `latency_bar.png`, `hallucination_bar.png`,
`faithfulness_bar.png`, `quality_by_route.png`, `cost_vs_quality.png`.

## 6. Discussion — model trade-offs

- **Claude** — Highest linguistic quality and instruction-following; the most reliable at honouring the 'narrate, don't invent' contract and at citation discipline on the knowledge route. Trade-off: hosted API (network + cost dependency), higher per-call latency than a warm local model.
- **Llama-3.3-70B (Groq)** — Balanced baseline with very fast hosted inference. Good general fluency; occasionally looser with numeric grounding than Claude. Trade-off: still an external API dependency.
- **Qwen2.5:3b (local)** — Strong technical reasoning and multilingual ability for its size, small memory footprint, fully local. A good fit when the shop floor needs an offline assistant with solid structured-instruction following. Trade-off: 3B capacity caps nuance vs the hosted models.
- **Mistral 7B (local)** — Fast, lightweight, efficient CPU inference; fully local and private. Good throughput-per-watt for edge deployment. Trade-off: less consistent citation discipline than Claude on the knowledge route.
- **Llama 3.2 (local)** — Balanced local baseline — no API key, fully offline, reasonable prose. Trade-off: CPU latency and a smaller model than the hosted 70B.

**Leading configuration this run:** `anthropic` (claude-haiku-4-5-20251001) at
composite quality **0.851**, mean latency
10.86 s, fallback rate 0.00.
For a *fully-offline* deployment the strongest local candidates are Qwen2.5:3b
(technical reasoning, small footprint) and Mistral 7B (fast CPU inference); for
*maximum linguistic quality and citation discipline* with a network budget,
hosted Claude leads.

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
