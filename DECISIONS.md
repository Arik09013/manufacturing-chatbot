# Architectural Decision Record — Phase 0

_These three decisions were required before any implementation work could begin._

---

## Decision 1 — No-RAG Architecture Confirmed

**Status:** Accepted

**Context:**
The original design considered a RAG (Retrieval-Augmented Generation) pipeline that would embed historical anomaly records and retrieve them at query time to give the LLM context. This adds complexity (vector database, embedding index, retrieval latency, hallucination risk from retrieved chunks) without a clear MVP benefit, because the real intelligence sits in the ML pipeline, not in historical text retrieval.

**Decision:**
No RAG. The LLM is used purely as a **natural-language synthesis layer**. It receives a structured JSON payload from the deterministic pipeline (anomaly flag, SHAP drivers, root-cause candidates, recommendation, confidence) and converts it to a clear conversational answer. The LLM does not retrieve, detect, or reason over raw data.

**Consequences:**
- Pipeline is fully transparent and auditable.
- LLM output is grounded in deterministic model outputs, not free-text retrieval.
- No vector DB dependency in MVP.
- Adding RAG for knowledge-base queries is straightforward in a later phase.

---

## Decision 2 — Base LLM Chosen: Anthropic Claude (API)

**Status:** Accepted

**Context:**
Three options considered:
1. **Anthropic Claude API** — mature API, strong instruction-following, fast, cost-effective at haiku tier, no GPU required.
2. **OpenAI API** — equivalent capability, slightly higher cost, equivalent integration complexity.
3. **Local model via Ollama** — zero API cost, full data privacy, but requires capable GPU and adds infra complexity for a 1-developer MVP.

**Decision:**
Use **Anthropic Claude** via the `anthropic` Python SDK. Specifically `claude-haiku-4-5-20251001` for synthesis (low latency, low cost per query) with the option to upgrade to Sonnet for richer explanations later.

**Rationale:**
- Fastest path to a working synthesis layer.
- `anthropic` SDK is stable and well-documented.
- haiku is sufficient for structured-payload-to-text conversion tasks.
- No GPU requirement keeps the MVP deployable on any developer machine.
- API key is the only external dependency (stored in `.env`).

**Consequences:**
- Requires `ANTHROPIC_API_KEY` in `.env`.
- Online data dependency — operator queries leave the local network. For fully offline operation, switch to Ollama (see `src/chat/synthesize.py` comments).

---

## Decision 3 — Dataset Strategy: Fully Synthetic, Programmatic Generation

**Status:** Accepted

**Context:**
Real manufacturing sensor data is either unavailable, commercially sensitive, or requires significant cleaning. The SRD flagged data availability as the biggest project risk.

Options considered:
1. **Public benchmark datasets** (e.g., NASA CMAPSS, SECOM) — real but single-modality, not tri-modal, require significant adaptation.
2. **Synthetic generation** — full control over modalities, anomaly types, ground-truth labels, and correlation structure.
3. **Hybrid** — seed synthetic parameters from a public benchmark. Deferred to later phase.

**Decision:**
Generate a fully synthetic, tri-modal dataset using `numpy` + `pandas`:
- **Sensor time-series:** 3 machines × 1 week × 1-minute intervals, 5 sensor channels each, with injected anomaly windows (10 anomalies per machine, 5 anomaly types).
- **Production logs:** structured event records correlated to anomaly windows.
- **Operator notes:** templated free-text notes loosely correlated to anomaly events.
- **Ground truth:** per-window labels with anomaly type and root cause.

**Consequences:**
- No external data dependency.
- Evaluation is meaningful because we know the ground truth exactly.
- Anomaly/cause correlation is controllable and realistic enough for MVP.
- Realistic enough for demonstrating the full pipeline; does not represent production variance.
