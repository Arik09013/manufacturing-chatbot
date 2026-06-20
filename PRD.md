# Product Requirements Document (PRD)

## Welding Process Optimization Chatbot with Explainable AI

| | |
|---|---|
| **Project** | LLM-Enhanced, Multimodal Welding Process Optimization with Explainable AI |
| **Domain** | MIG/MAG/TIG/SMAW arc welding (smart manufacturing, Industry 4.0) |
| **Author** | Tasnimur Rahman Fayad (ID: 2009026) |
| **Supervisor** | Kazi Naimur Rahman, Assistant Professor, Dept. of MIE, CUET |
| **Status** | MVP implemented & validated |
| **Version** | 1.0 |
| **Date** | June 2026 |

---

## 1. Overview

### 1.1 Problem Statement
Modern welding shops generate heterogeneous data — time-series sensor feeds, structured event logs, and free-text operator notes — yet operators still struggle to (a) choose **optimal weld parameters** for a given job, (b) **diagnose faults** quickly, and (c) get **trustworthy, explainable** guidance. Conventional ML tools act as black boxes, and raw LLMs can hallucinate unsafe numbers — unacceptable in a safety-critical welding context.

### 1.2 Solution
A chatbot that answers welding questions across three grounded routes:
1. **Parameter optimization** — computes the *best-efficiency* welding settings (current, voltage, travel speed, wire feed) from welding **physics** and published **standards windows**.
2. **Fault diagnosis** — detects weld anomalies from fused multimodal data using a supervised ML model, explained with **SHAP**.
3. **Welding knowledge (RAG)** — answers defect / troubleshooting / cost / quality questions using **hybrid retrieval-augmented generation**: it retrieves the most relevant passages from a welding corpus (curated knowledge base + reference documents) via combined semantic-embedding and keyword search, then grounds the LLM answer in those passages with inline **source citations**.

An LLM (Groq / Claude / Ollama) sits on top **only to phrase grounded results in plain language** — it never invents numbers, and on the knowledge route it answers strictly from retrieved, cited passages. This delivers the thesis goal: **transparent, traceable, grounded, hallucination-resistant** recommendations (numeric outputs from physics/ML; knowledge answers from cited retrieval).

### 1.3 Goals & Non-Goals
**Goals**
- Recommend optimal, physically-grounded weld parameters for supported material × thickness × process.
- Detect and explain weld faults from sensor + log + note data.
- Answer common welding knowledge questions with **RAG-grounded, source-cited** advisory answers.
- Make every output explainable (SHAP, sensitivity tables, confidence scores, retrieved-source citations).
- Ground knowledge answers in **retrieved sources (RAG)** and never use **LLM-generated numbers** for parameter or diagnosis outputs.

**Non-Goals (out of scope)**
- Physical hardware/PLC/SCADA integration.
- Financial or supply-chain analytics.
- Visual weld-defect inspection from images (future work).
- Real-time closed-loop machine control.

---

## 2. Users & Use Cases

| User | Need | Example query |
|---|---|---|
| Welding operator | Best settings for a job | "Best settings for 5mm stainless steel with MIG" |
| Maintenance engineer | Diagnose a fault | "Why did station_1 stop welding at 14:00?" |
| Apprentice / QC | Welding know-how | "Why am I getting porosity in stainless welds?" |
| Process engineer | Productivity / cost | "How can I increase deposition rate without losing quality?" |

---

## 3. Functional Requirements

### FR-1 — Intent Guard
- **FR-1.1** Classify each question as in-scope (welding/manufacturing) or out-of-scope.
- **FR-1.2** Reject out-of-scope questions (e.g. "what's the weather") with a helpful message listing supported query types.

### FR-2 — Question Router
- **FR-2.1** Route each in-scope question to one of: `param` (optimization), `anomaly` (diagnosis), `knowledge` (advice).
- **FR-2.2** Route to `anomaly` only when a station/line/machine is referenced; to `param` when a parameter optimization is requested; otherwise to `knowledge`.

### FR-3 — Parameter Optimizer (primary deliverable)
- **FR-3.1** Accept material, thickness, and process (defaults applied when omitted).
- **FR-3.2** Look up the standards window (current/voltage/speed/wire-feed/gas) per material × thickness band × process.
- **FR-3.3** Compute **heat input** `HI = (η·60·I·V)/(1000·S)` [kJ/mm] and **deposition rate** `DR = WFS·A·ρ·η_d` [g/min].
- **FR-3.4** Find optimal settings via deterministic **grid search** maximizing `deposition_rate × heat_input_window_score` (throughput within the safe thermal window).
- **FR-3.5** Produce a **sensitivity table**: nudge each parameter ±1 step, recompute metrics, and list qualitative effects.
- **FR-3.6** Return an **honest error** (never fabricated numbers) for unsupported materials (e.g. titanium, copper) or invalid combos (e.g. SMAW on aluminum).

### FR-4 — Anomaly Diagnosis
- **FR-4.1** Load and fuse sensor + log + operator-note data, filtered by station.
- **FR-4.2** Detect anomalies (5 fault types: arc_instability, wire_feed_fault, gas_flow_failure, overheating, underheat) with a supervised classifier.
- **FR-4.3** Explain the prediction with **SHAP** (top driving sensor channels).
- **FR-4.4** Map to a likely **root cause** and **recommended action**.
- **FR-4.5** Attach a **confidence score** = 0.7 × anomaly probability + 0.3 × SHAP agreement.

### FR-5 — Welding Knowledge (RAG)
- **FR-5.1** Build a retrieval corpus from the curated knowledge base (defects, troubleshooting, quality, productivity, cost, comparisons) plus drop-in reference documents (`data/knowledge_docs/`).
- **FR-5.2** Retrieve the most relevant passages via **hybrid search** — local sentence-embedding cosine similarity (semantic) blended with keyword overlap (lexical) — and rank them.
- **FR-5.3** Ground the LLM answer strictly in the retrieved passages, with **inline `[S#]` source citations**; fall back to keyword-matched entries (and then an advisory general-welding answer) if the embedding model is unavailable or nothing is relevant.
- **FR-5.4** Embeddings are computed **locally** (no API key, offline-capable); the vector index persists and rebuilds automatically when the corpus changes.

### FR-6 — LLM Synthesis Layer
- **FR-6.1** Convert structured pipeline output into a plain-language answer.
- **FR-6.2** Support pluggable backends: **Groq** (llama-3.3-70b), **Anthropic** (Claude), **Ollama** (local) via `SYNTHESIZER_BACKEND`.
- **FR-6.3** Never produce numbers not present in the computed payload (enforced by system prompt).
- **FR-6.4** Provide a deterministic text fallback when no LLM backend/key is available.

### FR-7 — Interfaces
- **FR-7.1** Streamlit chat UI (dark industrial theme) with parameter cards, SHAP charts, and knowledge cards.
- **FR-7.2** FastAPI backend: `POST /chat`, `POST /pipeline/raw`, `GET /health`.

---

## 4. Non-Functional Requirements
- **NFR-1 Trust/Safety:** All numeric outputs are computed by physics or ML — never by the LLM.
- **NFR-2 Explainability:** Every recommendation carries an explanation (SHAP, sensitivity, or grounded causes).
- **NFR-3 Grounded retrieval (RAG):** The welding-knowledge route uses a hybrid RAG layer (local semantic embeddings + keyword overlap) over a welding corpus; retrieved passages are cited inline. Numeric routes (parameter optimizer, anomaly detector) remain deterministic and never use retrieval or LLM-generated numbers. No cloud vector DB — embeddings are local and offline-capable.
- **NFR-4 Offline-capable:** Functions without internet via Ollama or deterministic fallback.
- **NFR-5 Performance:** Single-question response within a few seconds on CPU (model cached in memory).
- **NFR-6 Portability:** Pure-Python, Windows-friendly, no GPU required.

---

## 5. System Architecture

```
User question
     │
     ▼
[Intent guard] ──► reject out-of-scope
     │ in-scope
     ▼
[Router] ─────────┬──────────────────┬───────────────────┐
     ▼            ▼                   ▼
  PARAM         ANOMALY            KNOWLEDGE (RAG)
  physics       sensor+log+note     corpus: KB + docs
  grid search   → RandomForest      → embed + hybrid
  → HI, DR      → SHAP explain        retrieval (sem+kw)
  → sensitivity → root cause + conf  → cited passages [S#]
     └────────────┴───────────────────┘
                  │
                  ▼
        [LLM synthesis] (Groq / Claude / Ollama) — phrasing only
                  │
                  ▼
   Plain-language answer + explainability (Streamlit / API)
```

### 5.1 Key Modules
| Module | File | Responsibility |
|---|---|---|
| Intent classifier | `src/chat/intent.py` | scope gate |
| Router | `src/api/pipeline.py` | `route_question()` |
| Parameter advisor | `src/reasoning/param_advisor.py` | physics + grid-search optimization |
| Anomaly detector | `src/model/anomaly.py` | RandomForest |
| SHAP explainer | `src/explain/shap_explainer.py` | feature attribution |
| Knowledge base | `config/welding_knowledge.yaml` + `src/reasoning/knowledge.py` | curated entries + keyword fallback |
| RAG retriever | `src/rag/` (`corpus.py`, `index.py`, `retriever.py`) | corpus build, local embeddings, hybrid semantic+keyword retrieval, citations |
| LLM synthesis | `src/chat/synthesize.py`, `src/chat/prompts.py` | narration, backends |
| UI | `app/streamlit_app.py` | chat front-end |
| API | `src/api/main.py` | REST endpoints |

### 5.2 Configuration / Data
- `config/welding_params.yaml` — standards windows (AWS D1.1 / ISO 15614-1 style).
- `config/parameter_effects.yaml` — qualitative parameter→effect mapping.
- `config/welding_knowledge.yaml` — curated welding knowledge entries (also a RAG corpus source).
- `data/knowledge_docs/*.md|*.txt` — drop-in reference documents ingested into the RAG corpus.
- `models/rag_index/` — persisted vector index (embeddings + passages), auto-rebuilt when the corpus changes.
- `data/raw/*.csv` — synthetic sensor / logs / notes / ground-truth (3 stations × 7 days).

---

## 6. Domain Model (supported scope)

| Dimension | Supported values |
|---|---|
| Materials | mild steel, stainless steel, aluminum |
| Processes | MIG/MAG (GMAW), TIG (GTAW), SMAW (stick) |
| Thickness bands | thin / medium / thick (material-specific) |
| Sensor channels | welding_current, arc_voltage, welding_speed, wire_feed_rate, shielding_gas_flow, heat_input |
| Fault types | arc_instability, wire_feed_fault, gas_flow_failure, overheating, underheat |

**Example output — 5 mm stainless steel, MIG:** 187 A, 24.2 V, 450 mm/min, 9.3 m/min wire feed; heat input 0.493 kJ/mm (window 0.30–0.65); deposition 78.5 g/min; efficiency 8/10.

---

## 7. Success Metrics
| Metric | Target | Status |
|---|---|---|
| Anomaly detection F1 (5-fold CV) | ≥ 0.85 | **0.905** ✅ |
| ROC-AUC | ≥ 0.90 | **0.9996** ✅ (synthetic data) |
| SHAP alignment with injected channel | 100% | **100%** ✅ |
| Fabricated numbers in answers | 0 | **0** ✅ |
| Unit tests passing | all | **88/88** ✅ |

---

## 8. Risks & Limitations
| Risk / Limitation | Mitigation / Note |
|---|---|
| Synthetic (not real) factory data | Pipeline is data-agnostic; swap in real CSVs |
| ROC-AUC ~1.0 is optimistic | Injected faults are cleanly separable; real data noisier |
| LLM is narration-only (not fine-tuned) | Deliberate — prevents hallucinated numbers; fine-tuning is future work |
| Only SHAP implemented (no LIME/attention) | Stated future work |
| Single-fault-per-window detection | Multi-label detection is future work |
| No image/visual defect input | Multimodal vision is future work |

---

## 9. Future Work
- Fine-tune a domain LLM on welding logs/notes.
- Add LIME + attention-visualization explainability.
- Vision-language model for weld-defect images.
- Expand materials/processes (titanium, FCAW, SAW).
- Live MES/sensor-stream integration; human-in-the-loop confirmation UI.

---

## 10. How to Run
```powershell
# install
.venv\Scripts\python.exe -m pip install -r requirements.txt

# configure (copy and add a key)
copy .env.example .env   # set SYNTHESIZER_BACKEND=groq and GROQ_API_KEY

# run the chatbot UI
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py   # http://localhost:8501

# (optional) run the REST API
.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload     # http://localhost:8000

# run tests
.venv\Scripts\python.exe -m pytest tests/ -q                       # 88 passing
```
