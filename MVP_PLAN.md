# MVP Implementation Plan — LLM Multimodal Manufacturing Chatbot

**Scope:** MVP only (Phases 0–3 from the SRD). Phases 4–7 excluded.
**Goal:** Operator types a natural-language question → system returns anomaly status + likely root cause + recommendation + plain-language explanation (SHAP) + confidence.
**Assumed team:** 1 developer, working steadily. Time estimates are working-days.
**Total MVP estimate:** ~22–30 working days.

> Tasks are ordered by dependency. Don't start a task until its listed dependencies are checked off.

---

## T0 — Project setup & key decisions
- [ ] **1. What to build:** Repo scaffold, virtual environment, dependency manifest, folder structure, and a one-page decision record resolving the three Phase-0 blockers: (a) no-RAG confirmed, (b) base LLM chosen, (c) dataset strategy chosen.
- [ ] **2. Technologies:** Python 3.11+, `venv`/`conda`, `pip`/`poetry`, Git, a `README` + `DECISIONS.md`.
- [ ] **3. Expected files:** `requirements.txt` (or `pyproject.toml`), `README.md`, `DECISIONS.md`, `.gitignore`, empty package folders (`data/`, `src/`, `tests/`).
- [ ] **4. Estimated time:** 0.5–1 day.
- [ ] **5. Dependencies:** None (start here).

**Decision guidance:** For an explainability-heavy, self-hosted, no-RAG MVP, use the LLM only as a *natural-language synthesis layer* over deterministic model outputs — not as the detector. An API model (e.g., Anthropic/OpenAI) is fastest; a small local model is the alternative if data must stay offline.

---

## T1 — Synthetic dataset generator
- [ ] **1. What to build:** A script that generates a small, time-aligned tri-modal dataset: sensor time-series (with injected anomalies), production logs (timestamps + event codes), and operator notes (free text loosely correlated to anomalies). Include a ground-truth label column for anomalies/causes so you can evaluate later.
- [ ] **2. Technologies:** Python, `numpy`, `pandas`. Optionally a template/LLM call to generate varied operator-note text.
- [ ] **3. Expected files:** `src/data/generate_synthetic.py`, output `data/raw/sensors.csv`, `data/raw/logs.csv` (or `.json`), `data/raw/notes.csv`, `data/raw/ground_truth.csv`.
- [ ] **4. Estimated time:** 2–3 days.
- [ ] **5. Dependencies:** T0.

**Why first:** the SRD flagged data as the biggest risk. Controlling your own labeled data unblocks every downstream task and makes evaluation possible.

---

## T2 — Data ingestion / loaders
- [ ] **1. What to build:** Loader functions that read each modality from disk into clean in-memory objects with consistent schemas and validated types/timestamps.
- [ ] **2. Technologies:** Python, `pandas`, `pydantic` (schema validation, optional).
- [ ] **3. Expected files:** `src/data/loaders.py`, `src/data/schemas.py`.
- [ ] **4. Estimated time:** 1 day.
- [ ] **5. Dependencies:** T1 (needs a known data format).

---

## T3 — Per-modality preprocessing
- [ ] **1. What to build:** Three preprocessing routines. **Sensor:** missing-value imputation (linear or KNN), light denoising (moving-average), normalization (zero mean/unit variance), fixed-window segmentation. **Logs:** parse structured fields, dedup, format-correct. **Notes:** text cleaning, stop-word/symbol removal, tokenization, then embedding via a sentence encoder. (Wavelet denoising can be skipped for MVP; moving-average is enough.)
- [ ] **2. Technologies:** `pandas`, `numpy`, `scikit-learn` (imputation/scaling), `sentence-transformers` or `transformers` with DistilBERT for note embeddings.
- [ ] **3. Expected files:** `src/preprocess/sensor.py`, `src/preprocess/logs.py`, `src/preprocess/notes.py`, `src/preprocess/__init__.py`.
- [ ] **4. Estimated time:** 3–4 days.
- [ ] **5. Dependencies:** T2.

---

## T4 — Temporal alignment & simple fusion
- [ ] **1. What to build:** Align notes, logs, and sensor windows to shared process-event timestamps, then fuse into a single feature record per time window (concatenated sensor features + log features + note embeddings). Full cross-attention is deferred — concatenation/aligned join is the MVP approach.
- [ ] **2. Technologies:** `pandas` (time joins / `merge_asof`), `numpy`.
- [ ] **3. Expected files:** `src/fusion/align.py`, `src/fusion/fuse.py`, output `data/processed/fused.parquet`.
- [ ] **4. Estimated time:** 2 days.
- [ ] **5. Dependencies:** T3.

---

## T5 — Anomaly detection (baseline model)
- [ ] **1. What to build:** A baseline detector over the fused features that flags anomalous time windows and outputs an anomaly probability/score. No LLM fine-tuning at this stage.
- [ ] **2. Technologies:** `scikit-learn` (IsolationForest or RandomForest classifier if using labels), persisted with `joblib`.
- [ ] **3. Expected files:** `src/model/anomaly.py`, `src/model/train_anomaly.py`, saved model `models/anomaly.joblib`.
- [ ] **4. Estimated time:** 2 days.
- [ ] **5. Dependencies:** T4.

---

## T6 — SHAP explainability
- [ ] **1. What to build:** Wrap the detector with SHAP to produce per-prediction feature-importance values, plus a helper that turns the top contributors into a structured, plain-language explanation payload. (LIME and attention viz are deferred — SHAP only for MVP.)
- [ ] **2. Technologies:** `shap`, `matplotlib` (optional static plot).
- [ ] **3. Expected files:** `src/explain/shap_explainer.py`, optional `outputs/shap_plot.png`.
- [ ] **4. Estimated time:** 2 days.
- [ ] **5. Dependencies:** T5.

---

## T7 — Root-cause & recommendation logic
- [ ] **1. What to build:** For the MVP, a heuristic/rule-based mapper: given the anomaly and its top SHAP drivers + matching log events/notes, return a ranked list of likely causes and a corresponding recommended action (e.g., "vibration spike on motor 2 → check bearing / reduce setpoint"). Learned ranking (MRR/loss functions) is deferred to Phase 4.
- [ ] **2. Technologies:** Python (rules/lookup tables), optionally a small config file mapping driver patterns → causes/actions.
- [ ] **3. Expected files:** `src/reasoning/root_cause.py`, `src/reasoning/recommend.py`, `config/cause_action_map.yaml`.
- [ ] **4. Estimated time:** 2 days.
- [ ] **5. Dependencies:** T6.

---

## T8 — Confidence scoring
- [ ] **1. What to build:** Derive a confidence value per recommendation from the detector's probability and the strength/agreement of SHAP drivers; expose it as a number + qualitative band (low/medium/high).
- [ ] **2. Technologies:** Python (`numpy`).
- [ ] **3. Expected files:** `src/reasoning/confidence.py`.
- [ ] **4. Estimated time:** 0.5–1 day.
- [ ] **5. Dependencies:** T7.

---

## T9 — LLM natural-language synthesis layer
- [ ] **1. What to build:** A module that takes the structured outputs (anomaly, causes, recommendation, SHAP drivers, confidence) and composes a clear conversational answer. Also handles interpreting the user's NL question to decide what to query. The LLM does **not** detect anomalies — it only explains/synthesizes, keeping the pipeline transparent and no-RAG.
- [ ] **2. Technologies:** An LLM via API (Anthropic/OpenAI SDK) **or** a small local model (`transformers`/Ollama) per the T0 decision; prompt templates.
- [ ] **3. Expected files:** `src/chat/synthesize.py`, `src/chat/prompts.py`.
- [ ] **4. Estimated time:** 2–3 days.
- [ ] **5. Dependencies:** T8.

---

## T10 — Chat backend API
- [ ] **1. What to build:** An endpoint that receives a user question, runs the pipeline (load → preprocess → fuse → detect → explain → reason → confidence → synthesize), and returns the composed answer + structured payload.
- [ ] **2. Technologies:** `FastAPI` + `uvicorn`.
- [ ] **3. Expected files:** `src/api/main.py`, `src/api/pipeline.py`.
- [ ] **4. Estimated time:** 2 days.
- [ ] **5. Dependencies:** T9.

---

## T11 — Chat frontend
- [ ] **1. What to build:** A minimal chat UI: question box, answer display, an expandable panel showing the SHAP explanation and confidence. Optimize for speed of build, not polish.
- [ ] **2. Technologies:** **Streamlit** (fastest path for an MVP/demo); React is the alternative if a richer UI is needed later.
- [ ] **3. Expected files:** `app/streamlit_app.py` (or `frontend/` if React).
- [ ] **4. Estimated time:** 2 days.
- [ ] **5. Dependencies:** T10.

---

## T12 — Integration & demo scenario
- [ ] **1. What to build:** Wire everything end-to-end against the synthetic dataset and script the "definition of done" demo: *"Why did line 3 slow down at 14:00?"* → cause + action + explanation + confidence. Fix integration gaps.
- [ ] **2. Technologies:** all of the above; a short demo script/notebook.
- [ ] **3. Expected files:** `demo/walkthrough.md`, `demo/demo_queries.txt`.
- [ ] **4. Estimated time:** 1–2 days.
- [ ] **5. Dependencies:** T11.

---

## T13 — Lightweight evaluation
- [ ] **1. What to build:** Minimal sanity metrics on the labeled synthetic data — anomaly accuracy/F1 and a quick check that SHAP drivers align with the injected causes. Full 5-fold CV and KPI benchmarking are deferred to Phase 6.
- [ ] **2. Technologies:** `scikit-learn` metrics, `pandas`.
- [ ] **3. Expected files:** `tests/eval_mvp.py`, `outputs/eval_report.md`.
- [ ] **4. Estimated time:** 1 day.
- [ ] **5. Dependencies:** T12.

---

## Critical path & sequencing notes
- **Hard chain:** T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9 → T10 → T11 → T12 → T13. Most tasks are serial because each consumes the previous output.
- **Parallelizable if you get help:** T11 (frontend) skeleton can be stubbed against a mock API while T5–T9 are built. T1 note-text generation can overlap T2.
- **Biggest risk:** T1/T3 (data + preprocessing). If the synthetic data isn't realistic enough, anomaly detection and SHAP look meaningless. Budget extra buffer here.
- **Deferred to later phases (do NOT build now):** cross-attention fusion, LLM fine-tuning, LIME, attention visualization, learned root-cause ranking, 5-fold CV, KPI benchmarking, MES/ERP integration, human-in-the-loop accept/reject UI.
