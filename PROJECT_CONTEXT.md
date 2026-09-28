# PROJECT CONTEXT: LLM-Enhanced Multimodal Manufacturing Chatbot with Explainable AI

> **Document Type:** Comprehensive Autonomous Project Context & Handoff Specification  
> **Repository Path:** `D:\E\manufacturing-chatbot`  
> **Target Audience:** Future AI Assistant / Engineering Maintainer  
> **Primary Author:** Tasnimur Rahman Fayad (ID: 2009026), MIE Dept., CUET  
> **Supervisor:** Kazi Naimur Rahman, Assistant Professor, Dept. of MIE, CUET  
> **Current Active Git Branch:** `feat/qwen-mistral-backends-eval` (HEAD commit: `e18f5dd`)  
> **Last Updated:** 2026-09-29  

---

## 1. EXECUTIVE SUMMARY & PURPOSE

### 1.1 Project Purpose & Thesis Scope
This repository implements an **operator-facing, multimodal smart manufacturing assistant** specialized for arc welding operations (MIG/MAG/GMAW, TIG/GTAW, SMAW/Stick). Built as an undergraduate thesis project at Chittagong University of Engineering and Technology (CUET), the system unites **heterogeneous industrial data** (time-series sensor streams, discrete machine event logs, unstructured operator shift notes, and reference engineering documentation) with **explainable AI (XAI)** and **pluggable Large Language Models (LLMs)**.

### 1.2 Problem Being Solved
1. **Black-Box Opacity:** Conventional deep learning models diagnose anomalies without auditable evidence. Shop floor operators and quality engineers reject opaque predictions.
2. **LLM Hallucination Risk:** Off-the-shelf generative LLMs invent unsafe numerical parameters (amperage, voltage, travel speed, gas flow) or hallucinate non-existent root causes. In safety-critical welding, an incorrect heat-input setting can result in burn-through, cold cracking, or structural weld failure.
3. **Data Modality Fragmentation:** Factory data lives in silos—sensor feeds (time-series), programmable logic controller (PLC) alarm logs (event codes), and maintenance logs/shift notes (free text) are rarely fused temporally.
4. **Parameter Optimization Bottlenecks:** Operators rely on trial-and-error rather than physics-grounded standards windows to set up weld passes for specific materials and thicknesses.

### 1.3 Core Architectural Philosophy
- **Deterministic Compute, Narration-Only LLM:** All numerical calculations (heat input, deposition rate, standards search, sensitivity differentials) and statistical diagnostics (anomaly probability, feature attribution, root cause matching) are strictly computed by Python physics engines and deterministic machine learning pipelines.
- **Strict Grounding:** The LLM is **never** permitted to generate numbers out of thin air. It receives a structured JSON payload containing verified calculations and is constrained to natural-language synthesis.
- **Defense in Depth (Triple XAI):** Three complementary explainability mechanisms are implemented:
  - **SHAP (TreeExplainer):** Global and local feature attributions for tabular sensor/log data.
  - **LIME (Tabular):** Independent local surrogate perturbation checking SHAP agreement.
  - **DistilBERT Self-Attention:** Visualizing token attention weights across unstructured operator shift notes.
- **Multi-Route Modular Architecture:** Five specialized query routes prevent domain bleeding between welding physics, fault diagnosis, cited documentation retrieval, robotics simulation, and general factory engineering.

### 1.4 Target Users & Primary Scenarios
| User Persona | Primary Need | Example Query |
|---|---|---|
| **Welding Operator** | Optimal setup parameters for a specific workpiece | *"Best settings for 5mm stainless steel with MIG"* |
| **Maintenance Technician** | Diagnosing line stops, alarms, and erratic machine behavior | *"Why did station_1 stop welding at 14:00?"* |
| **QC / Welding Apprentice** | Remediation for metallurgical weld defects | *"Why am I getting porosity in my stainless steel welds?"* |
| **Robotics / Sim Engineer** | Setting up automated welding workcells in NVIDIA Isaac Sim | *"How do I add a depth camera and force-torque sensor to the Isaac Sim weld cell?"* |
| **Manufacturing Generalist** | Process guidance outside welding (CNC, injection molding, casting) | *"How do I reduce cycle time and eliminate sink marks in injection molding?"* |

---

## 2. CURRENT STATE SNAPSHOT

| Dimension | Current Snapshot Status | Evidence / Notes |
|---|---|---|
| **Project Lifecycle Status** | **Production-Ready MVP / Thesis-Validated** | T0 through T14 fully completed; core evaluation passed. |
| **Unit & Integration Tests** | **143 / 143 PASSED (100%)** | Verified via `.venv\Scripts\python.exe -m pytest -v` (execution time: 64.82s). |
| **Anomaly Detection Quality** | **F1 = 0.9516, ROC-AUC = 0.9992** | 5-fold Stratified Cross-Validation on 1,917 fused windows (`outputs/eval_report.md`). |
| **XAI Alignment** | **SHAP: 82% overall (100% on 4 of 5 fault types)** | Validated against synthetic injection ground truth. LIME agreement at 67%. |
| **Working Features** | **All 5 Query Routes Operational** | 1. Parameter Advisor (Grid Search)<br>2. Anomaly Diagnosis (RF + SHAP/LIME + Attention)<br>3. Welding Knowledge (Hybrid RAG + Citations)<br>4. Robotics / Isaac Sim Integration Advisor<br>5. General Manufacturing Advisor |
| **LLM Backends** | **5 Pluggable Backends Integrated** | 1. Anthropic Claude (Haiku / Opus)<br>2. Groq (Llama-3.3-70B-Versatile)<br>3. Ollama Llama 3.2 (Local)<br>4. Ollama Qwen2.5:3b (Local)<br>5. Ollama Mistral 7B (Local)<br>All with deterministic fallback on failure. |
| **UI Application** | **Fully Operational Streamlit Dashboard** | Professional dark industrial UI (`app/streamlit_app.py`) featuring SHAP plots, LIME cards, attention heatmaps, and parameter cards. |
| **API Backend** | **Fully Operational FastAPI Server** | Endpoints `/chat`, `/pipeline/raw`, and `/health` (`src/api/main.py`). |
| **Documentation & Reports** | **Complete** | PDF generator (`docs/build_documentation.py`), comparative evaluation report (`evaluation/report.md`), thesis gap analysis (`THESIS_GAP_ANALYSIS.md`). |
| **Broken / Failing Features** | **Zero Broken Code** | No crashing tests or dead imports. Groq backend showed rate-limiting fallbacks during massive batch runs (handled gracefully). |
| **Deployment Status** | **Local Development & Workstation Edge** | Runs locally on CPU/GPU Windows environments; containerization (Docker) is currently pending/future work. |

---

## 3. COMPLETE ARCHITECTURE & DATA FLOW

### 3.1 High-Level Architectural Diagram

```
                                  [ Operator Natural Language Query ]
                                                   │
                                                   ▼
                                      ┌─────────────────────────┐
                                      │  src/chat/intent.py     │
                                      │  Deterministic Scope    │
                                      │  Intent Guard (Keywords)│
                                      └────────────┬────────────┘
                                                   │ (If In-Scope)
                                                   ▼
                                      ┌─────────────────────────┐
                                      │  src/api/pipeline.py    │
                                      │  route_question()       │
                                      └──────┬──┬──┬──┬─────────┘
        ┌────────────────────────────────────┘  │  │  └──────────────────────────────────────┐
        ▼                                       ▼  ▼                                         ▼
┌──────────────────┐               ┌───────────────────────┐                    ┌───────────────────────┐
│  ROUTE 1: PARAM  │               │  ROUTE 2: ANOMALY     │                    │  ROUTE 5: GENERAL     │
│  Physics Grid    │               │  Multimodal Detection │                    │  Non-welding domains  │
│  Optimization    │               └───────────┬───────────┘                    │  (CNC, Molding, etc.) │
└────────┬─────────┘                           │                                └───────────┬───────────┘
         │                          ┌──────────┴──────────┐                                 │
         │                          ▼                     ▼                                 │
         │                 ┌──────────────────┐  ┌──────────────────┐                       │
         │                 │ Sensor/Log Data  │  │ Operator Notes   │                       │
         │                 │ Windowing & Aggs │  │ Preprocessing    │                       │
         │                 └────────┬─────────┘  └────────┬─────────┘                       │
         │                          └──────────┬──────────┘                                 │
         │                                     ▼                                            │
         │                         ┌───────────────────────┐                                │
         │                         │ Fused Parquet Window  │                                │
         │                         └───────────┬───────────┘                                │
         │                                     │                                            │
         │                    ┌────────────────┼────────────────┐                           │
         │                    ▼                ▼                ▼                           │
         │           ┌────────────────┐┌───────────────┐┌───────────────┐                   │
         │           │ RandomForest   ││ SHAP & LIME   ││ DistilBERT    │                   │
         │           │ Classifier     ││ Feature       ││ Text Self-    │                   │
         │           │ (or DistilBERT)││ Attributions  ││ Attention Map │                   │
         │           └────────┬───────┘└───────┬───────┘└───────┬───────┘                   │
         │                    └────────────────┼────────────────┘                           │
         │                                     ▼                                            │
         │                         ┌───────────────────────┐                                │
         │                         │ Root-Cause & Action   │                                │
         │                         │ Heuristic Mapper      │                                │
         │                         └───────────┬───────────┘                                │
         │                                     │                                            │
         ├─────────────────────────────────────┴───────────────┬────────────────────────────┤
         │                                                     │                            │
         ▼                                                     ▼                            │
┌───────────────────────────────┐             ┌────────────────────────────────┐            │
│  ROUTE 3: WELDING KNOWLEDGE   │             │  ROUTE 4: ROBOTICS INTEGRATION │            │
│  (HYBRID RAG RETRIEVER)       │             │  Isaac Sim, Sensors, Teleop    │            │
│  - all-MiniLM-L6-v2 Embeddings│             │  ROS 2, Workcell Automation    │            │
│  - 0.60 Cosine Semantic       │             │  - Curated Architecture Maps   │            │
│  - 0.25 Lexical Overlap       │             │  - Procedural Implementation   │            │
│  - 0.15 Title Keyword Boost   │             │  - ASCII Flowcharts            │            │
│  - Inline [S#] Citations      │             └────────────────┬───────────────┘            │
└────────┬──────────────────────┘                              │                            │
         └─────────────────────────────────────┬───────────────┴────────────────────────────┘
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Structured Diagnostic Payload   │
                              │  (Computed Numbers, Citations,   │
                              │   SHAP Drivers, Actions, Ranges) │
                              └────────────────┬─────────────────┘
                                               ▼
                              ┌──────────────────────────────────┐
                              │  src/chat/synthesize.py          │
                              │  LLM Synthesis Layer             │
                              │  (Narrator Contract: No Inventions)
                              └────────────────┬─────────────────┘
                                               ▼
                              ┌──────────────────────────────────┐
                              │  src/chat/backends.py            │
                              │  Pluggable Model Provider:       │
                              │  Claude | Groq | Qwen | Mistral  │
                              └────────────────┬─────────────────┘
                                               │ (On any failure: deterministic fallback)
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Operator Interface Presentation │
                              │  - Streamlit UI Dashboard        │
                              │  - FastAPI REST Endpoints        │
                              └──────────────────────────────────┘
```

### 3.2 Five Concrete Execution Flows

#### Route 1: Parameter Optimization (`param`)
1. **Trigger:** Query contains parameter keywords (e.g., "best settings", "what speed", "voltage for 5mm") without specifying an anomaly station.
2. **Parsing (`src/reasoning/param_advisor.py`):**
   - Natural language regex extracts material (Mild Steel, Stainless Steel, Aluminum), thickness in mm, welding process (GMAW/MIG, GTAW/TIG, SMAW/Stick), user-pinned parameter overrides (e.g. `wire_diameter=1.2`), bounds (`voltage >= 21`), or ranges (`speed 300 to 400`).
   - Unsupported materials (e.g., Titanium, Copper, Brass) or invalid process-material combinations (SMAW on Aluminum) generate an immediate, honest error string without defaulting.
   - Conversation state is preserved across turns via `merge_param_context()`.
3. **Standards Lookup:** Loads standards ranges from `config/welding_params.yaml` based on AWS D1.1 / ISO 15614-1 specifications.
4. **Deterministic Grid Search:**
   - Evaluates a 7-step grid across unpinned parameters.
   - Computes Heat Input: $\text{HI} = \frac{\eta \cdot 60 \cdot I \cdot V}{1000 \cdot S} \text{ [kJ/mm]}$
   - Computes Deposition Rate: $\text{DR} = \text{WFS} \cdot \left(\frac{\pi \cdot d^2}{4}\right) \cdot \rho \cdot \eta_d \text{ [g/min]}$
   - Maximizes throughput objective while scoring thermal compliance within the standards window.
5. **Sensitivity Analysis:** Perturbs each parameter by $\pm 1$ step, recomputes metrics, and attaches physical consequence labels from `config/parameter_effects.yaml`.
6. **Payload Construction:** Emits `pipeline_type="parameter_advice"` with exact numbers, sensitivity table, and efficiency score (1-10).

#### Route 2: Anomaly Detection & Diagnosis (`anomaly`)
1. **Trigger:** Query references a specific station or machine (`station_1`, `station_2`, `station_3`, `line 2`).
2. **Ingestion & Windowing (`src/preprocess/`):**
   - Loads 1-minute raw sensor time-series (`sensors.csv`), discrete event logs (`logs.csv`), and shift notes (`notes.csv`).
   - Normalizes and extracts 30-minute rolling aggregate features (mean, std, min, max, range) for 6 channels.
   - Merges modalities via `src/fusion/align.py` and `src/fusion/fuse.py`.
3. **Inference (`src/model/anomaly.py`):**
   - Evaluates feature vector using trained `RandomForestClassifier` (saved in `models/anomaly.joblib`).
   - Predicts binary `is_anomaly` and continuous `anomaly_prob`.
4. **Explainability Triad (`src/explain/`):**
   - **SHAP:** `AnomalyExplainer` executes `TreeExplainer` to identify the top-5 feature drivers pushing towards anomaly.
   - **LIME:** `LimeAnomalyExplainer` builds a local linear surrogate on 500 background samples to verify feature directions.
   - **Attention:** If an operator note exists within 90 minutes of the window, DistilBERT calculates token-level self-attention weights.
5. **Root-Cause & Recommendation (`src/reasoning/`):**
   - Matches SHAP drivers and event log codes against `config/cause_action_map.yaml`.
   - Computes confidence score: $\text{Confidence} = 0.70 \times \text{anomaly\_prob} + 0.30 \times \text{shap\_agreement}$.
6. **Payload Construction:** Emits diagnostic dictionary with anomaly flag, type, probability, ranked causes, actions, SHAP drivers, LIME drivers, and attention heatmap.

#### Route 3: Welding Knowledge Retrieval (`knowledge` / RAG)
1. **Trigger:** In-scope query concerning welding defects, troubleshooting, quality, or comparisons without a specific machine filter.
2. **Corpus & Index (`src/rag/`):**
   - Corpus unites 25 structured YAML topics (`config/welding_knowledge.yaml`) and markdown engineering reference docs (`data/knowledge_docs/`).
   - Passages are indexed via sentence-transformers `all-MiniLM-L6-v2` into an L2-normalized dense embedding matrix (`models/rag_index/`).
3. **Hybrid Retrieval (`src/rag/retriever.py`):**
   - Hybrid ranking formula:
     $$\text{Score} = 0.60 \times \text{CosineSimilarity} + 0.25 \times \text{LexicalOverlap} + 0.15 \times \text{TitleMatch}$$
   - Filters out passages below `min_score = 0.15` and labels the top-4 with citation tags `[S1]`, `[S2]`, `[S3]`, `[S4]`.
4. **Payload Construction:** Emits `pipeline_type="knowledge_advice"`, passing retrieved text blocks and citation tags into `src/chat/prompts.py:KNOWLEDGE_RAG_PROMPT`.

#### Route 4: Robotics & Simulation Integration (`knowledge_domain="integration"`)
1. **Trigger:** Query matches robotics or digital twin keywords (e.g., "isaac sim", "omniverse", "ros2", "teleop", "cloudxr", "urdf", "force/torque").
2. **Specialized Persona & Formatting:**
   - Swaps system persona to a Robotics Simulation & Systems Integration Engineer (`INTEGRATION_SYSTEM_PROMPT`).
   - Expands `max_tokens` headroom to 4,000 for full architectural walkthroughs.
   - Enforces procedural phases, hardware/software interface tables, version warnings (Isaac Sim / CloudXR), and ASCII component diagrams (`┌ ─ ┐ │ └ ┘`).

#### Route 5: General Manufacturing Advisor (`general`)
1. **Trigger:** Query identifies a non-welding manufacturing domain (CNC machining, lathe, milling, injection molding, sand casting, sheet metal stamping, additive manufacturing, Six Sigma, lean/OEE).
2. **Advisory Persona:**
   - Directs to `run_general_pipeline()`.
   - Instructs the model (`GENERAL_MANUFACTURING_SYSTEM_PROMPT`) to act as a senior industrial engineer.
   - Enforces strict safety disclaimers: all numbers must be explicitly stated as starting estimates requiring verification against machine spec sheets and tooling manuals.

---

## 4. DETAILED COMPONENT SPECIFICATIONS

### 4.1 Ingestion & Preprocessing Layer
- **`src/data/schemas.py`**:
  - Defines Pydantic v2 validation models: `SensorReading`, `LogEntry`, `OperatorNote`, and `GroundTruthWindow`.
  - Enforces physical parameter bounds: `welding_current` (0–400 A), `arc_voltage` (0–50 V), `welding_speed` (0–800 mm/min), `wire_feed_rate` (0–20 m/min), `shielding_gas_flow` (0–30 L/min), `heat_input` (0–2.0 kJ/mm).
  - Enforces valid machines (`station_1`, `station_2`, `station_3`) and 5 canonical anomaly types.
- **`src/data/loaders.py`**:
  - Robust, typed loaders for all raw CSV files in `data/raw/`.
  - Parses timestamps into UTC-naive `datetime64[ns]`, casts categorical and float columns, and provides optional Pydantic schema validation.
- **`src/data/generate_synthetic.py`**:
  - Deterministic synthetic generator (`seed=42`).
  - Simulates 3 stations over 7 days at 1-minute intervals (30,240 sensor rows).
  - Injects exactly 30 fault windows (10 per machine) across 5 defect classes with correlated alarm logs and operator shift notes.
- **`src/preprocess/sensor.py`**:
  - Handles missing data imputation (linear interpolation + forward/backward fill), denoising (rolling median/mean filter), StandardScaler normalization, and 30-minute tumbling window feature extraction (generating 30 statistical features across the 6 channels).
- **`src/preprocess/logs.py`**:
  - Parses discrete event logs, aggregates event counts per window (`n_alarm`, `n_warning`, `n_maintenance`, `n_diagnostic`), flags active alarms, and concatenates unique event code strings.
- **`src/preprocess/notes.py`**:
  - Cleans free-text operator notes and computes dense semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2`.

### 4.2 Fusion Layer
- **`src/fusion/align.py`**:
  - Executes temporal alignment between irregular event logs/notes and fixed 30-minute sensor windows using pandas interval matching.
- **`src/fusion/fuse.py`**:
  - Combines tabular sensor aggregates, log features, and note flags.
  - Labels windows by computing intersection over duration against `ground_truth.csv` ($\ge 50\%$ overlap marks `is_anomaly=True`).
  - Saves the resulting 2,010 windows to `data/processed/fused.parquet`.

### 4.3 Machine Learning & Model Layer
- **`src/model/anomaly.py`**:
  - `AnomalyDetector`: Wrapper class providing unified `fit`, `predict`, and `predict_proba` methods.
  - Primary Model: Supervised `RandomForestClassifier` (`n_estimators=200`, `min_samples_leaf=2`, `class_weight="balanced"`, `random_state=42`).
  - Secondary / Fallback Model: Unsupervised `IsolationForest` for zero-ground-truth settings.
  - Defines 39 tabular input feature columns (sensor aggregates, log counts, note presence flags).
- **`src/model/train_anomaly.py`**:
  - Script that loads `data/processed/fused.parquet`, fits `StandardScaler` and `RandomForestClassifier`, and persists the bundle to `models/anomaly.joblib`.
- **`src/model/bert_detector.py` & `finetune_distilbert.py`**:
  - Implements the thesis objective for fine-tuning a transformer.
  - Serializes each multimodal window into a structured sentence: *"Welding station window. current mean 185.0A std 2.1... Log: 0 alarms... Event codes: none. Operator note present."*
  - Fine-tunes `distilbert-base-uncased` for sequence classification (`models/distilbert_fault/`).

### 4.4 Explainable AI (XAI) Layer
- **`src/explain/shap_explainer.py`**:
  - Wraps the trained tree model in `shap.TreeExplainer(feature_perturbation="tree_path_dependent")`.
  - Computes exact SHAP values for class 1 (anomaly), extracts top-5 drivers with directional tags (`toward_anomaly` vs `toward_normal`), and generates plain-language explanations.
- **`src/explain/lime_explainer.py`**:
  - `LimeAnomalyExplainer`: Tabular surrogate model built on 500 background samples using `lime.lime_tabular.LimeTabularExplainer`.
  - Explains individual predictions by perturbing features and evaluating probabilities to verify whether LIME feature directions agree with SHAP.
- **`src/explain/attention_explainer.py`**:
  - Loads `distilbert-base-uncased` in eager attention mode (`output_attentions=True`).
  - Extracts multi-head self-attention weights for free-text notes, pools over layers and heads, merges sub-word tokens (`##`), and returns word-level saliency heatmaps.

### 4.5 Reasoning & Decision Logic Layer
- **`src/reasoning/param_advisor.py`**:
  - Core welding engineering engine (1,175 lines of pure deterministic logic).
  - Contains full physics formulas for arc heat input and electrode deposition rate.
  - Standard material thickness mapping (`THICKNESS_BANDS`), unit conversions, and multi-parameter grid search optimizer.
  - `parse_param_query()`: Extracts engineering intent, numbers, and constraints.
  - `merge_param_context()`: Preserves conversational memory across turns.
  - `_build_sensitivity_table()`: Computes directional derivative impacts for parameter nudges.
- **`src/reasoning/knowledge.py`**:
  - Curated retrieval matcher for `config/welding_knowledge.yaml`.
  - Token-level scoring prioritizing multi-word technical phrases (e.g., "lack of fusion").
  - Produces deterministic fallback summaries when no LLM is available.
- **`src/reasoning/root_cause.py`**:
  - Evaluates anomaly type, top SHAP driver features, and PLC event codes against `config/cause_action_map.yaml`.
  - Returns ranked root causes categorized by evidence strength (`strong`, `moderate`, `low`).
- **`src/reasoning/recommend.py`**:
  - Extracts immediate primary action, secondary verification step, and urgency level (`critical`, `high`, `medium`).
- **`src/reasoning/confidence.py`**:
  - Implements calibrated scoring formula combining model probability and SHAP driver consistency:
    $$\text{Score} = 0.70 \times P(\text{anomaly}) + 0.30 \times \text{Agreement}$$
  - Classifies predictions into `high` ($\ge 0.75$), `medium` ($0.45 - 0.75$), or `low` ($< 0.45$) confidence bands.

### 4.6 Retrieval-Augmented Generation (RAG) Layer
- **`src/rag/corpus.py`**:
  - Assembles passages from `config/welding_knowledge.yaml` and `.md`/`.txt` engineering manuals in `data/knowledge_docs/`.
  - Automatically splits documents by markdown headings and chunks sections exceeding 1,100 characters.
- **`src/rag/index.py`**:
  - Embeds passages into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.
  - L2-normalizes vectors and computes sha256 corpus hash for automatic cache invalidation (`models/rag_index/`).
- **`src/rag/retriever.py`**:
  - Public retrieval API executing hybrid search (semantic cosine + lexical overlap + title boost).
  - Ranks passages, assigns citation tags (`[S1]`, `[S2]`), and returns structured passage objects.
- **`src/rag/build_index.py`**:
  - Standalone CLI utility for manual index compilation and retrieval testing.

### 4.7 LLM Synthesis & Backends Layer
- **`src/chat/backends.py`**:
  - Unified interface: `generate(user_content, system_prompt, max_tokens, model)`.
  - Implementations: `AnthropicBackend` (Claude SDK), `GroqBackend` (Groq SDK), `OllamaBackend` (local HTTP streaming/REST via `urllib`).
  - Precedence: Explicit argument $\rightarrow$ Environment variable $\rightarrow$ `config/llm.yaml` $\rightarrow$ Built-in defaults.
  - Robust exception handling: Any communication timeout, missing key, unpulled model, or HTTP failure raises `BackendError`, intercepted cleanly to trigger deterministic grounded fallback.
- **`src/chat/synthesize.py`**:
  - Formats payloads into structured prompts, delegates to the active backend, measures wall-clock latency, and returns synthesized text or fallback markdown.
- **`src/chat/prompts.py`**:
  - Highly engineered system prompts enforcing strict adherence to provided numbers and citation discipline.
  - Templates: `SYSTEM_PROMPT` (anomaly), `PARAM_SYSTEM_PROMPT` (parameters), `KNOWLEDGE_SYSTEM_PROMPT` & `KNOWLEDGE_RAG_PROMPT` (curated/cited knowledge), `INTEGRATION_SYSTEM_PROMPT` (robotics/Isaac Sim), and `GENERAL_MANUFACTURING_SYSTEM_PROMPT` (general shop floor).
- **`src/chat/intent.py`**:
  - Pre-pipeline scope filter using keyword and regex matching across hundreds of manufacturing terms.
  - Instantly rejects general knowledge queries (sports, weather, politics) with a polite refusal, saving computation and API tokens.

### 4.8 Serving & User Interface Layer
- **`src/api/main.py`**:
  - FastAPI application exposing REST endpoints:
    - `POST /chat`: Accepts `{question, machine_id, query_time, param_context}`, executes pipeline, and returns synthesized answer + raw payload.
    - `POST /pipeline/raw`: Executes pipeline and returns structured JSON payload without invoking an LLM.
    - `GET /health`: Basic health and liveness probe.
- **`app/streamlit_app.py`**:
  - Comprehensive, interactive dashboard designed with custom CSS (dark industrial theme `#0d1117`).
  - Renders parameter cards, heat input gauges, sensitivity expanders, SHAP waterfall charts, LIME comparison cards, DistilBERT attention heatmaps, and RAG cited source snippets.

---

## 5. REPOSITORY STRUCTURE & FILE PURPOSES

```
D:\E\manufacturing-chatbot\
├── .claude/
│   └── settings.local.json          # Claude Code assistant settings (dev tooling)
├── .env                             # Local secrets & active backend settings (DO NOT COMMIT)
├── .env.example                     # Environment template defining supported variables
├── .gitignore                       # Git ignore rules for venv, data, models, logs
├── app/
│   └── streamlit_app.py             # Streamlit operator UI (interactive dashboard)
├── config/
│   ├── cause_action_map.yaml        # Fault-to-cause and corrective action knowledge base
│   ├── llm.yaml                     # Central registry of LLM backends and default models
│   ├── parameter_effects.yaml       # Qualitative physical consequences for parameter nudges
│   ├── welding_knowledge.yaml       # 25 curated welding defect and engineering topics
│   └── welding_params.yaml          # Standard welding parameter windows (AWS/ISO tables)
├── data/
│   ├── knowledge_docs/              # Reference documentation for RAG ingestion
│   │   ├── README.md                # Guide on adding external reference documents
│   │   └── shielding_gas_selection.md # Drop-in reference document for shielding gases
│   ├── processed/
│   │   └── fused.parquet            # Multimodal fused dataset (2,010 windows)
│   └── raw/
│       ├── ground_truth.csv         # Labeled anomaly windows for evaluation
│       ├── logs.csv                 # Machine event logs and alarm codes
│       ├── notes.csv                # Operator shift notes
│       └── sensors.csv              # High-frequency 1-min welding sensor feeds
├── demo/
│   ├── demo_queries.txt             # Curated list of demo questions across all routes
│   ├── run_demo.py                  # CLI smoke-test script running end-to-end queries
│   └── walkthrough.md               # Scripted presentation guide for thesis defense
├── docs/
│   ├── build_documentation.py       # ReportLab PDF report generation script
│   └── Welding_Chatbot_Project_Documentation.pdf # Generated thesis project report
├── evaluation/
│   ├── benchmark.py                 # Automated multi-model benchmark runner
│   ├── metrics.py                   # Deterministic quality metrics engine (no LLM judge)
│   ├── plots.py                     # Matplotlib chart generator (PNG figures)
│   ├── report.py                    # Markdown evaluation report builder
│   ├── report.md                    # Generated benchmark comparison report
│   ├── datasets/
│   │   └── benchmark_set.json       # 64 curated test prompts across all 5 routes
│   └── results/                     # Benchmark artifacts (CSVs, figures, raw JSON)
│       ├── comparison.csv           # Head-to-head comparison table across backends
│       ├── raw_outputs.json         # Raw outputs from all evaluated models
│       ├── figures/                 # Generated PNG plots (radar, latency, quality, etc.)
│       └── [model]_results.csv      # Per-model detailed metric logs
├── models/
│   ├── anomaly.joblib               # Serialized RandomForest detector + StandardScaler
│   ├── distilbert_fault/            # Fine-tuned DistilBERT weights & tokenizer files
│   └── rag_index/                   # Persisted RAG embeddings, metadata, and passages
├── outputs/
│   ├── eval_report.md               # 5-fold CV and XAI alignment evaluation report
│   └── finetune_report.md           # DistilBERT vs RandomForest comparison report
├── src/
│   ├── api/
│   │   ├── main.py                  # FastAPI REST service
│   │   └── pipeline.py              # End-to-end query router and execution coordinator
│   ├── chat/
│   │   ├── backends.py              # Pluggable LLM backend abstraction layer
│   │   ├── intent.py                # Pre-pipeline keyword scope classifier
│   │   ├── prompts.py               # Prompt templates and payload serializers
│   │   └── synthesize.py            # Synthesis orchestrator with graceful fallback
│   ├── data/
│   │   ├── generate_synthetic.py    # Synthetic dataset generator (seed=42)
│   │   ├── loaders.py               # Typed pandas data loaders
│   │   └── schemas.py               # Pydantic v2 schemas and validation bounds
│   ├── explain/
│   │   ├── attention_explainer.py   # DistilBERT token self-attention visualizer
│   │   ├── lime_explainer.py        # LIME tabular local surrogate explainer
│   │   └── shap_explainer.py        # SHAP TreeExplainer feature attribution
│   ├── fusion/
│   │   ├── align.py                 # Temporal alignment of sensors, logs, and notes
│   │   └── fuse.py                  # Multimodal feature concatenation & ground-truth labeling
│   ├── model/
│   │   ├── anomaly.py               # AnomalyDetector (RandomForest / IsolationForest)
│   │   ├── bert_detector.py         # DistilBERT inference wrapper
│   │   ├── finetune_distilbert.py   # DistilBERT fine-tuning script
│   │   └── train_anomaly.py         # Training script for anomaly.joblib
│   ├── preprocess/
│   │   ├── logs.py                  # Log cleaning, counts, and alarm flag extraction
│   │   ├── notes.py                 # Note text cleaning and sentence embeddings
│   │   └── sensor.py                # Sensor imputation, denoising, scaling, windowing
│   ├── rag/
│   │   ├── build_index.py           # Index building CLI tool
│   │   ├── corpus.py                # Document chunker and corpus builder
│   │   ├── index.py                 # Dense vector index with SHA256 caching
│   │   └── retriever.py             # Hybrid ranking retriever (semantic + lexical)
│   └── reasoning/
│       ├── confidence.py            # Combined confidence calibration logic
│       ├── knowledge.py             # Curated YAML knowledge matcher
│       ├── param_advisor.py         # Deterministic physics engine & grid search
│       ├── recommend.py             # Action recommendation extractor
│       └── root_cause.py            # Heuristic root-cause ranking engine
├── tests/
│   ├── eval_mvp.py                  # Evaluation script for 5-fold CV and XAI alignment
│   ├── test_backends.py             # Tests for LLM backends, fallbacks, and config
│   ├── test_evaluation.py           # Tests for deterministic benchmarking metrics
│   ├── test_explain.py              # Tests for LIME, SHAP, and Attention explainers
│   ├── test_intent.py               # Tests for intent classification and scope guarding
│   ├── test_knowledge.py            # Tests for knowledge matching and query routing
│   ├── test_pipeline_quick.py       # Fast end-to-end smoke test
│   ├── test_rag.py                  # Tests for RAG chunking, indexing, and retrieval
│   └── test_welding.py              # Tests for parameter advisor physics and conversions
├── DECISIONS.md                     # Architectural Decision Records (ADRs)
├── FILES_EXPLAINED.md               # User-friendly explanation of every file and folder
├── MVP_PLAN.md                      # Historical MVP build roadmap (Tasks T0–T14)
├── PRD.md                           # Product Requirements Document
├── PROGRESS.md                      # Implementation progress log
├── PROJECT_CONTEXT.md               # This document (Master autonomous context file)
├── RAG.md                           # Technical specification of the RAG architecture
├── README.md                        # Project landing page and quick-start instructions
├── requirements.txt                 # Pinned Python package dependencies
├── SETUP.md                         # Device transfer, installation, and GPU setup guide
└── THESIS_GAP_ANALYSIS.md           # Academic proposal vs implementation analysis
```

---

## 6. TECHNOLOGIES, FRAMEWORKS, LIBRARIES & VERSIONS

### 6.1 Core Environment & Platform
- **Operating System:** Windows 10 / 11 (Verified on win32 architecture).
- **Python Version:** `Python 3.11.0` (Virtual environment installed at `.venv\`).

### 6.2 Python Dependencies (`requirements.txt`)
| Category | Library | Minimum Spec | Verified Environment Role |
|---|---|---|---|
| **Data & Arrays** | `numpy` | `~=1.26.0` | Numerical calculations, grid search arrays, vector math |
| | `pandas` | `~=2.1.0` | DataFrame manipulation, rolling aggregations, time-series merge |
| | `pyarrow` | `~=14.0.0` | Parquet file I/O for fused multimodal records |
| **Machine Learning** | `scikit-learn` | `~=1.4.0` | `RandomForestClassifier`, `StandardScaler`, K-Fold CV, metrics |
| | `joblib` | `~=1.3.0` | Model artifact serialization (`anomaly.joblib`) |
| **NLP & Deep Learning** | `transformers` | `~=4.40.0` | Hugging Face DistilBERT models and tokenizers |
| | `sentence-transformers`| `~=2.6.0` | Local embedding model (`all-MiniLM-L6-v2`) for RAG & notes |
| | `accelerate` | `~=1.1.0` | Backend accelerator required by HuggingFace Trainer |
| | `torch` | *(transitive)* | PyTorch runtime (CPU default; CUDA optional for training) |
| **Explainable AI (XAI)** | `shap` | `~=0.44.0` | `TreeExplainer` feature attribution |
| | `lime` | `~=0.2.0` | `LimeTabularExplainer` local surrogate modeling |
| | `matplotlib` | `~=3.8.0` | Headless plot generation (`Agg` backend) for benchmark PNGs |
| **LLM APIs & Cloud** | `anthropic` | `~=0.25.0` | Anthropic Claude SDK (Haiku / Opus) |
| | `groq` | `~=0.11.0` | Groq high-speed Llama API client |
| | `python-dotenv` | `~=1.0.0` | Automated `.env` parsing |
| **Data Validation** | `pydantic` | `~=2.6.0` | Data schema validation and typing (`BaseModel`, validators) |
| **Configuration** | `pyyaml` | `~=6.0.1` | YAML configuration file parsing |
| **Backend & Serving** | `fastapi` | `~=0.110.0` | REST API routing and request handling |
| | `uvicorn[standard]`| `~=0.27.0` | ASGI web server |
| **Frontend UI** | `streamlit` | `~=1.32.0` | Web application framework for chat interface |
| **Testing** | `pytest` | `~=8.0.0` | Test runner and test fixtures |
| | `pytest-cov` | `~=4.1.0` | Code coverage reporting |
| **Documentation** | `reportlab` | *(optional)* | PDF document generation (`docs/build_documentation.py`) |

### 6.3 AI / ML Models Deployed
1. **`models/anomaly.joblib`**: Supervised `RandomForestClassifier` trained on 39 aggregated features from fused sensor, log, and note data.
2. **`models/distilbert_fault/`**: Fine-tuned `distilbert-base-uncased` sequence classification model trained on serialized multimodal text windows.
3. **`sentence-transformers/all-MiniLM-L6-v2`**: Local 384-dimensional dense sentence encoder used for RAG passage indexing and operator note embedding (runs offline on CPU).
4. **`distilbert-base-uncased`**: Standard Hugging Face transformer evaluated in eager attention mode for token heatmaps.
5. **Generative LLM Models**:
   - `claude-haiku-4-5-20251001` (Anthropic API — default narration model)
   - `claude-opus-4-8` (Anthropic API — general manufacturing route)
   - `llama-3.3-70b-versatile` (Groq API)
   - `llama3.2` (Local Ollama)
   - `qwen2.5:3b` (Local Ollama)
   - `mistral:7b` (Local Ollama)

---

## 7. ENVIRONMENT & CONFIGURATION SPECIFICATIONS

### 7.1 Configuration Hierarchy
Configuration is loaded with strict precedence (highest to lowest):
1. **Explicit API parameters** passed to function calls.
2. **Environment variables** in `.env` or system environment.
3. **YAML configuration files** (`config/llm.yaml`, `config/welding_params.yaml`, etc.).
4. **Baked-in hardcoded defaults** in `src/chat/backends.py`.

### 7.2 Environment Variables (`.env`)
```bash
# Active LLM backend: anthropic | groq | ollama | ollama_llama | ollama_qwen | ollama_mistral
SYNTHESIZER_BACKEND=ollama

# Anthropic Configuration
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_GENERAL_MODEL=claude-opus-4-8

# Groq Configuration
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Ollama Local Configuration
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT=120
```

---

## 8. INSTALLATION, VERIFIED COMMANDS & RUN GUIDE

### 8.1 Setup & Installation from Scratch
```powershell
# 1. Create Python 3.11 virtual environment
python -m venv .venv

# 2. Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\activate

# 3. Upgrade pip and install pinned dependencies
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. Configure local environment variables
copy .env.example .env
# Edit .env with your preferred keys or backends
```

### 8.2 Data & Model Regeneration Commands (Deterministic)
If data or model artifacts are missing:
```powershell
# Generate synthetic raw datasets (data/raw/*.csv)
.\.venv\Scripts\python.exe src\data\generate_synthetic.py

# Re-fuse modalities into parquet (data/processed/fused.parquet)
.\.venv\Scripts\python.exe -c "from src.data.loaders import load_all; from src.preprocess.sensor import preprocess_sensors; from src.preprocess.logs import preprocess_logs; from src.preprocess.notes import preprocess_notes; from src.fusion.fuse import fuse, save_fused; data = load_all(); sw, _ = preprocess_sensors(data['sensors']); lf = preprocess_logs(data['logs']); ne = preprocess_notes(data['notes']); save_fused(fuse(sw, lf, ne, data['ground_truth']))"

# Train and serialize the anomaly detector (models/anomaly.joblib)
.\.venv\Scripts\python.exe src\model\train_anomaly.py

# Build/rebuild the RAG vector index (models/rag_index/)
.\.venv\Scripts\python.exe src\rag\build_index.py "porosity in stainless steel"
```

### 8.3 Verified Working CLI Commands
The following exact commands were executed and verified during codebase audit:
- **Run Full Pytest Suite (143 Tests):**
  ```powershell
  .\.venv\Scripts\python.exe -m pytest -v
  ```
  *Result: 143 passed in ~65 seconds.*
- **Run Quick Pipeline Smoke Test:**
  ```powershell
  .\.venv\Scripts\python.exe tests\test_pipeline_quick.py
  ```
  *Result: Evaluates pipeline on machine_1, prints SHAP attributions, outputs PASS.*
- **Run Multi-Scenario Integration Demo:**
  ```powershell
  .\.venv\Scripts\python.exe demo\run_demo.py
  ```
  *Result: Runs 3 scenarios end-to-end with full diagnostic outputs.*
- **Launch Streamlit Web UI:**
  ```powershell
  .\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
  ```
  *Opens browser at `http://localhost:8501`.*
- **Launch FastAPI REST Backend:**
  ```powershell
  .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
  ```
  *Interactive OpenAPI documentation at `http://localhost:8000/docs`.*
- **Run Multi-Model Comparative Benchmark:**
  ```powershell
  .\.venv\Scripts\python.exe evaluation\benchmark.py --backends ollama_qwen ollama_mistral --limit 5
  ```

### 8.4 Attempted But Problematic Commands (Caveats)
1. **Direct `pytest.exe` invocation:** Calling `.\.venv\Scripts\pytest.exe` directly from PowerShell exited with return code 1 with suppressed error output due to Windows console redirection. **Always use** `.\.venv\Scripts\python.exe -m pytest -v`.
2. **Groq Batch Rate Limits:** Running 64 back-to-back prompts through Groq on a free-tier API key hit rate limits, resulting in a 93.75% fallback rate. Use local Ollama models (`ollama_qwen`, `ollama_mistral`) or Anthropic Claude for extensive batch evaluations.
3. **`localhost` Socket Timeout in Ollama:** In Windows, calling `localhost` can incur an IPv6 $\rightarrow$ IPv4 resolution delay of up to 4 seconds. The code explicitly implements raw TCP socket probes via `127.0.0.1:11434` in `src/chat/backends.py:_ollama_reachable()` to eliminate this latency.

---

## 9. TESTING, EVALUATION & BENCHMARK RESULTS

### 9.1 Pytest Suite Breakdown (143 Passing Tests)
- `tests/test_backends.py` (11 tests): Verifies backend resolution, model config mapping, error handling, fallback safety, and parameter isolation.
- `tests/test_evaluation.py` (12 tests): Verifies automated metrics (faithfulness, hallucination, citation preservation, readability, cost, determinism).
- `tests/test_explain.py` (4 tests): Verifies SHAP, LIME, and DistilBERT attention map generators and payload formatting.
- `tests/test_intent.py` (14 tests): Verifies keyword/regex scope filtering, out-of-scope refusals, and edge cases.
- `tests/test_knowledge.py` (17 tests): Verifies knowledge base matching, summary building, and 3-way routing logic.
- `tests/test_rag.py` (7 tests): Verifies markdown chunking, corpus building, embedding persistence, and hybrid retrieval ranking.
- `tests/test_welding.py` (78+ tests): Comprehensive validation of parameter parsing, unit conversions, physics formulas, AWS/ISO window bounds, multi-turn memory merging, and unsupported material handling.

### 9.2 Anomaly Detection Evaluation (`outputs/eval_report.md`)
Evaluated via 5-Fold Stratified Cross-Validation on 1,917 synthetic windows (60 anomalous):
- **Accuracy:** 0.9969
- **Precision:** 0.9219
- **Recall:** 0.9833
- **F1 Score:** 0.9516
- **ROC-AUC:** 0.9992

### 9.3 XAI Driver Alignment
- **SHAP Alignment:** Top-5 SHAP drivers matched injected anomaly channels in **82% of overall cases** (100% on Arc Instability, Underheat, Gas Flow Failure, Overheating; 39% on Wire Feed Fault due to strong coupling between wire feed and current).
- **LIME Alignment:** Independent local surrogate matched injected channels in **67% of cases** (100% on Arc Instability, Underheat, Overheating; 75% on Gas Flow Failure; 0% on Wire Feed Fault).

### 9.4 Prescriptive Setpoint & Root-Cause Quality
- **Root-Cause MRR:** 0.878 (Top-1: 82%, Top-3: 100%).
- **Prescriptive Setpoint Compliance:** 100% within standards window (evaluated over 72 test conditions).
- **Mean Normalized Deviation:** 0.378.

### 9.5 Comparative LLM Benchmark Summary (`evaluation/results/comparison.csv`)
Benchmark evaluated on 64 prompts spanning all 5 routes:
| Backend | Model | Quality Score | Numeric Faithfulness | Hallucination Rate | Latency | Tokens/s | Est. Cost / 1k | Fallback Rate |
|---|---|---|---|---|---|---|---|---|
| **Anthropic** | `claude-haiku-4-5` | **0.851** | 0.970 | 0.030 | 10.86s | 47.1 | $4.10 | 0.00% |
| **Ollama Mistral** | `mistral:7b` | **0.812** | 0.972 | 0.028 | 37.78s | 7.4 | $0.00 | 3.12% |
| **Ollama Llama** | `llama3.2` | **0.776** | 0.956 | 0.044 | 9.45s | 27.2 | $0.00 | 0.00% |
| **Groq** | `llama-3.3-70b` | **0.769** | 1.000* | 0.000* | 0.91s | 748.0 | $1.10 | 93.75%* |
| **Ollama Qwen** | `qwen2.5:3b` | **0.758** | 0.985 | 0.015 | 9.02s | 34.6 | $0.00 | 0.00% |

*\*Note on Groq: The high fallback rate was triggered by rate limits during batch processing; metrics reflect fallback text for those occurrences.*

---

## 10. KNOWN BUGS, LIMITATIONS & TECHNICAL DEBT

1. **Synthetic Data Realism:** The sensor dataset is generated programmatically (`src/data/generate_synthetic.py`). While correlations, noise, and fault signatures follow real physical dynamics, real-world shop floors introduce non-stationary noise, electrical interference, and uncalibrated sensor drift.
2. **Wire Feed Fault XAI Driver Coupling:** In constant-voltage GMAW welding, wire feed rate and welding current are physically coupled. When wire feed fails, current drops simultaneously. In some evaluation windows, the models identified current drop rather than wire feed as the primary driver, lowering individual alignment scores for that fault type.
3. **Single-Fault Window Assumption:** The anomaly detection pipeline currently assigns one dominant fault label per window. In complex factory breakdowns, multiple compounding faults (e.g., gas failure leading to severe arc instability) can occur concurrently.
4. **No Real-Time Streaming Ingestion:** Data is currently ingested via batch files (CSV/Parquet). Streaming integration with MQTT, OPC-UA, or Kafka is not yet implemented.
5. **No Visual Vision-Language Modality:** Weld pool cameras or post-weld bead inspection images are not ingested; the pipeline is limited to time-series sensors, event logs, and operator text notes.

---

## 11. SECURITY, SAFETY & TRUST BOUNDARIES

1. **Strict Numeric Guardrail:** LLM system prompts strictly forbid inventing, calculating, or rounding any numbers. All parameter values must be quoted verbatim from the pipeline payload.
2. **Offline Privacy Capability:** The entire stack can run 100% offline using local Ollama models (`qwen2.5:3b`, `mistral:7b`) and local sentence-transformers embeddings. No proprietary factory telemetry or operational notes need to leave the plant network.
3. **Engineering Advisory Disclaimer:** The assistant explicitly presents numbers as optimal engineering starting points that must be verified against qualified Welding Procedure Specifications (WPS) and structural engineering codes.
4. **Secret Management:** API keys (`ANTHROPIC_API_KEY`, `GROQ_API_KEY`) must reside exclusively in `.env` and are strictly excluded from version control via `.gitignore`.

---

## 12. PRESERVED TECHNICAL DECISIONS & CRITICAL CONSTRAINTS

### What a Future AI Coding Assistant MUST NOT Break:
1. **DO NOT introduce LLM-generated numbers into the parameter or anomaly routes.** The core thesis defense hinges on all numbers originating from deterministic Python physics formulas (`src/reasoning/param_advisor.py`) or the scikit-learn model.
2. **DO NOT remove the deterministic fallback in `src/chat/synthesize.py`.** If an LLM call fails, the system must return `_fallback_text(payload)` rather than crashing or throwing a 500 error.
3. **DO NOT attach wear effects to travel speed.** In `config/parameter_effects.yaml` and `param_advisor.py`, contact tip/tungsten wear is physically driven by current and heat input—never travel speed.
4. **DO NOT leak Anthropic model overrides to other backends.** In `src/chat/synthesize.py`, `route_model` (e.g. `claude-opus-4-8` for general manufacturing) must only apply to `anthropic`/`claude`. Passing a Claude model string to Groq or Ollama causes immediate 404 API crashes.
5. **DO NOT remove conversation memory merging.** In `src/reasoning/param_advisor.py:merge_param_context()`, follow-up queries like *"now use 1.5mm wire"* must inherit the previous turn's material and thickness.

---

## 13. AI ASSISTANT HANDOFF

### What This Project Is
An industrial AI assistant specialized in arc welding that provides physics-grounded parameter optimization, multimodal machine anomaly detection with triple explainability (SHAP, LIME, Attention), and hybrid RAG-retrieved knowledge advice.

### Current State
Fully functional, thoroughly evaluated, and completely tested (143/143 tests passing). The repository contains clean code, synchronized documentation, trained model artifacts, and pre-computed evaluation benchmarks.

### What Has Already Been Done
- Complete data generation, ingestion, windowing, and multimodal fusion.
- Supervised RandomForest anomaly detector + DistilBERT sequence classifier fine-tuning.
- Triple XAI implementation: SHAP TreeExplainer, LIME Tabular, and DistilBERT self-attention heatmaps.
- Physics-based parameter advisor with grid search, standards windows, and sensitivity analysis.
- Multi-source hybrid RAG retrieval engine with citation tracking.
- Pluggable synthesis layer supporting Anthropic, Groq, and 3 local Ollama models with automatic fallback.
- Streamlit interactive UI and FastAPI REST backend.
- Automated 64-prompt comparative evaluation benchmark with 12 deterministic metrics.

### What Should NOT Be Changed
- Do not refactor the core routing architecture in `src/api/pipeline.py:route_question()`.
- Do not bypass the intent guard in `src/chat/intent.py`.
- Do not modify the physics formulas in `src/reasoning/param_advisor.py` without consulting AWS/ISO standards.
- Do not alter the column names or ordering expected in `src/model/anomaly.py:ALL_FEATURE_COLS`.

### Known Issues
- Groq API can experience rate-limiting during rapid batch evaluations (handled by fallback).
- LIME explainer requires ~1-2 seconds on CPU to perturb inputs and fit its local surrogate.
- Direct invocation of `pytest.exe` on Windows PowerShell can suppress error text; use `python -m pytest -v`.

### What to Verify Before Making Changes
1. Run `.\.venv\Scripts\python.exe -m pytest -v` to ensure all 143 tests are green.
2. Run `.\.venv\Scripts\python.exe tests\test_pipeline_quick.py` to confirm the full pipeline executes cleanly.
3. If touching parameter logic, verify `tests/test_welding.py` specifically.
4. If touching backends, verify `tests/test_backends.py`.

### Likely Next Tasks (Recommended Roadmap)
1. **Containerization:** Create a multi-stage `Dockerfile` and `docker-compose.yml` to package the FastAPI backend, Streamlit UI, and local model weights for edge deployment.
2. **Live Telemetry Ingestion:** Implement an MQTT / OPC-UA subscriber service to stream live sensor readings into sliding window queues.
3. **Cross-Attention Multimodal Transformer:** Replace simple feature concatenation in `src/fusion/fuse.py` with a true deep multimodal transformer fusion network (as outlined in the thesis gap analysis).
4. **Computer Vision Defect Inspection:** Integrate a Vision-Language Model (VLM) to analyze bead photographs and radiographic weld inspection images.

### Important Commands
- Run Tests: `.\.venv\Scripts\python.exe -m pytest -v`
- Run UI: `.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py`
- Run API: `.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload`
- Run Demo: `.\.venv\Scripts\python.exe demo\run_demo.py`
- Rebuild RAG Index: `.\.venv\Scripts\python.exe src\rag\build_index.py`

### Important Files to Inspect First
1. `src/api/pipeline.py` — The core orchestrator directing traffic across all routes.
2. `src/reasoning/param_advisor.py` — The welding physics and parameter optimization engine.
3. `src/chat/backends.py` — The LLM abstraction layer and fallback handlers.
4. `src/model/anomaly.py` — The tabular feature matrix and anomaly classifier.
5. `app/streamlit_app.py` — The user interface rendering cards, charts, and metrics.

---

## 14. VERIFICATION NOTES

### VERIFIED FROM SOURCE
- Verified all 5 query routes and their corresponding pipelines in `src/api/pipeline.py`.
- Verified exact formulas for Heat Input ($HI$) and Deposition Rate ($DR$) in `src/reasoning/param_advisor.py`.
- Verified feature set of 39 columns (30 sensor stats, 8 log stats, 1 note flag) in `src/model/anomaly.py`.
- Verified SHAP `TreeExplainer` and LIME `LimeTabularExplainer` implementations in `src/explain/`.
- Verified DistilBERT attention map extraction via `AutoModel.from_pretrained("distilbert-base-uncased", output_attentions=True)` in `src/explain/attention_explainer.py`.
- Verified RAG hybrid search weighting (0.60 semantic, 0.25 lexical, 0.15 title) in `src/rag/retriever.py`.
- Verified LLM backend registry, model resolution, and TCP socket probe in `src/chat/backends.py` and `config/llm.yaml`.
- Verified Pydantic validation bounds in `src/data/schemas.py`.

### VERIFIED BY COMMAND / TEST
- **Python Version:** Verified `Python 3.11.0` via `.\.venv\Scripts\python.exe --version`.
- **Test Suite:** Verified **143 / 143 passed** in 64.82s via `.\.venv\Scripts\python.exe -m pytest -v`.
- **Pipeline Execution:** Verified end-to-end execution of `run_pipeline()` via `.\.venv\Scripts\python.exe tests\test_pipeline_quick.py`.
- **Demo Script:** Verified execution of all 3 demo scenarios via `.\.venv\Scripts\python.exe demo\run_demo.py`.
- **Git Status & Branch:** Verified branch `feat/qwen-mistral-backends-eval` and HEAD commit `e18f5dd` via `git status` and `git log`.
- **Direct Pytest Caveat:** Verified that `.\.venv\Scripts\pytest.exe` exits with code 1, whereas `python -m pytest` executes with code 0.

### INFERRED
- **Synthetic Data Generation Origin:** Inferred from `src/data/generate_synthetic.py` that timestamps are pinned around May 2026 to create a reproducible, static 7-day timeline.
- **Groq Fallback Behavior:** Inferred from `evaluation/results/comparison.csv` that Groq's 93.75% fallback rate was due to API tier rate-limits rather than code regressions.

### UNKNOWN / NEEDS VERIFICATION
- **Specific Hardware on Target Factory Floor:** Unknown whether the target production facility operates Windows or Linux on their shop floor edge gateways (the current repository is optimized for Windows PowerShell and local paths).
- **Physical Sensor Interfacing Protocols:** Unknown which exact PLC protocols (Modbus TCP, Profinet, EtherCAT, or OPC-UA) will be utilized when transitioning from synthetic CSV inputs to physical welding machines.
- **Real Factory Noise Thresholds:** Exact signal-to-noise ratios and baseline sensor drift on real production welding lines need empirical validation when real factory CSV feeds become available.
