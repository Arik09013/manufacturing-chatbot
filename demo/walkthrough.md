# Demo Walkthrough — Manufacturing Anomaly Chatbot MVP

## Definition of Done Scenario

**Query:** *"Why did line 3 slow down at 14:00?"*

**Expected pipeline flow:**

1. Intent parser extracts `machine_id = machine_3`, `query_time ≈ 14:00`
2. Sensor data for machine_3 is loaded, preprocessed into 30-min windows
3. Log events for the window are counted and alarm flags set
4. Anomaly detector (RandomForest) runs on the fused feature vector
5. SHAP values identify the top drivers (e.g., rpm_std, vibration_mean)
6. Root-cause mapper matches drivers to the `bearing_failure` cause entry
7. Recommendation retrieved: "Reduce speed, inspect bearing"
8. Confidence computed from anomaly probability + SHAP agreement
9. Claude (haiku) synthesizes the structured payload into a plain-language reply
10. Streamlit displays: answer + causes + recommendation + SHAP table + confidence badge

---

## How to Run the Demo

### Prerequisites

```bash
# 1. Activate virtual environment
.venv\Scripts\activate

# 2. Set your API key
copy .env.example .env
# Edit .env: ANTHROPIC_API_KEY=your_key

# 3. Generate synthetic data (if not already done)
python src/data/generate_synthetic.py

# 4. Train the anomaly model (if not already done)
python src/model/train_anomaly.py
```

### Option A — Streamlit UI (recommended for demo)

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

1. Type a query from `demo/demo_queries.txt` in the chat box
2. Press Enter to run
3. Read the plain-language answer
4. Expand "Root causes & recommendation" to see the structured output
5. Expand "SHAP explanation" to see which sensor signals drove the result

### Option B — FastAPI + curl

```bash
# Start the API
uvicorn src.api.main:app --reload --port 8000

# In another terminal:
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Why did line 3 slow down at 14:00?\"}"
```

### Option C — Python script

```python
import sys; sys.path.insert(0, '.')
from src.api.pipeline import run_pipeline
from src.chat.synthesize import synthesize

result = run_pipeline("Why did line 3 slow down at 14:00?")
print(synthesize(result))
print(result)
```

---

## What the System Does NOT Do (MVP Scope)

- No real-time data ingestion (runs on pre-generated synthetic CSV)
- No RAG / historical knowledge retrieval
- No LLM fine-tuning
- No cross-attention fusion (uses simple feature concatenation)
- No learned root-cause ranking (uses heuristic config map)
- No LIME or attention visualizations
- No MES/ERP integration
- No human-in-the-loop accept/reject UI

These are all deferred to later phases per the MVP plan.

---

## Key Files

| File | Purpose |
|---|---|
| `src/data/generate_synthetic.py` | Generate the tri-modal synthetic dataset |
| `src/model/train_anomaly.py` | Train the RandomForest anomaly detector |
| `app/streamlit_app.py` | Streamlit chat UI |
| `src/api/main.py` | FastAPI backend |
| `src/api/pipeline.py` | End-to-end inference pipeline |
| `src/chat/synthesize.py` | LLM synthesis via Claude API |
| `config/cause_action_map.yaml` | Root-cause / action heuristic map |
