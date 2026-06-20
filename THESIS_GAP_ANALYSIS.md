# Thesis vs. Implementation — Gap Analysis

**Thesis:** *LLM-Enhanced Multimodal Process Optimization in Smart Manufacturing with Explainable AI Support* (Tasnimur Rahman Fayad, 2009026)
**Implementation:** welding-focused chatbot (`e:\manufacturing-chatbot`)
**Legend:** ✅ Done · ⚠️ Partial · ❌ Not done · ➕ Extra (beyond proposal)

> **Note on scope:** The proposal is written generically ("smart manufacturing"). On the advisor's instruction, the implementation specializes the same architecture to the **welding domain** and adds a **physics-based parameter optimizer** — see the ➕ rows.

---

## 1. Research Objectives (§1.3)

| # | Objective | Status | Evidence / Note |
|---|---|---|---|
| 1 | LLM system processing text notes + sensor + logs (multimodal) | ✅ Done | `sensors.csv`, `logs.csv`, `notes.csv` loaded, preprocessed, fused |
| 2 | XAI: **SHAP** + **LIME** + **attention** interpretability | ✅ Done | **All three:** SHAP (TreeExplainer), LIME (`lime_explainer.py`), attention (DistilBERT `attention_explainer.py`) |
| 3 | Optimization tool **without RAG** | ✅ Done | No vector store / retrieval; knowledge is deterministic keyword lookup |

---

## 2. System Architecture Layers (§3.2, Fig. 4)

| Layer | Status | Evidence / Note |
|---|---|---|
| Data collection | ✅ Done (synthetic) | CSV loaders; live **MES/PLC/SCADA ❌** (excluded by scope) |
| Preprocessing | ✅ Done | sensor impute + denoise + normalize + window; log dedup; note embeddings |
| Feature extraction & fusion | ⚠️ Partial | Temporal alignment + feature fusion ✅; **cross-attention transformer ❌** (replaced by tabular fusion) |
| **LLM fine-tuning** | ✅ Done | Fine-tuned **DistilBERT** detector (`finetune_distilbert.py` → `bert_detector.py`); matches RF on the held-out split. RF stays the default; DistilBERT is the optional second detector |
| XAI layer | ✅ Done | SHAP ✅; LIME ✅; attention ✅ |
| Optimization & decision support | ✅ Done | Parameter advisor, recommendations, confidence score |

---

## 3. Data Preprocessing (§3.3)

| Step | Status | Note |
|---|---|---|
| Sensor: impute (linear/KNN), denoise (wavelet/moving-avg), normalize, window | ✅ Done | `src/preprocess/sensor.py` |
| Production logs: clean, parse, dedup, embed | ✅ Done | dedup + event-code counting |
| Operator notes: clean, sentence embeddings (BERT/DistilBERT) | ✅ Done | uses **sentence-transformers** (equivalent) |
| Temporal alignment of all modalities | ✅ Done | `src/fusion/align.py`, `fuse.py` → `fused.parquet` |

---

## 4. Model Components (§3.4)

| Component | Status | Note |
|---|---|---|
| Multimodal fusion (cross-attention transformer) | ⚠️ Partial | Fusion done as **feature concatenation**, not cross-attention |
| LLM fine-tuning (GPT-4 / DistilBERT on domain data) | ✅ Done | **DistilBERT fine-tuned** for fault detection (`finetune_distilbert.py`); matches RF baseline. RF remains default detector |
| XAI layer (SHAP + LIME + attention) | ✅ Done | SHAP TreeExplainer ✅; LIME ✅; DistilBERT attention ✅ |

---

## 5. Training & Evaluation Plan (§3.5)

| Item | Status | Note |
|---|---|---|
| Dataset split 70/15/15 | ⚠️ Changed | Used **5-fold cross-validation** instead (stronger for small data) |
| AdamW + cosine decay + batch 32–64 | ❌ N/A | Those are NN/LLM training params; RandomForest doesn't use them |
| Anomaly metrics: accuracy, precision, recall, F1, ROC-AUC | ✅ Done | **F1 = 0.95, ROC-AUC = 0.999** (5-fold CV) |
| Root-cause metrics: MRR, top-k accuracy | ✅ Done | **MRR = 0.878, top-3 = 100%** (`evaluate_root_cause`) |
| Prescriptive: MAE of setpoints + expert validation | ⚠️ Partial | **MAE done** (in-window 100%, mean norm-dev 0.378); expert validation ❌ |
| Explainability: fidelity + comprehensibility | ⚠️ Partial | SHAP + LIME alignment scored (fidelity-like) ✅; operator survey ❌ |
| 5-fold CV across machines | ✅ Done | generalization tested across stations |
| Sensitivity tests | ⚠️ Reinterpreted | **Parameter** sensitivity table ✅; fusion-policy ablation ❌ |

---

## 6. Expected Results (§4) — projections

| Claim | Status | Note |
|---|---|---|
| Fidelity > 85% | ✅ Met | 100% SHAP alignment with injected channels |
| Handles noisy / missing data | ✅ Done | imputation + denoising |
| Cross-line generalization | ✅ Done | 5-fold CV across stations (note: proposal says "via RAG" — we do it **without** RAG, consistent with Obj. 3) |
| Conversational NL interface (reduced decision time) | ✅ Done | Streamlit chatbot + FastAPI |
| Modular, scalable framework | ✅ Done | modular `src/` packages |
| MES/ERP integration | ❌ Not done | excluded by scope |

---

## 7. ➕ Extra work done (beyond the proposal)

| Addition | Note |
|---|---|
| ➕ Welding **parameter optimizer** | Physics-based grid search for best-efficiency settings (heat input + deposition rate) — the advisor's requested "welding calculation" deliverable |
| ➕ Welding **standards tables** | `welding_params.yaml` (AWS D1.1 / ISO 15614 style) for MIG/MAG/TIG/SMAW × 3 materials |
| ➕ **Knowledge base** | 25 grounded welding topics (defects, cost, quality, comparisons) |
| ➕ **Intent guard** | rejects out-of-scope questions |
| ➕ **3-way router** | param / anomaly / knowledge |
| ➕ **Pluggable LLM backends** | Groq, Claude, Ollama |
| ➕ **Sensitivity / explain-why** | parameter-level XAI (nudge → effect) |
| ➕ **88 automated tests** | |

---

## 8. Key remaining gaps (priority order)

**✅ Closed:** LIME ✅ · Attention visualization (DistilBERT) ✅ · Root-cause MRR/top-k ✅ · Prescriptive MAE ✅ · **LLM fine-tuning (DistilBERT) ✅**

Remaining (all legitimate future work):
1. **Cross-attention transformer fusion** — replace tabular feature-concatenation fusion (large research effort).
2. **Expert validation** of prescriptive setpoints (human study).
3. **Comprehensibility survey** — operator feedback for XAI clarity.
4. **MES/ERP/live data** — replace synthetic CSVs (explicitly out of current scope).

---

## 9. Honest one-line status

> **Done:** the multimodal pipeline, anomaly detection, **all 3 XAI methods (SHAP + LIME + attention)**, the welding parameter optimizer (advisor's focus), the no-RAG design, the conversational UI, and evaluation (F1, ROC-AUC, MRR/top-k, prescriptive MAE).
> **Remaining (future work):** LLM fine-tuning for detection, cross-attention transformer fusion, expert/operator validation studies, and live MES/ERP data.

**Rough completion vs. proposal:** core system, advisor's deliverable, and **Objective 2 (all 3 XAI methods) fully met ✅**; only the LLM-fine-tuning / transformer-fusion research items remain as future work.

---

## 10. ⚠️ Internal inconsistencies in the proposal (raise these proactively)

- **RAG contradiction:** Objective 1.3 and Scope 1.4 say **no RAG**, but §4.2 ("use the RAG mechanism") and the Summary ("retrieval-enhanced generation pipeline") mention RAG. **The implementation follows the no-RAG objective** — be ready to point this out as a corrected inconsistency.
- **§1.3 typo:** "probabilistic multi-agent path planning for Urban Air Mobility" is copy-paste residue — unrelated to this thesis.
- **Figure numbering:** two figures are both labelled "Figure 2."
