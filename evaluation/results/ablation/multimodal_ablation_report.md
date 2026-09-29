# Multimodal Modality Ablation Study Report

> **Objective:** Quantify the individual and synergistic predictive contributions of each information modality in the welding cell.
> **Methodology:** Leakage-safe chronological holdout (>=30 min embargo) and Leave-One-Machine-Out (LOMO) cross-validation. Preprocessing (StandardScaler) fitted strictly on training data.

---

## 1. Modality-to-Feature Mapping & Architectural Role

| Modality | Features Count | Feature Columns / Representation | Pipeline Role & Destination |
|---|---|---|---|
| **Sensor Telemetry** | 30 | `welding_current`, `arc_voltage`, `welding_speed`, `wire_feed_rate`, `shielding_gas_flow`, `heat_input` (mean, std, min, max, range) | **Direct Model Feature** (Fed to RF, LR, and DistilBERT text serializer) |
| **Operational Logs** | 9 | `n_events`, `n_alarm`, `n_warning`, `n_maintenance`, `n_production`, `n_diagnostic`, `n_operational`, `has_alarm`, `has_warning` | **Direct Model Feature** (Event counts and status flags) |
| **Operator Notes** | 1 | `has_note` (Boolean indicator flag) | **Direct Model Feature** (Presence of unstructured operator record) |
| **Engineering Context** | 0 | Free-text standards, procedures (`config/welding_knowledge.yaml`, `data/knowledge_docs/`) | **RAG Retrieval ONLY** (Indexed in vector store for advisory chat; **not** fed to anomaly classifier) |

---

## 2. Chronological Ablation Benchmark Results

Evaluated on held-out chronological test partition ($N=375$ windows, 10 anomalies, 30+ min embargo).

| Condition | Modalities Included | Model | Features | Accuracy | Precision | Recall | F1 Score | PR-AUC | $\Delta$F1 vs FULL | Rel. $\Delta$F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| `FULL` | Sensor + Logs + Notes | **Random Forest** | 40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.0000** | **0.00%** |
| `FULL` | Sensor + Logs + Notes | **Logistic Regression** | 40 | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | **0.0000** | **0.00%** |
| `FULL` | Sensor + Logs + Notes | **DistilBERT** | 40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.0000** | **0.00%** |
| `SENSOR_ONLY` | Sensor only | **Random Forest** | 30 | 0.9840 | 0.6667 | 0.8000 | 0.7273 | 0.9032 | **-0.2727** | **-27.27%** |
| `SENSOR_ONLY` | Sensor only | **Logistic Regression** | 30 | 0.9867 | 0.6667 | 1.0000 | 0.8000 | 0.9714 | **-0.1524** | **-16.00%** |
| `SENSOR_ONLY` | Sensor only | **DistilBERT** | 30 | 0.9733 | 0.0000 | 0.0000 | 0.0000 | 0.7307 | **-1.0000** | **-100.00%** |
| `WITHOUT_SENSOR` | Logs + Notes | **Random Forest** | 10 | 0.9813 | 0.5882 | 1.0000 | 0.7407 | 0.8765 | **-0.2593** | **-25.93%** |
| `WITHOUT_SENSOR` | Logs + Notes | **Logistic Regression** | 10 | 0.9813 | 0.5882 | 1.0000 | 0.7407 | 0.8765 | **-0.2117** | **-22.23%** |
| `WITHOUT_SENSOR` | Logs + Notes | **DistilBERT** | 10 | 0.9813 | 0.5882 | 1.0000 | 0.7407 | 0.8713 | **-0.2593** | **-25.93%** |
| `WITHOUT_LOGS` | Sensor + Notes | **Random Forest** | 31 | 0.9867 | 0.6923 | 0.9000 | 0.7826 | 0.8795 | **-0.2174** | **-21.74%** |
| `WITHOUT_LOGS` | Sensor + Notes | **Logistic Regression** | 31 | 0.9867 | 0.6667 | 1.0000 | 0.8000 | 0.9714 | **-0.1524** | **-16.00%** |
| `WITHOUT_LOGS` | Sensor + Notes | **DistilBERT** | 31 | 0.9733 | 0.0000 | 0.0000 | 0.0000 | 0.7234 | **-1.0000** | **-100.00%** |
| `WITHOUT_OPERATOR_NOTES` | Sensor + Logs | **Random Forest** | 39 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.0000** | **0.00%** |
| `WITHOUT_OPERATOR_NOTES` | Sensor + Logs | **Logistic Regression** | 39 | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | **0.0000** | **0.00%** |
| `WITHOUT_OPERATOR_NOTES` | Sensor + Logs | **DistilBERT** | 39 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.0000** | **0.00%** |

### Not Applicable Condition Note:
- **`WITHOUT_ENGINEERING_CONTEXT`:** *Not applicable — engineering documents are utilized exclusively in RAG retrieval for advisory questions, never as model features.*

---

## 3. Leave-One-Machine-Out (LOMO) Cross-Validation Ablation Results

Evaluates cross-machine generalizability across all 3 station holdouts ($N_{train}=1,278$, $N_{test}=639$ per fold).

| Condition | Modalities Included | Model | Features | Macro Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro PR-AUC | $\Delta$Macro F1 | Rel. $\Delta$F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| `FULL` | Sensor + Logs + Notes | **Random Forest** | 40 | 0.9979 | 0.9545 | 0.9833 | 0.9683 | 0.9817 | **0.0000** | **0.00%** |
| `FULL` | Sensor + Logs + Notes | **Logistic Regression** | 40 | 0.9995 | 0.9841 | 1.0000 | 0.9919 | 0.9945 | **0.0000** | **0.00%** |
| `SENSOR_ONLY` | Sensor only | **Random Forest** | 30 | 0.9765 | 0.5930 | 0.8167 | 0.6824 | 0.8288 | **-0.2859** | **-29.53%** |
| `SENSOR_ONLY` | Sensor only | **Logistic Regression** | 30 | 0.9927 | 0.8158 | 1.0000 | 0.8972 | 0.9836 | **-0.0947** | **-9.55%** |
| `WITHOUT_SENSOR` | Logs + Notes | **Random Forest** | 10 | 0.9765 | 0.5946 | 1.0000 | 0.7393 | 0.8276 | **-0.2290** | **-23.65%** |
| `WITHOUT_SENSOR` | Logs + Notes | **Logistic Regression** | 10 | 0.9765 | 0.5946 | 1.0000 | 0.7393 | 0.8892 | **-0.2526** | **-25.47%** |
| `WITHOUT_LOGS` | Sensor + Notes | **Random Forest** | 31 | 0.9776 | 0.5995 | 0.8500 | 0.7008 | 0.8243 | **-0.2675** | **-27.63%** |
| `WITHOUT_LOGS` | Sensor + Notes | **Logistic Regression** | 31 | 0.9927 | 0.8158 | 1.0000 | 0.8972 | 0.9836 | **-0.0947** | **-9.55%** |
| `WITHOUT_OPERATOR_NOTES` | Sensor + Logs | **Random Forest** | 39 | 0.9990 | 0.9833 | 0.9833 | 0.9833 | 0.9820 | **+0.0150** | **+1.55%** |
| `WITHOUT_OPERATOR_NOTES` | Sensor + Logs | **Logistic Regression** | 39 | 0.9995 | 0.9841 | 1.0000 | 0.9919 | 0.9945 | **0.0000** | **0.00%** |

---

## 4. Empirical Modality Contribution Findings

1. **Sensor + Log Synergy:**
   - Removing operational logs (`SENSOR_ONLY` or `WITHOUT_LOGS`) causes an F1 degradation of **-0.2174 to -0.2727** for Random Forest and **-0.1524** for Logistic Regression in chronological evaluation.
   - In LOMO cross-validation, removing logs drops Random Forest macro F1 from **0.9683 to 0.6824** (-29.53% relative degradation).
2. **Sensor Necessity:**
   - Removing sensor features (`WITHOUT_SENSOR`) causes an F1 drop of **-0.2593** for Random Forest and **-0.2117** for Logistic Regression, confirming that operational logs alone cannot fully resolve thermal/electrical drift anomalies.
3. **Operator Note Impact:**
   - Removing the single `has_note` flag (`WITHOUT_OPERATOR_NOTES`) causes minimal change ($\Delta F_1 = 0.0000$ for RF and LR in chronological holdout, $+0.0150$ in LOMO), confirming that operator notes serve primarily as qualitative context for LLM explanation rather than a primary statistical anomaly discriminator.
4. **DistilBERT Behavior under Text Ablation:**
   - Under `SENSOR_ONLY` and `WITHOUT_LOGS`, DistilBERT's fixed 0.5 classification threshold produces 0 positive predictions ($F_1=0.0$), yet its ranking ability remains high ($ROC-AUC=0.993+$), indicating that prompt phrasing omission shifts the raw logit calibration distribution.

---

## 5. Reproducibility Commands

```powershell
# Run multimodal ablation benchmark suite
.\.venv\Scripts\python.exe evaluation/eval_ablation.py

# Generate visualization plots
.\.venv\Scripts\python.exe evaluation/plot_ablation.py

# Run ablation unit tests
.\.venv\Scripts\pytest tests/test_ablation.py -v
```
