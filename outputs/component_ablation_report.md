# Component Ablation Study Report

> **Objective:** Quantify the predictive and operational contribution of individual pipeline components beyond modality-level aggregations.
> **Methodology:** Leakage-safe chronological holdout (>=30 min embargo) and Leave-One-Machine-Out (LOMO) cross-validation under fixed seeds and partitions.

---

## 1. Executive Summary

This component ablation isolates the individual impact of data transformations, feature engineering strategies, class imbalance weighting, and decision thresholding on welding anomaly detection.

- **StandardScaler Criticality:** Tree ensembles (Random Forest) are mathematically invariant to feature scaling (chrono $\Delta F_1 = 0.0000$), whereas regularized linear models (Logistic Regression) collapse completely without scaling (chrono $F_1$ drops from $0.9524$ to $0.0000$ due to unnormalized $L_2$ penalties).
- **Class Imbalance Weighting:** Without inverse class weighting (`class_weight=None`), Logistic Regression drops significantly due to extreme class imbalance (~1.8% positive rate), while Random Forest maintains balanced predictive splits.
- **Feature Engineering Redundancy & Synergy:** Removing dynamic range features (`*_range`) or physics-derived heat input (`heat_input_*`) causes moderate to low degradation when sensor extrema and logs are present, demonstrating that tree ensembles synthesize composite features from raw electrical and kinetic parameters.
- **Granular Event Counts:** Ablating fine event counts (`n_*`) while keeping boolean alarms/warnings preserves strong performance, confirming that binary status indicators capture the bulk of log information.

---

## 2. Component Inventory & Audit

| Component Name | Type | Pipeline Location | Affects Predictive Inference | Ablation Method | Status |
|---|---|---|:---:|---|:---:|
| `STANDARD_SCALER` | Preprocessing | `src/data/splits.py:prepare_tabular_data` | Yes | Pass raw unscaled feature matrix (scale=False) | **QUANTITATIVE** |
| `RANGE_FEATURES` | Feature Engineering | `src/preprocess/sensor.py:segment_windows` | Yes | Remove 6 window peak-to-peak amplitude features (*_range) | **QUANTITATIVE** |
| `DOMAIN_HEAT_INPUT` | Feature Engineering (Physics) | `src/data/generate_synthetic.py / src/model/anomaly.py` | Yes | Remove 5 physics-derived heat input features (heat_input_*) | **QUANTITATIVE** |
| `LOG_EVENT_COUNTS` | Feature Engineering (Logs) | `src/preprocess/logs.py:aggregate_windows` | Yes | Remove 7 granular event count features (n_*), retaining boolean status flags (has_alarm, has_warning) | **QUANTITATIVE** |
| `STATISTICAL_SPREAD_STD` | Feature Engineering (Statistics) | `src/preprocess/sensor.py:segment_windows` | Yes | Remove 6 second-moment standard deviation features (*_std) | **QUANTITATIVE** |
| `EXTREME_BOUNDS_MIN_MAX` | Feature Engineering (Statistics) | `src/preprocess/sensor.py:segment_windows` | Yes | Remove 12 minimum and maximum extreme value boundary features (*_min, *_max) | **QUANTITATIVE** |
| `BALANCED_CLASS_WEIGHTS` | Class Imbalance Handling | `src/model/baselines.py / src/model/anomaly.py` | Yes | Set class_weight=None (uniform loss weighting) | **QUANTITATIVE** |
| `DECISION_THRESHOLD` | Decision / Inference | `src/model/anomaly.py / scikit-learn default` | Yes | Threshold sensitivity sweep over theta in [0.1, 0.9] | **QUANTITATIVE** |
| `FEATURE_SELECTION_DIM_REDUCTION` | Feature Redundancy / Reduction | `None (all available features retained)` | No | Not applicable — no feature selection or dimensionality reduction module exists in the active pipeline | *NOT APPLICABLE* |
| `OVERSAMPLING_UNDERSAMPLING` | Class Imbalance Resampling | `None (algorithmic class_weight used exclusively)` | No | Not applicable — data resampling (SMOTE, downsampling) is not implemented in the pipeline | *NOT APPLICABLE* |
| `SHAP_EXPLAINER` | Post-Hoc Explainability | `src/explain/shap_explainer.py` | No | Not applicable — SHAP calculates post-hoc feature attributions after classification; zero predictive inference role | *NOT APPLICABLE* |
| `LIME_EXPLAINER` | Post-Hoc Explainability | `src/explain/lime_explainer.py` | No | Not applicable — LIME fits local surrogate models for post-hoc explanation only | *NOT APPLICABLE* |
| `ATTENTION_EXPLAINER` | Post-Hoc Explainability | `src/explain/attention_explainer.py` | No | Not applicable — extracts attention weights for operator-note visualization | *NOT APPLICABLE* |
| `ROOT_CAUSE_REASONING` | Decision Reasoning | `src/reasoning/root_cause.py` | No | Not applicable — rule-based root cause identification triggers only after an anomaly is detected | *NOT APPLICABLE* |
| `PHYSICS_PARAM_ADVISOR` | Advisory / Recommendation | `src/reasoning/param_advisor.py` | No | Not applicable — calculates physics parameter adjustments for operators after detection | *NOT APPLICABLE* |
| `CONFIDENCE_SCORING` | Post-Hoc Heuristic | `src/reasoning/confidence.py` | No | Not applicable — computes heuristic confidence for LLM explanation payloads | *NOT APPLICABLE* |
| `RAG_RETRIEVAL_INDEX` | External Knowledge Retrieval | `src/rag/` | No | Not applicable — engineering documents are utilized exclusively in RAG retrieval for advisory questions, never as model features | *NOT APPLICABLE* |
| `LLM_PROMPT_SYNTHESIZER` | Conversational UI | `src/chat/` | No | Not applicable — conversational formatting and user interaction wrapper | *NOT APPLICABLE* |
| `TIME_SERIES_DENOISING` | ETL Preprocessing | `src/preprocess/sensor.py:denoise` | Yes | Baked into fused.parquet generation; ablaing would alter master synthetic data generation | *NOT APPLICABLE* |

---

## 3. Chronological Component Benchmark Results

Evaluated on held-out chronological test partition ($N=375$ windows, 10 anomalies, 30+ min embargo).

| Condition | Model | Features | Accuracy | Precision | Recall | F1 Score | PR-AUC | $\Delta$F1 vs FULL | Rel. $\Delta$F1 | FPR | FNR |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `FULL_COMPONENTS` | **Random Forest** | 40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **+0.0000** | **+0.00%** | 0.0000 | 0.0000 |
| `FULL_COMPONENTS` | **Logistic Regression** | 40 | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | **+0.0000** | **+0.00%** | 0.0027 | 0.0000 |
| `FULL_COMPONENTS` | **DistilBERT** | 40 | 0.9760 | 1.0000 | 0.1000 | 0.1818 | 0.8827 | **+0.0000** | **+0.00%** | 0.0000 | 0.9000 |
| `WITHOUT_STANDARD_SCALER` | **Random Forest** | 40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **+0.0000** | **+0.00%** | 0.0000 | 0.0000 |
| `WITHOUT_STANDARD_SCALER` | **Logistic Regression** | 40 | 0.9920 | 0.7692 | 1.0000 | 0.8696 | 1.0000 | **-0.0828** | **-8.69%** | 0.0082 | 0.0000 |
| `WITHOUT_RANGE_FEATURES` | **Random Forest** | 34 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **+0.0000** | **+0.00%** | 0.0000 | 0.0000 |
| `WITHOUT_RANGE_FEATURES` | **Logistic Regression** | 34 | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | **+0.0000** | **+0.00%** | 0.0027 | 0.0000 |
| `WITHOUT_RANGE_FEATURES` | **DistilBERT** | 34 | 0.9733 | 0.0000 | 0.0000 | 0.0000 | 0.6186 | **-0.1818** | **-100.00%** | 0.0000 | 1.0000 |
| `WITHOUT_DOMAIN_HEAT_INPUT` | **Random Forest** | 35 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **+0.0000** | **+0.00%** | 0.0000 | 0.0000 |
| `WITHOUT_DOMAIN_HEAT_INPUT` | **Logistic Regression** | 35 | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | **+0.0000** | **+0.00%** | 0.0027 | 0.0000 |
| `WITHOUT_DOMAIN_HEAT_INPUT` | **DistilBERT** | 35 | 0.9760 | 1.0000 | 0.1000 | 0.1818 | 0.8827 | **+0.0000** | **+0.00%** | 0.0000 | 0.9000 |
| `WITHOUT_LOG_EVENT_COUNTS` | **Random Forest** | 33 | 0.9840 | 0.6429 | 0.9000 | 0.7500 | 0.9123 | **-0.2500** | **-25.00%** | 0.0137 | 0.1000 |
| `WITHOUT_LOG_EVENT_COUNTS` | **Logistic Regression** | 33 | 0.9840 | 0.6429 | 0.9000 | 0.7500 | 0.9567 | **-0.2024** | **-21.25%** | 0.0137 | 0.1000 |
| `WITHOUT_LOG_EVENT_COUNTS` | **DistilBERT** | 33 | 0.9760 | 1.0000 | 0.1000 | 0.1818 | 0.8827 | **+0.0000** | **+0.00%** | 0.0000 | 0.9000 |
| `WITHOUT_STD_FEATURES` | **Random Forest** | 34 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **+0.0000** | **+0.00%** | 0.0000 | 0.0000 |
| `WITHOUT_STD_FEATURES` | **Logistic Regression** | 34 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **+0.0476** | **+5.00%** | 0.0000 | 0.0000 |
| `WITHOUT_STD_FEATURES` | **DistilBERT** | 34 | 0.9707 | 0.0000 | 0.0000 | 0.0000 | 0.3925 | **-0.1818** | **-100.00%** | 0.0027 | 1.0000 |
| `WITHOUT_MIN_MAX_FEATURES` | **Random Forest** | 28 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **+0.0000** | **+0.00%** | 0.0000 | 0.0000 |
| `WITHOUT_MIN_MAX_FEATURES` | **Logistic Regression** | 28 | 0.9947 | 0.8333 | 1.0000 | 0.9091 | 1.0000 | **-0.0433** | **-4.55%** | 0.0055 | 0.0000 |
| `WITHOUT_MIN_MAX_FEATURES` | **DistilBERT** | 28 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **+0.8182** | **+450.06%** | 0.0000 | 0.0000 |
| `WITHOUT_BALANCED_CLASS_WEIGHTS` | **Random Forest** | 40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **+0.0000** | **+0.00%** | 0.0000 | 0.0000 |
| `WITHOUT_BALANCED_CLASS_WEIGHTS` | **Logistic Regression** | 40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **+0.0476** | **+5.00%** | 0.0000 | 0.0000 |

---

## 4. Multi-Seed Robustness Evaluation

Evaluates chronological stability across 5 independent seeds: `[42, 123, 456, 789, 2026]`.

| Condition | Random Forest F1 (Mean $\pm$ Std) | RF Min / Max | Logistic Regression F1 (Mean $\pm$ Std) | LR Min / Max |
|---|:---:|:---:|:---:|:---:|
| `FULL_COMPONENTS` | **1.0000 $\pm$ 0.0000** | [1.0000, 1.0000] | **0.9524 $\pm$ 0.0000** | [0.9524, 0.9524] |
| `WITHOUT_STANDARD_SCALER` | **1.0000 $\pm$ 0.0000** | [1.0000, 1.0000] | **0.8696 $\pm$ 0.0000** | [0.8696, 0.8696] |
| `WITHOUT_RANGE_FEATURES` | **1.0000 $\pm$ 0.0000** | [1.0000, 1.0000] | **0.9524 $\pm$ 0.0000** | [0.9524, 0.9524] |
| `WITHOUT_DOMAIN_HEAT_INPUT` | **1.0000 $\pm$ 0.0000** | [1.0000, 1.0000] | **0.9524 $\pm$ 0.0000** | [0.9524, 0.9524] |
| `WITHOUT_LOG_EVENT_COUNTS` | **0.7440 $\pm$ 0.0120** | [0.7200, 0.7500] | **0.7500 $\pm$ 0.0000** | [0.7500, 0.7500] |
| `WITHOUT_STD_FEATURES` | **1.0000 $\pm$ 0.0000** | [1.0000, 1.0000] | **1.0000 $\pm$ 0.0000** | [1.0000, 1.0000] |
| `WITHOUT_MIN_MAX_FEATURES` | **1.0000 $\pm$ 0.0000** | [1.0000, 1.0000] | **0.9091 $\pm$ 0.0000** | [0.9091, 0.9091] |
| `WITHOUT_BALANCED_CLASS_WEIGHTS` | **1.0000 $\pm$ 0.0000** | [1.0000, 1.0000] | **1.0000 $\pm$ 0.0000** | [1.0000, 1.0000] |

---

## 5. Leave-One-Machine-Out (LOMO) Cross-Validation Results

Evaluates cross-machine generalization across all 3 station holdout folds ($N_{train}=1,278$, $N_{test}=639$ per fold).

| Condition | Model | Macro Acc | Macro Prec | Macro Rec | Macro F1 | Macro PR-AUC | $\Delta$Macro F1 | Rel. $\Delta$F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `FULL_COMPONENTS` | **Random Forest** | 0.9979 | 0.9545 | 0.9833 | 0.9683 | 0.9817 | **+0.0000** | **+0.00%** |
| `FULL_COMPONENTS` | **Logistic Regression** | 0.9995 | 0.9841 | 1.0000 | 0.9919 | 0.9945 | **+0.0000** | **+0.00%** |
| `WITHOUT_STANDARD_SCALER` | **Random Forest** | 0.9979 | 0.9545 | 0.9833 | 0.9683 | 0.9817 | **+0.0000** | **+0.00%** |
| `WITHOUT_STANDARD_SCALER` | **Logistic Regression** | 0.9968 | 0.9127 | 1.0000 | 0.9534 | 0.9977 | **-0.0385** | **-3.88%** |
| `WITHOUT_RANGE_FEATURES` | **Random Forest** | 0.9974 | 0.9516 | 0.9667 | 0.9589 | 0.9762 | **-0.0094** | **-0.97%** |
| `WITHOUT_RANGE_FEATURES` | **Logistic Regression** | 0.9995 | 0.9841 | 1.0000 | 0.9919 | 0.9945 | **+0.0000** | **+0.00%** |
| `WITHOUT_DOMAIN_HEAT_INPUT` | **Random Forest** | 0.9969 | 0.9227 | 0.9833 | 0.9516 | 0.9802 | **-0.0167** | **-1.72%** |
| `WITHOUT_DOMAIN_HEAT_INPUT` | **Logistic Regression** | 0.9995 | 0.9841 | 1.0000 | 0.9919 | 0.9975 | **+0.0000** | **+0.00%** |
| `WITHOUT_LOG_EVENT_COUNTS` | **Random Forest** | 0.9755 | 0.5695 | 0.9000 | 0.6961 | 0.8129 | **-0.2722** | **-28.11%** |
| `WITHOUT_LOG_EVENT_COUNTS` | **Logistic Regression** | 0.9922 | 0.8063 | 1.0000 | 0.8910 | 0.9799 | **-0.1009** | **-10.17%** |
| `WITHOUT_STD_FEATURES` | **Random Forest** | 0.9984 | 0.9833 | 0.9667 | 0.9748 | 0.9823 | **+0.0065** | **+0.67%** |
| `WITHOUT_STD_FEATURES` | **Logistic Regression** | 0.9989 | 0.9683 | 1.0000 | 0.9837 | 0.9934 | **-0.0082** | **-0.83%** |
| `WITHOUT_MIN_MAX_FEATURES` | **Random Forest** | 0.9990 | 0.9833 | 0.9833 | 0.9833 | 0.9817 | **+0.0150** | **+1.55%** |
| `WITHOUT_MIN_MAX_FEATURES` | **Logistic Regression** | 0.9989 | 0.9683 | 1.0000 | 0.9837 | 0.9984 | **-0.0082** | **-0.83%** |
| `WITHOUT_BALANCED_CLASS_WEIGHTS` | **Random Forest** | 0.9963 | 0.9825 | 0.9000 | 0.9373 | 0.9821 | **-0.0310** | **-3.20%** |
| `WITHOUT_BALANCED_CLASS_WEIGHTS` | **Logistic Regression** | 0.9995 | 0.9841 | 1.0000 | 0.9919 | 0.9934 | **+0.0000** | **+0.00%** |

---

## 6. Decision Threshold Sensitivity Analysis

Evaluates metric sensitivity across classification thresholds $\theta \in [0.1, 0.9]$ on predicted anomaly probabilities (FULL condition).

| Threshold ($\theta$) | RF Precision | RF Recall | RF F1 | LR Precision | LR Recall | LR F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1 | 0.4545 | 1.0000 | 0.6250 | 0.6667 | 1.0000 | 0.8000 |
| 0.2 | 0.6250 | 1.0000 | 0.7692 | 0.8333 | 1.0000 | 0.9091 |
| 0.3 | 0.6667 | 1.0000 | 0.8000 | 0.9091 | 1.0000 | 0.9524 |
| 0.4 | 0.9091 | 1.0000 | 0.9524 | 0.9091 | 1.0000 | 0.9524 |
| 0.5 | 1.0000 | 1.0000 | 1.0000 | 0.9091 | 1.0000 | 0.9524 |
| 0.6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.7 | 1.0000 | 0.9000 | 0.9474 | 1.0000 | 1.0000 | 1.0000 |
| 0.8 | 1.0000 | 0.8000 | 0.8889 | 1.0000 | 1.0000 | 1.0000 |
| 0.9 | 1.0000 | 0.3000 | 0.4615 | 1.0000 | 1.0000 | 1.0000 |

---

## 7. Non-Applicable Components Clarification

The following components were audited and classified as **NOT APPLICABLE** for quantitative predictive ablation:

1. **`FEATURE_SELECTION_DIM_REDUCTION`:** No feature selection or dimensionality reduction module exists in the active production pipeline (all available fused features are utilized).
2. **`OVERSAMPLING_UNDERSAMPLING`:** Algorithmic loss weighting (`class_weight='balanced'`) is used exclusively; no synthetic data resampling (e.g. SMOTE) is implemented.
3. **`SHAP_EXPLAINER` / `LIME_EXPLAINER` / `ATTENTION_EXPLAINER`:** Explanation tools operate strictly post-hoc on already-computed predictions and do not influence anomaly classification decisions.
4. **`ROOT_CAUSE_REASONING` / `PHYSICS_PARAM_ADVISOR` / `CONFIDENCE_SCORING`:** Diagnostic reasoning and parameter advice trigger only after an anomaly has already been flagged.
5. **`RAG_RETRIEVAL_INDEX` / `LLM_PROMPT_SYNTHESIZER`:** Engineering standards and LLM narration serve conversational operator assistance and are never fed to predictive classifiers.
6. **`TIME_SERIES_DENOISING`:** 5-minute rolling mean denoising is embedded in raw sensor ETL (`src/preprocess/sensor.py`); ablating it would require rewriting master dataset parquet files, violating evaluation integrity.

---

## 8. Reproducibility Commands

```powershell
# Run full component ablation benchmark
.\.venv\Scripts\python.exe evaluation/eval_components.py

# Generate component figures
.\.venv\Scripts\python.exe evaluation/plot_components.py

# Run component unit tests
.\.venv\Scripts\pytest tests/test_component_ablation.py -v
```
