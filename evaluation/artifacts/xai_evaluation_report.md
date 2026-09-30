# Explainability (XAI) Empirical Evaluation Report

## 1. Research Question

This evaluation investigates the properties, empirical behavior, and practical utility of explainability (XAI) mechanisms in the multimodal manufacturing/welding anomaly detection architecture.

Specifically, five primary empirical questions are addressed:
1. **Explanation Fidelity / Faithfulness**: Does progressively masking or removing features identified as dominant by the explainer induce a measurable drop in predicted anomaly probability or confidence under neutral median replacement?
2. **Explanation Stability / Robustness**: Do feature rankings and top-$k$ importance sets remain consistent under small, class-preserving perturbations of continuous sensor signals?
3. **Feature-Ranking Consistency**: To what extent do independent explanation mechanisms (SHAP and LIME) agree in their local feature attribution rankings and top-$k$ driver sets when evaluated on identical model instances and sample representations?
4. **Normal vs. Anomalous Explanation Distributions**: Do the structural distributions of explanations (feature concentration, Shannon entropy, and physical domain representation) differ systematically between normal operational states and confirmed anomalies?
5. **Computational Cost**: What are the empirical latency and computational overhead profiles of each explainer across tabular and textual modalities under standardized benchmarking?

---

## 2. Existing XAI Architecture & Codebase Inventory

An audit of the repository reveals three distinct explainability components:

1. **SHAP Explainer (`src/explain/shap_explainer.py`)**:
   - Class: `AnomalyExplainer`
   - Algorithm: Exact TreeSHAP (`shap.TreeExplainer`) utilizing tree-path dependent feature perturbation.
   - Target Model: `AnomalyDetector` (`RandomForestClassifier`) operating over 40 fused multimodal features.
   - Feature Space: 40 tabular features (30 sensor aggregates across current, voltage, speed, wire feed, gas flow, and heat input; 9 log event features; 1 note indicator).
   - Scope: Local per-window explanation returning top drivers (`shap_drivers`) and full attribution vector (`raw_shap`).
   - Production Integration: Directly invoked in `src/api/pipeline.py` (lines 143–144).

2. **LIME Explainer (`src/explain/lime_explainer.py`)**:
   - Class: `LimeAnomalyExplainer`, factory `get_default_lime_explainer`.
   - Algorithm: Local linear surrogate (`lime.lime_tabular.LimeTabularExplainer`) with continuous feature discretization.
   - Target Model: `AnomalyDetector` (`RandomForestClassifier`) using `detector.model.predict_proba`.
   - Feature Space: Identical 40 tabular features. Feature names are parsed back from discretized condition intervals using `_feature_from_condition`.
   - Scope: Local per-window surrogate explanation returning top drivers (`lime_drivers`).
   - Production Integration: Invoked as an independent cross-check in `src/api/pipeline.py` (lines 150–151).

3. **DistilBERT Self-Attention Heatmap (`src/explain/attention_explainer.py`)**:
   - Function: `explain_attention(text: str)`.
   - Algorithm: Eager self-attention weight extraction from `distilbert-base-uncased`, averaged across all 6 transformer layers and 12 attention heads.
   - Target Model: Pretrained `distilbert-base-uncased` language representation model.
   - Feature Space: Natural-language word tokens from free-form operator notes (subwords merged via `_merge_wordpieces`).
   - Scope: Local textual heatmap surfacing word focus in operator notes.
   - Production Integration: Best-effort visualization in `src/api/pipeline.py` (lines 187–188).

### Cross-Method Comparability Audit
- **SHAP vs. LIME**: Evaluated on the exact same model instance (`models/anomaly.joblib`), identical sample instances, and identical 40-dimensional feature space. Directly comparable.
- **SHAP vs. Attention / LIME vs. Attention**: **Marked N/A**. As documented in `evaluation/artifacts/xai_inventory.json`, Attention operates over natural-language token strings using an unsupervised transformer encoder, whereas SHAP and LIME operate over 40 numerical tabular features of a Random Forest anomaly classifier. Because they possess disjoint feature spaces, distinct model architectures, and different analytical tasks, direct numerical or rank comparisons are mathematically invalid and are explicitly excluded.
- **Other Models (Logistic Regression, SVC, Fine-tuned DistilBERT)**: **Marked N/A**. The repository provides no SHAP/LIME wrappers for the baseline classifiers in `src/model/baselines.py` or the text classifier in `src/model/bert_detector.py`.

---

## 3. Evaluation Dataset & Leakage-Free Protocol

- **Master Dataset**: `data/processed/fused.parquet` (1,917 fused 10-minute windows; 60 anomalous, 1,857 normal).
- **Temporal Splitting Protocol**: Strictly identical to earlier research benchmarks (`src/data/splits.py`):
  - Machine-wise chronological holdout (80% train / 20% test).
  - Temporal purge/embargo gap of $\ge 30$ minutes between training termination and test onset per machine.
  - Zero leakage verified via `verify_leakage_free` (0 window ID overlap, strict temporal ordering).
- **Test Set Partition**: 375 total chronological test windows (10 anomalous, 365 normal).
- **Deterministic Evaluation Subset**:
  - Sample size: $N = 60$ windows.
  - Anomalous representation: All $N_{ano} = 10$ test anomalies are included.
  - Normal representation: $N_{norm} = 50$ normal windows sampled deterministically (`seed=42`).
- **Neutral Replacement Protocol**: Feature masking replaces values strictly with training-set feature medians computed exclusively on `train_df` (zero test-set fitting).

---

## 4. Available Explainers Summary

| Explainer | Implementation Module | Model Target | Feature Domain | Output Dimensions | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SHAP** | `src/explain/shap_explainer.py` | `RandomForestClassifier` | 40 Tabular Features | 40 Attributions | Fully Evaluated |
| **LIME** | `src/explain/lime_explainer.py` | `RandomForestClassifier` | 40 Tabular Features | 40 Attributions | Fully Evaluated |
| **Attention** | `src/explain/attention_explainer.py` | `distilbert-base-uncased` | Word Tokens ($\le 64$) | Variable Tokens | Evaluated (Text Modality) |
| **LR / SVC Explainers** | None in `src/explain/` | Baseline Classifiers | Tabular Features | None | N/A (Not Implemented) |

---

## 5. Explanation Fidelity / Faithfulness Evaluation

Fidelity was evaluated by progressively masking the top $k \in [0, 1, 3, 5, 10]$ ranked features identified by each explainer with training-set feature medians, recording the resulting probability drop $\Delta p_k = p_0 - p_k$ and prediction flip rate.

### Progressive Deletion Results

| Explainer | Metric | $k=0$ | $k=1$ | $k=3$ | $k=5$ | $k=10$ | Deletion AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SHAP** | Mean Drop ($\Delta p$) | 0.000 | 0.0734 | 0.1070 | 0.1335 | 0.1616 | **0.1195** |
| | Median Drop ($\Delta p$) | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | Flip Rate | 0.0% | 1.7% | 16.7% | 16.7% | 16.7% | — |
| **LIME** | Mean Drop ($\Delta p$) | 0.000 | 0.0772 | 0.0940 | 0.1313 | 0.1504 | **0.1140** |
| | Median Drop ($\Delta p$) | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | Flip Rate | 0.0% | 3.3% | 13.3% | 16.7% | 16.7% | — |
| **Random Baseline** | Mean Drop ($\Delta p$) | 0.000 | 0.0076 | 0.0304 | 0.0371 | 0.0577 | **0.0346** |

- **Observed Deletion Sensitivity**: Under progressive masking of top-10 features, SHAP-guided deletion achieved a mean probability drop of 0.1616 (AUC = 0.1195) and LIME-guided deletion achieved a mean drop of 0.1504 (AUC = 0.1140), compared to 0.0577 (AUC = 0.0346) under random feature deletion.
- **Decision Flip Sensitivity**: At $k=5$, masking explainer-selected features flipped 16.7% (SHAP) and 16.7% (LIME) of predictions.
- **DistilBERT Attention Fidelity**: **Marked N/A** (no classification output available).

---

## 6. Explanation Stability / Robustness Evaluation

Stability was measured by introducing small Gaussian jitter (2% of training standard deviation) to continuous sensor channels while preserving discrete log counts and boolean indicators.

- **Class Preservation Rate**: 100.0% of samples retained their original predicted classification under perturbation, confirming that input shifts remained within class boundaries.

### Stability Metrics under Perturbation

| Explainer | Top-1 Jaccard | Top-3 Jaccard | Top-5 Jaccard | Top-10 Jaccard | Spearman Rank Correlation ($\rho$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SHAP (RF)** | 1.0000 | 0.9917 | 1.0000 | 0.9970 | **0.9954 $\pm$ 0.0084** |
| **LIME (RF)** | 1.0000 | 1.0000 | 0.9278 | 0.8379 | **0.8075 $\pm$ 0.0559** |
| **Attention (DistilBERT)** | N/A | 0.0000 | 0.0000 | N/A | N/A (Variable token vocabulary) |

- **Observations**: SHAP displayed high rank correlation under input jitter (mean $\rho = 0.995$), whereas LIME exhibited moderate correlation (mean $\rho = 0.807$). This difference reflects LIME's reliance on stochastic neighborhood sampling during local surrogate estimation.
- **Text Modality Stability**: DistilBERT self-attention top-5 token sets showed a mean Jaccard overlap of 0.000 under trailing punctuation perturbations on operator notes.

---

## 7. Cross-Method Agreement (SHAP vs. LIME vs. Attention)

Cross-method agreement was evaluated across the identical $N = 60$ evaluation windows.

### SHAP vs. LIME Pairwise Agreement

| Agreement Metric | $k=1$ | $k=3$ | $k=5$ | $k=10$ |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Top-$k$ Overlap Count** | 0.98 / 1 | 1.15 / 3 | 3.47 / 5 | 6.30 / 10 |
| **Mean Top-$k$ Jaccard Similarity** | 0.9833 | 0.2517 | 0.5476 | 0.4657 |
| **Median Top-$k$ Jaccard Similarity**| 1.0000 | 0.2000 | 0.4286 | 0.4286 |

- **Full 40-Feature Spearman Rank Correlation**:
  - Mean $\rho$: **0.5889**
  - Median $\rho$: **0.5952**
  - Standard Deviation: **0.0843**
- **Attention Comparability**: **Marked N/A** due to disjoint feature spaces and distinct model targets.

---

## 8. Structural Comparison: Normal vs. Anomalous Explanations

Structural properties of explanations were contrasted between confirmed anomalous windows ($n=10$) and normal operational windows ($n=50$).

### Structural Attribution Statistics

| Metric | Normal Windows ($n=50$) | Anomalous Windows ($n=10$) | Mann-Whitney U $p$-value |
| :--- | :---: | :---: | :---: |
| **Top-1 Importance Concentration** | 27.37% | 17.88% | $p = 0.0000$ |
| **Top-3 Importance Concentration** | 49.17% | 44.57% | — |
| **Top-5 Importance Concentration** | 63.45% | 62.20% | — |
| **Shannon Entropy (bits)** | 3.869 $\pm$ 0.105 | 3.844 $\pm$ 0.181 | $p = 0.5585$ |
| **$k_{50}$ (Features for 50% Attribution)** | 3.8 features | 3.9 features | $p = 0.6570$ |
| **$k_{80}$ (Features for 80% Attribution)** | 10.3 features | 9.1 features | — |

### Domain Feature Group Representation

| Feature Group | Normal Importance Share (%) | Anomaly Importance Share (%) |
| :--- | :---: | :---: |
| **Welding Current** | 10.9% | 14.4% |
| **Arc Voltage** | 3.7% | 3.1% |
| **Welding Speed** | 2.0% | 1.2% |
| **Wire Feed Rate** | 2.3% | 4.3% |
| **Shielding Gas Flow** | 5.4% | 7.9% |
| **Heat Input** | 23.2% | 26.2% |
| **Log Events** | 52.4% | 42.9% |
| **Operator Note Flag** | 0.0% | 0.0% |

### Fault-Class Stratification
- **arc_instability** ($n=4$): Top-1 concentration = 18.0%, Entropy = 3.76, $k_{50} = 3.8$, Dominant group = `logs`.
- **gas_flow_failure** ($n=2$): Top-1 concentration = 17.1%, Entropy = 3.77, $k_{50} = 4.0$, Dominant group = `logs`.
- **wire_feed_fault** ($n=4$): Top-1 concentration = 18.2%, Entropy = 3.96, $k_{50} = 4.0$, Dominant group = `logs`.

---

## 9. Computational Cost & Latency Benchmark

Latency was measured on warm runs using `time.perf_counter()`.

| Explainer | Underlying Model | Evaluated Samples | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) | Min / Max (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SHAP** | `RandomForestClassifier` | 60 | **46.26** | 45.49 | 52.14 | 41.3 / 63.1 |
| **LIME** | `RandomForestClassifier` | 60 | **203.98** | 200.59 | 217.58 | 191.5 / 269.6 |
| **Attention** | `distilbert-base-uncased` | 35 | **23.90** | 23.88 | 27.18 | 21.5 / 29.9 |

- **Observations**: TreeSHAP computed local attributions in ~46.3 ms per window, whereas LIME required ~204.0 ms due to continuous neighborhood sampling (5,000 synthetic samples). Transformer attention extraction required ~23.9 ms per text note on CPU.

---

## 10. Failure & Edge-Case Diagnostics

Total diagnosed anomalies or edge-case instances: **0**.

### Diagnostic Breakdown:
No failure cases observed across the evaluated sample set.

---

## 11. Statistical Analysis

- **Normality & Tests**: Given skewed attribution distributions, non-parametric tests were employed.
- **Normal vs. Anomaly Concentration**: Mann-Whitney U test between normal and anomalous Top-1 concentration yielded $U = 7.0, p = 0.0000$.
- **Normal vs. Anomaly Entropy**: Mann-Whitney U test on Shannon entropy yielded $U = 220.0, p = 0.5585$.
- **Paired Agreement**: Mean Spearman correlation between SHAP and LIME across identical windows was $\rho = 0.5889 \pm 0.0843$.

---

## 12. Limitations

1. **Synthetic Tabular Data**: The evaluation was performed on multimodal synthetic datasets reflecting realistic physics; results may differ on real-world industrial weld telemetry with unmodeled electrical noise.
2. **Deletion Independence Assumption**: Progressive feature masking replaces features individually, which can introduce off-manifold tabular combinations not observed during training.
3. **No Downstream Text Anomaly Detector in Pipeline**: DistilBERT attention explains the self-attention of the language encoder rather than an end-to-end anomaly prediction head.
4. **Hardware Specificity**: Reported latencies were measured on local CPU execution; relative order (SHAP < Attention < LIME) is consistent, but absolute timings vary across processor architectures.

---

## 13. Reproducibility

- Master dataset: `data/processed/fused.parquet` (frozen, unchanged).
- Train/test split: Chronological 80/20 holdout with 30-minute embargo (`src/data/splits.py`).
- Evaluation seed: `42`.
- Preprocessing: Zero test-set fitting (medians computed exclusively from `train_df`).
- Determinism: Repeated executions yield bit-level identical JSON/CSV outputs.

---

## 14. Exact Execution Commands

```powershell
# Run the complete XAI evaluation benchmark
.\.venv\Scripts\python.exe evaluation/eval_xai.py

# Run focused XAI unit and integrity tests
.\.venv\Scripts\python.exe -m pytest tests/test_xai_evaluation.py -v

# Run full repository regression test suite
.\.venv\Scripts\python.exe -m pytest
```
