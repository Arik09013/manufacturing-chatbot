# Final Research & Reproducibility Audit

**Audit Date:** 2026-10-01T09:30:00Z  
**Audit Target:** Complete empirical evidence chain, artifacts, reproducibility pipelines, and cross-step consistency across Steps 1 through 12.  
**Repository:** `D:\E\manufacturing-chatbot`  
**Execution Environment:** Windows Server / Python 3.11 / PyTorch / Transformers / Scikit-Learn. Zero modification to production code, trained model weights, or master datasets.

---

## Executive Verdict

### **READY WITH DOCUMENTED FLAGS**

The empirical research repository exhibits outstanding methodological rigor, comprehensive experimental coverage, complete artifact preservation, and 100% test passing rates across both dedicated step test suites (116/116 passed) and the global test suite (259/259 passed).

The evidence chain across Steps 1–12 is empirically grounded, mathematically verified, and reproducible from committed code and data. No blocking code defects, data corruption, or computational fallacies were identified. However, several historical narrative artifacts, text-serialization sensitivities, and metric-aggregation distinctions require explicit qualification before inclusion in academic paper manuscripts. These flags are fully documented below with their authoritative empirical resolutions.

---

## 1. Step-by-Step Status

| Step | Title | Primary Artifacts | Dedicated Tests | Reproducibility | Status |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **Step 1** | Leakage-Free Evaluation | `evaluation/results/time_aware/` JSONs, `eval_report_time_aware.md` | `test_leakage_free.py` (7 tests) | REPRODUCIBLE | **COMPLETE** |
| **Step 2** | Stronger Baselines | `evaluation/results/baselines/` JSONs, `baseline_benchmark_report.md` | `test_baselines.py` (7 tests) | REPRODUCIBLE | **COMPLETE** |
| **Step 3** | Synthetic Robustness | `evaluation/results/robustness/` CSV/JSONs, `synthetic_robustness_report.md` | `test_robustness.py` (9 tests) | REPRODUCIBLE | **COMPLETE** |
| **Step 4** | Multimodal Ablation | `evaluation/results/ablation/` CSV/JSONs, `multimodal_ablation_report.md` | `test_ablation.py` (13 tests) | REPRODUCIBLE | **COMPLETE** |
| **Step 5** | Component Ablation | `evaluation/results/components/` CSV/JSONs, `component_ablation_report.md` | `test_component_ablation.py` (14 tests) | REPRODUCIBLE | **COMPLETE** (Flagged: Bert text shift) |
| **Step 6** | Model Comparison | `evaluation/results/comparison/` CSV/JSONs, `model_comparison_report.md` | `test_model_comparison.py` (9 tests) | REPRODUCIBLE | **COMPLETE** |
| **Step 7** | RAG Retrieval | `artifacts/rag_evaluation_results.csv`, `rag_evaluation_report.md` | `test_rag_evaluation.py` (8 tests) | PARTIALLY_REPRODUCIBLE | **COMPLETE** (Flagged: Scope of optimality) |
| **Step 8** | Intent Stress Benchmark | `artifacts/` & `evaluation/artifacts/intent_stress_*` | `test_intent_stress.py` (8 tests) | REPRODUCIBLE | **COMPLETE** (Reconciled) |
| **Step 9** | XAI Evaluation | `evaluation/artifacts/xai_evaluation_results.csv`, `xai_evaluation_report.md` | `test_xai_evaluation.py` (9 tests) | REPRODUCIBLE | **COMPLETE** (Flagged: Cross-modality scope) |
| **Step 10** | Physics Recommendation | `evaluation/artifacts/physics_evaluation_results.csv`, report | `test_physics_evaluation.py` (13 tests) | REPRODUCIBLE | **COMPLETE** |
| **Step 11** | Deployment Benchmark | `evaluation/artifacts/deployment_benchmark_results.csv`, report | `test_deployment_evaluation.py` (8 tests) | REPRODUCIBLE | **COMPLETE** (Flagged: 5/5 vs 4 named) |
| **Step 12** | Statistical Evaluation | `evaluation/artifacts/statistical_summary.json`, tests, report | `test_statistical_evaluation.py` (11 tests) | REPRODUCIBLE | **COMPLETE** |

---

## 2. Dataset Integrity

The canonical master dataset is frozen at [`data/processed/fused.parquet`](file:///D:/E/manufacturing-chatbot/data/processed/fused.parquet).

### Empirical Dataset Audit
* **Total Observations ($N$):** Exactly **1,917** windowed time-series samples across 48 schema columns.
* **Class Distribution:**
  * Normal Operational Windows: **1,857** (96.87%)
  * Confirmed Anomalous Windows: **60** (3.13% true prevalence)
* **Station / Machine Distribution:**
  * `station_1`: 639 windows (619 normal, 20 anomalous)
  * `station_2`: 639 windows (619 normal, 20 anomalous)
  * `station_3`: 639 windows (619 normal, 20 anomalous)
* **Partition Splits (Strictly Enforced):**
  * **Chronological Holdout (80/20):**
    * Training Set: **1,533** windows (50 anomalies)
    * Purged / Embargo Partition: **9** windows (30+ minute safety embargo preventing boundary leakage)
    * Test Set: **375** windows (10 anomalies: 4 arc instability, 3 wire feed fault, 2 underheat, 1 gas flow failure)
  * **Leave-One-Machine-Out (LOMO, $K=3$):** Each fold evaluates 1,278 training windows (40 anomalies) against 639 held-out test windows (20 anomalies).
* **Train-Only Transformations:** `StandardScaler` mean and variance vectors are strictly fitted on training partitions; automated assertions in `tests/test_leakage_free.py` verify that test feature values never alter transformer parameters.
* **Checkpoint & Data Modification Audit:** Checksum verification confirms that `data/processed/fused.parquet` and model checkpoints in `models/` have remained strictly unmodified throughout Steps 1–12.

### Resolution of the "70 vs 60 Anomalies" Historical Inconsistency
* **Investigation:** Exhaustive automated pattern scans across all git commits, commit logs, reports, and JSON artifacts disproved the existence of "70 anomalies" in any computation or authoritative report.
* **Finding:** The master dataset has always contained exactly **60 anomalies** (20 per station). In Step 3 multi-seed synthetic generation, alternative seeds produced small stochastic anomaly variations (`[60, 62, 63, 61, 62]`), but the master dataset `fused.parquet` remains exactly 60.
* **Conclusion:** The rumor of "70 anomalies" is an ungrounded external audit artifact; no calculation, table, or downstream artifact in the repository used 70 anomalies.

---

## 3. Model Evaluation Consistency

### Reconciliation: Step 2 RF Chronological F1 (0.8963) vs Step 5/6 RF Pooled F1 (1.0000)
A known apparent contradiction exists between the Random Forest chronological performance reported in Step 2 ($F_1 \approx 0.8963$) and that reported in Steps 4, 5, and 6 ($F_1 = 1.0000$).

* **Root-Cause Analysis:**
  * In **Step 2** ([`evaluation/eval_baselines.py`](file:///D:/E/manufacturing-chatbot/evaluation/eval_baselines.py#L122-L172)), the evaluation runner executed two distinct chronological protocols:
    1. `per_machine` (Unpooled): Trains three isolated models on single stations (~511 training windows, ~16 anomalies each) and evaluates each on its corresponding station test split (125 windows). The resulting per-station $F_1$ scores are:
       * Station 1: $F_1 = 0.8000$
       * Station 2: $F_1 = 0.8889$
       * Station 3: $F_1 = 1.0000$
       * **Unpooled Macro-Average across Stations:** $\frac{0.8000 + 0.8889 + 1.0000}{3} = \mathbf{0.8963}$.
    2. `pooled_all_machines`: Trains a single unified model on all pooled training windows ($N=1,533$, 50 anomalies) and evaluates on the pooled chronological test set ($N=375$, 10 anomalies). The resulting performance is **$F_1 = 1.0000$** (Precision = 1.0000, Recall = 1.0000, PR-AUC = 1.0000).
  * In **Steps 4, 5, and 6**, the benchmark evaluates the **multi-machine pooled model** on the pooled chronological test split.
* **Authoritative Finding:** Both values are 100% mathematically correct. They measure fundamentally different regimes:
  * $0.8963$ is the **unpooled single-station isolated model macro-average**.
  * $1.0000$ is the **cross-station pooled production model**.
* **Manuscript Guidance:** Papers must explicitly state whether numbers refer to single-machine isolated training ($F_1 = 0.8963$) or pooled multi-machine training ($F_1 = 1.0000$).

### Cross-Model Performance Baseline ($N=375$ Pooled Chronological)
* **Random Forest:** $F_1 = 1.0000$, Accuracy = 1.0000, PR-AUC = 1.0000
* **DistilBERT (theta=0.5):** $F_1 = 1.0000$, Accuracy = 1.0000, PR-AUC = 1.0000
* **Logistic Regression:** $F_1 = 0.9524$, Accuracy = 0.9973, Precision = 0.9091, Recall = 1.0000, PR-AUC = 0.9856 (1 false positive)
* **Support Vector Machine (RBF):** $F_1 = 0.8182$, Accuracy = 0.9893, Precision = 0.7500, Recall = 0.9000, PR-AUC = 0.9536 (3 false positives, 1 false negative)
* **Isolation Forest (Unsupervised):** $F_1 = 0.7143$, Accuracy = 0.9787, Precision = 0.5556, Recall = 1.0000, PR-AUC = 0.8501 (8 false positives)

---

## 4. Ablation Consistency

### Modality Ablation (Step 4)
* **Sensor Modality Indispensability:** Removing sensor features (`WITHOUT_SENSOR`) drops chronological $F_1$ from $1.0000$ to $0.7407$ for Random Forest ($\Delta = -0.2593$), $0.7407$ for Logistic Regression ($\Delta = -0.2117$), and $0.7407$ for DistilBERT ($\Delta = -0.2593$).
* **Log Event Indispensability:** Removing structured event logs (`WITHOUT_LOGS`) causes total collapse in text-serialized DistilBERT ($F_1: 1.0000 \to 0.0000$) and severe degradation in Random Forest ($F_1: 1.0000 \to 0.7826$).
* **Operator Notes Redundancy:** Removing unstructured operator notes (`WITHOUT_OPERATOR_NOTES`) causes zero degradation ($\Delta F_1 = 0.0000$) across all models, proving notes provide no marginal predictive value over tabular sensor and log features.

### Component Ablation Accounting & DistilBERT Out-of-Distribution Shift (Step 5)
* **Component Inventory Accounting:**
  * Exactly **19** pipeline components were audited in [`component_inventory.json`](file:///D:/E/manufacturing-chatbot/evaluation/results/components/component_inventory.json).
  * Exactly **7** components represent discrete quantitative feature/loss ablations:
    1. `WITHOUT_STANDARD_SCALER`
    2. `WITHOUT_RANGE_FEATURES`
    3. `WITHOUT_DOMAIN_HEAT_INPUT`
    4. `WITHOUT_LOG_EVENT_COUNTS`
    5. `WITHOUT_STD_FEATURES`
    6. `WITHOUT_MIN_MAX_FEATURES`
    7. `WITHOUT_BALANCED_CLASS_WEIGHTS`
  * Exactly **1** parametric sweep (`DECISION_THRESHOLD_SWEEP`) evaluates sensitivity across $\theta \in [0.1, 0.9]$.
  * The remaining **11** components represent post-hoc explainers, advisory reasoning, or conversational RAG modules operating downstream of the predictive classifier.
* **Component Effect Sizes:**
  * `WITHOUT_LOG_EVENT_COUNTS` represents the single largest component drop: Random Forest $F_1$ drops from $1.0000$ to $0.7440$ ($\Delta = -0.2560$, paired Cohen's $d_z = -19.0811$).
  * `WITHOUT_STANDARD_SCALER` has zero effect on Random Forest ($d=0.0$), but induces an 8.69% relative drop in Logistic Regression ($F_1: 0.9524 \to 0.8696$).
* **The DistilBERT Serialization Shift Flag:**
  * In Step 5, `FULL_COMPONENTS` DistilBERT achieved $F_1 = 0.1818$ rather than $1.0000$.
  * **Root Cause:** Fine-tuning in `src/model/bert_detector.py` serialized windows with `mean`, `std`, and `range`. Step 5's matrix generator introduced `min` and `max` tokens into the prompt string. This out-of-distribution textual prompt shift degraded the zero-shot inference of the frozen checkpoint.
  * **Verification:** In Step 5 condition `WITHOUT_MIN_MAX_FEATURES` (where `min` and `max` were omitted, matching the fine-tuning prompt template), DistilBERT immediately recovered to **$F_1 = 1.0000$**.

---

## 5. RAG Evaluation Consistency

* **Benchmark Corpus & Queries:** Audited across **41** specialized welding engineering passages (`data/knowledge_docs/`) and **40** expert-curated queries with binary and graded relevance judgments (`artifacts/rag_query_benchmark.json`).
* **Retrieval Completeness:**
  * Production Hybrid Retrieval achieves:
    * Recall@1: **0.8750** (35 / 40 queries)
    * Recall@3: **0.9750** (39 / 40 queries)
    * Recall@4: **1.0000** (40 / 40 queries)
    * Mean Reciprocal Rank (MRR): **0.9271**
* **Ablation Comparison:**
  * Dense-only: Recall@1 = 0.8000, MRR = 0.8925
  * Lexical-only (BM25): Recall@1 = 0.8250, MRR = 0.8833
  * Hybrid synergy yields a +0.0500 gain in Recall@1 and +0.0438 in MRR over pure lexical search.
* **Flagged Claims & Proper Scoping:**
  * Narrative reports described $\tau \in [0.15, 0.25]$ as the "Optimal Operating Window" that "guarantees 100% Recall@4".
  * **Audit Determination:** 100% empirical Recall@4 is observed on a discrete 40-query benchmark. As proven in Step 12, the exact Clopper-Pearson 95% confidence interval for 40/40 successes is $[91.19\%, 100.00\%]$.
  * **Manuscript Requirement:** Authors must replace "guaranteed" or "provably optimal" with "empirically preferred threshold range on the benchmark suite ($N=40$), with a 95% Clopper-Pearson lower bound of 91.19%".

---

## 6. Intent Evaluation Consistency

* **Benchmark Composition:** 100 stress queries (20 per intent: `anomaly`, `param`, `knowledge`, `general`, `out_of_scope`) encompassing 9 linguistic perturbation modes (typos, slang, code-switching, negation, ambiguity, etc.).
* **Reconciled Performance Metrics:**
  * Overall Classification Accuracy: **86.0% (86 / 100)**
  * Percentile Bootstrap 95% CI: **[79.00%, 93.00%]**
  * Binary Scope Guard Accuracy: **98.0% (98 / 100)**
  * Clopper-Pearson Exact 95% CI: **[92.96%, 99.76%]**
  * Macro $F_1$ Score: **0.8487**
  * Out-of-Scope Intent Rejection: **20 / 20 (100.0%)**
* **Multi-Directory Consistency Check:**
  * Cross-checking `artifacts/`, `evaluation/artifacts/`, and `outputs/` confirmed that `intent_stress_results.csv`, `intent_stress_results.json`, and `intent_stress_report.md` are **100% byte-for-byte identical** across all three locations.
  * Git working tree modifications reflect updated regex explanation strings (`matched_term` / `reason`), with zero change to predictions, ground-truth labels, or computed metrics.

---

## 7. XAI Evaluation Consistency

* **Sample Auditing:** Evaluated across **$N=60$** held-out test windows (10 confirmed anomalies, 50 normal operations) using TreeSHAP, LimeTabular, and DistilBERT self-attention.
* **Methodological Findings & Significance Control (Step 12 Linkage):**
  1. **Attribution Stability:** SHAP exhibits significantly higher top-5 Jaccard stability under 2% sensor noise jitter than LIME (Mean Jaccard $1.000$ vs $0.928$, Wilcoxon $W = 0.0$, raw $p = 0.00031$, Holm-adjusted $p_{\text{adj}} = 0.00155$).
  2. **Top-1 Attribution Concentration:** Anomalous windows exhibit significantly lower top-1 concentration than normal windows (Mean: $0.1788$ vs $0.2737$, Mann-Whitney $U = 7.0$, raw $p = 2.00 \times 10^{-6}$, Holm-adjusted $p_{\text{adj}} = 1.20 \times 10^{-5}$), reflecting distributed multi-feature root-cause signatures during welding failures.
  3. **Deletion Faithfulness:** While SHAP and LIME show raw unadjusted significance over random feature masking ($p < 0.05$), neither difference survives Holm-Bonferroni step-down correction ($p_{\text{adj}} = 0.13626$). SHAP vs LIME deletion AUC difference is likewise non-significant after correction ($W = 46.0, p_{\text{adj}} = 0.11038$).
* **Cross-Method Agreement Interpretation:**
  * Mean Spearman rank correlation across 40 features is $\rho = 0.5889 \pm 0.0843$.
  * At $k=5$, mean top-$k$ overlap is **3.47 / 5 features (69.4%)**, while the mean Jaccard similarity is **0.5476**.
  * **Reporting Requirement:** Do not conflate Jaccard similarity ($0.548$) with feature agreement percentage ($69.4\%$).
* **Attention Comparability Flag:**
  * DistilBERT self-attention operates on variable-length subword token embeddings ($\le 64$ tokens), whereas TreeSHAP and LIME operate on 40 continuous and discrete tabular features.
  * Attention is documented as **N/A** for direct numerical rank correlation against SHAP/LIME due to disjoint feature spaces and distinct targets.

---

## 8. Physics Evaluation Consistency

* **Deterministic Formulation:** Mathematical formulations for arc heat input ($\text{HI} = \frac{\eta \cdot 60 \cdot I \cdot V}{1000 \cdot S}$), wire deposition rate ($\text{DR} = \text{WFR} \cdot \pi (d/2)^2 \cdot \rho \cdot \eta_{\text{dep}}$), and triangular sweet-spot scoring are implemented in closed form without LLM intervention.
* **Empirical Concordance:**
  * Independent Solver Concordance: **100.00% (144 / 144 scenarios)**
  * 7x7 Grid Search Optimizer Concordance: **100.00% (144 / 144 scenarios)**
  * Engineering Constraints Compliance: **100.00% (641 / 641 checks)**
  * Boundary & Edge Case Handling: **100.00% (15 / 15 edge cases handled gracefully)**
* **Uncertainty Grounding:** As established in Step 12, exact Clopper-Pearson binomial 95% confidence intervals bound true population concordance to $[97.47\%, 100.00\%]$ (and $[78.20\%, 100.00\%]$ for edge cases). Sample success does not imply physical infallibility under unmodeled shop-floor dynamics.

---

## 9. Deployment Evaluation Consistency

* **Benchmark Query Profiling:** 80 curated queries across 5 routes evaluated for cold start, warm latency, memory footprint, throughput, and failure handling.
* **Bimodal Latency Architecture:**
  * Fast Deterministic Routes (`param`, `general`, `out_of_scope`, `knowledge`): Warm median latency operates between **0.25 ms and 17.96 ms** (P95 strictly $< 21$ ms), achieving up to 576 QPS.
  * Anomaly Diagnosis Route (`anomaly`): Warm median latency requires **3,300.44 ms** (P95: 3,430.13 ms) due to rolling sensor feature extraction (~350 ms) and LIME perturbation sampling (~570 ms).
  * Latency difference is statistically significant (Mann-Whitney $U = 935.0, p = 1.79 \times 10^{-8}$).
* **Linear Volume Scaling Scope:** Sequential batch throughput tests ($N=1$ to $N=100$) demonstrated linear time scaling ($O(N)$) and constant memory footprint. Papers must explicitly scope this to *single-thread sequential warm-batch execution* rather than concurrent multi-user ASGI production load.
* **Failure Scenarios Accounting Flag (5/5 vs 4 Named):**
  * In [`deployment_evaluation.py`](file:///D:/E/manufacturing-chatbot/evaluation/deployment_evaluation.py#L613-L687), exactly **5** failure scenarios are implemented and pass:
    1. `out_of_scope_guard`
    2. `offline_fallback_synthesis`
    3. `unsupported_material_param_advisor`
    4. `incompatible_process_rejection`
    5. `out_of_range_timestamp_handling`
  * [`deployment_summary.json`](file:///D:/E/manufacturing-chatbot/evaluation/artifacts/deployment_summary.json) correctly records `"failure_scenarios_passed": "5/5"`.
  * The markdown narrative in Section 13 of `deployment_evaluation_report.md` omitted bullet point 5 (`out_of_range_timestamp_handling`). Both the code and JSON authoritative sources confirm 5/5 passed.

---

## 10. Statistical Evaluation Consistency

Step 12 performed post-hoc uncertainty quantification, exact hypothesis testing, and multiple testing control. All four flags identified in Step 12.1 were verified resolved:

1. **FLAG 1 (Intent metrics):** Derived dynamically from computed results: Overall Accuracy = 86.0% (Bootstrap 95% CI: [79.0%, 93.0%]), Scope Guard = 98.0% (Clopper-Pearson: [92.96%, 99.76%]). Zero old pilot numbers (`94.0%`, `97.0%`) remain.
2. **FLAG 2 (XAI statistics):** Top-1 concentration Mann-Whitney $U = 7.0, p = 2.00 \times 10^{-6}$ (Holm $p_{\text{adj}} = 1.20 \times 10^{-5}$); SHAP vs LIME deletion $W = 46.0, p = 0.02759$ ($p_{\text{adj}} = 0.11038$). Zero pilot values (`U=81.0`, `W=178.50`) remain.
3. **FLAG 3 (Paired Cohen's $d_z$):** For `WITHOUT_LOG_EVENT_COUNTS` (RF), paired Cohen's $d_z = -19.0811$ ($F_1$ drops from 1.0000 to 0.7440, $-25.6\%$), calculated from paired differences on $n=5$ seeds.
4. **FLAG 4 (Deployment Mann-Whitney U):** Empirical Mann-Whitney $U = 935.0, p = 1.79 \times 10^{-8}$ between fast routes ($n=65$) and anomaly route ($n=15$), dynamically pulled from computed benchmark data.

---

## 11. Cross-Step Contradictions

| # | Topic & Exact Location | Observed Values | Authoritative Value | Empirical Impact | Recommended Treatment in Paper |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **RF Chronological F1**<br>`baseline_benchmark_report.md` vs `model_comparison_report.md` | Step 2 reports $F_1 = 0.8963$; Step 5/6 reports $F_1 = 1.0000$. | $0.8963$ is unpooled single-station model average; $1.0000$ is pooled multi-station model. | Low. Both are empirically true under distinct training protocols. | Explicitly report single-station isolated models ($F_1=0.8963$) separately from cross-station pooled training ($F_1=1.0000$). |
| **2** | **DistilBERT Step 5 Collapse**<br>`component_ablation_results.json` | $F_1 = 0.1818$ in Step 5 `FULL_COMPONENTS` vs $F_1 = 1.0000$ in Step 4/6. | Text prompt shift: fine-tuning template excluded `min`/`max`. Omitting `min`/`max` recovers $F_1 = 1.0000$. | Moderate. Misleading if reported as transformer inferiority. | Cite as empirical evidence of transformer vulnerability to prompt serialization drift under frozen checkpoints. |
| **3** | **RAG Optimality Overclaim**<br>`rag_evaluation_report.md` | Narrative states $\tau=0.15$ "guarantees 100% Recall@4" in an "Optimal Operating Window". | Sample success on 40 queries; Clopper-Pearson 95% CI is $[91.19\%, 100.00\%]$. | Low. Stylistic overclaim. | Rephrase to "empirically preferred threshold range on the benchmark suite ($N=40$)". |
| **4** | **Deployment Failure Count**<br>`deployment_evaluation_report.md` Section 13 | Narrative bullets list 4 scenarios; JSON summary reports 5/5. | 5 scenarios are implemented and pass in `deployment_evaluation.py`. | Negligible. Typo in report text. | Cite 5/5 passed failure scenarios as recorded in `deployment_summary.json`. |
| **5** | **XAI Agreement Interpretation**<br>`xai_evaluation_report.md` | Top-5 Jaccard overlap is $0.5476$; Top-5 overlap count is $3.47 / 5$ ($69.4\%$). | Both metrics valid; Jaccard penalizes set union size. | Low. Potential misinterpretation of agreement magnitude. | Report both top-$k$ intersection proportion ($69.4\%$) and formal Jaccard index ($0.548$) to avoid confusion. |

---

## 12. Reproducibility

* **Fully Reproducible Steps (11 of 12):** Steps 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, and 12 can be re-executed from scratch via their documented evaluation scripts using committed data and pinned seeds (`seed=42`).
* **Partially Reproducible Step (1 of 12):** Step 7 RAG retrieval metrics (Recall@K, MRR, parameter sweeps) reproduce deterministically. However, per-query reciprocal rank matrices across all 6 ablation conditions were not persisted to disk, which prevented paired per-query permutation significance testing in Step 12.
* **Non-Reproducible Steps:** 0.

---

## 13. Test Integrity

Automated test execution across dedicated step test suites:

```text
tests/test_leakage_free.py               7 passed
tests/test_baselines.py                  7 passed
tests/test_robustness.py                 9 passed
tests/test_ablation.py                  13 passed
tests/test_component_ablation.py        14 passed
tests/test_model_comparison.py           9 passed
tests/test_rag_evaluation.py             8 passed
tests/test_intent_stress.py              8 passed
tests/test_xai_evaluation.py             9 passed
tests/test_physics_evaluation.py        13 passed
tests/test_deployment_evaluation.py      8 passed
tests/test_statistical_evaluation.py    11 passed
--------------------------------------------------
Total Dedicated Step Tests:            116 PASSED (0 failed, 5 warnings, 67.38s)
Total Global Pytest Suite:             259 PASSED (0 failed, 5 warnings, 106.15s)
```

The 5 warnings are standard deprecation warnings from scikit-learn (`probability=True` in SVC and single-label confusion matrix sizing), none representing functional failures.

---

## 14. Git / Worktree Status

* **Branch:** `feat/qwen-mistral-backends-eval` (up to date with origin).
* **Recent Commit:** `6be4994 feat(eval): Step 12 - Statistical significance, uncertainty & effect-size analysis`.
* **Uncommitted Staged / Working Tree Changes:**
  * Step 8 intent stress files (`artifacts/intent_stress_*`, `evaluation/artifacts/intent_stress_*`, `outputs/intent_stress_report.md`): Working tree diff contains regex explanation text updates and timestamps; all classifications and metrics match HEAD.
  * Untracked audit artifacts:
    * `evaluation/artifacts/final_audit_inventory.json`
    * `evaluation/artifacts/final_cross_step_consistency.json`
    * `evaluation/artifacts/reproducibility_matrix.json`
    * `evaluation/artifacts/final_research_audit_report.md`
    * `artifacts/Manufacturing_AI_Chatbot_Case_Study.pdf`
    * `real_demo_interaction.json`
* **Zero Policy Violations:** No production code in `src/`, model weights in `models/`, or master data in `data/` were modified. No commits or pushes were performed.

---

## 15. Paper-Evidence Readiness

### A. Directly Supported Claims (Safe for High-Impact Paper Claims)
1. **Multimodal Synergy:** Sensor features are indispensable for defect detection ($-26\%$ drop upon removal). Operator notes provide zero incremental predictive value over tabular sensor and log features.
2. **Supervised vs Unsupervised Superiority:** Supervised models (Random Forest, Logistic Regression) statistically outperform unsupervised Isolation Forest on chronological holdout ($p < 0.01$).
3. **Cross-Station Generalization:** Station 3 exhibits measurable domain shift under LOMO cross-validation; Logistic Regression demonstrates superior cross-station stability ($CV = 1.42\%$) compared to Random Forest ($CV = 5.68\%$).
4. **Explainability Attribution Stability:** TreeSHAP exhibits statistically superior explanation stability over LIME under sensor noise perturbations ($p_{\text{adj}} = 0.00155$).
5. **Deterministic Physics Accuracy:** Parameter advisor recommendations exhibit 100% exact numerical concordance with independent mathematical formulations across 144 scenarios (Clopper-Pearson 95% CI: $[97.47\%, 100.00\%]$).
6. **Bimodal Deployment Architecture:** Sub-20 ms latency for deterministic query routes vs ~3.3 s for full sensor feature extraction and local LIME explanations ($p = 1.79 \times 10^{-8}$).

### B. Supported Claims with Explicit Limitations
1. **Model Equivalence (RF vs DistilBERT vs LogReg):** No statistically significant classification difference exists between supervised models on the clean chronological split ($p \ge 0.125$). *Limitation:* Chronological test set contains 10 anomalies ($N=375$); synthetic nature limits extrapolation to noisy physical factories.
2. **XAI Faithfulness Advantage:** While SHAP and LIME show nominal unadjusted superiority over random feature deletion ($p < 0.05$), neither difference survives Holm-Bonferroni multiple testing control ($p_{\text{adj}} = 0.136$).
3. **Transformer Deployment Feasibility:** Fine-tuned DistilBERT achieves parity ($F_1 = 1.0000$) on consistent prompt formats, but is highly sensitive to token prompt serialization shifts ($F_1 = 0.1818$) and requires 100x more compute than Random Forest.

### C. Currently Unsafe Claims (Do Not Make in Paper)
1. **"Guaranteed 100% RAG Retrieval":** Unsafe. Claim must be scoped to the 40-query benchmark with Clopper-Pearson bound $[91.19\%, 100.00\%]$.
2. **"TreeSHAP and Attention Agreement":** Unsafe. Attention is subword token attention from DistilBERT, not comparable to tabular feature SHAP values.
3. **"Linear Scaling under Factory Production Loads":** Unsafe. Linear throughput scaling was verified for single-thread sequential batches, not concurrent multi-user network load.
4. **"Flawless Physical Infallibility":** Unsafe. Physics recommendations are verified against discrete handbook tables, not continuous finite-element simulations or real-time weld pool sensor feedback.

---

## 16. Final Action List

### MUST FIX (Before Paper Submission)
1. **Distinguish Single-Station vs Pooled RF Performance:** Clearly state in the baseline results table that $F_1 = 0.8963$ corresponds to single-station unpooled models, whereas $F_1 = 1.0000$ represents multi-station pooled training.
2. **Clarify Step 5 DistilBERT Serialization Shift:** Explicitly document that DistilBERT's $F_1 = 0.1818$ in Step 5 resulted from adding un-trained `min`/`max` tokens into prompt strings, and cite `WITHOUT_MIN_MAX_FEATURES` ($F_1 = 1.0000$) as proof of recovery.
3. **Apply Multiple Testing Correction to XAI Claims:** Ensure the paper narrative reports Holm-adjusted $p$-values ($p_{\text{adj}} = 0.13626$), noting that explainer deletion faithfulness over random baseline is directional but not statistically significant after FWER control.

### SHOULD FIX (Recommended Quality Improvements)
1. **Clarify Deployment Failure Count:** Update report narrative from "four failure scenarios" to 5 failure scenarios matching `deployment_summary.json` and `deployment_evaluation.py`.
2. **Report Both Intersection Overlap and Jaccard Index for XAI:** Present both top-5 overlap count ($3.47 / 5 = 69.4\%$) and Jaccard similarity ($0.548$) to avoid confusion over agreement magnitude.
3. **Scope RAG Optimality:** Rephrase "Optimal Operating Window" to "Empirically Preferred Benchmark Threshold Range".

### OPTIONAL CLEANUP
1. **Commit Reconciled Intent Stress Files:** Clean git working tree by committing reconciled Step 8 intent stress explanation strings.
2. **Persist Per-Query RAG Rank Matrices:** In future benchmark extensions, save full 40x6 per-query rank matrices to enable paired permutation hypothesis tests.
