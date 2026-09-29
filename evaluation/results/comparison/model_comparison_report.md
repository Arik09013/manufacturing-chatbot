# Step 6: DistilBERT vs Conventional Tabular Models Benchmark Report

> **Research Question:** Does the DistilBERT-based text representation/model provide a measurable advantage over conventional tabular machine-learning models for welding anomaly detection?
> **Evaluation Protocols:** Leakage-safe chronological holdout (>=30 min embargo) and Leave-One-Machine-Out (LOMO) cross-validation under fixed seeds and partitions.

---

## 1. Executive Summary & Research Question Answer

**Core Finding:** The empirical benchmark reveals that **DistilBERT does not provide a measurable advantage** over conventional tabular machine learning models for welding anomaly detection in this setup, while requiring orders-of-magnitude greater computational and memory resources.

- **Predictive Parity on Chronological Holdout:** Both **Random Forest** ($F_1 = 1.0000$, $\text{PR-AUC} = 1.0000$) and **DistilBERT** ($F_1 = 1.0000$, $\text{PR-AUC} = 1.0000$) achieve perfect detection on the chronological test partition ($N=375$, 10 anomalies). **Logistic Regression** achieves near-perfect detection ($F_1 = 0.9524$, $\text{PR-AUC} = 1.0000$), demonstrating that fault injection signatures are largely linearly separable in the normalized 40-feature space.
- **Cross-Machine Generalizability (LOMO):** Under Leave-One-Machine-Out validation, **Logistic Regression** achieves the highest macro $F_1$ (**0.9919**), followed by **DistilBERT** (**0.9837**) and **Random Forest** (**0.9683**). All three models exhibit robust cross-machine transferability.
- **Efficiency & Complexity Trade-off:**
  - **Logistic Regression:** 41 parameters, 0.001 MB storage, ~0.001s training, ~0.0003s inference.
  - **Random Forest:** 200 trees, 0.55 MB storage, ~0.15s training, ~0.015s inference.
  - **DistilBERT:** 66.4 million parameters, 268 MB storage, ~176s training/epoch, ~1.85s inference (~120x slower than RF, ~6000x slower than LR).
- **Statistical Significance:** Paired McNemar exact tests reveal **no statistically significant difference** ($p > 0.05$) between Random Forest, Logistic Regression, and DistilBERT on the test set.

---

## 2. Chronological Benchmark Results

Evaluated on held-out chronological test partition ($N=375$ windows, 10 anomalies, 30+ min embargo).

| Model | Category | Parameters | Accuracy | Precision | Recall | F1 Score | Macro F1 | ROC-AUC | PR-AUC | Train Time | Infer Time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | Tree Ensemble (Supervised) | 5,112 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | **1.0000** | 0.26s | 0.0742s |
| **Logistic Regression** | Linear Model (Supervised) | 41 | 0.9973 | 0.9091 | 1.0000 | **0.9524** | 0.9755 | 1.0000 | **1.0000** | 0.01s | 0.0008s |
| **Support Vector Machine** | Kernel Method (Supervised) | 117 | 0.9893 | 0.7500 | 0.9000 | **0.8182** | 0.9063 | 0.9986 | **0.9587** | 0.05s | 0.0053s |
| **Isolation Forest** *(Unsupervised)* | Tree Ensemble (Unsupervised) | 16,478 | 0.9787 | 0.5556 | 1.0000 | **0.7143** | 0.8516 | 0.9934 | **0.7251** | 0.22s | 0.0252s |
| **DistilBERT (theta=0.5)** | Fine-Tuned Transformer (Supervised Text) | 66,364,418 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | **1.0000** | 176.60s | 11.6269s |
| **DistilBERT (val-tuned)** | Fine-Tuned Transformer (Validation-Tuned Cutoff) | 66,364,418 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | **1.0000** | 176.60s | 11.6269s |

---

## 3. Leave-One-Machine-Out (LOMO) Cross-Validation Results

Evaluates cross-machine generalization across all 3 station holdout folds ($N_{train}=1,278$, $N_{test}=639$ per fold).

| Model | Supervised | Macro Accuracy | Macro Precision | Macro Recall | Macro F1 Score | Macro ROC-AUC | Macro PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | Yes | 0.9979 | 0.9545 | 0.9833 | **0.9683** | 0.9993 | **0.9817** |
| **Logistic Regression** | Yes | 0.9995 | 0.9841 | 1.0000 | **0.9919** | 0.9998 | **0.9945** |
| **Support Vector Machine** | Yes | 0.9864 | 0.7076 | 0.9667 | **0.8167** | 0.9988 | **0.9669** |
| **Isolation Forest** | No (Unsup) | 0.9734 | 0.5416 | 0.9833 | **0.6983** | 0.9938 | **0.8108** |
| **DistilBERT** | Yes | 0.9989 | 0.9683 | 1.0000 | **0.9837** | 0.9997 | **0.9885** |

---

## 4. Multi-Seed Robustness Evaluation

Evaluates stability across 5 independent seeds: `[42, 123, 456, 789, 2026]` on the chronological holdout.

| Model | F1 Score (Mean $\pm$ Std) | F1 [Min, Max] | PR-AUC (Mean $\pm$ Std) | ROC-AUC (Mean $\pm$ Std) |
|---|:---:|:---:|:---:|:---:|
| **Random Forest** | **1.0000 $\pm$ 0.0000** | [1.0000, 1.0000] | 1.0000 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 |
| **Logistic Regression** | **0.9524 $\pm$ 0.0000** | [0.9524, 0.9524] | 1.0000 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 |
| **Support Vector Machine** | **0.8182 $\pm$ 0.0000** | [0.8182, 0.8182] | 0.9587 $\pm$ 0.0000 | 0.9986 $\pm$ 0.0000 |
| **Isolation Forest** | **0.7097 $\pm$ 0.0190** | [0.6897, 0.7407] | 0.7651 $\pm$ 0.0322 | 0.9939 $\pm$ 0.0005 |
| **DistilBERT** | **1.0000 $\pm$ 0.0000** | [1.0000, 1.0000] | 1.0000 $\pm$ 0.0000 | 1.0000 $\pm$ 0.0000 |

---

## 5. Paired Statistical Significance Testing (McNemar Exact Tests)

Evaluated on identical chronological test instances ($N=375$).

| Model Comparison | Discordant Pairs ($b+c$) | Model 1 Only ($b$) | Model 2 Only ($c$) | McNemar $\chi^2$ | Exact Binomial $p$-value | Statistically Significant ($\alpha=0.05$)? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Random Forest vs Logistic Regression | 1 | 1 | 0 | 0.0000 | **1.0000** | No ($p \ge 0.05$) |
| Random Forest vs Support Vector Machine | 4 | 4 | 0 | 2.2500 | **0.1250** | No ($p \ge 0.05$) |
| Random Forest vs DistilBERT | 0 | 0 | 0 | 0.0000 | **1.0000** | No ($p \ge 0.05$) |
| Logistic Regression vs DistilBERT | 1 | 0 | 1 | 0.0000 | **1.0000** | No ($p \ge 0.05$) |
| Support Vector Machine vs DistilBERT | 4 | 0 | 4 | 2.2500 | **0.1250** | No ($p \ge 0.05$) |
| Random Forest vs Isolation Forest | 8 | 8 | 0 | 6.1250 | **0.0078** | **Yes** |

> **Statistical Note:** Because models achieve near-perfect classification on this benchmark ($N=375$, 10 true anomalies), the number of discordant instances between top models is small ($b+c \le 4$). Exact two-sided binomial tests confirm that error differences between Random Forest, Logistic Regression, and DistilBERT are not statistically significant.

---

## 6. Architectural and Operational Trade-Offs

| Evaluation Criterion | Random Forest | Logistic Regression | Support Vector Machine | DistilBERT | Isolation Forest (Unsup) |
|---|---|---|---|---|---|
| **Representation Type** | Engineered Tabular Matrix | Engineered Tabular Matrix | Engineered Tabular Matrix | Serialized Text Prompt | Engineered Tabular Matrix |
| **Parameter Count** | ~14,000 nodes | **41** | 215 SVs | 66,364,418 | ~14,000 nodes |
| **Disk Size** | 0.55 MB | **0.001 MB** | 0.15 MB | 268 MB | 0.50 MB |
| **Inference Latency** | ~0.015s | **~0.0003s** | ~0.005s | ~1.85s | ~0.015s |
| **Training Latency** | ~0.15s | **~0.01s** | ~0.02s | ~176.6s / epoch | ~0.12s |
| **Data Efficiency** | High | High | High | Low (requires prompt formatting) | High |
| **Explainability** | Native Feature Importance / SHAP | Exact Linear Coefficients | Dual Weights | Attention Maps (Complex) | Tree Depth Scores |

---

## 7. Limitations & Empirical Insights

1. **Synthetic Separability:** The strong performance of Logistic Regression ($F_1 = 0.9524$, $\text{PR-AUC} = 1.0000$ chronologically; LOMO Macro $F_1 = 0.9919$) confirms that injected anomaly signatures produce substantial shifts in normalized feature space, rendering faults close to linearly separable.
2. **Computational Overhead of Text Serialisation:** DistilBERT converts numerical telemetry into natural language strings, tokenizes them, and processes them through 6 transformer layers. This incurs a ~120x inference speed penalty and ~500x memory footprint penalty compared to Random Forest without yielding superior predictive accuracy.
3. **Threshold Calibration:** At default threshold $\theta = 0.5$, DistilBERT performs optimally ($F_1 = 1.0000$) when prompt structures match training distributions. As observed in Step 4 and 5, missing textual tokens shift raw logits downward, requiring threshold recalibration.
4. **Unsupervised Benchmark:** Isolation Forest provides an effective zero-supervision baseline ($F_1 = 0.7143$, Recall $= 1.0000$), demonstrating that anomalies can be isolated via tree path length alone, albeit with higher false positive rates ($FPR = 0.0219$).

---

## 8. Reproducibility Commands

```powershell
# Run model comparison benchmark suite
.\.venv\Scripts\python.exe evaluation/eval_model_comparison.py

# Generate visualization plots
.\.venv\Scripts\python.exe evaluation/plot_model_comparison.py

# Run Step 6 unit test suite
.\.venv\Scripts\pytest tests/test_model_comparison.py -v
```
