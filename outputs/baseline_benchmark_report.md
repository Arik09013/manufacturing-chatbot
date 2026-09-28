# Conventional Baseline Benchmark Report

> **Evaluation Harness:** Leakage-safe time-aware evaluation with >=30 min purge embargo and Leave-One-Machine-Out validation.
> **Methodology:** Preprocessing (StandardScaler) fitted strictly on training data; identical feature matrix (40 columns); zero parameter tuning on test sets.

---

## 1. Experimental Overview & Model Configurations

| Model Name | Category | Imbalance Strategy | Active Hyperparameters | Status |
|---|---|---|---|---|
| **Random Forest** | Tree Ensemble (Supervised) | class_weight='balanced' (inversely proportional to class frequencies) | `n_estimators=200`, `min_samples_leaf=2`, `class_weight=balanced` | **Executed** |
| **Logistic Regression** | Linear Model (Supervised) | class_weight='balanced' (heuristically scales loss inversely to class frequencies) | `class_weight=balanced`, `max_iter=1000`, `solver=lbfgs` | **Executed** |
| **Support Vector Machine** | Kernel Method (Supervised) | class_weight='balanced' (scales penalty parameter C per class) | `C=1.0`, `kernel=rbf`, `class_weight=balanced`, `probability=True` | **Executed** |
| **Isolation Forest** | Tree Ensemble (Unsupervised) | None (unsupervised anomaly isolation based on tree path length) | `n_estimators=200`, `contamination=auto` | **Executed** |
| **XGBoost** | Gradient Boosted Trees (Supervised) | scale_pos_weight | None | *Not Executed (Unavailable)* |
| **LightGBM** | Gradient Boosted Trees (Supervised) | class_weight='balanced' | None | *Not Executed (Unavailable)* |
| **CatBoost** | Gradient Boosted Trees (Supervised) | auto_class_weights='Balanced' | None | *Not Executed (Unavailable)* |

---

## 2. Chronological Per-Machine Holdout Benchmark

Evaluated on the chronological test partition ($N_{test}=375$ windows across 3 stations, 10 anomalies) separated by an embargo buffer of 30+ minutes.

### A. Macro-Average Results (Across 3 Machines)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | 0.9920 | 0.8222 | 1.0000 | 0.8963 | 1.0000 | 1.0000 |
| **Logistic Regression** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Support Vector Machine** | 0.9920 | 0.8222 | 1.0000 | 0.8963 | 1.0000 | 1.0000 |
| **Isolation Forest** | 0.9493 | 0.3889 | 1.0000 | 0.5531 | 0.9952 | 0.8736 |

### B. Multi-Machine Pooled Model Results ($N=375$, 10 Anomalies)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Confusion Matrix |
|---|---|---|---|---|---|---|---|
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[365, 0], [0, 10]]` |
| **Logistic Regression** | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | 1.0000 | `[[364, 1], [0, 10]]` |
| **Support Vector Machine** | 0.9893 | 0.7500 | 0.9000 | 0.8182 | 0.9986 | 0.9587 | `[[362, 3], [1, 9]]` |
| **Isolation Forest** | 0.9787 | 0.5556 | 1.0000 | 0.7143 | 0.9934 | 0.7251 | `[[357, 8], [0, 10]]` |

### C. Station-Specific Breakdown (Chronological Test)

| Model | Station | Train / Test | Anomalies (Tr/Te) | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Random Forest | `station_1` | 511 / 125 | 16 / 4 | 0.9840 | 0.6667 | 1.0000 | 0.8000 | 1.0000 |
| Random Forest | `station_2` | 511 / 125 | 16 / 4 | 0.9920 | 0.8000 | 1.0000 | 0.8889 | 1.0000 |
| Random Forest | `station_3` | 511 / 125 | 18 / 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | `station_1` | 511 / 125 | 16 / 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | `station_2` | 511 / 125 | 16 / 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | `station_3` | 511 / 125 | 18 / 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Support Vector Machine | `station_1` | 511 / 125 | 16 / 4 | 0.9920 | 0.8000 | 1.0000 | 0.8889 | 1.0000 |
| Support Vector Machine | `station_2` | 511 / 125 | 16 / 4 | 0.9840 | 0.6667 | 1.0000 | 0.8000 | 1.0000 |
| Support Vector Machine | `station_3` | 511 / 125 | 18 / 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Isolation Forest | `station_1` | 511 / 125 | 16 / 4 | 0.9120 | 0.2667 | 1.0000 | 0.4211 | 0.9917 |
| Isolation Forest | `station_2` | 511 / 125 | 16 / 4 | 0.9520 | 0.4000 | 1.0000 | 0.5714 | 0.9938 |
| Isolation Forest | `station_3` | 511 / 125 | 18 / 2 | 0.9840 | 0.5000 | 1.0000 | 0.6667 | 1.0000 |

---

## 3. Leave-One-Machine-Out (LOMO) Cross-Validation Benchmark

Evaluates cross-machine generalizability: trained on 2 stations ($N=1,278$, 40 anomalies), tested on the held-out 3rd station ($N=639$, 20 anomalies).

### A. Macro-Average Results (Across 3 Folds)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | 0.9979 | 0.9545 | 0.9833 | 0.9683 | 0.9993 | 0.9817 |
| **Logistic Regression** | 0.9995 | 0.9841 | 1.0000 | 0.9919 | 0.9998 | 0.9945 |
| **Support Vector Machine** | 0.9864 | 0.7076 | 0.9667 | 0.8167 | 0.9988 | 0.9669 |
| **Isolation Forest** | 0.9734 | 0.5416 | 0.9833 | 0.6983 | 0.9938 | 0.8108 |

### B. Pooled Out-of-Fold Predictions ($N=1,917$, 60 Anomalies)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Confusion Matrix |
|---|---|---|---|---|---|---|---|
| **Random Forest** | 0.9979 | 0.9516 | 0.9833 | 0.9672 | 0.9994 | 0.9822 | `[[1854, 3], [1, 59]]` |
| **Logistic Regression** | 0.9995 | 0.9836 | 1.0000 | 0.9917 | 0.9998 | 0.9942 | `[[1856, 1], [0, 60]]` |
| **Support Vector Machine** | 0.9864 | 0.7073 | 0.9667 | 0.8169 | 0.9984 | 0.9536 | `[[1833, 24], [2, 58]]` |
| **Isolation Forest** | 0.9734 | 0.5413 | 0.9833 | 0.6982 | 0.9934 | 0.7935 | `[[1807, 50], [1, 59]]` |

### C. Fold Breakdown (Held-Out Stations)

| Model | Held-Out Station | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Confusion Matrix |
|---|---|---|---|---|---|---|---|---|
| Random Forest | `station_1` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[619, 0], [0, 20]]` |
| Random Forest | `station_2` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[619, 0], [0, 20]]` |
| Random Forest | `station_3` | 0.9937 | 0.8636 | 0.9500 | 0.9048 | 0.9980 | 0.9451 | `[[616, 3], [1, 19]]` |
| Logistic Regression | `station_1` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[619, 0], [0, 20]]` |
| Logistic Regression | `station_2` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[[619, 0], [0, 20]]` |
| Logistic Regression | `station_3` | 0.9984 | 0.9524 | 1.0000 | 0.9756 | 0.9995 | 0.9836 | `[[618, 1], [0, 20]]` |
| Support Vector Machine | `station_1` | 0.9844 | 0.6923 | 0.9000 | 0.7826 | 0.9974 | 0.9287 | `[[611, 8], [2, 18]]` |
| Support Vector Machine | `station_2` | 0.9890 | 0.7407 | 1.0000 | 0.8511 | 0.9994 | 0.9809 | `[[612, 7], [0, 20]]` |
| Support Vector Machine | `station_3` | 0.9859 | 0.6897 | 1.0000 | 0.8163 | 0.9997 | 0.9910 | `[[610, 9], [0, 20]]` |
| Isolation Forest | `station_1` | 0.9750 | 0.5556 | 1.0000 | 0.7143 | 0.9943 | 0.8501 | `[[603, 16], [0, 20]]` |
| Isolation Forest | `station_2` | 0.9718 | 0.5263 | 1.0000 | 0.6897 | 0.9956 | 0.8543 | `[[601, 18], [0, 20]]` |
| Isolation Forest | `station_3` | 0.9734 | 0.5429 | 0.9500 | 0.6909 | 0.9914 | 0.7279 | `[[603, 16], [1, 19]]` |

---

## 4. Computational Efficiency & Model Complexity

| Model | Train Time (Pooled) | Inference Time (Test Pool) | Inference / Sample | Serialized Size | Complexity / Parameters |
|---|---|---|---|---|---|
| **Random Forest** | 0.4264 s | 0.1216 s | 0.3242 ms | 483,383 B | 5112 decision tree nodes across 200 trees |
| **Logistic Regression** | 0.0183 s | 0.0007 s | 0.0017 ms | 872 B | 41 linear coefficients + bias |
| **Support Vector Machine** | 0.0718 s | 0.0143 s | 0.0382 ms | 41,096 B | 117 support vectors (4680 float coordinates) |
| **Isolation Forest** | 0.3514 s | 0.0386 s | 0.1030 ms | 1,583,501 B | 16478 decision tree nodes across 200 trees |

---

## 5. Dependency Audit & Status

- **Installed & Benchmarked:** `Random Forest`, `Logistic Regression`, `Support Vector Machine`, `Isolation Forest` (via `scikit-learn==1.9.0`).
- **Unavailable Dependencies:**
  - `XGBoost`: Package not installed in `.venv`. Status: *Not executed — dependency unavailable*.
  - `LightGBM`: Package not installed in `.venv`. Status: *Not executed — dependency unavailable*.
  - `CatBoost`: Package not installed in `.venv`. Status: *Not executed — dependency unavailable*.

---

## 6. Reproducibility Commands

```powershell
# Run baseline benchmark suite
.\.venv\Scripts\python.exe evaluation/eval_baselines.py

# Run unit tests for baselines and leakage
.\.venv\Scripts\pytest tests/test_baselines.py tests/test_leakage_free.py -v
```
