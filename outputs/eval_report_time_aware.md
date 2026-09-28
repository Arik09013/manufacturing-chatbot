# Leakage-Safe Time-Aware Evaluation Report

> **Evaluation Strategy:** Defensible time-aware holdout and cross-machine validation.
> **Leakage Prevention:** 30-minute temporal embargo between train and test windows; StandardScaler fit strictly on training data.

---

## 1. Leakage Verification Summary

| Check | Status | Verification Detail |
|---|---|---|
| **Window ID Overlap** | **PASS** | 0 overlapping window IDs across train and test |
| **Temporal Embargo** | **PASS** | Exactly 30.0 min minimum separation for all machines |
| **Machine Isolation (LOMO)** | **PASS** | Zero train/test machine overlap across all 3 folds |
| **Train-Only Scaling** | **PASS** | StandardScaler fit strictly on training feature matrix |

### Chronological Partition Breakdown

- **Total Dataset Windows:** 1917 (60 anomalies)
- **Training Set (Earliest ~80%):** 1533 windows (50 anomalies)
- **Purged Embargo Gap:** 9 windows (0 anomalies)
- **Test Set (Latest ~20%):** 375 windows (10 anomalies)

---

## 2. Random Forest Chronological Evaluation

### A. Machine-Specific Models (Independent Train & Test per Machine)

| Machine | Train (Anom) | Test (Anom) | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Confusion Matrix [TN, FP, FN, TP] |
|---|---|---|---|---|---|---|---|---|---|
| **station_1** | 511 (16) | 125 (4) | 0.9840 | 0.6667 | 1.0000 | 0.8000 | 1.0000 | 1.0000 | `[119, 2, 0, 4]` |
| **station_2** | 511 (16) | 125 (4) | 0.9920 | 0.8000 | 1.0000 | 0.8889 | 1.0000 | 1.0000 | `[120, 1, 0, 4]` |
| **station_3** | 511 (18) | 125 (2) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[123, 0, 0, 2]` |
| **Mean Across Machines** | — | — | **0.9920** | **0.8222** | **1.0000** | **0.8963** | **1.0000** | **1.0000** | — |

### B. Multi-Machine Pooled Model (Trained on all 3 machines' chronological train set)

| Metric | Pooled Chronological Test Score |
|---|---|
| **Accuracy** | 1.0000 |
| **Precision** | 1.0000 |
| **Recall** | 1.0000 |
| **F1 Score** | 1.0000 |
| **ROC-AUC** | 1.0000 |
| **PR-AUC** | 1.0000 |
| **Confusion Matrix** | `[[365, 0], [0, 10]]` |
| **Sample Counts** | Train: 1533 (Anom: 50) · Test: 375 (Anom: 10) |

---

## 3. Leave-One-Machine-Out (LOMO) Cross-Validation

Evaluates cross-machine generalization (train on 2 stations, test on the held-out 3rd station).

| Held-Out Machine | Train Windows | Test Windows | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Confusion Matrix [TN, FP, FN, TP] |
|---|---|---|---|---|---|---|---|---|---|
| **station_1** | 1278 (40) | 639 (20) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[619, 0, 0, 20]` |
| **station_2** | 1278 (40) | 639 (20) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `[619, 0, 0, 20]` |
| **station_3** | 1278 (40) | 639 (20) | 0.9937 | 0.8636 | 0.9500 | 0.9048 | 0.9980 | 0.9451 | `[616, 3, 1, 19]` |
| **Macro Mean (3 Folds)** | — | — | **0.9979** | **0.9545** | **0.9833** | **0.9683** | **0.9993** | **0.9817** | — |
| **Pooled Predictions (N=1917)** | — | 1917 (60) | **0.9979** | **0.9516** | **0.9833** | **0.9672** | **0.9994** | **0.9822** | `[[1854, 3], [1, 59]]` |

---

## 4. Methodological Distinction vs Legacy Evaluation

| Dimension | Legacy MVP Evaluation (outputs/eval_report.md) | Revised Leakage-Safe Evaluation (This Report) |
|---|---|---|
| **Splitting Scheme** | Random `StratifiedKFold(n_splits=5, shuffle=True)` | Chronological 80/20 Holdout + Leave-One-Machine-Out |
| **Temporal Embargo** | None (0 min) — overlapping windows split across folds | **>=30 min purge gap** between train end and test start |
| **Feature Scaling** | Global `StandardScaler` fit on `station_1` before windowing | `StandardScaler` fit **strictly on training fold/split** |
| **Generalization Claim** | In-distribution random window interpolation | True temporal forecasting & out-of-machine generalization |
