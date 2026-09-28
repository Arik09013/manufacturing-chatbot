# Synthetic Data Realism & Robustness Evaluation Report

> **Objective:** Empirically quantify the stability and degradation profile of anomaly detection models under realistic non-ideal synthetic conditions.
> **Key Integrity Guarantee:** Perturbations applied strictly to test data post-split. Zero perturbation leakage into training or feature scaling. Ground-truth labels strictly unchanged.

---

## 1. Multi-Seed Dataset Stability (Seeds: 42, 123, 456, 789, 2026)

### A. Dataset Variance Across Seeds

| Seed | Total Windows | Anomaly Windows | Anomaly Rate | Station 1 Anom. | Station 2 Anom. | Station 3 Anom. | Purged Embargo Windows |
|---|---|---|---|---|---|---|---|
| `42` | 1,917 | 60 | 3.13% | 20 | 20 | 20 | 9 |
| `123` | 1,917 | 62 | 3.23% | 21 | 20 | 21 | 9 |
| `456` | 1,917 | 63 | 3.29% | 21 | 20 | 22 | 9 |
| `789` | 1,917 | 61 | 3.18% | 20 | 21 | 20 | 9 |
| `2026` | 1,917 | 62 | 3.23% | 21 | 21 | 20 | 9 |

### B. Model Performance Stability Across Seeds (Chronological Split)

| Model | F1 (Mean ± Std) | F1 Range [Min, Max] | ROC-AUC (Mean ± Std) |
|---|---|---|---|
| **Random Forest** | 0.9359 ± 0.0577 | [0.8571, 1.0000] | 0.9996 ± 0.0009 |
| **Logistic Regression** | 0.9618 ± 0.0326 | [0.9231, 1.0000] | 0.9994 ± 0.0011 |

---

## 2. Controlled Test-Set Perturbation Results

Evaluated on the chronological test set ($N=375$ windows, 10 anomalies, 30+ min embargo).

| Scenario | Category | Model | Accuracy | Precision | Recall | F1 Score | PR-AUC | FPR | ΔF1 vs Clean | ΔPR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| `CLEAN` | Reference | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `CLEAN` | Reference | **Logistic Regression** | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | 0.0027 | 0.0000 | 0.0000 |
| `CLEAN` | Reference | **DistilBERT** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `NOISE_1%` | Sensor Noise | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `NOISE_1%` | Sensor Noise | **Logistic Regression** | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | 0.0027 | 0.0000 | 0.0000 |
| `NOISE_3%` | Sensor Noise | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `NOISE_3%` | Sensor Noise | **Logistic Regression** | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | 0.0027 | 0.0000 | 0.0000 |
| `NOISE_5%` | Sensor Noise | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `NOISE_5%` | Sensor Noise | **Logistic Regression** | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | 0.0027 | 0.0000 | 0.0000 |
| `NOISE_5%` | Sensor Noise | **DistilBERT** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `MISSING_5%` | Missingness | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `MISSING_5%` | Missingness | **Logistic Regression** | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | 0.0027 | 0.0000 | 0.0000 |
| `MISSING_10%` | Missingness | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `MISSING_10%` | Missingness | **Logistic Regression** | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | 0.0027 | 0.0000 | 0.0000 |
| `MISSING_10%` | Missingness | **DistilBERT** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `MISSING_BLOCK` | Missingness | **Random Forest** | 0.9973 | 1.0000 | 0.9000 | 0.9474 | 0.9345 | 0.0000 | -0.0526 | -0.0655 |
| `MISSING_BLOCK` | Missingness | **Logistic Regression** | 0.9947 | 1.0000 | 0.8000 | 0.8889 | 0.8883 | 0.0000 | -0.0635 | -0.1117 |
| `DRIFT_2%` | Temporal Drift | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `DRIFT_2%` | Temporal Drift | **Logistic Regression** | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | 0.0027 | 0.0000 | 0.0000 |
| `DRIFT_5%` | Temporal Drift | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `DRIFT_5%` | Temporal Drift | **Logistic Regression** | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | 0.0027 | 0.0000 | 0.0000 |
| `DRIFT_5%` | Temporal Drift | **DistilBERT** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `SEVERITY_75%` | Fault Severity | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `SEVERITY_75%` | Fault Severity | **Logistic Regression** | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | 0.0027 | 0.0000 | 0.0000 |
| `SEVERITY_50%` | Fault Severity | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `SEVERITY_50%` | Fault Severity | **Logistic Regression** | 0.9973 | 0.9091 | 1.0000 | 0.9524 | 1.0000 | 0.0027 | 0.0000 | 0.0000 |
| `SEVERITY_50%` | Fault Severity | **DistilBERT** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `SEVERITY_25%` | Fault Severity | **Random Forest** | 0.9920 | 1.0000 | 0.7000 | 0.8235 | 0.9198 | 0.0000 | -0.1765 | -0.0802 |
| `SEVERITY_25%` | Fault Severity | **Logistic Regression** | 0.9760 | 0.6667 | 0.2000 | 0.3077 | 0.8196 | 0.0027 | -0.6447 | -0.1804 |
| `DROPOUT_10%` | Feature Dropout | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `DROPOUT_10%` | Feature Dropout | **Logistic Regression** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | +0.0476 | 0.0000 |
| `DROPOUT_20%` | Feature Dropout | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `DROPOUT_20%` | Feature Dropout | **Logistic Regression** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | +0.0476 | 0.0000 |
| `DROPOUT_20%` | Feature Dropout | **DistilBERT** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

---

## 3. Perturbation Stress Analysis

### A. Sensor Noise Stress (1%, 3%, 5%)
- Gaussian sensor noise adds perturbation to all 30 continuous sensor features.
- High robustness observed for tree ensembles; linear boundaries shift gracefully.

### B. Sensor Missingness & Block Failures (5%, 10%, Contiguous Block)
- Random missingness simulates temporary communication dropped packets.
- Contiguous block simulates sensor subsystem dropout over 10 consecutive windows.

### C. Progressive Temporal Drift (2%, 5%)
- Simulates progressive electrode wear, nozzle spatter buildup, and thermal accumulation across shifts.

### D. Reduced Fault Severity (75%, 50%, 25%)
- Evaluates detection threshold sensitivity when physical fault signals are faint.
- As fault magnitude drops to 25%, recall degrades, revealing true detection boundaries.

### E. Feature Channel Dropout / Corruption (10%, 20%)
- Simulates complete hardware loss of 3 to 6 sensor channels.

---

## 4. Methodological Distinction: Synthetic Robustness vs. External Validation

> [!IMPORTANT]
> **Scientific Transparency Note:**
> This robustness evaluation demonstrates algorithmic stability against mathematical perturbations of the synthetic generator's output distribution. **It does NOT constitute validation on real-world industrial welding data or public benchmarks.** Real industrial environments feature unmodeled physical dynamics (e.g., base metal surface oxides, fit-up gaps, ambient draft), which require future empirical validation on hardware testbeds.

---

## 5. Reproducibility Commands

```powershell
# Run full synthetic robustness and multi-seed suite
.\.venv\Scripts\python.exe evaluation/eval_synthetic_robustness.py

# Generate visualization plots
.\.venv\Scripts\python.exe evaluation/plot_robustness.py

# Run unit tests for robustness perturbations and multi-seed
.\.venv\Scripts\pytest tests/test_robustness.py -v
```
