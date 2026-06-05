# MVP Evaluation Report

Generated: 2026-06-05

---

## 1. Anomaly Detection (5-fold Stratified CV)

| Metric | Score |
|---|---|
| Accuracy | 0.9935 |
| Precision | 0.8267 |
| Recall | 1.0 |
| F1 | 0.9051 |
| ROC-AUC | 0.9996 |
| Total windows | 2010 |
| Anomalous windows | 62 |

### Classification Report

```
              precision    recall  f1-score   support

      normal       1.00      0.99      1.00      1948
     anomaly       0.83      1.00      0.91        62

    accuracy                           0.99      2010
   macro avg       0.91      1.00      0.95      2010
weighted avg       0.99      0.99      0.99      2010

```

---

## 2. SHAP Driver Alignment

Fraction of anomalous windows where at least one top-5 SHAP driver
matches the injected anomaly channel for that anomaly type.

| Anomaly type | Alignment |
|---|---|
| overheating | 100% |
| bearing_failure | 100% |
| coolant_failure | 100% |
| pressure_loss | 100% |
| motor_overload | 100% |
| overall | 100% |

---

## 3. Confidence Calibration

| Confidence band | N | Accuracy |
|---|---|---|
| high | 62 | 100% |
| medium | 9 | 33% |
| low | 1939 | 100% |

---

## Notes

- CV metrics are computed on the synthetic dataset (labels known exactly).
- SHAP alignment validates that the explainability layer surfaces the correct
  sensor channels for each injected anomaly type.
- Full 5-fold CV, KPI benchmarking deferred to Phase 6.