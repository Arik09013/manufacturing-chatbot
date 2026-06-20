# MVP Evaluation Report

Generated: 2026-06-05

---

## 1. Anomaly Detection (5-fold Stratified CV)

| Metric | Score |
|---|---|
| Accuracy | 0.9969 |
| Precision | 0.9219 |
| Recall | 0.9833 |
| F1 | 0.9516 |
| ROC-AUC | 0.9992 |
| Total windows | 1917 |
| Anomalous windows | 60 |

### Classification Report

```
              precision    recall  f1-score   support

      normal       1.00      1.00      1.00      1857
     anomaly       0.92      0.98      0.95        60

    accuracy                           1.00      1917
   macro avg       0.96      0.99      0.97      1917
weighted avg       1.00      1.00      1.00      1917

```

---

## 2. SHAP Driver Alignment

Fraction of anomalous windows where at least one top-5 SHAP driver
matches the injected anomaly channel for that anomaly type.

| Anomaly type | Alignment |
|---|---|
| arc_instability | 100% |
| wire_feed_fault | 39% |
| underheat | 100% |
| gas_flow_failure | 100% |
| overheating | 100% |
| overall | 82% |

### LIME Driver Alignment (independent XAI cross-check)

| Anomaly type | Alignment |
|---|---|
| arc_instability | 100% |
| wire_feed_fault | 0% |
| underheat | 100% |
| gas_flow_failure | 75% |
| overheating | 100% |
| overall | 67% |

---

## 3. Root-Cause Ranking

Rank quality of the heuristic root-cause mapper for the canonical cause
of each anomaly type (given SHAP drivers).

| Metric | Score |
|---|---|
| MRR | 0.878 |
| Top-1 accuracy | 82% |
| Top-2 accuracy | 82% |
| Top-3 accuracy | 100% |
| Anomalous windows scored | 60 |

---

## 4. Prescriptive Setpoint Quality

Parameter-advisor recommendations across every supported
material x process x thickness band.

| Metric | Score |
|---|---|
| In-window compliance | 100% |
| Mean normalised deviation (MAE / range) | 0.378 |
| Setpoints evaluated | 72 |

---

## 5. Confidence Calibration

| Confidence band | N | Accuracy |
|---|---|---|
| high | 61 | 98% |
| medium | 15 | 100% |
| low | 1841 | 100% |

---

## Notes

- CV metrics are computed on the synthetic dataset (labels known exactly).
- SHAP alignment validates that the explainability layer surfaces the correct
  sensor channels for each injected anomaly type.
- Full 5-fold CV, KPI benchmarking deferred to Phase 6.