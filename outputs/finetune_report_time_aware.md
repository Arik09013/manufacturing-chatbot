# DistilBERT Fine-Tuning vs RandomForest (Leakage-Safe Evaluation)

**Device:** CPU  ·  **Epochs:** 3  ·  **Split:** Chronological 80/20 Holdout (embargo=30 min)
**Train Windows:** 1533 (Anomalies: 50)  ·  **Test Windows:** 375 (Anomalies: 10)

| Metric | RandomForest (Tabular) | DistilBERT (Fine-Tuned Text) |
|---|---|---|
| Accuracy | 1.0 | 1.0 |
| Precision | 1.0 | 1.0 |
| Recall | 1.0 | 1.0 |
| F1 Score | 1.0 | 1.0 |
| ROC-AUC | 1.0 | 1.0 |
| PR-AUC | 1.0 | 1.0 |
| Confusion Matrix | `[[365, 0], [0, 10]]` | `[[365, 0], [0, 10]]` |

### Preprocessing & Splitting Integrity Notes
- Splitting was performed strictly without random shuffling of overlapping windows.
- For chronological split, a >=30 minute temporal embargo was enforced between train and test windows.
- Tabular features were scaled with `StandardScaler` fit ONLY on training data.
- Previous report (`outputs/finetune_report.md`) used a random 80/20 split across overlapping windows (legacy leaky evaluation).