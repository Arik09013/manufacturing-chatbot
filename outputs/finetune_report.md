# DistilBERT Fine-Tuning vs RandomForest

Device: CPU  ·  Epochs: 3  ·  Test split: 20% stratified
Windows: 1917  ·  Anomalies: 60

| Metric | RandomForest | DistilBERT (fine-tuned) |
|---|---|---|
| precision | 1.0 | 1.0 |
| recall | 1.0 | 1.0 |
| f1 | 1.0 | 1.0 |
| roc_auc | 1.0 | 1.0 |

_Same train/test split for both. With ~60 anomalies the fine-tuned DistilBERT typically matches the RandomForest; the value is demonstrating the fine-tuning path and a like-for-like comparison._