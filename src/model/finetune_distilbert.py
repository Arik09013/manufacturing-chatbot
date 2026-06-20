"""
Fine-tune DistilBERT for welding-fault (anomaly) detection.

This is the "LLM fine-tuning" path from the thesis. Each fused multimodal window
is serialised to text (see `bert_detector.window_to_text`) and a DistilBERT
sequence classifier is fine-tuned to predict is_anomaly. The script then trains a
RandomForest on the *same* train/test split and reports both, so the comparison
is apples-to-apples.

Runs on GPU if available, else CPU (the dataset is small — CPU takes a few
minutes). The fine-tuned model is saved to `models/distilbert_fault/` (E drive);
nothing new is written to C.

Usage:
    python src/model/finetune_distilbert.py [--epochs 3] [--max-len 128]

Honest note: with only ~60 anomalies the fine-tuned model typically *matches*
rather than *beats* the RandomForest. The point is to demonstrate the
fine-tuning capability and provide a like-for-like comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.model.bert_detector import BASE_MODEL, MODEL_DIR, build_texts

REPORT_PATH = Path(__file__).parent.parent.parent / "outputs" / "finetune_report.md"


class _TextDataset:
    """Minimal torch Dataset over tokenised texts + integer labels."""

    def __init__(self, encodings, labels):
        import torch
        self.torch = torch
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: self.torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = self.torch.tensor(int(self.labels[idx]))
        return item


def _metrics(y_true, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments,
    )
    from src.fusion.fuse import load_fused
    from src.model.anomaly import AnomalyDetector, get_feature_matrix, get_labels

    use_cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"[finetune] device: {device}")

    # ── Data ──
    df = load_fused().reset_index(drop=True)
    texts = build_texts(df)
    y = get_labels(df)
    print(f"[finetune] {len(df)} windows, {int(y.sum())} anomalies")

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

    # ── DistilBERT fine-tuning ──
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def _encode(indices):
        return tokenizer([texts[i] for i in indices], truncation=True,
                         padding="max_length", max_length=args.max_len)

    train_ds = _TextDataset(_encode(tr_idx), y[tr_idx])
    test_ds = _TextDataset(_encode(te_idx), y[te_idx])

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    # class weights for the heavy imbalance (~3% anomalies)
    counts = np.bincount(y[tr_idx], minlength=2).astype(float)
    weights = torch.tensor((counts.sum() / (2.0 * np.maximum(counts, 1))), dtype=torch.float)
    print(f"[finetune] class weights: {weights.tolist()}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights.to(outputs.logits.device))
            loss = loss_fct(outputs.logits.view(-1, 2), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        return _metrics(labels, probs)

    targs = TrainingArguments(
        output_dir=str(MODEL_DIR.parent / "_distilbert_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        use_cpu=not use_cuda,
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("[finetune] training DistilBERT…")
    trainer.train()
    bert_eval = trainer.evaluate()
    bert_metrics = {k.replace("eval_", ""): v for k, v in bert_eval.items()
                    if k.replace("eval_", "") in ("precision", "recall", "f1", "roc_auc")}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"[finetune] saved -> {MODEL_DIR}")

    # ── RandomForest on the SAME split (apples-to-apples) ──
    X, _ = get_feature_matrix(df)
    rf = AnomalyDetector(mode="supervised")
    rf.model.fit(X[tr_idx], y[tr_idx])
    rf_prob = rf.model.predict_proba(X[te_idx])[:, 1]
    rf_metrics = _metrics(y[te_idx], rf_prob)

    # ── Report ──
    rows = [
        "# DistilBERT Fine-Tuning vs RandomForest",
        "",
        f"Device: {device}  ·  Epochs: {args.epochs}  ·  Test split: 20% stratified",
        f"Windows: {len(df)}  ·  Anomalies: {int(y.sum())}",
        "",
        "| Metric | RandomForest | DistilBERT (fine-tuned) |",
        "|---|---|---|",
    ]
    for m in ("precision", "recall", "f1", "roc_auc"):
        rows.append(f"| {m} | {rf_metrics.get(m)} | {bert_metrics.get(m)} |")
    rows += [
        "",
        "_Same train/test split for both. With ~60 anomalies the fine-tuned "
        "DistilBERT typically matches the RandomForest; the value is demonstrating "
        "the fine-tuning path and a like-for-like comparison._",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(rows), encoding="utf-8")

    print("\n" + "=" * 56)
    print("\n".join(rows))
    print("=" * 56)
    print(f"[finetune] report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
