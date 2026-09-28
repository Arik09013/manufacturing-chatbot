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
    ap.add_argument("--split-mode", type=str, default="chronological", choices=["chronological", "lomo"],
                    help="Splitting strategy: 'chronological' (default) or 'lomo' (Leave-One-Machine-Out)")
    ap.add_argument("--embargo-minutes", type=int, default=30,
                    help="Embargo gap in minutes between train and test windows (for chronological split)")
    ap.add_argument("--holdout-machine", type=str, default="station_1",
                    choices=["station_1", "station_2", "station_3"],
                    help="Machine to hold out when split-mode is 'lomo'")
    args = ap.parse_args()

    import json
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments,
    )
    from src.fusion.fuse import load_fused
    from src.model.anomaly import AnomalyDetector
    from src.data.splits import (
        split_chronological,
        verify_leakage_free,
        prepare_tabular_data,
        compute_classification_metrics,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"[finetune] device: {device}")

    # ── Leakage-Safe Data Split ──
    df = load_fused().reset_index(drop=True)

    if args.split_mode == "chronological":
        print(f"[finetune] Performing chronological split (embargo={args.embargo_minutes}m)...")
        train_df, test_df, purge_df = split_chronological(
            df, train_ratio=0.8, embargo_minutes=args.embargo_minutes
        )
        split_desc = f"Chronological 80/20 Holdout (embargo={args.embargo_minutes} min)"
    else:
        print(f"[finetune] Performing Leave-One-Machine-Out split (held-out: {args.holdout_machine})...")
        train_df = df[df["machine_id"] != args.holdout_machine].copy().reset_index(drop=True)
        test_df = df[df["machine_id"] == args.holdout_machine].copy().reset_index(drop=True)
        verify_leakage_free(train_df, test_df, mode="lomo", held_out_machine=args.holdout_machine)
        split_desc = f"Leave-One-Machine-Out (held-out: {args.holdout_machine})"

    train_texts = build_texts(train_df)
    y_train = train_df["is_anomaly"].astype(int).values

    test_texts = build_texts(test_df)
    y_test = test_df["is_anomaly"].astype(int).values

    print(f"[finetune] Train set: {len(train_df)} windows ({int(y_train.sum())} anomalies)")
    print(f"[finetune] Test set:  {len(test_df)} windows ({int(y_test.sum())} anomalies)")

    # ── DistilBERT fine-tuning ──
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_ds = _TextDataset(
        tokenizer(train_texts, truncation=True, padding="max_length", max_length=args.max_len),
        y_train,
    )
    test_ds = _TextDataset(
        tokenizer(test_texts, truncation=True, padding="max_length", max_length=args.max_len),
        y_test,
    )

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    # class weights for the heavy imbalance computed strictly on y_train
    counts = np.bincount(y_train, minlength=2).astype(float)
    weights = torch.tensor((counts.sum() / (2.0 * np.maximum(counts, 1))), dtype=torch.float)
    print(f"[finetune] class weights (from train set): {weights.tolist()}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights.to(outputs.logits.device))
            loss = loss_fct(outputs.logits.view(-1, 2), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

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
    )

    print("[finetune] training DistilBERT…")
    trainer.train()

    # Predict on test set
    preds_out = trainer.predict(test_ds)
    bert_logits = preds_out.predictions
    bert_probs = torch.softmax(torch.tensor(bert_logits), dim=-1)[:, 1].numpy()
    bert_pred = (bert_probs >= 0.5).astype(int)

    bert_metrics = compute_classification_metrics(
        y_test, bert_pred, bert_probs, n_train=len(train_df), n_anom_train=int(y_train.sum())
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"[finetune] saved -> {MODEL_DIR}")

    # ── RandomForest on the EXACT SAME leak-free split (apples-to-apples) ──
    X_tr_scaled, y_tr, X_te_scaled, y_te, _, _ = prepare_tabular_data(train_df, test_df, scale=True)
    rf = AnomalyDetector(mode="supervised")
    rf.model.fit(X_tr_scaled, y_tr)
    rf_pred = rf.model.predict(X_te_scaled)
    rf_prob = rf.model.predict_proba(X_te_scaled)[:, 1]
    rf_metrics = compute_classification_metrics(
        y_te, rf_pred, rf_prob, n_train=len(train_df), n_anom_train=int(y_train.sum())
    )

    # ── Save JSON Results ──
    results_dir = Path(__file__).parent.parent.parent / "evaluation" / "results" / "time_aware"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_filename = (
        "distilbert_chronological_results.json"
        if args.split_mode == "chronological"
        else f"distilbert_lomo_{args.holdout_machine}_results.json"
    )
    json_path = results_dir / json_filename

    results_payload = {
        "split_mode": args.split_mode,
        "split_description": split_desc,
        "embargo_minutes": args.embargo_minutes if args.split_mode == "chronological" else None,
        "holdout_machine": args.holdout_machine if args.split_mode == "lomo" else None,
        "distilbert": bert_metrics,
        "random_forest": rf_metrics,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[finetune] JSON results -> {json_path}")

    # ── Save Markdown Report ──
    report_time_aware_path = Path(__file__).parent.parent.parent / "outputs" / "finetune_report_time_aware.md"
    rows = [
        "# DistilBERT Fine-Tuning vs RandomForest (Leakage-Safe Evaluation)",
        "",
        f"**Device:** {device}  ·  **Epochs:** {args.epochs}  ·  **Split:** {split_desc}",
        f"**Train Windows:** {len(train_df)} (Anomalies: {int(y_train.sum())})  ·  **Test Windows:** {len(test_df)} (Anomalies: {int(y_test.sum())})",
        "",
        "| Metric | RandomForest (Tabular) | DistilBERT (Fine-Tuned Text) |",
        "|---|---|---|",
        f"| Accuracy | {rf_metrics['accuracy']} | {bert_metrics['accuracy']} |",
        f"| Precision | {rf_metrics['precision']} | {bert_metrics['precision']} |",
        f"| Recall | {rf_metrics['recall']} | {bert_metrics['recall']} |",
        f"| F1 Score | {rf_metrics['f1']} | {bert_metrics['f1']} |",
        f"| ROC-AUC | {rf_metrics['roc_auc']} | {bert_metrics['roc_auc']} |",
        f"| PR-AUC | {rf_metrics['pr_auc']} | {bert_metrics['pr_auc']} |",
        f"| Confusion Matrix | `{rf_metrics['confusion_matrix']}` | `{bert_metrics['confusion_matrix']}` |",
        "",
        "### Preprocessing & Splitting Integrity Notes",
        "- Splitting was performed strictly without random shuffling of overlapping windows.",
        "- For chronological split, a >=30 minute temporal embargo was enforced between train and test windows.",
        "- Tabular features were scaled with `StandardScaler` fit ONLY on training data.",
        "- Previous report (`outputs/finetune_report.md`) used a random 80/20 split across overlapping windows (legacy leaky evaluation).",
    ]
    report_time_aware_path.write_text("\n".join(rows), encoding="utf-8")

    print("\n" + "=" * 60)
    print("\n".join(rows))
    print("=" * 60)
    print(f"[finetune] report -> {report_time_aware_path}")


if __name__ == "__main__":
    main()
