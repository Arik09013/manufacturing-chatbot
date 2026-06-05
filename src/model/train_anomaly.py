"""
Train the anomaly detector and persist it to models/anomaly.joblib.

Usage:
    python src/model/train_anomaly.py          # supervised RF
    python src/model/train_anomaly.py --mode unsupervised
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loaders import load_sensors, load_logs, load_ground_truth
from src.preprocess.sensor import preprocess_sensors
from src.preprocess.logs import preprocess_logs
from src.fusion.fuse import fuse, save_fused
from src.model.anomaly import AnomalyDetector, get_feature_matrix, get_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MACHINES = ["machine_1", "machine_2", "machine_3"]
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "anomaly.joblib"


def build_fused_dataset(include_embeddings: bool = False) -> "pd.DataFrame":
    """Load, preprocess, and fuse data for all machines."""
    import pandas as pd
    from src.data.loaders import load_notes
    from src.preprocess.notes import preprocess_notes

    sensor_pieces, log_pieces = [], []
    scaler = None

    for m in MACHINES:
        s = load_sensors(machine_id=m)
        windowed, scaler = preprocess_sensors(s, scaler=scaler, fit_scaler_on_data=(scaler is None))
        sensor_pieces.append(windowed)

        l = load_logs(machine_id=m)
        log_pieces.append(preprocess_logs(l))

    sensor_all = pd.concat(sensor_pieces, ignore_index=True)
    log_all    = pd.concat(log_pieces, ignore_index=True)

    notes_embedded = None
    if include_embeddings:
        try:
            notes_pieces = []
            for m in MACHINES:
                n = load_notes(machine_id=m)
                notes_pieces.append(preprocess_notes(n))
            notes_embedded = pd.concat(notes_pieces, ignore_index=True)
        except ImportError:
            logger.warning("sentence-transformers not installed; skipping note embeddings")

    gt = load_ground_truth()
    fused = fuse(sensor_all, log_all, notes_embedded, gt, include_embeddings=include_embeddings)
    return fused, scaler


def train(mode: str = "supervised", include_embeddings: bool = False) -> AnomalyDetector:
    logger.info("Building fused dataset…")
    fused, scaler = build_fused_dataset(include_embeddings)

    logger.info(
        "Dataset: %d windows, %d anomalous (%.1f%%)",
        len(fused),
        fused["is_anomaly"].sum(),
        100 * fused["is_anomaly"].mean(),
    )

    save_fused(fused)

    detector = AnomalyDetector(mode=mode)
    detector.fit(fused)

    # Persist model + scaler as a bundle
    bundle = {"detector": detector, "scaler": scaler}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    logger.info("Model saved -> %s", MODEL_PATH)

    # Quick sanity metrics on training data
    if mode == "supervised":
        from sklearn.metrics import classification_report
        X, feat_names = get_feature_matrix(fused)
        y_true = get_labels(fused)
        y_pred = detector.predict(fused)
        print("\nTraining-set report (not held-out — use eval_mvp.py for real metrics):")
        print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"]))

    return detector


def load_model() -> tuple[AnomalyDetector, object]:
    """Load persisted detector + scaler bundle."""
    bundle = joblib.load(MODEL_PATH)
    return bundle["detector"], bundle["scaler"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["supervised", "unsupervised"], default="supervised"
    )
    parser.add_argument("--embeddings", action="store_true", default=False)
    args = parser.parse_args()
    train(mode=args.mode, include_embeddings=args.embeddings)
