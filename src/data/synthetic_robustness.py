"""
Synthetic Data Realism & Robustness Evaluation Utilities.

Provides:
  1. Multi-seed in-memory dataset generation and fusion (seeds: 42, 123, 456, 789, 2026).
  2. Controlled test-set perturbation transformations:
     - Sensor Noise (1%, 3%, 5%)
     - Missingness (5%, 10% random, and contiguous block)
     - Gradual Temporal Sensor Drift (2%, 5%)
     - Reduced Fault Severity (75%, 50%, 25% of anomalous magnitude)
     - Sensor Feature Corruption / Dropout (10%, 20%)

CRITICAL LEAKAGE & INTEGRITY GUARANTEES:
  - Perturbations are applied strictly to TEST data post-split.
  - Training data is never perturbed and never sees test noise.
  - StandardScaler is fitted strictly on CLEAN X_train.
  - Ground-truth anomaly labels (y_test) are NEVER modified.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.generate_synthetic import MACHINES, generate_synthetic_multimodal
from src.fusion.fuse import fuse
from src.model.anomaly import ALL_FEATURE_COLS, SENSOR_FEATURE_COLS
from src.preprocess.logs import preprocess_logs
from src.preprocess.sensor import preprocess_sensors

logger = logging.getLogger(__name__)

# Standard multi-seed evaluation list
ROBUSTNESS_SEEDS = [42, 123, 456, 789, 2026]


def build_fused_for_seed(seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic raw multimodal data for a specific seed and fuse it in-memory.
    Preserves existing raw disk files without overwriting.

    Returns
    -------
    pd.DataFrame of fused windows with ground-truth labels.
    """
    sensors, logs, notes, gt = generate_synthetic_multimodal(seed=seed)

    sensor_pieces, log_pieces = [], []
    scaler = None
    for m in MACHINES:
        s = sensors[sensors["machine_id"] == m].copy()
        w, scaler = preprocess_sensors(s, scaler=scaler, fit_scaler_on_data=(scaler is None))
        sensor_pieces.append(w)
        l = logs[logs["machine_id"] == m].copy()
        log_pieces.append(preprocess_logs(l))

    sensor_all = pd.concat(sensor_pieces, ignore_index=True)
    log_all = pd.concat(log_pieces, ignore_index=True)
    fused = fuse(sensor_all, log_all, None, gt, include_embeddings=False)
    return fused


# ── Perturbation Functions (Applied strictly to test data) ────────────────────

def get_sensor_column_indices(feature_names: List[str]) -> List[int]:
    """Return column indices of sensor features within the tabular feature vector."""
    return [i for i, name in enumerate(feature_names) if name in SENSOR_FEATURE_COLS]


def apply_sensor_noise(
    X_test: np.ndarray,
    noise_level: float = 0.01,
    seed: int = 42,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to sensor features proportional to their scale.
    Because X_test is already standardized, sigma=1.0, so noise std = noise_level.
    """
    X_pert = X_test.copy()
    rng = np.random.default_rng(seed)
    names = feature_names or ALL_FEATURE_COLS
    sensor_indices = get_sensor_column_indices(names)

    noise = rng.normal(loc=0.0, scale=noise_level, size=(X_pert.shape[0], len(sensor_indices)))
    X_pert[:, sensor_indices] += noise.astype(X_pert.dtype)
    return X_pert


def apply_missingness(
    X_test: np.ndarray,
    missing_ratio: float = 0.05,
    is_block: bool = False,
    block_len: int = 10,
    seed: int = 42,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Inject missing values into sensor channels (imputed with 0.0, the training mean in scaled space).
    """
    X_pert = X_test.copy()
    rng = np.random.default_rng(seed)
    names = feature_names or ALL_FEATURE_COLS
    sensor_indices = get_sensor_column_indices(names)

    if is_block:
        # Contiguous temporal block of sensor failure
        n_samples = X_pert.shape[0]
        max_start = max(0, n_samples - block_len)
        start_idx = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        end_idx = min(n_samples, start_idx + block_len)
        X_pert[start_idx:end_idx, sensor_indices] = 0.0
    else:
        # Random missing values
        mask = rng.random(size=(X_pert.shape[0], len(sensor_indices))) < missing_ratio
        for idx_col, col in enumerate(sensor_indices):
            X_pert[mask[:, idx_col], col] = 0.0

    return X_pert


def apply_sensor_drift(
    X_test: np.ndarray,
    drift_level: float = 0.02,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Introduce linear temporal drift evolving progressively across the chronological test horizon.
    """
    X_pert = X_test.copy()
    names = feature_names or ALL_FEATURE_COLS
    sensor_indices = get_sensor_column_indices(names)

    n_samples = X_pert.shape[0]
    if n_samples <= 1:
        return X_pert

    # Linear ramp from 0 to drift_level * sigma across time
    ramp = np.linspace(0.0, drift_level, n_samples, dtype=X_pert.dtype).reshape(-1, 1)
    X_pert[:, sensor_indices] += ramp
    return X_pert


def apply_reduced_severity(
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    severity_ratio: float = 0.75,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Attenuate anomaly magnitude towards normal baseline mean for anomalous test rows (y=1).
    Normal test rows (y=0) are untouched.
    """
    X_pert = X_test.copy()
    names = feature_names or ALL_FEATURE_COLS
    sensor_indices = get_sensor_column_indices(names)

    # Compute baseline normal mean from normal training windows (y_train == 0)
    normal_mask_tr = (y_train == 0)
    if np.any(normal_mask_tr):
        normal_mu = np.mean(X_train[normal_mask_tr], axis=0)
    else:
        normal_mu = np.zeros(X_train.shape[1], dtype=X_pert.dtype)

    # For anomalous test windows, scale deviation from normal mean by severity_ratio
    anom_mask_te = (y_test == 1)
    if np.any(anom_mask_te):
        for col in sensor_indices:
            diff = X_pert[anom_mask_te, col] - normal_mu[col]
            X_pert[anom_mask_te, col] = normal_mu[col] + (severity_ratio * diff)

    return X_pert


def apply_feature_dropout(
    X_test: np.ndarray,
    dropout_ratio: float = 0.10,
    seed: int = 42,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Mask a random subset of sensor features completely across all test windows.
    """
    X_pert = X_test.copy()
    rng = np.random.default_rng(seed)
    names = feature_names or ALL_FEATURE_COLS
    sensor_indices = get_sensor_column_indices(names)

    k = max(1, int(round(len(sensor_indices) * dropout_ratio)))
    chosen_indices = rng.choice(sensor_indices, size=k, replace=False)
    X_pert[:, chosen_indices] = 0.0
    return X_pert


def create_perturbed_test_dataframe(
    test_df: pd.DataFrame,
    X_test_perturbed: np.ndarray,
    scaler: StandardScaler,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Reconstruct a perturbed pandas DataFrame (in unscaled physical space)
    suitable for DistilBERT text serialization (window_to_text).
    """
    df_pert = test_df.copy()
    # Inverse transform perturbed features back to physical units
    X_raw_pert = scaler.inverse_transform(X_test_perturbed)
    for col_idx, col_name in enumerate(feature_names):
        if col_name in df_pert.columns:
            df_pert[col_name] = X_raw_pert[:, col_idx]
    return df_pert


def get_perturbation_scenarios(
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Tuple[np.ndarray, str, Dict[str, Any]]]:
    """
    Generate dictionary of all standardized perturbation test matrices.

    Returns
    -------
    dict: {scenario_name: (X_pert, category, metadata_dict)}
    """
    names = feature_names or ALL_FEATURE_COLS
    scenarios: Dict[str, Tuple[np.ndarray, str, Dict[str, Any]]] = {
        "CLEAN": (
            X_test.copy(),
            "Reference",
            {"description": "Clean unperturbed test set (reference baseline)", "level": 0.0},
        ),
        # Sensor Noise
        "NOISE_1%": (
            apply_sensor_noise(X_test, noise_level=0.01, seed=seed, feature_names=names),
            "Sensor Noise",
            {"description": "1% Gaussian sensor noise", "level": 0.01},
        ),
        "NOISE_3%": (
            apply_sensor_noise(X_test, noise_level=0.03, seed=seed, feature_names=names),
            "Sensor Noise",
            {"description": "3% Gaussian sensor noise", "level": 0.03},
        ),
        "NOISE_5%": (
            apply_sensor_noise(X_test, noise_level=0.05, seed=seed, feature_names=names),
            "Sensor Noise",
            {"description": "5% Gaussian sensor noise", "level": 0.05},
        ),
        # Missingness
        "MISSING_5%": (
            apply_missingness(X_test, missing_ratio=0.05, is_block=False, seed=seed, feature_names=names),
            "Missingness",
            {"description": "5% random sensor missingness (imputed with normal mean)", "level": 0.05},
        ),
        "MISSING_10%": (
            apply_missingness(X_test, missing_ratio=0.10, is_block=False, seed=seed, feature_names=names),
            "Missingness",
            {"description": "10% random sensor missingness (imputed with normal mean)", "level": 0.10},
        ),
        "MISSING_BLOCK": (
            apply_missingness(X_test, is_block=True, block_len=10, seed=seed, feature_names=names),
            "Missingness",
            {"description": "Contiguous temporal block missingness (10 consecutive windows)", "level": "10_windows"},
        ),
        # Drift
        "DRIFT_2%": (
            apply_sensor_drift(X_test, drift_level=0.02, feature_names=names),
            "Temporal Drift",
            {"description": "2% progressive linear sensor drift across test horizon", "level": 0.02},
        ),
        "DRIFT_5%": (
            apply_sensor_drift(X_test, drift_level=0.05, feature_names=names),
            "Temporal Drift",
            {"description": "5% progressive linear sensor drift across test horizon", "level": 0.05},
        ),
        # Severity
        "SEVERITY_75%": (
            apply_reduced_severity(X_test, y_test, X_train, y_train, severity_ratio=0.75, feature_names=names),
            "Fault Severity",
            {"description": "Anomalies reduced to 75% of original severity", "level": 0.75},
        ),
        "SEVERITY_50%": (
            apply_reduced_severity(X_test, y_test, X_train, y_train, severity_ratio=0.50, feature_names=names),
            "Fault Severity",
            {"description": "Anomalies reduced to 50% of original severity", "level": 0.50},
        ),
        "SEVERITY_25%": (
            apply_reduced_severity(X_test, y_test, X_train, y_train, severity_ratio=0.25, feature_names=names),
            "Fault Severity",
            {"description": "Anomalies reduced to 25% of original severity (subtle fault)", "level": 0.25},
        ),
        # Dropout / Corruption
        "DROPOUT_10%": (
            apply_feature_dropout(X_test, dropout_ratio=0.10, seed=seed, feature_names=names),
            "Feature Dropout",
            {"description": "10% sensor feature channels masked (3 channels)", "level": 0.10},
        ),
        "DROPOUT_20%": (
            apply_feature_dropout(X_test, dropout_ratio=0.20, seed=seed, feature_names=names),
            "Feature Dropout",
            {"description": "20% sensor feature channels masked (6 channels)", "level": 0.20},
        ),
    }
    return scenarios
