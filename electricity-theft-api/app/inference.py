"""
inference.py — Complete inference chain for one or more consumers.

Pipeline (matches notebook Cells 62–64, 74, 86–87, 89):
  raw_readings
    → preprocess_single()        [preprocessing.py]
        → fill_missing / cap_outliers / minmax
        → build X_1d (1035,) and X_2d (2, 148, 7)
    → CNN ensemble (5 WideDeepV2) → average sigmoid → CNN_Prediction
    → patch CNN_Prediction into stack_row[:, -1]
    → XGBoost stacking model     → theft_probability
    → threshold at DEFAULT_THRESHOLD → predicted_class
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

from app.config import (
    DEFAULT_THRESHOLD,
    LABEL_NAMES,
    MODEL_VERSION,
)
from app.model_loader import ModelStore
from app.preprocessing import preprocess_single, preprocess_batch

logger = logging.getLogger(__name__)


@torch.no_grad()
def _cnn_predict_batch(
    cnn_models: list,
    x_1d_batch: np.ndarray,   # (N, 1035)
    x_2d_batch: np.ndarray,   # (N, 2, 148, 7)
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run all CNN models on a batch, average their sigmoid outputs.

    Returns
    -------
    ndarray, shape (N,), values in [0, 1]
    """
    n = x_1d_batch.shape[0]
    t_1d = torch.tensor(x_1d_batch, dtype=torch.float32)
    t_2d = torch.tensor(x_2d_batch, dtype=torch.float32)

    all_probs = []
    for net in cnn_models:
        net.eval()
        model_probs = []
        for start in range(0, n, batch_size):
            end = start + batch_size
            logits = net(t_1d[start:end], t_2d[start:end])
            probs = torch.sigmoid(logits).cpu().numpy()
            model_probs.append(probs)
        all_probs.append(np.concatenate(model_probs))

    # Average over ensemble members
    return np.mean(all_probs, axis=0)


def run_inference(
    readings: List[Optional[float]],
    store: ModelStore,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Full inference pipeline for a single consumer.

    Parameters
    ----------
    readings  : list of 1034 float-or-None values
    store     : populated ModelStore
    threshold : decision threshold (default 0.5)

    Returns
    -------
    dict with keys:
        theft_probability  float [0, 1]
        predicted_class    int   {0, 1}
        label              str   "Normal" | "Theft flag"
        cnn_prediction     float [0, 1]  — intermediate CNN score
        missing_pct        float [0, 100]
        threshold          float
        model_version      str
    """
    if not store.is_ready:
        raise RuntimeError(
            "Model store is not ready.  "
            "Check /health for missing artifacts."
        )

    # ── Preprocessing ─────────────────────────────────────────────────────
    stack_row, x_1d, x_2d, miss_pct, _ = preprocess_single(readings)

    # ── CNN ensemble ──────────────────────────────────────────────────────
    cnn_pred = float(
        _cnn_predict_batch(store.cnn_models, x_1d, x_2d)[0]
    )

    # ── Patch CNN_Prediction into the stacking row ────────────────────────
    # stack_row[-1] is the CNN_Prediction placeholder (currently 0.0)
    stack_row[-1] = np.float32(cnn_pred)

    # ── XGBoost stacking model ────────────────────────────────────────────
    prob = float(
        store.stack_model.predict_proba(stack_row.reshape(1, -1))[0, 1]
    )

    predicted_class = int(prob >= threshold)

    return {
        "theft_probability": round(prob, 6),
        "predicted_class":   predicted_class,
        "label":             LABEL_NAMES[predicted_class],
        "cnn_prediction":    round(cnn_pred, 6),
        "missing_pct":       round(miss_pct, 4),
        "threshold":         threshold,
        "model_version":     MODEL_VERSION,
    }


def run_inference_batch(
    readings_list: List[List[Optional[float]]],
    store: ModelStore,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[dict]:
    """
    Full inference pipeline for multiple consumers.

    Parameters
    ----------
    readings_list : list of N lists, each of length EXPECTED_READINGS
    store         : populated ModelStore
    threshold     : decision threshold

    Returns
    -------
    list of N result dicts (same schema as run_inference)
    """
    if not store.is_ready:
        raise RuntimeError(
            "Model store is not ready.  "
            "Check /health for missing artifacts."
        )

    # ── Batch preprocessing ───────────────────────────────────────────────
    stack_rows, x_1d_batch, x_2d_batch, miss_pcts = preprocess_batch(
        readings_list
    )

    # ── CNN ensemble (batched) ────────────────────────────────────────────
    cnn_preds = _cnn_predict_batch(store.cnn_models, x_1d_batch, x_2d_batch)

    # ── Patch CNN predictions ─────────────────────────────────────────────
    stack_rows[:, -1] = cnn_preds.astype(np.float32)

    # ── XGBoost batch prediction ──────────────────────────────────────────
    probs = store.stack_model.predict_proba(stack_rows)[:, 1]

    results = []
    for i, (prob, cnn_pred, miss_pct) in enumerate(
        zip(probs, cnn_preds, miss_pcts)
    ):
        prob = float(prob)
        predicted_class = int(prob >= threshold)
        results.append(
            {
                "theft_probability": round(prob, 6),
                "predicted_class":   predicted_class,
                "label":             LABEL_NAMES[predicted_class],
                "cnn_prediction":    round(float(cnn_pred), 6),
                "missing_pct":       round(float(miss_pct), 4),
                "threshold":         threshold,
                "model_version":     MODEL_VERSION,
            }
        )
    return results


def explain_inference(
    readings: List[Optional[float]],
    store: ModelStore,
    top_n: int = 10,
) -> dict:
    """Return a local SHAP explanation for the final stack model."""
    if not store.is_ready:
        raise RuntimeError("Model store is not ready.")
    if store.explainer is None:
        raise RuntimeError("SHAP explanations are not available.")

    stack_row, x_1d, x_2d, _, _ = preprocess_single(readings)
    cnn_pred = float(_cnn_predict_batch(store.cnn_models, x_1d, x_2d)[0])
    stack_row[-1] = np.float32(cnn_pred)
    probability = float(store.stack_model.predict_proba(stack_row.reshape(1, -1))[0, 1])

    values = np.asarray(store.explainer.shap_values(stack_row.reshape(1, -1)))
    if values.ndim == 3:
        values = values[:, :, 1]
    values = values.reshape(-1)
    base = np.asarray(store.explainer.expected_value).reshape(-1)
    base_value = float(base[-1])
    raw_prediction = float(base_value + values.sum())
    indices = np.argsort(np.abs(values))[::-1][:top_n]
    return {
        "base_value": base_value,
        "prediction": probability,
        "raw_prediction": raw_prediction,
        "output_space": "raw",
        "features": [
            {
                "feature": store.feature_names[int(i)],
                "value": float(stack_row[int(i)]),
                "shap_value": float(values[int(i)]),
            }
            for i in indices
        ],
        "model_version": MODEL_VERSION,
    }
