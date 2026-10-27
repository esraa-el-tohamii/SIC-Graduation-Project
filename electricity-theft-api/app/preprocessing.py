"""
preprocessing.py — Exact reproduction of the notebook's preprocessing pipeline.

This module contains ONLY the transformations that are verified to have been
applied to the training data in sic-project.ipynb (Cells 62–64, 74, 86–87).

IMPORTANT NOTES
---------------
* All three cleaning functions (fill_missing, cap_outliers, minmax) operate
  PURELY on per-consumer statistics — no training-set-wide scaler is saved or
  needed.  This matches the notebook exactly.

* The raw (uncleaned) readings go into the XGBoost stacking feature vector.
  The cleaned readings are ONLY used to generate the CNN input.

* Missing readings are represented as NaN (float) in the raw vector.
  They remain NaN in the XGBoost input; XGBoost handles them natively.

* The Missing_Reading_Percentage is computed from the raw readings BEFORE any
  cleaning, as a fraction ×100.

* The CNN input has exactly two components:
    X_1d  shape (1035,)     — [cleaned_readings | missing_pct_fraction]
    X_2d  shape (2, 148, 7) — [cleaned_weekly_grid | binary_missing_mask]
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Optional, Tuple

from app.config import (
    EXPECTED_READINGS,
    NUM_WEEKS,
    PAD_DAYS,
    CNN_WIDE_IN,
    CNN_IN_CH,
)


# ── Per-consumer cleaning (verbatim from Cell 62) ─────────────────────────

def fill_missing(X: np.ndarray) -> np.ndarray:
    """
    Fill NaN values with the average of the immediately preceding and
    following observed reading (neighbour average).  If both neighbours
    are also NaN the reading is set to 0.0.

    Parameters
    ----------
    X : ndarray, shape (n_consumers, n_days)  — may contain NaN

    Returns
    -------
    ndarray, shape (n_consumers, n_days)  — no NaN values
    """
    X = X.copy()
    nan_mask = np.isnan(X)
    prev = np.full_like(X, np.nan)
    nxt  = np.full_like(X, np.nan)
    prev[:, 1:]  = X[:, :-1]
    nxt[:, :-1]  = X[:, 1:]
    avg = (prev + nxt) / 2.0
    X[nan_mask] = np.where(np.isnan(avg[nan_mask]), 0.0, avg[nan_mask])
    return X


def cap_outliers(X: np.ndarray) -> np.ndarray:
    """
    Clip each consumer's readings at mean + 2 × std (per-row).

    Parameters
    ----------
    X : ndarray, shape (n_consumers, n_days)  — no NaN expected

    Returns
    -------
    ndarray, same shape, clipped
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return np.minimum(X, mu + 2.0 * sd)


def minmax(X: np.ndarray) -> np.ndarray:
    """
    Scale each consumer's readings to [0, 1] using per-row min-max.
    Flat consumers (range == 0) are left at 0.0.

    Parameters
    ----------
    X : ndarray, shape (n_consumers, n_days)  — no NaN expected

    Returns
    -------
    ndarray, same shape, in [0, 1]
    """
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    return (X - mn) / rng


# ── High-level preprocessing entry point ──────────────────────────────────

def preprocess_single(
    readings: List[Optional[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Transform one consumer's raw readings into all inputs needed for inference.

    Parameters
    ----------
    readings : list of length EXPECTED_READINGS (1034)
        Each element is a daily electricity consumption value (float) or
        None / float('nan') for a missing day.

    Returns
    -------
    stack_row  : ndarray, shape (1036,)
        [raw_readings | Missing_Reading_Percentage | CNN_Prediction_placeholder]
        Note: CNN_Prediction is 0.0 here; it is filled in by inference.py
        after running the CNN ensemble.

    x_1d       : ndarray, shape (1, 1035)  — CNN wide input (batch dim added)
    x_2d       : ndarray, shape (1, 2, 148, 7) — CNN deep input (batch dim added)
    miss_pct   : float — missing reading percentage (0–100)
    miss_pct_fraction : float — same value as a fraction [0, 1]

    Raises
    ------
    ValueError
        If the readings list has the wrong length, contains inf, or is not
        one-dimensional.
    """
    # ── 1. Convert and validate ───────────────────────────────────────────
    raw = np.asarray(
        [np.nan if v is None else float(v) for v in readings],
        dtype=np.float32,
    )

    if raw.ndim != 1:
        raise ValueError("Readings must be a one-dimensional sequence.")
    if len(raw) != EXPECTED_READINGS:
        raise ValueError(
            f"Expected {EXPECTED_READINGS} readings, received {len(raw)}."
        )

    # Reject infinite values — NaN is allowed (missing reading)
    if np.any(np.isinf(raw)):
        raise ValueError(
            "Readings must not contain infinite values.  "
            "Use null/None for missing days."
        )

    # ── 2. Missing-reading statistics (from raw, BEFORE cleaning) ─────────
    nan_mask = np.isnan(raw)                         # shape (1034,)
    miss_pct_fraction = float(nan_mask.mean())       # [0, 1]
    miss_pct = miss_pct_fraction * 100.0             # [0, 100]

    # ── 3. Build the XGBoost stacking row (raw readings, NaN preserved) ───
    # The notebook's Xd_base/Xt_base uses the missing fraction, not a
    # percentage. The public response converts it to a percentage below.
    # Placeholder for CNN_Prediction; filled by inference.py
    stack_row = np.concatenate(
        [raw, [miss_pct_fraction, 0.0]], dtype=np.float32
    )  # shape (1036,)

    # ── 4. Clean the readings for the CNN (per-consumer, stateless) ───────
    raw_2d = raw.reshape(1, -1)                      # shape (1, 1034)
    clean_2d = minmax(cap_outliers(fill_missing(raw_2d))).astype(np.float32)
    # clean_2d has shape (1, 1034), no NaN

    # ── 5. Build CNN wide input X_1d ──────────────────────────────────────
    # shape (1, 1035) — [clean readings | missing_pct_fraction]
    x_1d = np.hstack(
        [clean_2d, [[miss_pct_fraction]]]
    ).astype(np.float32)  # shape (1, 1035)

    # ── 6. Build CNN deep input X_2d ──────────────────────────────────────
    # Channel 0: cleaned weekly grid (padded to 148 × 7)
    # Channel 1: binary missing mask (1 = was missing, 0 = observed)
    clean_padded = np.pad(
        clean_2d, ((0, 0), (0, PAD_DAYS))
    ).reshape(1, 1, NUM_WEEKS, 7).astype(np.float32)  # (1, 1, 148, 7)

    mask = nan_mask.astype(np.float32)               # shape (1034,)
    mask_padded = np.pad(
        mask.reshape(1, -1), ((0, 0), (0, PAD_DAYS))
    ).reshape(1, 1, NUM_WEEKS, 7).astype(np.float32)  # (1, 1, 148, 7)

    x_2d = np.concatenate(
        [clean_padded, mask_padded], axis=1
    )  # shape (1, 2, 148, 7)

    return stack_row, x_1d, x_2d, miss_pct, miss_pct_fraction


def preprocess_batch(
    readings_list: List[List[Optional[float]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch version of preprocess_single.

    Parameters
    ----------
    readings_list : list of N lists, each of length EXPECTED_READINGS

    Returns
    -------
    stack_rows : ndarray, shape (N, 1036)  — XGBoost input (CNN col = 0)
    x_1d_batch : ndarray, shape (N, 1035)
    x_2d_batch : ndarray, shape (N, 2, 148, 7)
    miss_pcts  : ndarray, shape (N,)       — missing pct (0–100) per consumer
    """
    results = [preprocess_single(r) for r in readings_list]
    stack_rows = np.stack([r[0] for r in results], axis=0)
    x_1d_batch = np.concatenate([r[1] for r in results], axis=0)
    x_2d_batch = np.concatenate([r[2] for r in results], axis=0)
    miss_pcts  = np.array([r[3] for r in results], dtype=np.float32)
    return stack_rows, x_1d_batch, x_2d_batch, miss_pcts
