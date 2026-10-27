"""
model_loader.py — Load all inference artifacts exactly once at startup.

Responsibilities
----------------
* Load the five CNN WideDeepV2 models from the combined checkpoint.
* Load the XGBoost stacking model from its .joblib file.
* Load the ordered stack_feature_names list.
* Expose a single `ModelStore` dataclass that the rest of the application
  imports for prediction.
* Track and report which artifacts failed to load (for /health endpoint).

Design decisions
----------------
* Models are loaded at import time (module-level singleton).  FastAPI's
  startup event calls `load_all_artifacts()` which sets the global store.
* The CNN architecture (WideDeepV2) is defined here verbatim from the
  notebook (Cell 74) so that the .pth weight files can be loaded without
  the notebook environment.
* All CNN models are loaded to CPU by default (no CUDA assumed).
* The XGBoost model is loaded from joblib, which handles device-agnostic
  inference automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn

from app.config import (
    STACK_MODEL_FILE,
    FEATURE_NAMES_FILE,
    CNN_ENSEMBLE_FILE,
    EXPLAINER_FILE,
    NUM_CNN_MODELS,
    CNN_WIDE_IN,
    CNN_IN_CH,
)

logger = logging.getLogger(__name__)


# ── WideDeepV2 architecture (verbatim from notebook Cell 74) ──────────────

def _conv_block(in_ch: int, out_ch: int, pool: bool) -> List[nn.Module]:
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    ]
    if pool:
        layers.append(nn.MaxPool2d((2, 1)))
    return layers


class WideDeepV2(nn.Module):
    """Wide & Deep CNN model (version 2 with BatchNorm and missing-mask channel)."""

    def __init__(self, wide_in: int = CNN_WIDE_IN, in_ch: int = CNN_IN_CH):
        super().__init__()
        self.wide = nn.Sequential(
            nn.Linear(wide_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.deep = nn.Sequential(
            *_conv_block(in_ch, 16, pool=True),
            *_conv_block(16, 32, pool=True),
            *_conv_block(32, 64, pool=False),
            nn.AdaptiveAvgPool2d((4, 7)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 4 * 7, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        x1d: torch.Tensor,  # shape (B, 1035)
        x2d: torch.Tensor,  # shape (B, 2, 148, 7)
    ) -> torch.Tensor:
        return self.head(
            torch.cat([self.wide(x1d), self.deep(x2d)], dim=1)
        ).squeeze(1)


# ── Model store ────────────────────────────────────────────────────────────

@dataclass
class ModelStore:
    """Container for all loaded inference artifacts."""
    cnn_models: List[WideDeepV2] = field(default_factory=list)
    stack_model: Optional[object] = None             # XGBClassifier
    feature_names: Optional[List[str]] = None
    explainer: Optional[object] = None
    prediction_capabilities: List[str] = field(default_factory=list)
    load_errors: List[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        """True only when every required artifact loaded successfully."""
        return (
            len(self.cnn_models) == NUM_CNN_MODELS
            and self.stack_model is not None
            and self.feature_names is not None
            and len(self.feature_names) == 1036
            and len(self.load_errors) == 0
        )


# Global singleton — populated by load_all_artifacts()
_store: ModelStore = ModelStore()


def get_store() -> ModelStore:
    """Return the global model store (populated at startup)."""
    return _store


def load_all_artifacts() -> ModelStore:
    """
    Load all model artifacts and populate the global ModelStore.

    Called once at application startup (FastAPI lifespan).
    Safe to call multiple times — will reload each time.

    Returns
    -------
    The populated ModelStore instance.
    """
    global _store
    _store = ModelStore()

    # ── Load XGBoost stacking model ───────────────────────────────────────
    try:
        _store.stack_model = joblib.load(STACK_MODEL_FILE)
        logger.info("Loaded stacking model from %s", STACK_MODEL_FILE)
    except Exception as exc:
        msg = f"Failed to load stacking model ({STACK_MODEL_FILE}): {exc}"
        logger.error(msg)
        _store.load_errors.append(msg)

    # ── Load feature names ────────────────────────────────────────────────
    try:
        _store.feature_names = list(joblib.load(FEATURE_NAMES_FILE))
        if len(_store.feature_names) != 1036:
            raise ValueError(
                f"Expected 1036 feature names, got {len(_store.feature_names)}"
            )
        logger.info(
            "Loaded %d feature names from %s",
            len(_store.feature_names),
            FEATURE_NAMES_FILE,
        )
    except Exception as exc:
        msg = f"Failed to load feature names ({FEATURE_NAMES_FILE}): {exc}"
        logger.error(msg)
        _store.load_errors.append(msg)

    # ── Load the exported CNN ensemble ────────────────────────────────────
    try:
        states = torch.load(
            CNN_ENSEMBLE_FILE,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(states, (list, tuple)) or len(states) != NUM_CNN_MODELS:
            raise ValueError(
                f"Expected {NUM_CNN_MODELS} CNN state dicts, got "
                f"{len(states) if hasattr(states, '__len__') else 'invalid data'}"
            )
        for i, state in enumerate(states, start=1):
            net = WideDeepV2(wide_in=CNN_WIDE_IN, in_ch=CNN_IN_CH)
            net.load_state_dict(state)
            net.eval()
            _store.cnn_models.append(net)
            logger.info("Loaded CNN model %d from %s", i, CNN_ENSEMBLE_FILE)
    except Exception as exc:
        msg = f"Failed to load CNN ensemble ({CNN_ENSEMBLE_FILE}): {exc}"
        logger.error(msg)
        _store.load_errors.append(msg)

    # Explanations are optional and do not make prediction readiness fail.
    try:
        _store.explainer = joblib.load(EXPLAINER_FILE)
        logger.info("Loaded SHAP explainer from %s", EXPLAINER_FILE)
    except Exception as exc:
        logger.warning("SHAP explainer unavailable (%s): %s", EXPLAINER_FILE, exc)

    if _store.stack_model is not None and hasattr(_store.stack_model, "predict"):
        _store.prediction_capabilities.append("predict")
    if _store.stack_model is not None and hasattr(_store.stack_model, "predict_proba"):
        _store.prediction_capabilities.append("predict_proba")

    if _store.is_ready:
        logger.info("All inference artifacts loaded successfully.")
    else:
        logger.warning(
            "Model store is NOT fully ready. Errors: %s",
            _store.load_errors,
        )

    return _store
