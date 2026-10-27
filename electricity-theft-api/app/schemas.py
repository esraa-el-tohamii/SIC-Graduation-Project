"""
schemas.py — Pydantic request and response models.

Design principles
-----------------
* The request accepts readings as a list of Optional[float] so that callers
  can pass JSON null for missing days — matching the model's NaN handling.
* Infinity and NaN are explicitly rejected (a missing day is null, not NaN).
* The response is a typed dataclass — no arbitrary dict pass-through.
* Batch requests are capped at 500 consumers to prevent abuse.
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import EXPECTED_READINGS

MAX_BATCH_SIZE = 500


# ── Helpers ────────────────────────────────────────────────────────────────

# ── Request models ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Request body for POST /predict.

    Fields
    ------
    consumer_id : optional string — passed back in the response unchanged.
    readings    : exactly 1034 daily electricity consumption values.
                  Use JSON null for days with missing meter readings.
                  Infinite values and JSON NaN are rejected.

    Example
    -------
    {
      "consumer_id": "CONS_12345",
      "readings": [12.5, null, 9.8, ...]  ← list of 1034 values
    }
    """

    consumer_id: Optional[str] = Field(
        default=None,
        description="Consumer identifier (returned unchanged in the response).",
        examples=["CONS_12345"],
    )
    readings: List[Optional[float]] = Field(
        description=(
            f"Daily electricity consumption readings "
            f"(exactly {EXPECTED_READINGS} values, chronological order "
            "2014-01-01 to 2016-10-31).  "
            "Use null for missing days."
        ),
        min_length=EXPECTED_READINGS,
        max_length=EXPECTED_READINGS,
        examples=[[0.0] * EXPECTED_READINGS],
    )

    @field_validator("readings", mode="before")
    @classmethod
    def validate_readings(cls, v: list) -> list:
        if not isinstance(v, list):
            raise ValueError("readings must be a JSON array.")
        if len(v) != EXPECTED_READINGS:
            raise ValueError(
                f"readings must contain exactly {EXPECTED_READINGS} values; "
                f"received {len(v)}."
            )
        validated = []
        for idx, item in enumerate(v):
            if item is None:
                validated.append(None)
                continue
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(
                    f"readings[{idx}] must be a number or null."
                )
            try:
                fval = float(item)
            except (TypeError, ValueError):
                raise ValueError(
                    f"readings[{idx}] = {item!r} is not a valid number."
                )
            if math.isnan(fval):
                raise ValueError(
                    f"readings[{idx}]: use null instead of NaN for missing days."
                )
            if math.isinf(fval):
                raise ValueError(
                    f"readings[{idx}] is infinite; only finite values or null are accepted."
                )
            if fval < 0:
                raise ValueError(
                    f"readings[{idx}] = {fval} is negative; "
                    "electricity consumption cannot be negative."
                )
            validated.append(fval)
        return validated

    model_config = {
        "json_schema_extra": {
            "example": {
                "consumer_id": "CONS_99001",
                "readings": [12.5, None, 9.8, 11.2] + [10.0] * (EXPECTED_READINGS - 4),
            }
        }
    }


class BatchPredictRequest(BaseModel):
    """
    Request body for POST /predict/batch.

    Fields
    ------
    consumers : list of PredictRequest objects (1 to 500 consumers).
    """

    consumers: List[PredictRequest] = Field(
        description=f"List of consumer requests (1–{MAX_BATCH_SIZE}).",
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )

    @field_validator("consumers")
    @classmethod
    def check_batch_size(cls, v: list) -> list:
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum of {MAX_BATCH_SIZE}."
            )
        return v


# ── Response models ────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    """
    Response body for POST /predict and each item in POST /predict/batch.

    Fields
    ------
    consumer_id       : echoed from the request (null if not supplied).
    predicted_class   : 0 (Normal) or 1 (Theft flag).
    label             : human-readable class name.
    theft_probability : model output score in [0, 1].
                        NOTE: this is NOT a calibrated probability.
                        It is the raw sigmoid output of the stacking model.
    cnn_prediction    : intermediate score from the 5-seed CNN ensemble.
    missing_pct       : percentage of readings that were missing (0–100).
    threshold         : decision threshold used (default 0.5).
    model_version     : identifier of the deployed model.
    interpretation    : plain-language description of the result.
    """

    consumer_id:       Optional[str]
    predicted_class:   int
    label:             str
    theft_probability: float
    cnn_prediction:    float
    missing_pct:       float
    threshold:         float
    model_version:     str
    interpretation:    str

    model_config = {
        "json_schema_extra": {
            "example": {
                "consumer_id": "CONS_99001",
                "predicted_class": 1,
                "label": "Theft flag",
                "theft_probability": 0.6254,
                "cnn_prediction": 0.7312,
                "missing_pct": 8.51,
                "threshold": 0.5,
                "model_version": "WideDeepV2-5seed + XGBoost-stack",
                "interpretation": (
                    "The model flagged this consumer as a potential theft risk "
                    "(score 0.6254 ≥ threshold 0.5). "
                    "This is a model prediction, not a confirmed finding. "
                    "Further investigation is recommended."
                ),
            }
        }
    }


class BatchPredictResponse(BaseModel):
    """Response body for POST /predict/batch."""

    results: List[PredictResponse]
    total:   int


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status:        str
    model_ready:   bool
    app_version:   str
    model_version: str
    artifacts_ok:  bool
    errors:        List[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "model_ready": True,
                "app_version": "1.0.0",
                "model_version": "WideDeepV2-5seed + XGBoost-stack",
                "artifacts_ok": True,
                "errors": [],
            }
        }
    }


class ModelInfoResponse(BaseModel):
    """Verified metadata about the loaded inference pipeline."""

    model_version: str
    artifact_directory_configured: bool
    expected_readings: int
    stack_feature_count: int
    cnn_ensemble_size: int
    prediction_capabilities: List[str]
    explainability: List[str]


class ExplanationItem(BaseModel):
    feature: str
    value: float
    shap_value: float


class ExplainResponse(BaseModel):
    consumer_id: Optional[str]
    base_value: float = Field(
        description="SHAP base value in the explainer's raw-margin output space."
    )
    prediction: float = Field(
        description="Final stack probability in [0, 1]."
    )
    raw_prediction: float = Field(
        description="Final stack raw margin explained by the SHAP values."
    )
    output_space: Literal["raw"]
    features: List[ExplanationItem]
    model_version: str


# ── Response builder helpers ───────────────────────────────────────────────

def build_interpretation(
    prob: float,
    predicted_class: int,
    threshold: float,
) -> str:
    """Return a plain-language description of the prediction."""
    direction = "above" if predicted_class == 1 else "below"
    flag_word = "flagged as a potential theft risk" if predicted_class == 1 else "classified as normal"
    disclaimer = (
        " This is a model prediction, not a confirmed finding. "
        "Further investigation is recommended."
        if predicted_class == 1
        else " No anomaly was detected by the model."
    )
    return (
        f"The model {flag_word} "
        f"(score {prob:.4f} is {direction} threshold {threshold})."
        + disclaimer
    )
