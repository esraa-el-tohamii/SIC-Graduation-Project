"""
main.py — FastAPI application entry point.

Endpoints
---------
GET  /health           — liveness + readiness check
POST /predict          — single-consumer theft detection
POST /predict/batch    — multi-consumer theft detection (up to 500)
GET  /docs             — Swagger UI (auto-generated)
GET  /redoc            — ReDoc UI (auto-generated)
"""

from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import (
    ALLOWED_ORIGINS,
    APP_VERSION,
    ARTIFACTS_DIR,
    DEFAULT_THRESHOLD,
    EXPECTED_READINGS,
    MODEL_VERSION,
)
from app.model_loader import load_all_artifacts, get_store
from app.schemas import (
    PredictRequest,
    BatchPredictRequest,
    PredictResponse,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    ExplainResponse,
    ExplanationItem,
    build_interpretation,
)
from app.inference import explain_inference, run_inference, run_inference_batch
from fastapi.middleware.cors import CORSMiddleware

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (replaces deprecated startup/shutdown events) ────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts …")
    store = load_all_artifacts()
    if store.is_ready:
        logger.info("All artifacts ready.  API is accepting requests.")
    else:
        logger.warning(
            "Some artifacts failed to load: %s", store.load_errors
        )
    yield
    logger.info("Application shutting down.")


# ── Application ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Electricity Theft Detection API",
    description=(
        "Inference service for the Samsung Innovation Campus electricity "
        "theft detection model.  "
        "The model is a two-stage stacking pipeline: a 5-seed WideDeepV2 CNN "
        "ensemble generates a probability score, which is then used as a "
        "feature by an XGBoost classifier trained on raw daily readings."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# ── Global exception handler ───────────────────────────────────────────────

def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert any value in a Pydantic error structure to something
    that is safe to pass to json.dumps.

    In Pydantic v2 the error dict can contain:
      - ``ctx``: a dict whose ``error`` key is an *Exception instance*
      - ``input``: the raw request value, which may be inf/nan
      - ``loc``:  a *tuple* (not JSON-serialisable as-is; convert to list)

    Handles:
      - Non-finite floats  → str ("inf", "-inf", "nan")
      - Exception objects  → str(exception)
      - tuple             → list (tuples are not in JSON)
      - dict / list       → recurse
      - Everything else   → attempt to keep; fall back to str()
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    if isinstance(obj, Exception):
        return str(obj)
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    # For everything else, attempt serialisation by converting to str
    # only if the standard types would fail (int, str, bool, None are safe).
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom 422 handler that guarantees a JSON-serialisable error response
    even when the request contained non-finite float values (inf, -inf, nan).
    Without this, FastAPI's default handler crashes with:
      ValueError: Out of range float values are not JSON compliant
    """
    safe_errors = _sanitize_for_json(exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": safe_errors},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred.  Please check server logs."},
    )


# ── A. Health endpoint ─────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root() -> dict:
    return {
        "message": "Electricity Theft Detection API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Return verified model metadata",
    tags=["System"],
)
def model_info() -> ModelInfoResponse:
    store = get_store()
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        artifact_directory_configured=ARTIFACTS_DIR.exists(),
        expected_readings=EXPECTED_READINGS,
        stack_feature_count=len(store.feature_names or []),
        cnn_ensemble_size=len(store.cnn_models),
        prediction_capabilities=store.prediction_capabilities,
        explainability=["SHAP"] if store.explainer is not None else [],
    )

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness check",
    tags=["System"],
)
def health() -> HealthResponse:
    """
    Returns HTTP 200 if the service is running and all model artifacts have
    been loaded successfully.  Returns HTTP 503 if a required artifact is
    missing or failed to load.
    """
    store = get_store()
    ready = store.is_ready
    public_errors = (
        []
        if ready
        else ["One or more required model artifacts failed to load."]
    )
    response = HealthResponse(
        status="ok" if ready else "degraded",
        model_ready=ready,
        app_version=APP_VERSION,
        model_version=MODEL_VERSION,
        artifacts_ok=len(store.load_errors) == 0,
        errors=public_errors,
    )
    if not ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(),
        )
    return response


# ── B. Single-consumer prediction endpoint ─────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict electricity theft for one consumer",
    tags=["Prediction"],
)
def predict(body: PredictRequest) -> PredictResponse:
    """
    Accept raw daily electricity consumption readings for one consumer and
    return a theft risk prediction.

    **Input**
    - `consumer_id`: optional identifier (returned unchanged).
    - `readings`: exactly 1034 floats in chronological order
      (2014-01-01 → 2016-10-31).  Use JSON `null` for missing days.

    **Output**
    - `predicted_class`: 0 = Normal, 1 = Theft flag.
    - `theft_probability`: model score in [0, 1] (not a calibrated probability).
    - `threshold`: 0.5 (not tuned on this consumer; see README).

    **Note**: A positive prediction is a model risk flag, not proof of theft.
    Human investigation is always required.
    """
    store = get_store()
    if not store.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Check /health for service status.",
        )

    try:
        result = run_inference(
            readings=body.readings,
            store=store,
            threshold=DEFAULT_THRESHOLD,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    return PredictResponse(
        consumer_id=body.consumer_id,
        predicted_class=result["predicted_class"],
        label=result["label"],
        theft_probability=result["theft_probability"],
        cnn_prediction=result["cnn_prediction"],
        missing_pct=result["missing_pct"],
        threshold=result["threshold"],
        model_version=result["model_version"],
        interpretation=build_interpretation(
            result["theft_probability"],
            result["predicted_class"],
            result["threshold"],
        ),
    )


# ── C. Batch prediction endpoint ───────────────────────────────────────────

@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Predict electricity theft for multiple consumers",
    tags=["Prediction"],
)
def predict_batch(body: BatchPredictRequest) -> BatchPredictResponse:
    """
    Accept up to 500 consumers in a single request.

    Each consumer follows the same schema as POST /predict.
    Results are returned in the same order as the input.
    """
    store = get_store()
    if not store.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Check /health for service status.",
        )

    readings_list = [c.readings for c in body.consumers]

    try:
        results = run_inference_batch(
            readings_list=readings_list,
            store=store,
            threshold=DEFAULT_THRESHOLD,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    responses = []
    for req, result in zip(body.consumers, results):
        responses.append(
            PredictResponse(
                consumer_id=req.consumer_id,
                predicted_class=result["predicted_class"],
                label=result["label"],
                theft_probability=result["theft_probability"],
                cnn_prediction=result["cnn_prediction"],
                missing_pct=result["missing_pct"],
                threshold=result["threshold"],
                model_version=result["model_version"],
                interpretation=build_interpretation(
                    result["theft_probability"],
                    result["predicted_class"],
                    result["threshold"],
                ),
            )
        )

    return BatchPredictResponse(results=responses, total=len(responses))


@app.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explain a prediction with the exported SHAP explainer",
    tags=["Prediction"],
)
def explain(
    body: PredictRequest,
    top_n: int = Query(default=10, ge=1, le=50),
) -> ExplainResponse:
    store = get_store()
    if not store.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready.",
        )
    if store.explainer is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="SHAP explanations are not available.",
        )
    try:
        result = explain_inference(body.readings, store, top_n=top_n)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    return ExplainResponse(
        consumer_id=body.consumer_id,
        base_value=result["base_value"],
        prediction=result["prediction"],
        raw_prediction=result["raw_prediction"],
        output_space=result["output_space"],
        features=[ExplanationItem(**item) for item in result["features"]],
        model_version=result["model_version"],
    )
