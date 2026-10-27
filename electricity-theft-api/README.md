# Electricity Theft Detection API

FastAPI inference service for the Samsung Innovation Campus electricity-theft
model. The service uses the exported trained pipeline and never retrains or
substitutes a dummy model.

## Project structure

```text
electricity-theft-api/
├── app/
│   ├── __init__.py
│   ├── config.py          # Root-relative paths and inference constants
│   ├── inference.py       # Preprocessing, CNN ensemble, stack prediction, SHAP
│   ├── main.py            # FastAPI app, lifecycle, endpoints, error handling
│   ├── model_loader.py    # Validated startup loading of exported models
│   ├── preprocessing.py   # Notebook-equivalent stateless preprocessing
│   └── schemas.py         # Pydantic request and response models
├── artifacts/
│   ├── models/            # CNN and XGBoost inference models
│   ├── explainability/    # SHAP explainer
│   └── metadata/          # Ordered feature names
├── docs/ARTIFACT_INVENTORY.md
├── tests/
├── .env.example
├── Dockerfile
├── requirements.txt
└── README.md
```

The existing preprocessing and focused health/prediction tests are retained;
they are not removed just to match a shorter example tree.

## Verified inference pipeline

The API accepts exactly 1,034 chronological daily readings. JSON `null`
represents a missing reading.

1. Raw readings are preserved for the XGBoost stack and missing percentage is
   appended.
2. The CNN branch applies the notebook's neighbour fill, per-consumer
   outlier cap, and per-consumer min-max scaling.
3. Five exported `WideDeepV2` models run on the cleaned wide input and the
   cleaned weekly grid plus missing-mask channel.
4. The averaged CNN score is appended to form the 1,036-feature stack input.
5. `final_stack_model.joblib` returns the final XGBoost prediction.

The stack's `Missing_Reading_Percentage` feature is the notebook's missing
fraction in `[0, 1]`; the API response field `missing_pct` presents the same
value as a human-readable percentage in `[0, 100]`.

Required production files:

- `artifacts/models/final_stack_model.joblib`
- `artifacts/models/cnn_ens_state_dicts.pt`
- `artifacts/metadata/stack_feature_names.joblib`

The full classification and disposition of retained and removed exports is in
[`docs/ARTIFACT_INVENTORY.md`](docs/ARTIFACT_INVENTORY.md). Training notebooks,
datasets, and source/export scripts remain outside this API directory.

## Installation and configuration

Python 3.10+ is supported. From the API directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The default artifact root is the repository's `artifacts\` directory. To use
another categorized artifact root:

```powershell
$env:MODEL_ARTIFACTS_DIR = "C:\path\to\electricity-theft-api\artifacts"
```

Optional browser CORS origins are configured with a comma-separated
`ALLOWED_ORIGINS` value. CORS is disabled when it is empty. Keep secrets out
of `.env.example` and source control.

## Run

```powershell
python -m uvicorn app.main:app --reload
```

Swagger UI is available at `http://127.0.0.1:8000/docs`; ReDoc is available at
`http://127.0.0.1:8000/redoc`.

## Endpoints

- `GET /` - welcome and service status.
- `GET /health` - readiness status. Missing or invalid required artifacts
  produce HTTP 503 without exposing filesystem paths.
- `GET /model-info` - verified dimensions, capabilities, CNN count, and
  explainability availability.
- `POST /predict` - prediction for one consumer.
- `POST /predict/batch` - ordered predictions for 1 to 500 consumers.
- `POST /explain?top_n=10` - local SHAP explanation using
  `artifacts/explainability/explainer.joblib`; returns HTTP 501 if that
  optional artifact is unavailable.

The explanation response contains `prediction` as the final probability, but
`base_value`, `raw_prediction`, and each `shap_value` are in the exported
SHAP explainer's raw-margin output space. Their additive relationship is
`base_value + sum(shap_value) ~= raw_prediction`.

Example request shape (the placeholder is explanatory and is not valid JSON):

```json
{
  "consumer_id": "CONS_001",
  "readings": [12.5, null, 9.8, "... 1031 more values ..."]
}
```

For a real request, replace the placeholder with exactly 1,034 JSON numbers or
`null` values. A frontend can generate a valid body as follows:

```python
payload = {
    "consumer_id": "CONS_001",
    "readings": readings,  # list with exactly 1034 float/int values or None
}
```

Successful prediction response shape:

```json
{
  "consumer_id": "CONS_001",
  "predicted_class": 0,
  "label": "Normal",
  "theft_probability": 0.092133,
  "cnn_prediction": 0.017035,
  "missing_pct": 0.0,
  "threshold": 0.5,
  "model_version": "WideDeepV2-5seed + XGBoost-stack",
  "interpretation": "The model classified as normal (score 0.0921 is below threshold 0.5). No anomaly was detected by the model."
}
```

Responses include the predicted class, label, model score, intermediate CNN
score, missing percentage, threshold, and model version. The model score is
not calibrated evidence of theft and should be reviewed by a human.

## Tests and verification

```powershell
python -m pytest -q
python -m compileall -q app tests
```

The test suite includes mocked API validation, preprocessing tests, and
real-artifact integration tests. The real model smoke test validates loading
the five CNN state dictionaries, the XGBoost stack, and SHAP explanation.

## Model provenance limitation

The supplied `final_stack_model.joblib` was trained with `Xt_stack`, whose
`CNN_Prediction` column was generated by the notebook's leakage-aware
cross-validation models (`test_cnn` in notebook Cell 86). the supplied `cnn_ens_state_dicts.pt` contains the separate five-seed ensemble
from Cell 79. The API loads and uses all five supplied CNN state dictionaries
correctly, but the export does not include the Cell 86 cross-validation CNN
models. During the audit, the saved evaluation files showed a reference
probability of `0.6254143119`, while the API's five-seed feature produced
`0.668077` after the verified preprocessing. Those evaluation files were
removed during final cleanup because they are not required by the backend.
The class remained `Theft flag`, but the probability was not equivalent.

To establish exact production equivalence, export the five CNN models (or
their predictions) used to generate `test_cnn`/`Xt_stack`, or retrain/export a
stack model whose CNN feature is generated by the supplied five-seed
checkpoint. Neither action was performed during this audit.

For production deployment, add authentication, rate limiting, and TLS at the
gateway or service boundary before exposing the API publicly.
