"""
test_predict.py — Tests for POST /predict and POST /predict/batch.

Test structure
--------------
* MOCKED tests (class TestPredictMocked):
  Tests that verify API request/response behaviour using a fake inference
  function.  These run without any model artifacts on disk.

* REAL MODEL tests (class TestPredictRealModel):
  Tests that require the actual model artifacts to be present under the
  categorized electricity-theft-api/artifacts/ directory. They are skipped
  automatically if the artifacts are absent.

  To run real model tests:
    1.  Place the required files in artifacts/models/ and artifacts/metadata/
    2.  Run:  pytest tests/test_predict.py -k "real_model" -v
"""

import math
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.config import EXPECTED_READINGS, STACK_MODEL_FILE, CNN_ENSEMBLE_FILE

# ── Shared helpers ─────────────────────────────────────────────────────────

def _valid_readings(n=EXPECTED_READINGS, missing_count=50):
    """Return a valid readings list with a few nulls."""
    readings = [float(i % 100 + 1) for i in range(n)]
    for i in range(0, missing_count * 3, 3):
        if i < n:
            readings[i] = None
    return readings


def _make_mock_store(ready=True):
    store = MagicMock()
    store.is_ready = ready
    store.load_errors = [] if ready else ["missing cnn_model_1.pth"]
    return store


def _make_inference_result():
    return {
        "theft_probability": 0.6254,
        "predicted_class": 1,
        "label": "Theft flag",
        "cnn_prediction": 0.7312,
        "missing_pct": 4.83,
        "threshold": 0.5,
        "model_version": "WideDeepV2-5seed + XGBoost-stack",
    }


# ── Mocked tests ───────────────────────────────────────────────────────────

class TestPredictMocked:
    """All tests use a mocked inference function — no artifacts required."""

    @pytest.fixture(autouse=True)
    def client(self):
        from app.main import app
        mock_store = _make_mock_store(ready=True)

        with patch("app.main.load_all_artifacts", return_value=mock_store), \
             patch("app.main.get_store", return_value=mock_store), \
             patch("app.main.run_inference", return_value=_make_inference_result()), \
             patch("app.main.run_inference_batch", return_value=[_make_inference_result()]):
            with TestClient(app, raise_server_exceptions=False) as c:
                self.client = c
                yield

    # ── Valid requests ─────────────────────────────────────────────────────

    def test_valid_request_returns_200(self):
        resp = self.client.post(
            "/predict",
            json={"readings": _valid_readings()},
        )
        assert resp.status_code == 200

    def test_valid_request_with_consumer_id(self):
        resp = self.client.post(
            "/predict",
            json={"consumer_id": "CONS_001", "readings": _valid_readings()},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["consumer_id"] == "CONS_001"

    def test_response_has_required_fields(self):
        resp = self.client.post(
            "/predict",
            json={"readings": _valid_readings()},
        )
        body = resp.json()
        required = [
            "predicted_class", "label", "theft_probability",
            "cnn_prediction", "missing_pct", "threshold",
            "model_version", "interpretation",
        ]
        for field in required:
            assert field in body, f"Missing field: {field}"

    def test_predicted_class_in_valid_set(self):
        resp = self.client.post(
            "/predict",
            json={"readings": _valid_readings()},
        )
        assert resp.json()["predicted_class"] in {0, 1}

    def test_theft_probability_in_range(self):
        resp = self.client.post(
            "/predict",
            json={"readings": _valid_readings()},
        )
        prob = resp.json()["theft_probability"]
        assert 0.0 <= prob <= 1.0

    def test_interpretation_is_nonempty_string(self):
        resp = self.client.post(
            "/predict",
            json={"readings": _valid_readings()},
        )
        assert isinstance(resp.json()["interpretation"], str)
        assert len(resp.json()["interpretation"]) > 10

    # ── Invalid requests ───────────────────────────────────────────────────

    def test_wrong_reading_count_short(self):
        resp = self.client.post(
            "/predict",
            json={"readings": _valid_readings(n=EXPECTED_READINGS - 1)},
        )
        assert resp.status_code == 422

    def test_wrong_reading_count_long(self):
        resp = self.client.post(
            "/predict",
            json={"readings": _valid_readings(n=EXPECTED_READINGS + 1)},
        )
        assert resp.status_code == 422

    def test_infinite_value_rejected(self):
        readings = _valid_readings()
        readings[10] = float("inf")
        # JSON cannot encode inf — simulate as string first
        import json
        # FastAPI will reject any non-finite JSON number at parse time
        resp = self.client.post(
            "/predict",
            content=json.dumps({"readings": [
                ("inf" if r is None else r) for r in readings
            ]}).replace('"inf"', 'Infinity'),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_missing_readings_field_rejected(self):
        resp = self.client.post(
            "/predict",
            json={"consumer_id": "X"},
        )
        assert resp.status_code == 422

    def test_empty_readings_rejected(self):
        resp = self.client.post(
            "/predict",
            json={"readings": []},
        )
        assert resp.status_code == 422

    def test_negative_value_rejected(self):
        readings = _valid_readings()
        readings[5] = -1.0
        resp = self.client.post(
            "/predict",
            json={"readings": readings},
        )
        assert resp.status_code == 422

    # ── Batch endpoint ─────────────────────────────────────────────────────

    def test_batch_valid(self):
        resp = self.client.post(
            "/predict/batch",
            json={"consumers": [{"readings": _valid_readings()}]},
        )
        assert resp.status_code == 200

    def test_batch_response_total_matches(self):
        consumers = [{"readings": _valid_readings()}] * 3
        with patch("app.main.run_inference_batch",
                   return_value=[_make_inference_result()] * 3):
            resp = self.client.post(
                "/predict/batch",
                json={"consumers": consumers},
            )
        assert resp.json()["total"] == 3

    def test_batch_empty_rejected(self):
        resp = self.client.post(
            "/predict/batch",
            json={"consumers": []},
        )
        assert resp.status_code == 422

    # ── Model not ready ────────────────────────────────────────────────────

    def test_predict_503_when_model_not_ready(self):
        from app.main import app
        not_ready_store = _make_mock_store(ready=False)
        with patch("app.main.load_all_artifacts", return_value=not_ready_store), \
             patch("app.main.get_store", return_value=not_ready_store):
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post(
                    "/predict",
                    json={"readings": _valid_readings()},
                )
        assert resp.status_code == 503


# ── Real model tests ───────────────────────────────────────────────────────

_ARTIFACTS_PRESENT = (
    STACK_MODEL_FILE.exists()
    and CNN_ENSEMBLE_FILE.exists()
)

@pytest.mark.skipif(
    not _ARTIFACTS_PRESENT,
    reason=(
        "Real model artifacts not found.  "
        "Provide final_stack_model.joblib and cnn_ens_state_dicts.pt "
        "in the configured artifact directory before running these tests."
    ),
)
class TestPredictRealModel:
    """
    Integration tests that use the actual trained artifacts.
    Skipped automatically when artifacts are absent.
    """

    @pytest.fixture(scope="class", autouse=True)
    def real_client(self):
        from app.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            self.__class__.client = c
            yield

    def test_health_ready_with_real_artifacts(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_ready"] is True

    def test_all_zero_readings_returns_valid_prediction(self):
        readings = [0.0] * EXPECTED_READINGS
        resp = self.client.post("/predict", json={"readings": readings})
        assert resp.status_code == 200
        body = resp.json()
        prob = body["theft_probability"]
        assert math.isfinite(prob)
        assert 0.0 <= prob <= 1.0
        assert body["predicted_class"] in {0, 1}

    def test_all_null_readings_accepted(self):
        readings = [None] * EXPECTED_READINGS
        resp = self.client.post("/predict", json={"readings": readings})
        assert resp.status_code == 200
        body = resp.json()
        assert math.isclose(body["missing_pct"], 100.0, abs_tol=0.01)

    def test_typical_readings_returns_finite_probability(self):
        import random
        random.seed(42)
        readings = [random.uniform(0, 20) for _ in range(EXPECTED_READINGS)]
        readings[::7] = [None] * len(readings[::7])  # ~14% missing
        resp = self.client.post(
            "/predict",
            json={"consumer_id": "TEST_001", "readings": readings},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert math.isfinite(body["theft_probability"])
        assert body["consumer_id"] == "TEST_001"

    def test_batch_produces_same_result_as_single(self):
        """Batch result must match single-consumer result for the same input."""
        readings = [float(i % 30 + 1) for i in range(EXPECTED_READINGS)]
        readings[::10] = [None] * len(readings[::10])

        single = self.client.post(
            "/predict", json={"readings": readings}
        ).json()

        batch = self.client.post(
            "/predict/batch",
            json={"consumers": [{"readings": readings}]},
        ).json()

        batch_item = batch["results"][0]

        assert math.isclose(
            single["theft_probability"],
            batch_item["theft_probability"],
            abs_tol=1e-4,
        ), (
            f"Single: {single['theft_probability']} != "
            f"Batch: {batch_item['theft_probability']}"
        )

    def test_cnn_prediction_in_range(self):
        readings = [5.0] * EXPECTED_READINGS
        resp = self.client.post("/predict", json={"readings": readings})
        cnn = resp.json()["cnn_prediction"]
        assert 0.0 <= cnn <= 1.0
