"""
test_health.py — Tests for GET /health.

These tests use a mocked model store so they do not require real artifacts.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ready_client():
    """TestClient with a fully mocked ready model store."""
    from app.main import app

    mock_store = MagicMock()
    mock_store.is_ready = True
    mock_store.load_errors = []

    with patch("app.main.load_all_artifacts", return_value=mock_store), \
         patch("app.main.get_store", return_value=mock_store):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


@pytest.fixture(scope="module")
def degraded_client():
    """TestClient with a mocked store that has load errors."""
    from app.main import app

    mock_store = MagicMock()
    mock_store.is_ready = False
    mock_store.load_errors = ["Failed to load cnn_model_1.pth"]

    with patch("app.main.load_all_artifacts", return_value=mock_store), \
         patch("app.main.get_store", return_value=mock_store):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


# ── Tests ──────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200_when_ready(self, ready_client):
        resp = ready_client.get("/health")
        assert resp.status_code == 200

    def test_health_body_model_ready_true(self, ready_client):
        body = ready_client.get("/health").json()
        assert body["model_ready"] is True

    def test_health_body_status_ok(self, ready_client):
        body = ready_client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_body_has_app_version(self, ready_client):
        body = ready_client.get("/health").json()
        assert "app_version" in body
        assert body["app_version"] != ""

    def test_health_body_has_model_version(self, ready_client):
        body = ready_client.get("/health").json()
        assert "model_version" in body

    def test_health_body_errors_empty_when_ready(self, ready_client):
        body = ready_client.get("/health").json()
        assert body["errors"] == []

    def test_health_returns_503_when_degraded(self, degraded_client):
        resp = degraded_client.get("/health")
        assert resp.status_code == 503

    def test_health_body_model_ready_false_when_degraded(self, degraded_client):
        body = degraded_client.get("/health").json()
        assert body["model_ready"] is False

    def test_health_body_errors_populated_when_degraded(self, degraded_client):
        body = degraded_client.get("/health").json()
        assert len(body["errors"]) > 0
