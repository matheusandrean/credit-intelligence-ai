"""Integration tests for the FastAPI application, using TestClient (no
running server needed) and Demo Mode (no API key needed)."""

from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDER", "demo")

import pytest
from fastapi.testclient import TestClient

from src.utils.config import get_settings


def _artifacts_available() -> bool:
    settings = get_settings()
    return (settings.models_path / "champion_calibrated.joblib").exists() and (
        settings.data_path / "processed" / "scored_portfolio.parquet"
    ).exists()


pytestmark = pytest.mark.skipif(
    not _artifacts_available(), reason="trained model artifacts not present"
)


@pytest.fixture(scope="module")
def client():
    from api.main import app

    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_portfolio_summary(client: TestClient) -> None:
    response = client.get("/portfolio/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["total_customers"] > 0


def test_customer_endpoints(client: TestClient) -> None:
    assert client.get("/customer/CUST_000001").status_code == 200
    assert client.get("/customer/CUST_000001/risk").status_code == 200
    assert client.get("/customer/CUST_000001/explanation").status_code == 200


def test_unknown_customer_returns_404(client: TestClient) -> None:
    response = client.get("/customer/CUST_999999/risk")
    assert response.status_code == 404


def test_model_metrics(client: TestClient) -> None:
    response = client.get("/model/metrics")
    assert response.status_code == 200
    assert "champion_model" in response.json()


def test_monitoring_drift(client: TestClient) -> None:
    response = client.get("/monitoring/drift")
    assert response.status_code == 200


def test_what_if_simulation(client: TestClient) -> None:
    response = client.post(
        "/simulation/what-if",
        json={"customer_id": "CUST_000001", "income_shock_pct": -0.1},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_simulation"] is True


def test_stress_test(client: TestClient) -> None:
    response = client.post("/stress-test", json={"scenario": "severe"})
    assert response.status_code == 200
    body = response.json()
    assert body["stressed"]["average_pd"] >= body["baseline"]["average_pd"]


def test_stress_test_invalid_scenario_returns_400(client: TestClient) -> None:
    response = client.post("/stress-test", json={"scenario": "apocalypse"})
    assert response.status_code == 400


def test_ai_chat(client: TestClient) -> None:
    response = client.post(
        "/ai/chat", json={"question": "What is the current risk profile of the portfolio?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "demo"
    assert "portfolio_summary" in body["tools_called"]


def test_openapi_lists_all_expected_paths(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = set(response.json()["paths"].keys())
    expected = {
        "/health",
        "/portfolio/summary",
        "/customer/{customer_id}",
        "/customer/{customer_id}/risk",
        "/customer/{customer_id}/explanation",
        "/model/metrics",
        "/monitoring/drift",
        "/simulation/what-if",
        "/stress-test",
        "/ai/chat",
    }
    assert expected <= paths
