import os

from fastapi.testclient import TestClient
from app.main import API_VERSION
from app.main import app
from app.services.model_service import model_service


TEST_API_KEY = "test-secret-key"

os.environ["API_KEY"] = TEST_API_KEY

client = TestClient(app)


def build_valid_payload():
    features = {
        feature_name: 0.0
        for feature_name in model_service.feature_names
    }

    return {
        "features": features
    }


def auth_headers():
    return {
        "X-API-Key": TEST_API_KEY
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "p8-credit-scoring-api"
    assert data["version"] == API_VERSION


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "P6_credit_scoring_default_risk_model"
    assert data["model_version"] == 2
    assert data["model_family"] == "XGBoost"
    assert data["mlflow_alias"] == "champion"
    assert data["decision_threshold"] == 0.45
    assert data["n_features"] == 656


def test_predict_without_api_key():
    response = client.post(
        "/predict",
        json=build_valid_payload(),
    )

    assert response.status_code == 401


def test_predict_with_invalid_api_key():
    response = client.post(
        "/predict",
        json=build_valid_payload(),
        headers={
            "X-API-Key": "wrong-key"
        },
    )

    assert response.status_code == 401


def test_predict_valid_payload():
    response = client.post(
        "/predict",
        json=build_valid_payload(),
        headers=auth_headers(),
    )

    assert response.status_code == 200

    data = response.json()

    assert 0.0 <= data["probability_default"] <= 1.0
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in [
        "client_risque",
        "client_non_risque",
    ]
    assert data["threshold"] == 0.45


def test_predict_missing_feature():
    payload = build_valid_payload()

    feature_to_remove = next(iter(payload["features"]))
    del payload["features"][feature_to_remove]

    response = client.post(
        "/predict",
        json=payload,
        headers=auth_headers(),
    )

    assert response.status_code == 422


def test_predict_extra_feature():
    payload = build_valid_payload()

    payload["features"]["FEATURE_INCONNUE"] = 123.0

    response = client.post(
        "/predict",
        json=payload,
        headers=auth_headers(),
    )

    assert response.status_code == 422


def test_predict_invalid_type():
    payload = build_valid_payload()

    first_feature = next(iter(payload["features"]))
    payload["features"][first_feature] = "texte_invalide"

    response = client.post(
        "/predict",
        json=payload,
        headers=auth_headers(),
    )

    assert response.status_code == 422