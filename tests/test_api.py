from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_model_metadata() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_name" in payload
    assert "metrics" in payload


def test_predict_endpoint_returns_eta_payload() -> None:
    payload = {
        "pickup_lat": 19.0760,
        "pickup_lng": 72.8777,
        "drop_lat": 18.5204,
        "drop_lng": 73.8567,
        "traffic_bucket": "medium",
        "weather_bucket": "clear",
        "vehicle_type": "van",
        "package_weight_kg": 3.6,
        "warehouse_load_pct": 67.0,
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200

    prediction = response.json()
    assert prediction["eta_minutes"] > 0
    assert prediction["eta_window_max"] >= prediction["eta_window_min"]
    assert isinstance(prediction["explanation"], list)
