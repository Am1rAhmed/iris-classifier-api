import httpx
from app.config import settings

BASE_URL = "https://iris-classifier-api-s7zg.onrender.com"
HEADERS = {"X-API-Key": settings.API_KEY}


def test_health_check_over_http():
    response = httpx.get(f"{BASE_URL}/api/v1/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_over_http():
    response = httpx.post(
        f"{BASE_URL}/api/v1/predict",
        json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        headers=HEADERS
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "setosa"


def test_predict_batch_over_http():
    response = httpx.post(
        f"{BASE_URL}/api/v1/predict-batch",
        json={"inputs": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 4.7, "sepal_width": 3.2, "petal_length": 1.3, "petal_width": 0.2},
    {"sepal_length": 5.0, "sepal_width": 3.6, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 5.4, "sepal_width": 3.9, "petal_length": 1.7, "petal_width": 0.4},
    {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4},
    {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5},
    {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 4.9, "petal_width": 1.5},
    {"sepal_length": 5.5, "sepal_width": 2.3, "petal_length": 4.0, "petal_width": 1.3},
    {"sepal_length": 6.5, "sepal_width": 2.8, "petal_length": 4.6, "petal_width": 1.5},
    {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
    {"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.9},
    {"sepal_length": 7.1, "sepal_width": 3.0, "petal_length": 5.9, "petal_width": 2.1},
    {"sepal_length": 6.3, "sepal_width": 2.9, "petal_length": 5.6, "petal_width": 1.8},
    {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2},
    {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3},
    {"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5}

        ]},
        headers=HEADERS
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 17


def test_metrics_over_http():
    response = httpx.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    assert "iris_predictions_total" in response.text