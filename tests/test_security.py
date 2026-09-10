from app.config import settings

API_KEY = settings.API_KEY

def test_predict_without_api_key_returns_401(client):
    response = client.post("/api/v1/predict", json={
        "sepal_length": 5.1, 
        "sepal_width": 3.5,
        "petal_length": 1.4, 
        "petal_width": 0.2
    })
    assert response.status_code == 401

def test_predict_with_invalid_api_key_returns_401(client):
    response = client.post(
        "/api/v1/predict",
        json={"sepal_length": 5.1, 
              "sepal_width": 3.5, 
              "petal_length": 1.4, 
              "petal_width": 0.2},
        headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 401

def test_predict_with_valid_api_key_returns_200(client):
    response = client.post(
        "/api/v1/predict",
        json={"sepal_length": 5.1, 
              "sepal_width": 3.5, 
              "petal_length": 1.4, 
              "petal_width": 0.2},
        headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 200

def test_predict_extra_field_rejected(client):
    response = client.post(
        "/api/v1/predict",
        json={
            "sepal_length": 5.1, 
            "sepal_width": 3.5,
            "petal_length": 1.4, 
            "petal_width": 0.2,
            "unexpected_field": "should be rejected"
        },
        headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 422