def test_predict_valid_input_returns_200(client):
    response = client.post("/api/v1/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "setosa"
    assert 0 <= data["confidence"] <= 1

def test_predict_missing_field_returns_422(client):
    response = client.post("/api/v1/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
        # petal_width missing on purpose
    })
    assert response.status_code == 422

def test_predict_invalid_type_returns_422(client):
    response = client.post("/api/v1/predict", json={
        "sepal_length": "not_a_number",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 422

def test_predict_negative_values_returns_422(client):
    response = client.post("/api/v1/predict", json={
        "sepal_length": -5.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 422

def test_predict_batch_oversized_rejected(client):
    huge_batch = {
        "inputs": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        ] * 100  # assuming MAX_BATCH_SIZE is less than 100
    }
    response = client.post("/api/v1/predict-batch", json=huge_batch)
    assert response.status_code == 400