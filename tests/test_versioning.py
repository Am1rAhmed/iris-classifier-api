def test_v1_and_v2_predict_different_shapes(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    v1_response = client.post("/api/v1/predict", json=payload)
    v2_response = client.post("/api/v2/predict", json=payload)

    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_data = v1_response.json()
    v2_data = v2_response.json()

    # v1 must still have its original shape — proves v1 wasn't broken
    assert "confidence" in v1_data
    assert isinstance(v1_data["confidence"], float)
    assert "probabilities" not in v1_data

    # v2 must have the new shape
    assert "probabilities" in v2_data
    assert isinstance(v2_data["probabilities"], dict)
    assert "confidence" not in v2_data

    # both agree on the actual prediction itself
    assert v1_data["prediction"] == v2_data["prediction"] == "setosa"