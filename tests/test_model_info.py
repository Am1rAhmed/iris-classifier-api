def test_model_info_returns_expected_keys(client):
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "model_version" in data
    assert "features" in data
    assert "classes" in data