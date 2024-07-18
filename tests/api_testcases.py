import pytest
from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

@pytest.fixture
def test_client():
    return client

@pytest.fixture
def valid_data():
    data = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "Bachelors",
        "education_num": 9,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Female",
        "capital_gain": 4236,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "Puerto-Rico"
    }
    return data

@pytest.fixture
def invalid_data():
    data = json.dumps({})
    return data

def test_welcome(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome!"

def test_predict_with_valid_data(test_client, valid_data):
    response = test_client.post("/predict", json=valid_data)
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_with_missing_feature(test_client, invalid_data):
    response = test_client.post("/predict", json=invalid_data)
    print(response.status_code)
    assert response.status_code == 422
