from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is up and running!"}

def test_favicon():
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.json() == ""

@patch("app.main.predict")
def test_predict_endpoint_success(mock_predict):
    # Arrange the mock
    mock_predict.return_value = [0.5, 0.8]
    payload = {"features": [1.0, 2.0, 3.5]}
    
    # Sending the request
    response = client.post("/predict", json=payload)
    
    # Assert
    assert response.status_code == 200
        
    assert response.json() == {"predictions": [0.5, 0.8]}
    mock_predict.assert_called_once_with([1.0, 2.0, 3.5])
    


def test_predict_endpoint_with_real_model():
    payload = {"features": [1.0, 2.0, 3.5]}
    
    # Sending the request
    response = client.post("/predict", json=payload)
    
    # Assert
    assert response.status_code == 200

    real_result = response.json()

    assert  real_result == {"predictions": [2.0, 4.0, 7.0]}


def test_predict_endpoint_invalid_data():
    payload = {"features": "inavid list"}

    # Sending the request
    response = client.post("/predict", json=payload)

    assert response.status_code == 422

