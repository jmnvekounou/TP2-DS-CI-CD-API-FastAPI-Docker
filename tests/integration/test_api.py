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

def test_predict_success():
    """
        Cette fonction test et valide une prédiction correcte avec les valeur: [1.0, 2.0, 3.0] 
    """
    payload = {"features": [1.0, 2.0, 3.0]}
    
    # Envoi de la requête
    response = client.post("/predict", json = payload)
    
    # Validation des assertions
    assert response.status_code == 200

    assert  response.json() == {"predictions": [2.0, 4.0, 6.0]}

