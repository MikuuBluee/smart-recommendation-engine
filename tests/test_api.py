import pytest
from fastapi.testclient import TestClient
from src.serving.main import app
import os

client = TestClient(app)
CI_ENV = os.getenv("CI", "false").lower() == "true"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

@pytest.mark.skipif(CI_ENV, reason="Skip in CI - Redis might not be ready")
def test_recommendations_new_user():
    response = client.get("/recommend/user_baru_test_123")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

@pytest.mark.skipif(CI_ENV, reason="Skip in CI - Redis might not be ready")
def test_get_recommendations_existing_user():
    response = client.get("/recommend/6f7c8b3369dd1f736a77264ecd928a47")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data