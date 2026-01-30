"""Tests for the FastAPI serving endpoint."""

import pytest


class TestServingEndpoints:
    """Tests for the inference API."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        from fastapi.testclient import TestClient

        from scripts.serve.app import app, load_pytorch_model, state

        # Load a demo model for testing
        if state.model is None:
            load_pytorch_model(None)

        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["num_items"] == 1000  # Demo model default
        assert data["embedding_dim"] == 64

    def test_recommend_endpoint(self, client):
        """Test the recommendation endpoint."""
        response = client.post(
            "/recommend",
            json={"session_items": [1, 2, 3], "k": 5},
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data["recommendations"]) == 5
        assert len(data["scores"]) == 5
        assert data["latency_ms"] > 0
        # Verify session items are excluded from recommendations
        for item in [1, 2, 3]:
            assert item not in data["recommendations"]

    def test_recommend_empty_session(self, client):
        """Test that empty sessions return 400."""
        response = client.post(
            "/recommend",
            json={"session_items": [], "k": 5},
        )
        assert response.status_code == 400

    def test_recommend_invalid_items(self, client):
        """Test that invalid item IDs are filtered."""
        response = client.post(
            "/recommend",
            json={"session_items": [99999, -1], "k": 5},
        )
        assert response.status_code == 400

    def test_recommend_batch_endpoint(self, client):
        """Test the batch recommendation endpoint."""
        response = client.post(
            "/recommend/batch",
            json=[
                {"session_items": [1, 2], "k": 3},
                {"session_items": [4, 5, 6], "k": 2},
            ],
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 2
        assert len(data[0]["recommendations"]) == 3
        assert len(data[1]["recommendations"]) == 2
