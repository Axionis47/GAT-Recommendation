"""Smoke tests for the real-model recommender and the app wiring.

The real-model tests skip automatically when the trained checkpoint is not
present (e.g. on CI without the weights), so they never cause false failures.
"""

from pathlib import Path

import pytest

from etpgt.serving import RecommendRequest, validate_request

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "best_model.pt"
GRAPH = ROOT / "data" / "processed" / "graph_edges.csv"

needs_model = pytest.mark.skipif(
    not (CKPT.exists() and GRAPH.exists()),
    reason="trained checkpoint / graph not present",
)

REAL_SESSION = [285930, 357564, 67045]


@pytest.fixture(scope="module")
def recommender():
    from etpgt.serving.recommender import Recommender

    return Recommender.from_default()


@needs_model
def test_recommends_k_valid_unseen_items(recommender):
    validated = validate_request(
        RecommendRequest(session_items=REAL_SESSION, k=5), recommender.num_items
    )
    ids, scores = recommender.recommend(validated)

    assert len(ids) == len(scores) == 5
    assert all(0 <= i < recommender.num_items for i in ids)
    assert set(ids).isdisjoint(REAL_SESSION)  # never recommend an item already seen
    assert scores == sorted(scores, reverse=True)  # best first


@needs_model
def test_health_reports_the_real_catalog(recommender):
    health = recommender.health()
    assert health["num_items"] == 466865
    assert round(health["val_recall_at_10"], 4) == 0.3828


def test_app_runs_the_gate_then_the_model(monkeypatch):
    """The app must reject bad input (422) and wire good input to the model."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from etpgt.serving import app as app_module

    class FakeRecommender:
        num_items = 100
        embedding_dim = 8

        def recommend(self, validated):
            picks = [10, 20, 30][: validated.k]
            return picks, [0.9, 0.8, 0.7][: validated.k]

    # Replace the loader so lifespan uses the fake instead of the 1.5 GB checkpoint.
    monkeypatch.setattr(
        app_module.Recommender, "from_default", classmethod(lambda cls, **kw: FakeRecommender())
    )

    with TestClient(app_module.app) as client:
        # empty session is stopped by the gate, before the model
        assert client.post("/recommend", json={"session_items": [], "k": 5}).status_code == 422

        # valid request flows through to the (fake) model
        ok = client.post("/recommend", json={"session_items": [1, 2, 3], "k": 2})
        assert ok.status_code == 200
        body = ok.json()
        assert body["recommendations"] == [10, 20]
        assert body["latency_ms"] >= 0

        health = client.get("/health").json()
        assert health["model_loaded"] is True
        assert health["num_items"] == 100
