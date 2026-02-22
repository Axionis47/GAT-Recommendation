#!/usr/bin/env python3
"""FastAPI server for session-based recommendations.

This provides a simple REST API for inference:
- POST /recommend: Get top-k recommendations for a session

Usage:
    # Start server
    uvicorn scripts.serve.app:app --host 0.0.0.0 --port 8000

    # Test endpoint
    curl -X POST http://localhost:8000/recommend \
        -H "Content-Type: application/json" \
        -d '{"session_items": [1, 2, 3], "k": 10}'
"""

import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from etpgt.model import create_graph_transformer_optimized


# Request/Response models
class RecommendRequest(BaseModel):
    session_items: list[int]
    k: int = 10


class RecommendResponse(BaseModel):
    recommendations: list[int]
    scores: list[float]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_items: int
    embedding_dim: int


# Global model state
class ModelState:
    def __init__(self):
        self.model = None
        self.ort_session = None
        self.item_embeddings = None
        self.num_items = 0
        self.embedding_dim = 0
        self.device = "cpu"


state = ModelState()


# FastAPI app
app = FastAPI(
    title="Session Recommendation API",
    description="GNN-based session recommendations",
    version="1.0.0",
)


def load_pytorch_model(checkpoint_path: Path = None):
    """Load PyTorch model for inference."""
    print("Loading PyTorch model...")

    # Default: create a demo model
    if checkpoint_path is None or not checkpoint_path.exists():
        print("No checkpoint found, creating demo model...")
        state.num_items = 1000
        state.embedding_dim = 64

        state.model = create_graph_transformer_optimized(
            num_items=state.num_items,
            embedding_dim=state.embedding_dim,
            hidden_dim=state.embedding_dim,
            use_laplacian_pe=False,
        )
    else:
        # Load from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        state.num_items = config.get("num_items", 1000)
        state.embedding_dim = config.get("hidden_dim", 64)

        state.model = create_graph_transformer_optimized(
            num_items=state.num_items,
            embedding_dim=config.get("embedding_dim", 64),
            hidden_dim=state.embedding_dim,
            use_laplacian_pe=False,
        )
        state.model.load_state_dict(checkpoint["model_state_dict"])

    state.model.eval()
    state.item_embeddings = state.model.get_item_embeddings().detach()
    print(f"Model loaded: {state.num_items} items, {state.embedding_dim}d embeddings")


def load_onnx_model(onnx_path: Path):
    """Load ONNX model for optimized inference."""
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX Runtime not installed")

    print(f"Loading ONNX model from {onnx_path}...")
    state.ort_session = ort.InferenceSession(str(onnx_path))
    print("ONNX model loaded")


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    project_root = Path(__file__).parent.parent.parent

    # Try to load checkpoint if exists
    checkpoint_path = project_root / "checkpoints" / "best_model.pt"
    load_pytorch_model(checkpoint_path if checkpoint_path.exists() else None)

    # Try to load ONNX if exists
    onnx_path = project_root / "exports" / "onnx" / "session_recommender.onnx"
    if ONNX_AVAILABLE and onnx_path.exists():
        load_onnx_model(onnx_path)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.model is not None,
        num_items=state.num_items,
        embedding_dim=state.embedding_dim,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Get recommendations for a session.

    Takes a list of item IDs the user has viewed and returns
    the top-k recommended items.
    """
    start_time = time.time()

    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate items
    session_items = request.session_items
    if not session_items:
        raise HTTPException(status_code=400, detail="Session items cannot be empty")

    # Filter to valid item IDs
    valid_items = [i for i in session_items if 0 <= i < state.num_items]
    if not valid_items:
        raise HTTPException(status_code=400, detail="No valid item IDs in session")

    # Compute session embedding (mean of item embeddings)
    # This is a simplified version - full inference would use GNN message passing
    item_indices = torch.tensor(valid_items, dtype=torch.long)
    session_embedding = state.item_embeddings[item_indices].mean(dim=0, keepdim=True)

    # Normalize for cosine similarity
    session_norm = session_embedding / (session_embedding.norm(dim=-1, keepdim=True) + 1e-8)
    item_norm = state.item_embeddings / (state.item_embeddings.norm(dim=-1, keepdim=True) + 1e-8)

    # Compute scores
    scores = torch.matmul(session_norm, item_norm.t()).squeeze(0)

    # Exclude items already in session
    for item_id in valid_items:
        scores[item_id] = float("-inf")

    # Get top-k
    k = min(request.k, state.num_items - len(valid_items))
    top_scores, top_indices = torch.topk(scores, k)

    latency_ms = (time.time() - start_time) * 1000

    return RecommendResponse(
        recommendations=top_indices.tolist(),
        scores=top_scores.tolist(),
        latency_ms=round(latency_ms, 3),
    )


@app.post("/recommend/batch")
async def recommend_batch(sessions: list[RecommendRequest]):
    """Batch recommendation endpoint for multiple sessions."""
    results = []
    for session in sessions:
        result = await recommend(session)
        results.append(result)
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
