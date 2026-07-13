"""FastAPI app for session recommendations.

Deliberately thin. All the real work lives in the sibling modules; this file
only wires them together in the one correct order:

    request  ->  validate_request (gate)  ->  Recommender.recommend  ->  response

Run locally:
    uvicorn etpgt.serving.app:app --host 0.0.0.0 --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from etpgt.serving.config import DEFAULT_LIMITS
from etpgt.serving.recommender import Recommender
from etpgt.serving.schemas import HealthResponse, RecommendRequest, RecommendResponse
from etpgt.serving.validation import InputValidationError, validate_request

_state: dict = {"recommender": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        _state["recommender"] = Recommender.from_default()
    except Exception as exc:  # missing checkpoint / graph: serve /health, refuse /recommend
        print(f"[serving] model not loaded: {exc}")
        _state["recommender"] = None
    yield
    _state["recommender"] = None


app = FastAPI(title="Session Recommendation API", version="2.0.0", lifespan=lifespan)


def _require_recommender() -> Recommender:
    recommender = _state["recommender"]
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return recommender


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    recommender = _state["recommender"]
    return HealthResponse(
        status="ok" if recommender is not None else "unavailable",
        model_loaded=recommender is not None,
        num_items=recommender.num_items if recommender else 0,
        embedding_dim=recommender.embedding_dim if recommender else 0,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    recommender = _require_recommender()

    # The gate runs first. Only a sanitized request reaches the model.
    try:
        validated = validate_request(request, recommender.num_items, DEFAULT_LIMITS)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    start = time.perf_counter()
    recommendations, scores = recommender.recommend(validated)
    latency_ms = (time.perf_counter() - start) * 1000

    return RecommendResponse(
        recommendations=recommendations,
        scores=scores,
        latency_ms=round(latency_ms, 3),
        dropped_items=validated.dropped_items,
        truncated=validated.truncated,
    )
