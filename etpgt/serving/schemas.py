"""Request and response shapes for the recommendation API.

Pydantic gives us the first, cheapest layer of validation for free: the JSON
must parse into these types at all (session_items must be a list of ints, k an
int) before our own semantic gate in validation.py even runs.
"""

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """A session to get recommendations for."""

    session_items: list[int] = Field(
        ..., description="Item IDs the shopper viewed this session, in order."
    )
    k: int | None = Field(
        default=None, description="How many items to recommend (defaults to the server's default_k)."
    )


class RecommendResponse(BaseModel):
    """Top-k recommendations for a session."""

    recommendations: list[int] = Field(..., description="Recommended item IDs, best first.")
    scores: list[float] = Field(..., description="Model score for each recommended item.")
    latency_ms: float = Field(..., description="Server-side inference time in milliseconds.")
    dropped_items: list[int] = Field(
        default_factory=list,
        description="Item IDs from the request that were out of range and ignored.",
    )
    truncated: bool = Field(
        default=False, description="True if the session was longer than the limit and was truncated."
    )


class HealthResponse(BaseModel):
    """Liveness and model-readiness."""

    status: str
    model_loaded: bool
    num_items: int
    embedding_dim: int
