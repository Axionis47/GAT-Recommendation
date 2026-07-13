"""Modular serving for session-based recommendations.

The serving flow is deliberately split into small, single-responsibility pieces:

    request  ->  validate_request (the gate)  ->  Recommender  ->  response

The validation gate runs BEFORE any model inference. Nothing unchecked ever
reaches the model. Each piece can be imported and tested on its own:

    config.py       static limits (session length, k bounds)
    schemas.py      request / response shapes (pydantic)
    validation.py   the gate: sanitize input or raise a typed error
    recommender.py  load the real checkpoint, run the GNN, return top-k
    app.py          thin FastAPI wiring
"""

from etpgt.serving.config import DEFAULT_LIMITS, ServingLimits
from etpgt.serving.schemas import HealthResponse, RecommendRequest, RecommendResponse
from etpgt.serving.validation import (
    InputValidationError,
    ValidatedRequest,
    validate_request,
)

__all__ = [
    "ServingLimits",
    "DEFAULT_LIMITS",
    "RecommendRequest",
    "RecommendResponse",
    "HealthResponse",
    "validate_request",
    "ValidatedRequest",
    "InputValidationError",
]
