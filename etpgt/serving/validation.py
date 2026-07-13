"""The input-validation gate.

Every request passes through `validate_request` BEFORE the model is touched.
It either returns a clean, bounded `ValidatedRequest` that inference can trust,
or raises `InputValidationError` with a message explaining exactly what was
wrong. This module has no torch and no web-framework imports on purpose: the
gate is pure logic, so it can be reasoned about and tested in isolation.
"""

from dataclasses import dataclass, field

from etpgt.serving.config import DEFAULT_LIMITS, ServingLimits
from etpgt.serving.schemas import RecommendRequest


class InputValidationError(ValueError):
    """A request failed validation. The message is safe to return to the caller."""


@dataclass
class ValidatedRequest:
    """A request the model can trust: bounded length, in-range ids, sane k.

    Attributes:
        session_items: In-range item ids, original order preserved, truncated
            to the session-length limit.
        k: Number of recommendations to return, within [1, max_k].
        dropped_items: Ids from the raw request that were out of range and removed.
        truncated: True if the session was longer than the limit and was trimmed.
    """

    session_items: list[int]
    k: int
    dropped_items: list[int] = field(default_factory=list)
    truncated: bool = False


def validate_request(
    request: RecommendRequest,
    num_items: int,
    limits: ServingLimits = DEFAULT_LIMITS,
) -> ValidatedRequest:
    """Sanitize a raw request, or raise InputValidationError.

    Args:
        request: The parsed request (pydantic has already enforced basic types).
        num_items: Catalog size. Valid item ids are [0, num_items).
        limits: Guard rails to enforce.

    Returns:
        A ValidatedRequest safe to pass to the model.

    Raises:
        InputValidationError: If the request cannot be made valid.
    """
    items = request.session_items

    # 1. Must contain something.
    if not items:
        raise InputValidationError("session_items must not be empty.")

    # 2. Must be plain integers. Guard against bools (a bool is an int in Python)
    #    and anything pydantic might have coerced loosely.
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            raise InputValidationError(
                f"session_items must be integers; got {item!r} of type {type(item).__name__}."
            )

    # 3. Keep only ids that exist in the catalog; report the rest.
    valid = [i for i in items if 0 <= i < num_items]
    dropped = [i for i in items if not (0 <= i < num_items)]
    if not valid:
        raise InputValidationError(
            f"no usable item ids in session: all {len(items)} were outside the "
            f"catalog range [0, {num_items})."
        )

    # 4. Bound the session length (keep the most recent items, as in training).
    truncated = len(valid) > limits.max_session_length
    if truncated:
        valid = valid[-limits.max_session_length :]

    # 5. Resolve and bound k.
    k = limits.default_k if request.k is None else request.k
    if k < 1:
        raise InputValidationError(f"k must be at least 1; got {k}.")
    # Cannot recommend more items than exist, nor more than the hard cap.
    k = min(k, limits.max_k, num_items - 1)

    return ValidatedRequest(
        session_items=valid,
        k=k,
        dropped_items=dropped,
        truncated=truncated,
    )
