"""Static serving limits.

These are the guard rails the validation gate enforces. They are separate from
the model so they can be tuned (or tested) without touching inference code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ServingLimits:
    """Bounds applied to every incoming request.

    Attributes:
        min_session_length: Fewest valid items a session may contain.
        max_session_length: Longest session we feed the model. Longer sessions
            are truncated to the most recent items (matches training, which used
            the last 50 events of a session).
        default_k: Recommendations returned when the request does not specify k.
        max_k: Largest number of recommendations we will return.
    """

    min_session_length: int = 1
    max_session_length: int = 50
    default_k: int = 10
    max_k: int = 100


DEFAULT_LIMITS = ServingLimits()
