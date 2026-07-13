"""Tests for the serving input-validation gate.

Every case here is something that must be caught BEFORE the model runs.
"""

import pytest

from etpgt.serving import InputValidationError, RecommendRequest, ServingLimits, validate_request

NUM_ITEMS = 1000
LIMITS = ServingLimits(max_session_length=5, default_k=10, max_k=20)


def req(items, k=None):
    return RecommendRequest(session_items=items, k=k)


def test_valid_request_passes_through_unchanged():
    v = validate_request(req([1, 2, 3], k=5), NUM_ITEMS, LIMITS)
    assert v.session_items == [1, 2, 3]
    assert v.k == 5
    assert v.dropped_items == []
    assert v.truncated is False


def test_empty_session_is_rejected():
    with pytest.raises(InputValidationError):
        validate_request(req([]), NUM_ITEMS, LIMITS)


def test_session_of_only_unknown_items_is_rejected():
    with pytest.raises(InputValidationError):
        validate_request(req([9999, -1]), NUM_ITEMS, LIMITS)


def test_out_of_range_items_are_dropped_and_order_is_preserved():
    v = validate_request(req([5, 9999, 7, -1, 3]), NUM_ITEMS, LIMITS)
    assert v.session_items == [5, 7, 3]
    assert v.dropped_items == [9999, -1]


def test_k_below_one_is_rejected():
    with pytest.raises(InputValidationError):
        validate_request(req([1, 2], k=0), NUM_ITEMS, LIMITS)


def test_k_defaults_when_not_given():
    v = validate_request(req([1, 2]), NUM_ITEMS, LIMITS)
    assert v.k == LIMITS.default_k


def test_k_is_clamped_to_the_cap():
    v = validate_request(req([1, 2], k=999), NUM_ITEMS, LIMITS)
    assert v.k == LIMITS.max_k


def test_over_long_session_keeps_the_most_recent_items():
    v = validate_request(req([1, 2, 3, 4, 5, 6, 7], k=3), NUM_ITEMS, LIMITS)
    assert v.session_items == [3, 4, 5, 6, 7]
    assert v.truncated is True


def test_non_integer_items_are_rejected():
    # bypass pydantic coercion to exercise the gate's own type guard
    bad = RecommendRequest.model_construct(session_items=[True, 2], k=None)
    with pytest.raises(InputValidationError):
        validate_request(bad, NUM_ITEMS, LIMITS)
