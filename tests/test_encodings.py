"""Tests for positional encodings."""

import torch

from etpgt.encodings.laplacian_pe import compute_laplacian_pe
from etpgt.encodings.path_encoding import path_length_to_bucket
from etpgt.encodings.temporal_encoding import time_delta_to_bucket


def test_lappe_shape() -> None:
    """Test that LapPE produces correct output shape."""
    # Create a simple graph (triangle)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    num_nodes = 3
    k = 16

    # Compute LapPE
    pe = compute_laplacian_pe(edge_index, num_nodes, k=k)

    # Check shape (should be [num_nodes, k] but k might be clamped to num_nodes-1)
    assert pe.shape[0] == num_nodes
    assert pe.shape[1] <= k  # Can be less if graph is small


def test_temporal_buckets_correct() -> None:
    """Test that temporal delta buckets are assigned correctly."""
    # Create time deltas in milliseconds
    time_deltas = torch.tensor(
        [
            30 * 1000,  # 30 seconds -> bucket 0 (0-1m)
            2 * 60 * 1000,  # 2 minutes -> bucket 1 (1-5m)
            10 * 60 * 1000,  # 10 minutes -> bucket 2 (5-30m)
            60 * 60 * 1000,  # 1 hour -> bucket 3 (30m-2h)
            12 * 60 * 60 * 1000,  # 12 hours -> bucket 4 (2h-24h)
            3 * 24 * 60 * 60 * 1000,  # 3 days -> bucket 5 (24h-7d)
            10 * 24 * 60 * 60 * 1000,  # 10 days -> bucket 6 (7d+)
        ]
    )

    expected_buckets = torch.tensor([0, 1, 2, 3, 4, 5, 6])

    # Assign to buckets
    buckets = time_delta_to_bucket(time_deltas)

    # Check correctness
    assert torch.equal(buckets, expected_buckets), f"Expected {expected_buckets}, got {buckets}"


def test_path_buckets_correct() -> None:
    """Test that path length buckets are assigned correctly."""
    # Create path lengths
    path_lengths = torch.tensor([1, 2, 3, 4, 5, 10, 100])

    expected_buckets = torch.tensor(
        [
            0,  # path=1 -> bucket 0
            1,  # path=2 -> bucket 1
            2,  # path=3 -> bucket 2
            2,  # path=4 -> bucket 2 (3+)
            2,  # path=5 -> bucket 2 (3+)
            2,  # path=10 -> bucket 2 (3+)
            2,  # path=100 -> bucket 2 (3+)
        ]
    )

    # Assign to buckets
    buckets = path_length_to_bucket(path_lengths)

    # Check correctness
    assert torch.equal(buckets, expected_buckets), f"Expected {expected_buckets}, got {buckets}"
