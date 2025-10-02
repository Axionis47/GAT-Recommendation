"""Tests for utility modules."""

import tempfile
from pathlib import Path

import pytest
import torch

from etpgt.utils.io import load_config, load_json, save_json
from etpgt.utils.metrics import compute_ndcg_at_k, compute_recall_at_k
from etpgt.utils.seed import set_seed


def test_set_seed() -> None:
    """Test that set_seed produces reproducible results."""
    set_seed(42)
    rand1 = torch.rand(10)

    set_seed(42)
    rand2 = torch.rand(10)

    assert torch.allclose(rand1, rand2), "Random tensors should be identical with same seed"


def test_save_and_load_json() -> None:
    """Test JSON save and load."""
    data = {"key": "value", "number": 42, "list": [1, 2, 3]}

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test.json"
        save_json(data, str(json_path))

        loaded_data = load_json(str(json_path))

        assert loaded_data == data, "Loaded data should match saved data"


def test_load_config() -> None:
    """Test YAML config loading."""
    config_content = """
model:
  hidden_dim: 256
  num_layers: 3
training:
  batch_size: 32
  lr: 0.001
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))

        assert config["model"]["hidden_dim"] == 256
        assert config["training"]["batch_size"] == 32


def test_compute_recall_at_k() -> None:
    """Test Recall@K computation."""
    # predictions: batch_size=3, num_candidates=5
    predictions = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])

    # targets: batch_size=3
    targets = torch.tensor([2, 9, 20])  # 2 is in top-5, 9 is in top-5, 20 is not

    recall_at_5 = compute_recall_at_k(predictions, targets, k=5)
    assert recall_at_5 == pytest.approx(2 / 3, abs=1e-6), "Recall@5 should be 2/3"

    recall_at_2 = compute_recall_at_k(predictions, targets, k=2)
    assert recall_at_2 == pytest.approx(1 / 3, abs=1e-6), "Recall@2 should be 1/3"


def test_compute_ndcg_at_k() -> None:
    """Test NDCG@K computation."""
    # predictions: batch_size=3, num_candidates=5
    predictions = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])

    # targets: batch_size=3
    targets = torch.tensor([1, 9, 20])  # 1 at pos 0, 9 at pos 3, 20 not in top-5

    ndcg_at_5 = compute_ndcg_at_k(predictions, targets, k=5)

    # Expected NDCG:
    # Sample 0: target at position 0 -> DCG = 1/log2(2) = 1.0
    # Sample 1: target at position 3 -> DCG = 1/log2(5) ≈ 0.4307
    # Sample 2: target not in top-5 -> DCG = 0.0
    # Average: (1.0 + 0.4307 + 0.0) / 3 ≈ 0.4769

    assert ndcg_at_5 > 0.4 and ndcg_at_5 < 0.5, f"NDCG@5 should be ~0.48, got {ndcg_at_5}"
