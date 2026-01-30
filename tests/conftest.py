"""Shared pytest fixtures for GAT-Recommendation tests."""

import pytest
import torch
from torch_geometric.data import Batch, Data


@pytest.fixture
def small_model_config():
    """Small model configuration for fast testing.

    Returns:
        dict: Model configuration with small dimensions.
    """
    return {
        "num_items": 100,
        "embedding_dim": 32,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.1,
    }


@pytest.fixture
def dummy_batch():
    """Create a minimal PyG Batch for testing models.

    Creates a batch with 2 small graphs (sessions):
    - Graph 1: 3 nodes, 4 edges
    - Graph 2: 4 nodes, 6 edges

    Returns:
        Batch: PyG Batch object with node features and edge indices.
    """
    # Graph 1: 3 nodes (items 1, 2, 3), 4 edges
    data1 = Data(
        x=torch.tensor([1, 2, 3]),  # Item indices
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),  # Bidirectional edges
    )

    # Graph 2: 4 nodes (items 4, 5, 6, 7), 6 edges
    data2 = Data(
        x=torch.tensor([4, 5, 6, 7]),  # Item indices
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
    )

    # Create batch
    batch = Batch.from_data_list([data1, data2])
    return batch


@pytest.fixture
def dummy_batch_with_targets(dummy_batch):
    """Create a batch with target and negative items for loss computation.

    Args:
        dummy_batch: Base batch fixture.

    Returns:
        Batch: PyG Batch with target_item and negative_items attributes.
    """
    batch = dummy_batch

    # Target items for each session (2 samples)
    batch.target_item = torch.tensor([10, 20])  # Items to predict

    # Negative items for contrastive learning (5 negatives per sample)
    batch.negative_items = torch.tensor(
        [[11, 12, 13, 14, 15], [21, 22, 23, 24, 25]]
    )

    return batch


@pytest.fixture
def dummy_session_embeddings():
    """Create dummy session embeddings for loss function testing.

    Returns:
        torch.Tensor: Session embeddings [batch_size=4, hidden_dim=32].
    """
    return torch.randn(4, 32)


@pytest.fixture
def dummy_item_embeddings():
    """Create dummy item embedding layer for loss function testing.

    Returns:
        torch.nn.Embedding: Item embedding layer [num_items=100, embedding_dim=32].
    """
    embedding = torch.nn.Embedding(100, 32)
    # Normalize for better numerical stability
    with torch.no_grad():
        embedding.weight.data = embedding.weight.data / embedding.weight.data.norm(
            dim=1, keepdim=True
        )
    return embedding


@pytest.fixture
def dummy_targets_and_negatives():
    """Create dummy target and negative item indices for loss testing.

    Returns:
        tuple: (target_items, negative_items) tensors.
    """
    batch_size = 4
    num_negatives = 5

    target_items = torch.randint(1, 100, (batch_size,))
    negative_items = torch.randint(1, 100, (batch_size, num_negatives))

    return target_items, negative_items


@pytest.fixture
def device():
    """Get the device to run tests on.

    Returns:
        torch.device: CPU device (CUDA not required for unit tests).
    """
    return torch.device("cpu")
