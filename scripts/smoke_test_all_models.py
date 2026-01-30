#!/usr/bin/env python3
"""Smoke test script to verify all models can train for a few epochs.

This script creates synthetic data and trains each model variant for 2 epochs
to ensure everything is working correctly before running full training.

Usage:
    python scripts/smoke_test_all_models.py --device cpu
    python scripts/smoke_test_all_models.py --device cuda
"""

import argparse
import sys
import time
from typing import Any

import torch
from torch_geometric.data import Batch, Data

from etpgt.model import (
    create_gat,
    create_graph_transformer,
    create_graph_transformer_optimized,
    create_graphsage,
)
from etpgt.train.losses import create_loss_function


def create_synthetic_batch(batch_size: int = 4, device: str = "cpu") -> Batch:
    """Create a synthetic batch for testing.

    Args:
        batch_size: Number of graphs in the batch.
        device: Device to place tensors on.

    Returns:
        PyG Batch object with synthetic data.
    """
    data_list = []
    for i in range(batch_size):
        num_nodes = 3 + i  # Varying graph sizes
        x = torch.randint(1, 100, (num_nodes,))

        # Create simple sequential edges
        edge_index = torch.tensor(
            [
                list(range(num_nodes - 1)) + list(range(1, num_nodes)),
                list(range(1, num_nodes)) + list(range(num_nodes - 1)),
            ]
        )

        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

    batch = Batch.from_data_list(data_list)

    # Add targets and negatives
    batch.target_item = torch.randint(1, 100, (batch_size,))
    batch.negative_items = torch.randint(1, 100, (batch_size, 5))

    # Move to device
    batch = batch.to(device)

    return batch


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
) -> float:
    """Train model for one step and return loss.

    Args:
        model: The model to train.
        loss_fn: Loss function.
        optimizer: Optimizer.
        batch: Training batch.

    Returns:
        Loss value.
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    session_embeddings = model(batch)

    # Compute loss
    loss = loss_fn(
        session_embeddings,
        batch.target_item,
        batch.negative_items,
        model.get_item_embeddings(),
    )

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def test_model(
    _model_name: str,
    create_fn: Any,
    config: dict,
    device: str,
    num_epochs: int = 2,
) -> tuple[bool, str, float]:
    """Test a single model variant.

    Args:
        model_name: Name of the model for logging.
        create_fn: Factory function to create the model.
        config: Model configuration.
        device: Device to run on.
        num_epochs: Number of epochs to train.

    Returns:
        Tuple of (success, message, duration).
    """
    start_time = time.time()

    try:
        # Create model
        model = create_fn(**config)
        model = model.to(device)

        # Create optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = create_loss_function("listwise")

        # Create synthetic batch
        batch = create_synthetic_batch(batch_size=4, device=device)

        # Train for a few epochs
        losses = []
        for _ in range(num_epochs):
            loss = train_one_epoch(model, loss_fn, optimizer, batch)
            losses.append(loss)

        duration = time.time() - start_time

        # Verify loss is valid
        if any(torch.isnan(torch.tensor(loss_val)) for loss_val in losses):
            return False, "NaN loss detected", duration

        return True, f"Losses: {losses}", duration

    except Exception as e:
        duration = time.time() - start_time
        return False, str(e), duration


def main() -> int:
    """Run smoke tests for all models.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Smoke test all models")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run tests on",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs per model",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Running smoke tests on device: {device}")
    print("=" * 60)

    # Base configuration
    base_config = {
        "num_items": 100,
        "embedding_dim": 32,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.1,
    }

    # Models to test
    models = [
        ("GraphSAGE", create_graphsage, base_config),
        ("GAT", create_gat, base_config),
        (
            "GraphTransformer (with FFN)",
            create_graph_transformer,
            {**base_config, "use_laplacian_pe": False, "use_ffn": True},
        ),
        (
            "GraphTransformer (no FFN)",
            create_graph_transformer_optimized,
            {**base_config, "use_laplacian_pe": False},
        ),
    ]

    results = []
    all_passed = True

    for model_name, create_fn, config in models:
        print(f"\nTesting {model_name}...")
        success, message, duration = test_model(
            model_name, create_fn, config, device, args.epochs
        )

        status = "PASS" if success else "FAIL"
        results.append((model_name, status, message, duration))

        if success:
            print(f"  [{status}] {message} ({duration:.2f}s)")
        else:
            print(f"  [{status}] {message} ({duration:.2f}s)")
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for model_name, status, _, duration in results:
        status_icon = "[OK]" if status == "PASS" else "[XX]"
        print(f"  {status_icon} {model_name}: {status} ({duration:.2f}s)")

    total_time = sum(r[3] for r in results)
    print(f"\nTotal time: {total_time:.2f}s")

    if all_passed:
        print("\nAll models passed smoke tests!")
        return 0
    else:
        print("\nSome models failed smoke tests!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
