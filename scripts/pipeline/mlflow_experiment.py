#!/usr/bin/env python3
"""MLflow experiment tracking for GAT-Recommendation.

This module provides MLflow integration for:
1. Experiment tracking
2. Model versioning
3. Artifact logging (checkpoints, metrics, configs)

Usage:
    python scripts/pipeline/mlflow_experiment.py --model graph_transformer_optimized
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARNING: MLflow not installed. Run: pip install mlflow")

from etpgt.model import (
    create_gat,
    create_graph_transformer,
    create_graph_transformer_optimized,
    create_graphsage,
)
from etpgt.train.losses import create_loss_function

# Model registry
MODEL_REGISTRY = {
    "graphsage": {
        "create_fn": create_graphsage,
        "description": "GraphSAGE baseline with mean aggregation",
        "training_notes": "Fastest training, good baseline",
    },
    "gat": {
        "create_fn": create_gat,
        "description": "Graph Attention Network with learned attention",
        "training_notes": "Moderate training time, better than GraphSAGE",
    },
    "graph_transformer": {
        "create_fn": create_graph_transformer,
        "description": "Full Graph Transformer with FFN",
        "training_notes": "Slowest training, highest capacity",
    },
    "graph_transformer_optimized": {
        "create_fn": create_graph_transformer_optimized,
        "description": "Optimized Graph Transformer without FFN",
        "training_notes": "88x faster than full, <3% accuracy loss",
    },
}


def create_batch_from_csv(sessions_path: Path, graph_path: Path, batch_size: int = 16):
    """Create a batch from CSV files."""
    from torch_geometric.data import Batch, Data

    sessions_df = pd.read_csv(sessions_path)
    graph_df = pd.read_csv(graph_path)

    # Get unique sessions
    session_ids = sessions_df['session_id'].unique()[:batch_size]

    # Build item to index mapping
    all_items = sorted(sessions_df['itemid'].unique())
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    num_items = len(all_items)

    data_list = []
    targets = []
    negatives_list = []

    for session_id in session_ids:
        session_data = sessions_df[sessions_df['session_id'] == session_id]
        session_data = session_data.sort_values('timestamp')
        items = session_data['itemid'].values

        if len(items) < 2:
            continue

        context_items = items[:-1]
        target_item = items[-1]
        context_indices = [item_to_idx[i] for i in context_items]
        target_idx = item_to_idx[target_item]

        local_items = sorted(set(context_indices))
        local_to_global = {i: idx for idx, i in enumerate(local_items)}

        context_set = set(context_items)
        session_edges = graph_df[
            (graph_df['item_i'].isin(context_set)) &
            (graph_df['item_j'].isin(context_set))
        ]

        if len(session_edges) > 0:
            edge_src = [local_to_global[item_to_idx[i]] for i in session_edges['item_i']]
            edge_dst = [local_to_global[item_to_idx[j]] for j in session_edges['item_j']]
            edge_index = torch.tensor([edge_src + edge_dst, edge_dst + edge_src], dtype=torch.long)
        else:
            n = len(local_items)
            edge_index = torch.tensor([list(range(n)), list(range(n))], dtype=torch.long)

        x = torch.tensor(local_items, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)
        targets.append(target_idx)

        session_items_set = set([item_to_idx[i] for i in items])
        available = [i for i in range(num_items) if i not in session_items_set][:5]
        neg_samples = torch.tensor(available if len(available) == 5 else list(range(5)), dtype=torch.long)
        negatives_list.append(neg_samples)

    batch = Batch.from_data_list(data_list)
    batch.target_item = torch.tensor(targets, dtype=torch.long)
    batch.negative_items = torch.stack(negatives_list)

    return batch, num_items


def train_with_mlflow(
    model_name: str,
    config: dict,
    sessions_path: Path,
    graph_path: Path,
    num_epochs: int = 5,
    experiment_name: str = "gat-recommendation",
):
    """Train a model with MLflow tracking.

    Args:
        model_name: Name of model from registry
        config: Model configuration
        sessions_path: Path to sessions CSV
        graph_path: Path to graph edges CSV
        num_epochs: Number of training epochs
        experiment_name: MLflow experiment name
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Install with: pip install mlflow")
        return

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_info = MODEL_REGISTRY[model_name]

    # Create batch
    batch, num_items = create_batch_from_csv(sessions_path, graph_path)
    config["num_items"] = num_items

    # Set experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_name}_{int(time.time())}"):
        # Log parameters
        mlflow.log_params(config)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("description", model_info["description"])
        mlflow.log_param("training_notes", model_info["training_notes"])

        # Create model with model-specific config
        create_fn = model_info["create_fn"]
        model_config = config.copy()

        # Remove non-model parameters
        lr = model_config.pop("lr", 0.001)

        # GraphSAGE doesn't use num_heads
        if model_name == "graphsage":
            model_config.pop("num_heads", None)
        elif model_name in ["gat", "graph_transformer", "graph_transformer_optimized"]:
            model_config.setdefault("num_heads", 2)

        # Add model-specific options
        if model_name in ["graph_transformer", "graph_transformer_optimized"]:
            model_config["use_laplacian_pe"] = False

        model = create_fn(**model_config)

        # Log model info
        param_count = sum(p.numel() for p in model.parameters())
        mlflow.log_metric("param_count", param_count)
        mlflow.log_metric("param_count_millions", param_count / 1e6)

        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = create_loss_function("listwise")

        print(f"\nTraining {model_name} with MLflow tracking...")
        print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            session_embeddings = model(batch)
            loss = loss_fn(
                session_embeddings,
                batch.target_item,
                batch.negative_items,
                model.item_embedding,  # Pass embedding layer, not weights
            )

            loss.backward()
            optimizer.step()

            # Log metrics
            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {loss.item():.4f}")

        # Log model
        mlflow.pytorch.log_model(model, "model")

        # Log final metrics
        mlflow.log_metric("final_loss", loss.item())

        print(f"\n✓ Run logged to MLflow: {mlflow.active_run().info.run_id}")

    return model


def main():
    parser = argparse.ArgumentParser(description="MLflow Experiment Tracking")
    parser.add_argument("--model", type=str, default="graph_transformer_optimized",
                       choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--experiment", type=str, default="gat-recommendation")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    sessions_path = project_root / "data/test_subset/sessions_subset.csv"
    graph_path = project_root / "data/test_subset/graph_subset.csv"

    # Fallback to full data if subset doesn't exist
    if not sessions_path.exists():
        sessions_path = project_root / "data/processed/train.csv"
        graph_path = project_root / "data/processed/graph_edges.csv"

    if not sessions_path.exists():
        print("ERROR: No data found. Run the data pipeline first.")
        return 1

    config = {
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": 0.1,
        "lr": 0.001,
    }

    train_with_mlflow(
        model_name=args.model,
        config=config,
        sessions_path=sessions_path,
        graph_path=graph_path,
        num_epochs=args.epochs,
        experiment_name=args.experiment,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
