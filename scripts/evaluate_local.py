#!/usr/bin/env python3
"""Local evaluation script for model checkpoints.

Loads each model checkpoint and evaluates on the test set.

Usage:
    python scripts/evaluate_local.py
    python scripts/evaluate_local.py --model graph_transformer_optimized
    python scripts/evaluate_local.py --batch-size 64 --num-samples 1000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from etpgt.model import (
    create_gat,
    create_graph_transformer,
    create_graph_transformer_optimized,
    create_graphsage,
)
from etpgt.train.dataloader import SessionDataset, collate_fn
from etpgt.utils.metrics import compute_ndcg_at_k, compute_recall_at_k

# Model configurations (matching training)
MODEL_CONFIGS = {
    "graphsage": {
        "factory": create_graphsage,
        "checkpoint": "graphsage_best.pt",
        "kwargs": {
            "embedding_dim": 256,
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.1,
            "readout_type": "mean",
            "aggregator": "mean",
        },
    },
    "gat": {
        "factory": create_gat,
        "checkpoint": "gat_best.pt",
        "kwargs": {
            "embedding_dim": 256,
            "hidden_dim": 256,
            "num_layers": 3,
            "num_heads": 4,
            "dropout": 0.1,
            "readout_type": "mean",
            "concat_heads": False,
        },
    },
    "graph_transformer": {
        "factory": create_graph_transformer,
        "checkpoint": "graph_transformer_best.pt",
        "kwargs": {
            "embedding_dim": 256,
            "hidden_dim": 256,
            "num_layers": 3,
            "num_heads": 4,
            "dropout": 0.1,
            "readout_type": "mean",
            "use_laplacian_pe": True,
            "laplacian_k": 16,
            "use_ffn": True,
        },
    },
    "graph_transformer_optimized": {
        "factory": create_graph_transformer_optimized,
        "checkpoint": "graph_transformer_optimized_best.pt",
        "kwargs": {
            "embedding_dim": 256,
            "hidden_dim": 256,
            "num_layers": 2,
            "num_heads": 2,
            "dropout": 0.1,
            "readout_type": "mean",
            "use_laplacian_pe": True,
            "laplacian_k": 16,
            "use_ffn": False,
        },
    },
}


def load_model(model_name: str, num_items: int, checkpoint_path: Path, device: str):
    """Load a model from checkpoint."""
    config = MODEL_CONFIGS[model_name]

    # Create model
    model = config["factory"](num_items=num_items, **config["kwargs"])

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Extract cached PE if present (for Graph Transformer models)
    cached_pe_key = "laplacian_pe._cached_pe"
    cached_pe = None
    if cached_pe_key in state_dict:
        cached_pe = state_dict.pop(cached_pe_key)

    # Load model weights (excluding cached PE which is not a parameter)
    model.load_state_dict(state_dict, strict=False)

    # Restore cached PE if model uses Laplacian PE
    if cached_pe is not None and hasattr(model, "laplacian_pe"):
        model.laplacian_pe._cached_pe = cached_pe

    model.to(device)
    model.eval()

    return model, checkpoint.get("epoch", "unknown")


def evaluate_model(model, dataloader, device: str, k_values: list[int] | None = None):
    """Evaluate model on test data."""
    if k_values is None:
        k_values = [10, 20]
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)

            # Get session embeddings
            session_emb = model(batch)  # [batch_size, hidden_dim]

            # Get item embeddings
            item_emb = model.get_item_embeddings()  # [num_items, hidden_dim]

            # Compute scores (dot product)
            scores = torch.matmul(session_emb, item_emb.t())  # [batch_size, num_items]

            # Get top-k predictions
            max_k = max(k_values)
            _, top_k_indices = torch.topk(scores, k=max_k, dim=1)

            all_predictions.append(top_k_indices.cpu())
            all_targets.append(batch.target_item.cpu())

    # Concatenate all predictions and targets
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = compute_recall_at_k(predictions, targets, k)
        metrics[f"ndcg@{k}"] = compute_ndcg_at_k(predictions, targets, k)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate models locally")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default="all",
        help="Model to evaluate (default: all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    checkpoint_dir = project_root / "checkpoints"

    # Check data exists
    test_path = data_dir / "test.csv"
    graph_path = data_dir / "graph_edges.csv"

    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        print("Run 'make data' first to generate processed data.")
        sys.exit(1)

    print(f"Loading test data from {test_path}...")
    print(f"Using device: {args.device}")

    # Get num_items from graph
    edges_df = pd.read_csv(graph_path)
    test_df = pd.read_csv(test_path)
    num_items = max(
        test_df["itemid"].max(),
        edges_df["item_i"].max(),
        edges_df["item_j"].max(),
    ) + 1
    print(f"Number of items: {num_items}")

    # Create dataset
    dataset = SessionDataset(
        sessions_path=test_path,
        graph_edges_path=graph_path,
        num_negatives=1,  # Minimal negatives for evaluation
        max_session_length=50,
    )

    # Optionally limit samples
    if args.num_samples:
        indices = list(range(min(args.num_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Evaluating on {len(indices)} samples (limited)")
    else:
        print(f"Evaluating on {len(dataset)} sessions")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Determine models to evaluate
    models_to_eval = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    # Results
    results = {}

    for model_name in models_to_eval:
        config = MODEL_CONFIGS[model_name]
        checkpoint_path = checkpoint_dir / config["checkpoint"]

        if not checkpoint_path.exists():
            print(f"\nSkipping {model_name}: checkpoint not found at {checkpoint_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Checkpoint: {checkpoint_path.name}")
        print(f"{'='*60}")

        # Load model
        start_time = time.time()
        model, epoch = load_model(model_name, num_items, checkpoint_path, args.device)
        load_time = time.time() - start_time
        print(f"Loaded model from epoch {epoch} in {load_time:.2f}s")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

        # Evaluate
        start_time = time.time()
        metrics = evaluate_model(model, dataloader, args.device)
        eval_time = time.time() - start_time

        # Store results
        results[model_name] = {
            "epoch": epoch,
            "num_params": num_params,
            "metrics": metrics,
            "eval_time": eval_time,
        }

        # Print results
        print("\nResults:")
        print(f"  Recall@10:  {metrics['recall@10']*100:.2f}%")
        print(f"  Recall@20:  {metrics['recall@20']*100:.2f}%")
        print(f"  NDCG@10:    {metrics['ndcg@10']*100:.2f}%")
        print(f"  NDCG@20:    {metrics['ndcg@20']*100:.2f}%")
        print(f"  Eval time:  {eval_time:.2f}s")

        # Clean up GPU memory
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Recall@10':>10} {'Recall@20':>10} {'NDCG@10':>10} {'Params':>12}")
    print("-" * 80)

    for model_name, result in results.items():
        m = result["metrics"]
        print(
            f"{model_name:<30} "
            f"{m['recall@10']*100:>9.2f}% "
            f"{m['recall@20']*100:>9.2f}% "
            f"{m['ndcg@10']*100:>9.2f}% "
            f"{result['num_params']:>12,}"
        )

    # Save results
    output_path = project_root / "evaluation_results.json"
    with open(output_path, "w") as f:
        # Convert metrics to serializable format
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                "epoch": result["epoch"],
                "num_params": result["num_params"],
                "eval_time": result["eval_time"],
                "metrics": {k: float(v) for k, v in result["metrics"].items()},
            }
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
