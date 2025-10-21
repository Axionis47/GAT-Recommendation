#!/usr/bin/env python3
"""Train baseline models (GraphSAGE, GAT, GraphTransformer)."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from google.cloud import storage
from torch_geometric.data import Data

from etpgt.model import (
    create_gat,
    create_graph_transformer,
    create_graph_transformer_optimized,
    create_graphsage,
)
from etpgt.train.dataloader import create_dataloader
from etpgt.train.trainer import Trainer
from etpgt.utils.logging import get_logger
from etpgt.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline models")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["graphsage", "gat", "graph_transformer", "graph_transformer_optimized"],
        help="Model type",
    )
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--readout-type",
        type=str,
        default="mean",
        choices=["mean", "max", "last", "attention"],
        help="Session readout type",
    )

    # Data arguments
    parser.add_argument(
        "--train-sessions", type=str, default="data/processed/train.csv", help="Training sessions"
    )
    parser.add_argument(
        "--val-sessions", type=str, default="data/processed/val.csv", help="Validation sessions"
    )
    parser.add_argument(
        "--graph-edges",
        type=str,
        default="data/processed/graph_edges.csv",
        help="Graph edges",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-negatives", type=int, default=5, help="Number of negative samples")
    parser.add_argument("--max-session-length", type=int, default=50, help="Maximum session length")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data workers")

    # Training arguments
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--gcs-bucket", type=str, default=None, help="GCS bucket for outputs")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    return parser.parse_args()


def download_from_gcs(bucket_name: str, source_path: str, dest_path: str) -> None:
    """Download file from GCS.

    Args:
        bucket_name: GCS bucket name.
        source_path: Source path in GCS.
        dest_path: Destination path locally.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_path)

    # Create parent directory
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

    blob.download_to_filename(dest_path)
    logger.info(f"Downloaded gs://{bucket_name}/{source_path} to {dest_path}")


def upload_to_gcs(bucket_name: str, source_path: str, dest_path: str) -> None:
    """Upload file to GCS.

    Args:
        bucket_name: GCS bucket name.
        source_path: Source path locally.
        dest_path: Destination path in GCS.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_path)

    blob.upload_from_filename(source_path)
    logger.info(f"Uploaded {source_path} to gs://{bucket_name}/{dest_path}")


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Log arguments
    logger.info("Training arguments:")
    logger.info(json.dumps(vars(args), indent=2))

    # Download data from GCS if needed
    if args.gcs_bucket:
        logger.info(f"Downloading data from GCS bucket: {args.gcs_bucket}")
        download_from_gcs(args.gcs_bucket, "data/processed/train.csv", args.train_sessions)
        download_from_gcs(args.gcs_bucket, "data/processed/val.csv", args.val_sessions)
        download_from_gcs(args.gcs_bucket, "data/processed/graph_edges.csv", args.graph_edges)
        # Download split_info.json to the same directory as train_sessions
        split_info_local = Path(args.train_sessions).parent / "split_info.json"
        download_from_gcs(args.gcs_bucket, "data/processed/split_info.json", str(split_info_local))

    # Load data info
    split_info_path = Path(args.train_sessions).parent / "split_info.json"
    with open(split_info_path) as f:
        split_info = json.load(f)

    num_items = split_info["num_items"]
    logger.info(f"Number of items: {num_items}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = create_dataloader(
        sessions_path=args.train_sessions,
        graph_edges_path=args.graph_edges,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        max_session_length=args.max_session_length,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = create_dataloader(
        sessions_path=args.val_sessions,
        graph_edges_path=args.graph_edges,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        max_session_length=args.max_session_length,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create model
    logger.info(f"Creating {args.model} model...")
    if args.model == "graphsage":
        model = create_graphsage(
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            readout_type=args.readout_type,
        )
    elif args.model == "gat":
        model = create_gat(
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            readout_type=args.readout_type,
        )
    elif args.model == "graph_transformer":
        model = create_graph_transformer(
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            readout_type=args.readout_type,
            use_laplacian_pe=True,
        )

        # Precompute Laplacian PE for the full graph
        logger.info("Precomputing Laplacian PE for the full graph...")
        graph_df = pd.read_csv(args.graph_edges)
        edge_index = torch.tensor(
            [graph_df["item_i"].values, graph_df["item_j"].values],
            dtype=torch.long,
        )
        graph_data = Data(edge_index=edge_index, num_nodes=num_items)
        model.laplacian_pe.precompute(graph_data)
        logger.info("Laplacian PE precomputed successfully")
    elif args.model == "graph_transformer_optimized":
        model = create_graph_transformer_optimized(
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            readout_type=args.readout_type,
            use_laplacian_pe=True,
            use_ffn=False,  # Optimized: No FFN for 29x speedup
            ffn_expansion=2,  # If FFN enabled, use 2x instead of 4x
        )

        # Precompute Laplacian PE for the full graph
        logger.info("Precomputing Laplacian PE for the full graph...")
        graph_df = pd.read_csv(args.graph_edges)
        edge_index = torch.tensor(
            [graph_df["item_i"].values, graph_df["item_j"].values],
            dtype=torch.long,
        )
        graph_data = Data(edge_index=edge_index, num_nodes=num_items)
        model.laplacian_pe.precompute(graph_data)
        logger.info("Laplacian PE precomputed successfully")
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create trainer
    output_dir = Path(args.output_dir) / args.model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=args.device,
        output_dir=output_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        eval_every=args.eval_every,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Log final results
    logger.info("Training complete!")
    logger.info(f"Best validation recall@10: {trainer.best_val_metric:.4f}")

    # Upload outputs to GCS if needed
    if args.gcs_bucket:
        logger.info(f"Uploading outputs to GCS bucket: {args.gcs_bucket}")
        for file_path in output_dir.glob("*"):
            if file_path.is_file():
                dest_path = f"outputs/{args.model}/{file_path.name}"
                upload_to_gcs(args.gcs_bucket, str(file_path), dest_path)


if __name__ == "__main__":
    main()
