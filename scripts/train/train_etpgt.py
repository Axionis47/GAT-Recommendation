#!/usr/bin/env python3
"""Training script for ETP-GT model."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from google.cloud import storage
from torch_geometric.data import Data

from etpgt.model.etpgt import create_etpgt
from etpgt.train.dataloader import create_dataloader
from etpgt.train.losses import create_loss_function
from etpgt.train.trainer import Trainer
from etpgt.utils.logging import get_logger
from etpgt.utils.seed import set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ETP-GT model")

    # Data arguments
    parser.add_argument("--train-sessions", type=str, default="data/processed/train.csv")
    parser.add_argument("--val-sessions", type=str, default="data/processed/val.csv")
    parser.add_argument("--graph-edges", type=str, default="data/processed/graph_edges.csv")
    parser.add_argument("--split-info", type=str, default="data/processed/split_info.json")

    # Model arguments
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-temporal-buckets", type=int, default=7)
    parser.add_argument("--num-path-buckets", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--readout-type", type=str, default="mean", choices=["mean", "max", "last", "attention"]
    )
    parser.add_argument("--use-laplacian-pe", action="store_true", default=True)
    parser.add_argument("--laplacian-k", type=int, default=16)
    parser.add_argument("--use-cls-token", action="store_true", default=False)

    # Loss arguments
    parser.add_argument(
        "--loss-type",
        type=str,
        default="dual",
        choices=["bpr", "listwise", "dual", "sampled_softmax"],
    )
    parser.add_argument(
        "--loss-alpha", type=float, default=0.7, help="Weight for listwise loss in dual loss"
    )
    parser.add_argument(
        "--loss-temperature", type=float, default=1.0, help="Temperature for softmax-based losses"
    )

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-negatives", type=int, default=5)
    parser.add_argument("--max-session-length", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--k-values", type=int, nargs="+", default=[10, 20])

    # System arguments
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/etpgt")

    # GCS arguments
    parser.add_argument(
        "--gcs-bucket", type=str, default=None, help="GCS bucket for data and outputs"
    )

    return parser.parse_args()


def download_from_gcs(bucket_name: str, source_path: str, dest_path: str):
    """Download file from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_path)

    # Create parent directory
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

    blob.download_to_filename(dest_path)
    logger = get_logger(__name__)
    logger.info(f"Downloaded gs://{bucket_name}/{source_path} to {dest_path}")


def upload_to_gcs(bucket_name: str, source_path: str, dest_path: str):
    """Upload file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_path)

    blob.upload_from_filename(source_path)
    logger = get_logger(__name__)
    logger.info(f"Uploaded {source_path} to gs://{bucket_name}/{dest_path}")


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    logger = get_logger(__name__)

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download data from GCS if needed
    if args.gcs_bucket:
        logger.info(f"Downloading data from GCS bucket: {args.gcs_bucket}")
        download_from_gcs(args.gcs_bucket, "data/processed/train.csv", args.train_sessions)
        download_from_gcs(args.gcs_bucket, "data/processed/val.csv", args.val_sessions)
        download_from_gcs(args.gcs_bucket, "data/processed/graph_edges.csv", args.graph_edges)
        download_from_gcs(args.gcs_bucket, "data/processed/split_info.json", args.split_info)

    # Load split info
    with open(args.split_info) as f:
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
        num_workers=args.num_workers,
        shuffle=True,
    )

    val_loader = create_dataloader(
        sessions_path=args.val_sessions,
        graph_edges_path=args.graph_edges,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        max_session_length=args.max_session_length,
        num_workers=args.num_workers,
        shuffle=False,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    logger.info("Creating ETP-GT model...")
    model = create_etpgt(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_temporal_buckets=args.num_temporal_buckets,
        num_path_buckets=args.num_path_buckets,
        dropout=args.dropout,
        readout_type=args.readout_type,
        use_laplacian_pe=args.use_laplacian_pe,
        laplacian_k=args.laplacian_k,
        use_cls_token=args.use_cls_token,
    )

    # Precompute Laplacian PE if enabled
    if args.use_laplacian_pe:
        logger.info("Precomputing Laplacian PE for the full graph...")
        graph_df = pd.read_csv(args.graph_edges)
        edge_index = torch.tensor(
            [graph_df["item_i"].values, graph_df["item_j"].values],
            dtype=torch.long,
        )
        graph_data = Data(edge_index=edge_index, num_nodes=num_items)
        model.laplacian_pe.precompute(graph_data)
        logger.info("Laplacian PE precomputed successfully")

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss function
    loss_fn = create_loss_function(
        loss_type=args.loss_type,
        alpha=args.loss_alpha,
        temperature=args.loss_temperature,
    )

    logger.info(f"Loss function: {args.loss_type}")
    if args.loss_type == "dual":
        logger.info(f"  Alpha (listwise weight): {args.loss_alpha}")
        logger.info(f"  Temperature: {args.loss_temperature}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=args.device,
        output_dir=str(output_dir),
        max_epochs=args.max_epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        k_values=args.k_values,
        loss_fn=loss_fn,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Log final results
    logger.info("Training complete!")
    logger.info(f"Best validation recall@10: {trainer.best_val_metric:.4f}")

    # Upload outputs to GCS
    if args.gcs_bucket:
        logger.info(f"Uploading outputs to GCS bucket: {args.gcs_bucket}")

        # Upload best checkpoint
        checkpoint_path = output_dir / "best_model.pt"
        if checkpoint_path.exists():
            upload_to_gcs(
                args.gcs_bucket,
                str(checkpoint_path),
                "models/etpgt/best_model.pt",
            )

        # Upload training history
        history_path = output_dir / "history.json"
        if history_path.exists():
            upload_to_gcs(
                args.gcs_bucket,
                str(history_path),
                "models/etpgt/history.json",
            )

    logger.info("Done!")


if __name__ == "__main__":
    main()
