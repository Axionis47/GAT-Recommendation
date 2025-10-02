#!/usr/bin/env python3
"""Local training script for GraphTransformerOptimized."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch_geometric.data import Data

from etpgt.model import create_graph_transformer_optimized
from etpgt.train.dataloader import create_dataloader
from etpgt.train.losses import create_loss_function
from etpgt.train.trainer import Trainer
from etpgt.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sessions", type=str, default="data/processed/train.csv")
    parser.add_argument("--val-sessions", type=str, default="data/processed/val.csv")
    parser.add_argument("--graph-edges", type=str, default="data/processed/graph_edges.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/graph_transformer_optimized_local")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 80)
    print("GRAPH TRANSFORMER OPTIMIZED - LOCAL TRAINING")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Patience: {args.patience}")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    train_sessions = pd.read_csv(args.train_sessions)
    val_sessions = pd.read_csv(args.val_sessions)
    graph_edges = pd.read_csv(args.graph_edges)
    
    print(f"Train sessions: {len(train_sessions):,}")
    print(f"Val sessions: {len(val_sessions):,}")
    print(f"Graph edges: {len(graph_edges):,}")
    
    # Get number of items
    num_items = max(
        train_sessions["itemid"].max(),
        val_sessions["itemid"].max(),
        graph_edges["item_i"].max(),
        graph_edges["item_j"].max()
    ) + 1
    print(f"Number of items: {num_items:,}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(
        sessions_path=args.train_sessions,
        graph_edges_path=args.graph_edges,
        batch_size=args.batch_size,
        num_negatives=5,
        max_session_length=50,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader(
        sessions_path=args.val_sessions,
        graph_edges_path=args.graph_edges,
        batch_size=args.batch_size,
        num_negatives=5,
        max_session_length=50,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_graph_transformer_optimized(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        readout_type="mean",
        use_laplacian_pe=True,
        use_ffn=False,  # Optimized: No FFN
        ffn_expansion=2,
    )

    # Precompute Laplacian PE
    print("Precomputing Laplacian PE...")
    # Create graph data for precomputation
    edge_index = torch.tensor(
        [graph_edges["item_i"].values, graph_edges["item_j"].values],
        dtype=torch.long,
    )
    graph_data = Data(edge_index=edge_index, num_nodes=num_items)
    model.laplacian_pe.precompute(graph_data)

    device = torch.device(args.device)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer and loss function
    print("\nCreating optimizer and loss function...")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = create_loss_function(loss_type="listwise", temperature=1.0)

    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir,
        max_epochs=args.max_epochs,
        patience=args.patience,
        eval_every=1,
        loss_fn=loss_fn,
    )

    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

