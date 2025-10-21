#!/usr/bin/env python3
"""Quick test script for all models."""

import torch
import pandas as pd
from torch_geometric.data import Data
from etpgt.model import create_gat, create_graphsage, create_graph_transformer
from etpgt.train.dataloader import create_dataloader

print("=== Testing all models ===\n")

# Create dataloader
print("Creating dataloader...")
train_loader = create_dataloader(
    sessions_path='data/processed/train.csv',
    graph_edges_path='data/processed/graph_edges.csv',
    batch_size=2,
    num_negatives=5,
    max_session_length=50,
    shuffle=False,
    num_workers=0,
)
batch = next(iter(train_loader))
print(f"✅ Dataloader created, batch shape: {batch.x.shape}\n")

# Test GAT
print("Testing GAT...")
gat_model = create_gat(
    num_items=466865,
    embedding_dim=32,
    hidden_dim=32,
    num_layers=1,
    num_heads=1,
    dropout=0.1,
    readout_type='mean',
)
gat_out = gat_model(batch)
print(f"✅ GAT works: {gat_out.shape}\n")

# Test GraphSAGE
print("Testing GraphSAGE...")
graphsage_model = create_graphsage(
    num_items=466865,
    embedding_dim=32,
    hidden_dim=32,
    num_layers=1,
    dropout=0.1,
    readout_type='mean',
)
graphsage_out = graphsage_model(batch)
print(f"✅ GraphSAGE works: {graphsage_out.shape}\n")

# Test GraphTransformer
print("Testing GraphTransformer...")
gt_model = create_graph_transformer(
    num_items=466865,
    embedding_dim=32,
    hidden_dim=32,
    num_layers=1,
    num_heads=1,
    dropout=0.1,
    readout_type='mean',
    use_laplacian_pe=True,
)

# Precompute Laplacian PE
print("Precomputing Laplacian PE...")
graph_df = pd.read_csv('data/processed/graph_edges.csv')
edge_index = torch.tensor(
    [graph_df['item_i'].values, graph_df['item_j'].values],
    dtype=torch.long,
)
graph_data = Data(edge_index=edge_index, num_nodes=466865)
gt_model.laplacian_pe.precompute(graph_data)
print("Laplacian PE precomputed")

gt_out = gt_model(batch)
print(f"✅ GraphTransformer works: {gt_out.shape}\n")

print("=== All models tested successfully! ===")

