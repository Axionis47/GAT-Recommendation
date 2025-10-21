"""GAT (Graph Attention Network) baseline for session-based recommendation."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from etpgt.model.base import BaseRecommendationModel, SessionReadout


class GAT(BaseRecommendationModel):
    """GAT model for session-based recommendation.

    Args:
        num_items: Number of items in the catalog.
        embedding_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of GAT layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        readout_type: Session readout type ('mean', 'max', 'last', 'attention').
        concat_heads: Whether to concatenate attention heads (vs. average).
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        readout_type: str = "mean",
        concat_heads: bool = False,
    ):
        super().__init__(num_items, embedding_dim, hidden_dim, num_layers, dropout)

        self.num_heads = num_heads
        self.readout_type = readout_type
        self.concat_heads = concat_heads

        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        if concat_heads:
            # Concatenate heads: output dim = hidden_dim * num_heads
            self.convs.append(
                GATConv(
                    embedding_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
            current_dim = hidden_dim * num_heads
        else:
            # Average heads: output dim = hidden_dim
            self.convs.append(
                GATConv(
                    embedding_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim

        # Hidden layers
        for _ in range(num_layers - 2):
            if concat_heads:
                self.convs.append(
                    GATConv(
                        current_dim,
                        hidden_dim,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                    )
                )
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
                current_dim = hidden_dim * num_heads
            else:
                self.convs.append(
                    GATConv(
                        current_dim,
                        hidden_dim,
                        heads=num_heads,
                        dropout=dropout,
                        concat=False,
                    )
                )
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                current_dim = hidden_dim

        # Last layer (always average heads for final output)
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    current_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Session readout
        self.readout = SessionReadout(hidden_dim, readout_type)

    def forward(self, batch):
        """Forward pass.

        Args:
            batch: PyG Batch object with:
                - x: Node features (item indices) [num_nodes]
                - edge_index: Edge index [2, num_edges]
                - batch: Batch indices [num_nodes]

        Returns:
            Session embeddings [batch_size, hidden_dim].
        """
        # Get item embeddings
        x = self.item_embedding(batch.x)  # [num_nodes, embedding_dim]
        edge_index = batch.edge_index

        # Apply GAT layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:  # No activation on last layer
                x = torch.relu(x)
                x = self.dropout_layer(x)

        # Session readout
        session_embeddings = self.readout(x, batch.batch)

        return session_embeddings


def create_gat(
    num_items: int,
    embedding_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_heads: int = 4,
    dropout: float = 0.1,
    readout_type: str = "mean",
    concat_heads: bool = False,
) -> GAT:
    """Factory function to create GAT model.

    Args:
        num_items: Number of items in the catalog.
        embedding_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of GAT layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        readout_type: Session readout type.
        concat_heads: Whether to concatenate attention heads.

    Returns:
        GAT model.
    """
    return GAT(
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        readout_type=readout_type,
        concat_heads=concat_heads,
    )
