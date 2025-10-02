"""GraphSAGE baseline for session-based recommendation."""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from etpgt.model.base import BaseRecommendationModel, SessionReadout


class GraphSAGE(BaseRecommendationModel):
    """GraphSAGE model for session-based recommendation.

    Args:
        num_items: Number of items in the catalog.
        embedding_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of GraphSAGE layers.
        dropout: Dropout rate.
        readout_type: Session readout type ('mean', 'max', 'last', 'attention').
        aggregator: GraphSAGE aggregator ('mean', 'max', 'lstm').
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        readout_type: str = "mean",
        aggregator: str = "mean",
    ):
        super().__init__(num_items, embedding_dim, hidden_dim, num_layers, dropout)

        self.aggregator = aggregator
        self.readout_type = readout_type

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(embedding_dim, hidden_dim, aggr=aggregator))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
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

        # Apply GraphSAGE layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout_layer(x)

        # Session readout
        session_embeddings = self.readout(x, batch.batch)

        return session_embeddings


def create_graphsage(
    num_items: int,
    embedding_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.1,
    readout_type: str = "mean",
    aggregator: str = "mean",
) -> GraphSAGE:
    """Factory function to create GraphSAGE model.

    Args:
        num_items: Number of items in the catalog.
        embedding_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of GraphSAGE layers.
        dropout: Dropout rate.
        readout_type: Session readout type.
        aggregator: GraphSAGE aggregator.

    Returns:
        GraphSAGE model.
    """
    return GraphSAGE(
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        readout_type=readout_type,
        aggregator=aggregator,
    )
