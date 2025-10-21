"""Graph Transformer with Laplacian PE baseline for session-based recommendation."""

import torch.nn as nn
from torch_geometric.nn import TransformerConv

from etpgt.encodings.laplacian_pe import LaplacianPECached
from etpgt.model.base import BaseRecommendationModel, SessionReadout


class GraphTransformer(BaseRecommendationModel):
    """Graph Transformer with Laplacian PE for session-based recommendation.

    Args:
        num_items: Number of items in the catalog.
        embedding_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        readout_type: Session readout type ('mean', 'max', 'last', 'attention').
        use_laplacian_pe: Whether to use Laplacian positional encoding.
        laplacian_k: Number of Laplacian eigenvectors.
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
        use_laplacian_pe: bool = True,
        laplacian_k: int = 16,
    ):
        super().__init__(num_items, embedding_dim, hidden_dim, num_layers, dropout)

        self.num_heads = num_heads
        self.readout_type = readout_type
        self.use_laplacian_pe = use_laplacian_pe
        self.laplacian_k = laplacian_k

        # Laplacian PE
        if use_laplacian_pe:
            self.laplacian_pe = LaplacianPECached(k=laplacian_k, embedding_dim=embedding_dim)

        # Transformer layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()

        # First layer
        self.convs.append(
            TransformerConv(
                embedding_dim,
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True,
                beta=True,  # Use gated residual connections
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.ffns.append(self._make_ffn(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                TransformerConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    beta=True,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.ffns.append(self._make_ffn(hidden_dim))

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Session readout
        self.readout = SessionReadout(hidden_dim, readout_type)

    def _make_ffn(self, hidden_dim: int) -> nn.Module:
        """Create feed-forward network.

        Args:
            hidden_dim: Hidden dimension.

        Returns:
            FFN module.
        """
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(self.dropout),
        )

    def forward(self, batch):
        """Forward pass.

        Args:
            batch: PyG Batch object with:
                - x: Node features (item indices) [num_nodes]
                - edge_index: Edge index [2, num_edges]
                - batch: Batch indices [num_nodes]
                - laplacian_pe (optional): Precomputed Laplacian PE [num_nodes, k]

        Returns:
            Session embeddings [batch_size, hidden_dim].
        """
        # Get item embeddings
        x = self.item_embedding(batch.x)  # [num_nodes, embedding_dim]
        edge_index = batch.edge_index

        # Add Laplacian PE
        if self.use_laplacian_pe:
            if hasattr(batch, "laplacian_pe") and batch.laplacian_pe is not None:
                # Use precomputed PE
                lap_pe = self.laplacian_pe.project(batch.laplacian_pe)
            else:
                # Compute on-the-fly using node indices from batch.x
                lap_pe = self.laplacian_pe(batch.x)

            x = x + lap_pe

        # Apply Transformer layers
        for conv, bn, ffn in zip(self.convs, self.batch_norms, self.ffns):
            # Self-attention
            x_residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = x + x_residual  # Residual connection
            x = self.dropout_layer(x)

            # Feed-forward
            x_residual = x
            x = ffn(x)
            x = x + x_residual  # Residual connection

        # Session readout
        session_embeddings = self.readout(x, batch.batch)

        return session_embeddings


def create_graph_transformer(
    num_items: int,
    embedding_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_heads: int = 4,
    dropout: float = 0.1,
    readout_type: str = "mean",
    use_laplacian_pe: bool = True,
    laplacian_k: int = 16,
) -> GraphTransformer:
    """Factory function to create Graph Transformer model.

    Args:
        num_items: Number of items in the catalog.
        embedding_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        readout_type: Session readout type.
        use_laplacian_pe: Whether to use Laplacian PE.
        laplacian_k: Number of Laplacian eigenvectors.

    Returns:
        GraphTransformer model.
    """
    return GraphTransformer(
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        readout_type=readout_type,
        use_laplacian_pe=use_laplacian_pe,
        laplacian_k=laplacian_k,
    )
