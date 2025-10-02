"""ETP-GT: Temporal & Path-Aware Graph Transformer for session-based recommendation."""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from etpgt.encodings.laplacian_pe import LaplacianPECached
from etpgt.encodings.path_encoding import PathBias
from etpgt.encodings.temporal_encoding import TemporalBias
from etpgt.model.base import BaseRecommendationModel, SessionReadout


class TemporalPathAttention(MessagePassing):
    """Temporal & Path-Aware Multi-Head Attention.

    Extends standard graph attention with:
    1. Temporal bias based on time deltas
    2. Path bias based on shortest path lengths
    3. Laplacian positional encoding

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension per head.
        num_heads: Number of attention heads.
        num_temporal_buckets: Number of temporal buckets (default: 7).
        num_path_buckets: Number of path buckets (default: 3).
        dropout: Dropout rate.
        concat: Whether to concatenate or average heads.
        beta: Whether to use gated residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        num_temporal_buckets: int = 7,
        num_path_buckets: int = 3,
        dropout: float = 0.1,
        concat: bool = True,
        beta: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        self.beta = beta

        # Linear transformations for Q, K, V
        self.lin_q = nn.Linear(in_channels, num_heads * out_channels, bias=False)
        self.lin_k = nn.Linear(in_channels, num_heads * out_channels, bias=False)
        self.lin_v = nn.Linear(in_channels, num_heads * out_channels, bias=False)

        # Temporal and path biases
        self.temporal_bias = TemporalBias(num_buckets=num_temporal_buckets, num_heads=num_heads)
        self.path_bias = PathBias(num_buckets=num_path_buckets, num_heads=num_heads)

        # Gated residual connection
        if beta:
            self.lin_beta = nn.Linear(3 * in_channels, 1, bias=False)
        else:
            self.lin_beta = None

        # Output projection
        if concat:
            self.lin_out = nn.Linear(num_heads * out_channels, num_heads * out_channels)
        else:
            self.lin_out = nn.Linear(out_channels, out_channels)

        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        if self.lin_beta is not None:
            nn.init.xavier_uniform_(self.lin_beta.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time_delta_ms: Optional[torch.Tensor] = None,
        path_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_channels].
            edge_index: Edge indices [2, num_edges].
            time_delta_ms: Time deltas in milliseconds [num_edges] (optional).
            path_length: Shortest path lengths [num_edges] (optional).

        Returns:
            Updated node features [num_nodes, out_channels * num_heads] if concat,
                                  [num_nodes, out_channels] otherwise.
        """
        # Compute Q, K, V
        q = self.lin_q(x).view(-1, self.num_heads, self.out_channels)
        k = self.lin_k(x).view(-1, self.num_heads, self.out_channels)
        v = self.lin_v(x).view(-1, self.num_heads, self.out_channels)

        # Message passing
        out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            time_delta_ms=time_delta_ms,
            path_length=path_length,
        )

        # Gated residual connection
        if self.lin_beta is not None:
            beta = self.lin_beta(torch.cat([x, out, out - x], dim=-1))
            beta = torch.sigmoid(beta)
            out = beta * out + (1 - beta) * x

        # Output projection
        out = self.lin_out(out)
        out = self.dropout_layer(out)

        return out

    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
        time_delta_ms: Optional[torch.Tensor] = None,
        path_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute messages with temporal and path-aware attention.

        Args:
            q_i: Query vectors [num_edges, num_heads, out_channels].
            k_j: Key vectors [num_edges, num_heads, out_channels].
            v_j: Value vectors [num_edges, num_heads, out_channels].
            index: Target node indices [num_edges].
            ptr: Pointer for CSR format (optional).
            size_i: Number of target nodes (optional).
            time_delta_ms: Time deltas [num_edges] (optional).
            path_length: Path lengths [num_edges] (optional).

        Returns:
            Attention-weighted messages [num_edges, num_heads, out_channels].
        """
        # Compute attention scores (dot product)
        alpha = (q_i * k_j).sum(dim=-1) / (self.out_channels**0.5)  # [num_edges, num_heads]

        # Add temporal bias
        if time_delta_ms is not None:
            # Compute temporal bias for each edge
            # time_delta_ms: [num_edges]
            # We need to convert to [1, 1, num_edges] for broadcasting
            temporal_bias = self.temporal_bias(time_delta_ms.unsqueeze(0).unsqueeze(0))
            # temporal_bias: [1, num_heads, 1, num_edges]
            # Squeeze to [num_heads, num_edges] and transpose
            temporal_bias = temporal_bias.squeeze(0).squeeze(1).t()  # [num_edges, num_heads]
            alpha = alpha + temporal_bias

        # Add path bias
        if path_length is not None:
            # Similar to temporal bias
            path_bias = self.path_bias(path_length.unsqueeze(0).unsqueeze(0))
            path_bias = path_bias.squeeze(0).squeeze(1).t()  # [num_edges, num_heads]
            alpha = alpha + path_bias

        # Softmax over neighbors
        alpha = softmax(alpha, index, ptr, size_i)  # [num_edges, num_heads]
        alpha = self.dropout_layer(alpha)

        # Weighted sum of values
        out = v_j * alpha.unsqueeze(-1)  # [num_edges, num_heads, out_channels]

        return out

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Update node features after aggregation.

        Args:
            aggr_out: Aggregated messages [num_nodes, num_heads, out_channels].

        Returns:
            Updated features [num_nodes, num_heads * out_channels] if concat,
                            [num_nodes, out_channels] otherwise.
        """
        if self.concat:
            # Concatenate heads
            return aggr_out.view(-1, self.num_heads * self.out_channels)
        else:
            # Average heads
            return aggr_out.mean(dim=1)


class ETPGT(BaseRecommendationModel):
    """ETP-GT: Temporal & Path-Aware Graph Transformer.

    Args:
        num_items: Number of items in the catalog.
        embedding_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_temporal_buckets: Number of temporal buckets (default: 7).
        num_path_buckets: Number of path buckets (default: 3).
        dropout: Dropout rate.
        readout_type: Session readout type ('mean', 'max', 'last', 'attention').
        use_laplacian_pe: Whether to use Laplacian positional encoding.
        laplacian_k: Number of Laplacian eigenvectors.
        use_cls_token: Whether to use a CLS token for session representation.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        num_temporal_buckets: int = 7,
        num_path_buckets: int = 3,
        dropout: float = 0.1,
        readout_type: str = "mean",
        use_laplacian_pe: bool = True,
        laplacian_k: int = 16,
        use_cls_token: bool = False,
    ):
        super().__init__(num_items, embedding_dim, hidden_dim, num_layers, dropout)

        self.num_heads = num_heads
        self.num_temporal_buckets = num_temporal_buckets
        self.num_path_buckets = num_path_buckets
        self.readout_type = readout_type
        self.use_laplacian_pe = use_laplacian_pe
        self.laplacian_k = laplacian_k
        self.use_cls_token = use_cls_token

        # Laplacian PE
        if use_laplacian_pe:
            self.laplacian_pe = LaplacianPECached(k=laplacian_k, embedding_dim=embedding_dim)

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, embedding_dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # Temporal & Path-Aware Attention layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()

        # First layer
        self.convs.append(
            TemporalPathAttention(
                embedding_dim,
                hidden_dim // num_heads,
                num_heads=num_heads,
                num_temporal_buckets=num_temporal_buckets,
                num_path_buckets=num_path_buckets,
                dropout=dropout,
                concat=True,
                beta=True,
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.ffns.append(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
        )

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                TemporalPathAttention(
                    hidden_dim,
                    hidden_dim // num_heads,
                    num_heads=num_heads,
                    num_temporal_buckets=num_temporal_buckets,
                    num_path_buckets=num_path_buckets,
                    dropout=dropout,
                    concat=True,
                    beta=True,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
            )

        # Session readout
        self.readout = SessionReadout(hidden_dim, readout_type)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, batch):
        """Forward pass.

        Args:
            batch: PyG Batch object with:
                - x: Node features (item indices) [num_nodes]
                - edge_index: Edge index [2, num_edges]
                - batch: Batch indices [num_nodes]
                - time_delta_ms (optional): Time deltas [num_edges]
                - path_length (optional): Path lengths [num_edges]
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

        # Add CLS token if enabled
        if self.use_cls_token:
            # Add CLS token to each graph in the batch
            # batch_size = batch.batch.max().item() + 1
            # cls_tokens = self.cls_token.expand(batch_size, -1)  # [batch_size, embedding_dim]

            # Concatenate CLS tokens with node features
            # This requires updating edge_index and batch indices
            # For simplicity, we'll add CLS tokens after message passing
            # TODO: Implement CLS token integration
            pass

        # Get temporal and path information
        time_delta_ms = batch.time_delta_ms if hasattr(batch, "time_delta_ms") else None
        path_length = batch.path_length if hasattr(batch, "path_length") else None

        # Apply Temporal & Path-Aware Attention layers
        for conv, bn, ffn in zip(self.convs, self.batch_norms, self.ffns):
            # Self-attention with temporal and path biases
            x_residual = x
            x = conv(x, edge_index, time_delta_ms, path_length)
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


def create_etpgt(
    num_items: int,
    embedding_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_heads: int = 4,
    num_temporal_buckets: int = 7,
    num_path_buckets: int = 3,
    dropout: float = 0.1,
    readout_type: str = "mean",
    use_laplacian_pe: bool = True,
    laplacian_k: int = 16,
    use_cls_token: bool = False,
) -> ETPGT:
    """Factory function to create ETP-GT model.

    Args:
        num_items: Number of items in the catalog.
        embedding_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_temporal_buckets: Number of temporal buckets.
        num_path_buckets: Number of path buckets.
        dropout: Dropout rate.
        readout_type: Session readout type.
        use_laplacian_pe: Whether to use Laplacian PE.
        laplacian_k: Number of Laplacian eigenvectors.
        use_cls_token: Whether to use CLS token.

    Returns:
        ETP-GT model.
    """
    return ETPGT(
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_temporal_buckets=num_temporal_buckets,
        num_path_buckets=num_path_buckets,
        dropout=dropout,
        readout_type=readout_type,
        use_laplacian_pe=use_laplacian_pe,
        laplacian_k=laplacian_k,
        use_cls_token=use_cls_token,
    )
