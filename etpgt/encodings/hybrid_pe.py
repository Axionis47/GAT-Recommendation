"""Hybrid Positional Encoding combining temporal, path, and Laplacian encodings."""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from etpgt.encodings.laplacian_pe import LaplacianPECached
from etpgt.encodings.path_encoding import PathEncoding
from etpgt.encodings.temporal_encoding import TemporalEncoding


class HybridPE(nn.Module):
    """Hybrid Positional Encoding.

    Combines three types of positional encodings:
    1. Temporal encoding (time deltas)
    2. Path encoding (shortest path lengths)
    3. Laplacian encoding (graph structure)

    Args:
        embedding_dim: Embedding dimension.
        num_temporal_buckets: Number of temporal buckets (default: 7).
        num_path_buckets: Number of path buckets (default: 3).
        laplacian_k: Number of Laplacian eigenvectors (default: 16).
        laplacian_normalization: Laplacian normalization ('sym' or 'rw').
        combine_method: How to combine encodings ('add', 'concat', 'gated').
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_temporal_buckets: int = 7,
        num_path_buckets: int = 3,
        laplacian_k: int = 16,
        laplacian_normalization: str = "sym",
        combine_method: str = "add",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.combine_method = combine_method

        # Temporal encoding
        self.temporal_encoding = TemporalEncoding(
            num_buckets=num_temporal_buckets,
            embedding_dim=embedding_dim,
        )

        # Path encoding
        self.path_encoding = PathEncoding(
            num_buckets=num_path_buckets,
            embedding_dim=embedding_dim,
        )

        # Laplacian encoding (cached for efficiency)
        self.laplacian_encoding = LaplacianPECached(
            k=laplacian_k,
            embedding_dim=embedding_dim,
            normalization=laplacian_normalization,
        )

        # Combination layer (if needed)
        if combine_method == "concat":
            # Concatenate all three encodings
            self.combine_projection = nn.Linear(embedding_dim * 3, embedding_dim)
            nn.init.xavier_uniform_(self.combine_projection.weight)
            nn.init.zeros_(self.combine_projection.bias)
        elif combine_method == "gated":
            # Gated combination
            self.gate_temporal = nn.Linear(embedding_dim, 1)
            self.gate_path = nn.Linear(embedding_dim, 1)
            self.gate_laplacian = nn.Linear(embedding_dim, 1)
            nn.init.xavier_uniform_(self.gate_temporal.weight)
            nn.init.xavier_uniform_(self.gate_path.weight)
            nn.init.xavier_uniform_(self.gate_laplacian.weight)
            nn.init.zeros_(self.gate_temporal.bias)
            nn.init.zeros_(self.gate_path.bias)
            nn.init.zeros_(self.gate_laplacian.bias)

    def precompute_laplacian(self, data: Data) -> None:
        """Precompute Laplacian PE for the graph.

        Args:
            data: PyG Data object with edge_index and num_nodes.
        """
        self.laplacian_encoding.precompute(data)

    def forward(
        self,
        node_indices: torch.Tensor,
        time_delta_ms: torch.Tensor,
        path_length: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hybrid positional encoding.

        Args:
            node_indices: Node indices [batch_size, seq_len].
            time_delta_ms: Time deltas in milliseconds [batch_size, seq_len].
            path_length: Shortest path lengths [batch_size, seq_len].

        Returns:
            Hybrid positional encodings [batch_size, seq_len, embedding_dim].
        """
        # Compute individual encodings
        temporal_pe = self.temporal_encoding(time_delta_ms)  # [batch_size, seq_len, embedding_dim]
        path_pe = self.path_encoding(path_length)  # [batch_size, seq_len, embedding_dim]
        laplacian_pe = self.laplacian_encoding(node_indices)  # [batch_size, seq_len, embedding_dim]

        # Combine encodings
        if self.combine_method == "add":
            # Simple addition
            hybrid_pe = temporal_pe + path_pe + laplacian_pe
        elif self.combine_method == "concat":
            # Concatenate and project
            hybrid_pe = torch.cat([temporal_pe, path_pe, laplacian_pe], dim=-1)
            hybrid_pe = self.combine_projection(hybrid_pe)
        elif self.combine_method == "gated":
            # Gated combination
            gate_t = torch.sigmoid(self.gate_temporal(temporal_pe))
            gate_p = torch.sigmoid(self.gate_path(path_pe))
            gate_l = torch.sigmoid(self.gate_laplacian(laplacian_pe))

            # Normalize gates
            gate_sum = gate_t + gate_p + gate_l
            gate_t = gate_t / gate_sum
            gate_p = gate_p / gate_sum
            gate_l = gate_l / gate_sum

            hybrid_pe = gate_t * temporal_pe + gate_p * path_pe + gate_l * laplacian_pe
        else:
            raise ValueError(f"Unknown combine_method: {self.combine_method}")

        return hybrid_pe


class HybridBias(nn.Module):
    """Hybrid attention bias combining temporal and path biases.

    Args:
        num_temporal_buckets: Number of temporal buckets (default: 7).
        num_path_buckets: Number of path buckets (default: 3).
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        num_temporal_buckets: int = 7,
        num_path_buckets: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Temporal bias
        self.temporal_bias = nn.Parameter(torch.zeros(num_heads, num_temporal_buckets))

        # Path bias
        self.path_bias = nn.Parameter(torch.zeros(num_heads, num_path_buckets))

        # Initialize
        nn.init.normal_(self.temporal_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.path_bias, mean=0.0, std=0.02)

    def forward(
        self,
        time_delta_ms: torch.Tensor,
        path_length: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hybrid attention bias.

        Args:
            time_delta_ms: Time deltas in milliseconds [batch_size, seq_len, seq_len].
            path_length: Shortest path lengths [batch_size, seq_len, seq_len].

        Returns:
            Attention bias [batch_size, num_heads, seq_len, seq_len].
        """
        from etpgt.encodings.path_encoding import path_length_to_bucket
        from etpgt.encodings.temporal_encoding import time_delta_to_bucket

        # Convert to buckets
        temporal_buckets = time_delta_to_bucket(time_delta_ms)
        path_buckets = path_length_to_bucket(path_length)

        batch_size, seq_len, _ = temporal_buckets.shape

        # Get temporal bias
        temporal_bias_expanded = self.temporal_bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        temporal_buckets_expanded = temporal_buckets.unsqueeze(1)
        temporal_attention_bias = torch.gather(
            temporal_bias_expanded.expand(
                batch_size, self.num_heads, seq_len, seq_len, self.temporal_bias.size(1)
            ),
            dim=-1,
            index=temporal_buckets_expanded.unsqueeze(1).expand(
                batch_size, self.num_heads, seq_len, seq_len, 1
            ),
        ).squeeze(-1)

        # Get path bias
        path_bias_expanded = self.path_bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        path_buckets_expanded = path_buckets.unsqueeze(1)
        path_attention_bias = torch.gather(
            path_bias_expanded.expand(
                batch_size, self.num_heads, seq_len, seq_len, self.path_bias.size(1)
            ),
            dim=-1,
            index=path_buckets_expanded.unsqueeze(1).expand(
                batch_size, self.num_heads, seq_len, seq_len, 1
            ),
        ).squeeze(-1)

        # Combine biases
        hybrid_bias = temporal_attention_bias + path_attention_bias

        return hybrid_bias
