"""Path encoding for graph transformers.

Encodes shortest path lengths into discrete buckets:
- {1, 2, 3+}
"""

import torch
import torch.nn as nn


def path_length_to_bucket(path_length: torch.Tensor) -> torch.Tensor:
    """Convert path lengths to bucket indices.

    Buckets:
    0: path length = 1
    1: path length = 2
    2: path length >= 3

    Args:
        path_length: Path lengths [batch_size] or [batch_size, seq_len, seq_len].

    Returns:
        Bucket indices [batch_size] or [batch_size, seq_len, seq_len].
    """
    buckets = torch.clamp(path_length - 1, min=0, max=2)
    return buckets


class PathEncoding(nn.Module):
    """Path encoding layer.

    Encodes shortest path lengths into learnable embeddings via bucketing.

    Args:
        num_buckets: Number of path buckets (default: 3 for {1, 2, 3+}).
        embedding_dim: Embedding dimension.
    """

    def __init__(self, num_buckets: int = 3, embedding_dim: int = 256):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim

        # Learnable bucket embeddings
        self.bucket_embedding = nn.Embedding(num_buckets, embedding_dim)

        # Initialize with small values
        nn.init.normal_(self.bucket_embedding.weight, mean=0.0, std=0.02)

    def forward(self, path_length: torch.Tensor) -> torch.Tensor:
        """Encode path lengths.

        Args:
            path_length: Path lengths [batch_size] or [batch_size, seq_len, seq_len].

        Returns:
            Path embeddings [batch_size, embedding_dim] or [batch_size, seq_len, seq_len, embedding_dim].
        """
        # Convert to buckets
        buckets = path_length_to_bucket(path_length)

        # Clamp to valid range
        buckets = torch.clamp(buckets, 0, self.num_buckets - 1)

        # Embed
        embeddings = self.bucket_embedding(buckets)

        return embeddings


class PathBias(nn.Module):
    """Path bias for attention mechanism.

    Adds learnable bias to attention scores based on path length buckets.

    Args:
        num_buckets: Number of path buckets (default: 3).
        num_heads: Number of attention heads.
    """

    def __init__(self, num_buckets: int = 3, num_heads: int = 4):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads

        # Learnable bias per bucket per head
        self.bias = nn.Parameter(torch.zeros(num_heads, num_buckets))

        # Initialize with small values
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, path_length: torch.Tensor) -> torch.Tensor:
        """Compute path bias for attention.

        Args:
            path_length: Path lengths [batch_size, seq_len, seq_len].

        Returns:
            Attention bias [batch_size, num_heads, seq_len, seq_len].
        """
        # Convert to buckets
        buckets = path_length_to_bucket(path_length)  # [batch_size, seq_len, seq_len]

        # Clamp to valid range
        buckets = torch.clamp(buckets, 0, self.num_buckets - 1)

        # Get bias for each bucket
        batch_size, seq_len, _ = buckets.shape

        # Expand bias to match buckets shape
        bias_expanded = (
            self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )  # [1, num_heads, 1, 1, num_buckets]
        buckets_expanded = buckets.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

        # Gather bias for each bucket
        attention_bias = torch.gather(
            bias_expanded.expand(batch_size, self.num_heads, seq_len, seq_len, self.num_buckets),
            dim=-1,
            index=buckets_expanded.unsqueeze(1).expand(
                batch_size, self.num_heads, seq_len, seq_len, 1
            ),
        ).squeeze(-1)

        return attention_bias
