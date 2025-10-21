"""Temporal encoding for time-aware graph transformers.

Encodes time deltas into discrete buckets:
- [0-1m, 1-5m, 5-30m, 30-120m, 2-24h, 1-7d, 7d+]
"""

import torch
import torch.nn as nn

# Time delta buckets (in milliseconds)
TIME_BUCKETS_MS = [
    60 * 1000,  # 1 minute
    5 * 60 * 1000,  # 5 minutes
    30 * 60 * 1000,  # 30 minutes
    120 * 60 * 1000,  # 2 hours
    24 * 60 * 60 * 1000,  # 24 hours
    7 * 24 * 60 * 60 * 1000,  # 7 days
]


def time_delta_to_bucket(time_delta_ms: torch.Tensor) -> torch.Tensor:
    """Convert time deltas to bucket indices.

    Buckets:
    0: [0-1m)
    1: [1m-5m)
    2: [5m-30m)
    3: [30m-2h)
    4: [2h-24h)
    5: [24h-7d)
    6: [7d+)

    Args:
        time_delta_ms: Time deltas in milliseconds [batch_size] or [batch_size, seq_len].

    Returns:
        Bucket indices [batch_size] or [batch_size, seq_len].
    """
    buckets = torch.zeros_like(time_delta_ms, dtype=torch.long)

    for i, threshold in enumerate(TIME_BUCKETS_MS):
        buckets[time_delta_ms >= threshold] = i + 1

    return buckets


class TemporalEncoding(nn.Module):
    """Temporal encoding layer.

    Encodes time deltas into learnable embeddings via bucketing.

    Args:
        num_buckets: Number of time buckets (default: 7).
        embedding_dim: Embedding dimension.
    """

    def __init__(self, num_buckets: int = 7, embedding_dim: int = 256):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim

        # Learnable bucket embeddings
        self.bucket_embedding = nn.Embedding(num_buckets, embedding_dim)

        # Initialize with small values
        nn.init.normal_(self.bucket_embedding.weight, mean=0.0, std=0.02)

    def forward(self, time_delta_ms: torch.Tensor) -> torch.Tensor:
        """Encode time deltas.

        Args:
            time_delta_ms: Time deltas in milliseconds [batch_size] or [batch_size, seq_len].

        Returns:
            Temporal embeddings [batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim].
        """
        # Convert to buckets
        buckets = time_delta_to_bucket(time_delta_ms)

        # Clamp to valid range
        buckets = torch.clamp(buckets, 0, self.num_buckets - 1)

        # Embed
        embeddings = self.bucket_embedding(buckets)

        return embeddings


class TemporalBias(nn.Module):
    """Temporal bias for attention mechanism.

    Adds learnable bias to attention scores based on time delta buckets.

    Args:
        num_buckets: Number of time buckets (default: 7).
        num_heads: Number of attention heads.
    """

    def __init__(self, num_buckets: int = 7, num_heads: int = 4):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads

        # Learnable bias per bucket per head
        self.bias = nn.Parameter(torch.zeros(num_heads, num_buckets))

        # Initialize with small values
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, time_delta_ms: torch.Tensor) -> torch.Tensor:
        """Compute temporal bias for attention.

        Args:
            time_delta_ms: Time deltas in milliseconds [batch_size, seq_len, seq_len].

        Returns:
            Attention bias [batch_size, num_heads, seq_len, seq_len].
        """
        # Convert to buckets
        buckets = time_delta_to_bucket(time_delta_ms)  # [batch_size, seq_len, seq_len]

        # Clamp to valid range
        buckets = torch.clamp(buckets, 0, self.num_buckets - 1)

        # Get bias for each bucket
        # bias: [num_heads, num_buckets]
        # buckets: [batch_size, seq_len, seq_len]
        # Output: [batch_size, num_heads, seq_len, seq_len]

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
