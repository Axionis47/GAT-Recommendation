"""Base model class for session-based recommendation."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseRecommendationModel(nn.Module, ABC):
    """Base class for session-based recommendation models.

    Args:
        num_items: Number of items in the catalog.
        embedding_dim: Embedding dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Item embeddings
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])  # Skip padding

    @abstractmethod
    def forward(self, batch):
        """Forward pass.

        Args:
            batch: Batch data.

        Returns:
            Session embeddings [batch_size, hidden_dim].
        """
        pass

    def get_item_embeddings(self) -> torch.Tensor:
        """Get all item embeddings.

        Returns:
            Item embeddings [num_items, embedding_dim].
        """
        return self.item_embedding.weight

    def predict(self, session_embeddings: torch.Tensor, k: int = 20) -> torch.Tensor:
        """Predict top-k items for sessions.

        Args:
            session_embeddings: Session embeddings [batch_size, hidden_dim].
            k: Number of items to predict.

        Returns:
            Top-k item indices [batch_size, k].
        """
        # Get all item embeddings
        item_embeddings = self.get_item_embeddings()  # [num_items, embedding_dim]

        # Compute scores (dot product)
        scores = torch.matmul(session_embeddings, item_embeddings.t())  # [batch_size, num_items]

        # Get top-k items
        _, top_k_items = torch.topk(scores, k=k, dim=1)

        return top_k_items

    def compute_loss(
        self,
        session_embeddings: torch.Tensor,
        target_items: torch.Tensor,
        negative_items: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            session_embeddings: Session embeddings [batch_size, hidden_dim].
            target_items: Target item indices [batch_size].
            negative_items: Negative item indices [batch_size, num_negatives].

        Returns:
            Loss scalar.
        """
        # Get embeddings
        target_embeddings = self.item_embedding(target_items)  # [batch_size, embedding_dim]
        negative_embeddings = self.item_embedding(
            negative_items
        )  # [batch_size, num_negatives, embedding_dim]

        # Compute positive scores
        pos_scores = (session_embeddings * target_embeddings).sum(dim=1)  # [batch_size]

        # Compute negative scores
        neg_scores = torch.bmm(negative_embeddings, session_embeddings.unsqueeze(2)).squeeze(
            2
        )  # [batch_size, num_negatives]

        # BPR loss: -log(sigmoid(pos - neg))
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8).mean()

        return loss


class SessionReadout(nn.Module):
    """Session readout layer.

    Aggregates node embeddings into session embedding.

    Args:
        hidden_dim: Hidden dimension.
        readout_type: Type of readout ('mean', 'max', 'last', 'attention').
    """

    def __init__(self, hidden_dim: int = 256, readout_type: str = "mean"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout_type = readout_type

        if readout_type == "attention":
            self.attention = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.attention.weight)
            nn.init.zeros_(self.attention.bias)

    def forward(self, node_embeddings: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """Aggregate node embeddings into session embeddings.

        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_dim].
            batch_indices: Batch indices for each node [num_nodes].

        Returns:
            Session embeddings [batch_size, hidden_dim].
        """
        batch_size = batch_indices.max().item() + 1

        if self.readout_type == "mean":
            # Mean pooling
            session_embeddings = torch.zeros(
                batch_size, self.hidden_dim, device=node_embeddings.device
            )
            for i in range(batch_size):
                mask = batch_indices == i
                session_embeddings[i] = node_embeddings[mask].mean(dim=0)

        elif self.readout_type == "max":
            # Max pooling
            session_embeddings = torch.zeros(
                batch_size, self.hidden_dim, device=node_embeddings.device
            )
            for i in range(batch_size):
                mask = batch_indices == i
                session_embeddings[i] = node_embeddings[mask].max(dim=0)[0]

        elif self.readout_type == "last":
            # Last item in session
            session_embeddings = torch.zeros(
                batch_size, self.hidden_dim, device=node_embeddings.device
            )
            for i in range(batch_size):
                mask = batch_indices == i
                session_embeddings[i] = node_embeddings[mask][-1]

        elif self.readout_type == "attention":
            # Attention-based pooling
            attention_scores = self.attention(node_embeddings).squeeze(-1)  # [num_nodes]
            attention_weights = torch.zeros(
                batch_size, node_embeddings.size(0), device=node_embeddings.device
            )

            for i in range(batch_size):
                mask = batch_indices == i
                attention_weights[i, mask] = torch.softmax(attention_scores[mask], dim=0)

            session_embeddings = torch.matmul(
                attention_weights, node_embeddings
            )  # [batch_size, hidden_dim]

        else:
            raise ValueError(f"Unknown readout type: {self.readout_type}")

        return session_embeddings
