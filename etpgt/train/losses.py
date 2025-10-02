"""Loss functions for session-based recommendation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking (BPR) loss.

    Contrastive loss that maximizes the margin between positive and negative items.

    Args:
        None
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        session_embeddings: torch.Tensor,
        target_items: torch.Tensor,
        negative_items: torch.Tensor,
        item_embeddings: nn.Embedding,
    ) -> torch.Tensor:
        """Compute BPR loss.

        Args:
            session_embeddings: Session representations [batch_size, hidden_dim].
            target_items: Target item indices [batch_size].
            negative_items: Negative item indices [batch_size, num_negatives].
            item_embeddings: Item embedding layer.

        Returns:
            BPR loss (scalar).
        """
        # Get item embeddings
        target_embeddings = item_embeddings(target_items)  # [batch_size, hidden_dim]
        negative_embeddings = item_embeddings(
            negative_items
        )  # [batch_size, num_negatives, hidden_dim]

        # Compute scores
        pos_scores = (session_embeddings * target_embeddings).sum(dim=1)  # [batch_size]
        neg_scores = torch.bmm(negative_embeddings, session_embeddings.unsqueeze(2)).squeeze(
            2
        )  # [batch_size, num_negatives]

        # BPR loss: -log(sigmoid(pos - neg))
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8).mean()

        return loss


class ListwiseLoss(nn.Module):
    """Listwise ranking loss (softmax cross-entropy).

    Treats the target item as the positive class and all other items as negatives.

    Args:
        temperature: Temperature for softmax (default: 1.0).
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        session_embeddings: torch.Tensor,
        target_items: torch.Tensor,
        negative_items: torch.Tensor,
        item_embeddings: nn.Embedding,
    ) -> torch.Tensor:
        """Compute listwise loss.

        Args:
            session_embeddings: Session representations [batch_size, hidden_dim].
            target_items: Target item indices [batch_size].
            negative_items: Negative item indices [batch_size, num_negatives].
            item_embeddings: Item embedding layer.

        Returns:
            Listwise loss (scalar).
        """
        # Get item embeddings
        target_embeddings = item_embeddings(target_items)  # [batch_size, hidden_dim]
        negative_embeddings = item_embeddings(
            negative_items
        )  # [batch_size, num_negatives, hidden_dim]

        # Compute scores
        pos_scores = (session_embeddings * target_embeddings).sum(dim=1)  # [batch_size]
        neg_scores = torch.bmm(negative_embeddings, session_embeddings.unsqueeze(2)).squeeze(
            2
        )  # [batch_size, num_negatives]

        # Concatenate positive and negative scores
        all_scores = torch.cat(
            [pos_scores.unsqueeze(1), neg_scores], dim=1
        )  # [batch_size, 1 + num_negatives]

        # Apply temperature
        all_scores = all_scores / self.temperature

        # Softmax cross-entropy (target is always index 0)
        targets = torch.zeros(all_scores.size(0), dtype=torch.long, device=all_scores.device)
        loss = F.cross_entropy(all_scores, targets)

        return loss


class DualLoss(nn.Module):
    """Dual loss combining listwise and contrastive (BPR) losses.

    Loss = alpha * listwise_loss + (1 - alpha) * bpr_loss

    Args:
        alpha: Weight for listwise loss (default: 0.7).
        temperature: Temperature for listwise loss (default: 1.0).
    """

    def __init__(self, alpha: float = 0.7, temperature: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.listwise_loss = ListwiseLoss(temperature=temperature)
        self.bpr_loss = BPRLoss()

    def forward(
        self,
        session_embeddings: torch.Tensor,
        target_items: torch.Tensor,
        negative_items: torch.Tensor,
        item_embeddings: nn.Embedding,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute dual loss.

        Args:
            session_embeddings: Session representations [batch_size, hidden_dim].
            target_items: Target item indices [batch_size].
            negative_items: Negative item indices [batch_size, num_negatives].
            item_embeddings: Item embedding layer.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses.
        """
        # Compute individual losses
        listwise = self.listwise_loss(
            session_embeddings, target_items, negative_items, item_embeddings
        )
        bpr = self.bpr_loss(session_embeddings, target_items, negative_items, item_embeddings)

        # Combine losses
        total_loss = self.alpha * listwise + (1 - self.alpha) * bpr

        # Return loss and components
        loss_dict = {
            "total": total_loss.item(),
            "listwise": listwise.item(),
            "bpr": bpr.item(),
        }

        return total_loss, loss_dict


class SampledSoftmaxLoss(nn.Module):
    """Sampled softmax loss for large-scale recommendation.

    Computes softmax over target + sampled negatives instead of all items.

    Args:
        temperature: Temperature for softmax (default: 1.0).
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        session_embeddings: torch.Tensor,
        target_items: torch.Tensor,
        negative_items: torch.Tensor,
        item_embeddings: nn.Embedding,
    ) -> torch.Tensor:
        """Compute sampled softmax loss.

        Args:
            session_embeddings: Session representations [batch_size, hidden_dim].
            target_items: Target item indices [batch_size].
            negative_items: Negative item indices [batch_size, num_negatives].
            item_embeddings: Item embedding layer.

        Returns:
            Sampled softmax loss (scalar).
        """
        # Same as listwise loss
        return ListwiseLoss(temperature=self.temperature).forward(
            session_embeddings, target_items, negative_items, item_embeddings
        )


def create_loss_function(
    loss_type: str = "dual",
    alpha: float = 0.7,
    temperature: float = 1.0,
) -> nn.Module:
    """Factory function to create loss function.

    Args:
        loss_type: Type of loss ('bpr', 'listwise', 'dual', 'sampled_softmax').
        alpha: Weight for listwise loss in dual loss (default: 0.7).
        temperature: Temperature for softmax-based losses (default: 1.0).

    Returns:
        Loss function module.
    """
    if loss_type == "bpr":
        return BPRLoss()
    elif loss_type == "listwise":
        return ListwiseLoss(temperature=temperature)
    elif loss_type == "dual":
        return DualLoss(alpha=alpha, temperature=temperature)
    elif loss_type == "sampled_softmax":
        return SampledSoftmaxLoss(temperature=temperature)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
