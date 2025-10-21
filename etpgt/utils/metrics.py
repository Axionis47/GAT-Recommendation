"""Evaluation metrics for recommendation systems."""

import torch


def compute_recall_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> float:
    """Compute Recall@K.

    Args:
        predictions: Predicted item IDs, shape (batch_size, num_candidates).
        targets: Ground truth item IDs, shape (batch_size,).
        k: Number of top predictions to consider.

    Returns:
        Recall@K score.
    """
    top_k_preds = predictions[:, :k]

    # Check if target is in top-k predictions
    hits = (top_k_preds == targets.unsqueeze(1)).any(dim=1).float()
    recall = hits.mean().item()

    return recall


def compute_ndcg_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> float:
    """Compute NDCG@K.

    Args:
        predictions: Predicted item IDs, shape (batch_size, num_candidates).
        targets: Ground truth item IDs, shape (batch_size,).
        k: Number of top predictions to consider.

    Returns:
        NDCG@K score.
    """
    top_k_preds = predictions[:, :k]

    # Find position of target in top-k predictions (0-indexed)
    matches = (top_k_preds == targets.unsqueeze(1)).float()
    positions = torch.argmax(matches, dim=1)

    # Compute DCG (only 1 relevant item per session)
    # DCG = 1 / log2(position + 2) if target in top-k, else 0
    has_match = matches.sum(dim=1) > 0
    dcg = torch.where(
        has_match,
        1.0 / torch.log2(positions.float() + 2.0),
        torch.zeros_like(positions, dtype=torch.float32),
    )

    # IDCG = 1 / log2(2) = 1.0 (only 1 relevant item)
    idcg = 1.0

    # NDCG = DCG / IDCG
    ndcg = (dcg / idcg).mean().item()

    return ndcg


def compute_stratified_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    strata: torch.Tensor,
    k_values: list[int] | None = None,
) -> dict:
    """Compute metrics stratified by a categorical variable.

    Args:
        predictions: Predicted item IDs, shape (batch_size, num_candidates).
        targets: Ground truth item IDs, shape (batch_size,).
        strata: Stratum labels, shape (batch_size,).
        k_values: List of k values for Recall@K and NDCG@K.

    Returns:
        Dictionary of stratified metrics.
    """
    if k_values is None:
        k_values = [10, 20]

    results = {}
    unique_strata = torch.unique(strata)

    for stratum in unique_strata:
        mask = strata == stratum
        stratum_preds = predictions[mask]
        stratum_targets = targets[mask]

        stratum_results = {"count": mask.sum().item()}

        for k in k_values:
            recall = compute_recall_at_k(stratum_preds, stratum_targets, k)
            ndcg = compute_ndcg_at_k(stratum_preds, stratum_targets, k)

            stratum_results[f"recall@{k}"] = recall
            stratum_results[f"ndcg@{k}"] = ndcg

        results[f"stratum_{stratum.item()}"] = stratum_results

    return results
