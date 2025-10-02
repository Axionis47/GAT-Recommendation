"""Trainer for session-based recommendation models."""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from etpgt.utils.metrics import compute_ndcg_at_k, compute_recall_at_k

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for session-based recommendation models.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer.
        device: Device to train on.
        output_dir: Output directory for checkpoints.
        max_epochs: Maximum number of epochs.
        patience: Early stopping patience.
        eval_every: Evaluate every N epochs.
        k_values: K values for evaluation metrics.
        loss_fn: Custom loss function (optional). If None, uses model's compute_loss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        output_dir: Path | str = "outputs",
        max_epochs: int = 100,
        patience: int = 10,
        eval_every: int = 1,
        k_values: list[int] | None = None,
        loss_fn: nn.Module | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        self.patience = patience
        self.eval_every = eval_every
        self.k_values = k_values if k_values is not None else [10, 20]
        self.loss_fn = loss_fn

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_metrics": []}

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            batch = batch.to(self.device)

            # Forward pass
            session_embeddings = self.model(batch)

            # Reshape negative_items from [batch_size * num_negatives] to [batch_size, num_negatives]
            batch_size = batch.target_item.shape[0]
            num_negatives = batch.negative_items.shape[0] // batch_size
            negative_items = batch.negative_items.view(batch_size, num_negatives)

            # Compute loss
            if self.loss_fn is not None:
                # Use custom loss function
                if (
                    hasattr(self.loss_fn, "forward")
                    and len(self.loss_fn.forward.__code__.co_varnames) > 4
                ):
                    # Dual loss returns (loss, loss_dict)
                    loss_output = self.loss_fn(
                        session_embeddings,
                        batch.target_item,
                        negative_items,
                        self.model.item_embedding,
                    )
                    if isinstance(loss_output, tuple):
                        loss, _ = loss_output
                    else:
                        loss = loss_output
                else:
                    loss = self.loss_fn(
                        session_embeddings,
                        batch.target_item,
                        negative_items,
                        self.model.item_embedding,
                    )
            else:
                # Use model's default loss
                loss = self.model.compute_loss(
                    session_embeddings,
                    batch.target_item,
                    negative_items,
                )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on validation set.

        Returns:
            Dictionary of metrics.
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch = batch.to(self.device)

            # Forward pass
            session_embeddings = self.model(batch)

            # Get predictions
            predictions = self.model.predict(session_embeddings, k=max(self.k_values))

            all_predictions.append(predictions.cpu())
            all_targets.append(batch.target_item.cpu())

        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        metrics = {}
        for k in self.k_values:
            recall = compute_recall_at_k(all_predictions[:, :k], all_targets, k=k)
            ndcg = compute_ndcg_at_k(all_predictions[:, :k], all_targets, k=k)
            metrics[f"recall@{k}"] = recall
            metrics[f"ndcg@{k}"] = ndcg

        return metrics

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save checkpoint.

        Args:
            is_best: Whether this is the best checkpoint.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_metric": self.best_val_metric,
            "history": self.history,
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

    def load_checkpoint(self, checkpoint_path: Path | str) -> None:
        """Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_metric = checkpoint["best_val_metric"]
        self.history = checkpoint["history"]
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def train(self) -> dict:
        """Train the model.

        Returns:
            Training history.
        """
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")

            # Evaluate
            if (epoch + 1) % self.eval_every == 0:
                val_metrics = self.evaluate()
                self.history["val_metrics"].append(val_metrics)

                # Log metrics
                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
                logger.info(f"Epoch {epoch}: {metrics_str}")

                # Check if best model
                val_metric = val_metrics[f"recall@{self.k_values[0]}"]
                is_best = val_metric > self.best_val_metric

                if is_best:
                    self.best_val_metric = val_metric
                    self.patience_counter = 0
                    logger.info(f"New best model! recall@{self.k_values[0]}={val_metric:.4f}")
                else:
                    self.patience_counter += 1

                # Save checkpoint
                self.save_checkpoint(is_best=is_best)

                # Early stopping
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Save final history
        history_path = self.output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        return self.history
