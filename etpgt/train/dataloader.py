"""Data loader for training session-based recommendation models."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


class SessionDataset(Dataset):
    """Dataset for session-based recommendation.

    Args:
        sessions_path: Path to sessions CSV file.
        graph_edges_path: Path to graph edges CSV file.
        num_negatives: Number of negative samples per positive.
        max_session_length: Maximum session length (truncate if longer).
    """

    def __init__(
        self,
        sessions_path: Path | str,
        graph_edges_path: Path | str,
        num_negatives: int = 5,
        max_session_length: int = 50,
    ):
        self.sessions_path = Path(sessions_path)
        self.graph_edges_path = Path(graph_edges_path)
        self.num_negatives = num_negatives
        self.max_session_length = max_session_length

        # Load sessions
        self.sessions_df = pd.read_csv(sessions_path)

        # Group by session_id
        self.session_groups = self.sessions_df.groupby("session_id")
        self.session_ids = list(self.session_groups.groups.keys())

        # Load graph edges
        self.graph_edges_df = pd.read_csv(graph_edges_path)

        # Build edge index (using item_i and item_j columns)
        self.edge_index = torch.tensor(
            np.array([self.graph_edges_df["item_i"].values, self.graph_edges_df["item_j"].values]),
            dtype=torch.long,
        )

        # Get number of items
        self.num_items = (
            max(
                self.sessions_df["itemid"].max(),
                self.graph_edges_df["item_i"].max(),
                self.graph_edges_df["item_j"].max(),
            )
            + 1
        )

    def __len__(self) -> int:
        """Get number of sessions."""
        return len(self.session_ids)

    def __getitem__(self, idx: int) -> dict:
        """Get a session.

        Args:
            idx: Session index.

        Returns:
            Dictionary with:
                - session_items: Item IDs in session [session_length]
                - target_item: Target item ID (last item in session)
                - negative_items: Negative item IDs [num_negatives]
                - edge_index: Edge index for session subgraph [2, num_edges]
        """
        session_id = self.session_ids[idx]
        session_data = self.session_groups.get_group(session_id)

        # Get session items (sorted by timestamp)
        session_data = session_data.sort_values("timestamp")
        session_items = session_data["itemid"].values

        # Truncate if too long
        if len(session_items) > self.max_session_length:
            session_items = session_items[-self.max_session_length:]

        # Target is last item
        target_item = session_items[-1]

        # Context is all items except last
        context_items = session_items[:-1]

        # Sample negative items (not in session)
        negative_items = self._sample_negatives(session_items)

        # Build session subgraph
        edge_index = self._build_session_subgraph(context_items)

        return {
            "session_items": torch.tensor(context_items, dtype=torch.long),
            "target_item": torch.tensor(target_item, dtype=torch.long),
            "negative_items": torch.tensor(negative_items, dtype=torch.long),
            "edge_index": edge_index,
        }

    def _sample_negatives(self, session_items: np.ndarray) -> list:
        """Sample negative items.

        Args:
            session_items: Items in session.

        Returns:
            List of negative item IDs.
        """
        session_items_set = set(session_items)
        negatives = []

        while len(negatives) < self.num_negatives:
            neg_item = torch.randint(1, self.num_items, (1,)).item()
            if neg_item not in session_items_set:
                negatives.append(neg_item)

        return negatives

    def _build_session_subgraph(self, context_items: np.ndarray) -> torch.Tensor:
        """Build session subgraph from context items.

        Args:
            context_items: Context items in session.

        Returns:
            Edge index [2, num_edges].
        """
        context_items_set = set(context_items)

        # Find edges where both item_i and item_j are in context
        mask = self.graph_edges_df["item_i"].isin(context_items_set) & self.graph_edges_df[
            "item_j"
        ].isin(context_items_set)

        if mask.sum() == 0:
            return torch.zeros((2, 0), dtype=torch.long)

        # Get filtered edges
        filtered_edges = self.graph_edges_df[mask]

        # Build edge index
        edge_index = torch.tensor(
            [filtered_edges["item_i"].values, filtered_edges["item_j"].values],
            dtype=torch.long,
        )

        return edge_index


def collate_fn(batch: list[dict]) -> Batch:
    """Collate function for DataLoader.

    Args:
        batch: List of session dictionaries.

    Returns:
        PyG Batch object.
    """
    data_list = []

    for item in batch:
        session_items = item["session_items"]
        edge_index = item["edge_index"]

        # Create mapping from global item IDs to local indices
        unique_items = session_items.unique()
        item_to_idx = {item_id.item(): idx for idx, item_id in enumerate(unique_items)}

        # Remap edge_index to local indices
        if edge_index.numel() > 0:
            valid_edges = []
            for i in range(edge_index.shape[1]):
                src_item = edge_index[0, i].item()
                tgt_item = edge_index[1, i].item()
                if src_item in item_to_idx and tgt_item in item_to_idx:
                    valid_edges.append([item_to_idx[src_item], item_to_idx[tgt_item]])

            if valid_edges:
                edge_index_local = torch.tensor(valid_edges, dtype=torch.long).t()
            else:
                edge_index_local = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index_local = edge_index

        # Create Data object
        data = Data(
            x=unique_items,
            edge_index=edge_index_local,
            target_item=item["target_item"],
            negative_items=item["negative_items"],
        )

        data_list.append(data)

    return Batch.from_data_list(data_list)


def create_dataloader(
    sessions_path: Path | str,
    graph_edges_path: Path | str,
    batch_size: int = 32,
    num_negatives: int = 5,
    max_session_length: int = 50,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for training.

    Args:
        sessions_path: Path to sessions CSV file.
        graph_edges_path: Path to graph edges CSV file.
        batch_size: Batch size.
        num_negatives: Number of negative samples.
        max_session_length: Maximum session length.
        shuffle: Whether to shuffle data.
        num_workers: Number of workers for data loading.

    Returns:
        DataLoader.
    """
    dataset = SessionDataset(
        sessions_path=sessions_path,
        graph_edges_path=graph_edges_path,
        num_negatives=num_negatives,
        max_session_length=max_session_length,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
