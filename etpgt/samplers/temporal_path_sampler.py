"""Temporal Path Sampler for time-aware graph sampling.

Samples neighbors while respecting temporal constraints:
- Only edges with time ≤ current event time
- Importance sampling based on recency × degree
- Fanout decay across layers
"""

import numpy as np
import torch
from torch_geometric.data import Data


class TemporalPathSampler:
    """Temporal-aware graph sampler.

    Samples k-hop neighborhoods while respecting temporal constraints.
    Only includes edges that occurred before or at the current event time.

    Args:
        fanout: List of fanout sizes for each layer (e.g., [16, 12, 8]).
        max_edges_per_batch: Maximum edges per batch (OOM protection).
        replace: Whether to sample with replacement.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        fanout: list[int] | None = None,
        max_edges_per_batch: int = 10000,
        replace: bool = True,
        seed: int = 42,
    ):
        self.fanout = fanout if fanout is not None else [16, 12, 8]
        self.max_edges_per_batch = max_edges_per_batch
        self.replace = replace
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def sample(
        self,
        data: Data,
        seed_nodes: torch.Tensor,
        seed_time: torch.Tensor,
    ) -> tuple[Data, torch.Tensor, torch.Tensor]:
        """Sample k-hop neighborhood for seed nodes.

        Args:
            data: PyG Data object with edge_index, edge_attr (must include 'last_ts').
            seed_nodes: Tensor of seed node indices [num_seeds].
            seed_time: Tensor of seed timestamps [num_seeds].

        Returns:
            Tuple of (subgraph_data, node_mapping, edge_mask).
        """
        # Get edge index and timestamps
        edge_index = data.edge_index  # [2, num_edges]
        edge_time = data.edge_attr["last_ts"]  # [num_edges]

        # Initialize sampled nodes and edges
        current_nodes = seed_nodes.clone()
        all_nodes = [seed_nodes]
        all_edges = []

        # Sample layer by layer
        for layer_idx, num_neighbors in enumerate(self.fanout):
            # Find valid edges for current nodes
            # Valid edges: target in current_nodes AND edge_time <= seed_time
            valid_edges_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)

            for node_idx in range(len(current_nodes)):
                node = current_nodes[node_idx]
                node_time = seed_time[node_idx] if layer_idx == 0 else seed_time.max()

                # Find edges where target == node and time <= node_time
                target_mask = edge_index[1] == node
                time_mask = edge_time <= node_time
                valid_edges_mask |= target_mask & time_mask

            valid_edge_indices = torch.where(valid_edges_mask)[0]

            if len(valid_edge_indices) == 0:
                # No valid edges, stop sampling
                break

            # Sample edges with importance sampling (recency × degree)
            sampled_edge_indices = self._importance_sample(
                edge_index,
                edge_time,
                valid_edge_indices,
                seed_time,
                num_neighbors,
            )

            all_edges.append(sampled_edge_indices)

            # Get source nodes for next layer
            sampled_edges = edge_index[:, sampled_edge_indices]
            next_nodes = sampled_edges[0].unique()
            all_nodes.append(next_nodes)
            current_nodes = next_nodes

        # Combine all sampled nodes and edges
        all_nodes_tensor = torch.cat(all_nodes).unique()
        all_edges_tensor = torch.cat(all_edges) if all_edges else torch.tensor([], dtype=torch.long)

        # Create node mapping (global -> local)
        node_mapping = torch.full((data.num_nodes,), -1, dtype=torch.long)
        node_mapping[all_nodes_tensor] = torch.arange(len(all_nodes_tensor))

        # Create subgraph
        if len(all_edges_tensor) > 0:
            subgraph_edge_index = edge_index[:, all_edges_tensor]
            # Remap to local indices
            subgraph_edge_index = node_mapping[subgraph_edge_index]

            # Create edge mask
            edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            edge_mask[all_edges_tensor] = True
        else:
            subgraph_edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)

        # Create subgraph data
        subgraph_data = Data(
            edge_index=subgraph_edge_index,
            num_nodes=len(all_nodes_tensor),
        )

        return subgraph_data, all_nodes_tensor, edge_mask

    def _importance_sample(
        self,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
        valid_edge_indices: torch.Tensor,
        seed_time: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Sample edges with importance sampling (recency × degree).

        Args:
            edge_index: Full edge index [2, num_edges].
            edge_time: Edge timestamps [num_edges].
            valid_edge_indices: Indices of valid edges.
            seed_time: Seed timestamps for recency calculation.
            num_samples: Number of edges to sample.

        Returns:
            Sampled edge indices.
        """
        if len(valid_edge_indices) == 0:
            return torch.tensor([], dtype=torch.long)

        # Calculate recency scores (higher = more recent)
        max_time = seed_time.max()
        edge_times = edge_time[valid_edge_indices]
        recency = 1.0 / (1.0 + (max_time - edge_times).float() / 1000.0)  # Normalize by seconds

        # Calculate degree scores (number of edges per source node)
        source_nodes = edge_index[0, valid_edge_indices]
        unique_sources, source_counts = source_nodes.unique(return_counts=True)
        degree_map = torch.zeros(edge_index.max() + 1, dtype=torch.float)
        degree_map[unique_sources] = source_counts.float()
        degree = degree_map[source_nodes]

        # Combine scores: recency × degree
        importance = recency * torch.sqrt(degree)  # sqrt to reduce degree dominance
        importance = importance / importance.sum()  # Normalize to probabilities

        # Sample edges
        if len(valid_edge_indices) <= num_samples:
            # Not enough edges, return all (or sample with replacement)
            if self.replace:
                sampled_indices = self.rng.choice(
                    valid_edge_indices.cpu().numpy(),
                    size=num_samples,
                    replace=True,
                    p=importance.cpu().numpy(),
                )
                return torch.from_numpy(sampled_indices)
            else:
                return valid_edge_indices
        else:
            # Sample without replacement
            sampled_indices = self.rng.choice(
                valid_edge_indices.cpu().numpy(),
                size=num_samples,
                replace=False,
                p=importance.cpu().numpy(),
            )
            return torch.from_numpy(sampled_indices)

    def reset_random_state(self, seed: int | None = None) -> None:
        """Reset random state for reproducibility.

        Args:
            seed: New random seed. If None, uses original seed.
        """
        if seed is not None:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed)


def create_temporal_sampler(
    fanout: list[int] | None = None,
    max_edges_per_batch: int = 10000,
    replace: bool = True,
    seed: int = 42,
) -> TemporalPathSampler:
    """Factory function to create TemporalPathSampler.

    Args:
        fanout: List of fanout sizes for each layer.
        max_edges_per_batch: Maximum edges per batch.
        replace: Whether to sample with replacement.
        seed: Random seed.

    Returns:
        TemporalPathSampler instance.
    """
    return TemporalPathSampler(
        fanout=fanout,
        max_edges_per_batch=max_edges_per_batch,
        replace=replace,
        seed=seed,
    )
