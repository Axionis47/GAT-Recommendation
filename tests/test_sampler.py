"""Tests for temporal path sampler."""

import torch
from torch_geometric.data import Data

from etpgt.samplers.temporal_path_sampler import TemporalPathSampler


def test_sampler_rejects_future_edges() -> None:
    """Test that sampler only includes edges with time <= current event time."""
    # Create a graph with temporal edges
    # Edge: 0 -> 1 (time=100), 0 -> 2 (time=200), 1 -> 2 (time=150)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    edge_time = torch.tensor([100, 200, 150], dtype=torch.long)

    data = Data(
        edge_index=edge_index,
        edge_attr={"last_ts": edge_time},
        num_nodes=3,
    )

    # Sample from node 2 at time 120
    # Should only include edge 0->1 (time=100), not 0->2 (time=200) or 1->2 (time=150)
    sampler = TemporalPathSampler(fanout=[2], seed=42)
    seed_nodes = torch.tensor([2])
    seed_time = torch.tensor([120])

    _, _, edge_mask = sampler.sample(data, seed_nodes, seed_time)

    # Check that no future edges are included
    sampled_edge_times = edge_time[edge_mask]
    assert all(sampled_edge_times <= 120), f"Found future edges: {sampled_edge_times}"


def test_sampler_fanout_respected() -> None:
    """Test that sampler respects fanout limits."""
    # Create a star graph: node 0 connected to nodes 1-20
    num_neighbors = 20
    edge_index = torch.stack(
        [
            torch.arange(num_neighbors),
            torch.zeros(num_neighbors, dtype=torch.long),
        ]
    )
    edge_time = torch.ones(num_neighbors, dtype=torch.long) * 100

    data = Data(
        edge_index=edge_index,
        edge_attr={"last_ts": edge_time},
        num_nodes=num_neighbors + 1,
    )

    # Sample from node 0 with fanout=[5]
    sampler = TemporalPathSampler(fanout=[5], seed=42)
    seed_nodes = torch.tensor([0])
    seed_time = torch.tensor([200])

    _, _, edge_mask = sampler.sample(data, seed_nodes, seed_time)

    # Check that at most 5 edges are sampled
    num_sampled_edges = edge_mask.sum().item()
    assert num_sampled_edges <= 5, f"Sampled {num_sampled_edges} edges, expected <= 5"


def test_sampler_deterministic_with_seed() -> None:
    """Test that sampler produces identical results with same seed."""
    # Create a simple graph
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [1, 2, 2, 3, 3]], dtype=torch.long)
    edge_time = torch.tensor([100, 110, 120, 130, 140], dtype=torch.long)

    data = Data(
        edge_index=edge_index,
        edge_attr={"last_ts": edge_time},
        num_nodes=4,
    )

    seed_nodes = torch.tensor([3])
    seed_time = torch.tensor([200])

    # Sample twice with same seed
    sampler1 = TemporalPathSampler(fanout=[2, 2], seed=42)
    subgraph1, nodes1, mask1 = sampler1.sample(data, seed_nodes, seed_time)

    sampler2 = TemporalPathSampler(fanout=[2, 2], seed=42)
    subgraph2, nodes2, mask2 = sampler2.sample(data, seed_nodes, seed_time)

    # Check that results are identical
    assert torch.equal(nodes1, nodes2), "Sampled nodes differ"
    assert torch.equal(mask1, mask2), "Edge masks differ"
    assert torch.equal(subgraph1.edge_index, subgraph2.edge_index), "Edge indices differ"
