#!/usr/bin/env python3
"""Build co-event graph from sessions.

Creates an undirected graph where:
- Nodes are items
- Edges connect items that co-occur within ±5 steps in sessions
- Edge features: co-occurrence count, last timestamp, event pair histogram
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from etpgt.utils.io import save_json
from etpgt.utils.logging import get_logger
from etpgt.utils.seed import set_seed

# Graph configuration (from DECISIONS.md)
CO_EVENT_WINDOW = 5  # ±5 steps


def build_co_event_graph(
    sessions_df: pd.DataFrame,
    window: int = CO_EVENT_WINDOW,
) -> tuple[pd.DataFrame, dict]:
    """Build co-event graph from sessions.

    Args:
        sessions_df: DataFrame with sessionized events.
        window: Co-occurrence window (±steps).

    Returns:
        Tuple of (edges_df, graph_stats).
    """
    logger = get_logger(__name__)

    logger.info(f"Building co-event graph with window=±{window} steps")

    # Group by session
    session_groups = sessions_df.groupby("session_id")

    # Edge accumulator: (item_i, item_j) -> {count, last_ts, event_pairs}
    edges = defaultdict(lambda: {"count": 0, "last_ts": 0, "event_pairs": defaultdict(int)})

    # Process each session
    for _, group in tqdm(session_groups, desc="Processing sessions"):
        # Sort by timestamp
        events = group.sort_values("timestamp").reset_index(drop=True)

        # Extract item IDs, timestamps, and event types
        items = events["itemid"].tolist()
        timestamps = events["timestamp"].tolist()
        event_types = events["event"].tolist()

        # Create edges within window
        for i in range(len(items)):
            for j in range(i + 1, min(i + window + 1, len(items))):
                item_i = items[i]
                item_j = items[j]

                # Ensure consistent ordering (smaller item ID first)
                if item_i > item_j:
                    item_i, item_j = item_j, item_i
                    event_i, event_j = event_types[j], event_types[i]
                    ts = timestamps[j]
                else:
                    event_i, event_j = event_types[i], event_types[j]
                    ts = timestamps[i]

                # Update edge
                edge_key = (item_i, item_j)
                edges[edge_key]["count"] += 1
                edges[edge_key]["last_ts"] = max(edges[edge_key]["last_ts"], ts)

                # Track event pair types
                event_pair = f"{event_i}_{event_j}"
                edges[edge_key]["event_pairs"][event_pair] += 1

    logger.info(f"Total edges: {len(edges):,}")

    # Convert to DataFrame
    edge_list = []
    for (item_i, item_j), data in tqdm(edges.items(), desc="Converting to DataFrame"):
        edge_list.append(
            {
                "item_i": item_i,
                "item_j": item_j,
                "count": data["count"],
                "last_ts": data["last_ts"],
                "event_pair_hist": dict(data["event_pairs"]),
            }
        )

    edges_df = pd.DataFrame(edge_list)

    # Sort by count (descending)
    edges_df = edges_df.sort_values("count", ascending=False).reset_index(drop=True)

    # Compute graph statistics
    num_nodes = len(set(edges_df["item_i"]) | set(edges_df["item_j"]))
    num_edges = len(edges_df)
    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

    graph_stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "edge_count_mean": float(edges_df["count"].mean()),
        "edge_count_median": float(edges_df["count"].median()),
        "edge_count_min": int(edges_df["count"].min()),
        "edge_count_max": int(edges_df["count"].max()),
    }

    logger.info("\nGraph statistics:")
    logger.info(f"  Nodes: {num_nodes:,}")
    logger.info(f"  Edges: {num_edges:,}")
    logger.info(f"  Avg degree: {avg_degree:.2f}")
    logger.info(f"  Edge count (mean): {graph_stats['edge_count_mean']:.2f}")
    logger.info(f"  Edge count (median): {graph_stats['edge_count_median']:.0f}")
    logger.info(
        f"  Edge count (min/max): {graph_stats['edge_count_min']} / {graph_stats['edge_count_max']}"
    )

    return edges_df, graph_stats


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Build co-event graph from sessions")
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/processed/train.csv",
        help="Input sessions CSV file (typically train split)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/graph_edges.csv",
        help="Output graph edges CSV file",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default="data/processed/graph_stats.json",
        help="Output graph statistics JSON file",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=CO_EVENT_WINDOW,
        help="Co-occurrence window (±steps)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    logger = get_logger(__name__)

    # Create output directory
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load sessions
    logger.info(f"Loading sessions from: {args.input_file}")
    sessions_df = pd.read_csv(args.input_file)

    logger.info(f"Total sessions: {sessions_df['session_id'].nunique():,}")
    logger.info(f"Total events: {len(sessions_df):,}")
    logger.info(f"Unique items: {sessions_df['itemid'].nunique():,}")

    # Build graph
    edges_df, graph_stats = build_co_event_graph(sessions_df, window=args.window)

    # Save edges
    logger.info(f"\nSaving edges to: {output_file}")
    edges_df.to_csv(output_file, index=False)

    # Save stats
    stats_file = Path(args.stats_file)
    save_json(graph_stats, str(stats_file))
    logger.info(f"Saved statistics to: {stats_file}")

    # Show sample edges
    logger.info("\nSample edges (top 10 by count):")
    logger.info(edges_df.head(10).to_string())

    logger.info("\n✓ Graph construction complete!")


if __name__ == "__main__":
    main()
