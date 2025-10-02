#!/usr/bin/env python3
"""Sessionize RetailRocket events.

Creates sessions from raw events using:
- 30-minute inactivity gap
- Minimum session length of 3 events
- Temporal ordering within sessions
"""

import argparse
from pathlib import Path

import pandas as pd

from etpgt.utils.io import save_json
from etpgt.utils.logging import get_logger
from etpgt.utils.seed import set_seed

# Session configuration (from DECISIONS.md)
SESSION_GAP_MINUTES = 30
MIN_SESSION_LENGTH = 3
SESSION_GAP_MS = SESSION_GAP_MINUTES * 60 * 1000


def sessionize_events(
    events_df: pd.DataFrame,
    gap_ms: int = SESSION_GAP_MS,
    min_length: int = MIN_SESSION_LENGTH,
) -> pd.DataFrame:
    """Sessionize events based on inactivity gap.

    Args:
        events_df: DataFrame with columns [timestamp, visitorid, event, itemid, transactionid].
        gap_ms: Maximum gap between events in milliseconds.
        min_length: Minimum number of events per session.

    Returns:
        DataFrame with additional session_id column.
    """
    logger = get_logger(__name__)

    # Sort by visitor and timestamp
    df = events_df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)

    logger.info(f"Total events: {len(df):,}")
    logger.info(f"Unique visitors: {df['visitorid'].nunique():,}")

    # Calculate time gaps within each visitor
    df["time_gap"] = df.groupby("visitorid")["timestamp"].diff()

    # Mark session boundaries (gap > threshold or new visitor)
    df["new_session"] = (df["time_gap"].isna()) | (df["time_gap"] > gap_ms)

    # Assign session IDs
    df["session_id"] = df["new_session"].cumsum()

    # Add session prefix
    df["session_id"] = "sess_" + df["session_id"].astype(str)

    # Count events per session
    session_lengths = df.groupby("session_id").size()

    logger.info(f"Total sessions (before filtering): {len(session_lengths):,}")
    logger.info("Session length distribution:")
    logger.info(f"  Mean: {session_lengths.mean():.2f}")
    logger.info(f"  Median: {session_lengths.median():.0f}")
    logger.info(f"  Min: {session_lengths.min()}")
    logger.info(f"  Max: {session_lengths.max()}")

    # Filter sessions by minimum length
    valid_sessions = session_lengths[session_lengths >= min_length].index
    df_filtered = df[df["session_id"].isin(valid_sessions)].copy()

    logger.info(f"Sessions after filtering (>= {min_length} events): {len(valid_sessions):,}")
    logger.info(f"Events after filtering: {len(df_filtered):,}")
    logger.info(f"Retention: {len(df_filtered) / len(df) * 100:.2f}%")

    # Drop temporary columns
    df_filtered = df_filtered.drop(columns=["time_gap", "new_session"])

    return df_filtered


def compute_session_stats(sessions_df: pd.DataFrame) -> dict:
    """Compute session statistics.

    Args:
        sessions_df: DataFrame with sessionized events.

    Returns:
        Dictionary of statistics.
    """
    stats = {}

    # Session-level stats
    session_groups = sessions_df.groupby("session_id")

    stats["num_sessions"] = len(session_groups)
    stats["num_events"] = len(sessions_df)
    stats["num_unique_visitors"] = sessions_df["visitorid"].nunique()
    stats["num_unique_items"] = sessions_df["itemid"].nunique()

    # Session lengths
    session_lengths = session_groups.size()
    stats["session_length_mean"] = float(session_lengths.mean())
    stats["session_length_median"] = float(session_lengths.median())
    stats["session_length_min"] = int(session_lengths.min())
    stats["session_length_max"] = int(session_lengths.max())
    stats["session_length_std"] = float(session_lengths.std())

    # Session durations (in minutes)
    session_durations = (session_groups["timestamp"].max() - session_groups["timestamp"].min()) / (
        60 * 1000
    )
    stats["session_duration_mean_min"] = float(session_durations.mean())
    stats["session_duration_median_min"] = float(session_durations.median())

    # Event type distribution
    event_counts = sessions_df["event"].value_counts().to_dict()
    stats["event_type_counts"] = event_counts

    # Items per session
    items_per_session = session_groups["itemid"].nunique()
    stats["unique_items_per_session_mean"] = float(items_per_session.mean())
    stats["unique_items_per_session_median"] = float(items_per_session.median())

    return stats


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Sessionize RetailRocket events")
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/raw/events.csv",
        help="Input events CSV file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/interim/sessions.csv",
        help="Output sessions CSV file",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default="data/interim/session_stats.json",
        help="Output statistics JSON file",
    )
    parser.add_argument(
        "--gap-minutes",
        type=int,
        default=SESSION_GAP_MINUTES,
        help="Session inactivity gap in minutes",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=MIN_SESSION_LENGTH,
        help="Minimum session length",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    logger = get_logger(__name__)

    # Create output directory
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load events
    logger.info(f"Loading events from: {args.input_file}")
    events_df = pd.read_csv(args.input_file)

    # Sessionize
    gap_ms = args.gap_minutes * 60 * 1000
    logger.info(f"Sessionizing with gap={args.gap_minutes}min, min_length={args.min_length}")

    sessions_df = sessionize_events(events_df, gap_ms=gap_ms, min_length=args.min_length)

    # Compute stats
    stats = compute_session_stats(sessions_df)

    # Save sessions
    logger.info(f"Saving sessions to: {output_file}")
    sessions_df.to_csv(output_file, index=False)

    # Save stats
    stats_file = Path(args.stats_file)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(stats, str(stats_file))
    logger.info(f"Saved statistics to: {stats_file}")

    # Print summary
    logger.info("\n=== Session Statistics ===")
    logger.info(f"Total sessions: {stats['num_sessions']:,}")
    logger.info(f"Total events: {stats['num_events']:,}")
    logger.info(f"Unique visitors: {stats['num_unique_visitors']:,}")
    logger.info(f"Unique items: {stats['num_unique_items']:,}")
    logger.info(f"Avg session length: {stats['session_length_mean']:.2f} events")
    logger.info(f"Avg session duration: {stats['session_duration_mean_min']:.2f} minutes")
    logger.info(f"Event types: {stats['event_type_counts']}")

    logger.info("\n✓ Sessionization complete!")


if __name__ == "__main__":
    main()
