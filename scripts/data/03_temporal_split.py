#!/usr/bin/env python3
"""Create temporal train/val/test splits with blackout periods.

Splits sessions temporally to prevent data leakage:
- 70% train, 15% validation, 15% test
- 1-3 day blackout periods between splits
- No session spans multiple splits
- Temporal ordering preserved
"""

import argparse
from pathlib import Path

import pandas as pd

from etpgt.utils.io import save_json
from etpgt.utils.logging import get_logger
from etpgt.utils.seed import set_seed

# Split configuration (from DECISIONS.md)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BLACKOUT_DAYS_MIN = 1
BLACKOUT_DAYS_MAX = 3


def create_temporal_splits(
    sessions_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    blackout_days: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Create temporal train/val/test splits with blackout periods.

    Args:
        sessions_df: DataFrame with sessionized events.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        blackout_days: Number of days for blackout period.

    Returns:
        Tuple of (train_df, val_df, test_df, split_info).
    """
    logger = get_logger(__name__)

    # Get session-level timestamps
    session_times = sessions_df.groupby("session_id")["timestamp"].agg(["min", "max"]).reset_index()
    session_times.columns = ["session_id", "start_ts", "end_ts"]

    # Sort by session start time
    session_times = session_times.sort_values("start_ts").reset_index(drop=True)

    # Calculate split points based on number of sessions
    num_sessions = len(session_times)
    train_end_idx = int(num_sessions * train_ratio)
    val_end_idx = int(num_sessions * (train_ratio + val_ratio))

    logger.info(f"Total sessions: {num_sessions:,}")
    logger.info(f"Target split: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")

    # Get split timestamps (end of last session in each split)
    train_end_ts = session_times.iloc[train_end_idx - 1]["end_ts"]
    val_end_ts = session_times.iloc[val_end_idx - 1]["end_ts"]

    # Add blackout periods
    blackout_ms = blackout_days * 24 * 60 * 60 * 1000
    train_blackout_end = train_end_ts + blackout_ms
    val_blackout_end = val_end_ts + blackout_ms

    logger.info(
        f"Blackout period: {blackout_days} days ({blackout_ms / (24 * 60 * 60 * 1000):.1f} days)"
    )

    # Assign sessions to splits based on start time
    # Sessions must not span blackout periods
    def assign_split(row):
        start_ts = row["start_ts"]
        end_ts = row["end_ts"]

        # Train: session ends before train_end_ts
        if end_ts <= train_end_ts:
            return "train"
        # Blackout 1: session starts or ends in blackout period
        elif start_ts <= train_blackout_end:
            return "blackout_1"
        # Val: session ends before val_end_ts
        elif end_ts <= val_end_ts:
            return "val"
        # Blackout 2: session starts or ends in blackout period
        elif start_ts <= val_blackout_end:
            return "blackout_2"
        # Test: remaining sessions
        else:
            return "test"

    session_times["split"] = session_times.apply(assign_split, axis=1)

    # Count sessions per split
    split_counts = session_times["split"].value_counts()
    logger.info("\nSession distribution:")
    for split in ["train", "blackout_1", "val", "blackout_2", "test"]:
        count = split_counts.get(split, 0)
        pct = count / num_sessions * 100
        logger.info(f"  {split:12s}: {count:7,} ({pct:5.2f}%)")

    # Remove blackout sessions
    valid_sessions = session_times[~session_times["split"].str.contains("blackout")].copy()

    # Merge split labels back to events
    sessions_with_split = sessions_df.merge(
        valid_sessions[["session_id", "split"]], on="session_id", how="inner"
    )

    # Split into train/val/test
    train_df = sessions_with_split[sessions_with_split["split"] == "train"].drop(columns=["split"])
    val_df = sessions_with_split[sessions_with_split["split"] == "val"].drop(columns=["split"])
    test_df = sessions_with_split[sessions_with_split["split"] == "test"].drop(columns=["split"])

    # Compute split info
    split_info = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "blackout_days": blackout_days,
        "train_sessions": len(train_df["session_id"].unique()),
        "val_sessions": len(val_df["session_id"].unique()),
        "test_sessions": len(test_df["session_id"].unique()),
        "train_events": len(train_df),
        "val_events": len(val_df),
        "test_events": len(test_df),
        "blackout_1_sessions": int(split_counts.get("blackout_1", 0)),
        "blackout_2_sessions": int(split_counts.get("blackout_2", 0)),
        "train_end_ts": int(train_end_ts),
        "val_end_ts": int(val_end_ts),
        "train_blackout_end_ts": int(train_blackout_end),
        "val_blackout_end_ts": int(val_blackout_end),
    }

    # Verify no temporal leakage
    train_max_ts = train_df["timestamp"].max()
    val_min_ts = val_df["timestamp"].min()
    val_max_ts = val_df["timestamp"].max()
    test_min_ts = test_df["timestamp"].min()

    logger.info("\nTemporal boundaries:")
    logger.info(f"  Train max:  {pd.to_datetime(train_max_ts, unit='ms')}")
    logger.info(f"  Val min:    {pd.to_datetime(val_min_ts, unit='ms')}")
    logger.info(f"  Val max:    {pd.to_datetime(val_max_ts, unit='ms')}")
    logger.info(f"  Test min:   {pd.to_datetime(test_min_ts, unit='ms')}")

    # Check for leakage
    if train_max_ts >= val_min_ts:
        logger.warning("⚠ Potential temporal leakage: train overlaps with val!")
    if val_max_ts >= test_min_ts:
        logger.warning("⚠ Potential temporal leakage: val overlaps with test!")

    logger.info("\n✓ Temporal splits created successfully")

    return train_df, val_df, test_df, split_info


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Create temporal train/val/test splits")
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/interim/sessions.csv",
        help="Input sessions CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
        help="Test set ratio",
    )
    parser.add_argument(
        "--blackout-days",
        type=int,
        default=2,
        help="Blackout period in days",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    logger = get_logger(__name__)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sessions
    logger.info(f"Loading sessions from: {args.input_file}")
    sessions_df = pd.read_csv(args.input_file)

    # Create splits
    logger.info("Creating temporal splits...")
    train_df, val_df, test_df, split_info = create_temporal_splits(
        sessions_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        blackout_days=args.blackout_days,
    )

    # Save splits
    train_file = output_dir / "train.csv"
    val_file = output_dir / "val.csv"
    test_file = output_dir / "test.csv"
    info_file = output_dir / "split_info.json"

    logger.info(f"Saving train to: {train_file}")
    train_df.to_csv(train_file, index=False)

    logger.info(f"Saving val to: {val_file}")
    val_df.to_csv(val_file, index=False)

    logger.info(f"Saving test to: {test_file}")
    test_df.to_csv(test_file, index=False)

    logger.info(f"Saving split info to: {info_file}")
    save_json(split_info, str(info_file))

    # Print summary
    logger.info("\n=== Split Summary ===")
    logger.info(
        f"Train: {split_info['train_sessions']:,} sessions, {split_info['train_events']:,} events"
    )
    logger.info(
        f"Val:   {split_info['val_sessions']:,} sessions, {split_info['val_events']:,} events"
    )
    logger.info(
        f"Test:  {split_info['test_sessions']:,} sessions, {split_info['test_events']:,} events"
    )
    logger.info(f"Blackout 1: {split_info['blackout_1_sessions']:,} sessions removed")
    logger.info(f"Blackout 2: {split_info['blackout_2_sessions']:,} sessions removed")

    logger.info("\n✓ Temporal splits complete!")


if __name__ == "__main__":
    main()
