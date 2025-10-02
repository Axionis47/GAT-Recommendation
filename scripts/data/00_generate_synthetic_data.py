#!/usr/bin/env python3
"""Generate synthetic RetailRocket-like dataset for development and testing.

This creates a realistic e-commerce event dataset with:
- Multiple sessions with varying lengths
- Temporal patterns (sessions span days/weeks)
- Different event types (view, addtocart, transaction)
- Co-occurrence patterns between items
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from etpgt.utils.logging import get_logger
from etpgt.utils.seed import set_seed


def generate_synthetic_events(
    num_sessions: int = 10000,
    num_items: int = 5000,
    min_session_length: int = 3,
    max_session_length: int = 20,
    start_date: str = "2024-01-01",
    duration_days: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic e-commerce events.

    Args:
        num_sessions: Number of sessions to generate.
        num_items: Number of unique items.
        min_session_length: Minimum events per session.
        max_session_length: Maximum events per session.
        start_date: Start date for events.
        duration_days: Duration of event period in days.
        seed: Random seed.

    Returns:
        DataFrame with columns: timestamp, visitorid, event, itemid, transactionid
    """
    set_seed(seed)
    logger = get_logger(__name__)

    logger.info(f"Generating {num_sessions:,} sessions with {num_items:,} items...")

    # Item popularity (power law distribution)
    item_popularity = np.random.zipf(1.5, num_items)
    item_popularity = item_popularity / item_popularity.sum()

    # Start timestamp
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = start_ts + (duration_days * 24 * 60 * 60 * 1000)

    events = []

    for session_idx in tqdm(range(num_sessions), desc="Generating sessions"):
        visitor_id = f"visitor_{session_idx}"

        # Session length
        session_length = random.randint(min_session_length, max_session_length)

        # Session start time (random within duration)
        session_start_ts = random.randint(start_ts, end_ts - 3600000)  # Leave 1 hour

        # Generate events for this session
        current_ts = session_start_ts
        viewed_items = set()
        cart_items = set()

        for event_idx in range(session_length):
            # Time gap between events (1-30 minutes, exponential distribution)
            if event_idx > 0:
                gap_seconds = int(np.random.exponential(300))  # Mean 5 minutes
                gap_seconds = min(gap_seconds, 1800)  # Max 30 minutes
                current_ts += gap_seconds * 1000

            # Select item (with popularity bias)
            if event_idx == 0 or random.random() < 0.7:
                # New item
                item_id = np.random.choice(num_items, p=item_popularity)
            else:
                # Re-view or interact with previously viewed item
                if viewed_items:
                    item_id = random.choice(list(viewed_items))
                else:
                    item_id = np.random.choice(num_items, p=item_popularity)

            # Determine event type
            if item_id in cart_items:
                # Items in cart have higher purchase probability
                event_probs = {"view": 0.3, "addtocart": 0.2, "transaction": 0.5}
            elif item_id in viewed_items:
                # Previously viewed items have higher cart probability
                event_probs = {"view": 0.6, "addtocart": 0.35, "transaction": 0.05}
            else:
                # New items are mostly views
                event_probs = {"view": 0.85, "addtocart": 0.13, "transaction": 0.02}

            event_type = random.choices(
                list(event_probs.keys()), weights=list(event_probs.values())
            )[0]

            # Transaction ID (only for purchases)
            transaction_id = None
            if event_type == "transaction":
                transaction_id = f"txn_{session_idx}_{event_idx}"

            # Update state
            viewed_items.add(item_id)
            if event_type == "addtocart":
                cart_items.add(item_id)

            events.append(
                {
                    "timestamp": current_ts,
                    "visitorid": visitor_id,
                    "event": event_type,
                    "itemid": int(item_id),
                    "transactionid": transaction_id,
                }
            )

    df = pd.DataFrame(events)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Generated {len(df):,} events")
    logger.info(f"Event type distribution:\n{df['event'].value_counts()}")
    logger.info(f"Unique visitors: {df['visitorid'].nunique():,}")
    logger.info(f"Unique items: {df['itemid'].nunique():,}")

    return df


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic RetailRocket-like dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory",
    )
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=10000,
        help="Number of sessions",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=5000,
        help="Number of unique items",
    )
    parser.add_argument(
        "--duration-days",
        type=int,
        default=90,
        help="Duration of event period in days",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    logger = get_logger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate events
    df = generate_synthetic_events(
        num_sessions=args.num_sessions,
        num_items=args.num_items,
        duration_days=args.duration_days,
        seed=args.seed,
    )

    # Save to CSV
    output_file = output_dir / "events.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved to: {output_file}")

    # Show sample
    logger.info(f"\nSample events:\n{df.head(10)}")

    # Show date range
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    logger.info(f"\nDate range: {df['datetime'].min()} to {df['datetime'].max()}")


if __name__ == "__main__":
    main()
