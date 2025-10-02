"""Tests for data contract validation."""

from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")


@pytest.fixture
def sessions_df():
    """Load sessionized data."""
    sessions_file = DATA_DIR / "sessions.csv"
    if not sessions_file.exists():
        pytest.skip("Sessions file not available")
    return pd.read_csv(sessions_file)


@pytest.fixture
def train_df():
    """Load train split."""
    train_file = PROCESSED_DIR / "train.csv"
    if not train_file.exists():
        pytest.skip("Train file not available")
    return pd.read_csv(train_file)


def test_event_schema_validation(sessions_df) -> None:
    """Test that events conform to schema."""
    # Required columns from DATA_CONTRACT.yaml
    required_columns = ["timestamp", "visitorid", "event", "itemid", "transactionid", "session_id"]

    # Check all required columns are present
    for col in required_columns:
        assert col in sessions_df.columns, f"Missing required column: {col}"

    # Check data types
    assert sessions_df["timestamp"].dtype in [int, "int64"], "timestamp must be integer (Unix ms)"
    assert sessions_df["itemid"].dtype in [int, "int64"], "itemid must be integer"

    # Check event types are valid
    valid_events = {"view", "addtocart", "transaction"}
    unique_events = set(sessions_df["event"].unique())
    assert unique_events.issubset(
        valid_events
    ), f"Invalid event types: {unique_events - valid_events}"

    # Check no null values in required fields (except transactionid)
    for col in ["timestamp", "visitorid", "event", "itemid", "session_id"]:
        null_count = sessions_df[col].isna().sum()
        assert null_count == 0, f"Column {col} has {null_count} null values"


def test_timestamps_monotonic_per_session(sessions_df) -> None:
    """Test that timestamps are strictly increasing within each session."""
    # Group by session
    for session_id, group in sessions_df.groupby("session_id"):
        timestamps = group.sort_index()["timestamp"].values

        # Check monotonically increasing (not strictly, as some events can have same timestamp)
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1], (
                f"Session {session_id}: timestamp[{i}]={timestamps[i]} > "
                f"timestamp[{i+1}]={timestamps[i+1]}"
            )


def test_session_min_length(sessions_df) -> None:
    """Test that all sessions have >= 3 events."""
    # Count events per session
    session_lengths = sessions_df.groupby("session_id").size()

    # Check minimum length
    min_length = session_lengths.min()
    assert min_length >= 3, f"Found session with length {min_length} < 3"

    # Check no sessions with length < 3
    short_sessions = session_lengths[session_lengths < 3]
    assert len(short_sessions) == 0, f"Found {len(short_sessions)} sessions with length < 3"
