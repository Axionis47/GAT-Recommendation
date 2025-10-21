"""Tests for temporal split validation (leakage prevention)."""

from pathlib import Path

import pandas as pd
import pytest

from etpgt.utils.io import load_json

DATA_DIR = Path("data/processed")


@pytest.fixture
def split_data():
    """Load train/val/test splits."""
    if not DATA_DIR.exists():
        pytest.skip("Processed data not available")

    train_file = DATA_DIR / "train.csv"
    val_file = DATA_DIR / "val.csv"
    test_file = DATA_DIR / "test.csv"
    info_file = DATA_DIR / "split_info.json"

    if not all([train_file.exists(), val_file.exists(), test_file.exists(), info_file.exists()]):
        pytest.skip("Split files not available")

    return {
        "train": pd.read_csv(train_file),
        "val": pd.read_csv(val_file),
        "test": pd.read_csv(test_file),
        "info": load_json(str(info_file)),
    }


def test_no_temporal_leakage(split_data) -> None:
    """Test that validation/test sets have no timestamps before train max timestamp."""
    train_df = split_data["train"]
    val_df = split_data["val"]
    test_df = split_data["test"]
    info = split_data["info"]

    # Get max/min timestamps
    train_max_ts = train_df["timestamp"].max()
    val_min_ts = val_df["timestamp"].min()
    val_max_ts = val_df["timestamp"].max()
    test_min_ts = test_df["timestamp"].min()

    # Blackout period in milliseconds
    blackout_ms = info["blackout_days"] * 24 * 60 * 60 * 1000

    # Check no temporal leakage with blackout period
    assert train_max_ts + blackout_ms <= val_min_ts, (
        f"Temporal leakage: train max ({train_max_ts}) + blackout ({blackout_ms}) "
        f"> val min ({val_min_ts})"
    )

    assert val_max_ts + blackout_ms <= test_min_ts, (
        f"Temporal leakage: val max ({val_max_ts}) + blackout ({blackout_ms}) "
        f"> test min ({test_min_ts})"
    )


def test_blackout_period_respected(split_data) -> None:
    """Test that blackout periods are respected between splits."""
    info = split_data["info"]

    blackout_days = info["blackout_days"]

    # Check blackout period is within expected range (1-3 days)
    assert 1 <= blackout_days <= 3, f"Blackout period {blackout_days} days not in range [1, 3]"


def test_no_session_spans_splits(split_data) -> None:
    """Test that no session appears in multiple splits."""
    train_df = split_data["train"]
    val_df = split_data["val"]
    test_df = split_data["test"]

    # Get unique session IDs from each split
    train_sessions = set(train_df["session_id"].unique())
    val_sessions = set(val_df["session_id"].unique())
    test_sessions = set(test_df["session_id"].unique())

    # Check no overlap
    train_val_overlap = train_sessions & val_sessions
    train_test_overlap = train_sessions & test_sessions
    val_test_overlap = val_sessions & test_sessions

    assert len(train_val_overlap) == 0, f"Train/Val overlap: {len(train_val_overlap)} sessions"
    assert len(train_test_overlap) == 0, f"Train/Test overlap: {len(train_test_overlap)} sessions"
    assert len(val_test_overlap) == 0, f"Val/Test overlap: {len(val_test_overlap)} sessions"
