#!/usr/bin/env python3
"""Download RetailRocket dataset using Kaggle API.

RetailRocket is an e-commerce dataset with user-item interactions.
Dataset: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

For this project, we use the events.csv file which contains:
- timestamp: Unix timestamp
- visitorid: User/session identifier
- event: Event type (view, addtocart, transaction)
- itemid: Item identifier
- transactionid: Transaction ID (for purchases)

Setup Kaggle API:
1. Create account at kaggle.com
2. Go to Account settings -> API -> Create New API Token
3. Download kaggle.json
4. Place at ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME and KAGGLE_KEY env vars)
5. chmod 600 ~/.kaggle/kaggle.json
"""

import argparse
from pathlib import Path

import pandas as pd

from etpgt.utils.logging import get_logger
from etpgt.utils.seed import set_seed

# Dataset information
DATASET_INFO = {
    "name": "RetailRocket E-commerce Dataset",
    "kaggle_dataset": "retailrocket/ecommerce-dataset",
    "source": "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset",
    "files": {
        "events.csv": {
            "description": "User interaction events",
            "expected_size_mb": 50,
            "columns": ["timestamp", "visitorid", "event", "itemid", "transactionid"],
        }
    },
}


def download_from_kaggle(dataset: str, output_dir: Path) -> None:
    """Download dataset from Kaggle using API.

    Args:
        dataset: Kaggle dataset identifier (e.g., 'retailrocket/ecommerce-dataset').
        output_dir: Directory to download files to.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise ImportError("Kaggle API not installed. Install with: pip install kaggle") from e

    logger = get_logger(__name__)

    # Initialize API
    api = KaggleApi()
    api.authenticate()

    logger.info(f"Downloading dataset: {dataset}")
    logger.info(f"Output directory: {output_dir}")

    # Download dataset
    api.dataset_download_files(dataset, path=str(output_dir), unzip=True)

    logger.info("✓ Download complete")


def verify_file(file_path: Path, expected_columns: list[str]) -> bool:
    """Verify downloaded file.

    Args:
        file_path: Path to file.
        expected_columns: Expected column names.

    Returns:
        True if file is valid.
    """
    logger = get_logger(__name__)

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    # Check file size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {size_mb:.2f} MB")

    # Check columns
    try:
        df = pd.read_csv(file_path, nrows=5)
        actual_columns = df.columns.tolist()

        if actual_columns != expected_columns:
            logger.error(f"Column mismatch. Expected: {expected_columns}, Got: {actual_columns}")
            return False

        logger.info(f"Columns verified: {actual_columns}")
        logger.info(f"Sample rows:\n{df.head()}")

        return True

    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return False


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Download RetailRocket dataset from Kaggle")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for raw data",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip download and only verify existing file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    logger = get_logger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events_file = output_dir / "events.csv"

    if not args.verify_only:
        # Download from Kaggle
        try:
            download_from_kaggle(DATASET_INFO["kaggle_dataset"], output_dir)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            logger.info("\nTroubleshooting:")
            logger.info("1. Install Kaggle API: pip install kaggle")
            logger.info("2. Setup credentials:")
            logger.info("   - Go to https://www.kaggle.com/settings/account")
            logger.info("   - Click 'Create New API Token'")
            logger.info("   - Place kaggle.json at ~/.kaggle/kaggle.json")
            logger.info("   - chmod 600 ~/.kaggle/kaggle.json")
            logger.info("3. Accept dataset terms at:")
            logger.info(f"   {DATASET_INFO['source']}")
            return

    # Verify file
    logger.info(f"\nVerifying file: {events_file}")

    if not events_file.exists():
        logger.error(f"File not found: {events_file}")
        return

    expected_columns = DATASET_INFO["files"]["events.csv"]["columns"]
    if verify_file(events_file, expected_columns):
        logger.info("✓ File verified successfully!")

        # Show basic stats
        df = pd.read_csv(events_file)
        logger.info("\nDataset statistics:")
        logger.info(f"  Total events: {len(df):,}")
        logger.info(f"  Unique visitors: {df['visitorid'].nunique():,}")
        logger.info(f"  Unique items: {df['itemid'].nunique():,}")
        logger.info(f"  Event types: {df['event'].value_counts().to_dict()}")

        # Date range
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        logger.info(f"  Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
    else:
        logger.error("✗ File verification failed!")


if __name__ == "__main__":
    main()
