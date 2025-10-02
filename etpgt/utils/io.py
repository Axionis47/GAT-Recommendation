"""I/O utilities for loading and saving data."""

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def save_json(data: dict[str, Any], output_path: str) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save.
        output_path: Output file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(input_path: str) -> dict[str, Any]:
    """Load data from JSON file.

    Args:
        input_path: Input file path.

    Returns:
        Loaded data.
    """
    with open(input_path) as f:
        data: dict[str, Any] = json.load(f)
    return data
