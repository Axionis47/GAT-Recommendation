"""Profiling utilities for performance measurement."""

import time
from collections.abc import Generator
from contextlib import contextmanager

import torch


@contextmanager
def timer(name: str, results: dict[str, float]) -> Generator[None, None, None]:
    """Context manager for timing code blocks.

    Args:
        name: Name of the timed block.
        results: Dictionary to store timing results.

    Yields:
        None

    Example:
        >>> results = {}
        >>> with timer("data_loading", results):
        ...     data = load_data()
        >>> print(results["data_loading"])
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        results[name] = (end - start) * 1000  # Convert to milliseconds


def measure_memory() -> dict[str, float]:
    """Measure current GPU memory usage.

    Returns:
        Dictionary with memory statistics in MB.
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "max_allocated_mb": max_allocated,
    }


def reset_memory_stats() -> None:
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
