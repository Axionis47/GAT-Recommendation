#!/usr/bin/env python3
"""Model quality gate for CI/CD deployment pipelines.

Validates a trained model checkpoint against configurable metric thresholds
before allowing deployment. Exits 0 (pass) or 1 (fail).

Usage:
    # With defaults (uses configs/quality_thresholds.yaml)
    python scripts/pipeline/model_quality_gate.py \\
        --checkpoint checkpoints/graph_transformer_optimized_best.pt \\
        --test-data data/processed/test.csv \\
        --graph-edges data/processed/graph_edges.csv

    # With custom thresholds
    python scripts/pipeline/model_quality_gate.py \\
        --checkpoint checkpoints/best_model.pt \\
        --min-recall-10 0.30 --min-ndcg-10 0.25

    # Quick validation (no eval data, just artifact checks)
    python scripts/pipeline/model_quality_gate.py \\
        --checkpoint checkpoints/best_model.pt \\
        --artifact-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from etpgt.model import (
    create_gat,
    create_graph_transformer,
    create_graph_transformer_optimized,
    create_graphsage,
)

# Model factory registry (same as evaluate_local.py)
MODEL_FACTORIES = {
    "graphsage": create_graphsage,
    "gat": create_gat,
    "graph_transformer": create_graph_transformer,
    "graph_transformer_optimized": create_graph_transformer_optimized,
}


def load_thresholds(config_path: Path | None, cli_overrides: dict) -> dict:
    """Load quality thresholds from config file, with CLI overrides."""
    defaults = {
        "recall_at_10": 0.30,
        "recall_at_20": 0.35,
        "ndcg_at_10": 0.25,
        "ndcg_at_20": 0.28,
        "max_size_mb": 500,
        "min_size_mb": 0.1,
        "nan_check": True,
    }

    # Load from YAML if exists
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        gate = config.get("quality_gate", {})
        metrics = gate.get("metrics", {})
        checks = gate.get("model_checks", {})
        latency_slo = gate.get("latency_slo", {})
        defaults.update(metrics)
        defaults.update(checks)
        defaults.update(latency_slo)

    # CLI overrides take precedence
    for key, value in cli_overrides.items():
        if value is not None:
            defaults[key] = value

    return defaults


def validate_artifact(checkpoint_path: Path, thresholds: dict) -> list[str]:
    """Validate model artifact integrity. Returns list of failures."""
    failures = []

    # Check file exists
    if not checkpoint_path.exists():
        failures.append(f"Checkpoint not found: {checkpoint_path}")
        return failures

    # Check file size
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    if size_mb < thresholds["min_size_mb"]:
        failures.append(
            f"Checkpoint too small: {size_mb:.2f}MB < {thresholds['min_size_mb']}MB"
        )
    if size_mb > thresholds["max_size_mb"]:
        failures.append(
            f"Checkpoint too large: {size_mb:.2f}MB > {thresholds['max_size_mb']}MB"
        )

    # Try loading checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        failures.append(f"Failed to load checkpoint: {e}")
        return failures

    # Check expected keys
    if "model_state_dict" not in checkpoint:
        failures.append("Checkpoint missing 'model_state_dict' key")
        return failures

    # NaN check on parameters
    if thresholds.get("nan_check", True):
        state_dict = checkpoint["model_state_dict"]
        for name, tensor in state_dict.items():
            if torch.is_floating_point(tensor) and torch.isnan(tensor).any():
                failures.append(f"NaN detected in parameter: {name}")

    return failures


def validate_latency(
    model,
    thresholds: dict,
    num_warmup: int = 5,
    num_runs: int = 50,
) -> tuple[dict, list[str]]:
    """Time inference on a dummy batch to check latency SLO.

    Args:
        model: Loaded model in eval mode.
        thresholds: Dict with optional p50_ms, p95_ms, p99_ms keys.
        num_warmup: Warmup iterations (not timed).
        num_runs: Timed iterations.

    Returns:
        (latency_metrics, list_of_failures).
    """
    from torch_geometric.data import Batch, Data

    dummy = Data(
        x=torch.tensor([1, 2, 3]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
    )
    batch = Batch.from_data_list([dummy])

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            model(batch)

        # Timed runs
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model(batch)
            latencies.append((time.perf_counter() - start) * 1000)

    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[len(sorted_latencies) // 2]
    p95 = sorted_latencies[int(0.95 * len(sorted_latencies))]
    p99 = sorted_latencies[int(0.99 * len(sorted_latencies))]

    latency_metrics = {
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "latency_mean_ms": round(sum(latencies) / len(latencies), 2),
    }

    failures = []
    slo_p50 = thresholds.get("p50_ms")
    slo_p95 = thresholds.get("p95_ms")
    slo_p99 = thresholds.get("p99_ms")

    if slo_p50 is not None and p50 > slo_p50:
        failures.append(f"Latency P50: {p50:.1f}ms > SLO {slo_p50}ms")
    if slo_p95 is not None and p95 > slo_p95:
        failures.append(f"Latency P95: {p95:.1f}ms > SLO {slo_p95}ms")
    if slo_p99 is not None and p99 > slo_p99:
        failures.append(f"Latency P99: {p99:.1f}ms > SLO {slo_p99}ms")

    return latency_metrics, failures


def validate_metrics(
    checkpoint_path: Path,
    test_data_path: Path,
    graph_edges_path: Path,
    model_name: str,
    thresholds: dict,
    num_samples: int | None = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> tuple[dict, list[str]]:
    """Evaluate model and check against metric thresholds.

    Returns (metrics_dict, list_of_failures).
    """
    import pandas as pd
    from torch.utils.data import DataLoader

    from etpgt.train.dataloader import SessionDataset, collate_fn
    from etpgt.utils.metrics import compute_ndcg_at_k, compute_recall_at_k

    failures = []

    # Load data
    test_df = pd.read_csv(test_data_path)
    edges_df = pd.read_csv(graph_edges_path)
    num_items = (
        max(test_df["itemid"].max(), edges_df["item_i"].max(), edges_df["item_j"].max())
        + 1
    )

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Detect model type from checkpoint if not specified
    if model_name not in MODEL_FACTORIES:
        failures.append(f"Unknown model type: {model_name}")
        return {}, failures

    # Determine model kwargs from checkpoint metadata
    model_kwargs = checkpoint.get("model_config", {})
    if not model_kwargs:
        # Fallback: use defaults matching evaluate_local.py
        model_kwargs = {
            "embedding_dim": 256,
            "hidden_dim": 256,
            "num_layers": 2,
            "num_heads": 2,
            "dropout": 0.1,
            "readout_type": "mean",
        }
        if model_name in ["graph_transformer", "graph_transformer_optimized"]:
            model_kwargs["use_laplacian_pe"] = True
            model_kwargs["laplacian_k"] = 16
        if model_name == "graph_transformer":
            model_kwargs["use_ffn"] = True
            model_kwargs["num_layers"] = 3
            model_kwargs["num_heads"] = 4
        elif model_name == "graph_transformer_optimized":
            model_kwargs["use_ffn"] = False

    factory = MODEL_FACTORIES[model_name]
    model = factory(num_items=num_items, **model_kwargs)

    # Handle cached PE key
    cached_pe_key = "laplacian_pe._cached_pe"
    cached_pe = state_dict.pop(cached_pe_key) if cached_pe_key in state_dict else None

    model.load_state_dict(state_dict, strict=False)

    if cached_pe is not None and hasattr(model, "laplacian_pe"):
        model.laplacian_pe._cached_pe = cached_pe

    model.to(device)
    model.eval()

    # Create dataset
    dataset = SessionDataset(
        sessions_path=test_data_path,
        graph_edges_path=graph_edges_path,
        num_negatives=1,
        max_session_length=50,
    )

    if num_samples:
        indices = list(range(min(num_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Evaluate
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            session_emb = model(batch)
            item_emb = model.get_item_embeddings()
            scores = torch.matmul(session_emb, item_emb.t())
            _, top_k = torch.topk(scores, k=20, dim=1)
            all_predictions.append(top_k.cpu())
            all_targets.append(batch.target_item.cpu())

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    metrics = {
        "recall_at_10": float(compute_recall_at_k(predictions, targets, 10)),
        "recall_at_20": float(compute_recall_at_k(predictions, targets, 20)),
        "ndcg_at_10": float(compute_ndcg_at_k(predictions, targets, 10)),
        "ndcg_at_20": float(compute_ndcg_at_k(predictions, targets, 20)),
    }

    # Check thresholds
    threshold_keys = ["recall_at_10", "recall_at_20", "ndcg_at_10", "ndcg_at_20"]
    for key in threshold_keys:
        threshold = thresholds.get(key)
        if threshold is not None and metrics[key] < threshold:
            failures.append(
                f"{key}: {metrics[key]:.4f} < threshold {threshold:.4f}"
            )

    return metrics, failures


def main():
    parser = argparse.ArgumentParser(
        description="Model quality gate for deployment pipelines"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_transformer_optimized",
        choices=list(MODEL_FACTORIES.keys()),
        help="Model type (default: graph_transformer_optimized)",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=None,
        help="Path to test data CSV (skip metric eval if not provided)",
    )
    parser.add_argument(
        "--graph-edges",
        type=Path,
        default=None,
        help="Path to graph edges CSV",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/quality_thresholds.yaml"),
        help="Path to threshold config (default: configs/quality_thresholds.yaml)",
    )
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help="Only validate artifact integrity (skip metric evaluation)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Evaluation batch size"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write results JSON to this path",
    )

    # CLI threshold overrides
    parser.add_argument("--min-recall-10", type=float, default=None)
    parser.add_argument("--min-recall-20", type=float, default=None)
    parser.add_argument("--min-ndcg-10", type=float, default=None)
    parser.add_argument("--min-ndcg-20", type=float, default=None)
    parser.add_argument(
        "--max-latency-p95",
        type=float,
        default=None,
        help="Override P95 latency SLO in ms",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip latency SLO check",
    )

    args = parser.parse_args()

    # Load thresholds
    cli_overrides = {
        "recall_at_10": args.min_recall_10,
        "recall_at_20": args.min_recall_20,
        "ndcg_at_10": args.min_ndcg_10,
        "ndcg_at_20": args.min_ndcg_20,
        "p95_ms": args.max_latency_p95,
    }
    thresholds = load_thresholds(args.config, cli_overrides)

    print("=" * 60)
    print("MODEL QUALITY GATE")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model}")
    print(f"Config:     {args.config}")
    print()

    all_failures = []
    metrics = {}

    # Phase 1: Artifact validation
    print("Phase 1: Artifact validation...")
    artifact_failures = validate_artifact(args.checkpoint, thresholds)
    all_failures.extend(artifact_failures)

    if artifact_failures:
        for f in artifact_failures:
            print(f"  FAIL: {f}")
    else:
        size_mb = args.checkpoint.stat().st_size / (1024 * 1024)
        print(f"  PASS: Checkpoint valid ({size_mb:.1f}MB, no NaN)")

    # Phase 2: Metric evaluation (if data provided and no artifact failures)
    if not args.artifact_only and not artifact_failures:
        test_data = args.test_data or Path("data/processed/test.csv")
        graph_edges = args.graph_edges or Path("data/processed/graph_edges.csv")

        if test_data.exists() and graph_edges.exists():
            print("\nPhase 2: Metric evaluation...")
            print(f"  Thresholds: recall@10>={thresholds['recall_at_10']:.2f}, "
                  f"ndcg@10>={thresholds['ndcg_at_10']:.2f}")

            metrics, metric_failures = validate_metrics(
                checkpoint_path=args.checkpoint,
                test_data_path=test_data,
                graph_edges_path=graph_edges,
                model_name=args.model,
                thresholds=thresholds,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
            )
            all_failures.extend(metric_failures)

            for key, value in metrics.items():
                threshold = thresholds.get(key)
                status = "PASS" if (threshold is None or value >= threshold) else "FAIL"
                threshold_str = f" (>= {threshold:.4f})" if threshold else ""
                print(f"  {status}: {key} = {value:.4f}{threshold_str}")
        else:
            print("\nPhase 2: SKIPPED (test data not found)")
            print(f"  Expected: {test_data}")

    # Phase 3: Latency SLO check (if not skipped and no artifact failures)
    latency_metrics = {}
    if not args.skip_latency and not artifact_failures:
        has_latency_slo = any(
            thresholds.get(k) is not None for k in ["p50_ms", "p95_ms", "p99_ms"]
        )
        if has_latency_slo:
            print("\nPhase 3: Latency SLO check...")

            # Build a minimal model for latency testing
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model_kwargs = checkpoint.get("model_config", {})

            # Infer num_items and embedding_dim from state dict
            embedding_key = "item_embedding.weight"
            num_items_latency = state_dict[embedding_key].shape[0] if embedding_key in state_dict else 100
            embedding_dim_latency = state_dict[embedding_key].shape[1] if embedding_key in state_dict else 256

            if not model_kwargs:
                model_kwargs = {
                    "embedding_dim": embedding_dim_latency,
                    "hidden_dim": embedding_dim_latency,
                    "num_layers": 2,
                    "num_heads": 2,
                    "dropout": 0.1,
                    "readout_type": "mean",
                }
                if args.model in ["graph_transformer", "graph_transformer_optimized"]:
                    model_kwargs["use_laplacian_pe"] = False
                if args.model == "graph_transformer":
                    model_kwargs["use_ffn"] = True
                    model_kwargs["num_layers"] = 3
                    model_kwargs["num_heads"] = 4
                elif args.model == "graph_transformer_optimized":
                    model_kwargs["use_ffn"] = False

            factory = MODEL_FACTORIES[args.model]
            latency_model = factory(num_items=num_items_latency, **model_kwargs)
            filtered = {k: v for k, v in state_dict.items() if "_cached_pe" not in k}
            latency_model.load_state_dict(filtered, strict=False)

            latency_metrics, latency_failures = validate_latency(
                latency_model, thresholds
            )
            all_failures.extend(latency_failures)

            for key, value in latency_metrics.items():
                slo_key = key.replace("latency_", "").replace("_ms", "_ms")
                print(f"  {key}: {value}ms")
            if latency_failures:
                for f in latency_failures:
                    print(f"  FAIL: {f}")
            else:
                print("  PASS: Latency within SLO")
        else:
            print("\nPhase 3: SKIPPED (no latency SLO configured)")

    # Result
    print()
    print("=" * 60)

    result = {
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "thresholds": {k: v for k, v in thresholds.items() if isinstance(v, (int, float))},
        "metrics": {**metrics, **latency_metrics},
        "failures": all_failures,
        "passed": len(all_failures) == 0,
    }

    if all_failures:
        print(f"QUALITY GATE: FAILED ({len(all_failures)} issue(s))")
        for f in all_failures:
            print(f"  - {f}")
    else:
        print("QUALITY GATE: PASSED")

    print("=" * 60)

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults written to {args.output}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
