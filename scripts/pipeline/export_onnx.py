#!/usr/bin/env python3
"""ONNX Export for GAT-Recommendation models.

This module provides ONNX export functionality for model serving:
1. Export trained models to ONNX format
2. Validate ONNX model inference
3. Optimize for production serving

Usage:
    python scripts/pipeline/export_onnx.py --model-path checkpoints/best_model.pt --output model.onnx

Note: Due to dynamic graph structures in GNNs, we export a simplified inference model
that takes pre-computed session embeddings. For full GNN inference, use TorchScript.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: ONNX not installed. Run: pip install onnx onnxruntime")


class SessionRecommender(nn.Module):
    """Simplified model for ONNX export.

    This model takes pre-computed session embeddings and item embeddings,
    then computes top-k recommendations via dot product similarity.

    Note: The full GNN forward pass has dynamic graph structure which is
    challenging to export to ONNX. This wrapper handles the scoring layer
    which is the inference-time bottleneck for serving.
    """

    def __init__(self, item_embeddings: torch.Tensor):
        """Initialize with frozen item embeddings.

        Args:
            item_embeddings: Tensor of shape [num_items, embedding_dim]
        """
        super().__init__()
        self.register_buffer("item_embeddings", item_embeddings)

    def forward(self, session_embedding: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores for all items.

        Args:
            session_embedding: Tensor of shape [batch_size, embedding_dim]

        Returns:
            Scores tensor of shape [batch_size, num_items]
        """
        # Normalize embeddings for cosine similarity
        session_norm = session_embedding / (session_embedding.norm(dim=-1, keepdim=True) + 1e-8)
        item_norm = self.item_embeddings / (self.item_embeddings.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute cosine similarity scores
        scores = torch.matmul(session_norm, item_norm.t())

        return scores


def export_to_onnx(
    item_embeddings: torch.Tensor,
    output_path: Path,
    embedding_dim: int,
    opset_version: int = 14,
):
    """Export the scoring model to ONNX format.

    Args:
        item_embeddings: Pre-trained item embeddings [num_items, embedding_dim]
        output_path: Path to save ONNX model
        embedding_dim: Embedding dimension
        opset_version: ONNX opset version
    """
    if not ONNX_AVAILABLE:
        print("ONNX not available. Install with: pip install onnx onnxruntime")
        return False

    print(f"\n{'='*60}")
    print("EXPORTING MODEL TO ONNX")
    print(f"{'='*60}")

    # Create scoring model
    model = SessionRecommender(item_embeddings)
    model.eval()

    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, embedding_dim)

    # Export to ONNX
    print(f"Exporting to: {output_path}")
    print(f"  Item embeddings: {item_embeddings.shape}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Opset version: {opset_version}")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["session_embedding"],
        output_names=["item_scores"],
        dynamic_axes={
            "session_embedding": {0: "batch_size"},
            "item_scores": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"\n✓ ONNX model exported successfully!")

    # Validate ONNX model
    print("\nValidating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validation passed!")

    return True


def validate_onnx_inference(
    onnx_path: Path,
    item_embeddings: torch.Tensor,
    embedding_dim: int,
):
    """Validate ONNX model inference matches PyTorch.

    Args:
        onnx_path: Path to ONNX model
        item_embeddings: Original item embeddings
        embedding_dim: Embedding dimension
    """
    if not ONNX_AVAILABLE:
        return

    print("\nValidating ONNX inference...")

    # PyTorch model
    torch_model = SessionRecommender(item_embeddings)
    torch_model.eval()

    # ONNX Runtime session
    ort_session = ort.InferenceSession(str(onnx_path))

    # Test input
    test_input = torch.randn(4, embedding_dim)

    # PyTorch inference
    with torch.no_grad():
        torch_output = torch_model(test_input).numpy()

    # ONNX inference
    ort_inputs = {"session_embedding": test_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    max_diff = abs(torch_output - ort_output).max()
    print(f"  Max difference: {max_diff:.6e}")

    if max_diff < 1e-5:
        print("✓ ONNX inference matches PyTorch!")
    else:
        print(f"⚠ Warning: Output difference > 1e-5")

    # Benchmark
    import time
    n_runs = 100

    # PyTorch timing
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = torch_model(test_input)
    torch_time = (time.time() - start) / n_runs * 1000

    # ONNX timing
    start = time.time()
    for _ in range(n_runs):
        _ = ort_session.run(None, ort_inputs)
    onnx_time = (time.time() - start) / n_runs * 1000

    print(f"\nInference benchmark (batch_size=4):")
    print(f"  PyTorch: {torch_time:.3f} ms")
    print(f"  ONNX:    {onnx_time:.3f} ms")
    print(f"  Speedup: {torch_time/onnx_time:.2f}x")


def export_full_model(
    model_path: Path,
    output_path: Path,
    model_class: str = "graph_transformer_optimized",
):
    """Export a full trained model to ONNX.

    Args:
        model_path: Path to PyTorch checkpoint
        output_path: Path to save ONNX model
        model_class: Model class name
    """
    from etpgt.model import (
        create_graph_transformer_optimized,
        create_graphsage,
        create_gat,
    )

    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Get config from checkpoint
    config = checkpoint.get("config", {})
    num_items = config.get("num_items", 100)
    embedding_dim = config.get("embedding_dim", 64)
    hidden_dim = config.get("hidden_dim", 64)

    # Create model
    if model_class == "graph_transformer_optimized":
        model = create_graph_transformer_optimized(
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            use_laplacian_pe=False,
        )
    elif model_class == "graphsage":
        model = create_graphsage(
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get item embeddings
    item_embeddings = model.get_item_embeddings()
    print(f"Item embeddings shape: {item_embeddings.shape}")

    # Export scoring layer to ONNX
    export_to_onnx(
        item_embeddings=item_embeddings,
        output_path=output_path,
        embedding_dim=hidden_dim,
    )

    # Validate
    validate_onnx_inference(
        onnx_path=output_path,
        item_embeddings=item_embeddings,
        embedding_dim=hidden_dim,
    )


def create_demo_export(output_dir: Path, num_items: int = 1000, embedding_dim: int = 64):
    """Create a demo ONNX export for testing.

    Args:
        output_dir: Output directory
        num_items: Number of items
        embedding_dim: Embedding dimension
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create random item embeddings (simulating trained embeddings)
    item_embeddings = torch.randn(num_items, embedding_dim)

    # Export
    onnx_path = output_dir / "session_recommender.onnx"
    export_to_onnx(
        item_embeddings=item_embeddings,
        output_path=onnx_path,
        embedding_dim=embedding_dim,
    )

    # Validate
    validate_onnx_inference(
        onnx_path=onnx_path,
        item_embeddings=item_embeddings,
        embedding_dim=embedding_dim,
    )

    # Save metadata
    metadata = {
        "num_items": num_items,
        "embedding_dim": embedding_dim,
        "model_type": "SessionRecommender",
        "note": "Demo export - replace with trained embeddings for production",
    }

    import json
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDemo export saved to: {output_dir}")
    print("Files:")
    print(f"  - {onnx_path}")
    print(f"  - {output_dir / 'model_metadata.json'}")


def export_production_model(
    model_path: Path,
    output_dir: Path,
    model_class: str = "graph_transformer_optimized",
):
    """Export production-ready ONNX model and embeddings for Vertex AI.

    This exports:
    1. Item embeddings as .npy file (for session embedding computation)
    2. ONNX scoring model (for inference)
    3. Metadata JSON with model configuration

    Args:
        model_path: Path to trained PyTorch checkpoint
        output_dir: Output directory for artifacts
        model_class: Model class name
    """
    import json
    import numpy as np
    from datetime import datetime

    from etpgt.model import (
        create_graph_transformer_optimized,
        create_graphsage,
        create_gat,
    )

    print(f"\n{'='*60}")
    print("EXPORTING PRODUCTION MODEL FOR VERTEX AI")
    print(f"{'='*60}")
    print(f"Checkpoint: {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"Model class: {model_class}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Infer config from state dict
    embedding_key = "item_embedding.weight"
    if embedding_key in state_dict:
        num_items = state_dict[embedding_key].shape[0]
        embedding_dim = state_dict[embedding_key].shape[1]
    else:
        config = checkpoint.get("config", {})
        num_items = config.get("num_items", 1000)
        embedding_dim = config.get("embedding_dim", 256)

    hidden_dim = embedding_dim  # Usually same

    print(f"\nModel config:")
    print(f"  num_items: {num_items}")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  hidden_dim: {hidden_dim}")

    # Create model
    model_factories = {
        "graph_transformer_optimized": lambda: create_graph_transformer_optimized(
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            use_laplacian_pe=False,
        ),
        "graphsage": lambda: create_graphsage(
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ),
        "gat": lambda: create_gat(
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ),
    }

    if model_class not in model_factories:
        raise ValueError(f"Unknown model class: {model_class}. Choose from: {list(model_factories.keys())}")

    model = model_factories[model_class]()

    # Filter out cached PE from state dict
    filtered_state = {k: v for k, v in state_dict.items() if "_cached_pe" not in k}
    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    # Extract item embeddings
    with torch.no_grad():
        item_embeddings = model.get_item_embeddings()
    print(f"\nItem embeddings shape: {item_embeddings.shape}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save item embeddings as numpy array
    embeddings_path = output_dir / "item_embeddings.npy"
    np.save(embeddings_path, item_embeddings.detach().numpy())
    embeddings_size = embeddings_path.stat().st_size / 1024 / 1024
    print(f"\nSaved embeddings: {embeddings_path} ({embeddings_size:.1f} MB)")

    # Export ONNX scoring model
    onnx_path = output_dir / "session_recommender.onnx"
    export_to_onnx(
        item_embeddings=item_embeddings,
        output_path=onnx_path,
        embedding_dim=hidden_dim,
    )
    onnx_size = onnx_path.stat().st_size / 1024 / 1024
    print(f"Saved ONNX model: {onnx_path} ({onnx_size:.1f} MB)")

    # Validate inference
    validate_onnx_inference(
        onnx_path=onnx_path,
        item_embeddings=item_embeddings,
        embedding_dim=hidden_dim,
    )

    # Save metadata
    metadata = {
        "model_class": model_class,
        "num_items": num_items,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "checkpoint_path": str(model_path),
        "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
        "files": {
            "onnx_model": "session_recommender.onnx",
            "item_embeddings": "item_embeddings.npy",
        },
        "metrics": {
            "recall@10": checkpoint.get("best_val_metric", 0.0),
        },
        "exported_at": datetime.utcnow().isoformat() + "Z",
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    print(f"\n{'='*60}")
    print("PRODUCTION EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"\nArtifacts in {output_dir}:")
    print(f"  - session_recommender.onnx ({onnx_size:.1f} MB)")
    print(f"  - item_embeddings.npy ({embeddings_size:.1f} MB)")
    print(f"  - model_metadata.json")
    print(f"\nTotal size: {onnx_size + embeddings_size:.1f} MB")
    print(f"\nNext step: Upload to GCS with:")
    print(f"  bash scripts/gcp/04_upload_model_artifacts.sh")


def main():
    parser = argparse.ArgumentParser(description="ONNX Export for GAT-Recommendation")
    parser.add_argument("--model-path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--output-dir", type=str, help="Output directory for production export")
    parser.add_argument("--model-class", type=str, default="graph_transformer_optimized",
                        choices=["graph_transformer_optimized", "graphsage", "gat"])
    parser.add_argument("--demo", action="store_true", help="Create demo export")
    parser.add_argument("--production", action="store_true",
                        help="Production export for Vertex AI (includes embeddings)")
    parser.add_argument("--num-items", type=int, default=1000, help="Number of items for demo")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent

    if args.demo:
        output_dir = project_root / "exports/onnx"
        create_demo_export(
            output_dir=output_dir,
            num_items=args.num_items,
            embedding_dim=args.embedding_dim,
        )
    elif args.production:
        # Production export for Vertex AI
        model_path = args.model_path or str(project_root / "checkpoints/graph_transformer_optimized_best.pt")
        output_dir = args.output_dir or str(project_root / "exports/onnx/production")
        export_production_model(
            model_path=Path(model_path),
            output_dir=Path(output_dir),
            model_class=args.model_class,
        )
    elif args.model_path:
        export_full_model(
            model_path=Path(args.model_path),
            output_path=Path(args.output),
            model_class=args.model_class,
        )
    else:
        print("ONNX Export for GAT-Recommendation")
        print("")
        print("Usage:")
        print("  Demo export:       python export_onnx.py --demo")
        print("  Production export: python export_onnx.py --production")
        print("  Custom export:     python export_onnx.py --model-path checkpoint.pt --output model.onnx")
        print("")
        print("Options:")
        print("  --production        Export for Vertex AI (includes embeddings as .npy)")
        print("  --model-path PATH   Path to PyTorch checkpoint")
        print("  --output-dir DIR    Output directory for production export")
        print("  --model-class NAME  Model class (graph_transformer_optimized, graphsage, gat)")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
