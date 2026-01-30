#!/usr/bin/env python3
"""Register model with Vertex AI Model Registry.

This script registers both PyTorch and ONNX model variants
with Vertex AI for managed deployment.

Usage:
    python scripts/gcp/05_register_vertex_model.py \
        --project-id your-project \
        --region us-central1 \
        --model-name gat-rec-pytorch-v1 \
        --container-image us-central1-docker.pkg.dev/project/repo/etpgt-serve:latest \
        --artifact-uri gs://bucket/models/serving/v1/pytorch \
        --variant pytorch
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def register_model(
    project_id: str,
    region: str,
    model_name: str,
    container_image_uri: str,
    artifact_uri: str,
    variant: str = "pytorch",
    description: str = "",
    labels: dict = None,
):
    """Register a model with Vertex AI.

    Args:
        project_id: GCP project ID
        region: GCP region
        model_name: Display name for the model
        container_image_uri: URI of the serving container image
        artifact_uri: GCS URI of model artifacts
        variant: Model variant ("pytorch" or "onnx")
        description: Model description
        labels: Labels for the model

    Returns:
        Registered model resource
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    print("=" * 60)
    print("REGISTERING MODEL WITH VERTEX AI")
    print("=" * 60)
    print(f"Project:   {project_id}")
    print(f"Region:    {region}")
    print(f"Model:     {model_name}")
    print(f"Variant:   {variant}")
    print(f"Container: {container_image_uri}")
    print(f"Artifacts: {artifact_uri}")
    print("=" * 60)

    # Set environment variables for the container
    if variant == "onnx":
        env_vars = {
            "MODEL_PATH": "/app/model/session_recommender.onnx",
            "EMBEDDINGS_PATH": "/app/model/item_embeddings.npy",
            "GCS_MODEL_URI": artifact_uri,
            "INFERENCE_MODE": "onnx",
        }
    else:
        env_vars = {
            "MODEL_PATH": "/app/model/model.pt",
            "GCS_MODEL_URI": artifact_uri,
            "INFERENCE_MODE": "pytorch",
        }

    # Default labels
    if labels is None:
        labels = {}
    labels.update({
        "model_type": "gat-recommendation",
        "variant": variant,
        "framework": variant,
    })

    # Register model
    print("\nUploading model to Vertex AI Model Registry...")

    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=container_image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        serving_container_environment_variables=env_vars,
        description=description or f"GAT-Recommendation {variant} model",
        labels=labels,
    )

    print("\n" + "=" * 60)
    print("MODEL REGISTERED SUCCESSFULLY")
    print("=" * 60)
    print(f"Model ID:      {model.resource_name}")
    print(f"Model Name:    {model.display_name}")
    print(f"Model Version: {model.version_id}")
    print("=" * 60)

    return model


def main():
    parser = argparse.ArgumentParser(description="Register Vertex AI Model")
    parser.add_argument("--project-id", type=str, required=True, help="GCP project ID")
    parser.add_argument("--region", type=str, default="us-central1", help="GCP region")
    parser.add_argument("--model-name", type=str, required=True, help="Model display name")
    parser.add_argument("--container-image", type=str, required=True,
                        help="Container image URI (Artifact Registry)")
    parser.add_argument("--artifact-uri", type=str, required=True,
                        help="GCS URI for model artifacts")
    parser.add_argument("--variant", type=str, choices=["pytorch", "onnx"], default="pytorch",
                        help="Model variant")
    parser.add_argument("--description", type=str, default="", help="Model description")
    args = parser.parse_args()

    try:
        model = register_model(
            project_id=args.project_id,
            region=args.region,
            model_name=args.model_name,
            container_image_uri=args.container_image,
            artifact_uri=args.artifact_uri,
            variant=args.variant,
            description=args.description,
        )

        print(f"\nNext step: Deploy to endpoint")
        print(f"  python scripts/gcp/06_deploy_endpoint.py \\")
        print(f"      --project-id {args.project_id} \\")
        print(f"      --region {args.region} \\")
        print(f"      --model-id {model.resource_name}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
