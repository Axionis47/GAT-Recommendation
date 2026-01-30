#!/usr/bin/env python3
"""Deploy models to Vertex AI Endpoint with traffic splitting.

This script creates an endpoint and deploys both PyTorch and ONNX
model variants with configurable traffic split for A/B testing.

Usage:
    # Single model deployment
    python scripts/gcp/06_deploy_endpoint.py \
        --project-id your-project \
        --region us-central1 \
        --model-id projects/xxx/locations/xxx/models/xxx

    # A/B test deployment (50/50 split)
    python scripts/gcp/06_deploy_endpoint.py \
        --project-id your-project \
        --region us-central1 \
        --pytorch-model-id projects/xxx/models/pytorch \
        --onnx-model-id projects/xxx/models/onnx \
        --ab-test
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_endpoint(
    project_id: str,
    region: str,
    endpoint_name: str,
    description: str = "",
):
    """Create a Vertex AI endpoint.

    Args:
        project_id: GCP project ID
        region: GCP region
        endpoint_name: Display name for the endpoint
        description: Endpoint description

    Returns:
        Vertex AI Endpoint
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    # Check if endpoint already exists
    existing_endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"'
    )

    if existing_endpoints:
        print(f"Endpoint '{endpoint_name}' already exists, reusing...")
        return existing_endpoints[0]

    print(f"Creating endpoint: {endpoint_name}")
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_name,
        description=description or "GAT-Recommendation inference endpoint",
    )

    print(f"Endpoint created: {endpoint.resource_name}")
    return endpoint


def deploy_model(
    endpoint,
    model_id: str,
    deployed_model_name: str,
    machine_type: str = "n1-standard-4",
    min_replica_count: int = 1,
    max_replica_count: int = 3,
    traffic_percentage: int = 100,
):
    """Deploy a model to an endpoint.

    Args:
        endpoint: Vertex AI Endpoint
        model_id: Model resource ID
        deployed_model_name: Display name for deployed model
        machine_type: Machine type for serving
        min_replica_count: Minimum replicas
        max_replica_count: Maximum replicas
        traffic_percentage: Traffic percentage (0-100)

    Returns:
        Deployed endpoint
    """
    from google.cloud import aiplatform

    model = aiplatform.Model(model_id)

    print(f"\nDeploying model: {model.display_name}")
    print(f"  Machine type: {machine_type}")
    print(f"  Replicas: {min_replica_count}-{max_replica_count}")
    print(f"  Traffic: {traffic_percentage}%")

    endpoint.deploy(
        model=model,
        deployed_model_display_name=deployed_model_name,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        traffic_percentage=traffic_percentage,
    )

    print(f"Model deployed successfully!")
    return endpoint


def deploy_ab_test(
    project_id: str,
    region: str,
    endpoint_name: str,
    pytorch_model_id: str,
    onnx_model_id: str,
    pytorch_traffic: int = 50,
    machine_type: str = "n1-standard-4",
    min_replicas: int = 1,
    max_replicas: int = 3,
):
    """Deploy both models for A/B testing with traffic split.

    Args:
        project_id: GCP project ID
        region: GCP region
        endpoint_name: Endpoint display name
        pytorch_model_id: PyTorch model resource ID
        onnx_model_id: ONNX model resource ID
        pytorch_traffic: Traffic percentage for PyTorch (ONNX gets remainder)
        machine_type: Machine type for serving
        min_replicas: Minimum replicas per model
        max_replicas: Maximum replicas per model

    Returns:
        Deployed endpoint
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    # Create endpoint
    endpoint = create_endpoint(
        project_id=project_id,
        region=region,
        endpoint_name=endpoint_name,
        description="GAT-Recommendation A/B test: PyTorch vs ONNX",
    )

    onnx_traffic = 100 - pytorch_traffic

    # Deploy PyTorch model first with 100% traffic (required for first model)
    print(f"\n{'='*60}")
    print(f"Deploying PyTorch model (initially 100% traffic)")
    print(f"{'='*60}")

    deploy_model(
        endpoint=endpoint,
        model_id=pytorch_model_id,
        deployed_model_name="gat-rec-pytorch",
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100,  # First model must have 100%
    )

    # Deploy ONNX model - this will automatically split traffic
    print(f"\n{'='*60}")
    print(f"Deploying ONNX model ({onnx_traffic}% traffic)")
    print(f"{'='*60}")

    deploy_model(
        endpoint=endpoint,
        model_id=onnx_model_id,
        deployed_model_name="gat-rec-onnx",
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=onnx_traffic,  # This triggers traffic rebalance
    )

    # Print summary
    print(f"\n{'='*60}")
    print("A/B TEST DEPLOYMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Endpoint:      {endpoint.resource_name}")
    print(f"Traffic split: PyTorch={pytorch_traffic}%, ONNX={onnx_traffic}%")
    print(f"\nTest the endpoint:")
    print(f"  curl -X POST '{region}-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict' \\")
    print(f'    -H "Authorization: Bearer $(gcloud auth print-access-token)" \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f"    -d '{{\"instances\": [{{\"session_items\": [1,2,3], \"k\": 10}}]}}'")
    print(f"{'='*60}")

    return endpoint


def main():
    parser = argparse.ArgumentParser(description="Deploy to Vertex AI Endpoint")
    parser.add_argument("--project-id", type=str, required=True, help="GCP project ID")
    parser.add_argument("--region", type=str, default="us-central1", help="GCP region")
    parser.add_argument("--endpoint-name", type=str, default="gat-recommendation-endpoint",
                        help="Endpoint display name")

    # Single model deployment
    parser.add_argument("--model-id", type=str, help="Single model resource ID")

    # A/B test deployment
    parser.add_argument("--ab-test", action="store_true",
                        help="Deploy both models for A/B testing")
    parser.add_argument("--pytorch-model-id", type=str, help="PyTorch model resource ID")
    parser.add_argument("--onnx-model-id", type=str, help="ONNX model resource ID")
    parser.add_argument("--pytorch-traffic", type=int, default=50,
                        help="Traffic percentage for PyTorch model (default: 50)")

    # Machine configuration
    parser.add_argument("--machine-type", type=str, default="n1-standard-4",
                        help="Machine type (default: n1-standard-4)")
    parser.add_argument("--min-replicas", type=int, default=1,
                        help="Minimum replica count (default: 1)")
    parser.add_argument("--max-replicas", type=int, default=3,
                        help="Maximum replica count (default: 3)")

    args = parser.parse_args()

    try:
        from google.cloud import aiplatform

        if args.ab_test:
            # A/B test deployment
            if not args.pytorch_model_id or not args.onnx_model_id:
                parser.error("--ab-test requires both --pytorch-model-id and --onnx-model-id")

            deploy_ab_test(
                project_id=args.project_id,
                region=args.region,
                endpoint_name=args.endpoint_name,
                pytorch_model_id=args.pytorch_model_id,
                onnx_model_id=args.onnx_model_id,
                pytorch_traffic=args.pytorch_traffic,
                machine_type=args.machine_type,
                min_replicas=args.min_replicas,
                max_replicas=args.max_replicas,
            )
        else:
            # Single model deployment
            if not args.model_id:
                parser.error("--model-id required for single model deployment")

            aiplatform.init(project=args.project_id, location=args.region)

            endpoint = create_endpoint(
                project_id=args.project_id,
                region=args.region,
                endpoint_name=args.endpoint_name,
            )

            deploy_model(
                endpoint=endpoint,
                model_id=args.model_id,
                deployed_model_name="gat-recommendation",
                machine_type=args.machine_type,
                min_replica_count=args.min_replicas,
                max_replica_count=args.max_replicas,
            )

            print(f"\n{'='*60}")
            print("DEPLOYMENT COMPLETE")
            print(f"{'='*60}")
            print(f"Endpoint: {endpoint.resource_name}")
            print(f"{'='*60}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
