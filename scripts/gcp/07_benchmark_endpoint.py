#!/usr/bin/env python3
"""Benchmark Vertex AI endpoint performance.

Compares latency, throughput, and cost between PyTorch and ONNX variants.

Usage:
    python scripts/gcp/07_benchmark_endpoint.py \
        --project-id your-project \
        --region us-central1 \
        --endpoint-name gat-recommendation-endpoint \
        --num-requests 100
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def generate_test_sessions(
    num_sessions: int,
    num_items: int = 10000,
    session_lengths: tuple = (3, 10),
) -> list[dict]:
    """Generate random test sessions.

    Args:
        num_sessions: Number of sessions to generate
        num_items: Maximum item ID
        session_lengths: Min and max session length

    Returns:
        List of session dicts with session_items and k
    """
    sessions = []
    for _ in range(num_sessions):
        length = np.random.randint(session_lengths[0], session_lengths[1] + 1)
        items = np.random.choice(num_items, size=length, replace=False).tolist()
        sessions.append({"session_items": items, "k": 10})
    return sessions


def benchmark_endpoint(
    endpoint,
    test_sessions: list[dict],
    num_requests: int = 100,
    concurrent_requests: int = 10,
) -> dict:
    """Benchmark an endpoint with test requests.

    Args:
        endpoint: Vertex AI Endpoint object
        test_sessions: List of test sessions
        num_requests: Total number of requests
        concurrent_requests: Number of concurrent requests

    Returns:
        Dictionary of benchmark metrics
    """
    latencies = []
    errors = 0

    def make_request(session: dict) -> tuple:
        """Make a single request and return latency."""
        start = time.time()
        try:
            response = endpoint.predict(instances=[session])
            latency = (time.time() - start) * 1000  # ms
            return latency, None
        except Exception as e:
            return None, str(e)

    print(f"Running {num_requests} requests ({concurrent_requests} concurrent)...")

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        # Submit requests
        futures = []
        for i in range(num_requests):
            session = test_sessions[i % len(test_sessions)]
            futures.append(executor.submit(make_request, session))

        # Collect results with progress
        completed = 0
        for future in as_completed(futures):
            latency, error = future.result()
            if error:
                errors += 1
            else:
                latencies.append(latency)

            completed += 1
            if completed % 10 == 0:
                print(f"  Progress: {completed}/{num_requests}")

    # Calculate metrics
    if latencies:
        metrics = {
            "total_requests": num_requests,
            "successful_requests": len(latencies),
            "failed_requests": errors,
            "error_rate_pct": round(errors / num_requests * 100, 2),
            "latency_p50_ms": round(np.percentile(latencies, 50), 2),
            "latency_p90_ms": round(np.percentile(latencies, 90), 2),
            "latency_p95_ms": round(np.percentile(latencies, 95), 2),
            "latency_p99_ms": round(np.percentile(latencies, 99), 2),
            "latency_mean_ms": round(statistics.mean(latencies), 2),
            "latency_std_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
            "latency_min_ms": round(min(latencies), 2),
            "latency_max_ms": round(max(latencies), 2),
        }

        # Calculate throughput
        total_time_s = sum(latencies) / 1000 / concurrent_requests
        metrics["throughput_rps"] = round(len(latencies) / total_time_s, 2)
    else:
        metrics = {
            "error": "All requests failed",
            "total_requests": num_requests,
            "failed_requests": errors,
        }

    return metrics


def estimate_costs(metrics: dict, machine_type: str = "n1-standard-4") -> dict:
    """Estimate serving costs based on benchmark results.

    Args:
        metrics: Benchmark metrics
        machine_type: Machine type for cost estimation

    Returns:
        Cost estimates
    """
    # Approximate hourly costs for common machine types
    machine_costs = {
        "n1-standard-2": 0.10,
        "n1-standard-4": 0.19,
        "n1-standard-8": 0.38,
        "n1-highmem-2": 0.13,
        "n1-highmem-4": 0.26,
    }

    hourly_cost = machine_costs.get(machine_type, 0.19)
    throughput = metrics.get("throughput_rps", 0)

    if throughput > 0:
        requests_per_hour = throughput * 3600
        cost_per_million = (hourly_cost / requests_per_hour) * 1_000_000
        monthly_cost_24_7 = hourly_cost * 24 * 30
    else:
        requests_per_hour = 0
        cost_per_million = float("inf")
        monthly_cost_24_7 = float("inf")

    return {
        "machine_type": machine_type,
        "hourly_cost_usd": hourly_cost,
        "requests_per_hour": round(requests_per_hour),
        "cost_per_million_requests_usd": round(cost_per_million, 2),
        "monthly_cost_24_7_usd": round(monthly_cost_24_7, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Vertex AI Endpoint")
    parser.add_argument("--project-id", type=str, required=True, help="GCP project ID")
    parser.add_argument("--region", type=str, default="us-central1", help="GCP region")
    parser.add_argument("--endpoint-name", type=str, default="gat-recommendation-endpoint",
                        help="Endpoint display name")
    parser.add_argument("--endpoint-id", type=str, help="Endpoint resource ID (alternative to name)")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Number of requests to run (default: 100)")
    parser.add_argument("--concurrent", type=int, default=10,
                        help="Number of concurrent requests (default: 10)")
    parser.add_argument("--num-items", type=int, default=10000,
                        help="Max item ID for test sessions (default: 10000)")
    parser.add_argument("--machine-type", type=str, default="n1-standard-4",
                        help="Machine type for cost estimation")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()

    try:
        from google.cloud import aiplatform

        aiplatform.init(project=args.project_id, location=args.region)

        # Find endpoint
        if args.endpoint_id:
            endpoint = aiplatform.Endpoint(args.endpoint_id)
        else:
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{args.endpoint_name}"'
            )
            if not endpoints:
                print(f"Endpoint '{args.endpoint_name}' not found")
                return 1
            endpoint = endpoints[0]

        print("=" * 60)
        print("VERTEX AI ENDPOINT BENCHMARK")
        print("=" * 60)
        print(f"Project:        {args.project_id}")
        print(f"Region:         {args.region}")
        print(f"Endpoint:       {endpoint.display_name}")
        print(f"Endpoint ID:    {endpoint.resource_name}")
        print(f"Requests:       {args.num_requests}")
        print(f"Concurrent:     {args.concurrent}")
        print("=" * 60)

        # List deployed models
        deployed_models = endpoint.gca_resource.deployed_models
        traffic_split = endpoint.gca_resource.traffic_split or {}
        print(f"\nDeployed models ({len(deployed_models)}):")
        for dm in deployed_models:
            pct = traffic_split.get(dm.id, "?")
            print(f"  - {dm.display_name}: {pct}% traffic")

        # Generate test data
        print(f"\nGenerating {args.num_requests} test sessions...")
        test_sessions = generate_test_sessions(
            num_sessions=min(100, args.num_requests),
            num_items=args.num_items,
        )

        # Run benchmark
        print("\n" + "=" * 60)
        print("RUNNING BENCHMARK")
        print("=" * 60)

        metrics = benchmark_endpoint(
            endpoint=endpoint,
            test_sessions=test_sessions,
            num_requests=args.num_requests,
            concurrent_requests=args.concurrent,
        )

        # Estimate costs
        costs = estimate_costs(metrics, args.machine_type)

        # Print results
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"\nLatency:")
        print(f"  P50:  {metrics.get('latency_p50_ms', 'N/A')} ms")
        print(f"  P90:  {metrics.get('latency_p90_ms', 'N/A')} ms")
        print(f"  P95:  {metrics.get('latency_p95_ms', 'N/A')} ms")
        print(f"  P99:  {metrics.get('latency_p99_ms', 'N/A')} ms")
        print(f"  Mean: {metrics.get('latency_mean_ms', 'N/A')} ms")

        print(f"\nThroughput:")
        print(f"  Requests/sec: {metrics.get('throughput_rps', 'N/A')}")
        print(f"  Error rate:   {metrics.get('error_rate_pct', 'N/A')}%")

        print(f"\nCost Estimate ({args.machine_type}):")
        print(f"  Hourly:          ${costs['hourly_cost_usd']}")
        print(f"  Monthly (24/7):  ${costs['monthly_cost_24_7_usd']}")
        print(f"  Per 1M requests: ${costs['cost_per_million_requests_usd']}")

        print("=" * 60)

        # Combine results
        results = {
            "endpoint": {
                "name": endpoint.display_name,
                "id": endpoint.resource_name,
                "deployed_models": [
                    {"name": dm.display_name, "traffic": traffic_split.get(dm.id, None)}
                    for dm in deployed_models
                ],
            },
            "config": {
                "num_requests": args.num_requests,
                "concurrent": args.concurrent,
                "num_items": args.num_items,
            },
            "metrics": metrics,
            "costs": costs,
        }

        # Save results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
