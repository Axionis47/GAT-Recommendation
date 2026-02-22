#!/usr/bin/env python3
"""Analyze A/B test results from Vertex AI endpoint logs.

This script queries Cloud Logging to extract latency metrics for each
deployed model variant (PyTorch vs ONNX) and provides insights for
informed decision-making.

Usage:
    python scripts/gcp/08_analyze_ab_results.py \
        --project-id your-project \
        --region us-central1 \
        --endpoint-id 1234567890 \
        --hours 24

Output includes:
    - Latency comparison (p50, p90, p95, p99)
    - Request counts per variant
    - Error rates
    - Statistical significance test
    - Recommendation
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta


def query_endpoint_logs(
    project_id: str,
    endpoint_id: str,
    hours: int = 24,
):
    """Query Cloud Logging for endpoint prediction logs.

    Args:
        project_id: GCP project ID
        endpoint_id: Vertex AI endpoint ID
        hours: Number of hours to analyze

    Returns:
        List of log entries with latency data
    """
    from google.cloud import logging as cloud_logging

    client = cloud_logging.Client(project=project_id)

    # Calculate time filter
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(hours=hours)

    # Build filter
    filter_str = f'''
        resource.type="aiplatform.googleapis.com/Endpoint"
        resource.labels.endpoint_id="{endpoint_id}"
        httpRequest.requestMethod="POST"
        timestamp >= "{start_time.isoformat()}"
        timestamp <= "{end_time.isoformat()}"
    '''

    print(f"Querying logs for last {hours} hours...")
    print(f"  Start: {start_time.isoformat()}")
    print(f"  End:   {end_time.isoformat()}")

    entries = list(client.list_entries(filter_=filter_str, max_results=10000))
    print(f"  Found: {len(entries)} log entries")

    return entries


def parse_latency_from_logs(entries: list) -> dict:
    """Parse latency data from log entries by deployed model.

    Args:
        entries: List of Cloud Logging entries

    Returns:
        Dictionary mapping deployed_model_id to list of latencies
    """
    results = defaultdict(lambda: {
        "latencies": [],
        "errors": 0,
        "requests": 0,
        "display_name": "unknown",
    })

    for entry in entries:
        try:
            # Get deployed model ID from labels
            labels = entry.labels or {}
            deployed_model_id = labels.get("deployed_model_id", "unknown")

            # Get HTTP request info
            http_request = entry.http_request
            if http_request:
                results[deployed_model_id]["requests"] += 1

                # Extract latency from httpRequest.latency
                latency_str = http_request.latency
                if latency_str:
                    # Parse duration string (e.g., "0.123s")
                    if isinstance(latency_str, str):
                        latency_ms = float(latency_str.rstrip('s')) * 1000
                    else:
                        # timedelta or Duration object
                        latency_ms = latency_str.total_seconds() * 1000
                    results[deployed_model_id]["latencies"].append(latency_ms)

                # Check for errors
                status = http_request.status
                if status and status >= 400:
                    results[deployed_model_id]["errors"] += 1

        except Exception:
            continue

    return dict(results)


def get_model_display_names(project_id: str, region: str, endpoint_id: str) -> dict:
    """Get display names for deployed models.

    Args:
        project_id: GCP project ID
        region: GCP region
        endpoint_id: Vertex AI endpoint ID

    Returns:
        Dictionary mapping deployed_model_id to display_name
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    endpoint = aiplatform.Endpoint(endpoint_id)
    names = {}

    for dm in endpoint.gca_resource.deployed_models:
        names[str(dm.id)] = dm.display_name

    return names


def calculate_statistics(latencies: list) -> dict:
    """Calculate latency statistics.

    Args:
        latencies: List of latency values in ms

    Returns:
        Dictionary of statistics
    """
    import numpy as np

    if not latencies:
        return {
            "count": 0,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "mean_ms": None,
            "std_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    arr = np.array(latencies)
    return {
        "count": len(arr),
        "p50_ms": round(np.percentile(arr, 50), 2),
        "p90_ms": round(np.percentile(arr, 90), 2),
        "p95_ms": round(np.percentile(arr, 95), 2),
        "p99_ms": round(np.percentile(arr, 99), 2),
        "mean_ms": round(np.mean(arr), 2),
        "std_ms": round(np.std(arr), 2),
        "min_ms": round(np.min(arr), 2),
        "max_ms": round(np.max(arr), 2),
    }


def perform_statistical_test(latencies_a: list, latencies_b: list) -> dict:
    """Perform statistical significance test (Mann-Whitney U).

    Args:
        latencies_a: Latencies for variant A
        latencies_b: Latencies for variant B

    Returns:
        Test results including p-value and conclusion
    """
    from scipy import stats

    if len(latencies_a) < 2 or len(latencies_b) < 2:
        return {
            "test": "Mann-Whitney U",
            "p_value": None,
            "significant": False,
            "conclusion": "Insufficient data for statistical test",
        }

    # Non-parametric test (doesn't assume normal distribution)
    stat, p_value = stats.mannwhitneyu(latencies_a, latencies_b, alternative='two-sided')

    # Calculate effect size
    mean_a = sum(latencies_a) / len(latencies_a)
    mean_b = sum(latencies_b) / len(latencies_b)
    diff_pct = ((mean_b - mean_a) / mean_a) * 100 if mean_a > 0 else 0

    significant = p_value < 0.05
    if significant:
        if diff_pct < 0:
            conclusion = f"Variant B is {abs(diff_pct):.1f}% faster (statistically significant)"
        else:
            conclusion = f"Variant A is {abs(diff_pct):.1f}% faster (statistically significant)"
    else:
        conclusion = "No statistically significant difference"

    return {
        "test": "Mann-Whitney U",
        "p_value": round(p_value, 4),
        "significant": significant,
        "effect_size_pct": round(diff_pct, 2),
        "conclusion": conclusion,
    }


def generate_recommendation(results: dict, stats_test: dict) -> str:
    """Generate actionable recommendation based on analysis.

    Args:
        results: Per-model results
        stats_test: Statistical test results

    Returns:
        Recommendation string
    """
    models = list(results.keys())
    if len(models) < 2:
        return "Need at least 2 deployed models for A/B comparison"

    model_a, model_b = models[0], models[1]
    stats_a = results[model_a]["stats"]
    stats_b = results[model_b]["stats"]
    name_a = results[model_a]["display_name"]
    name_b = results[model_b]["display_name"]

    if not stats_a["p50_ms"] or not stats_b["p50_ms"]:
        return "Insufficient latency data to make recommendation"

    # Compare p50 latencies
    p50_diff = stats_b["p50_ms"] - stats_a["p50_ms"]
    faster_model = name_a if p50_diff > 0 else name_b
    faster_p50 = stats_a["p50_ms"] if p50_diff > 0 else stats_b["p50_ms"]
    slower_p50 = stats_b["p50_ms"] if p50_diff > 0 else stats_a["p50_ms"]
    speedup = ((slower_p50 - faster_p50) / slower_p50) * 100 if slower_p50 > 0 else 0

    # Error rates
    error_a = results[model_a]["error_rate_pct"]
    error_b = results[model_b]["error_rate_pct"]

    recommendation = []
    recommendation.append("## Recommendation")
    recommendation.append("")

    if stats_test.get("significant"):
        recommendation.append(f"**{faster_model}** shows statistically significant better latency:")
        recommendation.append(f"- P50 latency: {faster_p50}ms vs {slower_p50}ms ({speedup:.1f}% faster)")
        recommendation.append("")
        if speedup > 20:
            recommendation.append("**Strong recommendation:** Shift traffic to the faster model")
        elif speedup > 10:
            recommendation.append("**Moderate recommendation:** Consider shifting traffic to the faster model")
        else:
            recommendation.append("**Mild recommendation:** Difference is small, consider other factors")
    else:
        recommendation.append("No statistically significant difference in latency between models.")
        recommendation.append("Consider:")
        recommendation.append("- Collecting more data (current sample may be too small)")
        recommendation.append("- Evaluating based on other metrics (cost, memory usage)")
        recommendation.append("- Choosing based on operational factors (ease of deployment, maintainability)")

    if error_a > 1 or error_b > 1:
        recommendation.append("")
        recommendation.append(f"**Note:** Error rates are elevated ({name_a}: {error_a}%, {name_b}: {error_b}%)")
        recommendation.append("Investigate errors before making traffic decisions.")

    return "\n".join(recommendation)


def main():
    parser = argparse.ArgumentParser(description="Analyze A/B test results from Vertex AI")
    parser.add_argument("--project-id", type=str, required=True, help="GCP project ID")
    parser.add_argument("--region", type=str, default="us-central1", help="GCP region")
    parser.add_argument("--endpoint-id", type=str, required=True, help="Endpoint resource ID")
    parser.add_argument("--hours", type=int, default=24, help="Hours of data to analyze (default: 24)")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()

    try:
        print("=" * 60)
        print("A/B TEST ANALYSIS")
        print("=" * 60)
        print(f"Project:  {args.project_id}")
        print(f"Region:   {args.region}")
        print(f"Endpoint: {args.endpoint_id}")
        print(f"Period:   Last {args.hours} hours")
        print("=" * 60)

        # Get model display names
        print("\nFetching deployed model info...")
        model_names = get_model_display_names(args.project_id, args.region, args.endpoint_id)
        for mid, name in model_names.items():
            print(f"  {mid}: {name}")

        # Query logs
        print("")
        entries = query_endpoint_logs(args.project_id, args.endpoint_id, args.hours)

        if not entries:
            print("\nNo log entries found. This could mean:")
            print("  - No requests have been made to the endpoint")
            print("  - Logs haven't been indexed yet (try again in a few minutes)")
            print("  - The endpoint ID is incorrect")
            return 1

        # Parse latency data
        print("\nParsing latency data...")
        results = parse_latency_from_logs(entries)

        # Calculate statistics for each model
        for model_id, data in results.items():
            data["display_name"] = model_names.get(model_id, model_id)
            data["stats"] = calculate_statistics(data["latencies"])
            data["error_rate_pct"] = round(
                (data["errors"] / data["requests"] * 100) if data["requests"] > 0 else 0,
                2
            )

        # Print results
        print("\n" + "=" * 60)
        print("LATENCY COMPARISON")
        print("=" * 60)

        for model_id, data in results.items():
            stats = data["stats"]
            print(f"\n{data['display_name']} ({model_id}):")
            print(f"  Requests:   {data['requests']}")
            print(f"  Errors:     {data['errors']} ({data['error_rate_pct']}%)")
            if stats["p50_ms"]:
                print(f"  P50:        {stats['p50_ms']} ms")
                print(f"  P90:        {stats['p90_ms']} ms")
                print(f"  P95:        {stats['p95_ms']} ms")
                print(f"  P99:        {stats['p99_ms']} ms")
                print(f"  Mean:       {stats['mean_ms']} ms (±{stats['std_ms']})")

        # Statistical test
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS")
        print("=" * 60)

        model_ids = list(results.keys())
        if len(model_ids) >= 2:
            lat_a = results[model_ids[0]]["latencies"]
            lat_b = results[model_ids[1]]["latencies"]
            stats_test = perform_statistical_test(lat_a, lat_b)

            print(f"\nTest: {stats_test['test']}")
            print(f"P-value: {stats_test['p_value']}")
            print(f"Significant (p<0.05): {stats_test['significant']}")
            print(f"Effect size: {stats_test.get('effect_size_pct', 'N/A')}%")
            print(f"Conclusion: {stats_test['conclusion']}")

            # Generate recommendation
            print("\n" + "=" * 60)
            recommendation = generate_recommendation(results, stats_test)
            print(recommendation)
            print("=" * 60)
        else:
            stats_test = {}
            print("\nNeed at least 2 models for comparison")

        # Save results
        if args.output:
            output_data = {
                "endpoint_id": args.endpoint_id,
                "analysis_period_hours": args.hours,
                "timestamp": datetime.now(UTC).isoformat(),
                "models": {
                    model_id: {
                        "display_name": data["display_name"],
                        "requests": data["requests"],
                        "errors": data["errors"],
                        "error_rate_pct": data["error_rate_pct"],
                        "latency_stats": data["stats"],
                    }
                    for model_id, data in results.items()
                },
                "statistical_test": stats_test,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
