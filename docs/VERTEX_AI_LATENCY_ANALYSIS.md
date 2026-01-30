# Vertex AI Endpoint Latency Analysis & Optimization Guide

## Deployment Summary

### Infrastructure Created

| Component | Status | Details |
|-----------|--------|---------|
| GCS Artifacts | ✅ Complete | `gs://plotpointe-etpgt-data/models/serving/20260127-154130/` |
| Docker Images | ✅ Complete | PyTorch & ONNX containers in Artifact Registry |
| Vertex AI Models | ✅ Registered | 4 model versions (PyTorch & ONNX, v1 & v2) |
| Vertex AI Endpoint | ✅ Created | `gat-recommendation-endpoint` (ID: 7562422284145655808) |
| A/B Test Logging | ✅ Created | `08_analyze_ab_results.py` |

### Model Artifacts

| Artifact | Size | Location |
|----------|------|----------|
| PyTorch Checkpoint | 1.4 GB | `gs://plotpointe-etpgt-data/models/serving/20260127-154130/pytorch/model.pt` |
| ONNX Model | 456 MB | `gs://plotpointe-etpgt-data/models/serving/20260127-154130/onnx/session_recommender.onnx` |
| Item Embeddings | 456 MB | `gs://plotpointe-etpgt-data/models/serving/20260127-154130/onnx/item_embeddings.npy` |

## Latency Benchmark (Local)

From the ONNX export benchmarks:

| Metric | PyTorch | ONNX | Improvement |
|--------|---------|------|-------------|
| Inference Time | 51.7 ms | 5.5 ms | **9.37x faster** |
| Model Size | 1.4 GB | 456 MB | 3x smaller |
| Cold Start | ~30s | ~10s | 3x faster |

## Expected Vertex AI Latencies

Based on infrastructure configuration (n1-standard-4):

| Metric | PyTorch (estimated) | ONNX (estimated) |
|--------|---------------------|------------------|
| P50 Latency | 80-120 ms | 20-40 ms |
| P95 Latency | 150-200 ms | 40-80 ms |
| P99 Latency | 200-300 ms | 60-100 ms |
| Cold Start | 60-90s | 30-45s |

*Note: Actual latencies include network overhead, GCS model download at startup, and serialization.*

## Latency Improvement Recommendations

### 1. Model-Level Optimizations

#### A. Reduce Embedding Dimensions (High Impact)
```python
# Current: 256 dimensions, 466K items = 456 MB embeddings
# Optimized: 128 dimensions = 228 MB embeddings

# In training config:
embedding_dim: 128  # Down from 256
```
**Expected improvement:** 30-40% faster inference, 50% smaller model

#### B. Quantize ONNX Model (High Impact)
```python
# Add to export_onnx.py
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="session_recommender.onnx",
    model_output="session_recommender_int8.onnx",
    weight_type=QuantType.QInt8,
)
```
**Expected improvement:** 2-3x faster inference, 4x smaller model

#### C. Use Float16 Embeddings (Medium Impact)
```python
# Save embeddings as float16 instead of float32
np.save(embeddings_path, item_embeddings.numpy().astype(np.float16))
```
**Expected improvement:** 50% memory reduction, 20-30% faster

### 2. Infrastructure Optimizations

#### A. Use GPU Instances (High Impact)
```bash
# In deploy command
--machine-type n1-standard-4-nvidia-t4

# Or for high throughput
--machine-type a2-highgpu-1g  # A100 GPU
```
**Expected improvement:** 5-10x faster for PyTorch, 2-3x for ONNX

#### B. Use Smaller Machine with ONNX (Cost Optimization)
```bash
# ONNX doesn't need as much CPU
--machine-type n1-standard-2  # Half the cost

# Cost: ~$72/month vs $137/month (47% savings)
```

#### C. Pre-warm with Min Replicas
```python
# Already configured:
min_replicas=1  # Prevents cold starts

# For high traffic:
min_replicas=2  # Better availability
```

### 3. Serving Optimizations

#### A. Batch Requests (High Impact)
```python
# Add batching to vertex_app.py
@app.post("/predict")
async def predict(request: VertexPredictRequest):
    # Process all instances in one batch
    all_embeddings = np.stack([
        compute_session_embedding(inst["session_items"])
        for inst in request.instances
    ])
    # Single matrix multiplication
    scores = np.matmul(all_embeddings, item_embeddings.T)
```
**Expected improvement:** 2-5x throughput for batch requests

#### B. Cache Common Sessions (Medium Impact)
```python
# Add LRU cache for frequent sessions
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_cached_recommendations(session_tuple: tuple, k: int):
    return compute_recommendations(list(session_tuple), k)
```
**Expected improvement:** 10-100x for cache hits

#### C. Use Connection Pooling
```python
# In client code
import httpx

# Reuse connections
async with httpx.AsyncClient(http2=True) as client:
    response = await client.post(endpoint_url, json=payload)
```
**Expected improvement:** 20-30% reduction in request overhead

### 4. A/B Testing Strategy

Use the analysis script to make data-driven decisions:

```bash
# Analyze last 24 hours of A/B test data
python scripts/gcp/08_analyze_ab_results.py \
    --project-id plotpointe \
    --region us-central1 \
    --endpoint-id 7562422284145655808 \
    --hours 24 \
    --output ab_results.json
```

#### Decision Framework

| Condition | Recommendation |
|-----------|----------------|
| ONNX p95 < 50ms | Shift 100% to ONNX |
| ONNX 2x+ faster | Shift 80% to ONNX, keep 20% PyTorch for comparison |
| No significant difference | Choose based on cost (ONNX is cheaper) |
| Error rate > 1% | Investigate before shifting traffic |

### 5. Monitoring & Alerting

Set up Cloud Monitoring alerts:

```bash
# Create alerting policy for high latency
gcloud alpha monitoring policies create \
    --notification-channels="CHANNEL_ID" \
    --condition-filter='resource.type="aiplatform.googleapis.com/Endpoint" AND metric.type="prediction_latency"' \
    --condition-threshold-value=200 \
    --condition-threshold-comparison=GT
```

## Cost-Performance Tradeoffs

| Configuration | Monthly Cost | P50 Latency | Best For |
|--------------|--------------|-------------|----------|
| n1-standard-4 + PyTorch | $137 | 100ms | Development |
| n1-standard-4 + ONNX | $137 | 30ms | General use |
| n1-standard-2 + ONNX | $72 | 40ms | Cost-sensitive |
| n1-standard-4 + ONNX + Quantized | $137 | 15ms | Low latency |
| T4 GPU + PyTorch | $250 | 20ms | High throughput |

## Next Steps

1. **Fix model loading issue**: Debug GCS download in containers
2. **Run benchmarks**: Collect real latency data from deployed endpoint
3. **Implement quantization**: Reduce ONNX model size and latency
4. **Shift to ONNX**: Once validated, move 100% traffic to ONNX
5. **Downsize machine**: Use n1-standard-2 for cost optimization

## Commands Reference

```bash
# Deploy ONNX-only (after testing)
python scripts/gcp/06_deploy_endpoint.py \
    --project-id plotpointe \
    --region us-central1 \
    --endpoint-name gat-recommendation-endpoint \
    --model-id "projects/359145045403/locations/us-central1/models/1447756647106609152" \
    --machine-type n1-standard-2

# Run benchmark
python scripts/gcp/07_benchmark_endpoint.py \
    --project-id plotpointe \
    --region us-central1 \
    --endpoint-id 7562422284145655808 \
    --num-requests 100 \
    --output benchmark_results.json

# Analyze A/B results
python scripts/gcp/08_analyze_ab_results.py \
    --project-id plotpointe \
    --endpoint-id 7562422284145655808 \
    --hours 24
```
