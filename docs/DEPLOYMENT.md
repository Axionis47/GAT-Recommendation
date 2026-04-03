# Deployment: Serving and Infrastructure

This document covers how models are served in production, the infrastructure that supports them, and how to monitor the deployed system.

## Serving Architecture

```
Client Request
       |
       v
Vertex AI Endpoint (Cloud Run)
       |
       v
FastAPI (vertex_app.py)
       |
       v
ONNX Runtime / PyTorch
       |
       v
Pre-computed Item Embeddings
       |
       v
Top-K Recommendations
```

At serving time, the GNN is not involved. Item embeddings are pre-computed during training and stored as a numpy array. Inference is a matrix multiply: session embedding (mean of item embeddings) dot-product with all item embeddings.

---

## API Endpoints

**File:** `scripts/serve/vertex_app.py`

### POST /predict (Vertex AI format)

The primary endpoint. Follows Vertex AI's prediction request format.

**Request:**
```json
{
  "instances": [
    {
      "session_items": [42, 1337, 99],
      "k": 10
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "recommendations": [501, 2043, 789, 1122, ...],
      "scores": [0.95, 0.91, 0.88, 0.85, ...]
    }
  ]
}
```

**Processing steps:**
1. Filter valid items (0 <= item < num_items)
2. Look up item embeddings from pre-computed numpy array
3. Compute session embedding = mean of item embeddings
4. L2 normalize session and item embeddings
5. Compute cosine similarity scores = session_norm @ item_norms.T
6. Exclude already-seen items (set scores to -inf)
7. Return top-K by score

### POST /recommend (Legacy format)

**Request:**
```json
{
  "session_items": [42, 1337, 99],
  "k": 10
}
```

**Response:**
```json
{
  "recommendations": [501, 2043, 789, ...],
  "scores": [0.95, 0.91, 0.88, ...]
}
```

### POST /recommend/batch

Same as `/recommend` but accepts a list of sessions.

### GET /health

```json
{
  "status": "healthy",
  "model_loaded": true,
  "inference_mode": "onnx",
  "num_items": 82173,
  "embedding_dim": 256
}
```

### GET /metrics

Prometheus-compatible metrics endpoint:
- `prediction_requests_total` (counter)
- `prediction_latency_seconds` (histogram)
- `model_loaded` (gauge)
- `model_num_items` (gauge)

### GET /drift

Evidently-based drift detection:
- Prediction score distribution shift
- Session length distribution shift
- Item diversity entropy check

---

## Inference Modes

### ONNX Mode (Production)

- Load `session_recommender.onnx` + `item_embeddings.npy`
- Uses ONNX Runtime for inference
- 5.5ms per request (local benchmark)
- 456 MB model size

### PyTorch Mode (Fallback/Debug)

- Load PyTorch checkpoint `.pt`
- Reconstruct model from state_dict keys
- Extract item embeddings via `model.get_item_embeddings()`
- 51.7ms per request
- 1.4 GB model size

### Performance Comparison

| Metric | PyTorch | ONNX | Improvement |
|--------|---------|------|-------------|
| Inference latency | 51.7ms | 5.5ms | 9.37x faster |
| Model size | 1.4 GB | 456 MB | 3x smaller |
| Cold start | ~30s | ~10s | 3x faster |

### Expected Vertex AI Latencies (n1-standard-4)

| Percentile | PyTorch | ONNX |
|------------|---------|------|
| P50 | 80-120ms | 20-40ms |
| P95 | 150-200ms | 40-80ms |
| P99 | 200-300ms | 60-100ms |
| Cold start | 60-90s | 30-45s |

Includes network overhead, GCS model download at startup, and serialization.

---

## ONNX Export Pipeline

**File:** `scripts/pipeline/export_onnx.py`

### What gets exported

Only the scoring layer is exported to ONNX. The full GNN has dynamic graph structure that ONNX cannot represent. Since item embeddings are pre-computed, the GNN is not needed at serving time.

**Exported model:**
- Input: `session_embedding [batch_size, 256]`
- Output: `scores [batch_size, num_items]`
- ONNX opset version: 14

**Separately saved:**
- `item_embeddings.npy`: Pre-computed numpy array `[num_items, 256]`
- `model_metadata.json`: Model class, dimensions, checkpoint info, export timestamp

### Validation

The export script validates the ONNX model:
1. Syntax check: `onnx.checker.check_model()`
2. Numerical equivalence: PyTorch vs ONNX Runtime output, max difference < 1e-5
3. Benchmark: Measures inference speedup

---

## Docker Images

### Training Image

**File:** `docker/train.Dockerfile`
**Base:** `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
**Size:** ~5-6 GB

Includes PyTorch Geometric, torch-scatter, torch-sparse, and all training dependencies.

### PyTorch Serving Image

**File:** `docker/serve.Dockerfile`
**Base:** `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
**Size:** ~4-5 GB

Same as training but with reduced dependencies. Used for debugging or when full GNN forward pass is needed.

### ONNX Serving Image (Production)

**File:** `docker/serve-onnx.Dockerfile`
**Base:** `python:3.11-slim`
**Size:** ~500 MB

Minimal: just ONNX Runtime + numpy + FastAPI + uvicorn. 10x smaller than PyTorch image. This is what runs in production.

---

## Infrastructure (Terraform)

**Directory:** `infra/`

### Resources

| Resource | Name | Purpose |
|----------|------|---------|
| GCS Bucket | `plotpointe-etpgt-data` | Data versioning, model artifacts |
| Artifact Registry | `etpgt` (us-central1) | Docker images |
| Service Account | `etpgt-sa` | IAM for training and serving |

### GCS Bucket Configuration

- Versioning enabled (3 versions retained)
- 30-day lifecycle rule
- Location: us-central1

### Service Account Roles

- `roles/storage.objectAdmin` (read/write GCS)
- `roles/aiplatform.user` (Vertex AI jobs)
- `roles/logging.logWriter` (Cloud Logging)
- `roles/monitoring.metricWriter` (Cloud Monitoring)

### GCP APIs Enabled

Storage, Artifact Registry, Vertex AI, Compute Engine, Cloud Run, IAM

---

## Cloud Build Pipeline

**Files:** `cloudbuild.yaml`, `cloudbuild-onnx.yaml`, `cloudbuild-pytorch.yaml`

### Build Process

```
Code Push
    |
    v
Cloud Build (E2_HIGHCPU_8, 30-min timeout)
    |
    v
Docker Build
    |
    v
Trivy Security Scan (CRITICAL + HIGH severity)
    |
    v
Push to Artifact Registry
    |
    v
Deploy to Vertex AI
```

### Security

- Trivy scans for CRITICAL and HIGH severity CVEs
- No secrets in Docker images (loaded from environment at runtime)
- Service account has minimum required permissions

---

## Model Artifacts (Production)

Stored in GCS at `gs://plotpointe-etpgt-data/models/serving/20260127-154130/`:

| Artifact | Path | Size |
|----------|------|------|
| PyTorch checkpoint | `pytorch/model.pt` | 1.4 GB |
| ONNX model | `onnx/session_recommender.onnx` | 456 MB |
| Item embeddings | `onnx/item_embeddings.npy` | 456 MB |

---

## Optimization Recommendations

### Model-level

| Optimization | Expected Impact | Effort |
|-------------|----------------|--------|
| Reduce embedding to 128 dims | 30-40% faster, 50% smaller | Retrain required |
| INT8 quantization | 2-3x faster, 4x smaller | Export script change |
| Float16 embeddings | 50% memory, 20-30% faster | Export script change |

### Infrastructure-level

| Optimization | Expected Impact | Cost Change |
|-------------|----------------|-------------|
| n1-standard-2 + ONNX | +10ms latency | -47% cost ($72/month vs $137) |
| T4 GPU + PyTorch | 5-10x faster inference | +82% cost ($250/month) |
| Min replicas = 1 | No cold starts | Baseline cost |
| LRU cache (10K sessions) | 10-100x for cache hits | No cost change |

### Cost-Performance Matrix

| Configuration | Monthly Cost | P50 Latency | Best For |
|--------------|-------------|-------------|----------|
| n1-standard-4 + PyTorch | $137 | 100ms | Development |
| n1-standard-4 + ONNX | $137 | 30ms | General use |
| n1-standard-2 + ONNX | $72 | 40ms | Budget-conscious |
| ONNX + INT8 quantized | $137 | 15ms | Low latency |
| T4 GPU + PyTorch | $250 | 20ms | High throughput |

---

## Monitoring and Alerting

### Key Metrics to Watch

| Metric | Threshold | Action |
|--------|-----------|--------|
| P95 latency | > 200ms | Scale up or investigate |
| Error rate | > 1% | Investigate model loading |
| Prediction diversity | Entropy drop > 20% | Check for embedding collapse |
| Cold start frequency | > 1/hour | Increase min replicas |

### Commands

```bash
# Deploy to Vertex AI
make gcp-deploy

# Run integration tests against deployed endpoint
make gcp-smoke

# Benchmark latency (100 requests)
python scripts/gcp/07_benchmark_endpoint.py \
    --endpoint-id 7562422284145655808 \
    --num-requests 100

# Analyze A/B test results
python scripts/gcp/08_analyze_ab_results.py \
    --endpoint-id 7562422284145655808 \
    --hours 24

# Check GCP setup
make gcp-validate
```
