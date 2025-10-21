# Reproducibility Guide

This document provides exact commands to reproduce all experiments and deployments for ETP-GT.

## Prerequisites

- Python 3.11+
- GCP project with billing enabled
- GitHub repository with OIDC configured (see [GCP_OIDC_SETUP.md](GCP_OIDC_SETUP.md))
- Docker installed (for local testing)

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/<org>/etp-gt.git
cd etp-gt
git checkout <commit-sha>  # Use specific commit for exact reproduction
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your GCP project details:
# - GCP_PROJECT_ID
# - GCP_REGION
# - GCS_BUCKET
# - AR_REPO
# etc.
```

### 3. Install Dependencies

```bash
make setup
source .venv/bin/activate
```

## Data Preparation

### Local Execution

```bash
# Download and process RetailRocket dataset
make data

# Verify outputs
ls -lh data/processed/
# Expected: train.parquet, val.parquet, test.parquet
ls -lh data/interim/
# Expected: graph_edges.parquet
```

### GCP Execution

```bash
# Upload to GCS
gsutil -m cp -r data/processed/* ${GCS_BUCKET}/data/processed/
gsutil -m cp -r data/interim/* ${GCS_BUCKET}/data/interim/
```

**Run Ledger**: `data/artifacts/prep_ledger.json`

## GCP Infrastructure Bootstrap

```bash
make gcp-bootstrap
```

**Outputs**:
- GCS bucket: `${GCS_BUCKET}`
- Artifact Registry: `${AR_REPO}`
- Service Account: `${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com`

**Documentation**: See [GCP_OIDC_SETUP.md](GCP_OIDC_SETUP.md) for Workload Identity Federation setup.

## Training

### Baselines on Vertex AI

```bash
# Build and push training image
make docker-train-build
make docker-train-push

# Submit training job
make gcp-train
# Or manually:
bash scripts/gcp/03_vertex_custom_job.sh \
  --config gs://${GCS_BUCKET}/configs/yoochoose_baselines.yaml \
  --job-name baseline-graphsage-20250115 \
  --image-tag latest
```

**Artifacts** (on GCS):
- Checkpoints: `${GCS_BUCKET}/artifacts/baselines/<job-name>/checkpoints/`
- Metrics: `${GCS_BUCKET}/artifacts/baselines/<job-name>/metrics.json`
- Logs: `${GCS_BUCKET}/artifacts/baselines/<job-name>/logs/`

**Run Ledger**: `${GCS_BUCKET}/artifacts/baselines/<job-name>/run_ledger.json`

### ETP-GT on Vertex AI

```bash
bash scripts/gcp/03_vertex_custom_job.sh \
  --config gs://${GCS_BUCKET}/configs/yoochoose_etpgt_small.yaml \
  --job-name etpgt-small-20250115 \
  --image-tag latest
```

**Artifacts** (on GCS):
- Checkpoints: `${GCS_BUCKET}/artifacts/etpgt/<job-name>/checkpoints/`
- Metrics: `${GCS_BUCKET}/artifacts/etpgt/<job-name>/metrics.json`
- Logs: `${GCS_BUCKET}/artifacts/etpgt/<job-name>/logs/`

**Run Ledger**: `${GCS_BUCKET}/artifacts/etpgt/<job-name>/run_ledger.json`

### Ablations

```bash
bash scripts/gcp/03_vertex_custom_job.sh \
  --config gs://${GCS_BUCKET}/configs/ablations.yaml \
  --job-name ablations-20250115 \
  --image-tag latest
```

**Artifacts**:
- Results: `${GCS_BUCKET}/artifacts/ablations/ablations.csv`

## Evaluation

### Local Evaluation (on downloaded checkpoints)

```bash
# Download checkpoint
gsutil cp gs://${GCS_BUCKET}/artifacts/etpgt/<job-name>/checkpoints/best.pt \
  data/artifacts/best.pt

# Run evaluation
python -m etpgt.cli.eval \
  --checkpoint data/artifacts/best.pt \
  --data data/processed/test.parquet \
  --output data/artifacts/eval_results.json
```

### Metrics

See [RESULTS.md](RESULTS.md) for complete results table.

## Serving

### Export Embeddings

```bash
python scripts/export_embeddings.py \
  --checkpoint data/artifacts/best.pt \
  --output data/artifacts/embeddings.npy

# Upload to GCS
gsutil cp data/artifacts/embeddings.npy \
  gs://${GCS_BUCKET}/artifacts/embeddings.npy
```

### Build and Deploy to Cloud Run

```bash
# Build inference image
make docker-infer-build
make docker-infer-push

# Deploy to Cloud Run
make gcp-deploy

# Run smoke test
make gcp-smoke
```

**Endpoint**: `https://${RUN_SERVICE}-<hash>-${GCP_REGION}.run.app`

### Test Inference

```bash
curl -X POST https://${RUN_SERVICE}-<hash>-${GCP_REGION}.run.app/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "session": [
      {"item_id": 123, "event_type": "view", "ts": 1700000000000},
      {"item_id": 456, "event_type": "view", "ts": 1700000060000}
    ],
    "K": 20
  }'
```

## GitHub Actions (Automated)

### Training via Workflow Dispatch

1. Go to Actions → "Train on Vertex AI"
2. Click "Run workflow"
3. Inputs:
   - `config_path`: `gs://${GCS_BUCKET}/configs/yoochoose_etpgt_small.yaml`
   - `job_name_suffix`: `etpgt-small-v1`
   - `image_tag`: `latest`

### Release

```bash
git tag v0.1.0
git push origin v0.1.0
# Release workflow automatically creates GitHub release
```

## Exact Reproduction

For exact reproduction of published results:

```bash
# Use specific commit
git checkout <commit-sha-from-paper>

# Use specific data snapshot
gsutil -m cp -r gs://${GCS_BUCKET}/data-snapshots/v0.1.0/* data/

# Use specific config
gsutil cp gs://${GCS_BUCKET}/configs-snapshots/v0.1.0/yoochoose_etpgt_small.yaml \
  configs/

# Train with fixed seed (already in config)
make gcp-train
```

## Troubleshooting

### OOM during training
- Reduce batch size in config
- Enable gradient checkpointing
- Use smaller model variant

### Latency exceeds budget
- Reduce fanout: [12, 8, 4]
- Use 1-layer re-ranker
- Pre-warm Cloud Run instances (min=1)

### Data leakage suspected
- Run unit tests: `pytest tests/test_splits.py -v`
- Check blackout periods in `docs/data_stats.md`

## Version Information

All experiments should log:
- Git SHA
- Python version
- PyTorch version
- CUDA version (if GPU)
- GCP machine type
- Timestamp

See run ledgers for complete provenance.

