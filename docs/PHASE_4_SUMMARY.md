# Phase 4: Baselines on Vertex AI - Summary

**Status**: ✅ COMPLETE (Jobs Submitted)  
**Date**: 2025-10-20  
**Gate Criteria**: At least one strong baseline trained and evaluated

---

## Overview

Phase 4 implements three baseline models for session-based recommendation and trains them on Google Cloud Vertex AI with GPU acceleration. This establishes performance benchmarks for the custom ETP-GT model in Phase 5.

---

## Deliverables

### 1. Baseline Models (3 models)

#### GraphSAGE
- **File**: `etpgt/model/graphsage.py`
- **Architecture**: SAGEConv layers with mean/max/lstm aggregation
- **Parameters**: 
  - Embedding dim: 256
  - Hidden dim: 256
  - Layers: 3
  - Dropout: 0.1
  - Aggregator: mean (default)
- **Session Readout**: mean/max/last/attention pooling

#### GAT (Graph Attention Network)
- **File**: `etpgt/model/gat.py`
- **Architecture**: Multi-head attention with GATConv layers
- **Parameters**:
  - Embedding dim: 256
  - Hidden dim: 256
  - Layers: 3
  - Attention heads: 4
  - Dropout: 0.1
  - Concat heads: False (average)
- **Session Readout**: mean/max/last/attention pooling

#### GraphTransformer with Laplacian PE
- **File**: `etpgt/model/graph_transformer.py`
- **Architecture**: TransformerConv with Laplacian Positional Encoding
- **Parameters**:
  - Embedding dim: 256
  - Hidden dim: 256
  - Layers: 3
  - Attention heads: 4
  - Dropout: 0.1
  - Laplacian PE: k=16 eigenvectors
  - Gated residual connections: Yes
- **Session Readout**: mean/max/last/attention pooling

### 2. Training Infrastructure

#### Base Model Class
- **File**: `etpgt/model/base.py`
- **Features**:
  - Abstract base class for all recommendation models
  - Item embeddings with Xavier initialization
  - BPR (Bayesian Personalized Ranking) contrastive loss
  - Top-k prediction with dot product scoring
  - SessionReadout module with 4 pooling strategies

#### Data Loading
- **File**: `etpgt/train/dataloader.py`
- **Features**:
  - SessionDataset for loading sessions and graph edges
  - Negative sampling (5 negatives per positive)
  - Session truncation (max 50 items)
  - Session subgraph construction
  - Custom collate function for PyG Batch objects

#### Training Loop
- **File**: `etpgt/train/trainer.py`
- **Features**:
  - Training epoch with BPR loss
  - Evaluation with Recall@K and NDCG@K
  - Early stopping with patience
  - Checkpoint saving (latest and best)
  - Training history tracking

#### Training Script
- **File**: `scripts/train/train_baseline.py`
- **Features**:
  - Command-line argument parsing
  - GCS integration for data download and output upload
  - Support for all three baseline models
  - AdamW optimizer with configurable hyperparameters
  - Comprehensive logging

### 3. Docker & Deployment

#### Training Dockerfile
- **File**: `docker/train.Dockerfile`
- **Base Image**: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
- **Python Version**: 3.10
- **Dependencies**:
  - PyTorch 2.1.0 with CUDA 11.8
  - PyTorch Geometric 2.7.0
  - torch-scatter, torch-sparse, torch-cluster, torch-spline-conv
  - Google Cloud Storage client
  - All project dependencies from requirements.txt

#### Build Script
- **File**: `scripts/gcp/02_build_training_image_cloudbuild.sh`
- **Method**: Cloud Build (no local Docker required)
- **Image URI**: `us-central1-docker.pkg.dev/plotpointe/etpgt/etpgt-train:latest`
- **Build Time**: ~13 minutes
- **Status**: ✅ Successfully built and pushed

#### Job Submission Script
- **File**: `scripts/gcp/03_submit_training_job.sh`
- **Features**:
  - YAML-based job configuration
  - Configurable model type, hyperparameters
  - GPU acceleration (NVIDIA L4)
  - GCS integration for data and outputs

### 4. Vertex AI Training Jobs

#### Job Configuration
- **Machine Type**: g2-standard-8 (8 vCPUs, 32 GB RAM)
- **Accelerator**: NVIDIA L4 x 1
- **Region**: us-central1
- **Image**: us-central1-docker.pkg.dev/plotpointe/etpgt/etpgt-train:latest

#### Submitted Jobs
1. **GraphSAGE**
   - Job ID: 4039383017804791808
   - Display Name: etpgt-graphsage-20251019-215612
   - Status: JOB_STATE_PENDING
   - Created: 2025-10-20T01:56:13Z

2. **GAT**
   - Job ID: 625654500257955840
   - Display Name: etpgt-gat-20251019-215620
   - Status: JOB_STATE_PENDING
   - Created: 2025-10-20T01:56:21Z

3. **GraphTransformer**
   - Job ID: 6840621986029240320
   - Display Name: etpgt-graph_transformer-20251019-215626
   - Status: JOB_STATE_PENDING
   - Created: 2025-10-20T01:56:27Z

---

## Training Configuration

### Hyperparameters
- **Embedding Dimension**: 256
- **Hidden Dimension**: 256
- **Number of Layers**: 3
- **Number of Attention Heads**: 4 (GAT, GraphTransformer)
- **Dropout**: 0.1
- **Session Readout**: mean
- **Batch Size**: 32
- **Max Epochs**: 100
- **Learning Rate**: 0.001
- **Optimizer**: AdamW
- **Weight Decay**: 1e-5
- **Patience**: 10 epochs
- **Negative Samples**: 5 per positive

### Evaluation Metrics
- **Recall@10**
- **Recall@20**
- **NDCG@10**
- **NDCG@20**

### Data
- **Training Sessions**: 120,436 sessions, 679,365 events
- **Validation Sessions**: 23,861 sessions, 127,461 events
- **Test Sessions**: 23,408 sessions, 125,363 events
- **Graph**: 82,173 nodes, 737,716 edges
- **GCS Bucket**: gs://plotpointe-etpgt-data

---

## Code Quality

### Linting & Formatting
- ✅ Ruff: All checks passed
- ✅ Black: All files formatted
- ✅ Isort: All imports sorted

### Type Checking
- ⚠️ Mypy: Skipped (PyTorch Geometric type stubs not available)

### Testing
- ✅ All existing tests passing (17/17)
- ⚠️ No baseline model tests yet (will add after training completes)

---

## Next Steps

### Immediate (Phase 4 Completion)
1. **Monitor Training Jobs**: Wait for jobs to complete (~1-2 hours per model)
2. **Download Checkpoints**: Retrieve trained models from GCS
3. **Evaluate on Test Set**: Run evaluation on test split
4. **Compare Baselines**: Analyze Recall@20 and NDCG@20 metrics
5. **Document Results**: Update this summary with final metrics

### Phase 5 (Custom ETP-GT)
1. **Implement ETP-GT Architecture**: Temporal & path-aware attention
2. **Implement Dual Loss**: 0.7 listwise + 0.3 contrastive
3. **Train on Vertex AI**: Submit ETP-GT training job
4. **Evaluate Against Baselines**: Compare with best baseline
5. **Gate Decision**: Proceed if ETP-GT beats ≥1 baseline

---

## Monitoring Commands

### List All Jobs
```bash
gcloud ai custom-jobs list \
    --region=us-central1 \
    --filter='displayName~etpgt-' \
    --format='table(displayName,state,createTime)'
```

### Stream Logs (GraphSAGE)
```bash
gcloud ai custom-jobs stream-logs 4039383017804791808 --region=us-central1
```

### Stream Logs (GAT)
```bash
gcloud ai custom-jobs stream-logs 625654500257955840 --region=us-central1
```

### Stream Logs (GraphTransformer)
```bash
gcloud ai custom-jobs stream-logs 6840621986029240320 --region=us-central1
```

### View in Console
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=plotpointe

---

## Files Created/Modified

### New Files (10)
1. `etpgt/model/base.py` - Base recommendation model class
2. `etpgt/model/graphsage.py` - GraphSAGE baseline
3. `etpgt/model/gat.py` - GAT baseline
4. `etpgt/model/graph_transformer.py` - GraphTransformer baseline
5. `etpgt/train/dataloader.py` - Session dataset and data loading
6. `etpgt/train/trainer.py` - Training loop and evaluation
7. `scripts/train/train_baseline.py` - Main training script
8. `docker/train.Dockerfile` - Training Docker image
9. `scripts/gcp/02_build_training_image_cloudbuild.sh` - Cloud Build script
10. `scripts/gcp/03_submit_training_job.sh` - Vertex AI job submission

### Modified Files (4)
1. `etpgt/model/__init__.py` - Added baseline model exports
2. `etpgt/train/__init__.py` - Added training module exports
3. `pyproject.toml` - Updated Python version requirement (>=3.10)
4. `.env` - Updated Vertex AI machine type (g2-standard-8)

---

## Gate Criteria Assessment

**Gate**: At least one strong baseline trained and evaluated

**Status**: 🟡 IN PROGRESS
- ✅ Three baseline models implemented
- ✅ Training infrastructure complete
- ✅ Docker image built and pushed
- ✅ Training jobs submitted to Vertex AI
- ⏳ Waiting for training to complete
- ⏳ Evaluation pending

**Next Gate Check**: After training jobs complete and models are evaluated on test set.

---

**Phase 4 Status**: 🟡 **JOBS SUBMITTED - AWAITING RESULTS**

