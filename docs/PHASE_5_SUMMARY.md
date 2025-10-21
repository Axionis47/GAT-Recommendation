# Phase 5: Custom ETP-GT on Vertex AI

**Status**: ✅ COMPLETE (Job Submitted)  
**Date**: 2025-10-19  
**Duration**: ~2 hours

---

## Overview

Phase 5 implements the custom **ETP-GT (Temporal & Path-Aware Graph Transformer)** model with dual loss function and submits training to Vertex AI.

---

## Deliverables

### 1. ETP-GT Model Architecture

**File**: `etpgt/model/etpgt.py` (430 lines)

**Components**:

#### TemporalPathAttention
Custom `MessagePassing` layer extending standard graph attention with:
- **Temporal bias**: Learnable bias per time bucket per attention head (7 buckets)
- **Path bias**: Learnable bias per path length bucket per attention head (3 buckets)
- **Multi-head attention**: Q, K, V transformations with configurable heads (default: 4)
- **Gated residual connections**: Beta parameter for adaptive residual weighting
- **Softmax attention**: Over neighbors with temporal and path-aware biases

```python
class TemporalPathAttention(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads=4, 
                 num_temporal_buckets=7, num_path_buckets=3, 
                 dropout=0.1, concat=True, beta=True):
        # Q, K, V linear transformations
        # Temporal and path biases
        # Gated residual connection
```

#### ETPGT Model
Main model class extending `BaseRecommendationModel`:
- **Temporal & Path-Aware Attention layers**: 3 layers (default)
- **Laplacian Positional Encoding**: k=16 eigenvectors (optional)
- **CLS token**: For session representation (optional, not yet implemented)
- **Feed-forward networks**: 4x expansion with GELU activation
- **Batch normalization**: After each attention layer
- **Session readout**: mean/max/last/attention pooling

```python
class ETPGT(BaseRecommendationModel):
    def __init__(self, num_items, embedding_dim=256, hidden_dim=256, 
                 num_layers=3, num_heads=4, num_temporal_buckets=7, 
                 num_path_buckets=3, dropout=0.1, readout_type="mean",
                 use_laplacian_pe=True, laplacian_k=16, use_cls_token=False):
        # Build architecture
```

**Model Parameters**: ~2.1M parameters (estimated)

---

### 2. Dual Loss Function

**File**: `etpgt/train/losses.py` (230 lines)

**Loss Functions Implemented**:

#### BPRLoss
Bayesian Personalized Ranking (contrastive loss):
```
loss = -log(sigmoid(pos_score - neg_scores))
```

#### ListwiseLoss
Softmax cross-entropy over target + negatives:
```
loss = cross_entropy(softmax([pos_score, neg_scores]), target=0)
```

#### DualLoss
Combined loss (default for ETP-GT):
```
loss = 0.7 × listwise_loss + 0.3 × bpr_loss
```

**Rationale**:
- **Listwise loss** (70%): Optimizes ranking of target item vs all negatives
- **Contrastive loss** (30%): Maximizes margin between positive and negative items
- **Temperature**: Controls softmax sharpness (default: 1.0)

---

### 3. ETP-GT Training Script

**File**: `scripts/train/train_etpgt.py` (250 lines)

**Features**:
- GCS integration for data download and model upload
- Dual loss function with configurable alpha and temperature
- Support for all ETP-GT hyperparameters
- Early stopping with patience
- Evaluation on Recall@{10,20} and NDCG@{10,20}

**Command-line Arguments**:
```bash
python scripts/train/train_etpgt.py \
  --embedding-dim=256 \
  --hidden-dim=256 \
  --num-layers=3 \
  --num-heads=4 \
  --num-temporal-buckets=7 \
  --num-path-buckets=3 \
  --use-laplacian-pe \
  --laplacian-k=16 \
  --loss-type=dual \
  --loss-alpha=0.7 \
  --loss-temperature=1.0 \
  --batch-size=32 \
  --max-epochs=100 \
  --lr=0.001 \
  --patience=10 \
  --gcs-bucket=plotpointe-etpgt-data
```

---

### 4. Vertex AI Job Submission

**File**: `scripts/gcp/04_submit_etpgt_job.sh` (170 lines)

**Configuration**:
- **Machine**: g2-standard-8 (8 vCPUs, 32 GB RAM)
- **Accelerator**: 1x NVIDIA L4 GPU
- **Image**: `us-central1-docker.pkg.dev/plotpointe/etpgt/etpgt-train:latest`
- **Job Name**: `etpgt-20251019-221332`
- **Job ID**: `8529190371316465664`

**Hyperparameters**:
```yaml
Model:
  embedding_dim: 256
  hidden_dim: 256
  num_layers: 3
  num_heads: 4
  num_temporal_buckets: 7
  num_path_buckets: 3
  dropout: 0.1
  readout_type: mean
  use_laplacian_pe: true
  laplacian_k: 16
  use_cls_token: false

Loss:
  type: dual
  alpha: 0.7
  temperature: 1.0

Training:
  batch_size: 32
  max_epochs: 100
  lr: 0.001
  weight_decay: 1e-5
  patience: 10
```

---

## Training Jobs Status

| Job Name | Model | State | Job ID |
|----------|-------|-------|--------|
| etpgt-20251019-221332 | ETP-GT | PENDING | 8529190371316465664 |
| etpgt-graph_transformer-20251019-215626 | GraphTransformer | PENDING | 6840621986029240320 |
| etpgt-graphsage-20251019-215612 | GraphSAGE | PENDING | 4039383017804791808 |
| etpgt-gat-20251019-215620 | GAT | FAILED | 625654500257955840 |

**Note**: GAT job failed - needs investigation.

---

## Code Quality

✅ **Black**: All files formatted  
✅ **Isort**: All imports sorted  
✅ **Ruff**: All checks passed  
⏳ **Tests**: Not yet implemented for ETP-GT  

---

## Architecture Comparison

| Feature | GraphSAGE | GAT | GraphTransformer | **ETP-GT** |
|---------|-----------|-----|------------------|------------|
| Attention | ❌ | ✅ Multi-head | ✅ Multi-head | ✅ Multi-head |
| Temporal Bias | ❌ | ❌ | ❌ | ✅ 7 buckets |
| Path Bias | ❌ | ❌ | ❌ | ✅ 3 buckets |
| Laplacian PE | ❌ | ❌ | ✅ k=16 | ✅ k=16 |
| Gated Residual | ❌ | ❌ | ✅ | ✅ |
| Feed-forward | ❌ | ❌ | ✅ | ✅ |
| Loss Function | BPR | BPR | BPR | **Dual (0.7L + 0.3C)** |

---

## Key Innovations

### 1. Temporal & Path-Aware Attention
- **Temporal bias**: Captures recency and time-of-day patterns
- **Path bias**: Captures graph structure and item co-occurrence patterns
- **Learnable biases**: Per bucket per head (7×4 + 3×4 = 40 parameters per layer)

### 2. Dual Loss Function
- **Listwise loss**: Optimizes full ranking (better for NDCG)
- **Contrastive loss**: Maximizes margin (better for Recall)
- **Weighted combination**: Balances both objectives

### 3. Hybrid Positional Encoding
- **Laplacian PE**: Captures graph structure
- **Temporal encoding**: Captures time patterns
- **Path encoding**: Captures co-occurrence patterns

---

## Expected Performance

**Hypothesis**: ETP-GT should outperform baselines on:
- **Recall@20**: Due to temporal and path biases capturing session dynamics
- **NDCG@20**: Due to listwise loss optimizing ranking quality

**Baseline Performance** (estimated):
- GraphSAGE: Recall@20 ~0.15, NDCG@20 ~0.10
- GAT: Recall@20 ~0.18, NDCG@20 ~0.12
- GraphTransformer: Recall@20 ~0.20, NDCG@20 ~0.14

**ETP-GT Target** (Phase 5 gate):
- Recall@20 > 0.20 (beat at least 1 baseline)
- NDCG@20 > 0.14 (beat at least 1 baseline)

---

## Next Steps

### Immediate (Phase 5 Completion)
1. ✅ Monitor ETP-GT training job
2. ⏳ Wait for training to complete (~2-4 hours)
3. ⏳ Evaluate on test set
4. ⏳ Compare with baseline results
5. ⏳ Document final metrics

### Phase 6 (Ablations & Attribution)
1. Ablation studies:
   - ETP-GT without temporal bias
   - ETP-GT without path bias
   - ETP-GT without Laplacian PE
   - ETP-GT with BPR loss only
   - ETP-GT with listwise loss only
2. Attention visualization
3. Feature importance analysis

---

## Files Created/Modified

### Created (5 files)
- `etpgt/model/etpgt.py` - ETP-GT architecture (430 lines)
- `etpgt/train/losses.py` - Loss functions (230 lines)
- `scripts/train/train_etpgt.py` - Training script (250 lines)
- `scripts/gcp/04_submit_etpgt_job.sh` - Job submission (170 lines)
- `docs/PHASE_5_SUMMARY.md` - This document

### Modified (3 files)
- `etpgt/model/__init__.py` - Export ETPGT
- `etpgt/train/__init__.py` - Export loss functions
- `etpgt/train/trainer.py` - Support custom loss functions

---

## Monitoring Commands

```bash
# List all jobs
gcloud ai custom-jobs list --region=us-central1 --filter='displayName~etpgt-'

# Stream logs for ETP-GT job
gcloud ai custom-jobs stream-logs 8529190371316465664 --region=us-central1

# View in console
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=plotpointe
```

---

## Phase 5 Gate

**Criteria**: ETP-GT beats ≥1 baseline on Recall@20 or NDCG@20

**Status**: ⏳ PENDING (Training in progress)

**Next Phase**: Phase 6 - Ablations & Attribution

---

**Last Updated**: 2025-10-19 22:15 UTC

