# Experimental Results

This document contains all experimental results for ETP-GT and baselines.

**Last Updated**: TBD  
**Dataset**: RetailRocket  
**Evaluation**: Test set (15% of data, temporal split with 1-3 day blackout)

## Summary

| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | VRAM (GB) | Train Time (h) | p95 Latency (ms) |
|-------|-----------|-----------|---------|---------|-----------|----------------|------------------|
| GraphSAGE | TBD | TBD | TBD | TBD | TBD | TBD | - |
| GAT | TBD | TBD | TBD | TBD | TBD | TBD | - |
| GraphTransformer | **0.3828** | **0.4129** | **0.3065** | **0.3141** | ~20 | ~15.5 | - |
| **ETP-GT** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

**Best Model**: GraphTransformer (baseline)
**Improvement over Best Baseline**: TBD (ETP-GT pending)

## Baseline Details

### GraphSAGE

**Configuration**:
- Layers: 2
- Hidden dim: 256
- Aggregation: Mean
- Dropout: 0.2
- Optimizer: AdamW (lr=1e-3)

**Results**:
- Recall@10: TBD
- Recall@20: TBD
- NDCG@10: TBD
- NDCG@20: TBD
- VRAM: TBD GB
- Training time: TBD hours

**Artifacts**:
- Checkpoint: `gs://<bucket>/artifacts/baselines/graphsage-<date>/checkpoints/best.pt`
- Metrics: `gs://<bucket>/artifacts/baselines/graphsage-<date>/metrics.json`

### GAT

**Configuration**:
- Layers: 2
- Hidden dim: 256
- Heads: 4
- Dropout: 0.2
- Optimizer: AdamW (lr=1e-3)

**Results**:
- Recall@10: TBD
- Recall@20: TBD
- NDCG@10: TBD
- NDCG@20: TBD
- VRAM: TBD GB
- Training time: TBD hours

**Artifacts**:
- Checkpoint: `gs://<bucket>/artifacts/baselines/gat-<date>/checkpoints/best.pt`
- Metrics: `gs://<bucket>/artifacts/baselines/gat-<date>/metrics.json`

### GraphTransformer (LapPE) - Optimized

**Configuration**:
- Layers: 2 (reduced from 3 for speed)
- Hidden dim: 256
- Heads: 2 (reduced from 4 for speed)
- LapPE: k=16
- FFN: Disabled (29x speedup)
- Dropout: 0.1
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-5)
- Batch size: 32
- Max epochs: 100
- Early stopping patience: 10

**Results**:
- **Recall@10: 0.3828** (Best epoch: 56)
- **Recall@20: 0.4129**
- **NDCG@10: 0.3065**
- **NDCG@20: 0.3141**
- VRAM: ~20 GB
- Training time: ~15.5 hours (67 epochs total, stopped at epoch 66)
- Model parameters: 120,050,688
- Training batches: 3,764
- Validation batches: 746

**Training Details**:
- Job Name: `etpgt-graph_transformer_optimized-queued`
- Job ID: `7056293390840758272`
- Started: 2025-10-21 19:39:00 UTC
- Completed: 2025-10-22 11:06:10 UTC
- Duration: ~15.5 hours
- Platform: Vertex AI (g2-standard-8 + 1x L4 GPU)
- Cost: ~$28.83 (at $1.86/hour)

**Performance Progression**:
- Epoch 0: Recall@10=0.1974, NDCG@10=0.1431
- Epoch 10: Recall@10=0.3126, NDCG@10=0.2487
- Epoch 20: Recall@10=0.3317, NDCG@10=0.2677
- Epoch 30: Recall@10=0.3589, NDCG@10=0.2877
- Epoch 40: Recall@10=0.3617, NDCG@10=0.2909
- Epoch 50: Recall@10=0.3717, NDCG@10=0.2980
- **Epoch 56: Recall@10=0.3828, NDCG@10=0.3065** (Best)
- Epoch 66: Recall@10=0.3803, NDCG@10=0.3058 (Final, early stopped)

**Optimization Impact**:
- **88x speedup** vs original GraphTransformer (40 hours/epoch → 27 minutes/epoch)
- FFN removal: 29x speedup
- Layer reduction (3→2): 1.5x speedup
- Head reduction (4→2): Additional speedup
- **Performance**: 38.28% Recall@10 (significantly better than expected!)

**Artifacts**:
- Checkpoint: `gs://plotpointe-etpgt-data/outputs/graph_transformer_optimized/checkpoint_best.pt`
- Latest Checkpoint: `gs://plotpointe-etpgt-data/outputs/graph_transformer_optimized/checkpoint_latest.pt`
- Metrics: `gs://plotpointe-etpgt-data/outputs/graph_transformer_optimized/history.json`

## ETP-GT Results

### Configuration

- Layers: 3
- Hidden dim: 256
- Heads: 4
- LapPE: k=16
- Temporal buckets: [0-1m, 1-5m, 5-30m, 30-120m, 2-24h, 1-7d, 7d+]
- Path buckets: {1, 2, 3+}
- Fanout: [16, 12, 8]
- Dropout: 0.2
- DropPath: 0.1
- Loss: 0.7 listwise + 0.3 contrastive
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)

### Overall Performance

- Recall@10: TBD
- Recall@20: TBD
- NDCG@10: TBD
- NDCG@20: TBD
- VRAM: TBD GB
- Training time: TBD hours
- Inference p50: TBD ms
- Inference p95: TBD ms

### Stratified Performance

**By Session Length**:

| Length | Count | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|--------|-------|-----------|-----------|---------|---------|
| 3-4 | TBD | TBD | TBD | TBD | TBD |
| 5-9 | TBD | TBD | TBD | TBD | TBD |
| 10+ | TBD | TBD | TBD | TBD | TBD |

**By Last Gap Δt**:

| Gap | Count | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|-----|-------|-----------|-----------|---------|---------|
| ≤5m | TBD | TBD | TBD | TBD | TBD |
| 5-30m | TBD | TBD | TBD | TBD | TBD |
| 30m-2h | TBD | TBD | TBD | TBD | TBD |
| >2h | TBD | TBD | TBD | TBD | TBD |

**Cold Items** (≤5 interactions):

- Count: TBD
- Recall@10: TBD
- Recall@20: TBD
- NDCG@10: TBD
- NDCG@20: TBD

### Artifacts

- Checkpoint: `gs://<bucket>/artifacts/etpgt/etpgt-small-<date>/checkpoints/best.pt`
- Metrics: `gs://<bucket>/artifacts/etpgt/etpgt-small-<date>/metrics.json`
- Embeddings: `gs://<bucket>/artifacts/embeddings.npy`
- ANN Index: `gs://<bucket>/artifacts/ann/index.faiss`

## Ablation Studies

**Configuration**: Same as ETP-GT, with components disabled

| Ablation | Recall@20 | NDCG@20 | Δ Recall@20 | Δ NDCG@20 |
|----------|-----------|---------|-------------|-----------|
| Full ETP-GT | TBD | TBD | - | - |
| No temporal bias | TBD | TBD | TBD | TBD |
| No edge bias | TBD | TBD | TBD | TBD |
| No path bias | TBD | TBD | TBD | TBD |
| No contrastive loss | TBD | TBD | TBD | TBD |
| No CLS token | TBD | TBD | TBD | TBD |

**Key Findings**: TBD

**Artifacts**:
- Results: `gs://<bucket>/artifacts/ablations/ablations.csv`

## Serving Performance

**Platform**: Cloud Run (GCP)  
**Configuration**: 2 CPU, 4GB RAM  
**Candidates**: K=200 (Faiss IVFPQ)  
**Re-ranker**: 1-layer ETP-GT

### Latency Breakdown

| Component | p50 (ms) | p95 (ms) | p99 (ms) |
|-----------|----------|----------|----------|
| ANN retrieval | TBD | TBD | TBD |
| Subgraph sampling | TBD | TBD | TBD |
| Re-ranking | TBD | TBD | TBD |
| Overhead | TBD | TBD | TBD |
| **Total** | **TBD** | **TBD** | **TBD** |

**Target**: p95 ≤ 120ms ✓/✗

### Throughput

- Requests per second (RPS): TBD
- Cold start time: TBD ms

## Hardware & Environment

### Training

- **Platform**: Vertex AI Custom Job
- **Machine**: n1-standard-8 (8 vCPU, 30GB RAM)
- **Accelerator**: NVIDIA L4 (1x)
- **Disk**: 100GB SSD
- **Region**: us-central1
- **PyTorch**: TBD
- **CUDA**: TBD

### Serving

- **Platform**: Cloud Run
- **CPU**: 2 vCPU
- **Memory**: 4GB
- **Region**: us-central1
- **Concurrency**: 80 requests per instance

## Reproducibility

### Seeds

- Global seed: 42
- Per-run seeds: Logged in run ledgers

### Commits

- Data prep: `<commit-sha>`
- Training: `<commit-sha>`
- Serving: `<commit-sha>`

### Configs

- Baselines: `gs://<bucket>/configs/yoochoose_baselines.yaml`
- ETP-GT: `gs://<bucket>/configs/yoochoose_etpgt_small.yaml`
- Ablations: `gs://<bucket>/configs/ablations.yaml`

## Conclusion

TBD: Summary of findings, acceptance criteria met/not met, next steps.

