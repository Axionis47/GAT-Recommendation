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
| GraphTransformer | TBD | TBD | TBD | TBD | TBD | TBD | - |
| **ETP-GT** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

**Best Model**: TBD  
**Improvement over Best Baseline**: TBD% Recall@20, TBD% NDCG@20

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

### GraphTransformer (LapPE)

**Configuration**:
- Layers: 3
- Hidden dim: 256
- Heads: 4
- LapPE: k=16
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
- Checkpoint: `gs://<bucket>/artifacts/baselines/graphtransformer-<date>/checkpoints/best.pt`
- Metrics: `gs://<bucket>/artifacts/baselines/graphtransformer-<date>/metrics.json`

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

