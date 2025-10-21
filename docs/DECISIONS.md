# Architectural Decision Record

This document captures **locked decisions** for ETP-GT. Changes require explicit justification and re-validation.

## Data Processing

### Session Definition
- **Gap threshold**: 30 minutes of inactivity
- **Min length**: ≥3 events
- **Deduplication**: Squash exact duplicates within 15s unless event escalates (view→cart→purchase)
- **Rationale**: Standard in session-based RecSys; balances signal vs. noise

### Temporal Split
- **Ratio**: 70% train / 15% val / 15% test
- **Method**: Chronological by timestamp
- **Blackout**: 1-3 days between splits
- **Rationale**: Prevents temporal leakage; simulates production deployment lag

### Graph Construction
- **Edge definition**: Co-occurrence within ±5 steps in any session
- **Edge attributes**:
  - `count`: Number of co-occurrences
  - `last_ts`: Most recent co-occurrence timestamp
  - `event_pair_hist`: Distribution of (event_i, event_j) types
- **Direction**: Undirected (symmetric exploration)
- **Rationale**: Captures item similarity beyond content features

## Temporal & Path Encodings

### Time Delta Buckets
```
[0-1m, 1-5m, 5-30m, 30-120m, 2-24h, 1-7d, 7d+]
```
- **Rationale**: Log-scale captures both micro (browsing) and macro (return visits) patterns

### Path Length Buckets
```
{1, 2, 3+}
```
- **Rationale**: Distinguishes direct connections, 2-hop neighbors, and longer paths

### Laplacian Positional Encoding (LapPE)
- **k**: 16 eigenvectors
- **Normalization**: Symmetric normalized Laplacian
- **Sign**: Absolute value (sign ambiguity)
- **Rationale**: Provides structural position signal; k=16 balances expressiveness vs. compute

## Sampling Strategy

### Temporal Path Sampler
- **Fanout**: [16, 12, 8] for layers 1, 2, 3
- **Temporal constraint**: Only edges with `time ≤ t` (current event)
- **Importance sampling**: `recency × degree`
- **Backoff**: If insufficient neighbors, sample with replacement
- **Cap**: Max 10,000 edges per batch (OOM protection)
- **Rationale**: Time-aware sampling prevents leakage; fanout decay balances depth vs. breadth

## Model Architecture

### ETP-GT Configuration (Small)
- **Embedding dim**: 256
- **Attention heads**: 4
- **Layers**: 3
- **Dropout**: 0.2
- **Activation**: SwiGLU (FFN)
- **DropPath**: 0.1
- **LayerScale**: Optional (ablation)
- **Readout**: CLS token + gated pooling
- **Rationale**: Proven in vision transformers; SwiGLU outperforms ReLU/GELU

### Attention Mechanism
- **Base**: Scaled dot-product (QK^T / √d)
- **Biases** (learned, additive):
  - `edgeTypeBias`: Per edge type (view-view, view-cart, etc.)
  - `dtBias`: Per time delta bucket
  - `pathBias`: Per path length bucket
- **Rationale**: Explicit bias terms are interpretable and efficient

### Loss Function
- **Primary**: Listwise softmax over K=200 candidates
- **Auxiliary**: Contrastive loss (edge-drop + time-jitter augmentation)
- **Weight**: 0.7 listwise + 0.3 contrastive
- **Rationale**: Listwise optimizes ranking; contrastive improves robustness

## Training

### Hyperparameters
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: Cosine annealing with warmup (5% of steps)
- **Batch size**: 256 sessions (adjust for GPU memory)
- **Epochs**: 50 (early stop on NDCG@20, patience=5)
- **Gradient clipping**: 1.0
- **Mixed precision**: FP16 (AMP)
- **Rationale**: Standard for transformers; early stopping prevents overfitting

### Negative Sampling
- **Strategy**: Uniform random from item catalog
- **Ratio**: 1 positive : 199 negatives (for K=200)
- **In-batch negatives**: Shared across batch for efficiency
- **Rationale**: Balances diversity vs. compute

## Serving

### Inference Pipeline
1. **Input**: Last N=10 items from session
2. **ANN**: Retrieve K=200 candidates (Faiss IVFPQ)
3. **Subgraph**: Build temporal neighborhood (fanout [16, 12, 8])
4. **Re-rank**: 1-layer ETP-GT (lightweight)
5. **Output**: Top-20 items + attributions

### Latency Budget
- **p50**: ≤60ms
- **p95**: ≤120ms
- **Breakdown target**:
  - ANN: ~15ms
  - Subgraph sampling: ~20ms
  - Re-rank: ~60ms
  - Overhead: ~25ms
- **Rationale**: Competitive with production systems; allows for network latency

### Deployment
- **Platform**: Cloud Run (GCP)
- **Resources**: 2-4 CPU, 2-4GB RAM
- **Scaling**: Min 0 instances (cost), max 10 (burst)
- **Rationale**: Serverless reduces ops burden; auto-scaling handles traffic

## Evaluation

### Metrics
- **Primary**: Recall@{10, 20}, NDCG@{10, 20}
- **Stratification**:
  - Session length: {3-4, 5-9, 10+}
  - Last gap Δt: {≤5m, 5-30m, 30m-2h, >2h}
  - Cold items: ≤5 interactions
- **Rationale**: Recall measures coverage; NDCG measures ranking quality; stratification reveals biases

### Baselines
1. **GraphSAGE**: Mean aggregation, 2 layers
2. **GAT**: Attention aggregation, 2 layers, 4 heads
3. **GraphTransformer**: LapPE, 3 layers, 4 heads (no temporal/path biases)
- **Rationale**: Covers spectrum from simple (SAGE) to complex (GT)

## Reproducibility

### Seeds
- **Global**: 42 (Python, NumPy, PyTorch, CUDA)
- **Per-run**: Logged in run ledger
- **Rationale**: Deterministic results for debugging and comparison

### Run Ledger
Every script writes:
- CLI command
- Environment variables
- Git SHA
- Input/output paths
- Timestamps
- **Rationale**: Full provenance for audits and rollbacks

## Version Control

### Branching
- **main**: Production-ready code
- **develop**: Integration branch
- **feature/***: Feature development
- **Rationale**: Git Flow for stability

### Tagging
- **Format**: `v<major>.<minor>.<patch>` (semantic versioning)
- **Trigger**: Release workflow on tag push
- **Rationale**: Clear versioning for deployments

## Security

### Authentication
- **Method**: GitHub OIDC → GCP Workload Identity Federation
- **No long-lived keys**: Service account keys prohibited
- **Rationale**: Reduces credential leakage risk

### Secrets Management
- **GitHub Secrets**: GCP project ID, region, bucket, etc.
- **Never in code**: `.env` files gitignored
- **Rationale**: Separation of config and code

## Change Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01-XX | Initial ADR | Bootstrap phase |
| TBD | Update if defaults change | Document in CHANGELOG.md |

