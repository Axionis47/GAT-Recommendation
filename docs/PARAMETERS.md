# Complete Parameter Reference

Every tunable parameter in the system, where it is configured, and why it has its current value.

---

## Data Pipeline Parameters

### Sessionization (`scripts/data/02_sessionize.py`)

| Parameter | Value | Variable | Line | Why |
|-----------|-------|----------|------|-----|
| Session gap | 30 minutes | `SESSION_GAP_MINUTES` | 20 | Industry standard for e-commerce. |
| Min session length | 3 events | `MIN_SESSION_LENGTH` | 21 | Need at least 2 context items + 1 target. |

### Temporal Split (`scripts/data/03_temporal_split.py`)

| Parameter | Value | Variable | Line | Why |
|-----------|-------|----------|------|-----|
| Train ratio | 0.70 | `TRAIN_RATIO` | 21 | Standard train split. |
| Val ratio | 0.15 | `VAL_RATIO` | 22 | Enough for reliable metric estimation. |
| Test ratio | 0.15 | `TEST_RATIO` | 23 | Held out for final evaluation. |
| Blackout days (min) | 1 | `BLACKOUT_DAYS_MIN` | 24 | Prevent leakage at split boundaries. |
| Blackout days (max) | 3 | `BLACKOUT_DAYS_MAX` | 25 | Upper bound on blackout period. |

### Graph Construction (`scripts/data/04_build_graph.py`)

| Parameter | Value | Variable | Line | Why |
|-----------|-------|----------|------|-----|
| Co-occurrence window | 5 steps | `CO_EVENT_WINDOW` | 22 | Captures most items in median-length sessions (4 items). |

---

## Model Parameters

### Shared Defaults (all models)

Configured in `params.yaml` and model `__init__` signatures:

| Parameter | Default | `params.yaml` Key | Purpose |
|-----------|---------|-------------------|---------|
| `num_items` | Dataset-dependent | Computed at runtime | Total items in catalog. |
| `embedding_dim` | 256 | `model.embedding_dim` | Dimension of item embeddings. |
| `hidden_dim` | 256 | `model.hidden_dim` | Dimension of GNN hidden layers. |
| `num_layers` | 2 | `model.num_layers` | Number of GNN layers. |
| `dropout` | 0.1 | `model.dropout` | Dropout rate. |

### GraphSAGE-Specific (`etpgt/model/graphsage.py`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `aggregator` | `"mean"` | Neighbor aggregation type. Options: mean, max, lstm. |
| `readout_type` | `"mean"` | Session readout type. Options: mean, max, last, attention. |

### GAT-Specific (`etpgt/model/gat.py`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `num_heads` | 4 | Number of attention heads. |
| `concat_heads` | `False` | If True: output dim = hidden_dim * num_heads. If False: average heads. |
| `readout_type` | `"mean"` | Session readout type. |

### Graph Transformer - Standard (`etpgt/model/graph_transformer.py`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `num_layers` | 3 | Transformer layers. |
| `num_heads` | 4 | Attention heads (each head: hidden_dim / num_heads = 64 dims). |
| `use_laplacian_pe` | `True` | Enable Laplacian positional encoding. |
| `laplacian_k` | 16 | Number of Laplacian eigenvectors. |
| `use_ffn` | `True` | Enable feed-forward network after each attention layer. |
| `ffn_expansion` | 4 | FFN inner dim = hidden_dim * 4 = 1024. |
| `readout_type` | `"mean"` | Session readout type. |

### Graph Transformer - Optimized (`create_graph_transformer_optimized`)

These differ from the standard version:

| Parameter | Optimized Value | Standard Value | Why Changed |
|-----------|----------------|---------------|-------------|
| `num_layers` | **2** | 3 | Over-smoothing at 3+ layers with avg degree 18. |
| `num_heads` | **2** | 4 | Sufficient for this graph structure. |
| `use_ffn` | **`False`** | `True` | 29x speedup. FFN underutilized for graph recommendation. |
| `ffn_expansion` | **2** | 4 | Reduced, only used if FFN is re-enabled. |

### Laplacian PE (`etpgt/encodings/laplacian_pe.py`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `k` | 16 | Number of eigenvectors to compute. |
| `embedding_dim` | 256 | Output dimension after linear projection. |
| `normalization` | `"sym"` | Laplacian normalization. Options: sym (symmetric), rw (random walk). |

---

## Training Parameters

### Main Training Config (`params.yaml`)

```yaml
train:
  batch_size: 32
  lr: 0.001
  num_epochs: 100

model:
  embedding_dim: 256
  hidden_dim: 256
  num_layers: 2
  num_heads: 2
  dropout: 0.1
```

### Trainer (`etpgt/train/trainer.py`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_epochs` | 100 | Maximum training epochs. |
| `patience` | 10 | Early stopping: stop after 10 epochs without improvement. |
| `eval_every` | 1 | Evaluate on validation set every N epochs. |
| `k_values` | `[10, 20]` | K values for Recall@K and NDCG@K metrics. |
| `device` | `"cuda"` | Training device. |

### DataLoader (`etpgt/train/dataloader.py`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `batch_size` | 32 | Sessions per batch. |
| `num_negatives` | 5 | Random negative items per positive target. |
| `max_session_length` | 50 | Truncate sessions longer than this (keep last N). |
| `shuffle` | `True` | Shuffle training data each epoch. |
| `num_workers` | 0 | DataLoader worker processes. |

### Optimizer

| Parameter | Value | Set In |
|-----------|-------|--------|
| Type | AdamW | `scripts/train/train_baseline.py` |
| Learning rate | 0.001 | `params.yaml` |
| Weight decay | 1e-5 | `scripts/train/train_baseline.py` |

### Loss Functions (`etpgt/train/losses.py`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `loss_type` | `"dual"` | Loss type: bpr, listwise, dual, sampled_softmax. |
| `alpha` | 0.7 | DualLoss: weight for listwise component. |
| `temperature` | 1.0 | Softmax temperature for listwise/sampled_softmax. |

---

## Advanced Training Config

For extended experiments, use YAML configs in `configs/`:

### `configs/yoochoose_etpgt_small.yaml`

```yaml
Training:
  batch_size: 256
  epochs: 50
  lr: 0.001
  weight_decay: 0.0001
  warmup_steps_ratio: 0.05
  gradient_clip: 1.0
  early_stop_patience: 5
  early_stop_metric: ndcg@20
  use_amp: true

Loss:
  listwise_weight: 0.7
  contrastive_weight: 0.3
  candidates_k: 200

Model:
  hidden_dim: 256
  num_layers: 3
  num_heads: 4
  dropout: 0.2
  drop_path: 0.1
  lappe_k: 16
  use_layer_scale: false
  use_cls_token: true
  use_gated_pooling: true

Sampling:
  fanout: [16, 12, 8]
  last_n: 10
  max_edges_per_batch: 10000

Evaluation:
  k_values: [10, 20]
  stratify_by:
    - session_length
    - last_gap_dt
    - cold_items
```

### `configs/quality_thresholds.yaml`

Defines minimum quality gates for model deployment. Models must pass these thresholds before serving.

---

## Serving Parameters

### FastAPI Server (`scripts/serve/vertex_app.py`)

| Parameter | Default | Environment Variable | Purpose |
|-----------|---------|---------------------|---------|
| Model path | - | `MODEL_PATH` | Path to ONNX or PyTorch model file |
| Embeddings path | - | `EMBEDDINGS_PATH` | Path to item_embeddings.npy (ONNX mode) |
| Inference mode | `"onnx"` | `INFERENCE_MODE` | `"onnx"` or `"pytorch"` |
| Default K | 10 | - | Default number of recommendations |
| Host | `0.0.0.0` | - | Server bind address |
| Port | 8080 | `PORT` | Server port |

### ONNX Export (`scripts/pipeline/export_onnx.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| ONNX opset version | 14 | ONNX operator set version |
| Validation threshold | 1e-5 | Max allowed difference between PyTorch and ONNX outputs |

---

## DVC Pipeline Parameters (`dvc.yaml`)

Stage 5 (train) references these `params.yaml` keys:

```yaml
params:
  - train.batch_size
  - train.lr
  - train.num_epochs
  - model.embedding_dim
  - model.hidden_dim
```

Changing any of these and running `dvc repro` will re-trigger the training stage.

---

## Where to Change Parameters

| To change... | Edit this file | Then run |
|-------------|---------------|----------|
| Training hyperparams (batch, lr, epochs) | `params.yaml` | `dvc repro` |
| Model architecture (layers, heads, dims) | `params.yaml` | `dvc repro` |
| Session gap or min length | `scripts/data/02_sessionize.py` | `dvc repro` (re-runs from sessionize) |
| Train/val/test split ratios | `scripts/data/03_temporal_split.py` | `dvc repro` (re-runs from split) |
| Co-occurrence window | `scripts/data/04_build_graph.py` | `dvc repro` (re-runs from build_graph) |
| Loss function | `scripts/train/train_baseline.py` | `make train` |
| Serving configuration | Environment variables | Rebuild Docker image |
| Advanced training config | `configs/*.yaml` | Passed via CLI args |
