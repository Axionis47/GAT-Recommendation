# Model Architectures

This document explains every model in the system: how they work, what every parameter does, and why each design decision was made.

> **Why these specific values?** Every parameter traces back to the data. Session length drives graph density, which drives layer count, which drives model capacity. See [Design Rationale](DESIGN_RATIONALE.md) for the full reasoning chain.

## Shared Architecture

All four models follow the same pattern:

```
Session Items [item_1, item_2, ..., item_N]
       |
       v
Item Embedding Layer  (nn.Embedding)
       |
       v
[num_items, 256] float vectors
       |
       v
GNN Layers  (model-specific message passing)
       |
       v
Session Readout  (mean pooling over nodes)
       |
       v
[batch_size, 256] session embedding
       |
       v
Dot Product with all item embeddings
       |
       v
Top-K Recommendations
```

### Base Class: `BaseRecommendationModel`

**File:** `etpgt/model/base.py`

Every model inherits from this. It provides:

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Item Embedding | `nn.Embedding(num_items, 256, padding_idx=0)` | Map item IDs to 256-dim vectors |
| Initialization | Xavier uniform (skip padding index 0) | Stable training start |
| `predict()` | `scores = session_emb @ item_emb.T; topk(scores, k)` | Dot-product ranking |
| `compute_loss()` | BPR: `-log(sigmoid(pos - neg))` | Default contrastive loss |
| `get_item_embeddings()` | Returns `self.item_embedding.weight` | For ONNX export |

### Session Readout: `SessionReadout`

**File:** `etpgt/model/base.py`, lines 116-193

Aggregates all node embeddings in a session subgraph into one session vector.

| Type | Formula | Used When |
|------|---------|-----------|
| `mean` | `session = mean(node_embeddings)` | Default. Works best overall. |
| `max` | `session = max(node_embeddings, dim=0)` | Emphasize dominant signals. |
| `last` | `session = node_embeddings[-1]` | Recency matters most. |
| `attention` | `session = sum(softmax(W * h) * h)` | Learned importance weighting. |

All models in this project use `mean` readout.

---

## Model 1: GraphSAGE (Baseline)

**File:** `etpgt/model/graphsage.py`
**Factory:** `create_graphsage()`

### Architecture

```
Item Embedding (num_items x 256)
       |
       v
SAGEConv Layer 1 (256 -> 256)
       |-> BatchNorm1d(256)
       |-> ReLU
       |-> Dropout(0.1)
       v
SAGEConv Layer 2 (256 -> 256)
       |-> BatchNorm1d(256)
       |-> ReLU
       |-> Dropout(0.1)
       v
SAGEConv Layer 3 (256 -> 256)
       |-> BatchNorm1d(256)
       |-> ReLU
       |-> Dropout(0.1)
       v
SessionReadout (mean)
       |
       v
Session Embedding [batch_size, 256]
```

### The Math

GraphSAGE aggregates neighbor embeddings using a simple mean:

```
h_v^(l+1) = ReLU( W * MEAN({ h_u^(l) : u in N(v) }) )
```

Where:
- `h_v^(l)` = embedding of node v at layer l
- `N(v)` = neighbors of v in the session subgraph
- `W` = learnable weight matrix

This is the simplest GNN. It treats all neighbors equally. No attention. No positional awareness.

### Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `embedding_dim` | 256 | Standard for recommendation. Large enough for 82K items. |
| `hidden_dim` | 256 | Same as embedding for residual-friendly architecture. |
| `num_layers` | 3 | Default, but 2 is better (see ablations). |
| `dropout` | 0.1 | Light regularization. |
| `aggregator` | `"mean"` | Most stable. Max and LSTM gave similar results. |
| `readout_type` | `"mean"` | Simple and effective. |

### Results

| Metric | Value |
|--------|-------|
| Recall@10 | 14.79% |
| Recall@20 | 18.19% |
| NDCG@10 | 9.87% |
| Best epoch | 34 |
| Training time | 12 hours |
| Parameters | 28,800 |

### Why this model is limited

GraphSAGE treats all neighbors equally. If item A is connected to items B, C, D, E, it averages all four embeddings with equal weight. But in reality, B might be highly relevant while D is noise. GraphSAGE cannot learn this distinction.

---

## Model 2: GAT (Graph Attention Network)

**File:** `etpgt/model/gat.py`
**Factory:** `create_gat()`

### Architecture

```
Item Embedding (num_items x 256)
       |
       v
GATConv Layer 1 (256 -> 256, 4 heads, averaged)
       |-> BatchNorm1d(256)
       |-> ReLU
       |-> Dropout(0.1)
       v
GATConv Layer 2 (256 -> 256, 4 heads, averaged)
       |-> BatchNorm1d(256)
       |-> ReLU
       |-> Dropout(0.1)
       v
GATConv Layer 3 (256 -> 256, 4 heads, averaged)
       |-> BatchNorm1d(256)
       [no activation on last layer]
       v
SessionReadout (mean)
       |
       v
Session Embedding [batch_size, 256]
```

### The Math

GAT uses **additive attention** to weight neighbors differently:

```
e(v, u) = LeakyReLU( a^T * [W*h_v || W*h_u] )
alpha(v, u) = softmax_over_neighbors( e(v, u) )
h_v^(l+1) = sum( alpha(v, u) * W * h_u )  for u in N(v)
```

Where:
- `||` = concatenation
- `a` = learnable attention vector
- `alpha` = normalized attention weight (how much to attend to each neighbor)

With 4 heads, 4 independent attention mechanisms run in parallel and results are averaged.

### Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `embedding_dim` | 256 | Same as baseline for fair comparison. |
| `hidden_dim` | 256 | Keeps dimension consistent. |
| `num_layers` | 3 | Same as baseline. |
| `num_heads` | 4 | 4 independent attention patterns. |
| `dropout` | 0.1 | Applied to attention weights and features. |
| `concat_heads` | `False` | Average heads instead of concatenating. Keeps dim at 256. |
| `readout_type` | `"mean"` | Consistent across models. |

### Results

| Metric | Value |
|--------|-------|
| Recall@10 | 20.10% |
| Recall@20 | 24.01% |
| NDCG@10 | 13.64% |
| Best epoch | 59 |
| Training time | 16 hours |
| Parameters | 29,312 |

### Why GAT improves over GraphSAGE

GAT can learn which neighbors matter more. If item A is connected to items B (relevant) and D (noise), GAT assigns higher attention to B and lower attention to D. This selective aggregation gives a 36% improvement in Recall@10 over GraphSAGE.

### Why GAT is still limited

GAT uses **additive** attention. The attention score depends on a learned vector `a`, not on the actual content of the embeddings. This limits the expressiveness. Dot-product attention (used in Transformers) compares embeddings directly, which is more powerful.

---

## Model 3: Graph Transformer (Standard)

**File:** `etpgt/model/graph_transformer.py`
**Factory:** `create_graph_transformer()`

### Architecture

```
Item Embedding (num_items x 256)
       |
       v
Laplacian Positional Encoding (k=16 eigenvectors -> 256)
       |-> x = embedding + PE
       v
TransformerConv Layer 1 (256 -> 256, 4 heads, concat, gated residual)
       |-> BatchNorm1d(256)
       |-> Residual connection
       |-> Dropout(0.1)
       |-> FFN: Linear(256 -> 1024) -> GELU -> Dropout -> Linear(1024 -> 256) -> Dropout
       |-> Residual connection
       v
TransformerConv Layer 2 (256 -> 256, same config)
       |-> [same as above]
       v
TransformerConv Layer 3 (256 -> 256, same config)
       |-> [same as above]
       v
SessionReadout (mean)
       |
       v
Session Embedding [batch_size, 256]
```

### The Math

Graph Transformer uses **scaled dot-product attention** between neighbors:

```
Q = W_Q * h_v        (query from target node)
K = W_K * h_u        (key from each neighbor)
V = W_V * h_u        (value from each neighbor)

alpha(v, u) = softmax( (Q * K) / sqrt(d_head) )
h_v' = sum( alpha(v, u) * V )

With gated residual (beta=True):
h_v_out = beta * h_v' + (1 - beta) * h_v
```

The FFN adds a two-layer MLP after each attention layer:
```
FFN(x) = Linear_2( Dropout( GELU( Linear_1(x) ) ) )
       = W_2 * GELU(W_1 * x + b_1) + b_2
```

### Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `embedding_dim` | 256 | Item representation dimension. |
| `hidden_dim` | 256 | TransformerConv output dimension. |
| `num_layers` | 3 | Standard depth. |
| `num_heads` | 4 | 4 independent attention heads (64 dims each). |
| `dropout` | 0.1 | Regularization. |
| `use_laplacian_pe` | `True` | Critical for distinguishing structurally different nodes. |
| `laplacian_k` | 16 | 16 eigenvectors capture enough graph structure. |
| `use_ffn` | `True` | Feed-forward network after each attention layer. |
| `ffn_expansion` | 4 | FFN inner dimension = 256 * 4 = 1024. |
| `readout_type` | `"mean"` | Session embedding aggregation. |

### Results

| Metric | Value |
|--------|-------|
| Recall@10 | 36.66% |
| Recall@20 | 39.61% |
| NDCG@10 | 29.75% |
| Best epoch | 86 |
| Training time | 40+ hours/epoch |
| Parameters | 112,128 |

### Cost Problem

**This model was estimated at $1,880 for 100 epochs on an NVIDIA L4 GPU at $0.47/hour.**

With a $300 budget spread across multiple projects, this was never going to happen. You would need to sell a kidney and a half just for the compute bill. So we built the optimized version instead.

---

## Model 4: Graph Transformer (Optimized) - THE ONE WE USE

**File:** `etpgt/model/graph_transformer.py`
**Factory:** `create_graph_transformer_optimized()`

### What changed

Three surgical cuts, each with a clear speedup:

| Change | From | To | Speedup |
|--------|------|----|---------|
| Remove FFN layers | `use_ffn=True` | `use_ffn=False` | 29x |
| Reduce layers | 3 | 2 | 1.5x |
| Reduce attention heads | 4 | 2 | ~2x |
| **Combined** | | | **~88x** |

### Architecture

```
Item Embedding (num_items x 256)
       |
       v
Laplacian Positional Encoding (k=16 eigenvectors -> 256)
       |-> x = embedding + PE
       v
TransformerConv Layer 1 (256 -> 256, 2 heads, concat, gated residual)
       |-> BatchNorm1d(256)
       |-> Residual connection
       |-> Dropout(0.1)
       v
TransformerConv Layer 2 (256 -> 256, 2 heads, concat, gated residual)
       |-> BatchNorm1d(256)
       |-> Residual connection
       |-> Dropout(0.1)
       v
SessionReadout (mean)
       |
       v
Session Embedding [batch_size, 256]
```

No FFN. No GELU. No 1024-dim expansion. Just attention, norm, residual, dropout. Done.

### Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `embedding_dim` | 256 | Same as standard. |
| `hidden_dim` | 256 | Same as standard. |
| `num_layers` | **2** | Avg degree 18. After 2 hops, each node sees ~324 nodes. 3 layers = over-smoothing. |
| `num_heads` | **2** | 2 heads (128 dims each). Enough for this graph. |
| `dropout` | 0.1 | Same as standard. |
| `use_laplacian_pe` | `True` | Still critical. |
| `laplacian_k` | 16 | Same as standard. |
| `use_ffn` | **`False`** | The big one. 29x speedup. FFN is underutilized for graph recommendation. |
| `ffn_expansion` | 2 | Only matters if FFN is re-enabled. Reduced from 4. |
| `readout_type` | `"mean"` | Same as standard. |

### Results

| Metric | Value |
|--------|-------|
| Recall@10 | **38.28%** |
| Recall@20 | **41.29%** |
| NDCG@10 | **30.65%** |
| Best epoch | 57 |
| Training time | **15.5 hours** (total) |
| Parameters | **46,000** |
| Estimated cost | **~$21** for 100 epochs |

### The punchline

The optimized model is not just cheaper. **It actually performs better.** 38.28% vs 36.66% Recall@10.

Why? The FFN was causing slight overfitting. With 112K parameters and relatively small session subgraphs, the FFN had more capacity than the data could support. Removing it acts as implicit regularization.

### Why removing FFN works for this task

In standard NLP Transformers, the FFN is essential. It provides the model's capacity for complex nonlinear transformations.

But graph recommendation is different:
1. **The input is already structured.** Graph topology encodes relationships. No need to learn them from raw text.
2. **Laplacian PE encodes global position.** The FFN is not needed for positional awareness.
3. **The task is similarity-based.** We just need good embeddings for dot-product ranking. We do not need complex reasoning.
4. **Sessions are short.** Mean length 5.55 items. The attention mechanism alone captures enough.

The attention layers do the heavy lifting. The FFN was adding compute without adding value.

---

## Model Comparison Summary

| | GraphSAGE | GAT | Graph Transformer | GT Optimized |
|---|-----------|-----|-------------------|--------------|
| **Attention type** | None (mean) | Additive | Dot-product | Dot-product |
| **Positional encoding** | No | No | Laplacian PE | Laplacian PE |
| **FFN layers** | No | No | Yes | **No** |
| **Layers** | 3 | 3 | 3 | **2** |
| **Heads** | - | 4 | 4 | **2** |
| **Parameters** | 28.8K | 29.3K | 112.1K | **46.0K** |
| **Recall@10** | 14.79% | 20.10% | 36.66% | **38.28%** |
| **Training cost** | ~$22 | ~$30 | **~$1,880** | **~$21** |
| **Actually trained?** | Yes | Yes | **No** | **Yes** |

---

## Loss Functions

**File:** `etpgt/train/losses.py`

### BPR Loss (Bayesian Personalized Ranking)

The default loss. Optimizes pairwise ranking.

```
loss = -log( sigmoid(score_positive - score_negative) )
```

For each session: the target item should score higher than every negative sample. The loss penalizes violations.

### Listwise Loss

Treats ranking as classification. The target item is class 0, negatives are classes 1 through N.

```
scores = [pos_score, neg_1_score, ..., neg_N_score]
loss = cross_entropy(scores / temperature, target=0)
```

### Dual Loss

Combines both:
```
loss = 0.7 * listwise + 0.3 * bpr
```

The alpha=0.7 weighting was determined empirically. Listwise loss provides a stronger gradient signal, while BPR adds pairwise margin enforcement.

### Sampled Softmax Loss

Same as Listwise. Provided as an alias for API consistency with literature that uses this term.

### Factory

```python
loss_fn = create_loss_function(
    loss_type="dual",     # or "bpr", "listwise", "sampled_softmax"
    alpha=0.7,            # weight for listwise in dual loss
    temperature=1.0,      # softmax temperature
)
```

---

## Evaluation Metrics

**File:** `etpgt/utils/metrics.py`

### Recall@K

"Did the true next item appear in the top K recommendations?"

```
Recall@10 = (number of sessions where target is in top 10) / (total sessions)
```

38.28% means: out of all test sessions, the model correctly places the actual next item in its top 10 list 38% of the time.

### NDCG@K (Normalized Discounted Cumulative Gain)

"Not just whether the item appears, but how high it ranks."

```
If target is at position p in the top-K list:
  DCG = 1 / log2(p + 2)
  NDCG = DCG / IDCG    where IDCG = 1 / log2(2) = 1.0
```

An item at position 1 scores 1.0. Position 2 scores 0.63. Position 10 scores 0.28. This rewards models that rank the target item higher.
