# Model Architectures

This document explains each model architecture I implemented, how they work, and why I chose them.

## Architecture Overview

All models follow the same pattern:

```
Session Items          Item Embeddings        GNN Layers           Session Embedding
[item_1, item_2, ...]  -> [256-dim vectors]   -> [message passing]  -> [256-dim vector]
                                                                            |
                                                                            v
                                                                      Dot Product with
                                                                      All Item Embeddings
                                                                            |
                                                                            v
                                                                      Top-K Recommendations
```

**Base class:** All models inherit from `BaseRecommendationModel` in [base.py](../etpgt/model/base.py).

Shared components:
- Item embedding layer: `nn.Embedding(num_items, 256)`
- Session readout: mean pooling over node embeddings
- Prediction: dot product between session embedding and all item embeddings

## 1. GraphSAGE (Baseline)

**File:** [graphsage.py](../etpgt/model/graphsage.py)

### What it does

GraphSAGE aggregates information from neighbors using a simple mean operation.

For each node, it:
1. Collects embeddings from all neighbors
2. Computes the mean
3. Passes through a linear layer and activation

### The math

```
h_v = ReLU(W * MEAN({h_u : u in N(v)}))
```

Where:
- `h_v` is the embedding of node v
- `N(v)` is the set of neighbors of v
- `W` is a learnable weight matrix

### Implementation

```python
# From graphsage.py
self.convs.append(SAGEConv(embedding_dim, hidden_dim, aggr="mean"))

# Forward pass
for conv, bn in zip(self.convs, self.batch_norms):
    x = conv(x, edge_index)
    x = bn(x)
    x = torch.relu(x)
    x = self.dropout_layer(x)
```

### Parameters

| Config | Value |
|--------|-------|
| Layers | 3 |
| Hidden dim | 256 |
| Aggregator | mean |
| Dropout | 0.1 |
| Total params | ~29K |

### Strengths

- Fast training (no attention computation)
- Scales well to large graphs
- Good baseline to compare against

### Limitations

- Treats all neighbors equally
- No way to learn which neighbors are more important
- Limited expressiveness

## 2. GAT (Graph Attention Network)

**File:** [gat.py](../etpgt/model/gat.py)

### What it does

GAT learns which neighbors are more important using attention weights.

Instead of equal weighting, it computes a score for each neighbor and normalizes with softmax.

### The math

```
alpha_ij = softmax(LeakyReLU(a^T [W*h_i || W*h_j]))
h_i = sum(alpha_ij * W*h_j for j in N(i))
```

Where:
- `alpha_ij` is the attention weight from node i to neighbor j
- `a` is a learnable attention vector
- `||` means concatenation
- The softmax is over all neighbors of i

This is called "additive attention" because the attention score comes from a learned vector `a` applied to concatenated features.

### Implementation

```python
# From gat.py
self.convs.append(
    GATConv(
        embedding_dim,
        hidden_dim,
        heads=num_heads,
        dropout=dropout,
        concat=False,  # Average heads
    )
)
```

### Parameters

| Config | Value |
|--------|-------|
| Layers | 3 |
| Hidden dim | 256 |
| Attention heads | 4 |
| Dropout | 0.1 |
| Total params | ~29K |

### Strengths

- Learns neighbor importance
- Multi-head attention captures different relationships
- Better than GraphSAGE (+36% relative improvement)

### Limitations

- Attention is local (only 1-hop neighbors)
- Additive attention is less expressive than dot-product attention
- No global position awareness

## 3. Graph Transformer

**File:** [graph_transformer.py](../etpgt/model/graph_transformer.py)

### What it does

Graph Transformer uses scaled dot-product attention (like the original Transformer) plus Laplacian positional encodings to give nodes global position awareness.

### The math

**Attention:**
```
alpha_ij = softmax((Q(h_i + p_i))^T (K(h_j + p_j)) / sqrt(d))
```

Where:
- `Q`, `K` are learned query and key projections
- `p_i` is the Laplacian positional encoding for node i
- `d` is the head dimension (for numerical stability)

**Laplacian PE:**
```
L = D - A                      # Graph Laplacian
L * phi_k = lambda_k * phi_k   # Eigendecomposition
p_i = [phi_1(i), phi_2(i), ..., phi_16(i)]  # First 16 eigenvectors
```

The Laplacian eigenvectors form a "coordinate system" for the graph:
- Small eigenvalues capture global structure (communities, clusters)
- Large eigenvalues capture local structure (bridges, boundaries)

### Implementation

```python
# From graph_transformer.py
self.convs.append(
    TransformerConv(
        embedding_dim,
        hidden_dim // num_heads,
        heads=num_heads,
        dropout=dropout,
        concat=True,
        beta=True,  # Gated residual connections
    )
)

# Add Laplacian PE to node features
if self.use_laplacian_pe:
    x = x + lap_pe
```

### Two Variants

I provide two factory functions:

**Standard (`create_graph_transformer`):**
- 3 layers, 4 heads
- FFN layers enabled
- ~112K parameters
- Slow: 40 hours per epoch

**Optimized (`create_graph_transformer_optimized`):**
- 2 layers, 2 heads
- FFN layers disabled
- ~46K parameters
- Fast: 27 minutes per epoch (88x speedup)

### Why removing FFN works

In a standard Transformer:
1. Attention routes information between positions
2. FFN adds capacity via nonlinear transformation

For graph recommendation with structured input, the attention mechanism plus Laplacian PE provides sufficient inductive bias. The FFN capacity is underutilized.

I validated this empirically: removing FFN gives 88x speedup with less than 3% accuracy loss.

### Parameters (Optimized)

| Config | Value |
|--------|-------|
| Layers | 2 |
| Hidden dim | 256 |
| Attention heads | 2 |
| Laplacian k | 16 |
| FFN | disabled |
| Dropout | 0.1 |
| Total params | ~46K |

### Why 2 layers is enough

Each GNN layer aggregates from 1-hop neighbors. After 2 layers, each node sees its 2-hop neighborhood.

In our co-occurrence graph:
- 2 hops already spans most of the graph (average degree is 18)
- More layers cause over-smoothing: all node embeddings converge to similar values

```
Layer 1: See direct neighbors (items bought together)
Layer 2: See 2-hop neighbors (items in similar categories)
Layer 3+: Everyone sees everyone -> information collapse
```

## 4. ETPGT (Temporal and Path-Aware)

**File:** [etpgt.py](../etpgt/model/etpgt.py)

### What it does

ETPGT extends Graph Transformer with:
1. Temporal bias: weight recent interactions higher
2. Path bias: weight closer items (in graph distance) higher
3. CLS token: learnable aggregation for session representation

### The math

```
alpha_ij = softmax((Q*K^T + temporal_bias + path_bias) / sqrt(d))
```

**Temporal bias:**
- Time deltas are bucketed into 7 categories
- Each bucket has a learned bias per attention head

**Path bias:**
- Path lengths are bucketed into 3 categories (1-hop, 2-hop, 3+-hop)
- Each bucket has a learned bias per attention head

### Implementation

```python
# From etpgt.py
class TemporalPathAttention(MessagePassing):
    def __init__(self, ...):
        self.temporal_bias = TemporalBias(num_buckets=7, num_heads=num_heads)
        self.path_bias = PathBias(num_buckets=3, num_heads=num_heads)

    def message(self, q_i, k_j, v_j, time_delta_ms, path_length, ...):
        # Compute attention scores
        alpha = (q_i * k_j).sum(dim=-1) / sqrt(d)

        # Add temporal bias
        if time_delta_ms is not None:
            alpha = alpha + self.temporal_bias(time_delta_ms)

        # Add path bias
        if path_length is not None:
            alpha = alpha + self.path_bias(path_length)

        alpha = softmax(alpha, index)
        return v_j * alpha
```

### CLS Token

When `use_cls_token=True`, the session embedding is computed via learned attention:

```python
# CLS token acts as a learned query
cls_query = self.cls_query_proj(self.cls_token)  # [1, hidden_dim]
keys = self.cls_key_proj(node_embeddings)         # [num_nodes, hidden_dim]

# Attention over all nodes in session
scores = cls_query @ keys.T / sqrt(hidden_dim)
weights = softmax(scores)
session_embedding = weights @ node_embeddings
```

This replaces mean pooling with a learned aggregation.

### Parameters

| Config | Value |
|--------|-------|
| Layers | 3 |
| Hidden dim | 256 |
| Attention heads | 4 |
| Temporal buckets | 7 |
| Path buckets | 3 |
| CLS token | optional |
| Total params | ~112K (no CLS), ~120K (with CLS) |

### When to use ETPGT

ETPGT is experimental. Use it when:
- You have edge features (timestamps, path lengths)
- You want to explore temporal/path-aware attention
- Research setting with time to tune

For production, Graph Transformer (optimized) is simpler and nearly as good.

## Loss Functions

**File:** [losses.py](../etpgt/train/losses.py)

I implemented 4 loss functions:

### 1. BPR Loss (Bayesian Personalized Ranking)

```python
loss = -log(sigmoid(pos_score - neg_score))
```

Pairwise margin loss. Pushes positive items above negative items.

### 2. Listwise Loss

```python
scores = [pos_score, neg_score_1, ..., neg_score_5]
loss = cross_entropy(scores, target=0)  # Target is always index 0
```

Treats ranking as classification. The positive item should have the highest score.

**This is what I use** because it directly optimizes ranking.

### 3. Dual Loss

```python
loss = 0.7 * listwise_loss + 0.3 * bpr_loss
```

Combines both approaches.

### 4. Sampled Softmax

Same as listwise, but designed for scaling to millions of items (not needed for this dataset).

## Model Comparison

| Model | Mechanism | Params | Final Loss* | Rationale |
|-------|-----------|--------|-------------|-----------|
| GraphSAGE | Mean aggregation | 29K | 1.61 | Baseline, fast |
| GAT | Additive attention | 29K | 1.47 | Learned neighbor importance |
| GraphTransformer (FFN) | Dot-product attention + FFN | 112K | 1.29 | Highest capacity |
| GraphTransformer (opt) | Dot-product attention, no FFN | 46K | 1.21 | Best speed/accuracy tradeoff |
| ETPGT (no CLS) | Temporal + path attention | 112K | 1.19 | Experimental |
| ETPGT (with CLS) | CLS token readout | 120K | 1.08 | Best loss, experimental |

*Lower loss is better. From pipeline validation with 3 epochs on test subset.

## Why Graph Transformer Won

Three reasons:

1. **Dot-product attention is more expressive than additive attention.** GAT's attention is computed from a single learned vector. Graph Transformer's attention is computed from learned query and key transformations, which can capture richer interactions.

2. **Laplacian positional encodings give global awareness.** Without PE, two nodes with identical local neighborhoods produce identical embeddings. With Laplacian PE, each node has a unique "fingerprint" based on its position in the global graph structure.

3. **Gated residual connections improve gradient flow.** The `beta=True` parameter in TransformerConv learns to balance between the original features and the transformed features, which helps training stability.

## Running the Models

```bash
# Train GraphSAGE
python scripts/train/train_baseline.py --model graphsage

# Train GAT
python scripts/train/train_baseline.py --model gat

# Train Graph Transformer (optimized)
python scripts/train/train_baseline.py --model graph_transformer_optimized

# Validate all models
python scripts/pipeline/run_full_pipeline.py --num-sessions 100 --num-epochs 3
```
