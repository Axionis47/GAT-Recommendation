# Experiments and Results

This document covers the experiments I ran, what I learned, and honest reflections on what worked and what did not.

## Experimental Setup

### Hardware

- Local development: MacBook Pro M1
- Cloud training: GCP Vertex AI with NVIDIA L4 GPU (24GB VRAM)

### Dataset

RetailRocket e-commerce events:
- 172,066 sessions (after filtering)
- 955,778 events
- 82,173 unique items in graph
- 737,716 co-occurrence edges

### Evaluation Metrics

**Recall@K:** Fraction of test samples where the true next item appears in the top K recommendations.

```
Recall@10 = (# samples where true item in top 10) / (# total samples)
```

**NDCG@K:** Normalized Discounted Cumulative Gain. Gives higher scores when the true item appears higher in the ranking.

```
NDCG = DCG / IDCG
DCG = sum(1 / log2(rank + 1) for correct items)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Batch size | 32 |
| Negative samples | 5 per positive |
| Loss function | Listwise (softmax cross-entropy) |
| Early stopping | Patience 10 epochs |

## Main Results

### Full Training (Vertex AI, L4 GPU)

| Model | Recall@10 | Recall@20 | NDCG@10 | Epochs | Best Epoch |
|-------|-----------|-----------|---------|--------|------------|
| GraphSAGE | 14.79% | 18.19% | 9.87% | 44 | 34 |
| GAT | 20.10% | 24.01% | 13.64% | 69 | 59 |
| Graph Transformer (FFN) | 36.66% | 39.61% | 29.75% | 96 | 86 |
| **Graph Transformer (no FFN)** | **38.28%** | **41.29%** | **30.65%** | **67** | **57** |

Graph Transformer (no FFN) achieved 2.6x better Recall@10 than GraphSAGE and 1.9x better than GAT.

**Key insight:** The optimized Graph Transformer without FFN not only trains faster but actually achieves better accuracy than the full version. This suggests the FFN was causing slight overfitting.

### Training Dynamics

**Graph Transformer (no FFN):**
```
Epoch 10:  Recall@10 = 30.55%  (rapid improvement)
Epoch 20:  Recall@10 = 33.17%
Epoch 30:  Recall@10 = 35.89%
Epoch 40:  Recall@10 = 36.51%
Epoch 50:  Recall@10 = 37.45%
Epoch 57:  Recall@10 = 38.28%  (best)
Epoch 67:  Recall@10 = 38.03%  (slight decay)
```

**GAT:**
```
Epoch 10:  Recall@10 = 14.71%
Epoch 20:  Recall@10 = 17.35%
Epoch 30:  Recall@10 = 18.00%
Epoch 40:  Recall@10 = 18.94%
Epoch 50:  Recall@10 = 19.66%
Epoch 59:  Recall@10 = 20.10%  (best)
```

**GraphSAGE:**
```
Epoch 10:  Recall@10 = 9.56%
Epoch 20:  Recall@10 = 13.04%
Epoch 30:  Recall@10 = 14.22%
Epoch 34:  Recall@10 = 14.79%  (best, early plateau)
```

### Pre-trained Weights

All model checkpoints are available in `checkpoints/`:

```
checkpoints/
├── best_model.pt                           # Symlink to best model
├── graph_transformer_optimized_best.pt    # 38.28% Recall@10
├── graph_transformer_best.pt              # 36.66% Recall@10
├── gat_best.pt                            # 20.10% Recall@10
├── graphsage_best.pt                      # 14.79% Recall@10
├── *_history.json                          # Training histories
```

### Quick Validation (Local, 3 epochs)

I also ran quick validation tests to verify all models work:

| Model | Final Loss | Parameters | Duration |
|-------|------------|------------|----------|
| GraphSAGE | 1.61 | 28,800 | 29ms |
| GAT | 1.47 | 29,312 | 25ms |
| GraphTransformer (FFN) | 1.29 | 112,128 | 21ms |
| GraphTransformer (no FFN) | 1.21 | 45,952 | 7ms |

Lower loss is better. These numbers are from the pipeline validation with a small batch.

## Ablation Studies

### 1. FFN vs No FFN

**Question:** Does the feed-forward network in Graph Transformer help?

**Setup:** Train Graph Transformer with and without FFN layers on the same data.

**Result:**

| Variant | Training Time | Relative Accuracy |
|---------|---------------|-------------------|
| With FFN | 40 hours/epoch | 100% |
| Without FFN | 27 minutes/epoch | 97% |

**Speedup: 88x**

**Conclusion:** For this task, the FFN capacity is underutilized. The attention mechanism with Laplacian PE provides sufficient inductive bias. Removing FFN gives massive speedup with minimal accuracy loss.

**Why this makes sense:** FFN adds capacity for learning complex nonlinear transformations. But in graph recommendation:
- The input is already structured (graph topology)
- Laplacian PE encodes global position
- The task is similarity-based (dot product ranking)

The attention layers capture the important relationships. FFN just adds unnecessary computation.

### 2. Laplacian PE vs No PE

**Question:** Do Laplacian positional encodings help?

**Setup:** Train Graph Transformer with and without Laplacian PE.

**Result:** With Laplacian PE, the model converges faster and achieves better final accuracy.

**Why this works:** Without PE, nodes with identical local neighborhoods produce identical embeddings. This is a problem because:
- Popular items have many neighbors, some similar, some different
- Hub items (homepage products) connect to many categories
- PE lets the model distinguish between structurally different nodes

### 3. Number of Layers

**Question:** How many GNN layers should we use?

**Setup:** Test 1, 2, 3, and 4 layer models.

**Result:**

| Layers | Performance | Notes |
|--------|-------------|-------|
| 1 | Worse | Only sees direct neighbors |
| 2 | Best | Sweet spot for this graph |
| 3 | Slightly worse | Beginning of over-smoothing |
| 4 | Significantly worse | Over-smoothing problem |

**Why 2 layers is optimal:**

Our graph has average degree 18. After 2 hops, a node already "sees" most of the graph. More layers cause over-smoothing: all node embeddings converge to similar values because they aggregate from overlapping neighborhoods.

```
1 layer:  See 18 neighbors
2 layers: See ~324 nodes (18^2)
3 layers: See ~5,832 nodes (18^3) -> most of the graph
4 layers: Information collapse
```

### 4. Negative Sampling Count

**Question:** How many negative samples per positive?

**Setup:** Test 1, 3, 5, 10, and 20 negatives.

**Result:**

| Negatives | Training Speed | Accuracy |
|-----------|----------------|----------|
| 1 | Fastest | Weak signal |
| 3 | Fast | Okay |
| 5 | Moderate | Good |
| 10 | Slow | Marginal improvement |
| 20 | Very slow | Diminishing returns |

I chose 5 negatives as the best trade-off.

## Cost Analysis

Training costs on GCP Vertex AI with NVIDIA L4 GPU ($0.47/hour):

| Model | GPU Hours | Cost |
|-------|-----------|------|
| GraphSAGE (100 epochs) | 47 | $22 |
| GAT (100 epochs) | 64 | $30 |
| Graph Transformer (FFN, 100 epochs) | 4,000 | $1,880 |
| Graph Transformer (no FFN, 100 epochs) | 45 | $21 |

The FFN ablation is critical for cost: it reduces training cost by 89x.

## What I Learned

### Things that worked

1. **Temporal train/test splits prevent data leakage.** Many tutorials use random splits, which inflates metrics. Temporal splits with blackout periods simulate the real deployment scenario.

2. **Laplacian PE gives significant improvement.** It solves the structural equivalence problem where nodes with identical neighborhoods get identical embeddings.

3. **Removing FFN is a huge win.** 88x speedup with minimal accuracy loss. This should be the default for graph recommendation tasks.

4. **2 layers is enough.** More layers cause over-smoothing on dense co-occurrence graphs.

### Things that surprised me

1. **GAT only moderately beats GraphSAGE.** I expected attention to help more, but GAT's additive attention is limited. The real gain comes from dot-product attention + positional encoding.

2. **Co-occurrence structure matters most.** I considered adding temporal attention biases (weighting recent items higher) but decided against it. The sessions are short (3-5 items on average), and the co-occurrence graph already captures timing implicitly: if items frequently appear together in sessions, they get connected regardless of order. I focused instead on the FFN ablation, which gave concrete 88x speedup.

### What I would do differently

1. **Add online evaluation.** The offline metrics look good, but real value comes from A/B testing with actual users. I would add a simple serving endpoint and measure click-through rate.

2. **Try contrastive learning.** Instead of predicting the next item, train the model to bring similar sessions closer in embedding space. This might generalize better.

3. **Explore graph construction.** I used a fixed co-occurrence window of 5 items. Tuning this or using different edge weighting schemes might help.

4. **Add item features.** The model only uses co-occurrence structure. Adding item categories, prices, or text embeddings could improve cold-start performance.

## Reproducibility

All experiments can be reproduced with:

```bash
# Download data
python scripts/data/01_download_retailrocket.py

# Run data pipeline
make data

# Validate all models
python scripts/pipeline/run_full_pipeline.py --num-sessions 100 --num-epochs 3

# Full training (requires GPU)
python scripts/train/train_baseline.py --model graph_transformer_optimized
```

Results are logged to:
- `data/interim/session_stats.json` - Sessionization statistics
- `data/processed/split_info.json` - Train/val/test split info
- `data/processed/graph_stats.json` - Graph construction stats
- `data/test_subset/pipeline_results.json` - Model validation results
