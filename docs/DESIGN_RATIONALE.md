# Design Rationale: From Data to Parameters

Every parameter in this system traces back to one fact: **sessions are short**. Average 5.55 items, median 4. This single data characteristic cascades into the co-occurrence window, graph density, number of layers, attention heads, and whether the FFN lives or dies.

This document follows that cascade step by step.

---

## The Full Chain

```
SESSION LENGTH (avg 5.55, median 4)
    |
    |-- "How wide should the co-occurrence window be?"
    v
CO-OCCURRENCE WINDOW = 5
    |
    |-- "Window covers full sessions -> every item pair becomes an edge"
    v
GRAPH DENSITY (avg degree 18, 737K edges)
    |
    |-- "Dense graph -> how many hops before over-smoothing?"
    v
NUMBER OF LAYERS = 2
    |
    |-- "Short sessions -> tiny subgraphs at training time (3-5 nodes)"
    v
TINY TRAINING SUBGRAPHS
    |
    |-- "Can't support high-capacity models"
    |
    |---- FFN REMOVED (88x speedup, better accuracy)
    |---- ATTENTION HEADS = 2 (enough for 18 neighbors)
    |---- EMBEDDING DIM = 256 (driven by catalog size, not session length)
    |
    |-- "Little per-session signal -> global graph must carry the knowledge"
    v
LAPLACIAN PE = CRITICAL (distinguishes nodes in dense graph)
```

---

## Step 1: Sessions Are Short

| Statistic | Value |
|-----------|-------|
| Mean session length | 5.55 items |
| Median session length | 4 items |
| Min session length | 3 items (filter threshold) |
| Max session length | 417 items |
| 75th percentile | ~6 items |

Most sessions are 3 to 6 items. A shopper views a few products and leaves. This is the cold-start problem in its purest form: you have almost no signal per visitor.

Everything else follows from this.

---

## Step 2: Session Length -> Window = 5

The co-occurrence window determines which item pairs become edges. Window = 5 means: for each item, connect it to the next 5 items in the session.

**Why 5?**

A 4-item session with window 5:
```
Session: [A, B, C, D]
Window = 5

From A: connect to B (step 1), C (step 2), D (step 3)   -> 3 edges
From B: connect to C (step 1), D (step 2)                -> 2 edges
From C: connect to D (step 1)                             -> 1 edge

Total: 6 edges = C(4,2) = ALL possible pairs
```

Window 5 is greater than or equal to the median session length (4). This means **most sessions become fully connected subgraphs** where every item pair gets an edge.

**What happens with other window sizes?**

| Window | Effect on median session (4 items) | Problem |
|--------|-----------------------------------|---------|
| 2 | Only adjacent pairs. Misses A-C, A-D, B-D. | Loses relationships between items that aren't neighbors in the browsing sequence. |
| 5 | All pairs captured. | Sweet spot. |
| 20 | All pairs captured (same as 5 for short sessions). | In long sessions (50+ items), connects items 20 steps apart. These are often unrelated. |

Window = 5 maximizes coverage for typical sessions without over-connecting long sessions.

---

## Step 3: Window -> Graph Density (Avg Degree 18)

Because the window covers most sessions fully, most sessions produce complete subgraphs. This makes the global graph dense.

**The math:**

- 120,436 training sessions
- Average session: ~5.55 items -> ~C(5,2) = 10 item pairs per session
- Many of these pairs overlap across sessions (popular items appear in thousands of sessions)
- Result: 82,173 nodes, 737,716 edges, average degree 17.96

**This density is a direct consequence of short sessions + full-coverage window.**

If sessions were 50 items long and window was still 5, each session would only create edges between nearby items (5 out of 50). Most pairs would NOT be connected. The graph would be much sparser, maybe average degree 5.

```
Short sessions + window >= median  -->  Dense graph (degree 18)
Long sessions  + window << length  -->  Sparse graph (degree ~5)
```

---

## Step 4: Graph Density -> 2 Layers

The number of GNN layers determines how far each node can "see" in the graph. Each layer expands the receptive field by one hop.

**Receptive field with average degree 18:**

```
Layer 0: Node sees itself                    =     1 node
Layer 1: Node sees its neighbors             =    18 nodes
Layer 2: Node sees neighbors of neighbors    =   324 nodes  (18^2)
Layer 3: Node sees 3-hop neighborhood        = 5,832 nodes  (18^3)
Layer 4: Node sees 4-hop neighborhood        = ~100K nodes  (most of graph)
```

**Why 2 is the sweet spot:**

- **1 layer:** 18 nodes. Only direct neighbors. Not enough context for meaningful recommendations.
- **2 layers:** 324 nodes (0.4% of graph). Enough to capture local clusters of related items. Each item "knows" about its extended neighborhood.
- **3 layers:** 5,832 nodes (7% of graph). Different nodes start seeing the same neighborhoods. Embeddings begin converging. This is over-smoothing.
- **4 layers:** Most of the graph. All embeddings collapse toward the graph's mean. Information destroyed.

**The ablation confirmed this:** 2 layers > 1 > 3 > 4 in terms of Recall@10.

**If the graph were sparser (degree 5):**

```
Layer 2: 5^2  =    25 nodes  (too small, need more layers)
Layer 3: 5^3  =   125 nodes  (reasonable coverage)
Layer 4: 5^4  =   625 nodes  (good coverage without over-smoothing)
```

Sparser graph -> more layers needed. Dense graph -> fewer layers needed. Our graph is dense because sessions are short.

---

## Step 5: Short Sessions -> Tiny Training Subgraphs

At training time, each session extracts its own subgraph from the global graph. The model trains on these subgraphs, not the full graph.

A typical training sample:
```
Session: [item_42, item_1337, item_99, item_55]
Context (input): [item_42, item_1337, item_99]   -> 3 nodes
Target (predict): item_55

Subgraph: edges from global graph where BOTH endpoints are in context
  Maybe: (42, 1337), (42, 99), (1337, 99) -> 3 edges

Training subgraph: 3 nodes, 3 edges
```

**The model sees 3-5 nodes and 3-10 edges per training step.** This is tiny. It has massive consequences for model capacity.

---

## Step 6: Tiny Subgraphs -> No FFN

The FFN (Feed-Forward Network) in standard Transformers is a two-layer MLP:
```
FFN(x) = Linear(256 -> 1024) -> GELU -> Linear(1024 -> 256)
```

It adds 2 * 256 * 1024 = 524,288 parameters per layer. This capacity is designed for learning complex nonlinear transformations over rich inputs.

**But our training subgraphs have 3-5 nodes.** The FFN has more parameters than data points per forward pass. It memorizes rather than generalizes.

**Evidence:** The model with FFN scores 36.66% Recall@10. Without FFN: 38.28%. The FFN was hurting, not helping. Removing it acts as implicit regularization - the model is forced to rely on attention (which works well on graphs) rather than brute-force capacity (which overfits on tiny subgraphs).

**The 88x speedup is a bonus.** The real reason to remove FFN is that tiny subgraphs cannot support it. The speedup (40 hours/epoch -> 27 minutes/epoch) is a consequence, not the motivation.

**When would FFN help?**
- Longer sessions (30-50 items) -> larger subgraphs with more nodes
- Richer node features (text embeddings, item metadata) -> more complex transformations needed
- Denser subgraphs with many edges -> more information to process

---

## Step 7: Tiny Subgraphs -> 2 Attention Heads

Each attention head learns one pattern of neighbor importance. With average degree 18, each node has about 18 neighbors to attend to.

**2 heads is enough because:**

The main distinction the model needs to learn is:
1. "Same category" neighbors (shoes -> other shoes)
2. "Complementary" neighbors (shoes -> socks, laces)

Two heads can capture these two patterns. More heads slice the 256-dim embedding into smaller chunks (4 heads = 64 dims per head), reducing each head's capacity on already-tiny subgraphs.

**The progression:**
- 1 head: one attention pattern. Can rank neighbors but cannot distinguish types.
- 2 heads: two patterns. Enough for category vs complement distinction.
- 4 heads: 64 dims per head. On a 3-node subgraph, diminishing returns.
- 8 heads: 32 dims per head. Not enough capacity per head.

---

## Step 8: Short Sessions -> Global Graph IS the Signal

This is the most important insight in the entire system.

**Long sessions (50 items):** The session itself contains rich signal. 49 context items, complex browsing patterns, category transitions, price comparisons. A sequence model (RNN, Transformer) could work without any graph.

**Short sessions (4 items):** 3 context items. That is almost nothing. You cannot learn meaningful patterns from 3 data points.

**The graph solves this.** When the GNN processes a 3-node subgraph, each node's embedding is not just its own embedding. It has been enriched by the global graph structure through the GNN layers:

```
Without graph: item_42's embedding = just item_42 (1 data point)
With graph:    item_42's embedding = item_42 + influence from 324 nodes
               (via 2-layer GNN on global graph with degree 18)
```

The session is the query: "These 3 items are what the user cares about." The graph is the knowledge base: "Here is what 120,436 sessions tell us about how items relate."

**This is why Laplacian PE is critical.**

In a dense graph (degree 18), many nodes have similar local neighborhoods. Item A and item B might both be connected to 15 of the same popular products. Without positional encoding, the GNN produces identical embeddings for A and B. Laplacian eigenvectors give each node a unique structural fingerprint based on its global position, breaking this symmetry.

If the graph were sparse, nodes would have more distinctive neighborhoods naturally, and PE would matter less.

---

## Step 9: Embedding Dimension (256)

This is the one parameter NOT driven by session length. It is driven by **catalog size**.

82,173 items need to be encoded as distinguishable points in embedding space. 256 dimensions provides enough room:

- 2^256 possible distinct binary vectors in 256 dimensions
- 82K items need to be well-separated, not just distinct
- 128 dims: works but less room for nuanced relationships
- 256 dims: sweet spot between expressiveness and compute
- 512 dims: marginal gains, doubles memory and compute

The embedding dimension also needs to match the hidden dimension (256) for clean residual connections.

---

## The Complete Parameter Table (With Reasoning Chain)

| Parameter | Value | Driven By | Reasoning |
|-----------|-------|-----------|-----------|
| `co_event_window` | 5 | Session length (median 4) | Window >= median -> full session coverage |
| `num_layers` | 2 | Graph density (degree 18) | 2 hops = 324 nodes. 3 hops = over-smoothing. |
| `num_heads` | 2 | Subgraph size (3-5 nodes) | 2 attention patterns sufficient for tiny subgraphs |
| `use_ffn` | False | Subgraph size (3-5 nodes) | FFN overfits on tiny subgraphs. Removal = regularization. |
| `use_laplacian_pe` | True | Graph density (degree 18) | Dense graph -> similar neighborhoods -> need PE to distinguish |
| `laplacian_k` | 16 | Graph structure | 16 eigenvectors capture enough global structure |
| `embedding_dim` | 256 | Catalog size (82K items) | Enough dimensions for 82K distinguishable embeddings |
| `hidden_dim` | 256 | embedding_dim | Match for residual connections |
| `dropout` | 0.1 | Standard | Light regularization |
| `batch_size` | 32 | Subgraph size | Tiny subgraphs -> batches are cheap -> 32 is fine |
| `num_negatives` | 5 | Training signal | 5 negatives per positive = sufficient contrast |
| `readout_type` | mean | Session length | Short sessions -> mean pooling over 3-5 nodes works well |

---

## What If the Data Were Different?

### What if sessions were 50 items long?

```
Window = 5 (keep same)
  -> Each session creates edges for 5-step windows, NOT all pairs
  -> Graph becomes sparser (degree ~5 instead of 18)
  -> Need 3-4 layers (not 2) for sufficient receptive field
  -> Larger subgraphs (49 context nodes, many edges)
  -> FFN might help (enough data per subgraph to use the capacity)
  -> 4-8 attention heads (more neighbors to attend to per step)
  -> Laplacian PE less critical (sparser graph = more distinctive neighborhoods)
```

### What if there were 1M items instead of 82K?

```
Embedding dim = 512 or 768 (more items need more room)
  -> Hidden dim increases to match
  -> More attention heads (larger dim per head)
  -> Training cost increases significantly
  -> ONNX model size grows (1M * 512 * 4 bytes = 2 GB embeddings)
```

### What if we had item features (categories, prices, images)?

```
Item embedding = learned embedding + feature projection
  -> FFN becomes more useful (complex feature transformations needed)
  -> Laplacian PE less critical (features already distinguish items)
  -> Cold-start for new items improves (features provide signal even without co-occurrence)
```

---

## Summary

The design is not a collection of independent parameter choices. It is a cascade where each decision follows logically from the data:

1. Sessions are short (median 4 items)
2. Therefore the window covers full sessions (window 5)
3. Therefore the graph is dense (degree 18)
4. Therefore 2 GNN layers are enough (3 = over-smoothing)
5. Therefore training subgraphs are tiny (3-5 nodes)
6. Therefore high-capacity components fail (FFN overfits)
7. Therefore the global graph must carry all the knowledge (Laplacian PE critical)

Change the session length, and most of these parameters should change with it. The numbers are not arbitrary. They are consequences.
