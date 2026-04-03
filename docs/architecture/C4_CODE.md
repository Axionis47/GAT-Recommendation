# C4: Code-Level Walkthroughs

This document walks through the most important code paths line by line. Read this to understand exactly what happens when data flows through the system.

## 1. Graph Transformer Forward Pass

**File:** `etpgt/model/graph_transformer.py`, lines 126-182

This is the core of the system. A batch of sessions enters, session embeddings come out.

```python
def forward(self, batch):
    # Step 1: Look up item embeddings
    # batch.x contains global item IDs: [item_42, item_1337, item_99, ...]
    # After lookup: [num_nodes, 256] float tensor
    x = self.item_embedding(batch.x)

    edge_index = batch.edge_index  # [2, num_edges] - which nodes connect

    # Step 2: Add positional encoding (if enabled)
    # Without this, items with identical neighbors get identical embeddings.
    # Laplacian eigenvectors give each node a unique structural fingerprint.
    if self.use_laplacian_pe:
        if hasattr(batch, "laplacian_pe") and batch.laplacian_pe is not None:
            lap_pe = self.laplacian_pe.project(batch.laplacian_pe)  # [num_nodes, 256]
        else:
            lap_pe = self.laplacian_pe(batch.x)  # compute on-the-fly
        x = x + lap_pe  # element-wise addition

    # Step 3: Apply Transformer layers (optimized path - no FFN)
    for conv, bn in zip(self.convs, self.batch_norms):
        x_residual = x                    # save for residual connection
        x = conv(x, edge_index)           # TransformerConv: multi-head attention
        x = bn(x)                         # batch normalization
        x = x + x_residual               # residual connection (prevents vanishing gradients)
        x = self.dropout_layer(x)         # dropout for regularization

    # Step 4: Aggregate node embeddings into session embeddings
    # batch.batch tells us which nodes belong to which session
    # Mean pooling: average all node embeddings in each session
    session_embeddings = self.readout(x, batch.batch)  # [batch_size, 256]

    return session_embeddings
```

**Shape trace for a batch of 32 sessions:**
```
Input:
  batch.x          = [150]           # 150 unique items across 32 sessions
  batch.edge_index = [2, 280]        # 280 edges in session subgraphs
  batch.batch      = [150]           # which session each item belongs to

After item_embedding:
  x = [150, 256]                     # 256-dim embedding per item

After laplacian_pe:
  x = [150, 256]                     # same shape, values shifted by PE

After TransformerConv layer 1:
  x = [150, 256]                     # attention between neighbors

After TransformerConv layer 2:
  x = [150, 256]                     # second round of attention

After readout (mean pooling):
  session_embeddings = [32, 256]     # one 256-dim vector per session
```

## 2. TransformerConv: What Happens Inside

**Library:** `torch_geometric.nn.TransformerConv`

TransformerConv implements scaled dot-product attention between graph neighbors. For each node, it attends only to its connected neighbors (not all nodes).

```
For node v with neighbors N(v):

    Query:  q_v = W_Q * h_v
    Key:    k_u = W_K * h_u    for each u in N(v)
    Value:  v_u = W_V * h_u    for each u in N(v)

    Attention weights:
    alpha(v, u) = softmax( (q_v * k_u) / sqrt(d_head) )

    Output:
    h_v' = sum( alpha(v, u) * v_u )  for u in N(v)
```

With `heads=2, hidden_dim=256`:
- Each head operates on `256 / 2 = 128` dimensions
- 2 heads run in parallel, outputs concatenated back to 256
- `beta=True` enables gated residual: `h_v' = beta * h_v' + (1-beta) * h_v`

**Why TransformerConv instead of GATConv?**
- GATConv uses *additive* attention: `alpha = LeakyReLU(a^T [Wh_v || Wh_u])`
- TransformerConv uses *scaled dot-product* attention: `alpha = softmax(q * k / sqrt(d))`
- Dot-product attention is more expressive for learning fine-grained similarity. Result: 38.28% vs 20.10% Recall@10.

## 3. Laplacian Positional Encoding Computation

**File:** `etpgt/encodings/laplacian_pe.py`, lines 19-66

```python
def compute_laplacian_pe(edge_index, num_nodes, k=16, normalization="sym"):
    # Step 1: Compute the graph Laplacian
    # L = D - A (degree matrix minus adjacency matrix)
    # With symmetric normalization: L_sym = D^(-1/2) * L * D^(-1/2)
    edge_index_lap, edge_weight_lap = get_laplacian(
        edge_index, normalization=normalization, num_nodes=num_nodes
    )

    # Step 2: Convert to scipy sparse matrix for eigendecomposition
    L = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes)

    # Step 3: Compute k+1 smallest eigenvectors
    # "SM" = Smallest Magnitude eigenvalues
    # The smallest eigenvalue is always 0 (trivial eigenvector = all ones)
    eigenvalues, eigenvectors = eigsh(L, k=k+1, which="SM")

    # Step 4: Skip trivial eigenvector, keep the next k
    eigenvectors = eigenvectors[:, 1:k+1]  # [num_nodes, k]

    # Step 5: Take absolute value
    # Eigenvectors have arbitrary sign (v and -v are both valid)
    # Absolute value makes the encoding sign-invariant
    eigenvectors = torch.from_numpy(eigenvectors).float().abs()

    return eigenvectors  # [num_nodes, 16]
```

**Intuition:** The Laplacian eigenvectors are like Fourier modes on the graph. The lowest-frequency eigenvectors capture the global structure: which parts of the graph are far apart, which are densely connected. Think of it as giving each node GPS coordinates in "graph space."

**Why 16 eigenvectors?** Empirically, 16 captures enough structural information without adding too much compute. Each eigenvector adds one dimension to the positional encoding. The linear projection then maps these 16 dimensions to 256.

## 4. Session Subgraph Extraction in collate_fn

**File:** `etpgt/train/dataloader.py`, lines 157-202

This is where global item IDs get remapped to local batch indices.

```python
def collate_fn(batch):
    data_list = []

    for item in batch:
        session_items = item["session_items"]  # e.g., [42, 1337, 99, 42]
        edge_index = item["edge_index"]        # global edges between these items

        # Step 1: Find unique items and assign local indices
        unique_items = session_items.unique()   # e.g., [42, 99, 1337]
        item_to_idx = {42: 0, 99: 1, 1337: 2}  # global -> local mapping

        # Step 2: Remap edges from global IDs to local indices
        # Global edge: (42, 1337) -> Local edge: (0, 2)
        if edge_index.numel() > 0:
            valid_edges = []
            for i in range(edge_index.shape[1]):
                src = edge_index[0, i].item()
                tgt = edge_index[1, i].item()
                if src in item_to_idx and tgt in item_to_idx:
                    valid_edges.append([item_to_idx[src], item_to_idx[tgt]])

        # Step 3: Create PyG Data object with local indices
        data = Data(
            x=unique_items,              # node features = item IDs
            edge_index=local_edges,       # edges in local coordinates
            target_item=item["target_item"],
            negative_items=item["negative_items"],
        )
        data_list.append(data)

    # Step 4: Merge all sessions into one batch
    # Batch.from_data_list handles index offsetting automatically
    return Batch.from_data_list(data_list)
```

**Why this remapping?** The global graph has 82,173 nodes. A single session might touch 5 items. The GNN should only compute over those 5 nodes and the edges between them, not the entire graph. Local remapping makes this efficient.

## 5. Graph Construction: From Session to Edges

**File:** `scripts/data/04_build_graph.py`, lines 49-80

```python
# For each session, create edges between co-occurring items
for _, group in session_groups:
    events = group.sort_values("timestamp")
    items = events["itemid"].tolist()
    timestamps = events["timestamp"].tolist()
    event_types = events["event"].tolist()

    for i in range(len(items)):
        # Look at items within +5 steps ahead
        for j in range(i + 1, min(i + window + 1, len(items))):
            item_i, item_j = items[i], items[j]

            # Canonical ordering: smaller ID first
            if item_i > item_j:
                item_i, item_j = item_j, item_i

            # Accumulate edge metadata
            edge_key = (item_i, item_j)
            edges[edge_key]["count"] += 1
            edges[edge_key]["last_ts"] = max(edges[edge_key]["last_ts"], ts)
            edges[edge_key]["event_pairs"][f"{event_i}_{event_j}"] += 1
```

**Concrete example with a real session:**

```
Session: [view item_A, view item_B, addtocart item_C, view item_D, view item_E]
Window = 5

Step i=0 (item_A):
  j=1: edge(A, B)  count=1  pair=view_view
  j=2: edge(A, C)  count=1  pair=view_addtocart
  j=3: edge(A, D)  count=1  pair=view_view
  j=4: edge(A, E)  count=1  pair=view_view

Step i=1 (item_B):
  j=2: edge(B, C)  count=1  pair=view_addtocart
  j=3: edge(B, D)  count=1  pair=view_view
  j=4: edge(B, E)  count=1  pair=view_view

Step i=2 (item_C):
  j=3: edge(C, D)  count=1  pair=addtocart_view
  j=4: edge(C, E)  count=1  pair=addtocart_view

Step i=3 (item_D):
  j=4: edge(D, E)  count=1  pair=view_view

Total from this session: 10 edges
If these items appear together in other sessions, counts increment.
```

## 6. ONNX Export: What Gets Exported vs What Stays

**File:** `scripts/pipeline/export_onnx.py`

Only the scoring layer is exported. The GNN is NOT in the ONNX model.

```python
class SessionRecommender(nn.Module):
    """This is what gets exported to ONNX."""
    def forward(self, session_embedding):
        # L2 normalize for cosine similarity
        session_norm = F.normalize(session_embedding, p=2, dim=1)
        # Dot product against all item embeddings
        scores = session_norm @ self.item_embeddings_norm.T
        return scores

# Export:
torch.onnx.export(
    model,
    dummy_input,                          # [1, 256] session embedding
    "session_recommender.onnx",
    input_names=["session_embedding"],
    output_names=["scores"],
    opset_version=14,
)

# Separately save pre-computed embeddings:
np.save("item_embeddings.npy", item_embeddings)
```

**At serving time:**
1. Look up item embeddings from the `.npy` file (simple array indexing)
2. Average them to get session embedding
3. Feed session embedding into ONNX model
4. ONNX model returns scores for all items
5. Take top-K

The GNN's job was to learn good item embeddings during training. Once trained, those embeddings are frozen and the GNN is no longer needed.

## 7. Training Loop: Loss Computation

**File:** `etpgt/train/trainer.py`, lines 80-136

```python
for batch in train_loader:
    batch = batch.to(device)

    # Forward: get session embeddings
    session_embeddings = model(batch)   # [batch_size, 256]

    # Reshape negatives
    batch_size = batch.target_item.shape[0]
    num_negatives = batch.negative_items.shape[0] // batch_size
    negative_items = batch.negative_items.view(batch_size, num_negatives)

    # Compute loss (BPR by default)
    # Positive score = dot(session_emb, target_emb)
    # Negative scores = dot(session_emb, negative_embs)
    # Loss = -log(sigmoid(positive - negative))
    loss = model.compute_loss(session_embeddings, batch.target_item, negative_items)

    # Standard PyTorch training step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Why BPR loss?** It directly optimizes the ranking: "the target item should score higher than random items." This is exactly what Recall@K measures. Cross-entropy over all 82K items would be more expensive and not necessarily better for ranking.
