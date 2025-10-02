# Graph Neural Networks for Session-Based E-Commerce Recommendations

A session-based recommendation system using Graph Neural Networks (GNNs) to predict next-item interactions. Benchmarks multiple GNN architectures and achieves **38.28% Recall@10** with an optimized Graph Transformer.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%3E%3D2.1.0-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-%3E%3D2.4.0-blue" alt="PyTorch Geometric">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/GCP-Vertex_AI-blue?logo=googlecloud" alt="GCP">
</p>

---

## What This Project Is (And Isn't)

This is an **applied ML engineering project**, not a research contribution. The core components — TransformerConv, Laplacian eigenvectors, BPR loss — come from existing literature and libraries. What I built is an end-to-end pipeline that combines these pieces for session-based recommendation:

- **Data pipeline:** Raw RetailRocket events → sessionization → temporal train/val/test split with blackout periods → item co-occurrence graph
- **Model comparison:** Benchmarked GraphSAGE, GAT, and Graph Transformer under identical conditions
- **Cloud training:** Containerized training on Vertex AI with GCS artifact storage
- **Empirical finding:** Adding Laplacian positional encodings to TransformerConv significantly outperformed GAT and GraphSAGE on this task

I learned a lot building this — especially about graph neural networks, proper evaluation methodology, and cloud ML workflows.

---

## Results

Trained on **RetailRocket dataset** (2.76M events, 168K sessions, 737K co-occurrence edges):

| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Training Time | Cost |
|-------|:---------:|:---------:|:-------:|:-------:|:-------------:|:----:|
| GraphSAGE | 14.80% | — | 9.87% | — | 12h | $22 |
| GAT | 20.10% | — | 13.60% | — | 16h | $30 |
| **Graph Transformer** | **38.28%** | **41.29%** | **30.65%** | **31.41%** | **15.5h** | **$29** |

Training performed on Google Cloud Vertex AI with NVIDIA L4 GPU.

---

## Why It Works: Mathematical Intuition

### 1. Laplacian Eigenvectors as Graph Coordinates

**The Problem:** Message-passing GNNs are *permutation equivariant* — they cannot distinguish between nodes with identical local neighborhoods. Two nodes with the same degree and neighbor structure produce identical embeddings, even if they occupy very different positions in the global graph.

**The Solution:** The graph Laplacian `L = D - A` (degree matrix minus adjacency) encodes the full graph structure. Its eigenvectors provide a natural coordinate system:

```
Lφₖ = λₖφₖ
```

- **Small eigenvalues (λ → 0):** Eigenvectors vary *slowly* across the graph — they capture global, low-frequency structure (communities, clusters)
- **Large eigenvalues:** Eigenvectors vary *rapidly* — they capture local, high-frequency structure (boundaries, bridges)

By taking the k smallest non-trivial eigenvectors as node features, each node gets a unique "spectral fingerprint" encoding its structural role. Nodes in the same community cluster together in this spectral space; bridge nodes are clearly separated.

```python
# Normalized Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
# Eigendecomposition gives orthonormal basis aligned with graph structure
eigenvalues, eigenvectors = eigsh(L, k=k+1, which="SM")
eigenvectors = eigenvectors[:, 1:k+1]  # Skip trivial eigenvector
positional_encoding = torch.from_numpy(eigenvectors).float().abs()  # Handle sign ambiguity
```

**Why k=16?** Empirically, the first ~16 eigenvectors capture the dominant structural patterns. Beyond that, eigenvectors encode increasingly localized (noisy) structure with diminishing returns.

### 2. Scaled Dot-Product Attention + Positional Awareness

**The Problem:** GAT uses additive attention over 1-hop neighbors:

```
αᵢⱼ = softmax(LeakyReLU(a^T [Whᵢ || Whⱼ]))  for j ∈ N(i)
```

This has two limitations:
1. Additive attention has limited expressiveness
2. No positional information — nodes with identical features get identical attention weights

**The Solution:** Graph Transformer uses scaled dot-product attention with positional encoding:

```
αᵢⱼ = softmax((Q(hᵢ + pᵢ))^T(K(hⱼ + pⱼ)) / √d)  for j ∈ N(i)
```

Where `pᵢ` is the Laplacian positional encoding. Both models attend over the same co-occurrence edges, but:
1. **Dot-product attention** is more expressive than additive (learns richer query-key interactions)
2. **Positional encoding** lets the model distinguish structurally different nodes even with identical features
3. **Gated residual connections** (`beta=True` in TransformerConv) improve gradient flow

### 3. Why Removing FFN Works

Standard Transformer blocks have:
1. **Multi-head attention** — routes information between positions
2. **Feed-forward network (FFN)** — adds capacity via nonlinear transformation

FFN adds model capacity but for graph recommendation with structured input, the inductive bias from attention + Laplacian PE is sufficient — the FFN capacity is underutilized.

**Empirical validation:** Removing FFN yielded significant speedup with <3% performance drop.

### 4. Why 2 Layers Suffice (Over-Smoothing)

Each GNN layer aggregates information from neighbors. After k layers, a node's representation mixes features from its k-hop neighborhood. With too many layers:

```
h_i^(L) → similar for all i  (over-smoothing)
```

Node representations converge to indistinguishable vectors. For item co-occurrence graphs with small diameter, 2-3 hops typically span the entire graph — more layers provide no new information while accelerating over-smoothing.

---

## Architecture

### Model Configuration (Graph Transformer Optimized)

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 256 |
| Hidden Dimension | 256 |
| Transformer Layers | 2 |
| Attention Heads | 2 |
| Laplacian PE (k) | 16 eigenvectors |
| Dropout | 0.1 |
| Readout | Mean pooling |
| Parameters | ~120M |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-5 |
| Batch Size | 32 |
| Negative Samples | 5 |
| Max Epochs | 100 |
| Early Stopping | Patience 10 |
| Loss | Listwise (softmax cross-entropy) |

---

## Installation

```bash
git clone https://github.com/Axionis47/GAT-Recommendation.git
cd GAT-Recommendation

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### Data Preparation

```bash
python scripts/data/01_download_retailrocket.py
python scripts/data/02_sessionize.py
python scripts/data/03_temporal_split.py
python scripts/data/04_build_graph.py
```

---

## Usage

### Training

```bash
# GraphSAGE
python scripts/train/train_baseline.py --model graphsage

# GAT
python scripts/train/train_baseline.py --model gat

# Graph Transformer (optimized)
python scripts/train/train_baseline.py --model graph_transformer_optimized
```

### Cloud Training (Vertex AI)

```bash
# Edit submit_graph_transformer_optimized.sh to set your PROJECT_ID and BUCKET_NAME
bash submit_graph_transformer_optimized.sh
```

### Inference

```python
from etpgt.model import create_graph_transformer_optimized
import torch

# Load trained model
model = create_graph_transformer_optimized(num_items=466865)
checkpoint = torch.load("best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# batch is a PyG Batch object from the dataloader
session_embedding = model(batch)
top_k_items = model.predict(session_embedding, k=20)
```

---

## Project Structure

```
GAT-Recommendation/
├── etpgt/
│   ├── model/
│   │   ├── base.py                     # Base model with predict()
│   │   ├── graphsage.py
│   │   ├── gat.py
│   │   ├── graph_transformer.py
│   │   ├── graph_transformer_optimized.py
│   │   └── etpgt.py
│   ├── encodings/
│   │   └── laplacian_pe.py
│   ├── train/
│   │   ├── dataloader.py
│   │   ├── trainer.py
│   │   └── losses.py                   # BPRLoss, ListwiseLoss, DualLoss, SampledSoftmaxLoss
│   └── utils/
│       ├── metrics.py
│       ├── logging.py
│       ├── io.py
│       ├── seed.py
│       └── profiler.py
├── scripts/
│   ├── data/
│   │   ├── 01_download_retailrocket.py
│   │   ├── 02_sessionize.py
│   │   ├── 03_temporal_split.py
│   │   └── 04_build_graph.py
│   ├── train/
│   │   ├── train_baseline.py
│   │   └── train_etpgt.py
│   └── gcp/
├── configs/
├── tests/
├── docker/
│   └── train.Dockerfile
└── requirements.txt
```

---

## Technical Stack

| Component | Version/Technology |
|-----------|-------------------|
| Deep Learning | PyTorch ≥2.1.0 |
| Graph Neural Networks | PyTorch Geometric ≥2.4.0 |
| Eigendecomposition | SciPy |
| Cloud Training | Google Cloud Vertex AI |
| GPU | NVIDIA L4 (24GB) |

---

## References

1. Hamilton et al. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.
2. Veličković et al. (2018). *Graph Attention Networks*. ICLR.
3. Dwivedi & Bresson (2020). *A Generalization of Transformer Networks to Graphs*.
4. Dwivedi et al. (2021). *Benchmarking Graph Neural Networks*. JMLR.

**Dataset:** [RetailRocket on Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
