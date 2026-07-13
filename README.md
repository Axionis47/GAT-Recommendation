# Session-Based Recommendations with Graph Neural Networks

> Predicting what an anonymous shopper buys next, from nothing but their last few clicks.
> **38.28% Recall@10** on RetailRocket, **2.6x the GraphSAGE baseline**, trained for about **$21**.

Most recommendation systems need user history. On a real store, **71% of visitors are a single anonymous click**: no account, no past, nothing to personalize on. All you have is the handful of items in the session in front of you. This project predicts the next item from that alone.

## The thinking

The whole design falls out of one honest look at the data. Each observation forces the next decision:

1. **The data has no users, only sessions and items.** We cannot model people, so we model the **relationships between items**, and a structure whose entire job is "things and how they relate" is a **graph**.
2. **An item means nothing on its own.** Its meaning is the company it keeps. So we read the graph with a **graph neural network**: each item's representation is built from its neighbors.
3. **Neighbor-averaging misses the shape of the graph.** A few hub products, a long tail, and *where* an item sits all carry signal. So we use a **graph transformer**: attention over neighbors plus **Laplacian positional encoding**, which gives every item a structural address.

A compute budget then shaped the architecture (see [The result](#the-result)), and the model went from a notebook to a served API.

## Read the full reasoning

The entire story, from 2.76M raw clicks to a served model, is one walkthrough notebook. It **rebuilds the pipeline live** and reproduces every number on this page, and GitHub renders it with all the charts inline:

### [notebooks/session_recsys_walkthrough.ipynb](notebooks/session_recsys_walkthrough.ipynb)

Understand the data (what we have, what is missing) &rarr; why graph networks &rarr; why a graph transformer specifically &rarr; training &rarr; serving.

_(If GitHub struggles to render it, open the same file via [nbviewer](https://nbviewer.org/github/Axionis47/GAT-Recommendation/blob/main/notebooks/session_recsys_walkthrough.ipynb).)_

## The result

38.28% Recall@10 on RetailRocket: the correct next item lands in the top 10 recommendations about 38% of the time, 2.6x better than the GraphSAGE baseline.

**The budget story.** The full Graph Transformer was estimated at $1,880 for training. With $300 spread across multiple projects, that was like being asked to park a yacht in a bicycle rack. So we removed the FFN layers (88x speedup), reduced layers and heads, and the optimized model actually scored *higher*. Total cost: about $21.

## Results

| Model | Recall@10 | NDCG@10 | Cost | Trained? |
|-------|-----------|---------|------|----------|
| GraphSAGE | 14.79% | 9.87% | ~$22 | Yes |
| GAT | 20.10% | 13.64% | ~$30 | Yes |
| Graph Transformer (FFN) | 36.66% | 29.75% | ~$1,880 | No |
| **Graph Transformer (optimized)** | **38.28%** | **30.65%** | **~$21** | **Yes** |

Numbers above are validation Recall@10. Held-out test numbers are lower (an honest generalization gap); the walkthrough notebook reports both.

## How it works

| Stage | What happens | Deep dive |
|-------|--------------|-----------|
| Data | 2.76M events &rarr; sessions (30-min gap, min 3) &rarr; temporal split with blackout &rarr; item co-occurrence graph (82k nodes, 738k edges) | [Data Pipeline](docs/DATA_PIPELINE.md) |
| Model | Item embedding + Laplacian PE &rarr; `TransformerConv` (attention over neighbors, gated residual) &rarr; session readout &rarr; dot-product scoring | [Models](docs/MODELS.md) |
| Training | BPR / listwise / dual loss, negative sampling from 82k items, temporal validation | [Experiments](docs/EXPERIMENTS.md) |
| Serving | Trained checkpoint or ONNX behind a FastAPI endpoint, deployable to GCP Vertex AI | [Deployment](docs/DEPLOYMENT.md) |

## Quick Start

```bash
# Setup
git clone https://github.com/Axionis47/GAT-Recommendation.git
cd GAT-Recommendation
make setup
source .venv/bin/activate

# Download data (requires Kaggle API key)
python scripts/data/01_download_retailrocket.py

# Run data pipeline
make data

# Validate all models (quick, no GPU)
python scripts/pipeline/run_full_pipeline.py --num-sessions 100 --num-epochs 3

# Full training (requires GPU)
python scripts/train/train_baseline.py --model graph_transformer_optimized
```

## Key Technical Decisions

1. **Temporal train/test splits with blackout periods.** Random splits leak future data. We split by time with 2-day blackout gaps.

2. **Laplacian positional encodings.** Standard GNNs cannot tell apart nodes with identical neighborhoods. Laplacian eigenvectors give each node a unique structural fingerprint.

3. **Removing FFN layers (88x speedup).** The feed-forward network in Transformer blocks is underutilized for graph recommendation. Removing it gave 88x speedup with better accuracy.

4. **Two GNN layers.** Average degree is 18. After 2 hops, each node sees ~324 nodes. More layers cause over-smoothing.

## Architecture and Deep Dives

### Architecture (C4 Model)

| Level | Document | What It Covers |
|-------|----------|----------------|
| Context | [C1: Context](docs/architecture/C1_CONTEXT.md) | What problem, who uses it, constraints |
| Container | [C2: Containers](docs/architecture/C2_CONTAINER.md) | Runtime containers, technology choices |
| Component | [C3: Components](docs/architecture/C3_COMPONENT.md) | Internal components, class diagrams |
| Code | [C4: Code](docs/architecture/C4_CODE.md) | Key code walkthroughs with snippets |

### Deep Dives

| Document | What It Covers |
|----------|----------------|
| [Data Pipeline](docs/DATA_PIPELINE.md) | How raw events become a graph, step by step |
| [Models](docs/MODELS.md) | Every model architecture with all parameters |
| [Experiments](docs/EXPERIMENTS.md) | Results, ablations, cost analysis, lessons |
| [Deployment](docs/DEPLOYMENT.md) | Serving, Docker, Terraform, monitoring |
| [Parameters](docs/PARAMETERS.md) | Complete parameter reference |
| [Design Rationale](docs/DESIGN_RATIONALE.md) | Why every parameter has its value (the reasoning chain) |

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Deep Learning | PyTorch 2.1+, PyTorch Geometric 2.4+ |
| Data | pandas, NumPy, SciPy, DVC |
| Serving | FastAPI, ONNX Runtime, Uvicorn |
| Cloud | GCP Vertex AI, Cloud Storage, Artifact Registry |
| Infrastructure | Terraform, Docker, Cloud Build |
| Quality | pytest, ruff, black, isort, mypy |
| Monitoring | Prometheus, Evidently, MLflow |

## Project Structure

```
GAT-Recommendation/
  notebooks/         The walkthrough (data to serving, in one story)
  etpgt/
    model/           GraphSAGE, GAT, Graph Transformer
    train/           DataLoader, Trainer, Loss functions
    encodings/       Laplacian positional encoding
    utils/           Metrics, logging, I/O
  scripts/
    data/            Data pipeline (sessionize, split, build graph)
    train/           Training scripts
    serve/           FastAPI inference server
    pipeline/        Full pipeline validation
    gcp/             GCP deployment scripts
  tests/             Unit and integration tests
  docs/              Architecture (C4) and deep dives
  infra/             Terraform (GCS, Artifact Registry, IAM)
  configs/           Experiment configurations
```

## License

See [LICENSE](LICENSE).
