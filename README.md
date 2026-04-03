# Session-Based Recommendations with Graph Neural Networks

A recommendation system that predicts what anonymous shoppers will buy next, using only their current browsing session.

**The problem:** Most recommendation systems need user history. Anonymous visitors have none. All you have is the last 3-10 items they clicked on.

**The solution:** Build a co-occurrence graph from browsing sessions. Items that appear together get connected. Use Graph Neural Networks to learn item relationships from this structure. Predict the next item.

**The result:** 38.28% Recall@10 on RetailRocket. The correct next item lands in the top 10 recommendations 38% of the time. 2.6x better than the GraphSAGE baseline.

**The budget story:** The full Graph Transformer was estimated at $1,880 for training. With $300 spread across multiple projects, that was like being asked to park a yacht in a bicycle rack. So we removed the FFN layers (88x speedup), reduced layers and heads, and the optimized model actually scored *higher*. Total cost: about $21.

## Results

| Model | Recall@10 | NDCG@10 | Cost | Trained? |
|-------|-----------|---------|------|----------|
| GraphSAGE | 14.79% | 9.87% | ~$22 | Yes |
| GAT | 20.10% | 13.64% | ~$30 | Yes |
| Graph Transformer (FFN) | 36.66% | 29.75% | ~$1,880 | No |
| **Graph Transformer (optimized)** | **38.28%** | **30.65%** | **~$21** | **Yes** |

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

## Project Structure

```
GAT-Recommendation/
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
  docs/              Documentation (you are here)
  infra/             Terraform (GCS, Artifact Registry, IAM)
  configs/           Experiment configurations
  checkpoints/       Pre-trained model weights
```

## Documentation

Start here and follow the links:

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

## Key Technical Decisions

1. **Temporal train/test splits with blackout periods.** Random splits leak future data. We split by time with 2-day blackout gaps.

2. **Laplacian positional encodings.** Standard GNNs cannot tell apart nodes with identical neighborhoods. Laplacian eigenvectors give each node a unique structural fingerprint.

3. **Removing FFN layers (88x speedup).** The feed-forward network in Transformer blocks is underutilized for graph recommendation. Removing it gave 88x speedup with better accuracy.

4. **Two GNN layers.** Average degree is 18. After 2 hops, each node sees ~324 nodes. More layers cause over-smoothing.

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

## License

See [LICENSE](LICENSE).
