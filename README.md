# ETP-GT: Temporal & Path-Aware Graph Transformer for Session Recommendations

**ETP-GT** is a production-ready session-based recommendation system built on Google Cloud Platform (GCP). It combines temporal dynamics, graph structure, and path-aware attention to deliver high-quality recommendations with strict latency guarantees.

## 🎯 Key Features

- **Temporal & Path-Aware**: Explicitly models time gaps and interaction paths in session graphs
- **GCP-Native**: Runs entirely on GCP (Vertex AI training, Cloud Run serving, GCS storage)
- **Production-Ready**: p95 latency ≤120ms for 200 candidates with full attribution
- **Reproducible**: Fixed seeds, comprehensive logging, automated CI/CD via GitHub Actions
- **Explainable**: Per-prediction attribution traces for edge types, time deltas, and paths

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- GCP project with billing enabled
- GitHub repository with OIDC configured for GCP Workload Identity Federation

### Local Setup

```bash
# Clone and setup
git clone https://github.com/<your-org>/etp-gt.git
cd etp-gt
make setup

# Configure environment
cp .env.example .env
# Edit .env with your GCP project details

# Run linting and tests
make fmt lint typecheck test
```

### GCP Setup

```bash
# 1. Validate GCP environment
make gcp-validate

# 2. Bootstrap GCP infrastructure (bucket, registry, service account)
make gcp-bootstrap

# 3. Prepare data
make data

# 4. Train baselines on Vertex AI
make gcp-train

# 5. Deploy to Cloud Run
make gcp-deploy
```

## 📊 Performance

| Model | Recall@20 | NDCG@20 | VRAM (GB) | p95 Latency (ms) |
|-------|-----------|---------|-----------|------------------|
| GraphSAGE | TBD | TBD | TBD | - |
| GAT | TBD | TBD | TBD | - |
| GraphTransformer | TBD | TBD | TBD | - |
| **ETP-GT** | **TBD** | **TBD** | **TBD** | **<120** |

See [docs/RESULTS.md](docs/RESULTS.md) for detailed results.

## 📁 Project Structure

```
etp-gt/
├── etpgt/              # Core library
│   ├── samplers/       # Temporal path sampling
│   ├── encodings/      # Positional encodings (LapPE, HybridPE)
│   ├── model/          # ETP-GT architecture
│   ├── train/          # Training loops
│   ├── serve/          # Inference API
│   └── cli/            # Command-line tools
├── configs/            # Model configurations
├── scripts/            # Automation scripts
│   └── gcp/            # GCP deployment scripts
├── tests/              # Unit tests
└── docs/               # Documentation
```

## 🔬 Methodology

1. **Temporal Splits**: 70/15/15 with 1-3 day blackout periods to prevent leakage
2. **Graph Construction**: Co-event edges within ±5 steps, temporal constraints
3. **Sampling**: Time-aware neighborhood sampling with fanout [16,12,8]
4. **Encodings**: Laplacian PE + temporal buckets + path buckets
5. **Architecture**: Multi-head attention with learned temporal/path biases
6. **Training**: Dual loss (listwise + contrastive) with early stopping

## 📚 Documentation

- [PRD.md](docs/PRD.md) - Product requirements and success metrics
- [DECISIONS.md](docs/DECISIONS.md) - Locked architectural decisions
- [REPRO.md](docs/REPRO.md) - Reproducibility guide
- [MODEL_CARD.md](docs/MODEL_CARD.md) - Model card
- [RESULTS.md](docs/RESULTS.md) - Experimental results
- [GCP_OIDC_SETUP.md](docs/GCP_OIDC_SETUP.md) - GitHub Actions OIDC setup

## 🛡️ Governance

- **Versioning**: Semantic versioning (v0.1.0+)
- **Testing**: ≥80% coverage for core modules
- **Code Quality**: Black, Ruff, Isort, Mypy (strict)
- **Security**: OIDC-based authentication, no long-lived keys

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Attribution

Built on PyTorch Geometric and inspired by recent advances in temporal graph learning.
See [ATTRIBUTION_NOTES.md](docs/ATTRIBUTION_NOTES.md) for detailed references.

