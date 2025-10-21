# Product Requirements Document: ETP-GT

## Overview

**ETP-GT** (Temporal & Path-Aware Graph Transformer) is a production-ready session-based recommendation system designed to run entirely on Google Cloud Platform (GCP). It addresses the limitations of existing graph-based recommenders by explicitly modeling temporal dynamics and interaction paths.

## Problem Statement

Traditional session-based recommenders fail to capture:
1. **Temporal dynamics**: Time gaps between interactions carry signal (e.g., quick succession vs. long pauses)
2. **Path structure**: The sequence and branching of user exploration patterns
3. **Production constraints**: Most research models are too slow or memory-intensive for real-time serving

## Success Metrics

### Quality Metrics (Primary)
- **Recall@20**: ≥ +2-5% absolute improvement over best baseline
- **NDCG@20**: ≥ +2-5% absolute improvement over best baseline
- Stratified by:
  - Session length: {3-4, 5-9, 10+}
  - Last gap Δt: {≤5m, 5-30m, 30m-2h, >2h}
  - Cold items: ≤5 interactions

### Efficiency Metrics (Primary)
- **Training VRAM**: -20-30% vs dense Graph Transformer at similar quality
- **Serving p95 latency**: ≤120ms for 200 candidates (Cloud Run)
- **Serving p50 latency**: ≤60ms for 200 candidates

### Explainability (Secondary)
- Per-prediction attribution traces exposing:
  - Edge type contributions
  - Temporal delta contributions
  - Path length contributions

## Dataset

**RetailRocket** e-commerce dataset:
- Events: view, add_to_cart, purchase
- Temporal split: 70/15/15 with 1-3 day blackout periods
- Sessionization: 30-minute inactivity gap, min length ≥3

## Architecture Constraints

### GCP-Only Requirement
- **Training**: Vertex AI Custom Jobs (n1-standard-8 + L4 GPU)
- **Storage**: Google Cloud Storage (GCS)
- **Serving**: Cloud Run (2-4 CPU, 2-4GB RAM)
- **Registry**: Artifact Registry
- **Auth**: GitHub OIDC → Workload Identity Federation (no long-lived keys)

### Model Constraints
- **Input**: Last N=10 items from session
- **Candidates**: K=200 via ANN (Faiss IVFPQ)
- **Re-ranking**: 1-layer ETP-GT for latency budget
- **Batch size**: Optimized for single-session inference

## Non-Goals

1. **Multi-dataset generalization**: Focus on RetailRocket only
2. **Real-time training**: Batch retraining is acceptable
3. **Multi-modal features**: Item content, images, text (future work)
4. **Cross-session modeling**: User-level personalization (future work)
5. **On-premise deployment**: GCP-only by design

## Stop Rule

If custom ETP-GT fails to achieve **≥1.5% absolute Recall@20 gain** over best baseline AND shows no VRAM/latency advantage after Phase 5, **STOP** custom development and ship best baseline with production infrastructure.

## Acceptance Criteria

### Phase 0-3: Foundation
- [ ] CI green with ≥80% coverage for core modules
- [ ] Data prep with leakage-safe splits (unit tests enforced)
- [ ] Sampler and encodings tested and documented

### Phase 4: Baselines
- [ ] At least one baseline (GraphSAGE/GAT/GraphTransformer) trained on Vertex AI
- [ ] Metrics logged to GCS and docs/RESULTS.md

### Phase 5: ETP-GT
- [ ] ETP-GT beats ≥1 baseline on validation OR stop rule triggered
- [ ] Ablation matrix completed
- [ ] Attribution examples documented

### Phase 6-9: Production
- [ ] Cloud Run deployment with p95 ≤120ms
- [ ] GitHub Actions OIDC pipeline functional
- [ ] v0.1.0 release with complete documentation

## Timeline

- **Phase 0-1**: 1-2 days (bootstrap)
- **Phase 2-3**: 2-3 days (data + sampler)
- **Phase 4**: 2-3 days (baselines)
- **Phase 5**: 3-5 days (ETP-GT + ablations)
- **Phase 6-9**: 3-4 days (serving + automation)

**Total**: ~2-3 weeks

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Temporal leakage in splits | High | Unit tests enforce blackout periods |
| OOM during training | Medium | Sampler caps, gradient checkpointing |
| Latency budget exceeded | High | 1-layer re-rank, pre-warm instances |
| Custom model underperforms | Medium | Stop rule + baseline fallback |
| GCP quota limits | Medium | Document requirements, request increases |

## Stakeholders

- **ML Team**: Model development, evaluation
- **MLOps Team**: Infrastructure, deployment, monitoring
- **Product**: Success metrics, business requirements

## References

- RetailRocket dataset: [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- Graph Transformers: Dwivedi et al. (2020)
- Temporal Graph Networks: Rossi et al. (2020)

