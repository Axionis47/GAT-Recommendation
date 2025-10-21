# Model Card: ETP-GT

## Model Details

**Model Name**: ETP-GT (Temporal & Path-Aware Graph Transformer)  
**Version**: 0.1.0  
**Date**: TBD  
**Organization**: TBD  
**License**: MIT  

### Model Description

ETP-GT is a graph transformer designed for session-based recommendation that explicitly models:
- **Temporal dynamics**: Time gaps between interactions via learned temporal biases
- **Path structure**: Interaction paths via learned path length biases
- **Graph structure**: Item co-occurrence via attention over temporal neighborhoods

### Architecture

- **Type**: Graph Transformer with temporal and path-aware attention
- **Layers**: 3 transformer blocks
- **Hidden dimension**: 256
- **Attention heads**: 4
- **Positional encoding**: Laplacian PE (k=16) + temporal buckets + path buckets
- **Readout**: CLS token + gated pooling
- **Parameters**: ~TBD million

## Intended Use

### Primary Use Cases

1. **Session-based recommendation**: Predict next items in e-commerce sessions
2. **Real-time serving**: Low-latency recommendations (<120ms p95)
3. **Explainable recommendations**: Attribution traces for model decisions

### Out-of-Scope Use Cases

- Cross-session user modeling (no user-level personalization)
- Cold-start items with zero interactions
- Multi-modal recommendations (text, images)
- Real-time model updates (batch retraining only)

## Training Data

### Dataset

**RetailRocket** e-commerce dataset:
- **Events**: view, add_to_cart, purchase
- **Sessions**: TBD (after filtering)
- **Items**: TBD unique items
- **Date range**: TBD

### Preprocessing

- **Sessionization**: 30-minute inactivity gap
- **Filtering**: Min session length ≥3 events
- **Deduplication**: Squash duplicates within 15s unless event escalates
- **Temporal split**: 70/15/15 with 1-3 day blackout periods

### Data Limitations

- **Domain**: E-commerce only (may not generalize to other domains)
- **Geography**: TBD (check RetailRocket metadata)
- **Temporal coverage**: TBD date range
- **Event distribution**: Heavily skewed toward views (~90%+)

## Evaluation

### Metrics

| Metric | Value | Baseline (Best) |
|--------|-------|-----------------|
| Recall@10 | TBD | TBD |
| Recall@20 | TBD | TBD |
| NDCG@10 | TBD | TBD |
| NDCG@20 | TBD | TBD |

### Stratified Performance

**By Session Length**:
| Length | Recall@20 | NDCG@20 |
|--------|-----------|---------|
| 3-4 | TBD | TBD |
| 5-9 | TBD | TBD |
| 10+ | TBD | TBD |

**By Last Gap Δt**:
| Gap | Recall@20 | NDCG@20 |
|-----|-----------|---------|
| ≤5m | TBD | TBD |
| 5-30m | TBD | TBD |
| 30m-2h | TBD | TBD |
| >2h | TBD | TBD |

**Cold Items** (≤5 interactions):
- Recall@20: TBD
- NDCG@20: TBD

### Efficiency

- **Training VRAM**: TBD GB (vs. TBD GB for dense GraphTransformer)
- **Training time**: TBD hours on L4 GPU
- **Inference p50**: TBD ms
- **Inference p95**: TBD ms (target: ≤120ms)

## Ethical Considerations

### Potential Biases

1. **Popularity bias**: Model may over-recommend popular items
   - **Mitigation**: Stratified evaluation includes cold items
   
2. **Temporal bias**: Recent sessions over-represented in training
   - **Mitigation**: Temporal split with blackout periods
   
3. **Event type bias**: Views dominate dataset (~90%+)
   - **Mitigation**: Event type embeddings; weighted loss (future work)

### Fairness

- **No demographic data**: Cannot assess fairness across user groups
- **Item fairness**: Long-tail items may receive fewer recommendations
- **Recommendation**: Monitor item coverage and diversity in production

### Privacy

- **Session-level**: No user identifiers; sessions are anonymous
- **Data retention**: Follow GDPR/CCPA guidelines for session data
- **Inference**: No user data stored; stateless API

## Limitations

1. **Domain-specific**: Trained on e-commerce; may not transfer to other domains
2. **Cold-start**: Poor performance on items with <5 interactions
3. **Session length**: Optimized for sessions of length 3-20; may degrade on very long sessions
4. **Temporal range**: Trained on TBD date range; may drift over time
5. **Latency-quality tradeoff**: 1-layer re-ranker sacrifices some quality for speed

## Recommendations

### Deployment

- **Monitoring**: Track latency (p50/p95), item coverage, diversity
- **Retraining**: Retrain monthly or when performance degrades >2%
- **Fallback**: Use popularity-based recommendations if model fails
- **A/B testing**: Gradual rollout with control group

### Future Improvements

1. **Multi-modal features**: Incorporate item text, images
2. **User-level personalization**: Cross-session modeling
3. **Diversity optimization**: Explicit diversity constraints
4. **Real-time updates**: Incremental learning for new items

## References

- **Graph Transformers**: Dwivedi et al., "Benchmarking Graph Neural Networks" (2020)
- **Temporal Graph Networks**: Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs" (2020)
- **Session-based RecSys**: Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks" (2016)

## Contact

For questions or issues, please open an issue on GitHub: https://github.com/<org>/etp-gt/issues

## Changelog

### v0.1.0 (TBD)
- Initial release
- Baselines: GraphSAGE, GAT, GraphTransformer
- ETP-GT with temporal and path-aware attention
- Cloud Run deployment with <120ms p95 latency

