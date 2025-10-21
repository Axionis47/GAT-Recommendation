# Attribution & Explainability

This document provides examples of how ETP-GT generates attribution traces for recommendations.

## Overview

ETP-GT provides per-prediction attribution by decomposing attention scores into three learned bias components:

1. **Edge Type Bias**: Contribution from the type of interaction (view-view, view-cart, etc.)
2. **Temporal Delta Bias**: Contribution from time gap between interactions
3. **Path Length Bias**: Contribution from graph distance (1-hop, 2-hop, 3+)

## Attribution Formula

For each recommended item `j`, the attention score from query item `i` is:

```
attention(i, j) = softmax(QK^T / √d + edge_bias + dt_bias + path_bias)
```

We extract the bias contributions before softmax normalization to provide interpretable scores.

## Example 1: Quick Browse Pattern

### Session
```json
{
  "session_id": "sess_001",
  "events": [
    {"item_id": 123, "event_type": "view", "ts": 1700000000000},
    {"item_id": 456, "event_type": "view", "ts": 1700000030000},  // +30s
    {"item_id": 789, "event_type": "view", "ts": 1700000060000}   // +30s
  ]
}
```

### Top Recommendation
```json
{
  "item_id": 234,
  "score": 0.873,
  "attribution": {
    "edge_contribution": 0.12,    // view-view edge (common pattern)
    "dt_contribution": 0.08,      // 0-1m bucket (quick succession)
    "path_contribution": 0.05     // 1-hop (direct connection)
  }
}
```

### Interpretation
- **Edge**: Item 234 is frequently viewed after item 789 (view-view pattern)
- **Temporal**: Short time gap suggests browsing momentum
- **Path**: Direct connection (1-hop) indicates strong co-occurrence

## Example 2: Deliberate Purchase Pattern

### Session
```json
{
  "session_id": "sess_002",
  "events": [
    {"item_id": 111, "event_type": "view", "ts": 1700000000000},
    {"item_id": 222, "event_type": "view", "ts": 1700000300000},      // +5m
    {"item_id": 222, "event_type": "add_to_cart", "ts": 1700000600000}, // +5m
    {"item_id": 333, "event_type": "view", "ts": 1700000900000}       // +5m
  ]
}
```

### Top Recommendation
```json
{
  "item_id": 444,
  "score": 0.921,
  "attribution": {
    "edge_contribution": 0.18,    // cart-view edge (strong signal)
    "dt_contribution": 0.11,      // 1-5m bucket (deliberate)
    "path_contribution": 0.03     // 2-hop (indirect connection)
  }
}
```

### Interpretation
- **Edge**: Item 444 is often viewed after adding item 222 to cart (complementary product)
- **Temporal**: Moderate time gaps suggest deliberate consideration
- **Path**: 2-hop connection (e.g., 222 → 333 → 444) indicates exploration

## Example 3: Return Visit

### Session
```json
{
  "session_id": "sess_003",
  "events": [
    {"item_id": 555, "event_type": "view", "ts": 1700000000000},
    {"item_id": 666, "event_type": "view", "ts": 1700086400000}  // +24h
  ]
}
```

### Top Recommendation
```json
{
  "item_id": 777,
  "score": 0.756,
  "attribution": {
    "edge_contribution": 0.09,    // view-view edge
    "dt_contribution": 0.14,      // 2-24h bucket (return visit)
    "path_contribution": 0.02     // 3+ hop (weak connection)
  }
}
```

### Interpretation
- **Edge**: Moderate view-view connection
- **Temporal**: Long gap indicates return visit (high temporal bias)
- **Path**: Weak graph connection (3+ hops) suggests popularity-based recommendation

## Aggregated Attribution

For each recommendation, we aggregate attributions across all query items in the session:

```python
total_attribution = {
    "edge": sum(edge_contributions) / len(session),
    "dt": sum(dt_contributions) / len(session),
    "path": sum(path_contributions) / len(session)
}
```

## Visualization (Future Work)

Potential visualizations:
1. **Heatmap**: Attribution scores across session items × recommended items
2. **Graph**: Subgraph with edge/temporal/path annotations
3. **Timeline**: Temporal evolution of attributions

## API Response Format

```json
{
  "items": [
    {"item_id": 234, "score": 0.873},
    {"item_id": 456, "score": 0.821},
    ...
  ],
  "attributions": [
    {
      "item_id": 234,
      "edge_contribution": 0.12,
      "dt_contribution": 0.08,
      "path_contribution": 0.05
    },
    {
      "item_id": 456,
      "edge_contribution": 0.10,
      "dt_contribution": 0.06,
      "path_contribution": 0.07
    },
    ...
  ],
  "timing": {...},
  "version": {...}
}
```

## Limitations

1. **Aggregation**: Attributions are averaged across session items; individual contributions may vary
2. **Normalization**: Bias values are pre-softmax; not directly comparable to final scores
3. **Interactions**: Does not capture interactions between bias types (e.g., edge × temporal)
4. **Causality**: Correlational, not causal (e.g., time gap may correlate with other factors)

## Use Cases

1. **Debugging**: Identify why certain items are recommended
2. **Trust**: Provide users with explanations (e.g., "Recommended because you viewed X 5 minutes ago")
3. **Bias detection**: Monitor for unwanted biases (e.g., over-reliance on popularity)
4. **Model improvement**: Identify weak components (e.g., path bias not contributing)

## Future Enhancements

1. **Counterfactual explanations**: "If you had viewed X instead of Y, we would recommend Z"
2. **Feature importance**: SHAP/LIME-style attributions
3. **User-facing explanations**: Natural language generation from attribution traces
4. **Interactive exploration**: Allow users to adjust biases and see updated recommendations

## References

- **Attention Rollout**: Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (2020)
- **Integrated Gradients**: Sundararajan et al., "Axiomatic Attribution for Deep Networks" (2017)
- **GNNExplainer**: Ying et al., "GNNExplainer: Generating Explanations for Graph Neural Networks" (2019)

