"""Graph neural network models for session-based recommendation."""

from etpgt.model.base import BaseRecommendationModel, SessionReadout
from etpgt.model.gat import GAT, create_gat
from etpgt.model.graph_transformer import (
    GraphTransformer,
    create_graph_transformer,
    create_graph_transformer_optimized,
)
from etpgt.model.graphsage import GraphSAGE, create_graphsage

__all__ = [
    # Base classes
    "BaseRecommendationModel",
    "SessionReadout",
    # GraphSAGE
    "GraphSAGE",
    "create_graphsage",
    # GAT
    "GAT",
    "create_gat",
    # Graph Transformer (standard and optimized)
    "GraphTransformer",
    "create_graph_transformer",
    "create_graph_transformer_optimized",
]
