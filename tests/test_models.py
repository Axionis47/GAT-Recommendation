"""Tests for all model variants in GAT-Recommendation."""

import pytest
import torch

from etpgt.model import (
    GAT,
    GraphSAGE,
    GraphTransformer,
    create_gat,
    create_graph_transformer,
    create_graph_transformer_optimized,
    create_graphsage,
)


class TestGraphSAGE:
    """Tests for GraphSAGE model."""

    @pytest.fixture
    def graphsage_config(self, small_model_config):
        """GraphSAGE config without num_heads."""
        return {k: v for k, v in small_model_config.items() if k != "num_heads"}

    def test_graphsage_instantiation(self, graphsage_config):
        """Test that GraphSAGE can be instantiated."""
        model = create_graphsage(**graphsage_config)
        assert isinstance(model, GraphSAGE)

    def test_graphsage_forward_pass(self, graphsage_config, dummy_batch):
        """Test GraphSAGE forward pass produces correct output shape."""
        model = create_graphsage(**graphsage_config)
        model.eval()

        with torch.no_grad():
            output = model(dummy_batch)

        batch_size = 2  # 2 graphs in dummy_batch
        hidden_dim = graphsage_config["hidden_dim"]

        assert output.shape == (batch_size, hidden_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_graphsage_gradient_flow(self, graphsage_config, dummy_batch):
        """Test that gradients flow through GraphSAGE."""
        model = create_graphsage(**graphsage_config)
        model.train()

        output = model(dummy_batch)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for item embeddings
        assert model.item_embedding.weight.grad is not None
        assert not torch.isnan(model.item_embedding.weight.grad).any()


class TestGAT:
    """Tests for GAT model."""

    def test_gat_instantiation(self, small_model_config):
        """Test that GAT can be instantiated."""
        model = create_gat(**small_model_config)
        assert isinstance(model, GAT)

    def test_gat_forward_pass(self, small_model_config, dummy_batch):
        """Test GAT forward pass produces correct output shape."""
        model = create_gat(**small_model_config)
        model.eval()

        with torch.no_grad():
            output = model(dummy_batch)

        batch_size = 2
        hidden_dim = small_model_config["hidden_dim"]

        assert output.shape == (batch_size, hidden_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gat_gradient_flow(self, small_model_config, dummy_batch):
        """Test that gradients flow through GAT."""
        model = create_gat(**small_model_config)
        model.train()

        output = model(dummy_batch)
        loss = output.sum()
        loss.backward()

        assert model.item_embedding.weight.grad is not None
        assert not torch.isnan(model.item_embedding.weight.grad).any()


class TestGraphTransformer:
    """Tests for Graph Transformer model."""

    def test_graph_transformer_instantiation(self, small_model_config):
        """Test that GraphTransformer can be instantiated."""
        # Need to disable laplacian_pe for simple tests (requires scipy)
        config = {**small_model_config, "use_laplacian_pe": False}
        model = create_graph_transformer(**config)
        assert isinstance(model, GraphTransformer)

    def test_graph_transformer_with_ffn(self, small_model_config, dummy_batch):
        """Test GraphTransformer with FFN enabled."""
        config = {**small_model_config, "use_laplacian_pe": False, "use_ffn": True}
        model = create_graph_transformer(**config)
        model.eval()

        with torch.no_grad():
            output = model(dummy_batch)

        batch_size = 2
        hidden_dim = small_model_config["hidden_dim"]

        assert output.shape == (batch_size, hidden_dim)
        assert not torch.isnan(output).any()

    def test_graph_transformer_without_ffn(self, small_model_config, dummy_batch):
        """Test GraphTransformer with FFN disabled (optimized mode)."""
        config = {**small_model_config, "use_laplacian_pe": False, "use_ffn": False}
        model = create_graph_transformer(**config)
        model.eval()

        with torch.no_grad():
            output = model(dummy_batch)

        batch_size = 2
        hidden_dim = small_model_config["hidden_dim"]

        assert output.shape == (batch_size, hidden_dim)
        assert not torch.isnan(output).any()

    def test_graph_transformer_optimized_factory(self, small_model_config, dummy_batch):
        """Test the optimized factory function creates correct configuration."""
        config = {
            "num_items": small_model_config["num_items"],
            "embedding_dim": small_model_config["embedding_dim"],
            "hidden_dim": small_model_config["hidden_dim"],
            "use_laplacian_pe": False,
        }
        model = create_graph_transformer_optimized(**config)

        # Verify optimized defaults
        assert model.use_ffn is False
        assert model.num_layers == 2
        assert model.num_heads == 2

        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)

        assert output.shape == (2, small_model_config["hidden_dim"])

    def test_graph_transformer_gradient_flow(self, small_model_config, dummy_batch):
        """Test that gradients flow through GraphTransformer."""
        config = {**small_model_config, "use_laplacian_pe": False}
        model = create_graph_transformer(**config)
        model.train()

        output = model(dummy_batch)
        loss = output.sum()
        loss.backward()

        assert model.item_embedding.weight.grad is not None
        assert not torch.isnan(model.item_embedding.weight.grad).any()


class TestModelConsistency:
    """Tests for consistency across model variants."""

    @pytest.mark.parametrize(
        "create_fn,needs_num_heads",
        [
            (create_graphsage, False),
            (create_gat, True),
            (lambda **k: create_graph_transformer(**k, use_laplacian_pe=False), True),
            (lambda **k: create_graph_transformer_optimized(**k, use_laplacian_pe=False), True),
        ],
    )
    def test_all_models_produce_same_output_shape(
        self, create_fn, needs_num_heads, small_model_config, dummy_batch
    ):
        """Test that all models produce the same output shape."""
        config = small_model_config if needs_num_heads else {
            k: v for k, v in small_model_config.items() if k != "num_heads"
        }
        model = create_fn(**config)
        model.eval()

        with torch.no_grad():
            output = model(dummy_batch)

        expected_shape = (2, small_model_config["hidden_dim"])
        assert output.shape == expected_shape

    @pytest.mark.parametrize(
        "create_fn,needs_num_heads",
        [
            (create_graphsage, False),
            (create_gat, True),
            (lambda **k: create_graph_transformer(**k, use_laplacian_pe=False), True),
        ],
    )
    def test_all_models_have_predict_method(
        self, create_fn, needs_num_heads, small_model_config, dummy_batch
    ):
        """Test that all models have working predict() method."""
        config = small_model_config if needs_num_heads else {
            k: v for k, v in small_model_config.items() if k != "num_heads"
        }
        model = create_fn(**config)
        model.eval()

        with torch.no_grad():
            session_embeddings = model(dummy_batch)
            predictions = model.predict(session_embeddings, k=10)

        assert predictions.shape == (2, 10)  # batch_size=2, k=10
        assert predictions.dtype == torch.long
