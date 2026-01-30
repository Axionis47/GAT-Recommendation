"""Ablation tests to verify all model variants and configurations work."""

import pytest
import torch

from etpgt.model import (
    create_gat,
    create_graph_transformer,
    create_graph_transformer_optimized,
    create_graphsage,
)
from etpgt.train.losses import create_loss_function


class TestModelAblations:
    """Smoke tests for all model variants."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for ablation tests."""
        return {
            "num_items": 100,
            "embedding_dim": 32,
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 2,
            "dropout": 0.1,
        }

    @pytest.fixture
    def dummy_batch(self):
        """Create a minimal batch for testing."""
        from torch_geometric.data import Batch, Data

        data1 = Data(
            x=torch.tensor([1, 2, 3]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        )
        data2 = Data(
            x=torch.tensor([4, 5, 6, 7]),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        )
        return Batch.from_data_list([data1, data2])

    @pytest.mark.parametrize(
        "model_type",
        ["graphsage", "gat", "graph_transformer", "graph_transformer_optimized"],
    )
    def test_model_training_smoke(self, model_type, base_config, dummy_batch):
        """Verify each model can complete 1 forward + backward pass."""
        # GraphSAGE doesn't use num_heads
        config_no_heads = {k: v for k, v in base_config.items() if k != "num_heads"}

        # Create model based on type
        if model_type == "graphsage":
            model = create_graphsage(**config_no_heads)
        elif model_type == "gat":
            model = create_gat(**base_config)
        elif model_type == "graph_transformer":
            model = create_graph_transformer(**base_config, use_laplacian_pe=False)
        elif model_type == "graph_transformer_optimized":
            model = create_graph_transformer_optimized(**base_config, use_laplacian_pe=False)
        else:
            pytest.fail(f"Unknown model type: {model_type}")

        model.train()

        # Forward pass
        output = model(dummy_batch)
        assert output.shape == (2, base_config["hidden_dim"])

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert model.item_embedding.weight.grad is not None

    @pytest.mark.parametrize("use_ffn", [True, False])
    def test_graph_transformer_ffn_ablation(self, use_ffn, base_config, dummy_batch):
        """Verify FFN ablation works."""
        model = create_graph_transformer(
            **base_config, use_laplacian_pe=False, use_ffn=use_ffn
        )
        model.train()

        output = model(dummy_batch)
        loss = output.sum()
        loss.backward()

        assert output.shape == (2, base_config["hidden_dim"])
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize(
        "readout_type", ["mean", "max", "last", "attention"]
    )
    def test_readout_type_ablation(self, readout_type, base_config, dummy_batch):
        """Verify all readout types work."""
        config_no_heads = {k: v for k, v in base_config.items() if k != "num_heads"}
        model = create_graphsage(**config_no_heads, readout_type=readout_type)
        model.eval()

        with torch.no_grad():
            output = model(dummy_batch)

        assert output.shape == (2, base_config["hidden_dim"])
        assert not torch.isnan(output).any()


class TestLossAblations:
    """Smoke tests for all loss function variants."""

    @pytest.fixture
    def loss_test_data(self):
        """Create data for loss function testing."""
        batch_size = 4
        hidden_dim = 32
        num_items = 100
        num_negatives = 5

        session_embeddings = torch.randn(batch_size, hidden_dim)
        item_embedding = torch.nn.Embedding(num_items, hidden_dim)
        target_items = torch.randint(0, num_items, (batch_size,))
        negative_items = torch.randint(0, num_items, (batch_size, num_negatives))

        return session_embeddings, target_items, negative_items, item_embedding

    @pytest.mark.parametrize(
        "loss_type", ["listwise", "bpr", "dual", "sampled_softmax"]
    )
    def test_loss_function_ablation(self, loss_type, loss_test_data):
        """Verify each loss function works in training."""
        session_emb, targets, negatives, item_emb = loss_test_data
        session_emb = session_emb.requires_grad_(True)

        loss_fn = create_loss_function(loss_type)
        result = loss_fn(session_emb, targets, negatives, item_emb)

        # DualLoss returns (loss, metrics_dict), others return just loss
        if isinstance(result, tuple):
            loss, metrics = result
            assert isinstance(metrics, dict)
        else:
            loss = result

        # Check loss is valid
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Check gradients flow
        loss.backward()
        assert session_emb.grad is not None
        assert not torch.isnan(session_emb.grad).any()


class TestEndToEndAblations:
    """End-to-end smoke tests combining models and losses."""

    @pytest.fixture
    def training_setup(self):
        """Create a minimal training setup."""
        from torch_geometric.data import Batch, Data

        config = {
            "num_items": 100,
            "embedding_dim": 32,
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 2,
            "dropout": 0.1,
        }

        data1 = Data(
            x=torch.tensor([1, 2, 3]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        )
        data2 = Data(
            x=torch.tensor([4, 5, 6, 7]),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        )
        batch = Batch.from_data_list([data1, data2])

        # Add targets and negatives
        batch.target_item = torch.tensor([10, 20])
        batch.negative_items = torch.tensor([[11, 12, 13], [21, 22, 23]])

        return config, batch

    @pytest.mark.parametrize(
        "model_type,loss_type",
        [
            ("graphsage", "listwise"),
            ("gat", "bpr"),
            ("graph_transformer", "dual"),
            ("graph_transformer_optimized", "sampled_softmax"),
        ],
    )
    def test_model_loss_combination(self, model_type, loss_type, training_setup):
        """Test that each model + loss combination works."""
        config, batch = training_setup

        # GraphSAGE doesn't use num_heads
        config_no_heads = {k: v for k, v in config.items() if k != "num_heads"}

        # Create model
        if model_type == "graphsage":
            model = create_graphsage(**config_no_heads)
        elif model_type == "gat":
            model = create_gat(**config)
        elif model_type == "graph_transformer":
            model = create_graph_transformer(**config, use_laplacian_pe=False)
        elif model_type == "graph_transformer_optimized":
            model = create_graph_transformer_optimized(**config, use_laplacian_pe=False)

        model.train()
        loss_fn = create_loss_function(loss_type)

        # Forward pass
        session_embeddings = model(batch)

        # Compute loss - pass the embedding layer, not the weights
        result = loss_fn(
            session_embeddings,
            batch.target_item,
            batch.negative_items,
            model.item_embedding,
        )

        # DualLoss returns (loss, metrics_dict), others return just loss
        if isinstance(result, tuple):
            loss, _ = result
        else:
            loss = result

        # Backward pass
        loss.backward()

        # Verify training step completed
        assert not torch.isnan(loss)
        assert model.item_embedding.weight.grad is not None
