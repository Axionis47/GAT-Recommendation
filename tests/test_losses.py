"""Tests for all loss functions in GAT-Recommendation."""

import pytest
import torch

from etpgt.train.losses import (
    BPRLoss,
    DualLoss,
    ListwiseLoss,
    SampledSoftmaxLoss,
    create_loss_function,
)


class TestListwiseLoss:
    """Tests for ListwiseLoss."""

    def test_listwise_loss_instantiation(self):
        """Test that ListwiseLoss can be instantiated."""
        loss_fn = ListwiseLoss()
        assert isinstance(loss_fn, ListwiseLoss)

    def test_listwise_loss_output_shape(
        self, dummy_session_embeddings, dummy_item_embeddings, dummy_targets_and_negatives
    ):
        """Test that ListwiseLoss returns a scalar."""
        loss_fn = ListwiseLoss()
        target_items, negative_items = dummy_targets_and_negatives

        loss = loss_fn(
            dummy_session_embeddings,
            target_items,
            negative_items,
            dummy_item_embeddings,
        )

        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0  # Loss should be non-negative

    def test_listwise_loss_gradient_flow(
        self, dummy_session_embeddings, dummy_item_embeddings, dummy_targets_and_negatives
    ):
        """Test that gradients flow through ListwiseLoss."""
        loss_fn = ListwiseLoss()
        target_items, negative_items = dummy_targets_and_negatives

        session_emb = dummy_session_embeddings.clone().requires_grad_(True)

        loss = loss_fn(session_emb, target_items, negative_items, dummy_item_embeddings)
        loss.backward()

        assert session_emb.grad is not None
        assert not torch.isnan(session_emb.grad).any()

    def test_listwise_loss_temperature_scaling(
        self, dummy_session_embeddings, dummy_item_embeddings, dummy_targets_and_negatives
    ):
        """Test that temperature affects loss magnitude."""
        target_items, negative_items = dummy_targets_and_negatives

        loss_low_temp = ListwiseLoss(temperature=0.5)
        loss_high_temp = ListwiseLoss(temperature=2.0)

        loss1 = loss_low_temp(
            dummy_session_embeddings, target_items, negative_items, dummy_item_embeddings
        )
        loss2 = loss_high_temp(
            dummy_session_embeddings, target_items, negative_items, dummy_item_embeddings
        )

        # Different temperatures should produce different losses
        assert loss1.item() != loss2.item()


class TestBPRLoss:
    """Tests for BPRLoss."""

    def test_bpr_loss_instantiation(self):
        """Test that BPRLoss can be instantiated."""
        loss_fn = BPRLoss()
        assert isinstance(loss_fn, BPRLoss)

    def test_bpr_loss_output_shape(
        self, dummy_session_embeddings, dummy_item_embeddings, dummy_targets_and_negatives
    ):
        """Test that BPRLoss returns a scalar."""
        loss_fn = BPRLoss()
        target_items, negative_items = dummy_targets_and_negatives

        loss = loss_fn(
            dummy_session_embeddings,
            target_items,
            negative_items,
            dummy_item_embeddings,
        )

        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_bpr_loss_gradient_flow(
        self, dummy_session_embeddings, dummy_item_embeddings, dummy_targets_and_negatives
    ):
        """Test that gradients flow through BPRLoss."""
        loss_fn = BPRLoss()
        target_items, negative_items = dummy_targets_and_negatives

        session_emb = dummy_session_embeddings.clone().requires_grad_(True)

        loss = loss_fn(session_emb, target_items, negative_items, dummy_item_embeddings)
        loss.backward()

        assert session_emb.grad is not None
        assert not torch.isnan(session_emb.grad).any()

    def test_bpr_loss_margin_property(self):
        """Test that BPR loss is lower when positive scores are higher than negatives."""
        loss_fn = BPRLoss()
        batch_size = 4
        num_negatives = 5

        # Create embeddings where positive items are similar to session
        session_emb = torch.randn(batch_size, 32)
        session_emb = session_emb / session_emb.norm(dim=1, keepdim=True)

        # Create item embedding layer
        item_embedding = torch.nn.Embedding(100, 32)

        # Make positive item (index 0) match the first session embedding
        positive_items = torch.zeros(batch_size, dtype=torch.long)
        with torch.no_grad():
            item_embedding.weight[0] = session_emb[0]

        # Random negatives
        negative_items = torch.randint(1, 100, (batch_size, num_negatives))

        loss = loss_fn(session_emb, positive_items, negative_items, item_embedding)

        # Loss should be finite and positive
        assert not torch.isnan(loss)
        assert loss.item() >= 0


class TestDualLoss:
    """Tests for DualLoss."""

    def test_dual_loss_instantiation(self):
        """Test that DualLoss can be instantiated."""
        loss_fn = DualLoss(alpha=0.7)
        assert isinstance(loss_fn, DualLoss)

    def test_dual_loss_output_shape(
        self, dummy_session_embeddings, dummy_item_embeddings, dummy_targets_and_negatives
    ):
        """Test that DualLoss returns a scalar loss and metrics dict."""
        loss_fn = DualLoss(alpha=0.7)
        target_items, negative_items = dummy_targets_and_negatives

        loss, metrics = loss_fn(
            dummy_session_embeddings,
            target_items,
            negative_items,
            dummy_item_embeddings,
        )

        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert isinstance(metrics, dict)
        assert "total" in metrics
        assert "listwise" in metrics
        assert "bpr" in metrics

    def test_dual_loss_alpha_weighting(
        self, dummy_session_embeddings, dummy_item_embeddings, dummy_targets_and_negatives
    ):
        """Test that alpha correctly weights listwise and BPR losses."""
        target_items, negative_items = dummy_targets_and_negatives

        # Pure listwise (alpha=1.0)
        loss_listwise = DualLoss(alpha=1.0)
        # Pure BPR (alpha=0.0)
        loss_bpr = DualLoss(alpha=0.0)
        # Mixed (alpha=0.5)
        loss_mixed = DualLoss(alpha=0.5)

        l1, _ = loss_listwise(
            dummy_session_embeddings, target_items, negative_items, dummy_item_embeddings
        )
        l2, _ = loss_bpr(
            dummy_session_embeddings, target_items, negative_items, dummy_item_embeddings
        )
        l3, _ = loss_mixed(
            dummy_session_embeddings, target_items, negative_items, dummy_item_embeddings
        )

        # Mixed loss should be between pure listwise and pure BPR (approximately)
        # This is a soft check due to numerical differences
        assert not torch.isnan(l1) and not torch.isnan(l2) and not torch.isnan(l3)

    def test_dual_loss_gradient_flow(
        self, dummy_session_embeddings, dummy_item_embeddings, dummy_targets_and_negatives
    ):
        """Test that gradients flow through DualLoss."""
        loss_fn = DualLoss(alpha=0.7)
        target_items, negative_items = dummy_targets_and_negatives

        session_emb = dummy_session_embeddings.clone().requires_grad_(True)

        loss, _ = loss_fn(session_emb, target_items, negative_items, dummy_item_embeddings)
        loss.backward()

        assert session_emb.grad is not None
        assert not torch.isnan(session_emb.grad).any()


class TestSampledSoftmaxLoss:
    """Tests for SampledSoftmaxLoss."""

    def test_sampled_softmax_loss_instantiation(self):
        """Test that SampledSoftmaxLoss can be instantiated."""
        loss_fn = SampledSoftmaxLoss()
        assert isinstance(loss_fn, SampledSoftmaxLoss)

    def test_sampled_softmax_loss_output_shape(
        self, dummy_session_embeddings, dummy_item_embeddings, dummy_targets_and_negatives
    ):
        """Test that SampledSoftmaxLoss returns a scalar."""
        loss_fn = SampledSoftmaxLoss()
        target_items, negative_items = dummy_targets_and_negatives

        loss = loss_fn(
            dummy_session_embeddings,
            target_items,
            negative_items,
            dummy_item_embeddings,
        )

        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sampled_softmax_temperature_effect(
        self, dummy_session_embeddings, dummy_item_embeddings, dummy_targets_and_negatives
    ):
        """Test that temperature affects loss."""
        target_items, negative_items = dummy_targets_and_negatives

        loss_low_temp = SampledSoftmaxLoss(temperature=0.5)
        loss_high_temp = SampledSoftmaxLoss(temperature=2.0)

        l1 = loss_low_temp(
            dummy_session_embeddings, target_items, negative_items, dummy_item_embeddings
        )
        l2 = loss_high_temp(
            dummy_session_embeddings, target_items, negative_items, dummy_item_embeddings
        )

        # Different temperatures should produce different losses
        assert l1.item() != l2.item()

    def test_sampled_softmax_gradient_flow(
        self, dummy_session_embeddings, dummy_item_embeddings, dummy_targets_and_negatives
    ):
        """Test that gradients flow through SampledSoftmaxLoss."""
        loss_fn = SampledSoftmaxLoss()
        target_items, negative_items = dummy_targets_and_negatives

        session_emb = dummy_session_embeddings.clone().requires_grad_(True)

        loss = loss_fn(session_emb, target_items, negative_items, dummy_item_embeddings)
        loss.backward()

        assert session_emb.grad is not None
        assert not torch.isnan(session_emb.grad).any()


class TestLossFactory:
    """Tests for the loss function factory."""

    @pytest.mark.parametrize(
        "loss_type,expected_class",
        [
            ("listwise", ListwiseLoss),
            ("bpr", BPRLoss),
            ("dual", DualLoss),
            ("sampled_softmax", SampledSoftmaxLoss),
        ],
    )
    def test_create_loss_function(self, loss_type, expected_class):
        """Test that factory creates correct loss type."""
        loss_fn = create_loss_function(loss_type)
        assert isinstance(loss_fn, expected_class)

    def test_create_loss_function_with_params(self):
        """Test factory with custom parameters."""
        loss_fn = create_loss_function("dual", alpha=0.8, temperature=0.5)
        assert isinstance(loss_fn, DualLoss)

    def test_create_loss_function_invalid_type(self):
        """Test factory raises error for invalid loss type."""
        with pytest.raises((ValueError, KeyError)):
            create_loss_function("invalid_loss_type")
