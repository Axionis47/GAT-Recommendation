"""Turn a validated session into recommendations with the real trained model.

This loads the optimized Graph Transformer checkpoint and runs the actual GNN
forward pass (not the mean-embedding approximation the old server used): it
builds the session's subgraph from the co-occurrence graph, passes it through
the model, and scores every item by dot product with the session embedding.
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from etpgt.model import create_graph_transformer_optimized
from etpgt.serving.validation import ValidatedRequest


def _repo_root() -> Path:
    # etpgt/serving/recommender.py -> parents[2] is the repo root
    return Path(__file__).resolve().parents[2]


class Recommender:
    """Loads the trained optimized model + co-occurrence graph and serves top-k.

    Args:
        checkpoint_path: Path to the optimized Graph Transformer checkpoint.
        graph_edges_path: Path to the co-occurrence graph edges CSV.
        device: Torch device string ("cpu" or "cuda").
    """

    def __init__(self, checkpoint_path: Path | str, graph_edges_path: Path | str, device: str = "cpu"):
        self.device = torch.device(device)
        self._load_model(Path(checkpoint_path))
        self._load_graph(Path(graph_edges_path))

    @classmethod
    def from_default(cls, device: str = "cpu") -> "Recommender":
        """Build a Recommender from the checkpoint and graph in the repo."""
        root = _repo_root()
        return cls(
            root / "checkpoints" / "best_model.pt",
            root / "data" / "processed" / "graph_edges.csv",
            device=device,
        )

    def _load_model(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = checkpoint["model_state_dict"]

        # This server targets the optimized (no-FFN) checkpoint. Fail loudly on others.
        if any(key.startswith("ffns.") for key in sd):
            raise RuntimeError(
                "This Recommender targets the optimized (no-FFN) checkpoint, but the "
                "given checkpoint has FFN layers. Load the optimized model instead."
            )

        # Read architecture straight off the tensors so we never guess catalog size.
        self.num_items, self.embedding_dim = (int(v) for v in sd["item_embedding.weight"].shape)
        hidden_dim = int(sd["batch_norms.0.weight"].shape[0])
        laplacian_k = int(sd["laplacian_pe.projection.weight"].shape[1])
        num_layers = len({key.split(".")[1] for key in sd if key.startswith("convs.")})

        model = create_graph_transformer_optimized(
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=2,  # the optimized configuration
            use_laplacian_pe=True,
            laplacian_k=laplacian_k,
        )
        # The cached-PE buffer starts as None on a fresh model; give it a real slot
        # so the checkpoint's precomputed positional encodings load into it.
        model.laplacian_pe._cached_pe = torch.empty_like(sd["laplacian_pe._cached_pe"])
        result = model.load_state_dict(sd, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"checkpoint does not match model: missing={result.missing_keys[:4]} "
                f"unexpected={result.unexpected_keys[:4]}"
            )

        self.model = model.to(self.device).eval()
        self.item_embeddings = self.model.get_item_embeddings().detach().to(self.device)
        self.checkpoint_epoch = int(checkpoint.get("epoch", -1))
        self.val_recall_at_10 = float(checkpoint.get("best_val_metric", float("nan")))
        del checkpoint

    def _load_graph(self, graph_edges_path: Path) -> None:
        edges = pd.read_csv(graph_edges_path, usecols=["item_i", "item_j"])
        adjacency: dict[int, set[int]] = defaultdict(set)
        for i, j in edges.itertuples(index=False):
            if i != j:  # skip self-loops for message passing
                adjacency[int(i)].add(int(j))
        self._adjacency = adjacency

    def _build_session_graph(self, items: list[int]) -> Data:
        """Build the session's induced subgraph, mirroring the training dataloader."""
        seen = set(items)
        unique = sorted(seen)
        local = {global_id: idx for idx, global_id in enumerate(unique)}
        # keep only stored edges whose both endpoints are in the session
        pairs = [(i, j) for i in seen for j in self._adjacency.get(i, ()) if j in seen]

        x = torch.tensor(unique, dtype=torch.long, device=self.device)
        if pairs:
            edge_index = torch.tensor(
                [[local[i], local[j]] for i, j in pairs], dtype=torch.long, device=self.device
            ).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(len(unique), dtype=torch.long, device=self.device)
        return data

    @torch.no_grad()
    def recommend(self, request: ValidatedRequest) -> tuple[list[int], list[float]]:
        """Return (item_ids, scores) for the top-k recommendations.

        Args:
            request: A validated request (see etpgt.serving.validation).

        Returns:
            Tuple of recommended item ids and their scores, best first.
        """
        data = self._build_session_graph(request.session_items)
        session_embedding = self.model(data)  # [1, hidden_dim]

        scores = (session_embedding @ self.item_embeddings.t()).squeeze(0)
        scores[list(set(request.session_items))] = float("-inf")  # do not recommend seen items
        scores[0] = float("-inf")  # padding index

        top = torch.topk(scores, request.k)
        return top.indices.tolist(), [float(v) for v in top.values.tolist()]

    def health(self) -> dict:
        """Model readiness and provenance."""
        return {
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
            "checkpoint_epoch": self.checkpoint_epoch,
            "val_recall_at_10": self.val_recall_at_10,
        }
