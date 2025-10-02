"""Laplacian Positional Encoding (LapPE) for graph transformers.

Computes positional encodings from the eigenvectors of the graph Laplacian.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

try:
    from scipy.sparse.linalg import eigsh

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compute_laplacian_pe(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 16,
    normalization: str = "sym",
) -> torch.Tensor:
    """Compute Laplacian Positional Encoding.

    Args:
        edge_index: Edge index [2, num_edges].
        num_nodes: Number of nodes.
        k: Number of eigenvectors to compute.
        normalization: Laplacian normalization ('sym' or 'rw').

    Returns:
        Positional encodings [num_nodes, k].
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for Laplacian PE computation")

    # Get Laplacian
    edge_index_lap, edge_weight_lap = get_laplacian(
        edge_index,
        normalization=normalization,
        num_nodes=num_nodes,
    )

    # Convert to scipy sparse matrix
    L = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes)

    # Compute k smallest eigenvectors (excluding the trivial one)
    # eigsh returns eigenvalues in ascending order
    try:
        eigenvalues, eigenvectors = eigsh(L, k=k + 1, which="SM", return_eigenvectors=True)
    except Exception:
        # Fallback: use dense computation for small graphs
        L_dense = L.toarray()
        eigenvalues, eigenvectors = torch.linalg.eigh(torch.from_numpy(L_dense).float())
        eigenvalues = eigenvalues.numpy()
        eigenvectors = eigenvectors.numpy()

    # Skip the first eigenvector (trivial, all ones)
    eigenvectors = eigenvectors[:, 1 : k + 1]

    # Take absolute value to handle sign ambiguity
    eigenvectors = torch.from_numpy(eigenvectors).float().abs()

    return eigenvectors


class LaplacianPE(nn.Module):
    """Laplacian Positional Encoding layer.

    Computes and projects Laplacian eigenvectors.

    Args:
        k: Number of eigenvectors (default: 16).
        embedding_dim: Output embedding dimension.
        normalization: Laplacian normalization ('sym' or 'rw').
    """

    def __init__(
        self,
        k: int = 16,
        embedding_dim: int = 256,
        normalization: str = "sym",
    ):
        super().__init__()
        self.k = k
        self.embedding_dim = embedding_dim
        self.normalization = normalization

        # Linear projection from k eigenvectors to embedding_dim
        self.projection = nn.Linear(k, embedding_dim)

        # Initialize projection
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, data: Data) -> torch.Tensor:
        """Compute Laplacian PE for graph.

        Args:
            data: PyG Data object with edge_index and num_nodes.

        Returns:
            Positional encodings [num_nodes, embedding_dim].
        """
        # Compute Laplacian eigenvectors
        pe = compute_laplacian_pe(
            data.edge_index,
            data.num_nodes,
            k=self.k,
            normalization=self.normalization,
        )

        # Move to same device as model
        pe = pe.to(self.projection.weight.device)

        # Project to embedding dimension
        pe_projected = self.projection(pe)

        return pe_projected


class LaplacianPECached(nn.Module):
    """Laplacian PE with caching for efficiency.

    Precomputes and caches Laplacian eigenvectors.

    Args:
        k: Number of eigenvectors (default: 16).
        embedding_dim: Output embedding dimension.
        normalization: Laplacian normalization ('sym' or 'rw').
    """

    def __init__(
        self,
        k: int = 16,
        embedding_dim: int = 256,
        normalization: str = "sym",
    ):
        super().__init__()
        self.k = k
        self.embedding_dim = embedding_dim
        self.normalization = normalization

        # Linear projection
        self.projection = nn.Linear(k, embedding_dim)

        # Initialize projection
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        # Cache for precomputed PE
        self.register_buffer("_cached_pe", None)

    def precompute(self, data: Data) -> None:
        """Precompute and cache Laplacian PE.

        Args:
            data: PyG Data object with edge_index and num_nodes.
        """
        pe = compute_laplacian_pe(
            data.edge_index,
            data.num_nodes,
            k=self.k,
            normalization=self.normalization,
        )
        self._cached_pe = pe.to(self.projection.weight.device)

    def project(self, pe: torch.Tensor) -> torch.Tensor:
        """Project precomputed Laplacian PE to embedding dimension.

        Args:
            pe: Precomputed positional encodings [num_nodes, k].

        Returns:
            Projected positional encodings [num_nodes, embedding_dim].
        """
        return self.projection(pe)

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """Get Laplacian PE for specific nodes.

        Args:
            node_indices: Node indices [batch_size] or [batch_size, seq_len].

        Returns:
            Positional encodings [batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim].
        """
        if self._cached_pe is None:
            raise RuntimeError("Laplacian PE not precomputed. Call precompute() first.")

        # Get PE for specified nodes
        pe = self._cached_pe[node_indices]

        # Project to embedding dimension
        pe_projected = self.projection(pe)

        return pe_projected
