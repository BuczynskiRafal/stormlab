"""Adjacency preprocessing shared by training and serving."""

import numpy as np
import scipy.sparse as sp


def preprocess_adjacency(adjacency_matrix, add_self_loops=True, normalize=True, max_hops=1):
    """One-time preprocessing of adjacency matrix with multi-hop support."""
    adj = adjacency_matrix.copy()
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    else:
        adj = adj.tocsr()

    # Add self-loops if requested
    if add_self_loops:
        shape = adj.shape
        n_rows = shape[0] if shape else 0
        adj = adj + sp.eye(n_rows, format="csr")

    # Create multi-hop adjacency matrix
    if max_hops > 1:
        multi_hop_adj = adj.copy()
        adj_power = adj.copy()

        for hop in range(2, max_hops + 1):
            adj_power = adj_power @ adj  # A^hop
            # Weight decreases with distance (closer neighbors more important)
            weight = 0.5 ** (hop - 1)  # 1.0, 0.5, 0.25, ...
            multi_hop_adj = multi_hop_adj + weight * adj_power

        # Remove self-connections from multi-hop (keep only from self-loops)
        multi_hop_adj.setdiag(adj.diagonal())
        adj = multi_hop_adj

    # Normalize if requested
    if normalize:
        # Row-wise normalization
        row_sums = np.array(adj.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        D_inv = sp.diags(1.0 / row_sums, format="csr")
        adj = D_inv @ adj

    return adj.tocsr()
