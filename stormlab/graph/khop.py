"""k-hop preprocessing of the conduit adjacency.

``legacy_khop_v1`` is the scheme the saved GraphSAGE weights were preprocessed
with. It is reproduced verbatim from ``app/sa/core/gnn/preprocessing.py``; the
implementation and its golden test are the definition, not the docstring
formula, because the arithmetic does not match the intent stated in the legacy
comment.

What it actually computes, for ``max_hops = K > 1``:

1. ``adj0 = A + I``;
2. ``M = Σ_{k=1..K} 0.5^(k-1) · adj0^k`` — note the self-loops are *inside* the
   power, so ``adj0^k = Σ_j C(k,j) A^j`` multiplies near neighbours repeatedly
   and the resulting hop profile is not monotone: on a plain chain the 1-hop
   weight is 3.25 and the 2-hop weight only 2.0;
3. ``M`` gets its diagonal overwritten with ``adj0``'s diagonal (1 with
   self-loops, 0 without), which discards the accumulated self terms;
4. row-wise ``D⁻¹M`` normalisation, with all-zero rows left untouched.

``max_hops = 1`` skips steps 2 and 3 entirely — ``A + I``, row-normalised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from stormlab.config import (
    GNN_OPERATOR_KHOP_SUM,
    GNN_OPERATOR_KHOP_SYM,
    GNN_OPERATOR_LEGACY_WEIGHTED_MEAN,
    GNN_OPERATOR_NOGRAPH,
    GNN_OPERATOR_SUPPORT_MAX,
    GNN_OPERATOR_SUPPORT_POOL,
    GNN_OPERATORS,
    KHOP_SCHEMES,
)
from stormlab.graph.build import n_rows

DEFAULT_KHOP_SCHEME = KHOP_SCHEMES[0]


@dataclass(frozen=True)
class KhopResult:
    """A preprocessed adjacency and the size of its support.

    Attributes:
        adjacency: row-normalised CSR matrix.
        n_edges_khop: stored non-zeros. This counts the self-loops on the
            diagonal as well, so it is not comparable to
            ``GraphBuildResult.n_edges_raw`` (which has a zero diagonal): an
            11-conduit chain gives ``n_edges_raw = 10`` but
            ``n_edges_khop = 21`` at ``max_hops=1`` and ``45`` at
            ``max_hops=4``.
    """

    adjacency: sp.csr_matrix
    n_edges_khop: int


def preprocess_adjacency(
    adjacency_matrix,
    add_self_loops: bool = True,
    normalize: bool = True,
    max_hops: int = 1,
) -> sp.csr_matrix:
    """One-time preprocessing of an adjacency matrix with multi-hop support.

    Args:
        adjacency_matrix: sparse or dense square matrix.
        add_self_loops: add ``I`` before the hop expansion.
        normalize: apply row-wise ``D⁻¹`` normalisation at the end.
        max_hops: highest power of ``A + I`` to accumulate; ``1`` disables the
            multi-hop combination.
    """
    adj = adjacency_matrix.copy()
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    else:
        adj = adj.tocsr()

    if add_self_loops:
        adj = sp.csr_matrix(adj + sp.eye(n_rows(adj), format="csr"))

    if max_hops > 1:
        multi_hop_adj = adj.copy()
        adj_power = adj.copy()

        for hop in range(2, max_hops + 1):
            adj_power = adj_power @ adj
            weight = 0.5 ** (hop - 1)
            multi_hop_adj = multi_hop_adj + weight * adj_power

        # The accumulated diagonal is thrown away: only the self-loop survives.
        multi_hop_adj.setdiag(adj.diagonal())
        adj = sp.csr_matrix(multi_hop_adj)

    if normalize:
        row_sums = np.array(adj.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        d_inv = sp.diags(1.0 / row_sums, format="csr")
        adj = sp.csr_matrix(d_inv @ adj)

    return sp.csr_matrix(adj.tocsr())


def apply_khop(
    adjacency,
    *,
    khop_scheme: str = DEFAULT_KHOP_SCHEME,
    max_hops: int = 1,
    add_self_loops: bool = True,
    normalize: bool = True,
) -> KhopResult:
    """Preprocess an adjacency under a named scheme.

    Args:
        adjacency: raw adjacency, as produced by ``graph.build``.
        khop_scheme: name of the scheme; only ``legacy_khop_v1`` exists.
        max_hops: highest power of ``A + I`` to accumulate.
        add_self_loops: add ``I`` before the hop expansion.
        normalize: apply row-wise normalisation.

    Raises:
        ValueError: on an unknown scheme or a non-positive ``max_hops``.
    """
    if khop_scheme not in KHOP_SCHEMES:
        raise ValueError(f"Unknown khop_scheme {khop_scheme!r}; expected one of {list(KHOP_SCHEMES)}")
    if max_hops < 1:
        raise ValueError(f"max_hops must be at least 1, got {max_hops}")

    processed = preprocess_adjacency(
        adjacency,
        add_self_loops=add_self_loops,
        normalize=normalize,
        max_hops=max_hops,
    )
    processed.eliminate_zeros()
    return KhopResult(adjacency=processed, n_edges_khop=int(processed.nnz))


ZERO_DEGREE_SUBSTITUTE = 1.0
"""What a zero row sum becomes before a division, in both normalisations.

``preprocess_adjacency`` already does this for ``D⁻¹`` and leaves such a row
untouched; the symmetric form does the same, for the same reason: a node with no
support has nothing to average and dividing it by zero would turn one isolated
conduit into a matrix of NaN. With self-loops on — which every path here uses —
no row is ever zero, so this is a guard and not a code path the study exercises."""


def symmetric_normalization(matrix) -> sp.csr_matrix:
    """``D^-1/2 M D^-1/2`` with ``D = diag(M·1)`` — the **row** degrees of ``M``.

    Applied to a matrix that has already been through the k-hop expansion, which
    is the whole point: normalising ``A`` first and expanding afterwards makes the
    hop count stop mattering, because the powers of a normalised matrix are
    normalised again by the expansion's own weights.

    The conduit graph is **directed** (``graph/build.py``), so ``M`` is asymmetric
    and stays asymmetric here: ``M_uv / sqrt(d_u · d_v)`` scales an entry by the
    degrees of both endpoints without moving it. Symmetrising ``M + Mᵀ`` first
    would give the textbook GCN operator and would erase the downstream/upstream
    axis that study (b2) is measuring, so it is deliberately not done — this is
    "the symmetric normalisation of a directed matrix", not "the symmetric
    operator".

    Args:
        matrix: a square sparse or dense matrix, already expanded and already
            carrying whatever diagonal the caller wants normalised.

    Returns:
        The scaled matrix, as CSR.
    """
    scaled = sp.csr_matrix(matrix) if not sp.issparse(matrix) else matrix.tocsr()
    degrees = np.asarray(scaled.sum(axis=1), dtype="float64").flatten()
    degrees[degrees == 0] = ZERO_DEGREE_SUBSTITUTE
    inverse_sqrt = sp.diags(1.0 / np.sqrt(degrees), format="csr")
    return sp.csr_matrix(inverse_sqrt @ scaled @ inverse_sqrt)


def support_mask(matrix) -> sp.csr_matrix:
    """The binary support of ``matrix``: ``1`` where it is non-zero, ``0`` elsewhere.

    What the ``support_*`` operators aggregate over. They pool ``max`` across the
    neighbourhood, so the hop weights of the expansion would be multiplied into
    the pooled values without ever being compared with each other — the mask is
    the honest input for them, and it is why ``k`` still matters to those
    operators (it decides *who* is a neighbour) while the ``0.5^(h-1)`` weights do
    not.

    The mask is taken **after** the expansion, so a node's support at ``k`` is
    every node within ``k`` hops of it, plus itself when self-loops were added.
    """
    values = sp.csr_matrix(matrix) if not sp.issparse(matrix) else matrix.tocsr()
    binary = values.copy()
    binary.data = np.ones_like(binary.data, dtype="float64")
    binary.eliminate_zeros()
    return sp.csr_matrix(binary)


def operator_adjacency(adjacency, *, operator: str, max_hops: int) -> sp.csr_matrix:
    """The matrix one of the five named graph operators is fed.

    The operator is a pair — preprocessing here, aggregation in
    :class:`stormlab.models.gnn.GraphOperatorLayer` — and this is the first half
    of it. All five start from the same k-hop expansion with self-loops; they
    differ in what happens next:

    ===========================  ====================================================
    ``legacy_weighted_mean``     row-wise ``D⁻¹`` — the untouched B1 path
    ``khop_sum``                 nothing; the expansion's own weights
    ``khop_sym``                 ``D^-1/2 M D^-1/2`` over the expanded matrix
    ``support_max``              the binary support of the expansion
    ``support_pool``             the binary support of the expansion
    ``nograph``                  the all-zero matrix; the layer reads no matrix
    ===========================  ====================================================

    ``nograph`` is the odd one and deliberately so. Its layer has no neighbour
    term, so *any* matrix would give the same answer; returning zeros rather than
    the expansion makes that visible in the artifact — the edge counts a run
    records for this arm are zero, which is what the arm claims — instead of
    leaving a reader to infer that a matrix that was passed was never read.

    ``legacy_weighted_mean`` goes through :func:`apply_khop` unchanged rather than
    through a branch that happens to compute the same thing, so the frozen B1
    weights keep exactly one preprocessing path.

    Args:
        adjacency: the raw adjacency, already transposed if the caller wanted
            upstream message passing.
        operator: one of :data:`stormlab.config.GNN_OPERATORS`.
        max_hops: ``k`` of the expansion.

    Raises:
        ValueError: on an unknown operator or a non-positive ``max_hops``.
    """
    if operator not in GNN_OPERATORS:
        raise ValueError(f"Unknown graph operator {operator!r}; expected one of {list(GNN_OPERATORS)}")
    if max_hops < 1:
        raise ValueError(f"max_hops must be at least 1, got {max_hops}")

    if operator == GNN_OPERATOR_NOGRAPH:
        matrix = sp.csr_matrix(adjacency)
        return sp.csr_matrix(matrix.shape, dtype="float64")

    if operator == GNN_OPERATOR_LEGACY_WEIGHTED_MEAN:
        return apply_khop(adjacency, max_hops=max_hops).adjacency

    expanded = preprocess_adjacency(adjacency, add_self_loops=True, normalize=False, max_hops=max_hops)
    expanded.eliminate_zeros()
    if operator == GNN_OPERATOR_KHOP_SUM:
        return expanded
    if operator == GNN_OPERATOR_KHOP_SYM:
        return symmetric_normalization(expanded)
    if operator in (GNN_OPERATOR_SUPPORT_MAX, GNN_OPERATOR_SUPPORT_POOL):
        return support_mask(expanded)
    raise ValueError(f"Graph operator {operator!r} has no preprocessing branch")  # pragma: no cover - closed set


__all__ = [
    "DEFAULT_KHOP_SCHEME",
    "ZERO_DEGREE_SUBSTITUTE",
    "KhopResult",
    "apply_khop",
    "operator_adjacency",
    "preprocess_adjacency",
    "support_mask",
    "symmetric_normalization",
]
