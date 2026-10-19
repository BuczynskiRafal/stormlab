"""Spatial consistency (SL) of a recommendation field over the conduit graph.

SL asks how often two hydraulically connected conduits get recommendations that
cannot sensibly stand next to each other. What counts as "cannot" is an explicit
9x9 boolean matrix, passed in, rather than a rule buried in the metric.

The historical ``calculate_spatial_consistency``
(``models/recomendations/gnn/experiments.py:26``) had no such matrix: it simply
checked ``argmax(i) == argmax(j)``, i.e. it treated *any* two different classes
as incompatible. :data:`INCOMPATIBLE_ARGMAX_IDENTITY` reproduces exactly that
definition, so the historical number stays reproducible while the concept it
encodes ("different means inconsistent") is visible instead of implied.

It is not a default, and this module defines no hydraulic rules of its own. A
hydraulically motivated 9x9 matrix is a modelling decision that belongs to the
experiment that makes it; ``incompatible`` is a required argument so that every
reported SL value names the matrix it was computed under.

Two deliberate departures from the historical function, both required by
plan11 §4b:

* it is computed on the **raw** adjacency, not on the k-hop preprocessed one —
  otherwise "neighbours" include hop-expanded and substitute pairs;
* it iterates **every** directed edge, not only the ``i < j`` half of the COO
  entries. The old filter silently dropped every edge whose upstream conduit
  happened to sit at a higher row index, making the metric depend on the row
  order of the INP file.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd
import scipy.sparse as sp

from stormlab.contracts.enums import RecommendationCategory


class SpatialConsistencyError(ValueError):
    """Raised when spatial consistency cannot be computed on the given input."""


def _identity_negation() -> pd.DataFrame:
    categories = list(RecommendationCategory)
    return pd.DataFrame(
        [[left is not right for right in categories] for left in categories],
        index=pd.Index(categories, name="upstream"),
        columns=pd.Index(categories, name="downstream"),
        dtype=bool,
    )


INCOMPATIBLE_ARGMAX_IDENTITY: pd.DataFrame = _identity_negation()
"""9x9 boolean matrix in which any two *different* recommendations clash.

The historical ``argmax(i) == argmax(j)`` rule written out in full. It is a named
constant so the legacy number stays reproducible and so the assumption behind it
("different means inconsistent") can be read, argued with and cited — never a
default. Which matrix a study uses is an input of that study: pass it explicitly,
under a name that says which one it is.
"""


def spatial_consistency(
    pred_labels: pd.Series,
    adjacency_raw: sp.csr_matrix,
    order: Sequence[str],
    incompatible: pd.DataFrame,
) -> float:
    """Share of graph edges whose two recommendations are compatible.

    Args:
        pred_labels: predicted category per conduit, indexed by conduit name.
        adjacency_raw: raw adjacency (zero diagonal, no k-hop expansion) whose
            rows follow ``order``.
        order: conduit names in adjacency row order.
        incompatible: 9x9 boolean matrix indexed by
            :class:`RecommendationCategory`; ``True`` marks a clashing pair.
            Required: pass :data:`INCOMPATIBLE_ARGMAX_IDENTITY` to reproduce the
            historical number, or the matrix the study argues for.

    Returns:
        ``1 - inconsistent_edges / edges``; ``1.0`` means no clash at all.

    Raises:
        SpatialConsistencyError: if the graph has no edges, if the adjacency
            does not match ``order``, or if a conduit has no prediction.
    """
    consistent, total = _count_edges(pred_labels, adjacency_raw, order, incompatible)
    if total == 0:
        raise SpatialConsistencyError(
            "Spatial consistency is undefined on a graph without edges; a returned 0.0 would be indistinguishable "
            "from a fully inconsistent network"
        )
    return consistent / total


def pooled_spatial_consistency(
    graphs: Iterable[tuple[pd.Series, sp.csr_matrix, Sequence[str]]],
    incompatible: pd.DataFrame,
) -> float:
    """Spatial consistency over several networks, pooled edge-wise.

    Edges from all graphs go into one population, so a network with three
    conduits does not weigh as much as one with sixty. That is what "over all
    test simulations" means; a mean of per-graph scores would be a different
    quantity.

    Args:
        graphs: ``(pred_labels, adjacency_raw, order)`` triples.
        incompatible: see :func:`spatial_consistency`.

    Raises:
        SpatialConsistencyError: if no graph contributes a single edge.
    """
    consistent_total = 0
    edges_total = 0
    for pred_labels, adjacency_raw, order in graphs:
        consistent, total = _count_edges(pred_labels, adjacency_raw, order, incompatible)
        consistent_total += consistent
        edges_total += total

    if edges_total == 0:
        raise SpatialConsistencyError("None of the supplied graphs has an edge; spatial consistency is undefined")
    return consistent_total / edges_total


def _count_edges(
    pred_labels: pd.Series,
    adjacency_raw: sp.csr_matrix,
    order: Sequence[str],
    incompatible: pd.DataFrame,
) -> tuple[int, int]:
    """Return ``(compatible_edges, total_edges)`` for one graph."""
    _validate_incompatible(incompatible)
    names = [str(name) for name in order]
    if adjacency_raw.shape != (len(names), len(names)):
        raise SpatialConsistencyError(f"Adjacency of shape {adjacency_raw.shape} does not match {len(names)} conduits in order")

    labels = _as_categories(pred_labels)
    missing = [name for name in names if name not in labels]
    if missing:
        raise SpatialConsistencyError(f"No prediction for conduit(s) {missing}")

    coo = adjacency_raw.tocoo()
    consistent = 0
    total = 0
    for row, column in zip(coo.row.tolist(), coo.col.tolist()):
        if row == column:
            continue
        total += 1
        if not bool(incompatible.at[labels[names[row]], labels[names[column]]]):
            consistent += 1
    return consistent, total


def _as_categories(pred_labels: pd.Series) -> Mapping[str, RecommendationCategory]:
    mapping: dict[str, RecommendationCategory] = {}
    for name, value in pred_labels.items():
        try:
            mapping[str(name)] = value if isinstance(value, RecommendationCategory) else RecommendationCategory(value)
        except ValueError as error:
            raise SpatialConsistencyError(f"Prediction {value!r} for conduit {name} is not a recommendation class") from error
    return mapping


def _validate_incompatible(incompatible: pd.DataFrame) -> None:
    categories = list(RecommendationCategory)
    if list(incompatible.index) != categories or list(incompatible.columns) != categories:
        raise SpatialConsistencyError(
            "The incompatible matrix must be indexed by RecommendationCategory in declaration order on both axes"
        )
    if not all(pd.api.types.is_bool_dtype(dtype) for dtype in incompatible.dtypes):
        raise SpatialConsistencyError("The incompatible matrix must be boolean")
