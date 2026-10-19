"""Conduit adjacency: nodes are conduits, edges are hydraulic connections.

The edge rule is copied one-for-one from ``app/sa/core/gnn/graph.py``: an edge
``A[upstream, current] = 1`` exists whenever ``upstream.OutletNode`` equals
``current.InletNode`` after ``strip().lower()``. Row-normalising such a matrix
and computing ``A @ X`` makes every conduit aggregate over its *downstream*
neighbours — the opposite of hydraulic intuition. That semantics is preserved
deliberately (the saved weights were trained with it); ``direction="upstream"``
transposes the result for the named counter-experiment.

Two behaviours of the legacy code are not preserved, both because they were
silent:

* conduits whose ``Name``/``InletNode``/``OutletNode`` is missing were dropped
  with a log line and left as empty rows; here they raise
  :class:`GraphBuildError` naming the offending conduits;
* similarity- and distance-based substitute edges (weights 0.5 and 0.3) were
  injected into the matrix whenever the graph looked too disconnected. They are
  not hydraulics, so they now sit behind ``surrogate_edges`` (off by default)
  and their count is reported in :class:`GraphBuildResult`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from stormlab.config import GRAPH_DIRECTIONS

logger = logging.getLogger(__name__)


def n_rows(matrix: sp.spmatrix) -> int:
    """Number of rows of a sparse matrix.

    ``scipy``'s type stubs declare ``shape`` as optional, so every ``shape[0]``
    reads as a possible subscript of ``None``. Narrowing it in one named place is
    the alternative to silencing the same warning at each call site.

    Raises:
        ValueError: if the matrix carries no shape at all.
    """
    shape = matrix.shape
    if shape is None:
        raise ValueError(f"Sparse matrix of type {type(matrix).__name__} carries no shape")
    return int(shape[0])


Direction = Literal["downstream", "upstream"]

NODE_COLUMNS: tuple[str, ...] = ("InletNode", "OutletNode")

SPATIAL_COLUMNS: tuple[str, ...] = ("X1", "Y1", "X2", "Y2", "Length")
SPATIAL_EDGE_WEIGHT = 0.5
SPATIAL_MAX_EDGES = 20
SPATIAL_LENGTH_TOLERANCE = 0.2

FEATURE_COLUMNS: tuple[str, ...] = ("Diameter", "Length", "Roughness")
FEATURE_EDGE_WEIGHT = 0.3
FEATURE_MAX_EDGES = 50
FEATURE_DISTANCE_THRESHOLD = 0.5

DISCONNECTED_COMPONENT_RATIO = 0.9


class GraphBuildError(ValueError):
    """Raised when the conduit frame cannot yield an unambiguous graph."""


@dataclass(frozen=True)
class GraphBuildResult:
    """A built adjacency together with everything needed to interpret it.

    Attributes:
        adjacency: ``float32`` CSR matrix with a zero diagonal, rows and columns
            aligned to ``order``.
        order: conduit names in matrix row order.
        n_edges_raw: number of hydraulic edges, counted before any substitute
            edge was considered.
        n_surrogate_edges: number of non-hydraulic substitute edges added
            (always ``0`` unless ``surrogate_edges=True``).
    """

    adjacency: sp.csr_matrix
    order: tuple[str, ...]
    n_edges_raw: int
    n_surrogate_edges: int


@dataclass(frozen=True)
class GraphChannels:
    """Both message-passing channels of one network, from one build of the edges.

    ``downstream`` is the matrix the pipeline has always built — the legacy
    orientation ``A[upstream, current] = 1`` — and ``upstream`` is its transpose,
    which is exactly what ``direction="upstream"`` has always returned. Nothing is
    recomputed: the two are the same edges read in the two directions, so a
    two-channel model and a one-channel model of either direction are fed
    identical numbers.

    The pair exists because ``direction="both"`` of plan16 §3 (E6) is **not** a
    third orientation of one matrix: it is two neighbourhoods aggregated by two
    separate sets of weights, and merging them into one adjacency is precisely
    what the plan forbids.

    Attributes:
        downstream: the stored orientation, ``float32`` CSR with a zero diagonal.
        upstream: its transpose, same dtype and same row order.
        order: conduit names in matrix row order; shared by both channels.
        n_edges_raw: hydraulic edges, counted once — transposing moves no edge.
        n_surrogate_edges: non-hydraulic substitute edges added, likewise counted
            once.
    """

    downstream: sp.csr_matrix
    upstream: sp.csr_matrix
    order: tuple[str, ...]
    n_edges_raw: int
    n_surrogate_edges: int

    def channel(self, direction: str) -> sp.csr_matrix:
        """The single channel named by ``direction``.

        Raises:
            ValueError: on anything but the two values of
                :data:`stormlab.config.GRAPH_DIRECTIONS`. ``"both"`` is
                deliberately refused: it names the pair, and answering it with one
                matrix is the merge this class exists to prevent.
        """
        if direction == "downstream":
            return self.downstream
        if direction == "upstream":
            return self.upstream
        raise ValueError(
            f"channel() answers a single direction, got {direction!r}; expected one of {list(GRAPH_DIRECTIONS)}. "
            "A two-channel model reads both attributes, it does not ask for one matrix"
        )


class SWMMGraphConstructor:
    """Builds the conduit adjacency for one network.

    Args:
        conduits_data: conduit frame; ``Name`` falls back to the index.
        conduit_order: conduit names defining the matrix row order. ``None``
            keeps the frame's own row order.
        direction: ``"downstream"`` keeps the legacy orientation,
            ``"upstream"`` returns its transpose.
        surrogate_edges: allow non-hydraulic substitute edges when the graph is
            almost fully disconnected.
    """

    def __init__(
        self,
        conduits_data: pd.DataFrame,
        conduit_order: Sequence[str] | None = None,
        *,
        direction: Direction = "downstream",
        surrogate_edges: bool = False,
    ) -> None:
        if direction not in GRAPH_DIRECTIONS:
            raise ValueError(f"Unknown graph direction {direction!r}; expected one of {list(GRAPH_DIRECTIONS)}")
        self.direction: Direction = direction
        self.surrogate_edges = surrogate_edges
        self.conduits_data = self._prepare_conduits_data(conduits_data, conduit_order)
        self.conduit_to_idx: dict[str, int] = {}
        self.idx_to_conduit: dict[int, str] = {}

    def _prepare_conduits_data(self, conduits_data: pd.DataFrame, conduit_order: Sequence[str] | None) -> pd.DataFrame:
        data = conduits_data.copy()
        if "Name" not in data.columns:
            data["Name"] = data.index.astype(str)

        data["Name"] = data["Name"].astype(str)

        if conduit_order is not None:
            order = [str(name) for name in conduit_order]
            if data["Name"].duplicated().any():
                duplicates = data.loc[data["Name"].duplicated(), "Name"].unique().tolist()
                raise GraphBuildError(f"Duplicate conduit names make conduit_order ambiguous: {duplicates}")

            indexed = data.set_index("Name", drop=False)
            missing = [name for name in order if name not in indexed.index]
            if missing:
                raise GraphBuildError(f"conduit_order contains names missing from conduits_data: {missing}")
            data = indexed.loc[order].reset_index(drop=True)

        return data.reset_index(drop=True)

    def build_conduit_graph(self) -> GraphBuildResult:
        """Build the directed conduit graph.

        Raises:
            GraphBuildError: if a required node column is absent, or if any
                conduit has a missing or blank inlet/outlet node.
        """
        data = self.conduits_data
        self._validate_node_columns(data)

        n_conduits = len(data)
        self.conduit_to_idx = {name: idx for idx, name in enumerate(data["Name"])}
        self.idx_to_conduit = {idx: name for name, idx in self.conduit_to_idx.items()}
        if len(self.conduit_to_idx) != n_conduits:
            duplicates = data.loc[data["Name"].duplicated(), "Name"].unique().tolist()
            raise GraphBuildError(f"Duplicate conduit names make the graph ambiguous: {duplicates}")

        inlets = _node_ids(data["InletNode"])
        outlets = _node_ids(data["OutletNode"])
        self._reject_missing_nodes(data["Name"], inlets, outlets)

        adjacency = sp.lil_matrix((n_conduits, n_conduits), dtype=np.float32)
        outlet_to_conduits: dict[str, set[int]] = defaultdict(set)
        for idx, outlet_node in enumerate(outlets):
            outlet_to_conduits[outlet_node].add(idx)

        added_edges: set[tuple[int, int]] = set()
        no_match_count = 0
        for current_idx, inlet_node in enumerate(inlets):
            upstream_conduits = outlet_to_conduits.get(inlet_node)
            if upstream_conduits is None:
                no_match_count += 1
                continue
            for upstream_idx in upstream_conduits:
                if upstream_idx == current_idx:
                    continue
                edge_key = (upstream_idx, current_idx)
                if edge_key in added_edges:
                    continue
                adjacency[upstream_idx, current_idx] = 1.0
                added_edges.add(edge_key)

        n_edges_raw = len(added_edges)
        logger.info(
            "Built raw conduit graph: nodes=%s, directed_edges=%s, conduits_without_upstream=%s",
            n_conduits,
            n_edges_raw,
            no_match_count,
        )

        n_surrogate_edges = self._maybe_add_surrogate_edges(data, adjacency, added_edges)

        csr = sp.csr_matrix(adjacency.tocsr(), dtype=np.float32)
        csr.setdiag(0)
        csr.eliminate_zeros()
        if self.direction == "upstream":
            csr = sp.csr_matrix(csr.transpose())

        self._log_connectivity(csr)
        return GraphBuildResult(
            adjacency=csr,
            order=tuple(data["Name"]),
            n_edges_raw=n_edges_raw,
            n_surrogate_edges=n_surrogate_edges,
        )

    @staticmethod
    def _validate_node_columns(data: pd.DataFrame) -> None:
        missing = [column for column in NODE_COLUMNS if column not in data.columns]
        if missing:
            raise GraphBuildError(f"Conduit frame is missing node columns {missing}; cannot derive hydraulic edges")

    @staticmethod
    def _reject_missing_nodes(names: pd.Series, inlets: list[str], outlets: list[str]) -> None:
        """Fail loudly on conduits the legacy code used to drop in a log line."""
        offenders = [
            f"{name} (InletNode={inlet!r}, OutletNode={outlet!r})"
            for name, inlet, outlet in zip(names, inlets, outlets)
            if not inlet or not outlet
        ]
        if offenders:
            raise GraphBuildError(
                f"{len(offenders)} conduit(s) have a missing or blank inlet/outlet node and cannot be placed "
                f"in the graph: {offenders}"
            )

    def _maybe_add_surrogate_edges(self, data: pd.DataFrame, adjacency: sp.lil_matrix, added_edges: set[tuple[int, int]]) -> int:
        n_conduits = len(data)
        if n_conduits == 0:
            return 0

        n_components = connected_components(adjacency.tocsr(), directed=True, connection="weak", return_labels=False)
        if n_components <= DISCONNECTED_COMPONENT_RATIO * n_conduits:
            return 0

        logger.warning(
            "Graph is highly disconnected (%s weak components for %s conduits).",
            n_components,
            n_conduits,
        )
        if not self.surrogate_edges:
            logger.info("Substitute edges are disabled; leaving the graph disconnected.")
            return 0

        extra_edges = self._try_spatial_connections(data, adjacency, added_edges)
        if extra_edges < 10:
            extra_edges += self._try_feature_based_connections(data, adjacency, added_edges, max_edges=FEATURE_MAX_EDGES)
        logger.warning(
            "Added %s non-hydraulic substitute edges (weights %s and %s).",
            extra_edges,
            SPATIAL_EDGE_WEIGHT,
            FEATURE_EDGE_WEIGHT,
        )
        return extra_edges

    def _try_spatial_connections(
        self, clean_data: pd.DataFrame, adjacency: sp.lil_matrix, added_edges: set[tuple[int, int]]
    ) -> int:
        """Connect conduits of similar ``Length``, in ascending length order."""
        available_spatial = [column for column in SPATIAL_COLUMNS if column in clean_data.columns]
        if len(available_spatial) < 2 or "Length" not in clean_data.columns:
            logger.warning("Not enough spatial data available to make spatial fallback connections.")
            return 0

        lengths = pd.to_numeric(clean_data["Length"], errors="coerce")
        sortable = clean_data.assign(_LengthNumeric=lengths).dropna(subset=["_LengthNumeric"]).sort_values("_LengthNumeric")

        extra_edges = 0
        for position in range(len(sortable) - 1):
            row1 = sortable.iloc[position]
            row2 = sortable.iloc[position + 1]
            length1 = float(row1["_LengthNumeric"])
            length2 = float(row2["_LengthNumeric"])
            if abs(length1 - length2) / max(length1, 1) >= SPATIAL_LENGTH_TOLERANCE:
                continue

            idx1 = self.conduit_to_idx[row1["Name"]]
            idx2 = self.conduit_to_idx[row2["Name"]]
            if idx1 == idx2:
                continue

            edge_key = (idx1, idx2)
            if edge_key in added_edges:
                continue

            adjacency[idx1, idx2] = SPATIAL_EDGE_WEIGHT
            added_edges.add(edge_key)
            extra_edges += 1
            if extra_edges >= SPATIAL_MAX_EDGES:
                break

        logger.info("Added %s spatial fallback connections.", extra_edges)
        return extra_edges

    def _try_feature_based_connections(
        self,
        clean_data: pd.DataFrame,
        adjacency: sp.lil_matrix,
        added_edges: set[tuple[int, int]],
        max_edges: int = FEATURE_MAX_EDGES,
    ) -> int:
        """Connect conduits close to each other in standardised feature space."""
        available_features = [column for column in FEATURE_COLUMNS if column in clean_data.columns]
        if not available_features or len(clean_data) < 2:
            logger.warning("No suitable features available for fallback comparison.")
            return 0

        normalized = clean_data[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        for column in available_features:
            spread = normalized[column].std()
            if spread > 0:
                normalized[column] = (normalized[column] - normalized[column].mean()) / spread

        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(normalized.values))
        extra_edges = 0
        for i in range(len(clean_data)):
            for j in range(i + 1, len(clean_data)):
                if distances[i, j] >= FEATURE_DISTANCE_THRESHOLD:
                    continue

                idx1 = self.conduit_to_idx[clean_data.iloc[i]["Name"]]
                idx2 = self.conduit_to_idx[clean_data.iloc[j]["Name"]]
                if idx1 == idx2:
                    continue

                if (idx1, idx2) not in added_edges:
                    adjacency[idx1, idx2] = FEATURE_EDGE_WEIGHT
                    added_edges.add((idx1, idx2))
                    extra_edges += 1

                if (idx2, idx1) not in added_edges and extra_edges < max_edges:
                    adjacency[idx2, idx1] = FEATURE_EDGE_WEIGHT
                    added_edges.add((idx2, idx1))
                    extra_edges += 1

                if extra_edges >= max_edges:
                    return extra_edges

        return extra_edges

    @staticmethod
    def _log_connectivity(adjacency: sp.csr_matrix) -> None:
        n_nodes = n_rows(adjacency)
        if n_nodes == 0:
            logger.info("Connectivity analysis: empty graph.")
            return

        n_components, labels = connected_components(adjacency, directed=True, connection="weak", return_labels=True)
        component_sizes = sorted(np.bincount(labels).tolist(), reverse=True) if labels.size else []
        in_degrees = np.asarray(adjacency.sum(axis=0)).ravel()
        out_degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        isolated = int(np.count_nonzero((in_degrees == 0) & (out_degrees == 0)))
        density = adjacency.nnz / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

        logger.info(
            "Connectivity analysis: components=%s, largest_components=%s, isolated=%s (%.1f%%), density=%.4f",
            n_components,
            component_sizes[:5],
            isolated,
            isolated / n_nodes * 100,
            density,
        )


def _node_ids(column: pd.Series) -> list[str]:
    """Node identifiers as the edge rule compares them: stripped and lowercased."""
    return ["" if pd.isna(value) else str(value).strip().lower() for value in column]


def build_graph(
    dfc: pd.DataFrame,
    conduit_order: Sequence[str] | None = None,
    *,
    direction: Direction = "downstream",
    surrogate_edges: bool = False,
) -> GraphBuildResult:
    """Build the conduit adjacency together with its order and edge counts.

    Args:
        dfc: conduit frame; ``Name`` falls back to the index.
        conduit_order: conduit names defining the matrix row order; ``None``
            keeps the frame's own row order.
        direction: ``"downstream"`` (legacy, ``A[upstream, current] = 1``) or
            ``"upstream"`` (its transpose).
        surrogate_edges: allow non-hydraulic substitute edges.

    Raises:
        GraphBuildError: on duplicate names, an unknown name in
            ``conduit_order``, absent node columns or missing node ids.
        ValueError: on an unknown ``direction``.
    """
    constructor = SWMMGraphConstructor(
        dfc,
        conduit_order=conduit_order,
        direction=direction,
        surrogate_edges=surrogate_edges,
    )
    return constructor.build_conduit_graph()


def build_graph_channels(
    dfc: pd.DataFrame,
    conduit_order: Sequence[str] | None = None,
    *,
    surrogate_edges: bool = False,
) -> GraphChannels:
    """Build the conduit graph once and return both of its channels.

    The downstream channel is :func:`build_graph`'s ``direction="downstream"``
    result object for object, and the upstream channel is its transpose — the
    same operation ``direction="upstream"`` applies. Building once and transposing
    is what makes the two channels provably the same edges; two calls could
    disagree if the frame were mutated between them.

    Args:
        dfc: conduit frame; ``Name`` falls back to the index.
        conduit_order: conduit names defining the matrix row order; ``None``
            keeps the frame's own row order.
        surrogate_edges: allow non-hydraulic substitute edges.

    Raises:
        GraphBuildError: for the reasons :func:`build_graph` raises it.
    """
    result = build_graph(
        dfc,
        conduit_order=conduit_order,
        direction="downstream",
        surrogate_edges=surrogate_edges,
    )
    return GraphChannels(
        downstream=result.adjacency,
        upstream=sp.csr_matrix(result.adjacency.transpose()),
        order=result.order,
        n_edges_raw=result.n_edges_raw,
        n_surrogate_edges=result.n_surrogate_edges,
    )


def build_adjacency_from_dfc(
    dfc: pd.DataFrame,
    conduit_order: Sequence[str] | None = None,
    *,
    direction: Direction = "downstream",
    surrogate_edges: bool = False,
) -> sp.csr_matrix:
    """The adjacency alone, for callers that already know the order."""
    return build_graph(
        dfc,
        conduit_order=conduit_order,
        direction=direction,
        surrogate_edges=surrogate_edges,
    ).adjacency
