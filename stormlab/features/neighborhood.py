"""The graph neighbourhood of a conduit, summarised as columns.

A tabular model sees one conduit at a time. A graph model sees its neighbours as
well, so a comparison between the two measures two things at once: the
architecture, and the information. These aggregates separate them -- the tabular
model is handed the same neighbourhood the message passing would have reached,
as ``min``/``max``/``mean`` of a few measured columns over the conduits within
``k`` hops upstream and downstream (plan16 §3, E6).

Three rules the module keeps and the tests pin.

* **Only measurements enter.** The aggregated columns are named by
  ``config.c1.neighborhood.columns`` and are raw quantities of the physical view;
  an aggregate over the neighbours' *labels* or decision flags would hand a model
  a summary of the answer its neighbours were given.
* **A missing value is not a zero.** The reductions are NaN-aware, so one
  neighbour whose ``min_slope_required`` was not computable does not erase the
  column for the whole neighbourhood. When *no* neighbour carries a finite value
  the aggregate is :data:`EMPTY_AGGREGATE`, which is the same answer as "no
  neighbour at all" -- and the pair ``count``/``present`` is what distinguishes
  the two cases, because they describe the graph rather than the values.
* **An isolated conduit gets zeros, not the population mean.** Substituting a
  dataset-wide average would tell the model something about the pool that the
  conduit's own network never said.

The column *names* and the registered ``v3_physical_nbr`` contract live one layer
down, in :mod:`stormlab.contracts.features`: what the aggregates are called is a
contract, not an implementation detail of the code that fills them, and
``config.c1.neighborhood`` is validated against the same two constants.

The adjacency is built by :mod:`stormlab.graph.build`, but it is *passed in*
rather than imported: ``features`` sits below ``graph`` in the layer stack
(``tests/contract/test_layering.py``), so the graph channels arrive through the
structural :class:`ConduitChannels` interface, which
``stormlab.graph.build.GraphChannels`` satisfies as it stands.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from stormlab.contracts.features import (
    FLAG_DTYPE,
    NEIGHBORHOOD_COUNT_SUFFIX,
    NEIGHBORHOOD_DIRECTIONS,
    NEIGHBORHOOD_PRESENT_SUFFIX,
    NEIGHBORHOOD_STATISTICS,
    neighborhood_column_names,
)

CONDUIT_COLUMN = "conduit"
"""The identity column joined against the graph's row order."""

EMPTY_AGGREGATE = 0.0
"""What a statistic is when the neighbourhood contributes no finite value."""


class NeighborhoodError(ValueError):
    """Raised when the aggregates cannot be computed from the frame and the graph given."""


class ConduitChannels(Protocol):
    """Both message-passing channels of one network, as this module reads them.

    Structural rather than imported, for the layering reason in the module
    docstring. The members are read-only properties because the object that
    satisfies this in production -- ``graph.build.GraphChannels`` -- is a frozen
    dataclass, and a writable protocol member would refuse it.
    """

    @property
    def upstream(self) -> sp.csr_matrix:
        """Row ``i`` holds the conduits immediately upstream of conduit ``i``."""
        ...

    @property
    def downstream(self) -> sp.csr_matrix:
        """Row ``i`` holds the conduits immediately downstream of conduit ``i``."""
        ...

    @property
    def order(self) -> tuple[str, ...]:
        """Conduit names in matrix row order."""
        ...


def neighborhood_aggregates(
    frame: pd.DataFrame,
    graph: ConduitChannels,
    *,
    columns: Sequence[str],
    hops: Sequence[int],
) -> pd.DataFrame:
    """Summaries of ``columns`` over the up- and downstream neighbourhoods of each conduit.

    Args:
        frame: one network's conduit rows, carrying :data:`CONDUIT_COLUMN` and
            every name in ``columns``. Row order is free -- the join is by conduit
            name -- and the result carries the frame's own index.
        graph: the two channels of the same network, as
            :func:`stormlab.graph.build.build_graph_channels` returns them.
        columns: the measured columns to aggregate; ``config.c1.neighborhood.columns``.
        hops: the neighbourhood radii; ``config.c1.neighborhood.hops``. The
            neighbourhood at ``k`` is every conduit reachable in 1..k hops,
            excluding the conduit itself.

    Returns:
        A frame indexed like ``frame``, holding exactly
        :func:`stormlab.contracts.features.neighborhood_column_names` in that order.

    Raises:
        NeighborhoodError: on a missing column, a conduit set that does not match
            the graph's row order, a repeated conduit name, or a non-numeric
            column.
    """
    generated = neighborhood_column_names(columns, hops)
    names = [str(item) for item in columns]
    steps = [int(item) for item in hops]

    missing = [column for column in (CONDUIT_COLUMN, *names) if column not in frame.columns]
    if missing:
        raise NeighborhoodError(f"Neighbourhood aggregates need column(s) {missing}; the frame has {list(frame.columns)}")

    position = _row_positions(frame, graph)
    # In *graph* row order, not the frame's: `neighbours.indices` are positions in
    # the adjacency, so a frame ordered differently would have every neighbour's
    # value read off the wrong conduit -- and the final reindex, which happens
    # after the reductions, could not undo it.
    values = {column: _numeric(frame, column)[_inverse(position, len(graph.order))] for column in names}

    block: dict[str, np.ndarray] = {}
    for direction, adjacency in zip(NEIGHBORHOOD_DIRECTIONS, (graph.upstream, graph.downstream)):
        for step in steps:
            neighbours = within_k_hops(adjacency, step)
            counts = np.diff(neighbours.indptr).astype("float64")
            for column in names:
                statistics = _reduce(neighbours, values[column])
                for statistic, result in zip(NEIGHBORHOOD_STATISTICS, statistics):
                    block[f"{direction}{step}_{column}_{statistic}"] = result
            block[f"{direction}{step}_{NEIGHBORHOOD_COUNT_SUFFIX}"] = counts
            block[f"{direction}{step}_{NEIGHBORHOOD_PRESENT_SUFFIX}"] = (counts > 0).astype(FLAG_DTYPE)

    return pd.DataFrame({name: block[name][position] for name in generated}, index=frame.index)


def within_k_hops(adjacency: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """The boolean support of ``A + A² + … + A^k`` with the diagonal cleared.

    That is exactly "every conduit reachable in one to ``k`` hops, itself
    excluded". The same set is what the graph models see: it equals
    ``graph.khop.support_mask(preprocess_adjacency(A, add_self_loops=True,
    normalize=False, max_hops=k))`` minus its diagonal, and a test asserts that
    equality on a real network so the two definitions of "within k hops" cannot
    drift apart. It is written here rather than imported because ``features`` sits
    below ``graph`` in the layer stack.

    Raises:
        NeighborhoodError: on a non-positive ``k``.
    """
    if k < 1:
        raise NeighborhoodError(f"A neighbourhood spans at least one hop, got {k}")
    step = _binary(adjacency)
    reached = step.copy()
    power = step
    for _ in range(2, k + 1):
        power = _binary(power @ step)
        reached = _binary(reached + power)
    reached.setdiag(0)
    reached.eliminate_zeros()
    return reached


def _binary(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """``matrix`` with every stored non-zero replaced by 1.0.

    Re-binarising after every product keeps the entries from growing with the
    number of distinct walks: the module asks *whether* a conduit is reachable,
    never how many ways there are to reach it.
    """
    result = sp.csr_matrix(matrix, dtype="float64", copy=True)
    result.eliminate_zeros()
    result.data = np.ones_like(result.data)
    return result


def _inverse(position: np.ndarray, n_rows: int) -> np.ndarray:
    """For each graph row, which row of the frame carries it.

    The inverse permutation of :func:`_row_positions`. Both directions are needed:
    this one to read the frame in graph order, that one to write the result back
    in frame order.
    """
    inverse = np.empty(n_rows, dtype="int64")
    inverse[position] = np.arange(len(position), dtype="int64")
    return inverse


def _row_positions(frame: pd.DataFrame, graph: ConduitChannels) -> np.ndarray:
    """For each row of ``frame``, its position in the graph's row order.

    Raises:
        NeighborhoodError: if the two disagree about which conduits the network
            has, or if the frame names one twice.
    """
    order = [str(name) for name in graph.order]
    width = graph.upstream.shape[0] if graph.upstream.shape is not None else 0
    if len(order) != width:
        raise NeighborhoodError(f"The graph names {len(order)} conduit(s) but its adjacency has {width} row(s)")

    conduits = frame[CONDUIT_COLUMN].astype(str)
    if conduits.duplicated().any():
        repeated = sorted(set(conduits[conduits.duplicated()]))
        raise NeighborhoodError(f"Conduit name(s) {repeated[:5]} occur more than once; a row is one conduit")
    index = {name: position for position, name in enumerate(order)}
    unknown = sorted(set(conduits) - set(index))
    if unknown:
        raise NeighborhoodError(f"{len(unknown)} conduit(s) are absent from the graph, e.g. {unknown[:5]}")
    absent = sorted(set(index) - set(conduits))
    if absent:
        raise NeighborhoodError(
            f"The graph holds {len(absent)} conduit(s) the frame does not, e.g. {absent[:5]}; "
            "the aggregates would then be taken over neighbours whose values are unknown"
        )
    return conduits.map(index).to_numpy(dtype="int64")


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    """One column as ``float64``, in the frame's own row order.

    Raises:
        NeighborhoodError: if the column cannot be read as a number.
    """
    try:
        return frame[column].to_numpy(dtype="float64")
    except (TypeError, ValueError) as error:
        raise NeighborhoodError(
            f"Column {column!r} is not numeric ({frame[column].dtype}); a neighbourhood aggregate has no meaning for it"
        ) from error


def _reduce(neighbours: sp.csr_matrix, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``min``, ``max`` and ``mean`` of ``values`` over each row's neighbours, NaN-aware.

    The neighbours of a row are contiguous in the CSR index array, so the two
    order statistics are one ``reduceat`` each over the gathered values;
    ``np.fmin``/``np.fmax`` skip NaN, which is the whole point -- a neighbour
    whose value was not measurable must not erase the others.

    A row whose neighbours carry no finite value at all -- because there are no
    neighbours, or because all of theirs are NaN -- gets
    :data:`EMPTY_AGGREGATE` for all three.
    """
    ordered_by_row = values[neighbours.indices]
    finite = np.isfinite(values).astype("float64")
    n_finite = np.asarray(neighbours @ finite, dtype="float64").reshape(-1)
    total = np.asarray(neighbours @ np.where(np.isfinite(values), values, 0.0), dtype="float64").reshape(-1)

    n_rows = neighbours.shape[0] if neighbours.shape is not None else 0
    if ordered_by_row.size == 0:
        empty = np.full(n_rows, EMPTY_AGGREGATE, dtype="float64")
        return empty, empty.copy(), empty.copy()

    # `reduceat` reads segment [starts[i], starts[i + 1]) and refuses a start past
    # the end of the array. One NaN of padding makes `nnz` a legal start, which is
    # what a trailing row with no neighbours produces; clamping it to `nnz - 1`
    # instead would silently shorten the *previous* row's segment by one neighbour.
    # NaN is the neutral element of `fmin`/`fmax`, and rows that read only padding
    # are overwritten by the `nothing` mask below anyway.
    padded = np.append(ordered_by_row, np.nan)
    starts = neighbours.indptr[:-1]
    minima = np.fmin.reduceat(padded, starts)
    maxima = np.fmax.reduceat(padded, starts)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(n_finite > 0, total / np.where(n_finite > 0, n_finite, 1.0), EMPTY_AGGREGATE)

    nothing = n_finite == 0
    minima = np.where(nothing, EMPTY_AGGREGATE, minima)
    maxima = np.where(nothing, EMPTY_AGGREGATE, maxima)
    return minima, maxima, means


__all__ = [
    "CONDUIT_COLUMN",
    "EMPTY_AGGREGATE",
    "ConduitChannels",
    "NeighborhoodError",
    "neighborhood_aggregates",
    "within_k_hops",
]
