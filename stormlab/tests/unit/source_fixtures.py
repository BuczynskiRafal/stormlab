"""Neutral F2/F3 chains shared by the four label-source test modules.

A helper module rather than a fixture inside one test file, for the reason
``motif_fixtures.py`` is one: the local sources, the geometry sources, the context
sources and the aggregator are tested from four sides and they have to agree about
what a *quiet* conduit is. If each file built its own neutral row, a source could
pass its own test and fail another's for reasons of fixture rather than of code.

"Neutral" here has a precise meaning: on a chain built from these values exactly one
source fires, ``no_violation``, and it fires because nothing is broken. Every other
source is silent. A test therefore says what it is about by naming the one or two
columns it overrides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from stormlab.config import C1Sources, load_config
from stormlab.contracts.features import FEATURE_ONEHOT_ORDER
from stormlab.graph.build import GraphBuildResult

FROST_ZONE_M = 0.8
MAX_DEPTH_M = 8.0
REQUIRED_COVER_M = 0.8
DIAMETER_M = 0.5
LENGTH_M = 50.0
NEUTRAL_MARGIN = 0.5
"""A margin comfortably on the satisfied side of every requirement."""

NEUTRAL_FLOW_CAPACITY_RATIO = 0.9
"""Loaded enough that no smaller standard diameter would carry the flow.

``oversized`` fires when ``D * ratio**(3/8)`` drops to or below the next size down;
at 0.5 m that needs a ratio under about 0.558, so 0.9 keeps the source quiet while
leaving ``m_capacity`` positive."""

NEUTRAL_NODE_DEPTH_M = 4.0
"""Manhole depth whose position in ``[frost + D, max depth]`` is neither shallow nor deep."""

IMPERVIOUS_CODE = FEATURE_ONEHOT_ORDER.index("urban_highly_impervious")
PERVIOUS_CODE = FEATURE_ONEHOT_ORDER.index("meadows")

NEUTRAL_IMPERV_PCT = 45.0
"""Imperviousness in the band where neither catchment premise speaks.

The C1 overlay infiltrates at or below 40 % and retains at or above 50 %, so a frame
that means to be silent about the catchment has to sit between the two — otherwise
every source test would carry a second, unasked-for assessment."""

PERVIOUS_IMPERV_PCT = 10.0
SEALED_IMPERV_PCT = 80.0
"""Two values well inside the two premises, so a test names its intent, not a boundary."""

FILE_ID = "f0"


def settings() -> C1Sources:
    """The ``c1.sources`` section of the real C1 overlay.

    Read from the shipped configuration rather than built by hand: every threshold a
    source compares against is part of the frozen protocol, and a test that invented
    its own would be testing a protocol nobody declared.
    """
    config = load_config(variant="C1", check_inputs=False)
    assert config.c1 is not None
    return config.c1.sources


def chain_graph(n_conduits: int) -> GraphBuildResult:
    """``C0 -> C1 -> ... -> C{n-1}``, in the stored orientation ``A[upstream, current] = 1``."""
    order = tuple(f"C{index}" for index in range(n_conduits))
    matrix = sp.lil_matrix((n_conduits, n_conduits), dtype="float32")
    for index in range(n_conduits - 1):
        matrix[index, index + 1] = 1.0
    adjacency = sp.csr_matrix(matrix)
    return GraphBuildResult(adjacency=adjacency, order=order, n_edges_raw=int(adjacency.nnz), n_surrogate_edges=0)


def isolated_graph(n_conduits: int) -> GraphBuildResult:
    """``n`` conduits and no edges, for a source that must have no neighbourhood."""
    order = tuple(f"C{index}" for index in range(n_conduits))
    adjacency = sp.csr_matrix((n_conduits, n_conduits), dtype="float32")
    return GraphBuildResult(adjacency=adjacency, order=order, n_edges_raw=0, n_surrogate_edges=0)


def frame(n_conduits: int, **overrides: object) -> pd.DataFrame:
    """A neutral F2/F3 frame for ``n`` conduits, indexed by conduit name.

    Args:
        n_conduits: how many rows.
        overrides: column name to a scalar (applied to every row) or to a sequence
            of length ``n_conduits``.

    Returns:
        A frame carrying the identity columns and every column any C1 source
        declares, on which only ``no_violation`` fires.
    """
    names = [f"C{index}" for index in range(n_conduits)]
    values: dict[str, object] = {
        "sample_id": [f"{FILE_ID}:{name}" for name in names],
        "file_id": [FILE_ID] * n_conduits,
        "conduit": names,
        # geometry as laid
        "diameter_m": DIAMETER_M,
        "length_m": LENGTH_M,
        "barrels": 1.0,
        "slope_signed": 0.005,
        "inlet_pipe_invert_m": [100.0 - index for index in range(n_conduits)],
        "outlet_pipe_invert_m": [99.0 - index for index in range(n_conduits)],
        "inlet_ground_m": [102.0 - index for index in range(n_conduits)],
        "outlet_ground_m": [101.0 - index for index in range(n_conduits)],
        "inlet_node_max_depth_m": NEUTRAL_NODE_DEPTH_M,
        "outlet_node_max_depth_m": NEUTRAL_NODE_DEPTH_M,
        # requirements in force
        "frost_zone_m": FROST_ZONE_M,
        "max_depth_m": MAX_DEPTH_M,
        "required_cover_m": REQUIRED_COVER_M,
        "max_filling_ratio": 0.827,
        "min_slope_required": 0.001,
        "max_slope_allowed": 0.05,
        # simulated state
        "max_flow_cms": 1.0,
        "max_full_depth_ratio": 0.1,
        "hours_capacity_limited": 0.0,
        "flow_capacity_ratio": NEUTRAL_FLOW_CAPACITY_RATIO,
        # catchment
        "sbc_category_code": float(IMPERVIOUS_CODE),
        "subcatchment_imperv_pct": NEUTRAL_IMPERV_PCT,
        # margins
        "m_filling": NEUTRAL_MARGIN,
        "m_velocity_min": NEUTRAL_MARGIN,
        "m_velocity_max": NEUTRAL_MARGIN,
        "m_slope_min": NEUTRAL_MARGIN,
        "m_slope_max": NEUTRAL_MARGIN,
        "m_capacity": 1.0 - NEUTRAL_FLOW_CAPACITY_RATIO,
        "m_cover_inlet": NEUTRAL_MARGIN,
        "m_cover_outlet": NEUTRAL_MARGIN,
        "m_depth_inlet": NEUTRAL_MARGIN,
        "m_depth_outlet": NEUTRAL_MARGIN,
        # masks
        "slope_nonpositive": 0,
        "inlet_ground_missing": 0,
        "outlet_ground_missing": 0,
        "inlet_node_depth_missing": 0,
        "outlet_node_depth_missing": 0,
        "min_slope_missing": 0,
        "slope_requirement_missing": 0,
        "sbc_missing": 0,
        "subcatchment_missing": 0,
        "range_contradictory": 0,
    }
    values.update(overrides)
    table = pd.DataFrame(
        {
            name: (value if isinstance(value, (list, tuple, np.ndarray, pd.Series)) else [value] * n_conduits)
            for name, value in values.items()
        },
        index=pd.Index(names, name="conduit_name"),
    )
    return table


def sample(index: int = 0) -> str:
    """The ``sample_id`` of conduit ``C{index}`` of a fixture frame."""
    return f"{FILE_ID}:C{index}"


__all__ = [
    "DIAMETER_M",
    "FILE_ID",
    "FROST_ZONE_M",
    "IMPERVIOUS_CODE",
    "LENGTH_M",
    "MAX_DEPTH_M",
    "NEUTRAL_FLOW_CAPACITY_RATIO",
    "NEUTRAL_IMPERV_PCT",
    "NEUTRAL_MARGIN",
    "NEUTRAL_NODE_DEPTH_M",
    "PERVIOUS_CODE",
    "PERVIOUS_IMPERV_PCT",
    "SEALED_IMPERV_PCT",
    "REQUIRED_COVER_M",
    "chain_graph",
    "frame",
    "isolated_graph",
    "sample",
    "settings",
]
