"""Synthetic chains and neutral F2 rows shared by the motif and penalty tests.

A helper module rather than a fixture in one of the two test files: the catalogue
and the penalty are tested from opposite sides — "is the pattern detected" and
"is the pattern charged" — and they have to agree about what a neutral chain is,
or a motif could pass one and fail the other for reasons of fixture rather than
of code. Same arrangement as ``physical_fixtures.py``.
"""

from __future__ import annotations

import pandas as pd

NEUTRAL_FLOW_CMS = 1.0
NEUTRAL_FILLING = 0.1
NEUTRAL_MAX_FILLING_RATIO = 0.827
NEUTRAL_MIN_SLOPE = 0.001
NEUTRAL_LENGTH_M = 50.0
NEUTRAL_MAX_DEPTH_M = 8.0
"""The admissible burial of the pipe invert; M5 reads it as the floor a regrade may reach."""


def chain(n_conduits: int) -> pd.DataFrame:
    """A straight chain ``C0 -> C1 -> ... -> C{n-1}`` through nodes ``J0..J{n}``."""
    return pd.DataFrame(
        {
            "Name": [f"C{index}" for index in range(n_conduits)],
            "InletNode": [f"J{index}" for index in range(n_conduits)],
            "OutletNode": [f"J{index + 1}" for index in range(n_conduits)],
        }
    )


def features_for(names: list[str], **overrides: list[float]) -> pd.DataFrame:
    """Neutral F2 rows for a chain: nothing fires unless a column is overridden.

    The neutral values are chosen so that every precondition is **false**: flow
    constant, filling well under the requirement, and a required regrading that
    fits in the available fall many times over. A motif that fired on these would
    fire on any chain.
    """
    n = len(names)
    frame = pd.DataFrame(
        {
            "max_flow_cms": [NEUTRAL_FLOW_CMS] * n,
            "max_full_depth_ratio": [NEUTRAL_FILLING] * n,
            "max_filling_ratio": [NEUTRAL_MAX_FILLING_RATIO] * n,
            "min_slope_required": [NEUTRAL_MIN_SLOPE] * n,
            "length_m": [NEUTRAL_LENGTH_M] * n,
            "inlet_pipe_invert_m": [100.0 - index for index in range(n)],
            "outlet_pipe_invert_m": [99.0 - index for index in range(n)],
            "outlet_ground_m": [101.0 - index for index in range(n)],
            "max_depth_m": [NEUTRAL_MAX_DEPTH_M] * n,
        },
        index=pd.Index(names, name="Name"),
    )
    for column, values in overrides.items():
        frame[column] = values
    return frame
