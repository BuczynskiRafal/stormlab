"""Frozen SWMM frames for the unit layer.

The unit tests never run SWMM: the engine writes ``.rpt``/``.out`` next to its
input, which would change tracked files under ``app/`` during a test run.
Instead the frames of ``app/sa/core/tests/test_file.inp`` are frozen as parquet
and loaded here.

The fixtures are function-scoped copies on purpose — the feature steps mutate the
frames they are handed, and a shared frame would leak state between tests.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"

TAG_INDEX_COLUMNS = ["ElementType", "Name"]


def _load(name: str) -> pd.DataFrame:
    path = FIXTURE_DIR / name
    if not path.is_file():
        pytest.skip(f"frozen fixture {path} is missing")
    return pd.read_parquet(path)


@pytest.fixture
def frozen_dfc() -> pd.DataFrame:
    """Conduit frame of the test network, as swmmio read it after a SWMM run."""
    return _load("dfc.parquet")


@pytest.fixture
def frozen_dfn() -> pd.DataFrame:
    """Node frame of the test network."""
    return _load("dfn.parquet")


@pytest.fixture
def frozen_dfs() -> pd.DataFrame:
    """Subcatchment frame of the test network."""
    return _load("dfs.parquet")


@pytest.fixture
def frozen_tags() -> pd.Series:
    """``[TAGS]`` of the test network, indexed by ``(ElementType, Name)``.

    The network tags nodes and links but no subcatchments, so it exercises the
    "no land cover" branch: an all-zero one-hot block. It is *not* the Wroclaw
    network, which tags all 37 of its subcatchments and lacks link tags instead
    (see ``test_subcatchments.TestWroclawTagProfile``).
    """
    frame = _load("tags.parquet")
    return frame.set_index(TAG_INDEX_COLUMNS)["Tag"]
