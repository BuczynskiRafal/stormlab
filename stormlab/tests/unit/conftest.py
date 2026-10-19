"""Frozen SWMM frames for the unit layer.

The unit tests never run SWMM: the engine writes ``.rpt``/``.out`` next to its
input, which is how tracked files under ``app/`` used to change during a test
run. Instead the frames of ``app/sa/core/tests/test_file.inp`` are frozen as
parquet by ``scripts/regenerate_test_fixtures.py`` and loaded here.

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
        pytest.skip(f"frozen fixture {path} is missing; regenerate it with scripts/regenerate_test_fixtures.py")
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
    "no land cover" branch: an all-zero one-hot block. This is a synthetic
    untagged network, *not* the Wroclaw one — Wroclaw tags all 37 of its
    subcatchments and lacks link tags instead (see
    ``test_subcatchments.TestWroclawTagProfile``).
    """
    frame = _load("tags.parquet")
    return frame.set_index(TAG_INDEX_COLUMNS)["Tag"]


@pytest.fixture
def synthetic_features():
    """Factory for a contract-shaped feature frame whose labels are linearly separable.

    The tabular models need a frame that satisfies the 33-column contract exactly
    — order, dtypes, no missing values — and a target they can actually learn, or
    a failing assertion says nothing about the code under test. Each class gets
    its own corner of the feature space: the flags encode the class index in
    binary and the normalised columns hold a class-specific level, so a handful of
    trees or a three-epoch network separates them cleanly.

    The frame also carries the identity columns of ``features.parquet``
    (``file_id``, ``conduit``, ``sample_id``), so the same fixture feeds the model
    tests and the experiment test.
    """
    import numpy as np

    from stormlab.contracts.features import FEATURE_COLUMNS, FLAG_COLUMNS, FLOAT_COLUMNS

    def build(classes, *, rows_per_class: int = 12, n_files: int = 6) -> pd.DataFrame:
        labels = [name for name in classes for _ in range(rows_per_class)]
        n_rows = len(labels)
        index = {name: position for position, name in enumerate(classes)}

        columns = {}
        for offset, column in enumerate(FLAG_COLUMNS):
            columns[column] = np.array([(index[label] >> (offset % 4)) & 1 for label in labels], dtype="int8")
        for offset, column in enumerate(FLOAT_COLUMNS):
            columns[column] = np.array(
                [((index[label] + 1) / (len(classes) + 1) + 0.01 * ((row + offset) % 3)) for row, label in enumerate(labels)],
                dtype="float64",
            )

        frame = pd.DataFrame(columns)[list(FEATURE_COLUMNS)]
        frame["label"] = labels
        file_ids = [f"file{row % n_files:02d}" for row in range(n_rows)]
        frame["file_id"] = file_ids
        frame["file"] = [f"{name}.inp" for name in file_ids]
        frame["conduit"] = [f"C{row}" for row in range(n_rows)]
        frame["sample_id"] = [f"{file_id}:C{row}" for row, file_id in enumerate(file_ids)]
        return frame

    return build
