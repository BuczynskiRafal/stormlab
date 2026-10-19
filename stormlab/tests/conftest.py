"""Shared fixtures and marker handling for the stormlab suite.

Parity tests run SWMM against a frozen golden file and are meant to run only when
asked for. ``addopts`` deselects them with ``-m 'not parity'``; ``--run-parity``
removes exactly that conjunct again. Because pytest's ``-m`` is single-valued, a
command line that passes its own ``-m`` (for example ``-m 'not slow'``) replaces
the configured expression entirely and would silently pull the parity tests back
in, so the collection hook below skips them as a second line of defence.

The ``--run-parity`` option itself is registered in the repository-root
``conftest.py``: ``testpaths`` also covers ``app/`` and pytest only accepts an
option registered at or above the rootdir of the run. This module keeps the
option name and the marker-expression surgery so both live next to the layer
they guard.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pytest

RUN_PARITY_OPTION = "--run-parity"


def parity_exclusion_removed(markexpr: str) -> str:
    """Drop the ``not parity`` conjunct from a marker expression, keeping the rest.

    ``addopts`` carries ``-m 'not parity'`` so that an ordinary run never collects
    the parity layer. ``--run-parity`` has to undo exactly that one conjunct:
    clearing the whole expression would also re-enable whatever else the caller
    excluded.
    """
    parts = [part.strip() for part in re.split(r"\band\b", markexpr) if part.strip()]
    kept = [part for part in parts if part.replace("(", "").replace(")", "").strip() != "not parity"]
    return " and ".join(kept)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip parity-marked tests unless they were explicitly requested."""
    if config.getoption(RUN_PARITY_OPTION, default=False):
        return
    skip = pytest.mark.skip(reason=f"parity layer is opt-in; pass {RUN_PARITY_OPTION} to run it")
    for item in items:
        if "parity" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def repo_root_path() -> Path:
    """Absolute path of the repository root."""
    from stormlab.config import repo_root

    return repo_root()


@pytest.fixture(scope="session")
def test_inp(repo_root_path: Path) -> Path:
    """The small tracked network used as a simulation smoke test."""
    candidate = repo_root_path / "app" / "sa" / "core" / "tests" / "test_file.inp"
    if not candidate.is_file():
        pytest.skip(f"test network not available at {candidate}")
    return candidate


@pytest.fixture
def inp_working_copy(test_inp: Path, tmp_path: Path) -> Iterator[Path]:
    """A copy of the test network inside ``tmp_path``, safe to simulate."""
    from stormlab.io.workdir import copy_inp

    yield copy_inp(test_inp, tmp_path).path
