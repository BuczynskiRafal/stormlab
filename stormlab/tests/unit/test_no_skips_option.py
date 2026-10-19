"""``--no-skips`` turns a skip into a failure, and leaves deselection alone.

The option is the only thing standing between "the verification layer passed" and
"the verification layer ran", so it is exercised against a real pytest process
rather than by calling the hook directly: what matters is the exit status a CI
step sees.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

SUITE = """
import pytest

def test_passes():
    assert True

@pytest.mark.skip(reason="a reason nobody reads")
def test_skips():
    assert False

@pytest.mark.slow
def test_is_slow():
    raise AssertionError("a deselected test must never run")
"""


@pytest.fixture()
def suite(tmp_path: Path) -> Path:
    """A three-test file outside the repository tree, with its own minimal config."""
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: slow test\n", encoding="utf-8")
    path = tmp_path / "test_sample.py"
    path.write_text(SUITE, encoding="utf-8")
    return path


def _pytest(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run pytest with the repository's root conftest loaded as a plugin.

    ``-p conftest`` rather than a copied file: the plugin under test is the real
    ``conftest.py`` of this repository, imported from ``PYTHONPATH``.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "conftest", "-p", "no:cacheprovider", "--no-cov", "-q", *args],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )


class TestNoSkips:
    def test_a_skip_is_a_pass_without_the_option(self, suite: Path) -> None:
        result = _pytest(suite.name, "-m", "not slow", cwd=suite.parent)

        assert result.returncode == 0, result.stdout
        assert "1 skipped" in result.stdout

    def test_a_skip_fails_the_run_with_the_option(self, suite: Path) -> None:
        result = _pytest(suite.name, "--no-skips", "-m", "not slow", cwd=suite.parent)

        assert result.returncode != 0, result.stdout

    def test_the_failure_names_every_test_that_skipped_and_why(self, suite: Path) -> None:
        result = _pytest(suite.name, "--no-skips", "-m", "not slow", cwd=suite.parent)

        assert "tests that skipped under --no-skips" in result.stdout
        assert "test_sample.py::test_skips" in result.stdout
        assert "a reason nobody reads" in result.stdout

    def test_a_deselected_test_is_not_a_skipped_one(self, suite: Path) -> None:
        """``-m 'not slow'`` says which tests the run is about; the rest were never promised."""
        result = _pytest(suite.name, "--no-skips", "-m", "not slow", "-k", "not test_skips", cwd=suite.parent)

        assert result.returncode == 0, result.stdout
        assert "2 deselected" in result.stdout
        assert "skipped" not in result.stdout
