"""Root-level pytest plugin: registers the ``--run-parity`` switch for the whole repo.

``testpaths`` in ``pyproject.toml`` spans two suites (``app/sa/core/tests`` and
``stormlab/tests``), and ``addopts`` carries a repository-wide ``-m 'not parity'``.
An option registered only in ``stormlab/tests/conftest.py`` is not visible when
pytest is started from the repository root with the full ``testpaths`` set, so
``pytest --run-parity`` failed with ``UsageError: unrecognized arguments``. The
registration therefore lives here, at the level the option applies to; the
collection-time deselection stays in the stormlab conftest, next to the layer it
guards.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    from stormlab.tests.conftest import RUN_PARITY_OPTION, parity_exclusion_removed
except ImportError:
    import re

    RUN_PARITY_OPTION = "--run-parity"

    def parity_exclusion_removed(markexpr: str) -> str:
        parts = [part.strip() for part in re.split(r"\band\b", markexpr) if part.strip()]
        kept = [part for part in parts if part.replace("(", "").replace(")", "").strip() != "not parity"]
        return " and ".join(kept)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the opt-in switch for the parity layer."""
    parser.addoption(
        RUN_PARITY_OPTION,
        action="store_true",
        default=False,
        help="run the parity tests that compare against the frozen golden extraction",
    )
    parser.addoption(
        NO_SKIPS_OPTION,
        action="store_true",
        default=False,
        dest="no_skips",
        help="fail on any skipped test instead of accepting it as a pass",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Let ``--run-parity`` win over the ``not parity`` expression from ``addopts``."""
    if not config.getoption(RUN_PARITY_OPTION, default=False):
        return
    config.option.markexpr = parity_exclusion_removed(config.option.markexpr or "")


NO_SKIPS_OPTION = "--no-skips"
"""Turn every skipped test into a failure, for the layer of CI that must not skip.

A skip is invisible in a green run: the parity layer skipping itself for want of
``--run-parity``, or a results-dependent test skipping because the run directory
it reads is not there, both report success. The verification layers of the
pipeline are exactly the ones where that is unacceptable — a CI job that "passed"
while never comparing anything against the golden extraction is a CI job that
proves nothing. Deselection is untouched: ``-m 'not slow'`` says which tests this
invocation is *about*, and a test outside that set was never promised."""

_SKIPPED_UNDER_NO_SKIPS: list[str] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    """Under ``--no-skips``, report a skip as a failure naming its reason."""
    outcome = yield
    if not item.config.getoption("no_skips", default=False):
        return
    report: pytest.TestReport = outcome.get_result()
    if report.outcome != "skipped" or report.when not in ("setup", "call"):
        return
    # longrepr of a skip is (path, lineno, "Skipped: <reason>"); an xfail arrives
    # here as skipped too, and it is a declared expectation rather than a silence.
    if hasattr(report, "wasxfail"):
        return
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    report.outcome = "failed"
    report.longrepr = f"{NO_SKIPS_OPTION} was given, but this test skipped: {reason}"
    _SKIPPED_UNDER_NO_SKIPS.append(f"{report.nodeid}: {reason}")


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """List every test that skipped under ``--no-skips``, so the fix is one read away."""
    if not config.getoption("no_skips", default=False) or not _SKIPPED_UNDER_NO_SKIPS:
        return
    terminalreporter.section(f"tests that skipped under {NO_SKIPS_OPTION}")
    for line in _SKIPPED_UNDER_NO_SKIPS:
        terminalreporter.line(line)
