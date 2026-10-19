"""``run_swmm`` only reports success when both artifacts of a run exist.

The report is validated line by line, but the binary ``.out`` is what every later
step reads through swmmio; a run whose engine wrote a clean report and no ``.out``
would otherwise be reported as a success and fail much later, far from its cause.
The engine itself is replaced at the import boundary — a stub ``pyswmm`` module —
so the check can be driven without a real simulation.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from stormlab.io.swmm import RPT_COMPLETION_MARKER, SwmmRunError, run_swmm

FINISHED_REPORT = f"""  Flow Routing Continuity
  Analysis begun on: Wed Aug 27 09:00:00 2025
  {RPT_COMPLETION_MARKER} Wed Aug 27 09:00:01 2025
"""


def _install_engine(monkeypatch: pytest.MonkeyPatch, *, writes: dict[str, str]) -> None:
    """Replace ``pyswmm`` with a Simulation that writes exactly ``writes`` next to the INP."""

    class Simulation:
        def __init__(self, path: str) -> None:
            self._inp = Path(path)

        def __enter__(self) -> "Simulation":
            return self

        def __exit__(self, *exc_info: object) -> None:
            for suffix, content in writes.items():
                self._inp.with_suffix(suffix).write_text(content, encoding="utf-8")

        def __iter__(self):
            return iter(())

    monkeypatch.setitem(sys.modules, "pyswmm", types.ModuleType("pyswmm"))
    monkeypatch.setattr(sys.modules["pyswmm"], "Simulation", Simulation, raising=False)


class TestBinaryOutput:
    """The ``.out`` file is not optional: it is the input of every downstream reader."""

    def test_a_run_without_a_binary_output_names_the_file_it_expected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange: an engine that finishes and reports cleanly, but writes no .out.
        inp = tmp_path / "network.inp"
        inp.write_text("[TITLE]\nstub network\n", encoding="utf-8")
        _install_engine(monkeypatch, writes={".rpt": FINISHED_REPORT})

        # Act / Assert
        with pytest.raises(SwmmRunError, match="produced no binary output") as error:
            run_swmm(inp)

        message = str(error.value)
        assert str(inp) in message and str(inp.with_suffix(".out")) in message

    def test_a_run_that_wrote_both_artifacts_is_reported_as_a_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The same stub, one file richer, must pass — otherwise the test above proves nothing."""
        inp = tmp_path / "network.inp"
        inp.write_text("[TITLE]\nstub network\n", encoding="utf-8")
        _install_engine(monkeypatch, writes={".rpt": FINISHED_REPORT, ".out": "binary"})

        run = run_swmm(inp)

        assert run.out.is_file()
        assert run.duration_s >= 0.0
