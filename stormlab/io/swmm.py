"""Running the SWMM engine on a working copy, with a hard check of the report.

``pyswmm`` reports failures through the report file rather than through
exceptions, so a run is only accepted once the ``.rpt`` exists, shows the
analysis finished, contains a result section and carries no ``ERROR`` line.
The guard on the input path exists because SWMM writes its outputs next to the
input: simulating a tracked file would modify the repository.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

logger = logging.getLogger(__name__)

FORBIDDEN_SUBTREES: tuple[str, ...] = ("models", "app")
"""Repository subtrees holding most tracked INP files; never simulated in place.

A named subtree is a fast rejection, not the guarantee: tracked INP files also
live outside these two (the ``stormwater analysis backup`` tree), so the real
check is :func:`is_tracked_by_git`.
"""

RPT_COMPLETION_MARKER = "Analysis ended on:"
RPT_RESULT_MARKERS: tuple[str, ...] = ("Summary", "Continuity")
RPT_ERROR_PATTERN = re.compile(r"^\s*ERROR.*$", re.MULTILINE)

SMALL_SLOPE_WARNING = "Slope is too small"

SMALL_SLOPE_WARNING_LOGGER = "stormlab.hydraulics.formulas"
"""Logger that emits the per-row small-slope warning (``calc_flow`` in that module)."""

_RPT_EXCERPT_CHARS = 2000


class SwmmRunError(RuntimeError):
    """Raised when a SWMM run did not produce a usable report."""


class UnsafeSimulationPathError(SwmmRunError):
    """Raised when a simulation would write next to a tracked input file."""


@dataclass(frozen=True)
class SwmmRun:
    """Artifacts of one successful SWMM run."""

    inp: Path
    rpt: Path
    out: Path
    duration_s: float


class SuppressedWarnings:
    """How many warnings a suppression block swallowed.

    A batch of 756 files silences the per-row "Slope is too small" warning to
    keep the log readable, but "how often did the dataset trip the minimum-slope
    rule" is a fact about the dataset, not log noise. Counting it is what keeps
    the suppression from being a silent degradation."""

    def __init__(self) -> None:
        self.count = 0


class _SmallSlopeFilter(logging.Filter):
    """Drops the per-row 'Slope is too small' warning emitted during batch runs."""

    def __init__(self, counter: SuppressedWarnings) -> None:
        super().__init__()
        self._counter = counter

    def filter(self, record: logging.LogRecord) -> bool:
        if SMALL_SLOPE_WARNING in record.getMessage():
            self._counter.count += 1
            return False
        return True


@contextmanager
def small_slope_warnings_suppressed(logger_name: str = SMALL_SLOPE_WARNING_LOGGER) -> Iterator[SuppressedWarnings]:
    """Silence the 'Slope is too small' warning for the duration of a batch.

    The default is the logger that *emits* the warning, and that is load-bearing:
    a ``logging.Filter`` installed on a logger only sees the records that logger
    creates itself, never the ones propagating up from its children. Defaulting to
    the root logger — as this used to — therefore filtered nothing at all while
    looking like it worked.

    Scoped to one logger and removed on exit, so interactive use of the same code
    keeps reporting the warning; the filter is never applied to the root logger,
    where it would also hide the warning in the places it is diagnostic.

    Yields:
        A live counter of the warnings dropped, readable after the block.
    """
    target = logging.getLogger(logger_name)
    counter = SuppressedWarnings()
    log_filter = _SmallSlopeFilter(counter)
    target.addFilter(log_filter)
    try:
        yield counter
    finally:
        target.removeFilter(log_filter)


def default_allowed_roots(work_dir: Path | str | None = None) -> tuple[Path, ...]:
    """Directories a simulation may write into: the run workdir and the system temp dir."""
    roots = [Path(tempfile.gettempdir()).resolve()]
    if work_dir is not None:
        roots.append(Path(work_dir).resolve())
    return tuple(roots)


def is_tracked_by_git(path: Path | str) -> bool:
    """Whether ``path`` is a file git has under version control.

    Answered by ``git ls-files --error-unmatch`` run from the repository root, so
    it holds for every tracked file rather than for a list of directories someone
    has to remember to extend. Anything that is not a definite "yes" — the path is
    outside the repository, git is not installed, the checkout is not a work tree
    — is reported as untracked and logged: the caller's other checks still have to
    pass, so a false negative cannot let a tracked file through on its own.
    """
    from stormlab.config import repo_root

    root = repo_root()
    if not (root / ".git").exists():
        logger.warning("%s is not a git work tree; treating %s as untracked", root, path)
        return False
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(path)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        logger.warning("git is unavailable (%s); treating %s as untracked", error, path)
        return False
    return result.returncode == 0


def assert_safe_simulation_path(inp: Path | str, *, allowed_roots: Sequence[Path] | None = None) -> Path:
    """Check that simulating ``inp`` cannot touch tracked files.

    Three independent checks, all of which must pass. ``allowed_roots`` widens the
    *location* rule only: it can never authorise a tracked file, because a caller
    that passes the repository as an allowed root would otherwise re-enable
    exactly the accident this function exists to prevent.

    Raises:
        UnsafeSimulationPathError: if the file is tracked by git, lives under a
            tracked subtree, or lies outside every allowed root.
    """
    from stormlab.config import repo_root

    candidate = Path(inp).resolve()
    root = repo_root()
    for subtree in FORBIDDEN_SUBTREES:
        if _is_within(candidate, root / subtree):
            raise UnsafeSimulationPathError(
                f"Refusing to run SWMM on {candidate}: it lives under the tracked subtree {root / subtree}. "
                "Copy the INP into the working directory first."
            )

    if is_tracked_by_git(candidate):
        raise UnsafeSimulationPathError(
            f"Refusing to run SWMM on {candidate}: git tracks it, and SWMM writes .rpt/.out/.bak next to its input. "
            "Copy the INP into the working directory first."
        )

    roots = tuple(allowed_roots) if allowed_roots is not None else default_allowed_roots()
    resolved_roots = tuple(Path(item).resolve() for item in roots)
    if not any(_is_within(candidate, allowed) for allowed in resolved_roots):
        raise UnsafeSimulationPathError(
            f"Refusing to run SWMM on {candidate}: it is outside the allowed roots {[str(item) for item in resolved_roots]}"
        )
    return candidate


def run_swmm(inp_copy: Path | str, *, allowed_roots: Sequence[Path] | None = None) -> SwmmRun:
    """Simulate a working copy of an INP file and validate the resulting report.

    Args:
        inp_copy: an INP file inside the working directory or a temp directory.
        allowed_roots: directories the file is allowed to live in; defaults to
            the system temp directory.

    Raises:
        UnsafeSimulationPathError: if ``inp_copy`` is not in a safe location.
        SwmmRunError: if the engine fails or the report is missing/incomplete/
            contains an error line. The message carries an excerpt of the report.
    """
    from pyswmm import Simulation

    path = assert_safe_simulation_path(inp_copy, allowed_roots=allowed_roots)
    if not path.is_file():
        raise SwmmRunError(f"INP file {path} does not exist")

    started = time.perf_counter()
    try:
        with Simulation(str(path)) as simulation:
            for _ in simulation:
                pass
    except Exception as error:  # pyswmm raises engine-specific errors
        raise SwmmRunError(f"SWMM engine failed on {path}: {type(error).__name__}: {error}") from error
    duration = time.perf_counter() - started

    rpt = path.with_suffix(".rpt")
    out = path.with_suffix(".out")
    validate_rpt(rpt)
    if not out.is_file():
        raise SwmmRunError(f"SWMM produced no binary output for {path}; expected {out}")
    return SwmmRun(inp=path, rpt=rpt, out=out, duration_s=duration)


def validate_rpt(rpt: Path | str) -> None:
    """Reject a report that is missing, truncated, or carries an ``ERROR`` line.

    Raises:
        SwmmRunError: with an excerpt of the report explaining what is wrong.
    """
    report = Path(rpt)
    if not report.is_file():
        raise SwmmRunError(f"SWMM report {report} was not produced")

    text = report.read_text(encoding="utf-8", errors="replace")
    errors = RPT_ERROR_PATTERN.findall(text)
    if errors:
        raise SwmmRunError(f"SWMM report {report} contains {len(errors)} error line(s):\n" + "\n".join(errors[:10]))
    if RPT_COMPLETION_MARKER not in text:
        raise SwmmRunError(
            f"SWMM report {report} does not contain {RPT_COMPLETION_MARKER!r}; the run did not finish.\n{_excerpt(text)}"
        )
    if not any(marker in text for marker in RPT_RESULT_MARKERS):
        raise SwmmRunError(
            f"SWMM report {report} contains no result section (looked for {list(RPT_RESULT_MARKERS)}).\n{_excerpt(text)}"
        )


def _excerpt(text: str) -> str:
    tail = text[-_RPT_EXCERPT_CHARS:]
    return f"--- report excerpt ---\n{tail}"


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True
