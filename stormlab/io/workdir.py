"""Working copies of INP files.

Running SWMM writes ``.rpt``/``.out``/``.bak`` next to the input file and
overwrites whatever is already there, so no tracked INP is ever simulated in
place. Every run copies its inputs into ``work/<run_id>/`` first, keeping the
original filename (labels, filename-encoded parameters and provenance all key on
it) and recording the sha256 of the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stormlab.provenance.record import sha256_file


class WorkdirError(RuntimeError):
    """Raised when a working copy cannot be prepared."""


@dataclass(frozen=True)
class InpCopy:
    """A working copy of an INP file and the hash of the file it came from."""

    source: Path
    path: Path
    sha256: str


FORBIDDEN_RUN_IDS: frozenset[str] = frozenset({".", ".."})
"""Run ids that name a directory instead of creating one."""


def prepare_workdir(work_dir: Path | str, run_id: str) -> Path:
    """Create and return ``<work_dir>/<run_id>``.

    The run id becomes a directory name and nothing else. A separator or a
    ``..`` would let the working directory escape ``work_dir``, and every later
    guarantee — that SWMM only ever writes inside the workdir, that a run's
    artifacts are removable as a unit — is a statement about that containment.
    The check is therefore made twice: on the string, and on the resolved path.

    Raises:
        WorkdirError: if ``run_id`` is empty, contains a path separator, is ``.``
            or ``..``, or the resulting directory would not sit directly inside
            ``work_dir``.
    """
    if not run_id or "/" in run_id or "\\" in run_id or run_id in FORBIDDEN_RUN_IDS:
        raise WorkdirError(f"Run id {run_id!r} must be a non-empty path-free identifier, and neither '.' nor '..'")

    parent = Path(work_dir).resolve()
    target = (parent / run_id).resolve()
    if target.parent != parent:
        raise WorkdirError(f"Run id {run_id!r} would place the working directory at {target}, outside {parent}")

    target.mkdir(parents=True, exist_ok=True)
    return target


def copy_inp(source: Path | str, destination_dir: Path | str, *, overwrite: bool = False) -> InpCopy:
    """Copy one INP into ``destination_dir`` under its original name.

    Args:
        source: the tracked INP file to copy.
        destination_dir: an existing or creatable working directory.
        overwrite: allow replacing an existing copy of the same name.

    Raises:
        WorkdirError: if the source is missing, is not a file, or a copy already
            exists and ``overwrite`` is off.
    """
    src = Path(source)
    if not src.is_file():
        raise WorkdirError(f"INP source {src} does not exist or is not a file")

    target_dir = Path(destination_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / src.name
    if target.resolve() == src.resolve():
        raise WorkdirError(f"Refusing to copy {src} onto itself; the working directory must differ from the source directory")
    if target.exists() and not overwrite:
        raise WorkdirError(f"Working copy {target} already exists; pass overwrite=True to replace it")

    digest = sha256_file(src)
    target.write_bytes(src.read_bytes())
    return InpCopy(source=src, path=target, sha256=digest)
