"""One JSON record per pipeline step, describing exactly how it ran.

A record pins the commit and working-tree cleanliness, the seeds, the sha256 of
every input and output artifact, and the versions of the libraries whose numeric
behaviour can move results. Records are written once per ``(run_id, step)``:
re-running a step over an existing record is refused rather than overwriting it,
so a run directory can never mix results from two different executions.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Mapping, Sequence

TRACKED_LIBRARIES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "tensorflow",
    "keras",
    "scikit-learn",
    "xgboost",
    "pyswmm",
    "swmmio",
    "swmm-toolkit",
)
"""Distributions whose version can move a number this pipeline reports.

``swmm-toolkit`` is the compiled SWMM engine ``pyswmm`` runs; it is what actually
integrates the hydraulics, so a record naming the wrapper and not the solver
pinned the interface and left the numerics unstated (plan16 §2 P2). Adding a name
changes only records written from here on -- frozen ones keep the set they were
written with, and :meth:`ProvenanceRecord.read` accepts either."""

PIPELINE_INPUT_PATHS: tuple[str, ...] = (
    "stormlab",
    "scripts",
    "conftest.py",
    "Makefile",
    "results/variants.yaml",
    "models",
    "app",
)
"""What "the worktree is dirty" is a statement about: the pipeline's **inputs**.

A run writes as it goes — ablation tables, verdicts, per-seed artifacts under
``results/`` — so a whole-tree ``git status`` turns green into red partway through
the chain it is describing. Every step after the first then records
``git_dirty: true`` because of files the run itself had just produced, and
``make update-golden`` refused to freeze a variant on the grounds that the
variant's own results existed. Neither is a statement about reproducibility.

What a reader of a provenance record needs to know is whether the **code,
configuration, models and manifest** the run was executed from match the commit it
names — because those are what another checkout would have to reproduce. Outputs
are not inputs, so they are outside the scope, and the scope is written into every
record (:attr:`ProvenanceRecord.git_dirty_scope`) so the flag cannot be read as
meaning something wider than it does."""

_HASH_CHUNK_BYTES = 1024 * 1024


class ProvenanceError(RuntimeError):
    """Raised when a provenance record cannot be built or written."""


def sha256_file(path: Path | str) -> str:
    """Hex sha256 digest of a file, read in chunks.

    Raises:
        ProvenanceError: if the path is not an existing file.
    """
    candidate = Path(path)
    if not candidate.is_file():
        raise ProvenanceError(f"Cannot hash {candidate}: not an existing file")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_versions(names: tuple[str, ...] = TRACKED_LIBRARIES) -> dict[str, str | None]:
    """Installed version of each tracked distribution; ``None`` when absent."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def worktree_status(repo: Path, *, paths: Sequence[str] = PIPELINE_INPUT_PATHS) -> tuple[str, tuple[str, ...]]:
    """``(commit_sha, dirty entries)`` for ``repo``, restricted to ``paths``.

    The entries are ``git status --porcelain`` lines, so a caller refusing to act
    on a dirty tree can name the files rather than only assert that some exist.
    Untracked files count: a file git does not track is a file the next checkout
    will not have, which is the same reproducibility problem as an uncommitted
    edit. A pathspec that matches nothing is not an error — ``app/`` is absent from
    some checkouts, and its absence is not a dirty worktree.

    Raises:
        ProvenanceError: if git is unavailable or ``repo`` is not a repository.
    """
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal", "--", *paths],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProvenanceError(f"Cannot read git state of {repo}: {error}") from error
    return commit, tuple(line.strip() for line in status.splitlines() if line.strip())


def git_state(repo: Path, *, paths: Sequence[str] = PIPELINE_INPUT_PATHS) -> tuple[str, bool]:
    """Return ``(commit_sha, is_dirty)`` for the repository at ``repo``.

    ``is_dirty`` is about :data:`PIPELINE_INPUT_PATHS` unless told otherwise; see
    that constant for why the pipeline's own outputs are outside the question.

    Raises:
        ProvenanceError: if git is unavailable or ``repo`` is not a repository.
    """
    commit, entries = worktree_status(repo, paths=paths)
    return commit, bool(entries)


@dataclass(frozen=True)
class ProvenanceRecord:
    """Everything needed to explain one step of one run."""

    step: str
    run_id: str
    git_commit: str
    git_dirty: bool
    seeds: dict[str, int]
    inputs: dict[str, str]
    outputs: dict[str, str]
    libraries: dict[str, str | None]
    python_version: str
    platform: str
    created_at: str
    extra: dict[str, object] = field(default_factory=dict)
    git_dirty_scope: list[str] = field(default_factory=list)
    """The paths :attr:`git_dirty` was measured over.

    Written down because the flag is meaningless without it. ``git_dirty: false``
    against the whole tree and against the pipeline's inputs are two different
    claims, and a record that only carried the boolean would silently change
    meaning the day the scope moved — which is exactly what happened here. Empty
    on records written before the scope existed, where it meant the whole tree."""
    variant: str | None = None
    """Which frozen result set of path B this step belongs to.

    A first-class field rather than one more entry in ``extra``: it is the answer
    to "which set of results is this", which every reader of a run directory has
    to ask, and burying it among the per-step facts would make it findable only by
    somebody who already knew it was there. Optional, because records written
    before variants existed carry none and are still readable."""

    def __post_init__(self) -> None:
        if not self.step or "/" in self.step or "\\" in self.step:
            raise ProvenanceError(f"Step name {self.step!r} must be a non-empty path-free identifier")
        if not self.run_id or "/" in self.run_id or "\\" in self.run_id:
            raise ProvenanceError(f"Run id {self.run_id!r} must be a non-empty path-free identifier")

    @classmethod
    def capture(
        cls,
        *,
        step: str,
        run_id: str,
        seeds: Mapping[str, int],
        inputs: Mapping[str, Path | str] | None = None,
        outputs: Mapping[str, Path | str] | None = None,
        repo: Path | None = None,
        extra: Mapping[str, object] | None = None,
        variant: str | None = None,
        dirty_scope: Sequence[str] = PIPELINE_INPUT_PATHS,
    ) -> "ProvenanceRecord":
        """Build a record, hashing every listed input and output file now.

        ``dirty_scope`` is the paths ``git_dirty`` is measured over and is stored
        beside it; see :data:`PIPELINE_INPUT_PATHS`.
        """
        from stormlab.config import repo_root

        repository = Path(repo) if repo is not None else repo_root()
        commit, dirty = git_state(repository, paths=dirty_scope)
        return cls(
            step=step,
            run_id=run_id,
            git_commit=commit,
            git_dirty=dirty,
            git_dirty_scope=list(dirty_scope),
            seeds=dict(seeds),
            inputs={name: sha256_file(path) for name, path in (inputs or {}).items()},
            outputs={name: sha256_file(path) for name, path in (outputs or {}).items()},
            libraries=library_versions(),
            python_version=platform.python_version(),
            platform=platform.platform(),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            extra=dict(extra or {}),
            variant=variant,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialise to a JSON-safe mapping."""
        return asdict(self)

    def write(self, root: Path | str) -> Path:
        """Write the record to ``<root>/<run_id>/<step>.json``.

        Raises:
            ProvenanceError: if a record for this step already exists.
        """
        target = Path(root) / self.run_id / f"{self.step}.json"
        if target.exists():
            raise ProvenanceError(
                f"Provenance for step {self.step!r} of run {self.run_id!r} already exists at {target}; "
                "provenance records are never overwritten — use a new run id"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read(cls, path: Path | str) -> "ProvenanceRecord":
        """Load a record previously written by :meth:`write`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)
