"""What a killed worker leaves behind, and how it is cleared without waiting for a restart.

An analysis runs inside the request. When gunicorn kills the worker (timeout, out of
memory) the session stays ``processing`` and the temporary ``.inp`` with the engine's
``.rpt`` and ``.out`` stay on disk. The master starts a new worker, but nothing else
notices until the container restarts.

* Every temporary file of an analysis lives in one directory of its own,
  :func:`analysis_temp_dir`, so cleaning up removes exactly what this application wrote.
* :func:`reap_stale_work` marks sessions stuck for :data:`STALE_MINUTES` as failed (one
  ``UPDATE`` over the ``(status, updated_at)`` index) and removes files of that age. The
  analysis form and the history page call it, and so does ``manage.py
  reap_stale_sessions``, which the container also runs at start.

No live analysis is touched: one cannot outlast the gunicorn timeout (180 s), far below
:data:`STALE_MINUTES`.
"""

from __future__ import annotations

import logging
import os
import stat
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from django.utils import timezone

from .models import CalculationSession

logger = logging.getLogger(__name__)

TEMP_DIR_NAME = "stormlab-analyses"
"""Directory under the system temporary directory that holds every file of a running analysis."""

SIBLING_SUFFIXES = (".rpt", ".out", ".bak")
"""What the engine and swmmio write next to the ``.inp`` they are given."""

STALE_MINUTES = 30
STALE_MESSAGE = "The analysis did not finish because the server stopped while it was running. Please run it again."


def analysis_temp_dir() -> Path:
    """The private directory for analysis files, created on first use (mode 0700).

    Raises:
        OSError: if the path exists and is not a directory of this process's user, e.g.
            a symbolic link planted in a shared ``/tmp``.
    """
    path = Path(tempfile.gettempdir()) / TEMP_DIR_NAME
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
        raise OSError(f"{path} is not a private directory of this user")
    return path


def new_analysis_file() -> str:
    """Create an empty ``.inp`` in :func:`analysis_temp_dir` and return its path."""
    handle = tempfile.NamedTemporaryFile(delete=False, dir=analysis_temp_dir(), prefix="analysis-", suffix=".inp")
    handle.close()
    return handle.name


def remove_analysis_files(inp_path: str) -> None:
    """Remove the ``.inp`` and the files the engine wrote beside it; missing ones are fine."""
    base, _ = os.path.splitext(inp_path)
    for candidate in (inp_path, *(base + suffix for suffix in SIBLING_SUFFIXES)):
        try:
            os.unlink(candidate)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Could not remove analysis file %s", candidate, exc_info=True)


def remove_stale_analysis_files(older_than: timedelta | None) -> int:
    """Remove files in :func:`analysis_temp_dir` last modified longer ago than *older_than*.

    ``None`` removes every file, which is safe only while no analysis runs (container
    start). Only plain files directly in the directory are considered. Returns the count.
    """
    try:
        directory = analysis_temp_dir()
    except OSError:
        logger.warning("Analysis directory unavailable; nothing removed", exc_info=True)
        return 0
    cutoff = None if older_than is None else time.time() - older_than.total_seconds()
    removed = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if cutoff is not None and entry.stat(follow_symlinks=False).st_mtime >= cutoff:
                    continue
                os.unlink(entry.path)
                removed += 1
            except FileNotFoundError:
                continue
    return removed


def reap_stale_sessions(minutes: int = STALE_MINUTES, *, now: datetime | None = None) -> int:
    """Mark sessions ``processing`` for longer than *minutes* as failed; return how many."""
    now = now or timezone.now()
    stale = CalculationSession.objects.filter(status="processing", updated_at__lt=now - timedelta(minutes=minutes))
    return stale.update(status="failed", error_message=STALE_MESSAGE, updated_at=now)


def reap_stale_work(minutes: int = STALE_MINUTES) -> None:
    """Clear what dead workers left: stuck sessions and their files. Never raises."""
    try:
        sessions = reap_stale_sessions(minutes)
        files = remove_stale_analysis_files(timedelta(minutes=minutes))
    except Exception:
        logger.exception("Clearing stale analyses failed")
        return
    if sessions or files:
        logger.info("Cleared %d stale session(s) and %d stale analysis file(s)", sessions, files)
