"""Lightweight parser for SWMM INP [OPTIONS] section values."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

MAX_SCANNED_LINES = 5000
"""[OPTIONS] is the first section of every SWMM file; a file that hides it deeper is not one we want."""


def read_flow_units(lines: Iterable[str]) -> str | None:
    """Return the FLOW_UNITS value from the [OPTIONS] section, or *None*.

    Scans *lines* (any iterable of strings — file, list, generator) for the
    ``[OPTIONS]`` header, then reads until the next section header (a line
    starting with ``[``).  Within [OPTIONS] it looks for a line whose first
    token (case-insensitive) is ``FLOW_UNITS`` and returns the second token
    upper-cased.  Everything after ``;`` on a line is treated as a comment. Only the
    first ``MAX_SCANNED_LINES`` lines are read, so a hostile file cannot make the caller
    buffer it whole.
    """
    in_options = False
    for raw_line in islice(lines, MAX_SCANNED_LINES):
        if "\x00" in raw_line:
            return None  # a NUL byte never occurs in a SWMM text file
        line = raw_line.split(";", maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            if line.upper() == "[OPTIONS]":
                in_options = True
                continue
            if in_options:
                # Reached the next section — stop.
                break
            continue
        if not in_options:
            continue
        tokens = line.split()
        if tokens[0].upper() == "FLOW_UNITS":
            return tokens[1].upper() if len(tokens) > 1 else None
    return None
