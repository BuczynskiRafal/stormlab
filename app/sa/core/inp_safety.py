"""Find SWMM input constructs that make the engine read or write other files.

An uploaded model must be self-contained: the engine opens any path named in it
(for example ``[FILES] SAVE HOTSTART "<path>"`` writes there). The scan follows the
way SWMM 5.2 reads an input file, so that what it accepts is what the engine sees:

* lines end at ``\\n`` only and are read in pieces of at most 1023 bytes, so a longer
  line would be split by the engine and is rejected here;
* a NUL byte ends the line, as in C; a ``;`` starts a comment even inside quotes; tokens are separated by space, tab
  and ``\\r``; a token that starts with ``"`` runs to the next ``"``;
* section headers and keywords are case-insensitive and match by prefix (``[FILE``
  matches ``[FILES]``), as the engine's keyword lookup does.

``[BACKDROP] FILE`` is allowed: the engine skips that section (it is map data).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

MAX_LINE_BYTES = 1023
"""Longest line the SWMM engine reads in one piece (its line buffer is 1024 bytes)."""

_TOKEN = re.compile(rb'"([^"\n]*)"?|([^ \t\r"][^ \t\r]*)')

_SECTIONS = {
    b"[FILE": "FILES",
    b"[OPTION": "OPTIONS",
    b"[RAINGAGE": "RAINGAGES",
    b"[TIMESERIES": "TIMESERIES",
    b"[TEMPERATURE": "TEMPERATURE",
    b"[EVAP": "EVAPORATION",
    b"[LID_USAGE": "LID_USAGE",
}
"""Header prefixes, as the engine spells them, of the sections that can name a file.
No other SWMM section keyword is a prefix of one of these, so the first match is the one."""


@dataclass(frozen=True)
class UnsafeInput:
    line: int
    reason: str


def _tokens(line: bytes) -> list[bytes]:
    code = line.split(b"\0", 1)[0].split(b";", 1)[0]
    return [unquoted or quoted for quoted, unquoted in _TOKEN.findall(code)]


def _section(header: bytes) -> str | None:
    upper = header.upper()
    return next((name for prefix, name in _SECTIONS.items() if upper.startswith(prefix)), None)


def _is(token: bytes, keyword: bytes) -> bool:
    # The engine's keyword match skips leading blanks, which a quoted token can carry.
    return token.lstrip(b" \t\r").upper().startswith(keyword)


def _external_file_reason(section: str | None, tokens: list[bytes]) -> str | None:
    """Why a data line of ``section`` names an external file, or ``None``."""
    if section == "OPTIONS" and _is(tokens[0], b"TEMPDIR"):
        return "the TEMPDIR option"
    if section == "RAINGAGES" and len(tokens) > 4 and _is(tokens[4], b"FILE"):
        return "a rain gage that reads a FILE"
    if section == "TIMESERIES" and len(tokens) > 1 and _is(tokens[1], b"FILE"):
        return "a time series that reads a FILE"
    if section == "TEMPERATURE" and (
        _is(tokens[0], b"FILE") or (_is(tokens[0], b"WINDSPEED") and len(tokens) > 1 and _is(tokens[1], b"FILE"))
    ):
        return "climate data read from a FILE"
    if section == "EVAPORATION" and _is(tokens[0], b"FILE"):
        return "evaporation read from a climate FILE"
    if section == "LID_USAGE" and len(tokens) > 8 and tokens[8] != b"*":
        return "an LID report file"
    return None


def _lines(chunks: Iterable[bytes]) -> Iterable[bytes | None]:
    """Yield the lines of the stream; ``None`` for a line too long for the engine."""
    pending = b""
    for chunk in chunks:
        pending += chunk
        *complete, pending = pending.split(b"\n")
        for line in complete:
            yield line if len(line) <= MAX_LINE_BYTES else None
        if len(pending) > MAX_LINE_BYTES:
            yield None
            return
    if pending:
        yield pending


def find_unsafe_input(chunks: Iterable[bytes]) -> UnsafeInput | None:
    """Return the first construct in the file that the app refuses, or ``None``.

    Memory stays bounded by one chunk plus one line, whatever the file size.
    """
    section: str | None = None
    for number, line in enumerate(_lines(chunks), start=1):
        if line is None:
            return UnsafeInput(number, f"more than {MAX_LINE_BYTES} characters, which SWMM would split into separate lines")
        tokens = _tokens(line)
        if not tokens:
            continue
        if tokens[0].startswith(b"["):
            section = _section(tokens[0])
            if section == "FILES":
                return UnsafeInput(number, "a [FILES] section")
            continue
        reason = _external_file_reason(section, tokens)
        if reason:
            return UnsafeInput(number, reason)
    return None
