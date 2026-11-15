"""Lightweight parsers for SWMM INP [OPTIONS] section values: the flow units and the simulated period."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import islice

MAX_SCANNED_LINES = 5000
"""[OPTIONS] is the first section of every SWMM file; a file that hides it deeper is not one we want."""


def read_flow_units(lines: Iterable[str]) -> str | None:
    """Return the FLOW_UNITS value from the [OPTIONS] section, or *None*.

    Scans *lines* (any iterable of strings — file, list, iterator) for the
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


# ---------------------------------------------------------------------------
# Simulated period and report steps, for the cost limits of the upload form
# ---------------------------------------------------------------------------

PERIOD_OPTIONS = ("START_DATE", "START_TIME", "END_DATE", "END_TIME", "REPORT_START_DATE", "REPORT_START_TIME", "REPORT_STEP")
"""The option keywords the period is computed from. The engine matches a keyword by prefix
(``END_DATE_X`` sets END_DATE) and no SWMM 5.2 option word is a prefix of another, so a
token names the one word it starts with."""

DEFAULT_DATE = datetime(2004, 1, 1)
"""SWMM 5.2 starts *and* ends a simulation on 01/01/2004 when the file names no date (an
input without END_DATE therefore ends before it starts and the engine refuses it)."""

DEFAULT_REPORT_STEP_SECONDS = 15 * 60

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_DATE = re.compile(r"^(\d+|[A-Za-z]{3})[/-](\d+)[/-](\d+)$")
_DECIMAL_HOURS = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_CLOCK = re.compile(r"^(\d+):(\d+)(?::(\d+))?$")


class OptionValueError(ValueError):
    """An option value written in a form StormLab does not read."""

    def __init__(self, option: str, value: str):
        super().__init__(f"{option} {value}")
        self.option = option
        self.value = value


@dataclass(frozen=True)
class SimulationPeriod:
    duration_seconds: float
    report_steps: int

    @property
    def duration_hours(self) -> float:
        return self.duration_seconds / 3600.0


def period_options(lines: Iterable[tuple[str, str | None]]) -> dict[str, str]:
    """The value of each of :data:`PERIOD_OPTIONS`, the last line winning as in the engine.

    *lines* are ``(keyword, value)`` pairs of every ``[OPTIONS]`` section, in file order
    (:attr:`sa.core.inp_structure.InpSurvey.options`); a line without a value is skipped.
    """
    found: dict[str, str] = {}
    for keyword, value in lines:
        option = next((word for word in PERIOD_OPTIONS if keyword.upper().startswith(word)), None)
        if option is not None and value is not None:
            found[option] = value
    return found


def parse_date(option: str, value: str) -> datetime:
    """A SWMM date: month/day/year or month-day-year, the month as a number or its first three letters."""
    match = _DATE.match(value.strip())
    if not match:
        raise OptionValueError(option, value)
    month_text, day, year = match.groups()
    if month_text.isdigit():
        month = int(month_text)
    elif month_text.upper() in _MONTHS:
        month = _MONTHS.index(month_text.upper()) + 1
    else:
        raise OptionValueError(option, value)
    try:
        return datetime(int(year), month, int(day))
    except ValueError:
        raise OptionValueError(option, value) from None


def parse_clock_seconds(option: str, value: str) -> float:
    """A SWMM time or time step in seconds: decimal hours (``6.5``) or hours:minutes[:seconds].

    Hours may exceed 24 (``30:00:00`` ends six hours into the next day), as the engine reads
    them. The engine first tries the whole value as a C number, in hours, so ``1e3`` is a
    thousand hours, not one: anything that is neither a plain number nor a plain clock
    (``inf``, ``0x10``, ``12:00abc``) is refused rather than guessed.
    """
    text = value.strip()
    if _DECIMAL_HOURS.match(text):
        return float(text) * 3600.0
    match = _CLOCK.match(text)
    if not match:
        raise OptionValueError(option, value)
    hours, minutes, seconds = (int(part) if part else 0 for part in match.groups())
    return hours * 3600.0 + minutes * 60.0 + seconds


def simulation_period(lines: Iterable[tuple[str, str | None]]) -> SimulationPeriod:
    """How long the file simulates and how many report steps that writes, with SWMM's defaults.

    Reporting starts at REPORT_START_DATE and REPORT_START_TIME when both are given, never
    before the start of the simulation; otherwise at the start. The steps are counted over
    the reporting period. A period that ends before it starts is left for the engine to
    refuse and counts as zero.

    Raises:
        OptionValueError: for a date, time or step in a form this reader does not know,
            or a report step that is not longer than zero.
    """
    options = period_options(lines)

    def moment(date_option: str, time_option: str, default_date: datetime) -> datetime:
        date = parse_date(date_option, options[date_option]) if date_option in options else default_date
        seconds = parse_clock_seconds(time_option, options[time_option]) if time_option in options else 0.0
        return date + timedelta(seconds=seconds)

    start = moment("START_DATE", "START_TIME", DEFAULT_DATE)
    end = moment("END_DATE", "END_TIME", DEFAULT_DATE)
    report_start = start
    if "REPORT_START_DATE" in options and "REPORT_START_TIME" in options:
        report_start = max(start, moment("REPORT_START_DATE", "REPORT_START_TIME", DEFAULT_DATE))
    step = DEFAULT_REPORT_STEP_SECONDS
    if "REPORT_STEP" in options:
        step = parse_clock_seconds("REPORT_STEP", options["REPORT_STEP"])
        if step <= 0:
            raise OptionValueError("REPORT_STEP", options["REPORT_STEP"])
    duration = max(0.0, (end - start).total_seconds())
    reported = max(0.0, (end - report_start).total_seconds())
    return SimulationPeriod(duration_seconds=duration, report_steps=int(reported // step))
