"""Reading the summary tables of a SWMM report.

``swmmio`` can turn some report sections into frames, but not the ones the
physical view needs: ``get_rpt_sections_details`` gives *Conduit Surcharge
Summary* no column list at all (it comes back as ``blob``), and the
"section printed as a sentence instead of a table" case is special-cased for
node flooding only. Both are load-bearing here -- plan16 §2 P1 asks for the
surcharge hours by name, and asks that an absent table read as ``0``, not as
missing data -- so the four tables are parsed here instead.

The layout SWMM writes is fixed and simple: a banner of asterisks around the
heading, then a dashed rule, the column headings, a second dashed rule, and the
data rows. Every table this module reads follows it, so one row-finder serves
all four, and each table only declares how many fields its rows carry and which
of them it keeps.

Two absences with different meanings, kept in separate code paths on purpose:

* *Conduit Surcharge Summary* and *Node Flooding Summary* are lists of
  exceptions. SWMM replaces them with a single sentence when nothing qualifies,
  and a conduit missing from a table that is present is in exactly the same
  state as every conduit when the table is absent: **zero** hours, zero flooding.
  Turning that into a missing value would make "nothing happened" unmeasured.
* *Node Depth Summary* and *Link Flow Summary* report every element. A node or
  link that is not there was not measured, and the caller has to see the
  difference, so those two never invent a row.

Kinematic wave semantics (plan16 §2 P1): the surcharge hours mean the flow
exceeded the normal-flow capacity of the conduit, not that the pipe was under
pressure, and node flooding means inflow exceeded the outflow capacity of the
node. The numbers are read as reported; naming them is the caller's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

LINK_FLOW_SUMMARY = "Link Flow Summary"
CONDUIT_SURCHARGE_SUMMARY = "Conduit Surcharge Summary"
NODE_DEPTH_SUMMARY = "Node Depth Summary"
NODE_FLOODING_SUMMARY = "Node Flooding Summary"

NO_CONDUITS_SURCHARGED = "No conduits were surcharged."
NO_NODES_FLOODED = "No nodes were flooded."
"""The sentences SWMM prints in place of an empty table.

Verbatim from reports produced by the engine this study runs -- for example
``models/recomendations/dataset-backup/proba/generated/
01_recomendation_template_D1000_EulerRain60_S6_0_rural.rpt`` -- rather than from
the manual, because the parser has to recognise the exact string.
"""

CONDUIT_LINK_TYPE = "CONDUIT"
"""The only link type whose row carries the full flow-summary layout.

Pumps, orifices, weirs and outlets are printed with fewer fields, and none of
them is a conduit of the physical view, so they are skipped rather than parsed
into a row with holes."""

_ANALYSIS_END_MARKER = "Analysis begun on"
_MIN_RULE_LENGTH = 10

_CONTINUITY = re.compile(r"Continuity Error \(%\)\s*\.*\s*(-?\d+(?:\.\d+)?)")
"""Duplicated from :mod:`stormlab.datasets.eval_sets`, which may not be edited here.

The two readings are the same number under the same pattern; when the build path
of plan16 §2 P2 takes over the continuity gate, ``eval_sets`` is the caller that
should come to this module rather than keep its own copy.
"""


class RptParseError(RuntimeError):
    """Raised when a report does not carry a section, or a row does not parse."""


@dataclass(frozen=True)
class ReportTables:
    """The four summary tables of one report, plus its two mass balances.

    Attributes:
        link_flow: peak flow, velocity and the two full-section ratios, indexed
            by conduit name.
        conduit_surcharge: the five hour counts, indexed by conduit name. Empty
            when nothing surcharged, which the caller reads as zero hours.
        node_depth: average and maximum depth, maximum HGL and the reported
            maximum depth, indexed by node name.
        node_flooding: flood hours, peak rate, total and ponded volume, indexed
            by node name. Empty when nothing flooded.
        runoff_continuity_error_percent: signed runoff mass balance error [%].
        routing_continuity_error_percent: signed routing mass balance error [%].
    """

    link_flow: pd.DataFrame
    conduit_surcharge: pd.DataFrame
    node_depth: pd.DataFrame
    node_flooding: pd.DataFrame
    runoff_continuity_error_percent: float
    routing_continuity_error_percent: float


LINK_FLOW_COLUMNS: tuple[str, ...] = (
    "max_flow_cms",
    "max_velocity_mps",
    "max_full_flow_ratio",
    "max_full_depth_ratio",
)
CONDUIT_SURCHARGE_COLUMNS: tuple[str, ...] = (
    "hours_full_both_ends",
    "hours_full_upstream",
    "hours_full_downstream",
    "hours_above_normal_flow",
    "hours_capacity_limited",
)
NODE_DEPTH_COLUMNS: tuple[str, ...] = (
    "avg_depth_m",
    "max_depth_m",
    "max_hgl_m",
    "max_depth_reported_m",
)
NODE_FLOODING_COLUMNS: tuple[str, ...] = (
    "hours_flooded",
    "max_flood_rate_cms",
    "total_flood_volume_1e6l",
    "max_ponded_volume_1000m3",
)

_LINK_FLOW_FIELDS = 8
_CONDUIT_SURCHARGE_FIELDS = 6
_NODE_DEPTH_FIELDS = 8
_NODE_FLOODING_FIELDS = 7

# Field positions inside a row, counting the name as 0. The two columns holding
# the time of the maximum ("days hr:min") are not kept: no source of the C1
# series reads a time of occurrence, and returning them as an unparsed string
# pair would be a column nothing can compute with.
_LINK_FLOW_POSITIONS: tuple[int, ...] = (2, 5, 6, 7)
_CONDUIT_SURCHARGE_POSITIONS: tuple[int, ...] = (1, 2, 3, 4, 5)
_NODE_DEPTH_POSITIONS: tuple[int, ...] = (2, 3, 4, 7)
_NODE_FLOODING_POSITIONS: tuple[int, ...] = (1, 2, 5, 6)


def read_report(path: Path | str) -> ReportTables:
    """Parse the report at ``path``.

    Raises:
        RptParseError: if the file is missing or a required section is absent.
    """
    report = Path(path)
    if not report.is_file():
        raise RptParseError(f"SWMM report {report} does not exist")
    return parse_report(report.read_text(encoding="utf-8", errors="replace"), source=report)


def parse_report(text: str, *, source: Path | str | None = None) -> ReportTables:
    """Parse an already-read report text.

    Args:
        text: the whole report.
        source: the file the text came from, used only in error messages.

    Raises:
        RptParseError: if a required section is absent or a row does not parse.
    """
    runoff, routing = continuity_errors(text, source=source)
    return ReportTables(
        link_flow=link_flow_summary(text, source=source),
        conduit_surcharge=conduit_surcharge_summary(text, source=source),
        node_depth=node_depth_summary(text, source=source),
        node_flooding=node_flooding_summary(text, source=source),
        runoff_continuity_error_percent=runoff,
        routing_continuity_error_percent=routing,
    )


def link_flow_summary(text: str, *, source: Path | str | None = None) -> pd.DataFrame:
    """Peak flow, velocity and the two full-section ratios of every conduit."""
    rows = _data_rows(text, LINK_FLOW_SUMMARY, empty_sentence=None, source=source)
    conduits = [row for row in rows if len(row) > 1 and row[1] == CONDUIT_LINK_TYPE]
    return _numeric_frame(
        conduits,
        heading=LINK_FLOW_SUMMARY,
        columns=LINK_FLOW_COLUMNS,
        positions=_LINK_FLOW_POSITIONS,
        n_fields=_LINK_FLOW_FIELDS,
        index_name="conduit",
        source=source,
    )


def conduit_surcharge_summary(text: str, *, source: Path | str | None = None) -> pd.DataFrame:
    """The five hour counts of every surcharged conduit; empty when none was."""
    rows = _data_rows(text, CONDUIT_SURCHARGE_SUMMARY, empty_sentence=NO_CONDUITS_SURCHARGED, source=source)
    return _numeric_frame(
        rows,
        heading=CONDUIT_SURCHARGE_SUMMARY,
        columns=CONDUIT_SURCHARGE_COLUMNS,
        positions=_CONDUIT_SURCHARGE_POSITIONS,
        n_fields=_CONDUIT_SURCHARGE_FIELDS,
        index_name="conduit",
        source=source,
    )


def node_depth_summary(text: str, *, source: Path | str | None = None) -> pd.DataFrame:
    """Average and maximum depth, maximum HGL and reported maximum depth per node."""
    rows = _data_rows(text, NODE_DEPTH_SUMMARY, empty_sentence=None, source=source)
    return _numeric_frame(
        rows,
        heading=NODE_DEPTH_SUMMARY,
        columns=NODE_DEPTH_COLUMNS,
        positions=_NODE_DEPTH_POSITIONS,
        n_fields=_NODE_DEPTH_FIELDS,
        index_name="node",
        source=source,
    )


def node_flooding_summary(text: str, *, source: Path | str | None = None) -> pd.DataFrame:
    """Flood hours, peak rate and volumes of every flooded node; empty when none was."""
    rows = _data_rows(text, NODE_FLOODING_SUMMARY, empty_sentence=NO_NODES_FLOODED, source=source)
    return _numeric_frame(
        rows,
        heading=NODE_FLOODING_SUMMARY,
        columns=NODE_FLOODING_COLUMNS,
        positions=_NODE_FLOODING_POSITIONS,
        n_fields=_NODE_FLOODING_FIELDS,
        index_name="node",
        source=source,
    )


def continuity_errors(text: str, *, source: Path | str | None = None) -> tuple[float, float]:
    """The signed runoff and routing mass balance errors [%], in that order.

    Raises:
        RptParseError: if either continuity section, or its error line, is absent.
    """
    values: list[float] = []
    for name, heading in (("runoff", "Runoff Quantity Continuity"), ("routing", "Flow Routing Continuity")):
        position = text.find(heading)
        if position < 0:
            raise RptParseError(f"{_where(source)} has no {name} continuity section ({heading!r})")
        match = _CONTINUITY.search(text, position)
        if match is None:
            raise RptParseError(f"{_where(source)} has no continuity error line after the {name} heading")
        values.append(float(match.group(1)))
    return values[0], values[1]


def _data_rows(
    text: str,
    heading: str,
    *,
    empty_sentence: str | None,
    source: Path | str | None,
) -> list[list[str]]:
    """Whitespace-split data rows of one summary table.

    Returns an empty list when the section carries ``empty_sentence`` instead of
    a table, which only the two exception tables declare.

    Raises:
        RptParseError: if the section is absent, or is present without a table
            and without the sentence that legitimately replaces one.
    """
    body = _section_body(text, heading, source=source)
    if empty_sentence is not None and any(line.strip() == empty_sentence for line in body):
        return []

    rules = [index for index, line in enumerate(body) if _is_rule(line)]
    if len(rules) < 2:
        instead = f" nor the sentence {empty_sentence!r}" if empty_sentence is not None else ""
        raise RptParseError(f"{_where(source)}: section {heading!r} carries no table{instead}; first lines were {body[:4]}")

    rows: list[list[str]] = []
    for line in body[rules[1] + 1 :]:
        if _is_rule(line):
            break
        if not line.strip():
            continue
        rows.append(line.split())
    return rows


def _section_body(text: str, heading: str, *, source: Path | str | None) -> list[str]:
    """Lines between a section's heading banner and the next one.

    Raises:
        RptParseError: if no banner carries ``heading``.
    """
    lines = text.splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() == heading
            and 0 < index < len(lines) - 1
            and _is_banner(lines[index - 1])
            and _is_banner(lines[index + 1])
        ),
        None,
    )
    if start is None:
        raise RptParseError(f"{_where(source)} has no {heading!r} section")

    body: list[str] = []
    for line in lines[start + 2 :]:
        if _is_banner(line) or line.strip().startswith(_ANALYSIS_END_MARKER):
            break
        body.append(line)
    return body


def _numeric_frame(
    rows: Sequence[Sequence[str]],
    *,
    heading: str,
    columns: tuple[str, ...],
    positions: tuple[int, ...],
    n_fields: int,
    index_name: str,
    source: Path | str | None,
) -> pd.DataFrame:
    """Build a float frame indexed by element name from already-split rows.

    Raises:
        RptParseError: if a row has the wrong number of fields, carries a value
            that is not a number, or names an element twice.
    """
    names: list[str] = []
    values: list[list[float]] = []
    for row in rows:
        if len(row) != n_fields:
            raise RptParseError(
                f"{_where(source)}: row {' '.join(row)!r} of {heading!r} has {len(row)} fields, expected {n_fields}"
            )
        names.append(row[0])
        try:
            values.append([float(row[position]) for position in positions])
        except ValueError as error:
            raise RptParseError(f"{_where(source)}: row {' '.join(row)!r} of {heading!r} is not numeric: {error}") from error

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RptParseError(f"{_where(source)}: {heading!r} names {duplicates} more than once")

    frame = pd.DataFrame(values, columns=list(columns), index=pd.Index(names, name=index_name, dtype="object"), dtype="float64")
    return frame


def _is_rule(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= _MIN_RULE_LENGTH and set(stripped) == {"-"}


def _is_banner(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {"*"}


def _where(source: Path | str | None) -> str:
    return f"SWMM report {source}" if source is not None else "SWMM report"


def zero_filled(frame: pd.DataFrame, elements: Sequence[str], columns: tuple[str, ...]) -> pd.DataFrame:
    """Reindex an exception table onto ``elements``, filling absences with zero.

    The one place the "absent means zero" reading of the surcharge and flooding
    tables is applied, so it cannot be confused with the "absent means
    unmeasured" reading the reported tables get.

    Args:
        frame: a table parsed from an exception section.
        elements: every element the caller expects an answer for.
        columns: the columns to return, in order.
    """
    index = pd.Index([str(element) for element in elements], name=frame.index.name)
    return frame.reindex(index=index, columns=list(columns)).fillna(0.0).astype("float64")


__all__ = [
    "CONDUIT_LINK_TYPE",
    "CONDUIT_SURCHARGE_COLUMNS",
    "CONDUIT_SURCHARGE_SUMMARY",
    "LINK_FLOW_COLUMNS",
    "LINK_FLOW_SUMMARY",
    "NODE_DEPTH_COLUMNS",
    "NODE_DEPTH_SUMMARY",
    "NODE_FLOODING_COLUMNS",
    "NODE_FLOODING_SUMMARY",
    "NO_CONDUITS_SURCHARGED",
    "NO_NODES_FLOODED",
    "ReportTables",
    "RptParseError",
    "conduit_surcharge_summary",
    "continuity_errors",
    "link_flow_summary",
    "node_depth_summary",
    "node_flooding_summary",
    "parse_report",
    "read_report",
    "zero_filled",
]
