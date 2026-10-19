"""Tiny SWMM inputs and reports written from scratch, for the physical view tests.

Not a test module: pytest collects ``test_*.py`` only. It lives here because two
test modules need the same two builders, and because the point of these fixtures
is that every quantity under test -- an offset, a barrel count, an adverse slope,
a node missing from the depth table -- is set by the test rather than found in a
file someone else produced.

The INP is the smallest thing ``swmmio`` will open and the report is the smallest
thing :mod:`stormlab.io.rpt` will read: the sections, their banners and their
column layouts are copied from real SWMM output, the rows are not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

DEFAULT_TAG = "rural"

_HEADER = """[TITLE]
;;Project Title/Notes
physical fixture

[OPTIONS]
;;Option             Value
FLOW_UNITS           CMS
INFILTRATION         HORTON
FLOW_ROUTING         KINWAVE
LINK_OFFSETS         {link_offsets}
ALLOW_PONDING        NO
SKIP_STEADY_STATE    NO
START_DATE           06/17/2022
START_TIME           00:00:00
REPORT_START_DATE    06/17/2022
REPORT_START_TIME    00:00:00
END_DATE             06/17/2022
END_TIME             02:00:00
SWEEP_START          01/01
SWEEP_END            12/31
DRY_DAYS             0
REPORT_STEP          00:15:00
WET_STEP             00:05:00
DRY_STEP             01:00:00
ROUTING_STEP         0:01:00
MIN_SLOPE            0
THREADS              1

[EVAPORATION]
;;Data Source    Parameters
CONSTANT         0.0
DRY_ONLY         NO

[RAINGAGES]
;;Name           Format    Interval SCF      Source
G1               INTENSITY 0:05     1.0      TIMESERIES S1

[SUBCATCHMENTS]
;;Name           Rain Gage        Outlet           Area     %Imperv  Width    %Slope   CurbLen
{subcatchment_lines}

[SUBAREAS]
;;Subcatchment   N-Imperv   N-Perv     S-Imperv   S-Perv     PctZero    RouteTo
{subarea_lines}

[INFILTRATION]
;;Subcatchment   MaxRate    MinRate    Decay      DryTime    MaxInfil
{infiltration_lines}
"""

_FOOTER = """
[TIMESERIES]
;;Name           Date       Time       Value
S1                          0:00       0.0
S1                          0:05       30.0
S1                          0:10       60.0
S1                          0:15       30.0

[REPORT]
;;Reporting Options
SUBCATCHMENTS ALL
NODES ALL
LINKS ALL

[TAGS]
{tag_lines}

[COORDINATES]
;;Node           X-Coord            Y-Coord
{coordinates}

[Polygons]
;;Subcatchment   X-Coord            Y-Coord
{polygon_lines}
"""


def write_inp(
    path: Path,
    *,
    junctions: Sequence[Mapping[str, object]],
    outfalls: Sequence[Mapping[str, object]] = (),
    conduits: Sequence[Mapping[str, object]],
    link_offsets: str = "DEPTH",
    tag: str = DEFAULT_TAG,
    catchment_outlet: str | None = None,
    catchments: Sequence[Mapping[str, object]] | None = None,
) -> Path:
    """Write a minimal but complete INP and return its path.

    Args:
        path: where to write.
        junctions: mappings with ``name``, ``invert``, ``max_depth``.
        outfalls: mappings with ``name``, ``invert`` and optionally ``kind``
            (``FREE`` by default) and ``stage``.
        conduits: mappings with ``name``, ``inlet``, ``outlet``, ``length``,
            ``diameter`` and optionally ``roughness``, ``in_offset``,
            ``out_offset``, ``barrels``.
        link_offsets: the ``LINK_OFFSETS`` mode the offsets are written in.
        tag: the land cover class tagged onto every subcatchment that does not
            name its own.
        catchment_outlet: the node the single default subcatchment drains into;
            the first junction by default. Ignored when ``catchments`` is given.
        catchments: mappings with ``name``, ``outlet`` and optionally ``area``,
            ``imperv``, ``width``, ``slope`` and ``tag``. One catchment named
            ``SC1`` draining into ``catchment_outlet`` by default, which is what
            every test that does not care about the catchment gets.
    """
    stated = (
        list(catchments) if catchments is not None else [{"name": "SC1", "outlet": catchment_outlet or str(junctions[0]["name"])}]
    )
    if not stated:
        raise ValueError("write_inp needs at least one catchment; swmmio will not open a file without one")
    catchment_names = [str(item["name"]) for item in stated]
    subcatchment_lines = "\n".join(
        f"{str(item['name']):<17s}G1               {str(item['outlet']):<17s}"
        f"{float(item.get('area', 1.0)):<9.1f}{float(item.get('imperv', 50.0)):<9.1f}"
        f"{float(item.get('width', 100.0)):<9.1f}{float(item.get('slope', 1.0)):<9.1f}0"
        for item in stated
    )
    subarea_lines = "\n".join(
        f"{name:<17s}0.01       0.1        0.05       0.05       25         OUTLET" for name in catchment_names
    )
    infiltration_lines = "\n".join(f"{name:<17s}3.0        0.5        4          7          0" for name in catchment_names)
    tag_lines = "\n".join(f"Subcatch         {str(item['name']):<17s}{str(item.get('tag', tag))}" for item in stated)
    polygon_lines = "\n".join(f"{name:<17s}{index * 10:<19.3f}0.000" for index, name in enumerate(catchment_names))
    nodes = list(junctions) + list(outfalls)
    coordinates = "\n".join(f"{str(node['name']):<17s}{index * 100:<19.3f}0.000" for index, node in enumerate(nodes))

    junction_lines = "\n".join(
        f"{str(node['name']):<17s}{float(node['invert']):<11.4f}{float(node['max_depth']):<11.3f}0          0          0"
        for node in junctions
    )
    outfall_lines = "\n".join(
        f"{str(node['name']):<17s}{float(node['invert']):<11.4f}{str(node.get('kind', 'FREE')):<11s}"
        f"{'' if node.get('stage') is None else float(node['stage']):<17}NO"
        for node in outfalls
    )
    conduit_lines = "\n".join(
        f"{str(item['name']):<17s}{str(item['inlet']):<17s}{str(item['outlet']):<17s}"
        f"{float(item['length']):<11.3f}{float(item.get('roughness', 0.013)):<11.3f}"
        f"{float(item.get('in_offset', 0.0)):<11.4f}{float(item.get('out_offset', 0.0)):<11.4f}0          0"
        for item in conduits
    )
    xsection_lines = "\n".join(
        f"{str(item['name']):<17s}CIRCULAR     {float(item['diameter']):<17.3f}0          0          0          "
        f"{int(item.get('barrels', 1))}"
        for item in conduits
    )

    text = (
        _HEADER.format(
            link_offsets=link_offsets,
            subcatchment_lines=subcatchment_lines,
            subarea_lines=subarea_lines,
            infiltration_lines=infiltration_lines,
        )
        + "\n[JUNCTIONS]\n;;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded\n"
        + junction_lines
        + "\n\n[OUTFALLS]\n;;Name           Elevation  Type       Stage Data       Gated\n"
        + outfall_lines
        + "\n\n[CONDUITS]\n;;Name           From Node        To Node          Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow\n"
        + conduit_lines
        + "\n\n[XSECTIONS]\n;;Link           Shape        Geom1            Geom2      Geom3      Geom4      Barrels\n"
        + xsection_lines
        + "\n"
        + _FOOTER.format(tag_lines=tag_lines, coordinates=coordinates, polygon_lines=polygon_lines)
    )
    path.write_text(text, encoding="utf-8")
    return path


def write_rpt(
    path: Path,
    *,
    links: Mapping[str, Sequence[float]],
    nodes: Mapping[str, Sequence[float]],
    surcharge: Mapping[str, Sequence[float]] | None = None,
    flooding: Mapping[str, Sequence[float]] | None = None,
    runoff_error: float = -0.3,
    routing_error: float = 0.4,
) -> Path:
    """Write a report carrying exactly the rows given, and return its path.

    Args:
        path: where to write.
        links: conduit name to ``(max_flow, max_velocity, full_flow_ratio,
            full_depth_ratio)``.
        nodes: node name to ``(avg_depth, max_depth, max_hgl,
            reported_max_depth)``. A node left out is absent from the table,
            which is how "not measured" is set up.
        surcharge: conduit name to the five hour counts; ``None`` writes the
            sentence SWMM writes when nothing surcharged.
        flooding: node name to ``(hours, rate, volume, ponded)``; ``None`` writes
            the sentence SWMM writes when nothing flooded.
        runoff_error: the runoff continuity error [%].
        routing_error: the routing continuity error [%].
    """
    from stormlab.io.rpt import NO_CONDUITS_SURCHARGED, NO_NODES_FLOODED

    link_rows = "".join(
        f"  {name:<20s} CONDUIT  {values[0]:>9.3f}     0  00:17   {values[1]:>7.2f} {values[2]:>7.2f} {values[3]:>7.2f}\n"
        for name, values in links.items()
    )
    node_rows = "".join(
        f"  {name:<20s} JUNCTION {values[0]:>8.2f} {values[1]:>8.2f} {values[2]:>8.2f}     0  00:16 {values[3]:>11.2f}\n"
        for name, values in nodes.items()
    )
    if surcharge is None:
        surcharge_body = f"  {NO_CONDUITS_SURCHARGED}\n"
    else:
        surcharge_body = (
            "  " + "-" * 76 + "\n"
            "                                                           Hours        Hours\n"
            "                         --------- Hours Full --------   Above Full   Capacity\n"
            "  Conduit                Both Ends  Upstream  Dnstream   Normal Flow   Limited\n"
            "  " + "-" * 76 + "\n"
        ) + "".join(
            f"  {name:<20s} {values[0]:>9.2f} {values[1]:>9.2f} {values[2]:>9.2f} {values[3]:>9.2f} {values[4]:>9.2f}\n"
            for name, values in surcharge.items()
        )
    if flooding is None:
        flooding_body = f"  {NO_NODES_FLOODED}\n"
    else:
        flooding_body = (
            "  " + "-" * 74 + "\n"
            "                                                             Total   Maximum\n"
            "                                 Maximum   Time of Max       Flood    Ponded\n"
            "                        Hours       Rate    Occurrence      Volume    Volume\n"
            "  Node                 Flooded       CMS   days hr:min    10^6 ltr   1000 m3\n"
            "  " + "-" * 74 + "\n"
        ) + "".join(
            f"  {name:<20s} {values[0]:>8.2f} {values[1]:>9.3f}      0  00:17 {values[2]:>11.3f} {values[3]:>9.3f}\n"
            for name, values in flooding.items()
        )

    text = (
        "  " + "*" * 13 + "\n"
        "  Runoff Quantity Continuity     hectare-m            mm\n"
        "  " + "*" * 13 + "\n"
        f"  Continuity Error (%) .....     {runoff_error}\n"
        "\n"
        "  " + "*" * 13 + "\n"
        "  Flow Routing Continuity        hectare-m      10^6 ltr\n"
        "  " + "*" * 13 + "\n"
        f"  Continuity Error (%) .....     {routing_error}\n"
        "\n"
        "  " + "*" * 18 + "\n"
        "  Node Depth Summary\n"
        "  " + "*" * 18 + "\n"
        "\n"
        "  " + "-" * 81 + "\n"
        "                                 Average  Maximum  Maximum  Time of Max    Reported\n"
        "                                   Depth    Depth      HGL   Occurrence   Max Depth\n"
        "  Node                 Type       Meters   Meters   Meters  days hr:min      Meters\n"
        "  " + "-" * 81 + "\n"
        f"{node_rows}"
        "\n"
        "  " + "*" * 21 + "\n"
        "  Node Flooding Summary\n"
        "  " + "*" * 21 + "\n"
        "\n"
        f"{flooding_body}"
        "\n"
        "  " + "*" * 18 + "\n"
        "  Link Flow Summary\n"
        "  " + "*" * 18 + "\n"
        "\n"
        "  " + "-" * 77 + "\n"
        "                                 Maximum  Time of Max   Maximum    Max/    Max/\n"
        "                                  |Flow|   Occurrence   |Veloc|    Full    Full\n"
        "  Link                 Type          CMS  days hr:min     m/sec    Flow   Depth\n"
        "  " + "-" * 77 + "\n"
        f"{link_rows}"
        "\n"
        "  " + "*" * 25 + "\n"
        "  Conduit Surcharge Summary\n"
        "  " + "*" * 25 + "\n"
        "\n"
        f"{surcharge_body}"
        "\n"
        "  Analysis begun on:  Mon Aug  4 06:13:55 2025\n"
        "  Analysis ended on:  Mon Aug  4 06:13:55 2025\n"
        "  Total elapsed time: < 1 sec\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


__all__ = ["DEFAULT_TAG", "write_inp", "write_rpt"]
