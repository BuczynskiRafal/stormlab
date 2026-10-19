"""Physical state of a conduit, measured and undecided: the F2 and F3 views.

plan16 §2 P1 puts a seam between *what the network is* and *what should be done
about it*. The historical :mod:`stormlab.features.conduits` has no such seam: one
pass sizes a replacement diameter, raises validity flags and squeezes every
quantity onto ``[0, 1]``, so a violation twice as large reads the same as one at
the threshold. This module is the other side of the seam and does none of that.

What it reads:

* from the INP -- diameter, length, roughness, barrels, both conduit offsets and
  the ``LINK_OFFSETS`` mode they are written in, node inverts and manhole depths,
  the outfall of the network, the catchment draining into the conduit (see below)
  and the rainfall series driving the file;
* from the RPT -- Link Flow Summary, Conduit Surcharge Summary, Node Depth
  Summary and Node Flooding Summary, through :mod:`stormlab.io.rpt`.

What it deliberately does not read: the binary ``.out``. No source of the C1
series asks for a time series, and a reader for one would be an unused
dependency on a much larger file per case.

The catchment of a conduit is everything **above** it (author's decision of
2026-09-07). ``subcatchment_*`` used to describe only the catchments wired to the
conduit's own inlet node, which made 31 % of the C1 training conduits catchment-less
even though every one of them carries water off some terrain -- and it disagreed with
the historical view, which propagates the land cover class down the network. The
columns now describe the **cumulative inflowing catchment**: every catchment attached
to the inlet node and to every node upstream of it in the conduit graph, area summed
and the intensive parameters area-weighted over that sum. The mask is set only where
nothing upstream states an area.

Two facts about the geometry the historical path does not carry:

* the invert of the **pipe** is the invert of the node plus the offset, and is
  not the invert of the manhole. With the zero offsets of the B4 pool the two
  coincide, which is why the difference is verified by the DEPTH/ELEVATION
  equivalence test rather than by the data;
* the slope keeps its sign. A flat or adverse conduit gets no capacity at all
  (``full_flow_cms`` is missing, ``slope_nonpositive`` is 1) instead of a
  fictitious one.

Kinematic wave semantics (plan16 §2 P1). Node flooding is inflow in excess of
what the node can pass on, and the surcharge hours count time above the *normal
flow* capacity of the conduit -- not backwater, not pressurised flow, neither of
which this routing scheme models. The columns are named after what the report
measures, and the report is read the same way for both.

The reading does not change under a dynamic wave, but what it *means* does: there
the head at a node is computed rather than assumed, so ``{inlet,outlet}_node_max_hgl_m``
and the freeboard derived from it describe a real water level and node flooding is a
surcharge that reached the ground. This module states the measurement either way; it
is the protocol that decides which routing scheme produced the report.

Missing versus zero, kept apart on purpose. The surcharge and flooding tables are
lists of exceptions, so a conduit absent from them has genuinely zero hours; the
node and link tables report everything, so an element absent from them was not
measured and its columns are missing, with the matching mask set. That is the
one asymmetry the whole "brak pomiaru nie jest pomiarem zerowym" test rests on.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import swmmio as sw

from stormlab.config import Config
from stormlab.contracts.features import (
    FEATURE_CONTRACT_V3_PHYSICAL,
    FEATURE_CONTRACT_V4_MARGINS,
    FEATURE_ONEHOT_ORDER,
    FeatureContract,
    FeatureContractError,
)
from stormlab.features.conduits import ConduitFeatureSteps, resolve_required_cover
from stormlab.features.nodes import assign_subcatchment_info
from stormlab.features.subcatchments import SubcatchmentFeatureError, assign_categories, encode_categories
from stormlab.hydraulics.round import MANNING_FULL_PIPE_COEFFICIENT, max_slope, max_velocity_value, min_slope
from stormlab.io import rpt
from stormlab.io.inp import (
    link_offsets_mode,
    load_model,
    offsets_as_depth,
    read_conduit_geometry,
    read_nodes,
    read_subcatchments,
    read_tags,
    subcatchment_raingages,
)

PHYSICAL_IDENTITY_COLUMNS: tuple[str, ...] = (
    "file_id",
    "file",
    "conduit",
    "inlet_node",
    "outlet_node",
    "rain_gage",
    "rain_series",
)
"""Which case each row is, next to the contract columns.

``sample_id`` is **not** here: it is the dataset layer's identifier
(``datasets/schema.sample_id``), which this layer may not import and which
``datasets/build.py`` already attaches to the historical feature frame after the
extraction returns. The node names are spelled as ``topology.parquet`` spells
them, because that is the frame the graph layer joins these rows against.
"""

RAINFALL_INTENSITY_FORMAT = "INTENSITY"
RAINFALL_TIMESERIES_SOURCE = "TIMESERIES"
"""The only rainfall description this extraction converts into scenario columns.

Both are what every generated file and every Wroclaw file declare. A gage given
as a volume, a cumulative depth or an external file describes the same storm in
units this code would have to convert differently, so it is refused rather than
summed as if it were an intensity.
"""

CATCHMENT_COLUMNS: tuple[str, ...] = (
    "subcatchment_area_ha",
    "subcatchment_imperv_pct",
    "subcatchment_width_m",
    "subcatchment_slope_pct",
)
"""The four columns describing the cumulative catchment above a conduit, in write order."""

NULLABLE_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "inlet_ground_m": ("inlet_ground_missing",),
        "inlet_pipe_depth_m": ("inlet_ground_missing",),
        "inlet_cover_m": ("inlet_ground_missing",),
        "inlet_node_max_depth_m": ("inlet_ground_missing",),
        "outlet_ground_m": ("outlet_ground_missing",),
        "outlet_pipe_depth_m": ("outlet_ground_missing",),
        "outlet_cover_m": ("outlet_ground_missing",),
        "outlet_node_max_depth_m": ("outlet_ground_missing",),
        "inlet_node_max_depth_reported_m": ("inlet_node_depth_missing",),
        "inlet_node_max_hgl_m": ("inlet_node_depth_missing",),
        "outlet_node_max_depth_reported_m": ("outlet_node_depth_missing",),
        "outlet_node_max_hgl_m": ("outlet_node_depth_missing",),
        "inlet_node_freeboard_m": ("inlet_ground_missing", "inlet_node_depth_missing"),
        "outlet_node_freeboard_m": ("outlet_ground_missing", "outlet_node_depth_missing"),
        "full_flow_cms": ("slope_nonpositive",),
        "flow_capacity_ratio": ("slope_nonpositive",),
        "min_slope_required": ("slope_requirement_missing", "min_slope_missing"),
        "max_slope_allowed": ("slope_requirement_missing",),
        "sbc_category_code": ("sbc_missing",),
        "subcatchment_area_ha": ("subcatchment_missing",),
        "subcatchment_imperv_pct": ("subcatchment_missing",),
        "subcatchment_width_m": ("subcatchment_missing",),
        "subcatchment_slope_pct": ("subcatchment_missing",),
        "outfall_fixed_stage_m": ("outfall_stage_missing",),
        "m_cover_inlet": ("inlet_ground_missing",),
        "m_cover_outlet": ("outlet_ground_missing",),
        "m_depth_inlet": ("inlet_ground_missing",),
        "m_depth_outlet": ("outlet_ground_missing",),
        "r_cover_inlet": ("inlet_ground_missing", "range_contradictory"),
        "r_cover_outlet": ("outlet_ground_missing", "range_contradictory"),
        "m_slope_min": ("slope_requirement_missing", "min_slope_missing"),
        "m_slope_max": ("slope_requirement_missing",),
        "m_capacity": ("slope_nonpositive",),
    }
)
"""Which columns may be missing, and which mask has to say so.

A value may be absent only where the row also declares *why*. Without this pair
"not measured" and "not computed" would both arrive as a bare ``NaN``, and the
view whose whole point is to keep absence distinguishable from zero would have
lost the distinction one layer further on.
"""


class PhysicalFeatureError(ValueError):
    """Raised when a case cannot be turned into physical features."""


def extract_physical_features(inp_path: Path | str, rpt_path: Path | str, *, config: Config, file_id: str) -> pd.DataFrame:
    """One row per conduit: identity, the F2 measurements, the F3 margins and the masks.

    Args:
        inp_path: the INP the case was simulated from. Read, never simulated:
            running SWMM is :mod:`stormlab.io.swmm`'s business and happens on a
            working copy before this is called.
        rpt_path: the report of that simulation.
        config: the run's configuration; every requirement (cover, frost zone,
            maximum depth, filling ratio, self-cleansing velocity, the slope
            constants) is read from it rather than from a literal here.
        file_id: the identifier the dataset layer knows this file by.

    Returns:
        A frame indexed 0..n-1 carrying :data:`PHYSICAL_IDENTITY_COLUMNS`
        followed by the columns of ``v4_margins``, which contains ``v3_physical``
        and ``v4_physical`` as prefixes and every other physical contract as a
        subset -- so one extraction serves a C1 run and a C2 one, and a caller
        picks its generation with :func:`physical_contract_frame`. Values are not
        clipped and not repaired.

    Raises:
        PhysicalFeatureError: if the case is not extractable -- a conduit absent
            from the report, a zero-length conduit, a rainfall description this
            module does not convert, a land cover class outside the canonical
            order.
        InpReadError, RptParseError: propagated from the readers.
    """
    inp = Path(inp_path)
    tables = rpt.read_report(rpt_path)
    model = load_model(inp, include_rpt=False)

    geometry = read_conduit_geometry(model)
    nodes = read_nodes(model, include_rpt=False)
    subcatchments = read_subcatchments(model, include_rpt=False)

    conduits = [str(name) for name in geometry.index]
    inlet_nodes = geometry["InletNode"].astype(str)
    outlet_nodes = geometry["OutletNode"].astype(str)

    frame = pd.DataFrame(index=pd.RangeIndex(len(conduits)))
    frame["file_id"] = str(file_id)
    frame["file"] = inp.name
    frame["conduit"] = conduits
    frame["inlet_node"] = inlet_nodes.to_numpy()
    frame["outlet_node"] = outlet_nodes.to_numpy()

    scenario = _rainfall_scenario(model)
    frame["rain_gage"] = scenario.gage
    frame["rain_series"] = scenario.series

    _add_geometry(frame, geometry, nodes, model=model)
    _add_report(frame, tables, conduits=conduits, inlet_nodes=inlet_nodes, outlet_nodes=outlet_nodes, source=rpt_path)
    _add_freeboard(frame)
    _add_capacity(frame)
    _add_slope_requirements(frame, config=config)
    _add_context(frame, geometry=geometry, nodes=nodes, subcatchments=subcatchments, model=model, config=config)
    frame["rain_total_mm"] = scenario.total_mm
    frame["rain_duration_h"] = scenario.duration_h
    frame["rain_peak_intensity_mmph"] = scenario.peak_intensity_mmph
    _add_requirements(frame, config=config)
    _add_receiver(frame, nodes=nodes, outlet_nodes=outlet_nodes)
    _add_margins(frame)

    ordered = list(PHYSICAL_IDENTITY_COLUMNS) + list(FEATURE_CONTRACT_V4_MARGINS.columns)
    missing = [column for column in ordered if column not in frame.columns]
    if missing:
        raise PhysicalFeatureError(f"Physical extraction of {inp} produced no {missing}")
    result = frame.loc[:, ordered].copy()
    for column, dtype in FEATURE_CONTRACT_V4_MARGINS.dtypes:
        result[column] = result[column].astype(dtype)
    validate_physical_features(result, contract=FEATURE_CONTRACT_V4_MARGINS)
    return result


def physical_contract_frame(
    df: pd.DataFrame,
    contract: FeatureContract = FEATURE_CONTRACT_V3_PHYSICAL,
) -> pd.DataFrame:
    """Project a physical frame onto one of the physical contracts.

    The counterpart of :func:`stormlab.features.conduits.conduit_feature_matrix`
    for the physical view. It cannot reuse that function, or the
    :func:`stormlab.contracts.features.validate_features` behind it, because
    those refuse every missing value -- correctly for v1/v2, where a ``NaN``
    could only be a hole, and wrongly here, where an absent measurement is a
    declared state carried by a mask.

    Raises:
        FeatureContractError: if a contract column is absent or cannot be cast.
        PhysicalFeatureError: if a value is missing where no mask accounts for it.
    """
    missing = [column for column in contract.columns if column not in df.columns]
    if missing:
        raise FeatureContractError(
            f"Cannot build the {contract.name} matrix: {len(missing)} contract column(s) missing: {missing}"
        )

    matrix = df.loc[:, list(contract.columns)].copy()
    for column in contract.columns:
        target = contract.dtypes_map[column]
        try:
            matrix[column] = matrix[column].astype(target)
        except (TypeError, ValueError) as error:
            raise FeatureContractError(
                f"Column {column!r} cannot be cast to {target}: {type(error).__name__}: {error}"
            ) from error

    validate_physical_features(matrix, contract=contract)
    return matrix


def validate_physical_features(df: pd.DataFrame, *, contract: FeatureContract = FEATURE_CONTRACT_V3_PHYSICAL) -> None:
    """Check presence, order, dtypes and that every missing value is declared.

    Raises:
        FeatureContractError: on a missing column, an out-of-contract order or a
            wrong dtype.
        PhysicalFeatureError: on a missing value in a column that may not have
            one, or on one whose mask does not say the value is absent.
    """
    if not isinstance(df, pd.DataFrame):
        raise FeatureContractError(f"Expected a DataFrame, got {type(df).__name__}")

    required = list(contract.columns)
    absent = [column for column in required if column not in df.columns]
    if absent:
        raise FeatureContractError(f"Missing feature columns ({len(absent)}): {absent}")

    present = [column for column in df.columns if column in set(required)]
    if present != required:
        raise FeatureContractError(f"Feature columns are out of contract order; expected {required}, got {present}")

    wanted = contract.dtypes_map
    wrong = {column: str(df[column].dtype) for column in required if str(df[column].dtype) != wanted[column]}
    if wrong:
        raise FeatureContractError(f"Wrong dtypes: {wrong} (expected {[(c, wanted[c]) for c in wrong]})")

    for column in required:
        holes = df[column].isna()
        if not holes.any():
            continue
        masks = NULLABLE_COLUMNS.get(column, ())
        if not masks:
            raise PhysicalFeatureError(
                f"Column {column!r} has {int(holes.sum())} missing value(s) but declares no mask that could explain them"
            )
        absent = [mask for mask in masks if mask not in df.columns]
        if absent:
            raise PhysicalFeatureError(
                f"Column {column!r} has missing values and its mask(s) {absent} are not in the frame, "
                "so nothing can say whether the absence was declared"
            )
        declared = np.zeros(len(df), dtype=bool)
        for mask in masks:
            declared |= df[mask].to_numpy().astype(bool)
        undeclared = holes.to_numpy() & ~declared
        if undeclared.any():
            positions = np.flatnonzero(undeclared)[:5].tolist()
            raise PhysicalFeatureError(
                f"Column {column!r} is missing at rows {positions} while none of {list(masks)} marks the row as missing"
            )


def _add_geometry(frame: pd.DataFrame, geometry: pd.DataFrame, nodes: pd.DataFrame, *, model: sw.Model) -> None:
    """Diameter, offsets, pipe inverts, ground, depth and cover at both ends."""
    inlet_nodes = geometry["InletNode"].astype(str)
    outlet_nodes = geometry["OutletNode"].astype(str)
    node_invert = _node_column(nodes, "InvertElev")
    node_depth = _node_column(nodes, "MaxDepth")

    diameter = geometry["Geom1"].to_numpy(dtype="float64")
    length = geometry["Length"].to_numpy(dtype="float64")
    if not np.all(length > 0):
        bad = [conduit for conduit, value in zip(geometry.index, length) if value <= 0]
        raise PhysicalFeatureError(f"Conduits {bad[:5]} have a non-positive length; a slope cannot be computed for them")
    if not np.all(diameter > 0):
        bad = [conduit for conduit, value in zip(geometry.index, diameter) if value <= 0]
        raise PhysicalFeatureError(f"Conduits {bad[:5]} have a non-positive diameter")

    mode = link_offsets_mode(model)
    inlet_invert = inlet_nodes.map(node_invert)
    outlet_invert = outlet_nodes.map(node_invert)
    if inlet_invert.isna().any() or outlet_invert.isna().any():
        unknown = sorted(set(inlet_nodes[inlet_invert.isna()]) | set(outlet_nodes[outlet_invert.isna()]))
        raise PhysicalFeatureError(f"Nodes {unknown[:5]} carry no invert elevation, so the pipe inverts are unknown")

    in_offset = offsets_as_depth(geometry["InOffset"], inlet_invert, mode=mode).to_numpy()
    out_offset = offsets_as_depth(geometry["OutOffset"], outlet_invert, mode=mode).to_numpy()
    inlet_pipe_invert = inlet_invert.to_numpy() + in_offset
    outlet_pipe_invert = outlet_invert.to_numpy() + out_offset
    slope_signed = (inlet_pipe_invert - outlet_pipe_invert) / length

    inlet_manhole = inlet_nodes.map(node_depth).to_numpy(dtype="float64")
    outlet_manhole = outlet_nodes.map(node_depth).to_numpy(dtype="float64")
    inlet_ground_missing = np.isnan(inlet_manhole)
    outlet_ground_missing = np.isnan(outlet_manhole)

    # An outfall has no manhole depth, so it has no ground elevation. The
    # historical path (`ConduitFeatureSteps.calculate_max_depth`) invents one as
    # inlet depth *minus* the pipe fall, which on a steep, long conduit comes out
    # metres below the invert (author's finding of 2026-09-17: cover of -23 m on
    # 1154 conduits of the C3b pool). The physical view does not repeat that: the
    # four outlet-end quantities stay missing where the depth is missing, the
    # mask `outlet_ground_missing` says so, and the imputation of the model
    # inputs (`models/imputation.py`) fills them like every other declared gap.
    # The historical view is left as it was, so the parity gate still holds.
    inlet_ground = inlet_invert.to_numpy() + inlet_manhole
    outlet_ground = np.where(outlet_ground_missing, np.nan, outlet_invert.to_numpy() + outlet_manhole)

    frame["diameter_m"] = diameter
    frame["length_m"] = length
    frame["roughness"] = geometry["Roughness"].to_numpy(dtype="float64")
    frame["barrels"] = geometry["Barrels"].to_numpy(dtype="float64")
    frame["in_offset_m"] = in_offset
    frame["out_offset_m"] = out_offset
    frame["inlet_pipe_invert_m"] = inlet_pipe_invert
    frame["outlet_pipe_invert_m"] = outlet_pipe_invert
    frame["slope_signed"] = slope_signed
    frame["inlet_ground_m"] = inlet_ground
    frame["outlet_ground_m"] = outlet_ground
    frame["inlet_pipe_depth_m"] = inlet_ground - inlet_pipe_invert
    frame["outlet_pipe_depth_m"] = outlet_ground - outlet_pipe_invert
    frame["inlet_cover_m"] = inlet_ground - inlet_pipe_invert - diameter
    frame["outlet_cover_m"] = outlet_ground - outlet_pipe_invert - diameter
    frame["inlet_node_max_depth_m"] = inlet_manhole
    frame["outlet_node_max_depth_m"] = outlet_manhole
    frame["inlet_ground_missing"] = inlet_ground_missing.astype("int8")
    frame["outlet_ground_missing"] = outlet_ground_missing.astype("int8")
    frame["slope_nonpositive"] = (slope_signed <= 0).astype("int8")


def _add_report(
    frame: pd.DataFrame,
    tables: rpt.ReportTables,
    *,
    conduits: Sequence[str],
    inlet_nodes: pd.Series,
    outlet_nodes: pd.Series,
    source: Path | str,
) -> None:
    """The simulated state: link flow, surcharge hours, node depths and flooding."""
    flow = tables.link_flow.reindex(index=list(conduits))
    unreported = [conduit for conduit, missing in zip(conduits, flow.isna().any(axis=1)) if missing]
    if unreported:
        raise PhysicalFeatureError(
            f"{len(unreported)} conduit(s) are absent from the Link Flow Summary of {source}: {unreported[:5]}; "
            "the report does not cover the network it was produced from"
        )
    for column in rpt.LINK_FLOW_COLUMNS:
        frame[column] = flow[column].to_numpy(dtype="float64")

    surcharge = rpt.zero_filled(tables.conduit_surcharge, conduits, rpt.CONDUIT_SURCHARGE_COLUMNS)
    for column in rpt.CONDUIT_SURCHARGE_COLUMNS:
        frame[column] = surcharge[column].to_numpy(dtype="float64")

    for end, node_names in (("inlet", inlet_nodes), ("outlet", outlet_nodes)):
        depths = tables.node_depth.reindex(index=list(node_names))
        frame[f"{end}_node_max_depth_reported_m"] = depths["max_depth_reported_m"].to_numpy(dtype="float64")
        frame[f"{end}_node_max_hgl_m"] = depths["max_hgl_m"].to_numpy(dtype="float64")
        frame[f"{end}_node_depth_missing"] = depths["max_depth_reported_m"].isna().to_numpy().astype("int8")

        flooding = rpt.zero_filled(tables.node_flooding, list(node_names), rpt.NODE_FLOODING_COLUMNS)
        frame[f"{end}_node_flood_hours"] = flooding["hours_flooded"].to_numpy(dtype="float64")
        frame[f"{end}_node_flood_volume_1e6l"] = flooding["total_flood_volume_1e6l"].to_numpy(dtype="float64")


def _add_freeboard(frame: pd.DataFrame) -> None:
    """How far the highest water level at each end stayed below the terrain.

    Arithmetic on two columns the frame already carries rather than a fifth table
    read out of the report, and stated as a column rather than left to every
    consumer because it is the premise a whole family of label sources rests on:
    under a dynamic wave a node whose head reaches the ground surcharges out of it,
    and the sign of this number is what says so.

    Missing wherever either term is missing -- an outlet whose ground is inferred
    from nothing, or a node the report never listed -- and both masks are already
    on the row, which is why the column needs no new one.
    """
    for end in ("inlet", "outlet"):
        frame[f"{end}_node_freeboard_m"] = frame[f"{end}_ground_m"].to_numpy(dtype="float64") - frame[
            f"{end}_node_max_hgl_m"
        ].to_numpy(dtype="float64")


def _add_capacity(frame: pd.DataFrame) -> None:
    """Full-pipe capacity and the ratio of the peak flow to it.

    The capacity is the denominator of the ``manning_v2`` flow law -- the
    corrected law variant B4 and the C1 profile run under -- so the two describe
    the same pipe. It is computed only where the slope is positive: ``sqrt`` of a
    negative fall is not a smaller capacity, it is no capacity, and a flat
    conduit divided into would produce an infinity that later reads as an
    enormous overload.

    It is the capacity of **one barrel**, exactly as the historical ``NMaxQ`` is,
    while ``max_flow_cms`` from the report is the flow through all of them. The
    two agree wherever ``barrels`` is 1, which is every conduit of the B4 pool
    (33 040 of 33 040) but not every conduit of Wroclaw, where ``K58`` and ``K60``
    carry two barrels and ``flow_capacity_ratio`` therefore reads about twice the
    load one pipe sees. The column keeps the historical meaning; ``barrels`` rides
    next to it so a consumer can tell which rows that meaning covers.
    """
    slope = frame["slope_signed"].to_numpy()
    roughness = frame["roughness"].to_numpy()
    diameter = frame["diameter_m"].to_numpy()
    if not np.all(roughness > 0):
        raise PhysicalFeatureError("A conduit declares a non-positive roughness; the Manning capacity is undefined for it")

    usable = slope > 0
    full_flow = np.full(len(frame), np.nan)
    full_flow[usable] = MANNING_FULL_PIPE_COEFFICIENT / roughness[usable] * diameter[usable] ** (8 / 3) * np.sqrt(slope[usable])
    ratio = np.full(len(frame), np.nan)
    ratio[usable] = frame["max_flow_cms"].to_numpy()[usable] / full_flow[usable]

    frame["full_flow_cms"] = full_flow
    frame["flow_capacity_ratio"] = ratio


def _add_slope_requirements(frame: pd.DataFrame, *, config: Config) -> None:
    """``min_slope_required`` and ``max_slope_allowed``, in m/m, each failing on its own.

    Both come from the very functions the historical ``MinRequiredSlope`` and
    ``MaxAllowableSlope`` come from, under this run's ``theta``/``g``/Manning, so
    the requirement means the same thing in both views. The stored unit is
    **m/m**, not the per mille those functions return, because the margins divide
    them by ``slope_signed``; a per-mille requirement next to an m/m slope would
    read as a factor of a thousand.

    The two requirements depend on different things, so they are computed apart:

    * the maximum slope is the one at which a **full** pipe still runs at the
      admissible velocity, a function of the diameter alone. It stays valid
      whatever the simulated filling was, and is memoised per diameter (the
      bisection behind it costs far more than the lookup);
    * the minimum slope is the self-cleansing criterion, which needs a filling.
      A filling at or above the diameter is evaluated at the **full section**
      -- ``check_dimensions`` refuses anything above the diameter, and the
      criterion at a filling the pipe cannot physically hold is not a stricter
      requirement, it is an undefined one.

    A conduit that carried no flow leaves ``min_slope_required`` absent and sets
    ``min_slope_missing``. :func:`stormlab.hydraulics.round.min_slope` would
    answer for it -- it clamps an empty pipe to a 1 mm filling -- but the answer
    is an artefact of that clamp (roughly 0.23 m/m, two orders above any laid
    sewer), not a requirement anything was designed against. A diameter outside
    the tabulated range leaves both columns absent and sets
    ``slope_requirement_missing``; one such conduit must not abort the extraction
    of the whole network.
    """
    theta = config.hydraulics.theta
    g = config.hydraulics.g
    manning = config.hydraulics.manning_coefficient
    diameter = frame["diameter_m"].to_numpy()
    filling = frame["max_full_depth_ratio"].to_numpy() * diameter

    maximum_by_diameter: dict[float, float] = {}
    minimum = np.full(len(frame), np.nan)
    maximum = np.full(len(frame), np.nan)
    for position in range(len(frame)):
        size = float(diameter[position])
        try:
            if size not in maximum_by_diameter:
                maximum_by_diameter[size] = max_slope(size, theta=theta, g=g, manning=manning) / 1000.0
        except (TypeError, ValueError):
            # `check_dimensions` refuses a diameter the criteria were never
            # tabulated for. That is a fact about this conduit, recorded as one,
            # and it takes the minimum slope down with it: the same check guards
            # both functions.
            continue
        maximum[position] = maximum_by_diameter[size]

        depth = float(filling[position])
        if not depth > 0.0:
            continue
        minimum[position] = min_slope(min(depth, size), size, theta=theta, g=g) / 1000.0

    frame["min_slope_required"] = minimum
    frame["max_slope_allowed"] = maximum
    frame["slope_requirement_missing"] = np.isnan(maximum).astype("int8")
    frame["min_slope_missing"] = np.isnan(minimum).astype("int8")


def _add_context(
    frame: pd.DataFrame,
    *,
    geometry: pd.DataFrame,
    nodes: pd.DataFrame,
    subcatchments: pd.DataFrame,
    model: sw.Model,
    config: Config,
) -> None:
    """Land cover class of the conduit and the whole catchment draining into it from above."""
    tagged = assign_categories(subcatchments, read_tags(model))
    with_subcatchments = assign_subcatchment_info(nodes, tagged)

    # The class a conduit inherits has to be the *same* class the historical view
    # gives it, otherwise the two views would disagree about the same network, so
    # the two steps that compute it are called rather than reimplemented. They
    # need a frost zone to construct, and never read it.
    steps = ConduitFeatureSteps(
        geometry.loc[:, ["InletNode", "OutletNode"]].copy(),
        with_subcatchments.copy(),
        config.frost_zone_dataset,
    )
    steps.conduits_subcatchment_info()
    steps.propagate_subcatchment_info()
    categories = steps.dfc["SbcCategory"]

    try:
        encoded = encode_categories(categories, FEATURE_ONEHOT_ORDER, what="conduit")
    except SubcatchmentFeatureError as error:
        raise PhysicalFeatureError(f"Cannot code SbcCategory of {frame['file'].iloc[0]}: {error}") from error
    known = encoded.to_numpy().sum(axis=1) == 1
    code = np.full(len(frame), np.nan)
    code[known] = encoded.to_numpy().argmax(axis=1)[known]
    frame["sbc_category_code"] = code
    frame["sbc_missing"] = (~known).astype("int8")

    cumulative = _upstream_catchment(_catchment_source(tagged), _upstream_nodes(geometry))
    inlet_nodes = geometry["InletNode"].astype(str)
    for column in CATCHMENT_COLUMNS:
        frame[column] = inlet_nodes.map(cumulative[column]).to_numpy(dtype="float64")
    # A catchment whose area the file does not state is as unusable as no
    # catchment at all: every aggregate of it is missing, so the mask has to say
    # so rather than leave four holes no column accounts for. With the cumulative
    # definition the mask is set only where *nothing* above the conduit states an
    # area -- a conduit at the top of a run with no catchment of its own inherits
    # the catchments of the branches feeding it, and is not missing.
    frame["subcatchment_missing"] = (~np.isfinite(frame["subcatchment_area_ha"].to_numpy())).astype("int8")


def _add_requirements(frame: pd.DataFrame, *, config: Config) -> None:
    """The design rules in force, one constant column each.

    They are read from the configuration and written into the row so that a
    margin can be recomputed from the row alone. The dataset frost zone is the
    one that applies: this view is a dataset artifact, and the serving frost zone
    belongs to the historical scoring path in ``experiments/e_wroclaw.py``.
    """
    frame["required_cover_m"] = resolve_required_cover(config.frost_zone_dataset, config.required_cover)
    frame["frost_zone_m"] = config.frost_zone_dataset
    frame["max_depth_m"] = config.max_depth
    frame["max_filling_ratio"] = config.max_filling_ratio
    frame["v_min_mps"] = config.hydraulics.v_min
    frame["v_max_mps"] = max_velocity_value


def _add_receiver(frame: pd.DataFrame, *, nodes: pd.DataFrame, outlet_nodes: pd.Series) -> None:
    """Whether the conduit discharges into an outfall, and at what stage if fixed."""
    types = _node_column(nodes, "OutfallType", required=False)
    stages = _node_column(nodes, "StageOrTimeseries", required=False)
    outlet_type = outlet_nodes.map(types).astype("object") if types is not None else pd.Series([None] * len(frame))
    is_outfall = outlet_type.notna().to_numpy()

    stage = np.full(len(frame), np.nan)
    if stages is not None:
        fixed = outlet_type.astype(str).str.upper() == "FIXED"
        values = pd.to_numeric(outlet_nodes.map(stages), errors="coerce").to_numpy(dtype="float64")
        stage = np.where(fixed.to_numpy(), values, np.nan)

    frame["outfall_fixed_stage_m"] = stage
    frame["is_outfall_conduit"] = is_outfall.astype("int8")
    frame["outfall_stage_missing"] = np.isnan(stage).astype("int8")


def _add_margins(frame: pd.DataFrame) -> None:
    """Relative distance to every requirement, negative where the requirement is broken.

    Nothing is clipped: two conduits short of cover by 0.1 m and by 0.9 m have to
    stay two different rows, which is exactly what a saturated ``[0, 1]``
    normalisation destroys (plan16 §2 P3). ``r_C`` is reported next to ``m_C``
    rather than instead of it, because it compresses the whole deficit range into
    roughly 1.04-1.14 and would flatten those two rows into one.
    """
    required_cover = frame["required_cover_m"].to_numpy()
    max_depth = frame["max_depth_m"].to_numpy()
    diameter = frame["diameter_m"].to_numpy()
    max_cover = max_depth - diameter
    band = max_cover - required_cover
    contradictory = band <= 0

    for end in ("inlet", "outlet"):
        cover = frame[f"{end}_cover_m"].to_numpy()
        depth = frame[f"{end}_pipe_depth_m"].to_numpy()
        frame[f"m_cover_{end}"] = (cover - required_cover) / required_cover
        frame[f"m_depth_{end}"] = (max_depth - depth) / max_depth
        ratio = np.full(len(frame), np.nan)
        usable = ~contradictory
        ratio[usable] = (max_cover[usable] - cover[usable]) / band[usable]
        frame[f"r_cover_{end}"] = ratio

    max_filling_ratio = frame["max_filling_ratio"].to_numpy()
    velocity = frame["max_velocity_mps"].to_numpy()
    slope = frame["slope_signed"].to_numpy()
    frame["m_filling"] = (max_filling_ratio - frame["max_full_depth_ratio"].to_numpy()) / max_filling_ratio
    frame["m_velocity_min"] = (velocity - frame["v_min_mps"].to_numpy()) / frame["v_min_mps"].to_numpy()
    frame["m_velocity_max"] = (frame["v_max_mps"].to_numpy() - velocity) / frame["v_max_mps"].to_numpy()
    frame["m_slope_min"] = (slope - frame["min_slope_required"].to_numpy()) / frame["min_slope_required"].to_numpy()
    frame["m_slope_max"] = (frame["max_slope_allowed"].to_numpy() - slope) / frame["max_slope_allowed"].to_numpy()
    frame["m_capacity"] = 1.0 - frame["flow_capacity_ratio"].to_numpy()
    frame["range_contradictory"] = contradictory.astype("int8")


@dataclass(frozen=True)
class _RainfallScenario:
    """The storm one file was driven by, as far as the INP states it."""

    gage: str
    series: str
    total_mm: float
    duration_h: float
    peak_intensity_mmph: float


def _rainfall_scenario(model: sw.Model) -> _RainfallScenario:
    """Identify the storm and measure it, rather than coding its name.

    The scenario is a property of the file: ``[SUBCATCHMENTS]`` names the gage,
    the gage names a time series, and the series is the storm. Turning the name
    into an arbitrary integer would give the model a code whose meaning depends
    on which files happened to be in the pool; the depth, the duration and the
    peak intensity mean the same thing in every file, so those are what the
    contract carries and the two names travel as identity columns.

    Raises:
        PhysicalFeatureError: if the subcatchments are driven by more than one
            gage, or the gage is not an intensity time series.
    """
    gages = subcatchment_raingages(model)
    used = sorted({str(name) for name in gages})
    if len(used) != 1:
        raise PhysicalFeatureError(
            f"{model.inp.path} drives its subcatchments from {len(used)} raingages {used[:5]}; "
            "the rainfall scenario of a case is a single storm"
        )
    gage = used[0]

    raingages = getattr(model.inp, "raingages", None)
    if raingages is None or gage not in raingages.index:
        raise PhysicalFeatureError(f"{model.inp.path} uses raingage {gage!r}, which its [RAINGAGES] section does not declare")
    row = raingages.loc[gage]
    rain_format = str(row.get("RainType", "")).strip().upper()
    data_source = str(row.get("DataSource", "")).strip().upper()
    if rain_format != RAINFALL_INTENSITY_FORMAT or data_source != RAINFALL_TIMESERIES_SOURCE:
        raise PhysicalFeatureError(
            f"Raingage {gage!r} of {model.inp.path} is {rain_format!r}/{data_source!r}; this extraction converts only "
            f"{RAINFALL_INTENSITY_FORMAT}/{RAINFALL_TIMESERIES_SOURCE} gages into scenario columns"
        )

    series_name = str(row["DataSourceName"]).strip()
    timeseries = getattr(model.inp, "timeseries", None)
    if timeseries is None or series_name not in timeseries.index:
        raise PhysicalFeatureError(f"{model.inp.path} has no [TIMESERIES] {series_name!r} behind raingage {gage!r}")

    series = timeseries.loc[[series_name]]
    values = pd.to_numeric(series["Value"], errors="coerce")
    if values.isna().any():
        raise PhysicalFeatureError(f"Time series {series_name!r} of {model.inp.path} carries a non-numeric intensity")
    hours = series["Time"].map(_hours)
    interval = _hours(str(row["TimeIntrvl"]))
    if interval <= 0:
        raise PhysicalFeatureError(f"Raingage {gage!r} of {model.inp.path} declares a non-positive interval")

    return _RainfallScenario(
        gage=gage,
        series=series_name,
        total_mm=float(values.sum()) * interval,
        duration_h=float(hours.max() - hours.min()),
        peak_intensity_mmph=float(values.max()),
    )


def _hours(value: str) -> float:
    """A SWMM ``H:MM`` or ``H:MM:SS`` duration in hours.

    Raises:
        PhysicalFeatureError: if the field is not a duration.
    """
    parts = str(value).strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise PhysicalFeatureError(f"{value!r} is not a SWMM duration")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as error:
        raise PhysicalFeatureError(f"{value!r} is not a SWMM duration: {error}") from error
    while len(numbers) < 3:
        numbers.append(0.0)
    return numbers[0] + numbers[1] / 60.0 + numbers[2] / 3600.0


def _catchment_source(subcatchments: pd.DataFrame) -> pd.DataFrame:
    """The four catchment parameters as numbers, keyed by the node each drains into.

    One row per subcatchment, not per node: the aggregation happens later over a
    whole upstream set, and averaging twice -- once per node, once over the nodes
    -- would weight a node with one large catchment the same as a node with five
    small ones.
    """
    if "Outlet" not in subcatchments.columns or len(subcatchments) == 0:
        return pd.DataFrame({name: pd.Series(dtype="float64") for name in ("area", "imperv", "width", "slope")}).assign(
            outlet=pd.Series(dtype="object")
        )
    return pd.DataFrame(
        {
            "outlet": subcatchments["Outlet"].astype(str).to_numpy(),
            "area": pd.to_numeric(subcatchments.get("Area"), errors="coerce").to_numpy(dtype="float64"),
            "imperv": pd.to_numeric(subcatchments.get("PercImperv"), errors="coerce").to_numpy(dtype="float64"),
            "width": pd.to_numeric(subcatchments.get("Width"), errors="coerce").to_numpy(dtype="float64"),
            "slope": pd.to_numeric(subcatchments.get("PercSlope"), errors="coerce").to_numpy(dtype="float64"),
        }
    )


def _upstream_nodes(geometry: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Every node that drains into each inlet node, the node itself included.

    A breadth-first walk up the conduit graph, over the edges the geometry states
    (``InletNode`` -> ``OutletNode``). Two properties it has to have and does: it
    visits each node once, so a node reachable along two branches of a converging
    topology is counted once rather than twice; and it is deterministic, because
    both the starting nodes and the feeders of each node are kept in the order the
    conduits appear in the file. A cycle -- which a drainage network does not have,
    and which a malformed file could still state -- terminates on the visited set
    instead of looping.
    """
    inlets = geometry["InletNode"].astype(str).to_numpy()
    outlets = geometry["OutletNode"].astype(str).to_numpy()
    feeders: dict[str, list[str]] = {}
    for inlet, outlet in zip(inlets, outlets):
        feeders.setdefault(str(outlet), []).append(str(inlet))

    above: dict[str, tuple[str, ...]] = {}
    for start in dict.fromkeys(str(node) for node in inlets):
        seen = {start}
        order = [start]
        queue = deque([start])
        while queue:
            for feeder in feeders.get(queue.popleft(), ()):
                if feeder not in seen:
                    seen.add(feeder)
                    order.append(feeder)
                    queue.append(feeder)
        above[start] = tuple(order)
    return above


def _upstream_catchment(source: pd.DataFrame, above: Mapping[str, tuple[str, ...]]) -> pd.DataFrame:
    """Cumulative catchment above each node: area summed, the rest area-weighted.

    The catchment a conduit drains is everything upstream of it, so the area is a
    sum over the whole upstream set and the intensive parameters are averaged over
    that sum -- a 0.01 ha strip must not outvote a 2 ha catchment, whichever node
    either of them hangs off.
    """
    by_outlet = {str(outlet): group for outlet, group in source.groupby("outlet", sort=False)}
    rows: dict[str, list[float]] = {}
    for node, upstream in above.items():
        parts = [by_outlet[name] for name in upstream if name in by_outlet]
        if not parts:
            rows[node] = [math.nan, math.nan, math.nan, math.nan]
            continue
        block = parts[0] if len(parts) == 1 else pd.concat(parts)
        weights = block["area"].to_numpy(dtype="float64")
        # `nansum` of nothing is 0.0, which would turn "the file states no area"
        # into "the catchment has none" -- and, with the intensive parameters
        # correctly missing beside it, into a row whose mask says nothing is
        # missing. An unstated area is an absence, so it is written as one.
        total = float(np.nansum(weights)) if np.isfinite(weights).any() else math.nan
        rows[node] = [
            total,
            _weighted(block["imperv"].to_numpy(dtype="float64"), weights),
            _weighted(block["width"].to_numpy(dtype="float64"), weights),
            _weighted(block["slope"].to_numpy(dtype="float64"), weights),
        ]
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(CATCHMENT_COLUMNS)).rename_axis("node")


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    """Area-weighted mean, falling back to the plain mean when every area is zero."""
    finite = np.isfinite(values) & np.isfinite(weights)
    if not finite.any():
        return math.nan
    total = float(weights[finite].sum())
    if total <= 0:
        return float(values[finite].mean())
    return float((values[finite] * weights[finite]).sum() / total)


def _node_column(nodes: pd.DataFrame, column: str, *, required: bool = True) -> pd.Series | None:
    """One column of the node frame, keyed by node name as a string.

    Raises:
        PhysicalFeatureError: if a required column is absent from the frame.
    """
    if column not in nodes.columns:
        if required:
            raise PhysicalFeatureError(f"Node frame has no {column!r} column; got {list(nodes.columns)}")
        return None
    series = nodes[column]
    return pd.Series(series.to_numpy(), index=pd.Index([str(name) for name in nodes.index]))


__all__ = [
    "NULLABLE_COLUMNS",
    "PHYSICAL_IDENTITY_COLUMNS",
    "PhysicalFeatureError",
    "extract_physical_features",
    "physical_contract_frame",
    "validate_physical_features",
]
