"""Where every model input comes from, declared rather than inferred.

This is the *taint allowlist* of plan14 §4.3 (f). It replaces the mutual-information
test that used to stand in for "no feature is made out of the label". That test
asked whether a column determines the label, which in this study is true of the
features by construction — the label *is* a deterministic function of them
(``labels/rules.py``) — so it passed vacuously on B1--B3 and would have started
failing on a richer grid for a reason that has nothing to do with leakage.

The question worth asking is the other one: **what was each column made out of?**
That is not measurable from the data; it is a fact about the extraction code, so
it is written down here, column by column, and held against the contract by
:mod:`stormlab.tests.contract.test_leakage`.

Vocabulary of a source, closed on purpose:

``inp_section:<SECTION>``
    A value read straight out of a section of the SWMM input file.
``swmm_report:<SECTION>``
    A value produced by the simulation and read back from the ``.rpt``, named by
    the summary section it was read from.
``hydraulics:<function>``
    A value computed by a closed-form hydraulic rule in
    :mod:`stormlab.hydraulics`, from constants and from other columns.
``config:<key>``
    A requirement read from the run's configuration -- the required cover, the
    maximum admissible depth -- rather than measured on the network. Its own root
    form because it is neither a measurement nor a hydraulic computation: it is a
    design rule the study chose, and a column that rests on one is answering
    partly with the study's own assumptions. plan16 §2 P3 asks for it by name, for
    the requirement margins of the F3 view. No v1/v2 column declares it; the
    vocabulary is widened here so the v3 contract can, without touching a single
    declaration the frozen variants were audited under.
``derived:<a>,<b>``
    A value computed from other columns named here, which are resolved onward
    until only the root forms above remain.

``labels:<...>`` exists in the vocabulary only as the thing that must never
appear: :func:`root_sources` resolves a contract column to its roots and
:data:`FORBIDDEN_SOURCE_PREFIX` is what the test refuses to find among them.

The intermediates are declared too. Without them ``derived:`` would be an
unrooted promise — "made from ``Filling``" says nothing until ``Filling`` itself
has a source — and the resolution would have nowhere to bottom out.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from stormlab.contracts.features import (
    FEATURE_CONTRACT_V1,
    FEATURE_CONTRACT_V2,
    FEATURE_CONTRACT_V2_NOFLAGS,
    FEATURE_CONTRACT_V3_MARGINS,
    FEATURE_CONTRACT_V3_PHYSICAL,
    FEATURE_CONTRACT_V3_PHYSICAL_NBR,
    FEATURE_CONTRACT_V3_PHYSICAL_NBR_NOCAT,
    FEATURE_CONTRACT_V3_PHYSICAL_NOCAT,
    FEATURE_CONTRACT_V4_MARGINS,
    FEATURE_CONTRACT_V4_NOFLOOD_MARGINS,
    FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL,
    FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL_NBR,
    FEATURE_CONTRACT_V4_PHYSICAL,
    FEATURE_CONTRACT_V4_PHYSICAL_NBR,
    FEATURE_CONTRACT_V4_PHYSICAL_NBR_NOCAT,
    FEATURE_CONTRACT_V4_PHYSICAL_NOCAT,
    FEATURE_ONEHOT_ORDER,
    NEIGHBORHOOD_CONTRACT_COLUMNS,
    NEIGHBORHOOD_COUNT_SUFFIX,
    NEIGHBORHOOD_PRESENT_SUFFIX,
    FeatureContract,
    neighborhood_column_names,
)

FORBIDDEN_SOURCE_PREFIX = "labels:"
"""No model input may resolve to this. The label is downstream of every feature."""

ROOT_SOURCE_PREFIXES: tuple[str, ...] = ("inp_section:", "swmm_report:", "hydraulics:", "config:")
"""Source forms that terminate the resolution; ``derived:`` is the only recursive one."""

DERIVED_PREFIX = "derived:"


class ProvenanceError(ValueError):
    """Raised when the declaration is incomplete, circular or names an unknown form."""


@dataclass(frozen=True)
class ColumnProvenance:
    """One column's declared origin.

    Attributes:
        column: the column named — a contract column or an intermediate.
        sources: its immediate origins, each in the vocabulary above.
        intended_onehot: ``True`` for the twelve land cover columns. They are the
            one place where a *categorical* value of the network becomes a model
            input, and plan14 §4.5 asks the allowlist to say so explicitly: the
            raw ``SbcCategory`` never enters ``X``, its encoding deliberately
            does, and a check that could not tell those apart would either ban
            the block or wave the raw column through.
        note: why this origin is what it is, where the formula is not obvious.
    """

    column: str
    sources: tuple[str, ...]
    intended_onehot: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not self.sources:
            raise ProvenanceError(f"Column {self.column!r} declares no source")
        for source in self.sources:
            if source.startswith(DERIVED_PREFIX):
                if not source[len(DERIVED_PREFIX) :].strip():
                    raise ProvenanceError(f"Column {self.column!r} declares {source!r} without naming a column")
            elif not source.startswith(ROOT_SOURCE_PREFIXES + (FORBIDDEN_SOURCE_PREFIX,)):
                raise ProvenanceError(
                    f"Column {self.column!r} declares source {source!r}, which is not one of "
                    f"{list(ROOT_SOURCE_PREFIXES)} or {DERIVED_PREFIX!r}"
                )


def _derived(*columns: str) -> str:
    return DERIVED_PREFIX + ",".join(columns)


_INTERMEDIATES: tuple[ColumnProvenance, ...] = (
    # --- straight out of the INP -------------------------------------------------
    ColumnProvenance("Geom1", ("inp_section:XSECTIONS",), note="Pipe diameter [m]."),
    ColumnProvenance("Length", ("inp_section:CONDUITS",)),
    ColumnProvenance("Roughness", ("inp_section:CONDUITS",), note="Manning's n as laid down in the file."),
    ColumnProvenance(
        "SlopeFtPerFt", ("inp_section:CONDUITS", "inp_section:JUNCTIONS"), note="Fall over length, from the inverts."
    ),
    ColumnProvenance("InletNode", ("inp_section:CONDUITS",)),
    ColumnProvenance("OutletNode", ("inp_section:CONDUITS",)),
    ColumnProvenance("InletNodeInvert", ("inp_section:JUNCTIONS",)),
    ColumnProvenance("OutletNodeInvert", ("inp_section:JUNCTIONS", "inp_section:OUTFALLS")),
    ColumnProvenance("InletMaxDepth", ("inp_section:JUNCTIONS",), note="Manhole depth of the inlet node."),
    ColumnProvenance(
        "OutletMaxDepth",
        ("inp_section:JUNCTIONS", "inp_section:OUTFALLS", _derived("InletMaxDepth", "Length", "SlopeFtPerFt")),
        note="Read from the node where it exists; inferred from the inlet depth and the fall where it does not.",
    ),
    ColumnProvenance(
        "SbcCategory",
        ("inp_section:TAGS", "inp_section:SUBCATCHMENTS"),
        note="Land cover class of the subcatchment draining to the conduit, tagged in the file and propagated along the network.",
    ),
    # --- read back from the simulation ------------------------------------------
    ColumnProvenance("MaxDPerc", ("swmm_report:Link Flow Summary",), note="Peak filling as a fraction of the diameter."),
    ColumnProvenance("MaxV", ("swmm_report:Link Flow Summary",)),
    ColumnProvenance("MaxQ", ("swmm_report:Link Flow Summary",)),
    # --- computed -----------------------------------------------------------------
    ColumnProvenance("Filling", (_derived("MaxDPerc", "Geom1"),), note="MaxDPerc * Geom1 [m]."),
    ColumnProvenance("SlopePerMile", (_derived("SlopeFtPerFt"),), note="SlopeFtPerFt * 1000."),
    ColumnProvenance("InletGroundElevation", (_derived("InletNodeInvert", "InletMaxDepth"),)),
    ColumnProvenance("OutletGroundElevation", (_derived("OutletNodeInvert", "OutletMaxDepth"),)),
    ColumnProvenance("InletGroundCover", (_derived("InletGroundElevation", "InletNodeInvert", "Geom1"),)),
    ColumnProvenance("OutletGroundCover", (_derived("OutletGroundElevation", "OutletNodeInvert", "Geom1"),)),
    ColumnProvenance(
        "MinRequiredSlope",
        ("hydraulics:round.min_slope", _derived("Filling", "Geom1")),
        note="Hydromechanical minimum slope; a rule, not an observation.",
    ),
    ColumnProvenance("MaxAllowableSlope", ("hydraulics:round.max_slope", _derived("Geom1"))),
    ColumnProvenance(
        "MinDiameter",
        ("hydraulics:round.calc_filling", _derived("MaxQ", "Geom1", "Filling", "SlopeFtPerFt", "Roughness")),
        note="Smallest standard diameter that still conveys MaxQ under the configured flow law.",
    ),
)

_CONTRACT_COLUMNS: tuple[ColumnProvenance, ...] = (
    # --- 7 hydraulic validity flags ------------------------------------------------
    ColumnProvenance("ValMaxFill", (_derived("Filling", "Geom1"),), note="Filling within max_filling_ratio * D."),
    ColumnProvenance("ValMaxV", (_derived("MaxV"),)),
    ColumnProvenance("ValMinV", (_derived("MaxV"),)),
    ColumnProvenance("ValMaxSlope", (_derived("SlopePerMile", "MaxAllowableSlope"),)),
    ColumnProvenance("ValMinSlope", (_derived("SlopePerMile", "MinRequiredSlope"),)),
    ColumnProvenance(
        "ValDepth",
        (_derived("InletNodeInvert", "InletGroundElevation", "OutletNodeInvert", "OutletGroundElevation"),),
        note="Neither end deeper than the admissible 8 m.",
    ),
    ColumnProvenance(
        "ValCoverage", (_derived("InletGroundCover", "OutletGroundCover"),), note="Both ends covered to the frost zone."
    ),
    # --- 5 decision flags ----------------------------------------------------------
    ColumnProvenance("isMinDiameter", (_derived("Geom1", "MinDiameter"),)),
    ColumnProvenance("IncreaseDia", (_derived("Geom1", "MinDiameter"),)),
    ColumnProvenance("ReduceDia", (_derived("Geom1", "MinDiameter"),)),
    ColumnProvenance("IncreaseSlope", (_derived("SlopePerMile", "MinRequiredSlope"),)),
    ColumnProvenance("ReduceSlope", (_derived("SlopePerMile", "MaxAllowableSlope"),)),
    # --- 9 (v1) / 7 (v2) normalised values -----------------------------------------
    ColumnProvenance("NRoughness", (_derived("Roughness"),), note="Affine over the configured PVC-to-stone range."),
    ColumnProvenance("NMaxV", (_derived("MaxV"),), note="Fraction of the admissible maximum velocity."),
    ColumnProvenance("NInletDepth", (_derived("InletMaxDepth", "Geom1"),), note="Over [frost + D, 8 m]."),
    ColumnProvenance("NOutletDepth", (_derived("OutletMaxDepth", "Geom1"),), note="Over [frost + D, 8 m]."),
    ColumnProvenance("NFilling", (_derived("Filling", "Geom1"),), note="Filling over max_filling_ratio * D."),
    ColumnProvenance(
        "NMaxQ",
        ("hydraulics:round.max_flow", _derived("MaxQ", "Geom1", "SlopeFtPerFt", "Roughness")),
        note="Observed flow over the theoretical full-pipe capacity of the configured flow law.",
    ),
    ColumnProvenance("NInletGroundCover", (_derived("InletGroundCover", "Geom1"),), note="v1 only; over [frost, 8 m - D]."),
    ColumnProvenance("NOutletGroundCover", (_derived("OutletGroundCover", "Geom1"),), note="v1 only; over [frost, 8 m - D]."),
    ColumnProvenance("NSlope", (_derived("SlopePerMile", "MinRequiredSlope", "MaxAllowableSlope"),)),
    # --- 12 land cover one-hots ----------------------------------------------------
    *(
        ColumnProvenance(
            name,
            (_derived("SbcCategory"),),
            intended_onehot=True,
            note="One of the twelve columns of the SbcCategory block: the encoding is a model input, the raw column is not.",
        )
        for name in FEATURE_ONEHOT_ORDER
    ),
)

_PHYSICAL_COLUMNS: tuple[ColumnProvenance, ...] = (
    # --- geometry, straight out of the INP -----------------------------------------
    ColumnProvenance("diameter_m", ("inp_section:XSECTIONS",), note="Geom1 of a circular section [m]."),
    ColumnProvenance("length_m", ("inp_section:CONDUITS",)),
    ColumnProvenance("roughness", ("inp_section:CONDUITS",)),
    ColumnProvenance("barrels", ("inp_section:XSECTIONS",), note="Parallel pipes; 1 where the file does not say."),
    ColumnProvenance(
        "in_offset_m",
        ("inp_section:CONDUITS", "inp_section:OPTIONS", "inp_section:JUNCTIONS", "inp_section:OUTFALLS"),
        note="The written offset expressed as a height above the node invert; LINK_OFFSETS says which it was.",
    ),
    ColumnProvenance(
        "out_offset_m",
        ("inp_section:CONDUITS", "inp_section:OPTIONS", "inp_section:JUNCTIONS", "inp_section:OUTFALLS"),
    ),
    ColumnProvenance(
        "inlet_pipe_invert_m",
        ("inp_section:JUNCTIONS", "inp_section:OUTFALLS", _derived("in_offset_m")),
        note="Node invert plus the offset: the invert of the pipe, not of the manhole.",
    ),
    ColumnProvenance(
        "outlet_pipe_invert_m",
        ("inp_section:JUNCTIONS", "inp_section:OUTFALLS", _derived("out_offset_m")),
    ),
    ColumnProvenance(
        "slope_signed",
        (_derived("inlet_pipe_invert_m", "outlet_pipe_invert_m", "length_m"),),
        note="Fall over length [m/m], sign kept: an adverse pipe is not a flat one.",
    ),
    ColumnProvenance("inlet_node_max_depth_m", ("inp_section:JUNCTIONS",), note="Manhole depth of the inlet node."),
    ColumnProvenance("outlet_node_max_depth_m", ("inp_section:JUNCTIONS", "inp_section:OUTFALLS")),
    ColumnProvenance("inlet_ground_m", ("inp_section:JUNCTIONS", _derived("inlet_node_max_depth_m"))),
    ColumnProvenance(
        "outlet_ground_m",
        (
            "inp_section:JUNCTIONS",
            "inp_section:OUTFALLS",
            _derived("outlet_node_max_depth_m", "inlet_node_max_depth_m", "length_m", "slope_signed"),
        ),
        note="Inferred from the inlet depth and the fall where the outlet node declares no depth, as v1/v2 do.",
    ),
    ColumnProvenance("inlet_pipe_depth_m", (_derived("inlet_ground_m", "inlet_pipe_invert_m"),)),
    ColumnProvenance("outlet_pipe_depth_m", (_derived("outlet_ground_m", "outlet_pipe_invert_m"),)),
    ColumnProvenance("inlet_cover_m", (_derived("inlet_pipe_depth_m", "diameter_m"),), note="Soil above the crown [m]."),
    ColumnProvenance("outlet_cover_m", (_derived("outlet_pipe_depth_m", "diameter_m"),)),
    # --- the simulated state, read back from the report ------------------------------
    ColumnProvenance("max_flow_cms", ("swmm_report:Link Flow Summary",)),
    ColumnProvenance("max_velocity_mps", ("swmm_report:Link Flow Summary",)),
    ColumnProvenance("max_full_flow_ratio", ("swmm_report:Link Flow Summary",), note="SWMM's own Max/Full Flow ratio."),
    ColumnProvenance("max_full_depth_ratio", ("swmm_report:Link Flow Summary",), note="Max/Full Depth; not clipped."),
    ColumnProvenance("hours_full_both_ends", ("swmm_report:Conduit Surcharge Summary",)),
    ColumnProvenance("hours_full_upstream", ("swmm_report:Conduit Surcharge Summary",)),
    ColumnProvenance("hours_full_downstream", ("swmm_report:Conduit Surcharge Summary",)),
    ColumnProvenance("hours_above_normal_flow", ("swmm_report:Conduit Surcharge Summary",)),
    ColumnProvenance("hours_capacity_limited", ("swmm_report:Conduit Surcharge Summary",)),
    ColumnProvenance("inlet_node_max_depth_reported_m", ("swmm_report:Node Depth Summary",)),
    ColumnProvenance("outlet_node_max_depth_reported_m", ("swmm_report:Node Depth Summary",)),
    ColumnProvenance("inlet_node_max_hgl_m", ("swmm_report:Node Depth Summary",)),
    ColumnProvenance("outlet_node_max_hgl_m", ("swmm_report:Node Depth Summary",)),
    ColumnProvenance("inlet_node_flood_hours", ("swmm_report:Node Flooding Summary",)),
    ColumnProvenance("inlet_node_flood_volume_1e6l", ("swmm_report:Node Flooding Summary",)),
    ColumnProvenance("outlet_node_flood_hours", ("swmm_report:Node Flooding Summary",)),
    ColumnProvenance("outlet_node_flood_volume_1e6l", ("swmm_report:Node Flooding Summary",)),
    ColumnProvenance(
        "full_flow_cms",
        ("hydraulics:round.MANNING_FULL_PIPE_COEFFICIENT", _derived("diameter_m", "roughness", "slope_signed")),
        note=(
            "Full-pipe Manning capacity of one barrel, the denominator the manning_v2 flow law uses; "
            "undefined at a non-positive slope."
        ),
    ),
    ColumnProvenance("flow_capacity_ratio", (_derived("max_flow_cms", "full_flow_cms"),)),
    ColumnProvenance(
        "inlet_node_freeboard_m",
        (_derived("inlet_ground_m", "inlet_node_max_hgl_m"),),
        note="Terrain minus the highest water level at the inlet node; negative is a head above the ground.",
    ),
    ColumnProvenance("outlet_node_freeboard_m", (_derived("outlet_ground_m", "outlet_node_max_hgl_m"),)),
    # --- catchment and rainfall scenario ---------------------------------------------
    ColumnProvenance(
        "sbc_category_code",
        ("inp_section:TAGS", "inp_section:SUBCATCHMENTS"),
        note="Index of the land cover class in FEATURE_ONEHOT_ORDER; a code of the category, not a decision.",
    ),
    # The cumulative inflowing catchment (author's decision of 2026-09-07): every
    # catchment above the conduit, not only the ones wired to its inlet node. The
    # conduit graph is therefore a source of these columns and is declared as one.
    ColumnProvenance(
        "subcatchment_area_ha",
        ("inp_section:SUBCATCHMENTS", "inp_section:CONDUITS"),
        note="Area of every catchment above the conduit, summed over the upstream node set.",
    ),
    ColumnProvenance(
        "subcatchment_imperv_pct",
        ("inp_section:SUBCATCHMENTS", "inp_section:CONDUITS"),
        note="Imperviousness of the catchment above the conduit, averaged over its area.",
    ),
    ColumnProvenance(
        "subcatchment_width_m",
        ("inp_section:SUBCATCHMENTS", "inp_section:CONDUITS"),
        note="Width of the catchment above the conduit, averaged over its area.",
    ),
    ColumnProvenance(
        "subcatchment_slope_pct",
        ("inp_section:SUBCATCHMENTS", "inp_section:CONDUITS"),
        note="Slope of the catchment above the conduit, averaged over its area.",
    ),
    ColumnProvenance(
        "rain_total_mm",
        ("inp_section:TIMESERIES", "inp_section:RAINGAGES", "inp_section:SUBCATCHMENTS"),
        note="Sum of the intensity series times the gage interval.",
    ),
    ColumnProvenance("rain_duration_h", ("inp_section:TIMESERIES", "inp_section:RAINGAGES", "inp_section:SUBCATCHMENTS")),
    ColumnProvenance(
        "rain_peak_intensity_mmph", ("inp_section:TIMESERIES", "inp_section:RAINGAGES", "inp_section:SUBCATCHMENTS")
    ),
    # --- the requirements in force ----------------------------------------------------
    ColumnProvenance("required_cover_m", ("config:required_cover", "config:frost_zone_dataset")),
    ColumnProvenance("frost_zone_m", ("config:frost_zone_dataset",)),
    ColumnProvenance("max_depth_m", ("config:max_depth",)),
    ColumnProvenance("max_filling_ratio", ("config:max_filling_ratio",)),
    ColumnProvenance("v_min_mps", ("config:hydraulics.v_min",)),
    ColumnProvenance(
        "v_max_mps",
        ("hydraulics:round.max_velocity",),
        note="Still a module constant: plan16 §2 P0 separated three requirements into configuration keys, not this one.",
    ),
    ColumnProvenance(
        "min_slope_required",
        ("hydraulics:round.min_slope", _derived("diameter_m", "max_full_depth_ratio")),
        note=(
            "The hydromechanical criterion of the historical MinRequiredSlope, converted from per mille to m/m; "
            "evaluated at the full section where the filling reaches the diameter, and absent for a conduit that "
            "carried no flow."
        ),
    ),
    ColumnProvenance(
        "max_slope_allowed",
        ("hydraulics:round.max_slope", _derived("diameter_m")),
        note="Depends on the diameter alone, so it survives a filling the minimum-slope criterion cannot use.",
    ),
    # --- receiver ---------------------------------------------------------------------
    ColumnProvenance("outfall_fixed_stage_m", ("inp_section:OUTFALLS",)),
    ColumnProvenance("is_outfall_conduit", ("inp_section:OUTFALLS", "inp_section:CONDUITS")),
    # --- masks: which values are absent rather than zero ------------------------------
    ColumnProvenance("inlet_ground_missing", ("inp_section:JUNCTIONS", "inp_section:OUTFALLS")),
    ColumnProvenance("outlet_ground_missing", ("inp_section:JUNCTIONS", "inp_section:OUTFALLS")),
    ColumnProvenance("slope_nonpositive", (_derived("slope_signed"),)),
    ColumnProvenance(
        "slope_requirement_missing",
        ("hydraulics:round.max_slope", _derived("diameter_m")),
        note="1 where the diameter is outside the tabulated range, so neither slope requirement is defined.",
    ),
    ColumnProvenance(
        "min_slope_missing",
        ("hydraulics:round.min_slope", _derived("diameter_m", "max_full_depth_ratio")),
        note="1 where min_slope_required is absent: an untabulated diameter, or a conduit with no measured filling.",
    ),
    ColumnProvenance("sbc_missing", ("inp_section:TAGS", "inp_section:SUBCATCHMENTS")),
    ColumnProvenance(
        "subcatchment_missing",
        ("inp_section:SUBCATCHMENTS", "inp_section:CONDUITS"),
        note="Set only where no catchment anywhere above the conduit states an area.",
    ),
    ColumnProvenance("outfall_stage_missing", ("inp_section:OUTFALLS",)),
    ColumnProvenance("inlet_node_depth_missing", ("swmm_report:Node Depth Summary",)),
    ColumnProvenance("outlet_node_depth_missing", ("swmm_report:Node Depth Summary",)),
)

_MARGIN_COLUMNS: tuple[ColumnProvenance, ...] = (
    ColumnProvenance("m_cover_inlet", (_derived("inlet_cover_m", "required_cover_m"),), note="(C - C_req) / C_req."),
    ColumnProvenance("m_cover_outlet", (_derived("outlet_cover_m", "required_cover_m"),)),
    ColumnProvenance("m_depth_inlet", (_derived("inlet_pipe_depth_m", "max_depth_m"),), note="(d_max - d) / d_max."),
    ColumnProvenance("m_depth_outlet", (_derived("outlet_pipe_depth_m", "max_depth_m"),)),
    ColumnProvenance(
        "r_cover_inlet",
        (_derived("inlet_cover_m", "required_cover_m", "max_depth_m", "diameter_m"),),
        note="Position in the admissible cover band; absent where the band is empty.",
    ),
    ColumnProvenance("r_cover_outlet", (_derived("outlet_cover_m", "required_cover_m", "max_depth_m", "diameter_m"),)),
    ColumnProvenance("m_filling", (_derived("max_full_depth_ratio", "max_filling_ratio"),)),
    ColumnProvenance("m_velocity_min", (_derived("max_velocity_mps", "v_min_mps"),)),
    ColumnProvenance("m_velocity_max", (_derived("max_velocity_mps", "v_max_mps"),)),
    ColumnProvenance("m_slope_min", (_derived("slope_signed", "min_slope_required"),)),
    ColumnProvenance("m_slope_max", (_derived("slope_signed", "max_slope_allowed"),)),
    ColumnProvenance("m_capacity", (_derived("flow_capacity_ratio"),)),
    ColumnProvenance("range_contradictory", (_derived("max_depth_m", "diameter_m", "required_cover_m"),)),
)


def _neighborhood_provenance() -> tuple[ColumnProvenance, ...]:
    """The aggregate columns of every declared neighbourhood, from the generator that names them.

    The one block written by a comprehension rather than by hand. Its column names
    are themselves generated from :data:`NEIGHBORHOOD_CONTRACT_COLUMNS` and
    :data:`NEIGHBORHOOD_HOPS`, so the literals would be a second copy of the
    generator -- and the failure mode that copy invites is exactly the one this
    file exists to prevent: a declaration that no longer describes the columns.

    Every registered neighbourhood is walked, not just C1's: the variants overlap on
    three of their four measurements, so the generated names are deduplicated and the
    column each statistic summarises is looked up in the **union** of the declared
    lists. A neighbourhood declared by one variant and undeclared here would be a
    model input nothing audits.

    A statistic rests on **two** things and says so: the value comes from the F2
    measurement it summarises, the membership from the hydraulic edges, which are
    the inlet and outlet node names of ``[CONDUITS]``. ``count`` and ``present``
    describe the graph alone.
    """
    edges = "inp_section:CONDUITS"
    measurements = tuple(dict.fromkeys(sum(NEIGHBORHOOD_CONTRACT_COLUMNS.values(), ())))
    declared: dict[str, ColumnProvenance] = {}
    for columns in NEIGHBORHOOD_CONTRACT_COLUMNS.values():
        for name in neighborhood_column_names(columns):
            if name in declared:
                continue
            prefix, _, tail = name.partition("_")
            if tail in (NEIGHBORHOOD_COUNT_SUFFIX, NEIGHBORHOOD_PRESENT_SUFFIX):
                declared[name] = ColumnProvenance(
                    name,
                    (edges,),
                    note=f"How many conduits the {prefix} neighbourhood holds, and whether it holds any.",
                )
                continue
            source = next(column for column in measurements if tail.startswith(f"{column}_"))
            declared[name] = ColumnProvenance(
                name,
                (_derived(source), edges),
                note=f"{tail[len(source) + 1 :]} of {source!r} over the {prefix} neighbourhood.",
            )
    return tuple(declared.values())


_NEIGHBORHOOD_COLUMNS: tuple[ColumnProvenance, ...] = _neighborhood_provenance()


PROVENANCE: Mapping[str, ColumnProvenance] = MappingProxyType(
    {
        item.column: item
        for item in _INTERMEDIATES + _CONTRACT_COLUMNS + _PHYSICAL_COLUMNS + _MARGIN_COLUMNS + _NEIGHBORHOOD_COLUMNS
    }
)
"""Every declared column, contract and intermediate alike, keyed by name."""

if len(PROVENANCE) != len(_INTERMEDIATES) + len(_CONTRACT_COLUMNS) + len(_PHYSICAL_COLUMNS) + len(_MARGIN_COLUMNS) + len(
    _NEIGHBORHOOD_COLUMNS
):  # pragma: no cover - guards the literal above
    raise ProvenanceError("provenance_columns declares the same column twice")


def declared_columns() -> tuple[str, ...]:
    """Names of every declared column, in declaration order."""
    return tuple(PROVENANCE)


def root_sources(column: str, *, _seen: frozenset[str] = frozenset()) -> frozenset[str]:
    """Resolve ``column`` through ``derived:`` until only root forms remain.

    Args:
        column: a declared column name.

    Returns:
        The set of ``inp_section:`` / ``swmm_report:`` / ``hydraulics:`` /
        ``config:`` (and, if anything ever declared one, ``labels:``) sources the
        column ultimately rests on.

    Raises:
        ProvenanceError: if the column is not declared, if a ``derived:`` source
            names a column that is not declared, or if the declaration is
            circular. All three are defects in this file, not in the data.
    """
    if column in _seen:
        raise ProvenanceError(f"Provenance of {column!r} is circular: {sorted(_seen | {column})}")
    try:
        entry = PROVENANCE[column]
    except KeyError as error:
        raise ProvenanceError(f"Column {column!r} has no declared provenance") from error

    roots: set[str] = set()
    for source in entry.sources:
        if not source.startswith(DERIVED_PREFIX):
            roots.add(source)
            continue
        for parent in source[len(DERIVED_PREFIX) :].split(","):
            roots |= root_sources(parent.strip(), _seen=_seen | {column})
    return frozenset(roots)


def missing_for(contract: FeatureContract) -> tuple[str, ...]:
    """Contract columns with no declared provenance, in contract order."""
    return tuple(column for column in contract.columns if column not in PROVENANCE)


def tainted_for(contract: FeatureContract) -> Mapping[str, frozenset[str]]:
    """Contract columns that resolve to a forbidden source, with the offending roots."""
    tainted: dict[str, frozenset[str]] = {}
    for column in contract.columns:
        if column not in PROVENANCE:
            continue
        bad = frozenset(source for source in root_sources(column) if source.startswith(FORBIDDEN_SOURCE_PREFIX))
        if bad:
            tainted[column] = bad
    return MappingProxyType(tainted)


def onehot_columns() -> tuple[str, ...]:
    """The columns declared as the intended one-hot encoding of a categorical input."""
    return tuple(name for name, item in PROVENANCE.items() if item.intended_onehot)


AUDITED_CONTRACTS: tuple[FeatureContract, ...] = (
    FEATURE_CONTRACT_V1,
    FEATURE_CONTRACT_V2,
    FEATURE_CONTRACT_V2_NOFLAGS,
    FEATURE_CONTRACT_V3_PHYSICAL,
    FEATURE_CONTRACT_V3_MARGINS,
    FEATURE_CONTRACT_V3_PHYSICAL_NBR,
    FEATURE_CONTRACT_V3_PHYSICAL_NOCAT,
    FEATURE_CONTRACT_V3_PHYSICAL_NBR_NOCAT,
    FEATURE_CONTRACT_V4_PHYSICAL,
    FEATURE_CONTRACT_V4_MARGINS,
    FEATURE_CONTRACT_V4_PHYSICAL_NBR,
    FEATURE_CONTRACT_V4_PHYSICAL_NOCAT,
    FEATURE_CONTRACT_V4_PHYSICAL_NBR_NOCAT,
    FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL,
    FEATURE_CONTRACT_V4_NOFLOOD_MARGINS,
    FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL_NBR,
)
"""Every contract the allowlist is held against.

All of them, not just the one a given build projects: the check is about what a
column was *made out of*, which does not change with the contract that selects
it, and a contract left out of this tuple is a model input nothing audits.
``v2_noflags`` is a subset of v2 and adds no declaration; the two v3 contracts add
the whole physical view, and ``v3_physical_nbr`` adds the neighbourhood block,
whose declarations are generated from the same list that names its columns
(``NEIGHBORHOOD_COLUMNS`` x ``NEIGHBORHOOD_HOPS``). The two ``*_nocat`` arms are
subsets of the two they ablate and likewise add no declaration -- they are listed
so that the registry and this tuple stay the same set, which is what the contract
test checks. The v4 contracts add exactly the two freeboard columns and the
neighbourhood block of the C2 measurement list between them; everything else they
name is a column one of the v3 contracts already declares, and the three
``v4_noflood_*`` arms are subsets of the v4 views they ablate."""


__all__ = [
    "AUDITED_CONTRACTS",
    "DERIVED_PREFIX",
    "FORBIDDEN_SOURCE_PREFIX",
    "PROVENANCE",
    "ROOT_SOURCE_PREFIXES",
    "ColumnProvenance",
    "ProvenanceError",
    "declared_columns",
    "missing_for",
    "onehot_columns",
    "root_sources",
    "tainted_for",
]
