"""Versioned conduit feature contracts, their order and their dtypes.

The order is load-bearing: it is the input layout of the saved recommendation
MLP and of the GraphSAGE weights. It is written here as a single explicit
literal, replacing the two unsynchronised copies that used to live in
``app/sa/core/constants.py`` and ``app/sa/core/feature_contract.py``.

Layout: 7 hydraulic validity flags, 5 decision flags, 9 normalised values and
12 one-hot land cover columns. The 12 flags are ``int8``; everything else
(normalised values *and* the one-hot block) is ``float64`` — the model input is
a float matrix, so the one-hot columns are cast, not left as integers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import pandas as pd

VALIDITY_FLAG_COLUMNS: tuple[str, ...] = (
    "ValMaxFill",
    "ValMaxV",
    "ValMinV",
    "ValMaxSlope",
    "ValMinSlope",
    "ValDepth",
    "ValCoverage",
)

DECISION_FLAG_COLUMNS: tuple[str, ...] = (
    "isMinDiameter",
    "IncreaseDia",
    "ReduceDia",
    "IncreaseSlope",
    "ReduceSlope",
)

NORMALIZED_COLUMNS: tuple[str, ...] = (
    "NRoughness",
    "NMaxV",
    "NInletDepth",
    "NOutletDepth",
    "NFilling",
    "NMaxQ",
    "NInletGroundCover",
    "NOutletGroundCover",
    "NSlope",
)

REDUNDANT_GROUND_COVER_COLUMNS: tuple[str, ...] = (
    "NInletGroundCover",
    "NOutletGroundCover",
)
"""Columns removed by v2.

They duplicate the normalised inlet/outlet depth to the precision carried by
the generated dataset.  The physical, unnormalised ground-cover intermediates
remain in ``features.parquet`` for audits; only the model-input projection is
reduced.
"""

FEATURE_ONEHOT_ORDER: tuple[str, ...] = (
    "marshes",
    "suburban_highly_impervious",
    "suburban_weakly_impervious",
    "arable",
    "meadows",
    "forests",
    "rural",
    "urban_weakly_impervious",
    "urban_moderately_impervious",
    "urban_highly_impervious",
    "mountains_rocky",
    "mountains_vegetated",
)
"""One-hot block order of the conduit feature matrix.

This is NOT the class order of the catchment classifier — see
``catchment_features.CLASSIFIER_CLASS_ORDER``. The two must never be derived
from one another.
"""

_FEATURE_COLUMNS_V1: tuple[str, ...] = VALIDITY_FLAG_COLUMNS + DECISION_FLAG_COLUMNS + NORMALIZED_COLUMNS + FEATURE_ONEHOT_ORDER
_FEATURE_COLUMNS_V2: tuple[str, ...] = tuple(
    column for column in _FEATURE_COLUMNS_V1 if column not in REDUNDANT_GROUND_COVER_COLUMNS
)

FLAG_COLUMNS: tuple[str, ...] = VALIDITY_FLAG_COLUMNS + DECISION_FLAG_COLUMNS
"""The 12 columns stored as ``int8``."""

_FEATURE_COLUMNS_V2_NOFLAGS: tuple[str, ...] = tuple(column for column in _FEATURE_COLUMNS_V2 if column not in set(FLAG_COLUMNS))
"""v2 without the 12 flags: 7 normalised values and the 12 one-hot columns.

Derived from v2 rather than retyped, so the two can never disagree about which
columns they share or in what order they stand.
"""

FLOAT_COLUMNS: tuple[str, ...] = NORMALIZED_COLUMNS + FEATURE_ONEHOT_ORDER
"""The 21 columns stored as ``float64``."""

FLAG_DTYPE = "int8"
FLOAT_DTYPE = "float64"


@dataclass(frozen=True)
class FeatureContract:
    """One immutable model-input schema with a content-derived identity.

    The digest is over the ordered column names alone.  That recipe is frozen,
    not chosen: it is the one every registry entry of variants B1--B3 was
    written under, and a digest that also covered dtypes would relabel models
    that have not changed.  Two contracts therefore differ in ``sha`` exactly
    when they differ in which columns they name, or in what order.
    """

    name: str
    columns: tuple[str, ...]
    dtypes: tuple[tuple[str, str], ...]
    schema_version: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature contract name must not be empty")
        if self.schema_version < 1:
            raise ValueError(f"Feature contract schema_version must be positive, got {self.schema_version}")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Feature contract columns must be unique")
        dtype_names = {pair[0] for pair in self.dtypes}
        if dtype_names != set(self.columns):
            raise ValueError("Feature contract dtypes must name exactly its columns")

    @property
    def dtypes_map(self) -> Mapping[str, str]:
        """Read-only mapping from column name to dtype string."""
        return MappingProxyType(dict(self.dtypes))

    @property
    def n_columns(self) -> int:
        """Number of model inputs in the contract."""
        return len(self.columns)

    @property
    def sha(self) -> str:
        """SHA-256 of the ordered column names, using the frozen v1 recipe."""
        return hashlib.sha256("\n".join(self.columns).encode("utf-8")).hexdigest()


class FeatureContractError(ValueError):
    """Raised when a frame does not satisfy the 33-column feature contract."""


_DTYPES_V1: tuple[tuple[str, str], ...] = tuple(
    (column, FLAG_DTYPE) for column in VALIDITY_FLAG_COLUMNS + DECISION_FLAG_COLUMNS
) + tuple((column, FLOAT_DTYPE) for column in NORMALIZED_COLUMNS + FEATURE_ONEHOT_ORDER)
_DTYPES_V2: tuple[tuple[str, str], ...] = tuple(pair for pair in _DTYPES_V1 if pair[0] in _FEATURE_COLUMNS_V2)

_DTYPES_V2_NOFLAGS: tuple[tuple[str, str], ...] = tuple(pair for pair in _DTYPES_V2 if pair[0] in _FEATURE_COLUMNS_V2_NOFLAGS)

FEATURE_CONTRACT_V1 = FeatureContract(name="v1", columns=_FEATURE_COLUMNS_V1, dtypes=_DTYPES_V1, schema_version=1)
FEATURE_CONTRACT_V2 = FeatureContract(name="v2", columns=_FEATURE_COLUMNS_V2, dtypes=_DTYPES_V2, schema_version=2)
FEATURE_CONTRACT_V2_NOFLAGS = FeatureContract(
    name="v2_noflags",
    columns=_FEATURE_COLUMNS_V2_NOFLAGS,
    dtypes=_DTYPES_V2_NOFLAGS,
    # 2, not 3: `schema_version` names the *build* a contract projects, and this
    # one projects a v2 build -- it needs no new `features.parquet`, no new entry
    # in `FEATURES_REQUIRED_COLUMNS` and no rebuild of the pool. The next integer
    # belongs to a contract that does (plan16 section 5.2, contract v3).
    schema_version=2,
)
"""The 19-column ablation arm: v2 with the seven validity and five decision flags removed.

The flags are the cascade's own intermediates -- ``labels/rules.py`` reads them
to decide the label -- so a model given them is being handed a summary of the
answer. Dropping them is what makes the comparison of plan16 section 3 (E1)
possible; the contract exists so that comparison runs through the same
validated layout every other model input goes through.
"""

PHYSICAL_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "diameter_m",
    "length_m",
    "roughness",
    "barrels",
    "in_offset_m",
    "out_offset_m",
    "inlet_pipe_invert_m",
    "outlet_pipe_invert_m",
    "slope_signed",
    "inlet_ground_m",
    "outlet_ground_m",
    "inlet_pipe_depth_m",
    "outlet_pipe_depth_m",
    "inlet_cover_m",
    "outlet_cover_m",
    "inlet_node_max_depth_m",
    "outlet_node_max_depth_m",
)
"""Geometry of the conduit as laid down in the INP, offsets resolved.

The invert of the *pipe* end and the depth of the *manhole* are separate columns
on purpose: the historical pipeline identified them, which is only true where the
offsets are zero (plan16 §2 P1).
"""

PHYSICAL_HYDRAULIC_COLUMNS: tuple[str, ...] = (
    "max_flow_cms",
    "max_velocity_mps",
    "max_full_flow_ratio",
    "max_full_depth_ratio",
    "hours_full_both_ends",
    "hours_full_upstream",
    "hours_full_downstream",
    "hours_above_normal_flow",
    "hours_capacity_limited",
    "inlet_node_max_depth_reported_m",
    "outlet_node_max_depth_reported_m",
    "inlet_node_max_hgl_m",
    "outlet_node_max_hgl_m",
    "inlet_node_flood_hours",
    "inlet_node_flood_volume_1e6l",
    "outlet_node_flood_hours",
    "outlet_node_flood_volume_1e6l",
    "full_flow_cms",
    "flow_capacity_ratio",
)
"""The simulated state, read back from the report, plus the two capacity terms.

Nothing here is clipped. ``max_full_depth_ratio`` and ``max_full_flow_ratio`` are
SWMM's own ratios and may exceed 1; ``full_flow_cms`` and ``flow_capacity_ratio``
are undefined on a flat or adverse conduit and are missing there rather than
being given a fictitious value.

``max_full_depth_ratio`` **is** the relative filling of the conduit; there is no
second column repeating it under another name.

``full_flow_cms`` is the capacity of **one barrel**, as the historical ``NMaxQ``
is, while ``max_flow_cms`` from the report is the flow through all of them --
so ``flow_capacity_ratio`` is comparable across conduits only where ``barrels``
is 1 (see :mod:`stormlab.features.physical`).
"""

NODE_FLOOD_COLUMNS: tuple[str, ...] = (
    "inlet_node_flood_hours",
    "inlet_node_flood_volume_1e6l",
    "outlet_node_flood_hours",
    "outlet_node_flood_volume_1e6l",
)
"""The four columns that say a node spilled onto the surface, and how much.

One list, read by the contract that carries them and by the ablation that removes
them -- the pattern :data:`LAND_COVER_CATEGORY_COLUMNS` follows, and for the same
reason: an ablation whose column list is retyped beside the contract's is an
ablation that can stop ablating without anything failing.

They are removed in the ``v4_noflood_*`` arm because under a dynamic wave the
overload premise of variant C2 *is* the flooding of the inlet node, so a model given
these columns is reading the label rather than predicting it (plan17 §Etap 1b). That
is a trivialised task, not leaked data: the target is derived from the measurement
openly, and the arm exists so the benchmark can say how much of the reading is the
derivation.
"""

PHYSICAL_CONTEXT_COLUMNS: tuple[str, ...] = (
    "sbc_category_code",
    "subcatchment_area_ha",
    "subcatchment_imperv_pct",
    "subcatchment_width_m",
    "subcatchment_slope_pct",
    "rain_total_mm",
    "rain_duration_h",
    "rain_peak_intensity_mmph",
)
"""Catchment draining into the inlet node, and the rainfall scenario of the file."""

LAND_COVER_CATEGORY_COLUMNS: tuple[str, ...] = ("sbc_category_code", "sbc_missing")
"""The two v3 columns that name the land cover class of the catchment, and its absence.

The same pair :data:`stormlab.labels.sources.registry.CATEGORY_COLUMNS` names on the
label side, and for the same reason: the category is the one input of series C1 that
enters both a feature view and a label source, so removing it has to be one list read
from one place. The remaining ``subcatchment_*`` columns stay -- area, imperviousness,
width and slope are SWMM inputs of the run, not a land cover label.
"""

PHYSICAL_REQUIREMENT_COLUMNS: tuple[str, ...] = (
    "required_cover_m",
    "frost_zone_m",
    "max_depth_m",
    "max_filling_ratio",
    "v_min_mps",
    "v_max_mps",
    "min_slope_required",
    "max_slope_allowed",
)
"""The design requirements in force, carried alongside the measurements.

Constant per run for the first six and per diameter for the last two. They are
inputs so that a margin is reconstructible from the row that carries it; a run
that moves a requirement moves the column, not a hidden constant.
"""

PHYSICAL_RECEIVER_COLUMNS: tuple[str, ...] = ("outfall_fixed_stage_m",)

PHYSICAL_MASK_COLUMNS: tuple[str, ...] = (
    "is_outfall_conduit",
    "inlet_ground_missing",
    "outlet_ground_missing",
    "slope_nonpositive",
    "slope_requirement_missing",
    "min_slope_missing",
    "sbc_missing",
    "subcatchment_missing",
    "outfall_stage_missing",
    "inlet_node_depth_missing",
    "outlet_node_depth_missing",
)
"""Indicator columns of the physical view, stored as ``int8``.

Every one of them states a fact about the *network or the file*, never a
diagnosis or a recommended action -- that is the line plan16 §2 P3 draws around
the twelve flags of v1/v2. Nine of them mark a value that is missing rather
than zero, so a model can tell "not measured" from "measured as nothing".

The two slope masks are separate because the two requirements fail separately:
``max_slope_allowed`` depends on the diameter alone, ``min_slope_required`` also
on the filling, so a conduit that carried no flow loses the second and keeps the
first.
"""

MARGIN_COLUMNS: tuple[str, ...] = (
    "m_cover_inlet",
    "m_cover_outlet",
    "m_depth_inlet",
    "m_depth_outlet",
    "r_cover_inlet",
    "r_cover_outlet",
    "m_filling",
    "m_velocity_min",
    "m_velocity_max",
    "m_slope_min",
    "m_slope_max",
    "m_capacity",
)
"""Relative distance from each requirement; negative means the requirement is broken.

Unclipped by construction, which is what keeps different sizes of the same
violation distinguishable (plan16 §2 P3).
"""

MARGIN_MASK_COLUMNS: tuple[str, ...] = ("range_contradictory",)
"""1 where ``C_max <= C_wym``, i.e. the admissible cover band is empty.

The row is kept and marked. Deciding what to do with a network whose requirements
cannot all hold is the business of the stage that builds the dataset, not of the
extraction that measured it.
"""

PHYSICAL_FREEBOARD_COLUMNS: tuple[str, ...] = ("inlet_node_freeboard_m", "outlet_node_freeboard_m")
"""How far the highest water level at each end stayed below the terrain, in metres.

``<end>_ground_m - <end>_node_max_hgl_m``: positive is a water table under the
ground, negative is a head standing above it. Both terms are already columns of
``v3_physical``, so this is arithmetic on the row rather than a new reading of the
report -- but it is the arithmetic that says whether a node surcharged out onto the
surface, which under a dynamic wave is the premise the label sources of variant C2
rest on, and asking every consumer to subtract two columns would leave the premise
undeclared (plan17 §Etap 1).

Missing wherever either term is: the mask pair of each column names both ways.
"""

_FEATURE_COLUMNS_V3_PHYSICAL: tuple[str, ...] = (
    PHYSICAL_GEOMETRY_COLUMNS
    + PHYSICAL_HYDRAULIC_COLUMNS
    + PHYSICAL_CONTEXT_COLUMNS
    + PHYSICAL_REQUIREMENT_COLUMNS
    + PHYSICAL_RECEIVER_COLUMNS
    + PHYSICAL_MASK_COLUMNS
)
_FEATURE_COLUMNS_V3_MARGINS: tuple[str, ...] = _FEATURE_COLUMNS_V3_PHYSICAL + MARGIN_COLUMNS + MARGIN_MASK_COLUMNS
_FEATURE_COLUMNS_V3_PHYSICAL_NOCAT: tuple[str, ...] = tuple(
    column for column in _FEATURE_COLUMNS_V3_PHYSICAL if column not in set(LAND_COVER_CATEGORY_COLUMNS)
)
"""v3_physical minus the land cover category and its missing-value mask.

Derived by filtering rather than retyped, so the ablation arm and the contract it
ablates cannot disagree about the 63 columns they share or about their order.
"""

PHYSICAL_FLAG_COLUMNS: tuple[str, ...] = PHYSICAL_MASK_COLUMNS + MARGIN_MASK_COLUMNS
"""The ``int8`` block of the v3 contracts."""


def _physical_dtypes(columns: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """``int8`` for the indicator columns of the physical view, ``float64`` for the rest."""
    flags = set(PHYSICAL_FLAG_COLUMNS)
    return tuple((column, FLAG_DTYPE if column in flags else FLOAT_DTYPE) for column in columns)


FEATURE_CONTRACT_V3_PHYSICAL = FeatureContract(
    name="v3_physical",
    columns=_FEATURE_COLUMNS_V3_PHYSICAL,
    dtypes=_physical_dtypes(_FEATURE_COLUMNS_V3_PHYSICAL),
    # 3, not 2: unlike `v2_noflags` this contract projects a *different build*.
    # Not one of its columns exists in a v2 `features.parquet`, and producing them
    # needs the physical extraction of `features/physical.py` rather than a
    # reprojection of what is already there.
    schema_version=3,
)
"""F2 of plan16 §2 P3: the physical state of a conduit, unclipped and undecided.

Nothing here is a validity flag or a proposed action, and nothing is normalised
onto ``[0, 1]``: the point of the view is that a violation twice as large reads
as twice as large. Missing values are permitted and are always paired with a mask
of :data:`PHYSICAL_MASK_COLUMNS`, which is why frames of this contract are
validated by :func:`stormlab.features.physical.validate_physical_features` rather
than by :func:`validate_features`.
"""

FEATURE_CONTRACT_V3_MARGINS = FeatureContract(
    name="v3_margins",
    columns=_FEATURE_COLUMNS_V3_MARGINS,
    dtypes=_physical_dtypes(_FEATURE_COLUMNS_V3_MARGINS),
    schema_version=3,
)
"""F3 of plan16 §2 P3: F2 plus the relative distance to every requirement.

Derived from F2 rather than declared beside it, so the two can never disagree
about the measurements they share.
"""

FEATURE_CONTRACT_V3_PHYSICAL_NOCAT = FeatureContract(
    name="v3_physical_nocat",
    columns=_FEATURE_COLUMNS_V3_PHYSICAL_NOCAT,
    dtypes=_physical_dtypes(_FEATURE_COLUMNS_V3_PHYSICAL_NOCAT),
    schema_version=3,
)
"""F2 without the land cover category: the feature side of the category ablation.

Sixty-eight per cent of the conduits of the generated pool drain terrain that is not
urban, because the category is drawn uniformly from twelve classes -- so a result that
leans on it leans on a proportion nobody designed. The arm this contract feeds asks how
much of the E6 reading survives when neither the features nor the label sources may see
it (author's decision of 2026-09-07).

The four ``subcatchment_*`` parameters stay: they are the numbers SWMM was run with, not
a name given to the terrain. Only :data:`LAND_COVER_CATEGORY_COLUMNS` goes.
"""

_FEATURE_COLUMNS_V4_PHYSICAL: tuple[str, ...] = _FEATURE_COLUMNS_V3_PHYSICAL + PHYSICAL_FREEBOARD_COLUMNS
_FEATURE_COLUMNS_V4_MARGINS: tuple[str, ...] = _FEATURE_COLUMNS_V4_PHYSICAL + MARGIN_COLUMNS + MARGIN_MASK_COLUMNS
_FEATURE_COLUMNS_V4_PHYSICAL_NOCAT: tuple[str, ...] = tuple(
    column for column in _FEATURE_COLUMNS_V4_PHYSICAL if column not in set(LAND_COVER_CATEGORY_COLUMNS)
)
"""The v4 views, each derived from the v3 list it extends rather than retyped beside it.

``v3_physical`` is therefore a *prefix* of ``v4_physical`` and of ``v4_margins``,
which is what lets one extraction serve both generations: a frame written to the v4
layout still projects onto either v3 contract, and the frozen digests of v3 cannot
move because nothing was inserted into their lists.
"""

FEATURE_CONTRACT_V4_PHYSICAL = FeatureContract(
    name="v4_physical",
    columns=_FEATURE_COLUMNS_V4_PHYSICAL,
    dtypes=_physical_dtypes(_FEATURE_COLUMNS_V4_PHYSICAL),
    # 4, not 3: `schema_version` names the *build* a contract projects, and a v3
    # `features.parquet` carries no freeboard column. The two terms it is made of
    # are in there, but a projection does not compute -- producing these columns
    # needs the extraction to run again.
    schema_version=4,
)
"""``v3_physical`` plus the freeboard at each end: the F2 view of variant C2.

Two columns and nothing else. Everything that made the physical view what it is --
unclipped, undecided, missing rather than repaired -- is unchanged, and the pair
that is added is the one measurement of a surcharge that reaches the surface.
"""

FEATURE_CONTRACT_V4_MARGINS = FeatureContract(
    name="v4_margins",
    columns=_FEATURE_COLUMNS_V4_MARGINS,
    dtypes=_physical_dtypes(_FEATURE_COLUMNS_V4_MARGINS),
    schema_version=4,
)
"""F3 of variant C2: ``v4_physical`` followed by the same twelve margins and their mask.

The margins are the v3 ones, in the v3 order. What moved is where they start, which
is why ``v3_margins`` is not a prefix of this contract while ``v3_physical`` is: the
frame is projected by name, so both still resolve, and the frozen digest of
``v3_margins`` names the same twelve columns it always did.
"""

FEATURE_CONTRACT_V4_PHYSICAL_NOCAT = FeatureContract(
    name="v4_physical_nocat",
    columns=_FEATURE_COLUMNS_V4_PHYSICAL_NOCAT,
    dtypes=_physical_dtypes(_FEATURE_COLUMNS_V4_PHYSICAL_NOCAT),
    schema_version=4,
)
"""The category ablation of the C2 feature view, filtered from it exactly as v3's is."""

_FEATURE_COLUMNS_V4_NOFLOOD_PHYSICAL: tuple[str, ...] = tuple(
    column for column in _FEATURE_COLUMNS_V4_PHYSICAL if column not in set(NODE_FLOOD_COLUMNS)
)
_FEATURE_COLUMNS_V4_NOFLOOD_MARGINS: tuple[str, ...] = tuple(
    column for column in _FEATURE_COLUMNS_V4_MARGINS if column not in set(NODE_FLOOD_COLUMNS)
)
"""The two v4 views with :data:`NODE_FLOOD_COLUMNS` filtered out, order otherwise kept.

Filtered from the lists they ablate rather than retyped beside them, so the arm and
the view it is compared against cannot disagree about the columns they share or about
the order they carry them in."""

FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL = FeatureContract(
    name="v4_noflood_physical",
    columns=_FEATURE_COLUMNS_V4_NOFLOOD_PHYSICAL,
    dtypes=_physical_dtypes(_FEATURE_COLUMNS_V4_NOFLOOD_PHYSICAL),
    schema_version=4,
)
"""``v4_physical`` without the four node-flooding columns: the ablation arm of C2.

What stays is everything that says a conduit is under strain without saying it
overflowed -- ``max_full_depth_ratio``, the five surcharge-hour columns, both maximum
heads, both reported node depths, both freeboards, the velocities, the capacity ratio
and the whole of the geometry, the catchment and the storm. The research question of
the arm is whether that strain is enough to predict the need for retention *before*
the spill is on the record (plan17 §Etap 1b)."""

FEATURE_CONTRACT_V4_NOFLOOD_MARGINS = FeatureContract(
    name="v4_noflood_margins",
    columns=_FEATURE_COLUMNS_V4_NOFLOOD_MARGINS,
    dtypes=_physical_dtypes(_FEATURE_COLUMNS_V4_NOFLOOD_MARGINS),
    schema_version=4,
)
"""The F3 counterpart of the arm: ``v4_margins`` with the same four columns removed.

The twelve margins stay whole -- none of them is computed from a flood column -- so
this contract differs from ``v4_margins`` in exactly what ``v4_noflood_physical``
differs from ``v4_physical`` in."""

NEIGHBORHOOD_COLUMNS: tuple[str, ...] = (
    "max_full_depth_ratio",
    "slope_signed",
    "max_flow_cms",
    "hours_capacity_limited",
)
"""The F2 measurements variant C1 summarises over a conduit's graph neighbourhood.

Every name is a *measurement* -- an aggregate over a neighbour's mask, margin or
label would hand a model a summary of what was decided about it. The fourth is the
premise C1 overloads on: hours during which the conduit's capacity limited the flow.

One of several declared neighbourhoods, not the only one; which contract carries
which list is :data:`NEIGHBORHOOD_CONTRACT_COLUMNS`, and that is what
``config.c1.neighborhood.columns`` is validated against.
"""

NEIGHBORHOOD_COLUMNS_V4: tuple[str, ...] = (
    "max_full_depth_ratio",
    "slope_signed",
    "max_flow_cms",
    "inlet_node_flood_hours",
)
"""The same neighbourhood under variant C2, with the C2 overload premise in it.

Three of the four are C1's. The fourth is not a renaming: under a dynamic wave the
premise of an overload is that the upper node spilled, so the quantity a neighbour
has to contribute is its flooding, not the hours its capacity limited the flow
(plan17 §Etap 2). The width of the generated block is unchanged, so the C2
neighbourhood contract is as wide as C1's and says something else."""

NEIGHBORHOOD_COLUMNS_V4_NOFLOOD: tuple[str, ...] = tuple(
    column for column in NEIGHBORHOOD_COLUMNS_V4 if column not in set(NODE_FLOOD_COLUMNS)
)
"""The C2 neighbourhood of the ``noflood`` arm: :data:`NEIGHBORHOOD_COLUMNS_V4` minus the flood column.

A **subset**, computed by the same filter that builds the ``noflood`` feature lists,
and not a substitution -- no replacement column takes the place of the one removed.
The aggregates are computed per column independently, so the three that stay hold
exactly the values they hold in the full arm, and the protocol keeps declaring four
columns while this arm reads three of them."""

NEIGHBORHOOD_HOPS: tuple[int, ...] = (1, 2)
"""Radii of the neighbourhood, in hops; ``config.c1.neighborhood.hops``."""

NEIGHBORHOOD_DIRECTIONS: tuple[str, ...] = ("up", "down")
"""Name prefixes of the two neighbourhoods, in column order."""

NEIGHBORHOOD_STATISTICS: tuple[str, ...] = ("min", "max", "mean")
"""The three reductions, in column order."""

NEIGHBORHOOD_COUNT_SUFFIX = "count"
NEIGHBORHOOD_PRESENT_SUFFIX = "present"

NEIGHBORHOOD_CONTRACT_NAME = "v3_physical_nbr"
NEIGHBORHOOD_CONTRACT_NOCAT_NAME = "v3_physical_nbr_nocat"
NEIGHBORHOOD_CONTRACT_V4_NAME = "v4_physical_nbr"
NEIGHBORHOOD_CONTRACT_V4_NOCAT_NAME = "v4_physical_nbr_nocat"
NEIGHBORHOOD_CONTRACT_V4_NOFLOOD_NAME = "v4_noflood_physical_nbr"
"""The same block over the ablated and over the C2 bases; named here so the matrix resolves each by name.

Spelled out rather than derived from the base's name because the registry is keyed
by these strings: two contracts that happened to agree on a name would collapse into
one entry instead of failing.
"""


def neighborhood_column_names(
    columns: Sequence[str] = NEIGHBORHOOD_COLUMNS,
    hops: Sequence[int] = NEIGHBORHOOD_HOPS,
) -> tuple[str, ...]:
    """The generated neighbourhood block: ``{up|down}{k}_{column}_{statistic}``, then count and presence.

    A pure function of the two lists, which is what lets the derived contract be
    identified by the ordinary digest over its names. It lives here rather than
    beside the aggregation because ``features`` sits *above* ``contracts`` in the
    layer stack: the names have to be readable where the aggregation is not.

    Raises:
        FeatureContractError: on an empty or repeating list, or a non-positive hop.
    """
    names = [str(item) for item in columns]
    steps = [int(item) for item in hops]
    if not names or len(set(names)) != len(names):
        raise FeatureContractError(f"The neighbourhood columns must be a non-empty list of distinct names, got {names}")
    if not steps or len(set(steps)) != len(steps) or any(step < 1 for step in steps):
        raise FeatureContractError(f"The neighbourhood hops must be a non-empty list of distinct positive integers, got {steps}")

    generated: list[str] = []
    for direction in NEIGHBORHOOD_DIRECTIONS:
        for step in steps:
            for column in names:
                generated.extend(f"{direction}{step}_{column}_{statistic}" for statistic in NEIGHBORHOOD_STATISTICS)
            generated.append(f"{direction}{step}_{NEIGHBORHOOD_COUNT_SUFFIX}")
            generated.append(f"{direction}{step}_{NEIGHBORHOOD_PRESENT_SUFFIX}")
    return tuple(generated)


def neighborhood_contract(
    base: FeatureContract,
    *,
    columns: Sequence[str] = NEIGHBORHOOD_COLUMNS,
    hops: Sequence[int] = NEIGHBORHOOD_HOPS,
    name: str = NEIGHBORHOOD_CONTRACT_NAME,
) -> FeatureContract:
    """``base`` followed by the generated neighbourhood block.

    Derived rather than written out, so the registered contract and the frame
    :func:`stormlab.features.neighborhood.neighborhood_aggregates` produces come
    from one generator and cannot disagree about a name or an order.

    ``schema_version`` is the base's: the aggregates need no new
    ``features.parquet``, only the graph of the files one already holds.

    Raises:
        FeatureContractError: for the reasons :func:`neighborhood_column_names`
            raises it, or if a generated name would shadow a column of ``base``.
    """
    generated = neighborhood_column_names(columns, hops)
    collisions = sorted(set(generated) & set(base.columns))
    if collisions:
        raise FeatureContractError(f"The neighbourhood block would shadow column(s) {collisions} of contract {base.name!r}")

    flags = {item for item in generated if item.endswith(f"_{NEIGHBORHOOD_PRESENT_SUFFIX}")}
    return FeatureContract(
        name=name,
        columns=base.columns + generated,
        dtypes=base.dtypes + tuple((item, FLAG_DTYPE if item in flags else FLOAT_DTYPE) for item in generated),
        schema_version=base.schema_version,
    )


NEIGHBORHOOD_BLOCK_COLUMNS: tuple[str, ...] = neighborhood_column_names()
"""The 56 generated columns of the declared neighbourhood, in contract order."""

FEATURE_CONTRACT_V3_PHYSICAL_NBR = neighborhood_contract(FEATURE_CONTRACT_V3_PHYSICAL)
"""F2 plus its neighbourhood block: the tabular arm's answer to the graph models.

The comparison of plan16 §3 (E6) is between a model that passes messages and a
model handed the same neighbourhood as columns -- not between a model with context
and one with none at all. Derived from ``v3_physical`` so the shared columns can
never drift; the two contracts it is built from are untouched by its existence.
"""

FEATURE_CONTRACT_V3_PHYSICAL_NBR_NOCAT = neighborhood_contract(
    FEATURE_CONTRACT_V3_PHYSICAL_NOCAT, name=NEIGHBORHOOD_CONTRACT_NOCAT_NAME
)
"""The neighbourhood arm of the category ablation: ``v3_physical_nocat`` plus the block.

The generated block is the same fifty-six columns as under ``v3_physical_nbr`` -- the
aggregated measurements are declared in :data:`NEIGHBORHOOD_COLUMNS` and none of them is
the category -- so the ablation removes two columns from the arm and nothing else, and
the very same aggregate frame serves both contracts.
"""


FEATURE_CONTRACT_V4_PHYSICAL_NBR = neighborhood_contract(
    FEATURE_CONTRACT_V4_PHYSICAL, columns=NEIGHBORHOOD_COLUMNS_V4, name=NEIGHBORHOOD_CONTRACT_V4_NAME
)
"""``v4_physical`` plus a neighbourhood block of the same width over C2's own list.

Fifty-six columns, as under ``v3_physical_nbr``, and not the same fifty-six: the
fourth aggregated measurement is the flooding of the neighbour's inlet node rather
than the hours its capacity limited the flow, which is the premise variant C2
overloads on."""

FEATURE_CONTRACT_V4_PHYSICAL_NBR_NOCAT = neighborhood_contract(
    FEATURE_CONTRACT_V4_PHYSICAL_NOCAT, columns=NEIGHBORHOOD_COLUMNS_V4, name=NEIGHBORHOOD_CONTRACT_V4_NOCAT_NAME
)
"""The neighbourhood arm of the C2 category ablation, over the same C2 neighbourhood."""

FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL_NBR = neighborhood_contract(
    FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL,
    columns=NEIGHBORHOOD_COLUMNS_V4_NOFLOOD,
    name=NEIGHBORHOOD_CONTRACT_V4_NOFLOOD_NAME,
)
"""The neighbourhood arm of the ``noflood`` ablation: 62 columns plus a block of 44.

Three aggregated measurements instead of four, so the generator yields
``2 x 2 x (3 x 3 + 2) = 44`` columns rather than 56. Removing the flood column from
the base and leaving it in the neighbourhood would let a model read off its
neighbours what it may not read off itself, so the same filter runs on both."""


NEIGHBORHOOD_CONTRACT_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        NEIGHBORHOOD_CONTRACT_NAME: NEIGHBORHOOD_COLUMNS,
        NEIGHBORHOOD_CONTRACT_NOCAT_NAME: NEIGHBORHOOD_COLUMNS,
        NEIGHBORHOOD_CONTRACT_V4_NAME: NEIGHBORHOOD_COLUMNS_V4,
        NEIGHBORHOOD_CONTRACT_V4_NOCAT_NAME: NEIGHBORHOOD_COLUMNS_V4,
        NEIGHBORHOOD_CONTRACT_V4_NOFLOOD_NAME: NEIGHBORHOOD_COLUMNS_V4_NOFLOOD,
    }
)
"""Which measurements each registered neighbourhood contract was generated from.

The list stopped being a single constant the moment a second variant declared a
different premise, so the thing a protocol is checked against is the list of the
contract it names, not a module-level default. Declared here rather than recovered
from the column names, because parsing ``up1_max_flow_cms_mean`` back into a column
and a statistic is guesswork on any name that contains an underscore."""


def neighborhood_columns_of(name: str) -> tuple[str, ...]:
    """The measurements the named neighbourhood contract aggregates.

    Raises:
        FeatureContractError: if ``name`` is not a registered neighbourhood contract.
    """
    try:
        return NEIGHBORHOOD_CONTRACT_COLUMNS[name]
    except KeyError as error:
        raise FeatureContractError(
            f"Unknown neighbourhood contract {name!r}; expected one of {sorted(NEIGHBORHOOD_CONTRACT_COLUMNS)}"
        ) from error


FEATURE_CONTRACTS: Mapping[str, FeatureContract] = MappingProxyType(
    {
        contract.name: contract
        for contract in (
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
    }
)
DEFAULT_FEATURE_CONTRACT = FEATURE_CONTRACT_V1

# Compatibility aliases remain the import surface for models until contracts are
# threaded through them in a later migration.
FEATURE_COLUMNS: tuple[str, ...] = FEATURE_CONTRACT_V1.columns
EXPECTED_N_FEATURES = FEATURE_CONTRACT_V1.n_columns


def class_order(kind: str = "recommendation") -> tuple[str, ...]:
    """Return a named canonical class order.

    Args:
        kind: ``"recommendation"`` for the nine intervention classes (enum
            declaration order) or ``"feature_onehot"`` for the twelve land
            cover columns of the feature matrix.

    Raises:
        FeatureContractError: if ``kind`` is not a known order name.
    """
    from stormlab.contracts.enums import RECOMMENDATION_CLASS_ORDER

    orders = {
        "recommendation": RECOMMENDATION_CLASS_ORDER,
        "feature_onehot": FEATURE_ONEHOT_ORDER,
    }
    if kind not in orders:
        raise FeatureContractError(f"Unknown class order {kind!r}; known orders: {sorted(orders)}")
    return orders[kind]


def feature_contract(name: str) -> FeatureContract:
    """Resolve a configured contract name without a permissive fallback."""
    try:
        return FEATURE_CONTRACTS[name]
    except KeyError as error:
        raise FeatureContractError(f"Unknown feature contract {name!r}; expected one of {sorted(FEATURE_CONTRACTS)}") from error


def expected_dtypes(contract: FeatureContract = DEFAULT_FEATURE_CONTRACT) -> dict[str, str]:
    """Map every contract column to its required dtype string."""
    return dict(contract.dtypes_map)


def validate_features(
    df: pd.DataFrame,
    *,
    contract: FeatureContract = DEFAULT_FEATURE_CONTRACT,
    columns: Sequence[str] | None = None,
) -> None:
    """Check that ``df`` carries ``contract``.

    The column list and its width come from ``contract``, not from a constant:
    v1 has 33 columns and v2 has 31, and a frame is correct against the contract
    its build declared, never against whichever one is the default.

    Verifies presence and *relative* order of the contract columns, their dtypes
    and the absence of missing values. Never repairs the frame: a missing feature
    is an error, not a column of zeros.

    Extra columns are allowed, deliberately, and this is where it differs from
    :func:`stormlab.contracts.catchment_features.validate_catchment_features`,
    which demands the frame be *exactly* the contract. The conduit frame travels
    through the pipeline carrying its own identity — ``file``, ``conduit``,
    ``sample_id``, the intermediates a parity failure is diagnosed from — and
    stripping that off just to validate would mean carrying two frames in step.
    What the model consumes is not this frame but the projection
    :func:`stormlab.features.conduits.conduit_feature_matrix` makes of it, and
    that one is exactly 33 columns wide. The catchment frame has no such
    companion metadata, so it is validated as the matrix it already is.

    Args:
        df: the frame to check; may carry columns beyond the contract.
        columns: the contract columns to require, in order.

    Raises:
        FeatureContractError: with the offending column names in the message.
    """
    if not isinstance(df, pd.DataFrame):
        raise FeatureContractError(f"Expected a DataFrame, got {type(df).__name__}")

    required = list(contract.columns if columns is None else columns)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise FeatureContractError(f"Missing feature columns ({len(missing)}): {missing}")

    present = [column for column in df.columns if column in set(required)]
    if present != required:
        raise FeatureContractError(f"Feature columns are out of contract order; expected {required}, got {present}")

    wanted = expected_dtypes(contract)
    wrong_dtype = {
        column: str(df[column].dtype) for column in required if column in wanted and str(df[column].dtype) != wanted[column]
    }
    if wrong_dtype:
        raise FeatureContractError(f"Wrong dtypes: {wrong_dtype} (expected {[(c, wanted[c]) for c in wrong_dtype]})")

    na_counts = {column: int(df[column].isna().sum()) for column in required if df[column].isna().any()}
    if na_counts:
        raise FeatureContractError(f"Missing values in feature columns: {na_counts}")


def feature_contract_sha(contract: FeatureContract = DEFAULT_FEATURE_CONTRACT) -> str:
    """Digest identifying the exact contract a model was trained against.

    Defined as the sha256 of the newline-joined column names, encoded
    as UTF-8, with **no trailing newline**. The exact recipe matters more than
    its elegance: the digest is stored next to every trained artifact, and a
    model whose recorded digest differs from the current one was trained on a
    different input layout — so the recipe is frozen here and pinned by a test
    asserting the literal value.
    """
    return contract.sha
