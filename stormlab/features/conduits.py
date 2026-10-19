"""Conduit feature engineering: the 33-column model input and its intermediates.

Ported from ``ConduitFeatureEngineeringService`` in ``app/sa/core/conduits.py``.
The arithmetic is unchanged; what changed is ownership and honesty about
parameters:

* :class:`ConduitFeatureSteps` still mutates its frames in place, step by step,
  because the steps genuinely depend on each other's columns. It is the
  implementation, not the interface.
* :func:`extract_conduit_features` is the interface. It owns the step order,
  which used to live in ``DataManager.feature_engineering`` and was therefore
  impossible to reuse without the whole Django-era object, validates its inputs
  up front, and works on copies so the caller's frames survive.
* Every value the thesis backlog wants to vary — the flow constant, the two
  minimum slope rules, the maximum filling ratio, the self-cleansing velocity —
  is an explicit argument rather than a module constant.

The historical pipeline used *two different* minimum slope rules at once: the
slope features come from the hydromechanical criterion of
:func:`stormlab.hydraulics.round.min_slope`, while the hydraulic validation
inside the minimum diameter search goes through Imhoff's 1/D. That is reproduced
here rather than tidied away.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from stormlab.contracts.features import (
    DEFAULT_FEATURE_CONTRACT,
    FEATURE_ONEHOT_ORDER,
    FeatureContract,
    FeatureContractError,
    validate_features,
)
from stormlab.features.subcatchments import SubcatchmentFeatureError, encode_categories
from stormlab.hydraulics import formulas
from stormlab.hydraulics.round import (
    DEFAULT_G,
    DEFAULT_THETA,
    MANNING_COEFFICIENT,
    MANNING_FULL_PIPE_COEFFICIENT,
    MAX_FILLING_RATIO,
    common_diameters,
    max_depth_value,
    max_filling,
    max_slope,
    max_velocity_value,
    min_slope,
    min_velocity_value,
)
from stormlab.hydraulics.validators import (
    MinSlopeRule,
    validate_filling,
    validate_max_velocity,
    validate_min_velocity,
)

logger = logging.getLogger(__name__)

DEFAULT_FLOW_CONSTANT = 0.312
"""Empirical ``k`` of the legacy ``Q_max = k * D^2.5 * S^0.5`` law.

The constant does not contain Manning roughness ``n``; ``manning_v2`` reads
roughness separately from each conduit.
"""

DEFAULT_MIN_ROUGHNESS = 0.009
DEFAULT_MAX_ROUGHNESS = 0.020
"""Manning roughness range used to normalise ``Roughness`` (PVC to stone channels).

Defaults, not constants: ``config.yaml::hydraulics`` carries the pair and
:func:`extract_conduit_features` forwards it, so widening the range is a
configuration change and a rerun rather than an edit here."""


def resolve_required_cover(frost_zone: float, required_cover: float | None) -> float:
    """The soil cover a conduit must carry above its crown [m].

    The one place the ``None`` of ``config.required_cover`` is turned into a
    number. ``None`` means "the frost zone in force for this extraction", which is
    what the historical code did with a single value serving as both the climate
    parameter and the design rule -- and it has to stay a rule rather than a
    repeated literal, because the frost zone reaches the extraction as two
    different configured values (``frost_zone_dataset`` when the pool is built,
    ``frost_zone_serving`` when Wroclaw is scored) and one number in the file
    could only mirror one of them (plan16 §2 P0 point 5).

    Callers that record the extraction settings record the value this returns,
    not the ``None``: a manifest whose reader has to apply a rule to learn what a
    build ran under is not a record of what it ran under.
    """
    return frost_zone if required_cover is None else required_cover


SLOPE_MULTIPLIERS: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)
"""Slope factors retried by the minimum diameter search when a slope is too small."""

SEARCH_OUTCOME_MESSAGES: frozenset[str] = frozenset(
    {
        formulas.SLOPE_TOO_SMALL_MESSAGE,
        formulas.SLOPE_TOO_LARGE_MESSAGE,
        formulas.FILLING_EXCEEDS_DIAMETER_MESSAGE,
    }
)
"""``ValueError`` messages of :func:`stormlab.hydraulics.formulas.calc_filling` that state a
result about the network rather than a defect in the call.

The minimum diameter search is a trial-and-error loop: "this slope is outside the
admissible band for this diameter" and "the pipe fills up before it carries the
flow" are how a trial reports that a candidate does not work, and the legacy
search relies on exactly that to move to the next candidate. Every other
``ValueError`` from ``calc_filling`` (a diameter outside the standard range, a
negative flow) means the *caller* is wrong, is not an outcome of the search, and
is raised with the row that produced it instead of being swallowed.

Measured on the six golden networks: 72 x ``SLOPE_TOO_SMALL_MESSAGE`` and 37 x
``FILLING_EXCEEDS_DIAMETER_MESSAGE``, no other message."""

LOW_FILLING_RATIO_THRESHOLD = 0.3
TARGET_FILLING_RATIO = 0.5
VERY_SMALL_FLOW_THRESHOLD = 0.01
MIN_PRACTICAL_DIAMETER = 0.3
SUBCATCHMENT_PROPAGATION_ITERATIONS = 5

FeatureMinSlopeRule = Literal["hydromechanical"]
FlowLaw = Literal["legacy_k_d25", "manning_v2"]

FLOW_LAWS: tuple[str, ...] = ("legacy_k_d25", "manning_v2")

INTERMEDIATE_COLUMNS: tuple[str, ...] = (
    "MinRequiredSlope",
    "MaxAllowableSlope",
    "MinDiameter",
    "Filling",
    "MaxQ",
    "InletGroundCover",
    "OutletGroundCover",
    "SbcCategory",
)
"""Columns compared alongside the 33 features so a parity break points at a step."""

REQUIRED_CONDUIT_COLUMNS: tuple[str, ...] = (
    "MaxDPerc",
    "Geom1",
    "MaxV",
    "MaxQ",
    "SlopeFtPerFt",
    "Length",
    "InletNode",
    "OutletNode",
    "InletNodeInvert",
    "OutletNodeInvert",
    "Roughness",
)

REQUIRED_NODE_COLUMNS: tuple[str, ...] = ("MaxDepth", "Subcatchment", "SbcCategory")


class ConduitFeatureError(ValueError):
    """Raised when the conduit or node frames cannot support feature extraction."""


class ConduitFeatureSteps:
    """The individual extraction steps, each mutating ``dfc`` (and sometimes ``dfn``).

    Kept as a class because the steps share state and because the migrated unit
    tests exercise them one at a time. Callers that want features should use
    :func:`extract_conduit_features`, which fixes the order.
    """

    def __init__(
        self,
        dfc: pd.DataFrame,
        dfn: pd.DataFrame | None,
        frost_zone: float,
        *,
        flow_constant: float = DEFAULT_FLOW_CONSTANT,
        flow_law: FlowLaw = "legacy_k_d25",
        nmaxq_clip: bool = True,
        hydraulic_validation_min_slope_rule: MinSlopeRule = "imhoff",
        max_filling_ratio: float = MAX_FILLING_RATIO,
        v_min: float = min_velocity_value,
        max_depth: float = max_depth_value,
        required_cover: float | None = None,
        theta: float = DEFAULT_THETA,
        g: float = DEFAULT_G,
        manning_coefficient: float = MANNING_COEFFICIENT,
        min_roughness: float = DEFAULT_MIN_ROUGHNESS,
        max_roughness: float = DEFAULT_MAX_ROUGHNESS,
    ) -> None:
        self.dfc = dfc
        self.dfn = dfn
        self.frost_zone = frost_zone
        # Not `self.max_depth`: `max_depth()` is already the step that copies the
        # node depths onto the conduit ends, and an attribute of that name would
        # shadow it. The two are different quantities -- a measured manhole depth
        # and the admissible ceiling on burial -- so they get different names
        # rather than one of them being renamed into the other's shape.
        self.max_admissible_depth = max_depth
        self.required_cover = resolve_required_cover(frost_zone, required_cover)
        self.flow_constant = flow_constant
        self.flow_law = flow_law
        self.nmaxq_clip = nmaxq_clip
        self.hydraulic_validation_min_slope_rule = hydraulic_validation_min_slope_rule
        self.max_filling_ratio = max_filling_ratio
        self.v_min = v_min
        self.theta = theta
        self.g = g
        self.manning_coefficient = manning_coefficient
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness

    def calculate_filling(self) -> None:
        """Water level in the pipe [m] as ``MaxDPerc * Geom1``."""
        if self.dfc is not None:
            self.dfc["Filling"] = self.dfc["MaxDPerc"] * self.dfc["Geom1"]

    def filling_is_valid(self) -> None:
        """``ValMaxFill``: 1 when the filling stays within the usable filling of the pipe."""
        if self.dfc is not None:
            self.dfc["ValMaxFill"] = self.dfc.apply(
                lambda row: validate_filling(row["Filling"], row["Geom1"], ratio=self.max_filling_ratio), axis=1
            ).astype(int)

    def velocity_is_valid(self) -> None:
        """``ValMaxV`` / ``ValMinV``: velocity against the admissible envelope."""
        if self.dfc is not None:
            self.dfc["ValMaxV"] = self.dfc.apply(lambda r: validate_max_velocity(r["MaxV"]), axis=1).astype(int)
            self.dfc["ValMinV"] = self.dfc.apply(lambda r: validate_min_velocity(r["MaxV"], v_min=self.v_min), axis=1).astype(int)

    def slope_per_mile(self) -> None:
        """``SlopePerMile``: the SWMM slope [m/m] expressed in per mille."""
        if self.dfc is not None:
            self.dfc["SlopePerMile"] = self.dfc["SlopeFtPerFt"] * 1000

    def slopes_is_valid(self) -> None:
        """``ValMaxSlope`` / ``ValMinSlope`` against the columns written by :meth:`normalize_slope`."""
        if self.dfc is not None:
            self.dfc["ValMaxSlope"] = self.dfc.apply(lambda r: 1 if r["SlopePerMile"] <= r["MaxAllowableSlope"] else 0, axis=1)
            self.dfc["ValMinSlope"] = self.dfc.apply(lambda r: 1 if r["SlopePerMile"] >= r["MinRequiredSlope"] else 0, axis=1)

    def max_depth(self) -> None:
        """Copy the node ``MaxDepth`` onto both ends of each conduit."""
        if self.dfc is not None and self.dfn is not None:
            self.dfc["InletMaxDepth"] = self.dfc["InletNode"].map(self.dfn["MaxDepth"])
            self.dfc["OutletMaxDepth"] = self.dfc["OutletNode"].map(self.dfn["MaxDepth"])

    def calculate_max_depth(self) -> None:
        """Infer a missing outlet depth from the inlet depth and the pipe fall."""
        if self.dfc is not None:
            nan_rows = pd.isna(self.dfc["OutletMaxDepth"])
            self.dfc.loc[nan_rows, "OutletMaxDepth"] = self.dfc.loc[nan_rows, "InletMaxDepth"] - (
                self.dfc.loc[nan_rows, "Length"] * self.dfc.loc[nan_rows, "SlopeFtPerFt"]
            )

    def calculate_ground_elevation(self) -> None:
        """Ground elevation at both ends as invert plus manhole depth."""
        if self.dfc is not None:
            self.dfc["InletGroundElevation"] = self.dfc["InletNodeInvert"] + self.dfc["InletMaxDepth"]
            self.dfc["OutletGroundElevation"] = self.dfc["OutletNodeInvert"] + self.dfc["OutletMaxDepth"]

    def ground_cover(self) -> None:
        """Soil cover above the pipe crown at both ends [m].

        Raises:
            ConduitFeatureError: if the elevation columns are missing, i.e. the
                step order was broken.
        """
        if self.dfc is None:
            return
        required_cols = ["InletGroundElevation", "InletNodeInvert", "Geom1"]
        if not all(col in self.dfc.columns for col in required_cols):
            raise ConduitFeatureError(f"Missing required columns: {required_cols}")

        self.dfc["InletGroundCover"] = self.dfc["InletGroundElevation"] - self.dfc["InletNodeInvert"] - self.dfc["Geom1"]
        self.dfc["OutletGroundCover"] = self.dfc["OutletGroundElevation"] - self.dfc["OutletNodeInvert"] - self.dfc["Geom1"]

    def max_ground_cover_is_valid(self) -> None:
        """``ValDepth``: 1 when neither end is deeper than the admissible depth."""
        if self.dfc is not None:
            self.dfc["ValDepth"] = (
                (self.dfc["InletNodeInvert"] >= (self.dfc["InletGroundElevation"] - self.max_admissible_depth))
                & (self.dfc["OutletNodeInvert"] >= (self.dfc["OutletGroundElevation"] - self.max_admissible_depth))
            ).astype(int)

    def min_ground_cover_is_valid(self) -> None:
        """``ValCoverage``: 1 when both ends carry at least the required cover.

        The requirement, not the frost zone: the two were one number until
        plan16 §2 P0 point 5 separated them, and they still are wherever
        ``config.required_cover`` is ``None``.
        """
        if self.dfc is None:
            return
        if "InletGroundCover" not in self.dfc.columns:
            self.ground_cover()
        self.dfc["ValCoverage"] = (
            (self.dfc["InletGroundCover"] >= self.required_cover) & (self.dfc["OutletGroundCover"] >= self.required_cover)
        ).astype(int)

    def min_conduit_diameter(self) -> None:
        """``MinDiameter``: smallest standard diameter that still conveys ``MaxQ``.

        Workflow per row:

        1. Filling already out of limits (``ValMaxFill != 1``) -> step up to the
           next standard diameter that fits the filling.
        2. Filling above the usable filling of the current diameter -> same.
        3. Very small flow -> the smallest practical diameter.
        4. Non-standard diameter -> left untouched.
        5. Current diameter insufficient for the computed flow -> step up.
        6. Otherwise walk down the standard diameters, retrying each with a
           steeper slope while the failure is "slope too small".
        7. If nothing moved and the pipe is barely filled, aim at the target
           filling ratio instead.
        """
        if self.dfc is None:
            return

        common_diameters_sorted: list[float] = sorted(common_diameters)
        diam_index_map: dict[float, int] = {d: i for i, d in enumerate(common_diameters_sorted)}
        min_slope_rule = self.hydraulic_validation_min_slope_rule
        max_filling_ratio = self.max_filling_ratio
        theta = self.theta
        g = self.g

        def get_max_filling_height(diameter: float) -> float:
            """Maximum allowed filling height for a given diameter."""
            return max_filling(diameter, max_filling_ratio)

        def find_larger_diameter(current_diam: float, required_filling: float) -> float:
            """Next larger standard diameter that can hold the required filling."""
            idx = diam_index_map.get(current_diam, -1)
            if idx != -1 and idx + 1 < len(common_diameters_sorted):
                for larger_diam in common_diameters_sorted[idx + 1 :]:
                    max_filling_height = get_max_filling_height(larger_diam)
                    if required_filling <= max_filling_height:
                        return larger_diam
            logger.warning("Could not find a suitable larger diameter for filling: %s", required_filling)
            return current_diam

        def handle_excessive_filling(row: pd.Series) -> float:
            """Case where the filling already exceeds the limits."""
            current_diam = row["Geom1"]
            current_filling = row.get("Filling", 0.0)
            return find_larger_diameter(current_diam, current_filling)

        def try_smaller_diameters(
            conduit_name: object, current_diam: float, idx: int, max_q: float, original_slope: float
        ) -> float:
            """Walk down the standard diameters while the flow still fits."""
            best_diam = current_diam

            for candidate in reversed(common_diameters_sorted[:idx]):
                diameter_works = False

                for multiplier in SLOPE_MULTIPLIERS:
                    adjusted_slope = original_slope * multiplier

                    try:
                        fill_height = formulas.calc_filling(
                            max_q,
                            candidate,
                            adjusted_slope,
                            min_slope_rule=min_slope_rule,
                            max_filling_ratio=max_filling_ratio,
                            theta=theta,
                            g=g,
                        )

                        if validate_filling(fill_height, candidate, ratio=max_filling_ratio):
                            best_diam = candidate
                            diameter_works = True
                            break
                        break  # This diameter does not work even with this slope
                    except ValueError as error:
                        _reraise_unless_search_outcome(error, conduit_name, candidate, adjusted_slope)
                        # The message is the signal: only a too-small slope is worth retrying.
                        if str(error) == formulas.SLOPE_TOO_SMALL_MESSAGE and multiplier < SLOPE_MULTIPLIERS[-1]:
                            continue
                        break

                if not diameter_works:
                    break

            return best_diam

        def handle_small_filling_ratio(current_diam: float, current_filling: float) -> float:
            """Case where the pipe is barely filled: aim at the target filling ratio."""
            if current_filling / current_diam < LOW_FILLING_RATIO_THRESHOLD:
                target_diameter = current_filling / TARGET_FILLING_RATIO

                for candidate in common_diameters_sorted:
                    if candidate >= target_diameter and candidate < current_diam:
                        return candidate

            return current_diam

        def find_min_diam(row: pd.Series) -> float:
            if row.get("ValMaxFill", 0) != 1:
                return handle_excessive_filling(row)

            current_diam = float(row["Geom1"])
            current_filling = float(row.get("Filling", 0.0) or 0.0)
            max_q = float(row.get("MaxQ", 0.0) or 0.0)

            max_allowed_filling_height = get_max_filling_height(current_diam)
            if current_filling > max_allowed_filling_height:
                return find_larger_diameter(current_diam, current_filling)

            if max_q < VERY_SMALL_FLOW_THRESHOLD:
                for candidate in common_diameters_sorted:
                    if candidate >= MIN_PRACTICAL_DIAMETER:
                        return candidate
                return common_diameters_sorted[0]

            idx: int = diam_index_map.get(current_diam, -1)
            if idx == -1:  # Non-standard diameter - leave as is
                return current_diam

            original_slope = float(row.get("SlopeFtPerFt", 0.0) or 0.0)

            try:
                current_fill_height = formulas.calc_filling(
                    max_q,
                    current_diam,
                    original_slope,
                    min_slope_rule=min_slope_rule,
                    max_filling_ratio=max_filling_ratio,
                    theta=theta,
                    g=g,
                )

                if not validate_filling(current_fill_height, current_diam, ratio=max_filling_ratio):
                    logger.info("ROW %s: Current diameter is insufficient for calculated flow", row.name)

                    for larger_diam in common_diameters_sorted[idx + 1 :] if idx + 1 < len(common_diameters_sorted) else []:
                        try:
                            larger_fill_height = formulas.calc_filling(
                                max_q,
                                larger_diam,
                                original_slope,
                                min_slope_rule=min_slope_rule,
                                max_filling_ratio=max_filling_ratio,
                                theta=theta,
                                g=g,
                            )

                            if validate_filling(larger_fill_height, larger_diam, ratio=max_filling_ratio):
                                logger.info("ROW %s: Increasing diameter from %sm to %sm", row.name, current_diam, larger_diam)
                                return larger_diam
                        except ValueError as error:
                            _reraise_unless_search_outcome(error, row.name, larger_diam, original_slope)
                            continue  # This larger diameter does not work either; try the next one.
            except ValueError as error:
                _reraise_unless_search_outcome(error, row.name, current_diam, original_slope)
                # The current diameter cannot convey the flow at this slope; fall
                # through to the search for a smaller one, as the legacy code did.

            best_diam = try_smaller_diameters(row.name, current_diam, idx, max_q, original_slope)

            if best_diam == current_diam:
                best_diam = handle_small_filling_ratio(current_diam, current_filling)

            return best_diam

        self.dfc["MinDiameter"] = self.dfc.apply(find_min_diam, axis=1)

    def diameter_features(self) -> None:
        """``isMinDiameter`` / ``IncreaseDia`` / ``ReduceDia`` from ``Geom1`` vs ``MinDiameter``."""
        if self.dfc is not None:
            self.dfc["isMinDiameter"] = np.where(self.dfc["Geom1"] == self.dfc["MinDiameter"], 1, 0)
            self.dfc["IncreaseDia"] = np.where(self.dfc["Geom1"] < self.dfc["MinDiameter"], 1, 0)
            self.dfc["ReduceDia"] = np.where(self.dfc["Geom1"] > self.dfc["MinDiameter"], 1, 0)

    def normalize_roughness(self) -> None:
        """``NRoughness``: Manning's n mapped onto [0, 1] over the PVC-to-stone range."""
        if self.dfc is None:
            return

        roughness_clipped = self.dfc["Roughness"].clip(self.min_roughness, self.max_roughness)
        self.dfc["NRoughness"] = (roughness_clipped - self.min_roughness) / (self.max_roughness - self.min_roughness)

    def normalize_max_velocity(self) -> None:
        """``NMaxV``: velocity as a fraction of the admissible maximum."""
        if self.dfc is None:
            return

        max_v_clipped = self.dfc["MaxV"].clip(0, max_velocity_value)
        self.dfc["NMaxV"] = max_v_clipped / max_velocity_value

    def normalize_depth(self) -> None:
        """``NInletDepth`` / ``NOutletDepth`` over [frost zone + diameter, max depth].

        Rows whose admissible range collapses to zero stay at 0.0 rather than
        dividing by zero.

        The lower end stays the **frost zone**, not the required cover. This is a
        contract v1/v2 column of the historical F0/F1 path, which plan16 §2 P3
        keeps untouched -- the physical view F2 is a new module rather than a
        re-parameterised ``normalize_*`` -- so moving its floor onto a parameter
        the C1 series may change would silently redefine a frozen column.
        """
        if self.dfc is None:
            return

        min_depth = self.frost_zone + self.dfc["Geom1"]
        inlet_depth_clipped = self.dfc["InletMaxDepth"].clip(lower=min_depth, upper=self.max_admissible_depth)
        outlet_depth_clipped = self.dfc["OutletMaxDepth"].clip(lower=min_depth, upper=self.max_admissible_depth)
        depth_range = self.max_admissible_depth - min_depth
        valid_range = depth_range > 0
        self.dfc["NInletDepth"] = 0.0
        self.dfc["NOutletDepth"] = 0.0
        self.dfc.loc[valid_range, "NInletDepth"] = (
            inlet_depth_clipped.loc[valid_range] - min_depth.loc[valid_range]
        ) / depth_range.loc[valid_range]
        self.dfc.loc[valid_range, "NOutletDepth"] = (
            outlet_depth_clipped.loc[valid_range] - min_depth.loc[valid_range]
        ) / depth_range.loc[valid_range]

    def normalize_filling(self) -> None:
        """``NFilling``: filling relative to usable filling.

        A value greater than or equal to 1 means the pipe has reached or
        exceeded the usable threshold, ``0.827 * D`` under the default profile.
        """
        if self.dfc is None:
            return

        max_effective_filling = self.max_filling_ratio * self.dfc["Geom1"]
        self.dfc["NFilling"] = self.dfc["Filling"] / max_effective_filling

    def normalize_max_q(self) -> None:
        """``NMaxQ``: observed flow divided by theoretical full-pipe capacity.

        ``legacy_k_d25`` uses ``flow_constant * D**2.5 * sqrt(S_per_mille)``.
        ``manning_v2`` uses the circular full-pipe Manning equation with
        per-conduit ``Roughness`` and slope in m/m. ``nmaxq_clip`` optionally
        clips either ratio to ``[0, 1]``; disabling it preserves overload values.
        Lower bounds are validated for every ``manning_v2`` run and for an
        unclipped legacy run. Legacy with clipping deliberately retains its
        historical behaviour without this validation.

        Raises:
            ConduitFeatureError: if an enforced physical lower bound is
                violated, or if ``flow_law`` is unsupported.
        """
        if self.dfc is None:
            return

        if self.flow_law not in FLOW_LAWS:
            raise ConduitFeatureError(f"Unknown flow_law {self.flow_law!r}")

        if not self.nmaxq_clip or self.flow_law == "manning_v2":
            self._validate_nmaxq_lower_bounds()

        if self.flow_law == "legacy_k_d25":
            theoretical_max_q = self.flow_constant * (self.dfc["Geom1"] ** 2.5) * (self.dfc["SlopePerMile"] ** 0.5)
        elif self.flow_law == "manning_v2":
            theoretical_max_q = (
                MANNING_FULL_PIPE_COEFFICIENT
                / self.dfc["Roughness"]
                * self.dfc["Geom1"] ** (8 / 3)
                * self.dfc["SlopeFtPerFt"] ** 0.5
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = self.dfc["MaxQ"] / theoretical_max_q
        if not self.nmaxq_clip:
            zero_flow_at_zero_slope = (self.dfc["SlopeFtPerFt"] == 0) & (self.dfc["MaxQ"] == 0)
            normalized.loc[zero_flow_at_zero_slope] = 0.0
        self.dfc["NMaxQ"] = normalized.clip(0, 1) if self.nmaxq_clip else normalized

    def _validate_nmaxq_lower_bounds(self) -> None:
        """Reject the first conduit with an invalid physical lower bound."""
        assert self.dfc is not None
        requirements = {"Geom1": "> 0", "SlopeFtPerFt": ">= 0", "MaxQ": ">= 0"}
        zero_slope_with_flow = self.dfc["SlopeFtPerFt"].eq(0) & self.dfc["MaxQ"].gt(0)
        invalid = {
            "Geom1": ~self.dfc["Geom1"].gt(0),
            "SlopeFtPerFt": ~self.dfc["SlopeFtPerFt"].ge(0) | zero_slope_with_flow,
            "MaxQ": ~self.dfc["MaxQ"].ge(0),
        }
        if self.flow_law == "manning_v2":
            requirements = {"Geom1": "> 0", "Roughness": "> 0", **requirements}
            invalid = {"Geom1": invalid["Geom1"], "Roughness": ~self.dfc["Roughness"].gt(0), **invalid}

        invalid_frame = pd.DataFrame(invalid, index=self.dfc.index)
        invalid_rows = invalid_frame.any(axis="columns").to_numpy()
        if invalid_rows.any():
            position = int(np.flatnonzero(invalid_rows)[0])
            column = str(invalid_frame.columns[invalid_frame.iloc[position].to_numpy()][0])
            conduit = self.dfc.index[position]
            value = self.dfc.iloc[position][column]
            if column == "SlopeFtPerFt" and zero_slope_with_flow.iloc[position]:
                max_q = self.dfc.iloc[position]["MaxQ"]
                raise ConduitFeatureError(f"Conduit {conduit!r} has SlopeFtPerFt=0 with MaxQ={max_q!r}; NMaxQ would be infinite")
            raise ConduitFeatureError(f"Conduit {conduit!r} has {column}={value!r}; expected {column} {requirements[column]}")

    def normalize_ground_cover(self) -> None:
        """``NInletGroundCover`` / ``NOutletGroundCover`` over [required cover, max depth - diameter].

        0 means the minimum acceptable cover, 1 the deepest possible burial. The
        lower end is the cover *requirement*, which equals the frost zone wherever
        ``config.required_cover`` is ``None`` (plan16 §2 P0 point 5).
        """
        if self.dfc is None:
            return

        if "InletGroundCover" not in self.dfc.columns or "OutletGroundCover" not in self.dfc.columns:
            self.ground_cover()

        min_cover = self.required_cover
        max_cover = self.max_admissible_depth - self.dfc["Geom1"]

        inlet_cover_clipped = self.dfc["InletGroundCover"].clip(lower=min_cover, upper=max_cover)
        outlet_cover_clipped = self.dfc["OutletGroundCover"].clip(lower=min_cover, upper=max_cover)

        cover_range = max_cover - min_cover
        valid_range = cover_range > 0

        self.dfc["NInletGroundCover"] = 0.0
        self.dfc["NOutletGroundCover"] = 0.0

        self.dfc.loc[valid_range, "NInletGroundCover"] = (inlet_cover_clipped.loc[valid_range] - min_cover) / cover_range.loc[
            valid_range
        ]
        self.dfc.loc[valid_range, "NOutletGroundCover"] = (outlet_cover_clipped.loc[valid_range] - min_cover) / cover_range.loc[
            valid_range
        ]

    def _min_slope(self, filling: float, diameter: float) -> float:
        """The hydromechanical minimum slope under this run's ``theta``/``g``."""
        return min_slope(filling, diameter, theta=self.theta, g=self.g)

    def _max_slope(self, diameter: float) -> float:
        """The maximum slope under this run's ``theta``/``g``/``manning``.

        Computed per call rather than looked up in
        :data:`stormlab.hydraulics.round.max_slopes`: that dictionary is built at
        import with the default constants and would answer with them however the
        run is configured.
        """
        return max_slope(diameter, theta=self.theta, g=self.g, manning=self.manning_coefficient)

    def normalize_slope(self) -> None:
        """``MinRequiredSlope``, ``MaxAllowableSlope`` and ``NSlope`` in [0, 1].

        The minimum required slope is the hydromechanical criterion of
        :func:`stormlab.hydraulics.round.min_slope` — deliberately a different
        rule from the Imhoff check used by the diameter search.
        """
        if self.dfc is None or self.dfc.empty:
            return

        self.dfc["MinRequiredSlope"] = self.dfc.apply(lambda r: self._min_slope(r["Filling"], r["Geom1"]), axis=1)
        self.dfc["MaxAllowableSlope"] = self.dfc.apply(lambda r: self._max_slope(r["Geom1"]), axis=1)

        def normalize(row: pd.Series) -> float:
            min_slope_val = row["MinRequiredSlope"]
            max_slope_val = row["MaxAllowableSlope"]
            actual_slope = row["SlopePerMile"]

            if min_slope_val >= max_slope_val:  # If min and max are too close
                return 0.5 if actual_slope >= min_slope_val else 0.0

            if actual_slope <= min_slope_val:
                return 0.0
            elif actual_slope >= max_slope_val:
                return 1.0
            else:
                return float((actual_slope - min_slope_val) / (max_slope_val - min_slope_val))

        self.dfc["NSlope"] = self.dfc.apply(normalize, axis=1)

    def slope_increase(self) -> None:
        """``IncreaseSlope``: 1 when the pipe is laid flatter than the minimum required."""
        if self.dfc is None:
            return

        if "MinRequiredSlope" not in self.dfc.columns:
            self.dfc["MinRequiredSlope"] = self.dfc.apply(lambda r: self._min_slope(r["Filling"], r["Geom1"]), axis=1)

        self.dfc["IncreaseSlope"] = np.where(self.dfc["SlopePerMile"] < self.dfc["MinRequiredSlope"], 1, 0)

    def slope_reduction(self) -> None:
        """``ReduceSlope``: 1 when the pipe is laid steeper than the maximum allowed."""
        if self.dfc is None or self.dfc.empty:
            return

        if "MaxAllowableSlope" not in self.dfc.columns:
            self.dfc["MaxAllowableSlope"] = self.dfc.apply(lambda r: self._max_slope(r["Geom1"]), axis=1)

        self.dfc["ReduceSlope"] = np.where(self.dfc["SlopePerMile"] > self.dfc["MaxAllowableSlope"], 1, 0)

    def conduits_subcatchment_info(self) -> None:
        """Attach the subcatchment of the outlet node, falling back to the inlet node."""
        if self.dfc is None or self.dfn is None:
            return

        if "Subcatchment" not in self.dfn.columns or "SbcCategory" not in self.dfn.columns:
            return

        self.dfc["Subcatchment"] = "-"
        self.dfc["SbcCategory"] = "-"

        for idx, row in self.dfc.iterrows():
            inlet_node = row.get("InletNode")
            outlet_node = row.get("OutletNode")

            if outlet_node in self.dfn.index and self.dfn.at[outlet_node, "Subcatchment"] != "-":
                self.dfc.at[idx, "Subcatchment"] = self.dfn.at[outlet_node, "Subcatchment"]
                self.dfc.at[idx, "SbcCategory"] = self.dfn.at[outlet_node, "SbcCategory"]
            elif inlet_node in self.dfn.index and self.dfn.at[inlet_node, "Subcatchment"] != "-":
                self.dfc.at[idx, "Subcatchment"] = self.dfn.at[inlet_node, "Subcatchment"]
                self.dfc.at[idx, "SbcCategory"] = self.dfn.at[inlet_node, "SbcCategory"]

    def propagate_subcatchment_info(self) -> None:
        """Spread subcatchment membership upstream to nodes and conduits that lack it.

        Mutates ``dfn`` as well as ``dfc``: a node without a subcatchment inherits
        from any of its inlet nodes, and the result feeds the next iteration.
        """
        if self.dfc is None or self.dfn is None:
            return

        node_connections: dict[object, list[object]] = {}

        for _idx, row in self.dfc.iterrows():
            inlet_node = row.get("InletNode")
            outlet_node = row.get("OutletNode")

            if outlet_node not in node_connections:
                node_connections[outlet_node] = []

            if inlet_node:
                node_connections[outlet_node].append(inlet_node)

        for _ in range(SUBCATCHMENT_PROPAGATION_ITERATIONS):
            changes_made = False

            for outlet_node, inlet_nodes in node_connections.items():
                if outlet_node in self.dfn.index and self.dfn.at[outlet_node, "Subcatchment"] == "-":
                    for inlet_node in inlet_nodes:
                        if inlet_node in self.dfn.index and self.dfn.at[inlet_node, "Subcatchment"] != "-":
                            self.dfn.at[outlet_node, "Subcatchment"] = self.dfn.at[inlet_node, "Subcatchment"]
                            self.dfn.at[outlet_node, "SbcCategory"] = self.dfn.at[inlet_node, "SbcCategory"]
                            changes_made = True
                            break

            for idx, row in self.dfc.iterrows():
                if self.dfc.at[idx, "Subcatchment"] == "-":
                    inlet_node = row.get("InletNode")
                    outlet_node = row.get("OutletNode")

                    if outlet_node in self.dfn.index and self.dfn.at[outlet_node, "Subcatchment"] != "-":
                        self.dfc.at[idx, "Subcatchment"] = self.dfn.at[outlet_node, "Subcatchment"]
                        self.dfc.at[idx, "SbcCategory"] = self.dfn.at[outlet_node, "SbcCategory"]
                        changes_made = True
                    elif inlet_node in self.dfn.index and self.dfn.at[inlet_node, "Subcatchment"] != "-":
                        self.dfc.at[idx, "Subcatchment"] = self.dfn.at[inlet_node, "Subcatchment"]
                        self.dfc.at[idx, "SbcCategory"] = self.dfn.at[inlet_node, "SbcCategory"]
                        changes_made = True

            if not changes_made:
                break

    def encode_sbc_category(self) -> None:
        """One-hot encode ``SbcCategory``, always emitting all twelve land cover columns.

        A conduit that no subcatchment reaches gets zeros across the whole block:
        both spellings of "no land cover information" — ``NaN`` and the ``"-"``
        sentinel written by :meth:`conduits_subcatchment_info` — are the
        documented absence branch. Any *other* value is rejected, because a
        silently zeroed block would make a mistyped class indistinguishable from
        a documented absence while still costing the model all twelve columns.

        Raises:
            ConduitFeatureError: naming the conduits and the categories that are
                not part of :data:`FEATURE_ONEHOT_ORDER`.
        """
        if self.dfc is None or len(self.dfc) == 0:
            return

        if "SbcCategory" not in self.dfc.columns:
            return

        try:
            encoded = encode_categories(self.dfc["SbcCategory"], FEATURE_ONEHOT_ORDER, what="conduit")
        except SubcatchmentFeatureError as error:
            raise ConduitFeatureError(f"Cannot one-hot encode SbcCategory: {error}") from error

        for column in encoded.columns:
            self.dfc[column] = encoded[column]


def extract_conduit_features(
    dfc: pd.DataFrame,
    dfn: pd.DataFrame,
    frost_zone: float,
    *,
    flow_constant: float = DEFAULT_FLOW_CONSTANT,
    flow_law: FlowLaw = "legacy_k_d25",
    nmaxq_clip: bool = True,
    feature_min_slope_rule: FeatureMinSlopeRule = "hydromechanical",
    hydraulic_validation_min_slope_rule: MinSlopeRule = "imhoff",
    max_filling_ratio: float = MAX_FILLING_RATIO,
    v_min: float = min_velocity_value,
    max_depth: float = max_depth_value,
    required_cover: float | None = None,
    theta: float = DEFAULT_THETA,
    g: float = DEFAULT_G,
    manning_coefficient: float = MANNING_COEFFICIENT,
    min_roughness: float = DEFAULT_MIN_ROUGHNESS,
    max_roughness: float = DEFAULT_MAX_ROUGHNESS,
) -> pd.DataFrame:
    """Run the whole conduit feature pipeline and return an enriched copy of ``dfc``.

    Args:
        dfc: conduit frame as read from the INP and its report.
        dfn: node frame, already carrying ``Subcatchment``/``SbcCategory`` from
            :func:`stormlab.features.nodes.assign_subcatchment_info`.
        frost_zone: frost penetration depth [m]; 1.6 for the dataset and its
            labels, 0.8 for the chapter-4 serving run.
        flow_constant: ``k`` of the theoretical capacity used by ``NMaxQ``.
        flow_law: capacity law used as the denominator of ``NMaxQ``. The legacy
            law uses ``flow_constant``; ``manning_v2`` uses each conduit's
            ``Roughness``.
        nmaxq_clip: whether ``NMaxQ`` is clipped to ``[0, 1]``. Disabling it
            exposes overload. ``manning_v2`` validates denominator inputs with
            either setting; clipped legacy keeps the historical permissive path.
        feature_min_slope_rule: rule behind ``MinRequiredSlope``, ``NSlope``,
            ``ValMinSlope`` and ``IncreaseSlope``.
        hydraulic_validation_min_slope_rule: rule gating the flow computations
            inside the minimum diameter search.
        max_filling_ratio: relative filling at full capacity. One value drives
            every place the legacy code hard-coded 0.827: ``ValMaxFill``, the
            usable filling height and the flow gate of the minimum diameter
            search, and ``NFilling``. Changing it therefore moves the flags, the
            recommended diameter and the normalised filling together instead of
            leaving them in disagreement.
        v_min: self-cleansing velocity behind ``ValMinV`` [m/s].
        max_depth: deepest admissible pipe invert below ground [m]. The threshold
            behind ``ValDepth`` and the upper end of both the depth and the cover
            normalisation, so it moves the three together. It was a module literal
            (``hydraulics.round.MAX_DEPTH``), which is why it defaults to that
            value here and is configured through ``config.max_depth``.
        required_cover: soil cover a conduit must carry above its crown [m], behind
            ``ValCoverage`` and the lower end of the cover normalisation. ``None``
            means the frost zone in force for this extraction, which is what the
            historical pipeline did with one value serving as both
            (:func:`resolve_required_cover`).
        theta: shear stress of the hydromechanical minimum slope criterion [Pa].
            Moves ``MinRequiredSlope``, ``MaxAllowableSlope``, ``NSlope``,
            ``ValMinSlope``, ``ValMaxSlope``, ``IncreaseSlope`` and ``ReduceSlope``
            together.
        g: specific weight of water ``gamma = rho * g`` [kN/m3]. With ``theta``
            in Pa, ``theta / g`` is numerically in millimetres and the canonical
            minimum-slope function returns per mille directly.
        manning_coefficient: Manning roughness behind the velocity the maximum
            slope is bisected against.
        min_roughness: lower end of the range ``NRoughness`` maps onto ``[0, 1]``.
        max_roughness: upper end of that range.

    Returns:
        A copy of ``dfc`` with the feature, flag and intermediate columns added.
        Neither input frame is modified.

    Raises:
        ConduitFeatureError: if a required column is missing from either frame,
            or if ``feature_min_slope_rule`` is not supported.
    """
    if feature_min_slope_rule != "hydromechanical":
        raise ConduitFeatureError(
            f"feature_min_slope_rule={feature_min_slope_rule!r} is not supported: the slope features compare against "
            "SlopePerMile, and only the hydromechanical criterion is expressed in per mille"
        )
    _assert_columns(dfc, REQUIRED_CONDUIT_COLUMNS, "conduit")
    _assert_columns(dfn, REQUIRED_NODE_COLUMNS, "node")
    if dfc.empty:
        # Without this the pipeline dies several steps later inside pandas with
        # "Cannot set a DataFrame with multiple columns...", which says nothing
        # about the actual problem: the network carries no conduits at all.
        raise ConduitFeatureError("Conduit frame is empty: there is nothing to extract features from")

    steps = ConduitFeatureSteps(
        dfc.copy(),
        dfn.copy(),
        frost_zone,
        flow_constant=flow_constant,
        flow_law=flow_law,
        nmaxq_clip=nmaxq_clip,
        hydraulic_validation_min_slope_rule=hydraulic_validation_min_slope_rule,
        max_filling_ratio=max_filling_ratio,
        v_min=v_min,
        max_depth=max_depth,
        required_cover=required_cover,
        theta=theta,
        g=g,
        manning_coefficient=manning_coefficient,
        min_roughness=min_roughness,
        max_roughness=max_roughness,
    )

    # The order is load-bearing: each step reads columns written by earlier ones.
    steps.calculate_filling()
    steps.filling_is_valid()
    steps.velocity_is_valid()
    steps.slope_per_mile()
    steps.max_depth()
    steps.calculate_max_depth()
    steps.calculate_ground_elevation()
    steps.ground_cover()
    steps.max_ground_cover_is_valid()
    steps.min_ground_cover_is_valid()
    steps.min_conduit_diameter()
    steps.diameter_features()
    steps.conduits_subcatchment_info()
    steps.propagate_subcatchment_info()
    steps.normalize_roughness()
    steps.normalize_max_velocity()
    steps.normalize_depth()
    steps.normalize_filling()
    steps.normalize_max_q()
    steps.normalize_ground_cover()
    steps.normalize_slope()
    steps.slopes_is_valid()
    steps.slope_increase()
    steps.slope_reduction()
    steps.encode_sbc_category()

    return steps.dfc


def conduit_feature_matrix(
    df: pd.DataFrame,
    contract: FeatureContract = DEFAULT_FEATURE_CONTRACT,
) -> pd.DataFrame:
    """Project an enriched conduit frame onto the selected contract columns.

    Selects the contract columns in contract order and casts them to the contract
    dtypes. A missing column is an error: filling it with zeros is how the
    historical pipeline silently degraded a model to a single class.

    Raises:
        FeatureContractError: if a contract column is absent, cannot be cast, or
            still carries missing values after the cast.
    """
    missing = [column for column in contract.columns if column not in df.columns]
    if missing:
        raise FeatureContractError(
            f"Cannot build the feature matrix: {len(missing)} contract column(s) missing from the conduit frame: {missing}"
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

    validate_features(matrix, contract=contract)
    return matrix


def _reraise_unless_search_outcome(error: ValueError, conduit_name: object, diameter: float, slope: float) -> None:
    """Let a trial-and-error outcome through; turn anything else into a named failure.

    Raises:
        ConduitFeatureError: if ``error`` is not one of
            :data:`SEARCH_OUTCOME_MESSAGES`, carrying the conduit, the diameter
            and the slope that produced it.
    """
    if isinstance(error, ConduitFeatureError):
        # Already wrapped by an inner handler; ConduitFeatureError is a ValueError,
        # so the enclosing `except ValueError` would otherwise wrap it a second time
        # and report the diameter of the outer frame instead of the failing one.
        raise error
    if str(error) in SEARCH_OUTCOME_MESSAGES:
        return
    raise ConduitFeatureError(
        f"Minimum diameter search failed on conduit {conduit_name!r} at diameter={diameter} m, slope={slope} m/m: "
        f"{type(error).__name__}: {error}"
    ) from error


def _assert_columns(frame: pd.DataFrame, required: tuple[str, ...], what: str) -> None:
    """Fail immediately, and by name, on an input frame that cannot support the pipeline."""
    if not isinstance(frame, pd.DataFrame):
        raise ConduitFeatureError(f"Expected a DataFrame for the {what} frame, got {type(frame).__name__}")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ConduitFeatureError(f"{what.capitalize()} frame is missing {len(missing)} column(s): {missing}")


__all__ = [
    "DEFAULT_FLOW_CONSTANT",
    "FLOW_LAWS",
    "INTERMEDIATE_COLUMNS",
    "SEARCH_OUTCOME_MESSAGES",
    "REQUIRED_CONDUIT_COLUMNS",
    "REQUIRED_NODE_COLUMNS",
    "ConduitFeatureError",
    "ConduitFeatureSteps",
    "FeatureMinSlopeRule",
    "FlowLaw",
    "conduit_feature_matrix",
    "extract_conduit_features",
    "resolve_required_cover",
]
