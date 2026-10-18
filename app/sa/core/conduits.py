"""
Feature engineering services for conduits (pipes) and recommendation generation.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import (
    CIRCULAR_PIPE_FLOW_CONSTANT,
    FEATURE_COLUMNS,
    LOW_FILLING_RATIO_THRESHOLD,
    MAX_FILLING_RATIO,
    MAX_ROUGHNESS,
    MIN_PRACTICAL_DIAMETER,
    MIN_ROUGHNESS,
    SLOPE_MULTIPLIERS,
    SUBCATCHMENT_CATEGORIES,
    SUBCATCHMENT_PROPAGATION_ITERATIONS,
    TARGET_FILLING_RATIO,
    VERY_SMALL_FLOW_THRESHOLD,
)
from .enums import RecommendationCategory
from .round import common_diameters, max_depth_value, max_filling, max_slope, max_slopes, max_velocity_value, min_slope
from .services import HydraulicCalculationsService
from .valid_round import (
    validate_filling,
    validate_max_velocity,
    validate_min_velocity,
)

logger = logging.getLogger(__name__)
_CANONICAL_MAX_SLOPE = max_slope


class ConduitFeatureEngineeringService:
    def __init__(self, dfc: pd.DataFrame, dfn: pd.DataFrame, frost_zone: float):
        self.dfc = dfc
        self.dfn = dfn
        self.frost_zone = frost_zone

    def calculate_filling(self) -> None:
        """
        Calculates the filling height of conduits in meters [m].

        Filling is calculated as the product of maximum depth percentage (MaxDPerc)
        and the conduit diameter (Geom1). The result represents the actual water
        level inside the pipe in meters.

        This method adds a 'Filling' column to the conduits dataframe (self.dfc).
        """
        if self.dfc is not None:
            self.dfc["Filling"] = self.dfc["MaxDPerc"] * self.dfc["Geom1"]

    def filling_is_valid(self) -> None:
        """
        Validates if the filling height is within acceptable limits for each conduit.

        This method checks if the filling height (in meters [m]) is valid for the given
        pipe diameter using the validate_filling function. The result is stored as a
        binary value (0 or 1) in the 'ValMaxFill' column, where:
        - 1 indicates the filling is valid (within acceptable limits)
        - 0 indicates the filling exceeds the maximum allowed value

        The validation is based on the maximum filling ratio defined in the system
        (typically 80-90% of pipe diameter).
        """
        if self.dfc is not None:
            self.dfc["ValMaxFill"] = self.dfc.apply(lambda row: validate_filling(row["Filling"], row["Geom1"]), axis=1).astype(
                int
            )

    def velocity_is_valid(self) -> None:
        if self.dfc is not None:
            self.dfc["ValMaxV"] = self.dfc.apply(lambda r: validate_max_velocity(r["MaxV"]), axis=1).astype(int)
            self.dfc["ValMinV"] = self.dfc.apply(lambda r: validate_min_velocity(r["MaxV"]), axis=1).astype(int)

    def slope_per_mile(self) -> None:
        if self.dfc is not None:
            self.dfc["SlopePerMile"] = self.dfc["SlopeFtPerFt"] * 1000

    def slopes_is_valid(self) -> None:
        if self.dfc is not None:
            # Use the same functions as in normalize_slope for consistency
            self.dfc["ValMaxSlope"] = self.dfc.apply(lambda r: 1 if r["SlopePerMile"] <= r["MaxAllowableSlope"] else 0, axis=1)
            self.dfc["ValMinSlope"] = self.dfc.apply(lambda r: 1 if r["SlopePerMile"] >= r["MinRequiredSlope"] else 0, axis=1)

    def max_depth(self) -> None:
        if self.dfc is not None and self.dfn is not None:
            self.dfc["InletMaxDepth"] = self.dfc["InletNode"].map(self.dfn["MaxDepth"])
            self.dfc["OutletMaxDepth"] = self.dfc["OutletNode"].map(self.dfn["MaxDepth"])

    def calculate_max_depth(self) -> None:
        if self.dfc is not None:
            nan_rows = pd.isna(self.dfc["OutletMaxDepth"])
            self.dfc.loc[nan_rows, "OutletMaxDepth"] = self.dfc.loc[nan_rows, "InletMaxDepth"] - (
                self.dfc.loc[nan_rows, "Length"] * self.dfc.loc[nan_rows, "SlopeFtPerFt"]
            )

    def calculate_ground_elevation(self) -> None:
        if self.dfc is not None:
            self.dfc["InletGroundElevation"] = self.dfc["InletNodeInvert"] + self.dfc["InletMaxDepth"]
            self.dfc["OutletGroundElevation"] = self.dfc["OutletNodeInvert"] + self.dfc["OutletMaxDepth"]

    def ground_cover(self) -> None:
        if self.dfc is None:
            return
        required_cols = ["InletGroundElevation", "InletNodeInvert", "Geom1"]
        if not all(col in self.dfc.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        self.dfc["InletGroundCover"] = self.dfc["InletGroundElevation"] - self.dfc["InletNodeInvert"] - self.dfc["Geom1"]
        self.dfc["OutletGroundCover"] = self.dfc["OutletGroundElevation"] - self.dfc["OutletNodeInvert"] - self.dfc["Geom1"]

    def max_ground_cover_is_valid(self) -> None:
        if self.dfc is not None:
            self.dfc["ValDepth"] = (
                (self.dfc["InletNodeInvert"] >= (self.dfc["InletGroundElevation"] - max_depth_value))
                & (self.dfc["OutletNodeInvert"] >= (self.dfc["OutletGroundElevation"] - max_depth_value))
            ).astype(int)

    def min_ground_cover_is_valid(self) -> None:
        if self.dfc is None:
            return
        if "InletGroundCover" not in self.dfc.columns:
            self.ground_cover()
        self.dfc["ValCoverage"] = (
            (self.dfc["InletGroundCover"] >= self.frost_zone) & (self.dfc["OutletGroundCover"] >= self.frost_zone)
        ).astype(int)

    def min_conduit_diameter(self) -> None:
        """
        Determine the smallest standard conduit diameter that can safely convey
        the design flow (MaxQ) while meeting the maximum filling constraint.

        A new column *MinDiameter* is added to self.dfc.

        Workflow
        --------
        1. For rows where filling already exceeds limits (ValMaxFill == 0),
            increase the diameter to the next larger standard size that can handle the flow.
        2. For rows where filling is within limits (ValMaxFill == 1):
            a. Find the index of the current diameter in *common_diameters*.
                If it is not present, leave the diameter unchanged.
            b. Check if current filling exceeds maximum allowed filling (0.827 * diameter).
                If so, increase the diameter to the next larger standard size.
            c. Iterate over smaller standard diameters (descending).
                • For each candidate compute the filling height required to pass MaxQ.
                • If validate_filling succeeds, update *best_diam* and continue.
                • If it fails due to slope constraints, try with increased slope.
                • If it fails due to other reasons, break the loop.
        """
        if self.dfc is None:
            return

        common_diameters_sorted: List[float] = sorted(common_diameters)
        diam_index_map: Dict[float, int] = {d: i for i, d in enumerate(common_diameters_sorted)}

        def get_max_filling_height(diameter: float) -> float:
            """Calculate maximum allowed filling height for a given diameter."""
            calculated_max_filling_height = max_filling(diameter)
            return calculated_max_filling_height

        def find_larger_diameter(current_diam: float, required_filling: float) -> float:
            """Find the next larger diameter that can handle the required filling."""
            idx = diam_index_map.get(current_diam, -1)
            if idx != -1 and idx + 1 < len(common_diameters_sorted):
                for larger_diam in common_diameters_sorted[idx + 1 :]:
                    max_filling_height = get_max_filling_height(larger_diam)
                    if required_filling <= max_filling_height:
                        return larger_diam
            logger.warning(f"Could not find a suitable larger diameter for filling: {required_filling}")
            return current_diam

        def handle_excessive_filling(row: pd.Series) -> float:
            """Handle case where filling already exceeds limits."""
            current_diam = row["Geom1"]
            current_filling = row.get("Filling", 0.0)
            new_diam = find_larger_diameter(current_diam, current_filling)
            return new_diam

        def try_smaller_diameters(row: pd.Series, current_diam: float, idx: int, max_q: float, original_slope: float) -> float:
            """Try to find a smaller diameter that can still handle the flow."""
            best_diam = current_diam

            for candidate in reversed(common_diameters_sorted[:idx]):
                diameter_works = False

                for multiplier in SLOPE_MULTIPLIERS:
                    adjusted_slope = original_slope * multiplier

                    try:
                        fill_height = HydraulicCalculationsService.calc_filling(max_q, candidate, adjusted_slope)

                        if validate_filling(fill_height, candidate):
                            best_diam = candidate
                            diameter_works = True
                            break  # Exit multiplier loop
                        else:
                            break  # This diameter doesn't work even with this slope
                    except ValueError as e:
                        error_msg = str(e)
                        if "Slope is too small" in error_msg and multiplier < SLOPE_MULTIPLIERS[-1]:
                            # Try next multiplier
                            continue
                        else:
                            # Other error or reached max multiplier
                            break

                if diameter_works:
                    # Continue checking even smaller diameters
                    continue
                else:
                    # This diameter doesn't work with any slope, stop checking smaller ones
                    break

            return best_diam

        def handle_small_filling_ratio(row: pd.Series, current_diam: float, current_filling: float) -> float:
            """Handle the case where filling ratio is very small."""
            if current_filling / current_diam < LOW_FILLING_RATIO_THRESHOLD:
                target_diameter = current_filling / TARGET_FILLING_RATIO

                for candidate in common_diameters_sorted:
                    if candidate >= target_diameter and candidate < current_diam:
                        return candidate

            return current_diam

        def find_min_diam(row: pd.Series) -> float:
            # Step 1: Handle rows where filling already exceeds limits
            if row.get("ValMaxFill", 0) != 1:
                return handle_excessive_filling(row)

            # Get common parameters
            current_diam = float(row["Geom1"])
            current_filling = float(row.get("Filling", 0.0) or 0.0)
            max_q = float(row.get("MaxQ", 0.0) or 0.0)

            # Step 2: Check if current filling exceeds maximum allowed
            max_allowed_filling_height = get_max_filling_height(current_diam)
            if current_filling > max_allowed_filling_height:
                return find_larger_diameter(current_diam, current_filling)

            # Step 3: Handle very small flows
            if max_q < VERY_SMALL_FLOW_THRESHOLD:
                for candidate in common_diameters_sorted:
                    if candidate >= MIN_PRACTICAL_DIAMETER:
                        return candidate
                return common_diameters_sorted[0]

            # Step 4: Find current diameter index
            idx: int = diam_index_map.get(current_diam, -1)
            if idx == -1:  # Non-standard diameter - leave as is
                return current_diam

            # Step 5: Check if current diameter is sufficient based on calculated filling
            original_slope = float(row.get("SlopeFtPerFt", 0.0) or 0.0)
            best_diam: float = current_diam

            try:
                current_fill_height = HydraulicCalculationsService.calc_filling(max_q, current_diam, original_slope)
                needs_larger_diameter = not validate_filling(current_fill_height, current_diam)
            except ValueError as e:
                needs_larger_diameter = "Filling exceeds diameter" in str(e)

            if needs_larger_diameter:
                logger.info(f"ROW {row.name}: Current diameter is insufficient for calculated flow")

                # Find larger diameter that works
                for larger_diam in common_diameters_sorted[idx + 1 :] if idx + 1 < len(common_diameters_sorted) else []:
                    try:
                        larger_fill_height = HydraulicCalculationsService.calc_filling(max_q, larger_diam, original_slope)

                        if validate_filling(larger_fill_height, larger_diam):
                            logger.info(f"ROW {row.name}: Increasing diameter from {current_diam}m to {larger_diam}m")
                            return larger_diam
                    except ValueError:
                        continue
                return common_diameters_sorted[-1]

            # Step 6: Try to find smaller diameters that still meet requirements
            best_diam = try_smaller_diameters(row, current_diam, idx, max_q, original_slope)

            # Step 7: For cases with very low filling, try an alternative approach
            if best_diam == current_diam:
                best_diam = handle_small_filling_ratio(row, current_diam, current_filling)

            return best_diam

        # Apply to every row
        self.dfc["MinDiameter"] = self.dfc.apply(find_min_diam, axis=1)

    def diameter_features(self) -> None:
        """
        Calculates three features related to pipe diameter optimization:

        - isMinDiameter: 1 if current diameter equals calculated minimum diameter, 0 otherwise
        - IncreaseDia: 1 if current diameter is too small and should be increased, 0 otherwise
        - ReduceDia: 1 if current diameter is larger than necessary and could be reduced, 0 otherwise

        These features provide more detailed information for the neural network model
        about the relationship between current diameter (Geom1) and calculated
        minimum diameter (MinDiameter).
        """
        if self.dfc is not None:
            # Check if current diameter equals calculated minimum diameter
            self.dfc["isMinDiameter"] = np.where(self.dfc["Geom1"] == self.dfc["MinDiameter"], 1, 0)

            # Check if current diameter is too small and should be increased
            self.dfc["IncreaseDia"] = np.where(self.dfc["Geom1"] < self.dfc["MinDiameter"], 1, 0)

            # Check if current diameter is larger than necessary and could be reduced
            self.dfc["ReduceDia"] = np.where(self.dfc["Geom1"] > self.dfc["MinDiameter"], 1, 0)

    def normalize_roughness(self) -> None:
        """
        Normalizes the Roughness coefficient (Manning's n) in the dataframe.

        The Roughness coefficient is normalized to a range of [0, 1] based on typical
        values for different pipe materials used in stormwater drainage systems.

        Typical Manning's n values for different pipe materials:
        - PVC/Plastic pipes: 0.009 - 0.013 (Smooth surfaces)
        - Concrete pipes: 0.011 - 0.015 (Moderately smooth)
        - Cast iron pipes: 0.012 - 0.016 (Moderately rough)
        - Steel pipes: 0.010 - 0.014 (Varies based on coating)
        - Brick channels: 0.012 - 0.018 (Rough surfaces)
        - Stone channels: 0.013 - 0.020 (Very rough surfaces)

        The normalization uses the formula:
        Roughness_normalized = (Roughness - MIN_ROUGHNESS) / (MAX_ROUGHNESS - MIN_ROUGHNESS)

        Where:
        - MIN_ROUGHNESS = 0.009 (smoothest plastic pipes)
        - MAX_ROUGHNESS = 0.020 (roughest stone channels)

        Note:
        Values outside the range [0.009, 0.020] will be clipped to ensure
        the normalized values stay within [0, 1].
        """
        if self.dfc is None:
            return

        roughness_clipped = self.dfc["Roughness"].clip(MIN_ROUGHNESS, MAX_ROUGHNESS)
        self.dfc["NRoughness"] = (roughness_clipped - MIN_ROUGHNESS) / (MAX_ROUGHNESS - MIN_ROUGHNESS)

    def normalize_max_velocity(self) -> None:
        """
        Normalizes the MaxV (maximum velocity) values in the dataframe.

        The maximum velocity is normalized to a range of [0, 1] based on
        the maximum allowed velocity value from max_velocity_value.

        The normalization uses the formula:
        NMaxV = MaxV / max_velocity_value

        Where:
        - MaxV = Original maximum velocity value
        - max_velocity_value = Maximum allowed velocity (typically 5.0 m/s)

        Note:
        Values greater than max_velocity_value will be clipped to ensure
        the normalized values stay within [0, 1].
        """
        if self.dfc is None:
            return

        max_v_clipped = self.dfc["MaxV"].clip(0, max_velocity_value)
        self.dfc["NMaxV"] = max_v_clipped / max_velocity_value

    def normalize_depth(self) -> None:
        """
        Normalizes depth values for inlet and outlet nodes to range [0, 1].

        Normalization is based on the valid depth range for pipes:
        - Minimum valid depth: frost_zone + pipe diameter (frost protection + pipe size)
        - Maximum valid depth: max_depth_value (maximum allowed installation depth)

        Values outside the valid range are clipped before normalization.
        Rows where min equals max (zero range) are set to 0 to avoid division by zero.
        """
        if self.dfc is None:
            return

        min_depth = self.frost_zone + self.dfc["Geom1"]
        inlet_depth_clipped = self.dfc["InletMaxDepth"].clip(lower=min_depth, upper=max_depth_value)
        outlet_depth_clipped = self.dfc["OutletMaxDepth"].clip(lower=min_depth, upper=max_depth_value)
        depth_range = max_depth_value - min_depth
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
        """
        Normalizes the Filling values to range [0, 1].

        Normalization is based on the ratio of filling to maximum effective filling
        calculated as 0.827 * pipe diameter (Geom1), according to Colebrook-White formula.
        Values exceeding this threshold indicate hydraulic overload.

        According to hydraulic principles, a circular pipe reaches its maximum flow capacity
        at approximately 82.7% of its diameter, not at full diameter.
        """
        if self.dfc is None:
            return

        max_effective_filling = MAX_FILLING_RATIO * self.dfc["Geom1"]
        self.dfc["NFilling"] = self.dfc["Filling"] / max_effective_filling

    def normalize_max_q(self) -> None:
        """
        Normalizes the MaxQ (maximum flow) values in the dataframe.

        The normalization is based on the theoretical maximum flow capacity
        for each pipe, calculated using pipe diameter and slope.
        """
        if self.dfc is None:
            return

        # Calculate theoretical max flow capacity for each pipe (frozen v1 feature contract for
        # legacy 2025 MLP weights; GNN v4_physical uses manning_v2 in stormlab with D^(8/3) and m/m slope)
        theoretical_max_q = CIRCULAR_PIPE_FLOW_CONSTANT * (self.dfc["Geom1"] ** 2.5) * (self.dfc["SlopePerMile"] ** 0.5)

        # Normalize actual MaxQ by theoretical max flow
        # Clip to [0, 1] range to handle cases where actual flow exceeds theoretical max
        self.dfc["NMaxQ"] = (self.dfc["MaxQ"] / theoretical_max_q).clip(0, 1)

    def normalize_ground_cover(self) -> None:
        """
        Normalizes ground cover values (InletGroundCover and OutletGroundCover) to range [0, 1].

        Normalization is based on the valid ground cover range for pipes:
        - Minimum valid cover: self.frost_zone (frost protection)
        - Maximum valid cover: max_depth_value - Geom1 (maximum installation depth minus pipe diameter)

        Values outside the valid range are clipped before normalization.
        Rows where min equals max (zero range) are set to 0 to avoid division by zero.

        A ground cover value of 0 means minimum acceptable cover (frost_zone).
        A value of 1 means maximum possible cover (max_depth_value - pipe diameter).
        """
        if self.dfc is None:
            return

        if "InletGroundCover" not in self.dfc.columns or "OutletGroundCover" not in self.dfc.columns:
            self.ground_cover()

        min_cover = self.frost_zone
        max_cover = max_depth_value - self.dfc["Geom1"]

        # Clip values to valid range
        inlet_cover_clipped = self.dfc["InletGroundCover"].clip(lower=min_cover, upper=max_cover)
        outlet_cover_clipped = self.dfc["OutletGroundCover"].clip(lower=min_cover, upper=max_cover)

        # Calculate range (avoiding division by zero)
        cover_range = max_cover - min_cover
        valid_range = cover_range > 0

        # Initialize with zeros
        self.dfc["NInletGroundCover"] = 0.0
        self.dfc["NOutletGroundCover"] = 0.0

        # Normalize only where range is valid
        self.dfc.loc[valid_range, "NInletGroundCover"] = (inlet_cover_clipped.loc[valid_range] - min_cover) / cover_range.loc[
            valid_range
        ]
        self.dfc.loc[valid_range, "NOutletGroundCover"] = (outlet_cover_clipped.loc[valid_range] - min_cover) / cover_range.loc[
            valid_range
        ]

    @staticmethod
    def _safe_min_slope(filling: float, diameter: float) -> float:
        """Compute minimum required slope [‰] while clamping surcharged or NaN filling heights."""
        if not pd.isna(diameter) and 0.2 <= float(diameter) <= 2.0:
            safe_d = float(diameter)
            safe_f = 0.0 if pd.isna(filling) else float(np.clip(float(filling), 0.0, safe_d))
            return float(min_slope(safe_f, safe_d))
        return float(min_slope(filling, diameter))

    @staticmethod
    def _fast_max_slope(diameter: float) -> float:
        """Return maximum allowable slope [‰], using precomputed lookup for catalog diameters."""
        if max_slope is _CANONICAL_MAX_SLOPE and not pd.isna(diameter):
            cached = max_slopes.get(str(round(float(diameter), 2)))
            if cached is not None:
                return float(cached)
        return float(max_slope(diameter))

    def normalize_slope(self) -> None:
        """
        Normalizes the slope values in the dataframe to a range [0, 1] based on both
        minimum required slope and maximum allowable slope.

        The normalization maps the slope values to a range where:
        - 0 represents a slope at or below the minimum required slope
        - 0.5 represents an optimal slope (midpoint between min and max)
        - 1 represents a slope at or above the maximum allowable slope

        This ensures that slopes are evaluated relative to their hydraulic requirements
        based on pipe diameter and filling.
        """
        if self.dfc is None or self.dfc.empty:
            return

        # Calculate minimum required slope for each pipe based on its filling and diameter
        self.dfc["MinRequiredSlope"] = self.dfc.apply(lambda r: self._safe_min_slope(r["Filling"], r["Geom1"]), axis=1)

        # Calculate maximum allowable slope for each pipe based on its diameter
        self.dfc["MaxAllowableSlope"] = self.dfc.apply(lambda r: self._fast_max_slope(r["Geom1"]), axis=1)

        # Normalize slope to range [0, 1] where:
        # 0 = at or below minimum required slope
        # 1 = at or above maximum allowable slope
        def normalize(row):
            min_slope_val = row["MinRequiredSlope"]
            max_slope_val = row["MaxAllowableSlope"]
            actual_slope = row["SlopePerMile"]

            if pd.isna(actual_slope):
                return 0.0

            # Handle edge cases
            if min_slope_val >= max_slope_val:  # If min and max are too close
                return 0.5 if actual_slope >= min_slope_val else 0.0

            # Linear normalization between min and max
            if actual_slope <= min_slope_val:
                return 0.0
            elif actual_slope >= max_slope_val:
                return 1.0
            else:
                # Map to range [0, 1]
                return (actual_slope - min_slope_val) / (max_slope_val - min_slope_val)

        self.dfc["NSlope"] = self.dfc.apply(normalize, axis=1)

    def slope_increase(self) -> None:
        """
        Identifies pipes that need slope increase because their current slope is too small.

        This method adds a binary indicator column 'SlopeIncrease' to the dataframe where:
        - 1 indicates the pipe needs slope increase (current slope < minimum required slope)
        - 0 indicates the pipe does not need slope increase

        The minimum required slope is calculated based on the pipe's filling height and diameter
        using the min_slope function.
        """
        if self.dfc is None:
            return

        # Ensure MinRequiredSlope is calculated
        if "MinRequiredSlope" not in self.dfc.columns:
            self.dfc["MinRequiredSlope"] = self.dfc.apply(lambda r: self._safe_min_slope(r["Filling"], r["Geom1"]), axis=1)

        # Check if current slope is less than minimum required slope
        self.dfc["IncreaseSlope"] = np.where(self.dfc["SlopePerMile"] < self.dfc["MinRequiredSlope"], 1, 0)

    def slope_reduction(self) -> None:
        """
        Identifies pipes that need slope reduction because their current slope is too large.

        This method adds a binary indicator column 'SlopeReduction' to the dataframe where:
        - 1 indicates the pipe needs slope reduction (current slope > maximum allowable slope)
        - 0 indicates the pipe does not need slope reduction

        The maximum allowable slope is calculated based on the pipe's diameter
        using the max_slope function.
        """
        if self.dfc is None or self.dfc.empty:
            return

        # Ensure MaxAllowableSlope is calculated
        if "MaxAllowableSlope" not in self.dfc.columns:
            self.dfc["MaxAllowableSlope"] = self.dfc.apply(lambda r: self._fast_max_slope(r["Geom1"]), axis=1)

        # Check if current slope is greater than maximum allowable slope
        self.dfc["ReduceSlope"] = np.where(self.dfc["SlopePerMile"] > self.dfc["MaxAllowableSlope"], 1, 0)

    def conduits_subcatchment_info(self) -> None:
        """
        Maps subcatchment information to conduits based on their inlet and outlet nodes.

        This method assigns subcatchment ID and category to conduits based on:
        1. Outlet node's subcatchment information if available
        2. Inlet node's subcatchment information if outlet node has no subcatchment

        If neither node has subcatchment information, conduit's subcatchment fields remain empty.
        """
        if self.dfc is None or self.dfn is None:
            return

        # Check if nodes dataframe has the required subcatchment columns
        if "Subcatchment" not in self.dfn.columns or "SbcCategory" not in self.dfn.columns:
            return

        # Initialize conduit subcatchment columns
        self.dfc["Subcatchment"] = "-"
        self.dfc["SbcCategory"] = "-"

        # Map node subcatchment info to conduits
        for idx, row in self.dfc.iterrows():
            inlet_node = row.get("InletNode")
            outlet_node = row.get("OutletNode")

            # First try outlet node for subcatchment info
            if outlet_node in self.dfn.index and self.dfn.at[outlet_node, "Subcatchment"] != "-":
                self.dfc.at[idx, "Subcatchment"] = self.dfn.at[outlet_node, "Subcatchment"]
                self.dfc.at[idx, "SbcCategory"] = self.dfn.at[outlet_node, "SbcCategory"]
            # If no info on outlet node, try inlet node
            elif inlet_node in self.dfn.index and self.dfn.at[inlet_node, "Subcatchment"] != "-":
                self.dfc.at[idx, "Subcatchment"] = self.dfn.at[inlet_node, "Subcatchment"]
                self.dfc.at[idx, "SbcCategory"] = self.dfn.at[inlet_node, "SbcCategory"]

    def propagate_subcatchment_info(self) -> None:
        """
        Propagates subcatchment information through the network for nodes and conduits
        that don't have directly connected subcatchments.

        If a node doesn't have subcatchment info, it inherits from its inlet nodes.
        If a conduit doesn't have subcatchment info, it inherits from its nodes.

        This method should be called after conduits_subcatchment_info().
        """
        if self.dfc is None or self.dfn is None:
            return

        # First, gather all connections (which node is connected to which)
        node_connections = {}  # Dictionary mapping outlet_node -> list of inlet_nodes

        for idx, row in self.dfc.iterrows():
            inlet_node = row.get("InletNode")
            outlet_node = row.get("OutletNode")

            if outlet_node not in node_connections:
                node_connections[outlet_node] = []

            if inlet_node:
                node_connections[outlet_node].append(inlet_node)

        # Propagate subcatchment info through the network
        # Multiple iterations to ensure info propagates through the entire network
        for _ in range(SUBCATCHMENT_PROPAGATION_ITERATIONS):
            changes_made = False

            # Update nodes based on their inlet nodes
            for outlet_node, inlet_nodes in node_connections.items():
                if outlet_node in self.dfn.index and self.dfn.at[outlet_node, "Subcatchment"] == "-":
                    # Try to find any inlet node with subcatchment info
                    for inlet_node in inlet_nodes:
                        if inlet_node in self.dfn.index and self.dfn.at[inlet_node, "Subcatchment"] != "-":
                            self.dfn.at[outlet_node, "Subcatchment"] = self.dfn.at[inlet_node, "Subcatchment"]
                            self.dfn.at[outlet_node, "SbcCategory"] = self.dfn.at[inlet_node, "SbcCategory"]
                            changes_made = True
                            break

            # Update conduits based on their nodes
            for idx, row in self.dfc.iterrows():
                if self.dfc.at[idx, "Subcatchment"] == "-":
                    inlet_node = row.get("InletNode")
                    outlet_node = row.get("OutletNode")

                    # First try outlet node
                    if outlet_node in self.dfn.index and self.dfn.at[outlet_node, "Subcatchment"] != "-":
                        self.dfc.at[idx, "Subcatchment"] = self.dfn.at[outlet_node, "Subcatchment"]
                        self.dfc.at[idx, "SbcCategory"] = self.dfn.at[outlet_node, "SbcCategory"]
                        changes_made = True
                    # Then try inlet node
                    elif inlet_node in self.dfn.index and self.dfn.at[inlet_node, "Subcatchment"] != "-":
                        self.dfc.at[idx, "Subcatchment"] = self.dfn.at[inlet_node, "Subcatchment"]
                        self.dfc.at[idx, "SbcCategory"] = self.dfn.at[inlet_node, "SbcCategory"]
                        changes_made = True

            # If no changes were made in this iteration, we've reached a stable state
            if not changes_made:
                break

    def encode_sbc_category(self) -> None:
        """One-hot encodes the SbcCategory column for use in machine learning models.

        Creates columns for all possible subcatchment categories, not just those present in the data,
        ensuring consistent feature sets across different datasets.

        This method adds one-hot encoded columns directly to the conduits dataframe (self.dfc).
        Each column is named 'SbcCategory_{category}' and contains integer values (0 or 1).
        """
        if self.dfc is None or len(self.dfc) == 0:
            return

        if "SbcCategory" not in self.dfc.columns:
            return

        encoded_categories = pd.get_dummies(self.dfc["SbcCategory"], drop_first=False)

        for category in SUBCATCHMENT_CATEGORIES:
            col_name = f"{category}"
            if col_name not in encoded_categories.columns:
                encoded_categories[col_name] = 0
            else:
                encoded_categories[col_name] = encoded_categories[col_name].astype(int)

        for col in encoded_categories.columns:
            self.dfc[col] = encoded_categories[col]


class RecommendationService:
    """
    Unified recommendation service that supports both MLP and GNN models.
    """

    def __init__(
        self,
        dfc: pd.DataFrame,
        model: Optional[object],
        model_name: str = "MLP",
        adjacency=None,
        conduit_order=None,
        inp_path: Optional[str] = None,
        rpt_path: Optional[str] = None,
    ):
        """
        Initialize the recommendation service.

        Args:
            dfc: DataFrame containing conduit data
            model: The ML model (MLP or GNN) to use for predictions
            model_name: Name of the model for logging ("MLP" or "GNN")
            adjacency: Preprocessed adjacency matrix required for GNN predictions
            conduit_order: Conduit ordering matching adjacency matrix rows
            inp_path: Optional path to the SWMM .inp file
            rpt_path: Optional path to the SWMM .rpt file
        """
        self.dfc = dfc
        self.model = model
        self.model_name = model_name
        self.adjacency = adjacency
        self.conduit_order = conduit_order
        self.inp_path = inp_path
        self.rpt_path = rpt_path

    def recommendations(self) -> pd.DataFrame:
        """
        Generate recommendations using the provided model.
        Adds 'recommendation' column and confidence scores for each category.

        Returns:
            pd.DataFrame: DataFrame with added recommendation results.
        """
        logger.info(f"Generating recommendations using {self.model_name} model")
        if self.dfc is None or self.model is None:
            raise ValueError("DataFrame and model must be provided")

        spec = getattr(self.model, "spec", None)

        if self.model_name == "GNN":
            if self.adjacency is None:
                raise ValueError("GNN model requires adjacency matrix; refusing to serve GNN without graph (K1)")

            if spec is not None:
                contract = getattr(self.model, "feature_contract", None)
                if contract is not None and getattr(contract, "name", "") == "v4_physical":
                    from stormlab.config import load_config
                    from stormlab.features.physical import extract_physical_features
                    from stormlab.contracts.features import FEATURE_CONTRACT_V4_PHYSICAL
                    from stormlab.graph.order import canonical_order

                    order = self.conduit_order or canonical_order(self.dfc, "inp")

                    if self.inp_path and self.rpt_path and os.path.isfile(self.inp_path) and os.path.isfile(self.rpt_path):
                        config = load_config(variant="C3", check_inputs=False)
                        phys_df = extract_physical_features(self.inp_path, self.rpt_path, config=config, file_id="app_serving")
                        phys_indexed = phys_df.set_index("conduit").loc[:, list(FEATURE_CONTRACT_V4_PHYSICAL.columns)].copy()
                    else:
                        phys_indexed = pd.DataFrame(index=order)
                        for col in FEATURE_CONTRACT_V4_PHYSICAL.columns:
                            if col in self.dfc.columns:
                                if "Name" in self.dfc.columns:
                                    phys_indexed[col] = self.dfc.set_index("Name")[col]
                                else:
                                    phys_indexed[col] = self.dfc[col].values
                            else:
                                phys_indexed[col] = np.nan

                        if bool(phys_indexed.isna().all().all()):
                            raise ValueError(
                                "GNN v4_physical model requires valid inp_path and rpt_path or precomputed "
                                "v4_physical columns; refusing to impute 100% missing features"
                            )

                    imputation = getattr(self.model, "imputation_plan", None)
                    if imputation and "fills" in imputation:
                        for col in FEATURE_CONTRACT_V4_PHYSICAL.columns:
                            fill_val = imputation["fills"].get(col, 0.0)
                            phys_indexed[col] = phys_indexed[col].fillna(fill_val)
                    else:
                        phys_indexed = phys_indexed.fillna(0.0)

                    features_aligned = phys_indexed.loc[list(order)].to_numpy(dtype="float32")
                    adj_dense = np.asarray(
                        self.adjacency.todense() if hasattr(self.adjacency, "todense") else self.adjacency,
                        dtype="float32",
                    )
                    preds = self.model([features_aligned, adj_dense], training=False).numpy()

                    from stormlab.contracts.enums import RecommendationCategory as RC

                    class_order = [c.value for c in RC]
                    proba = pd.DataFrame(preds, index=list(order), columns=[f"p_{c}" for c in class_order])
                    self._assign_aligned_probabilities(proba)

                    logger.info(f"{self.model_name} (C3c v4_physical) recommendations generated for {len(self.dfc)} conduits")
                    return self.dfc

                from stormlab.features.conduits import conduit_feature_matrix
                from stormlab.contracts.features import FEATURE_CONTRACT_V2
                from stormlab.models.predict import predict_gnn
                from stormlab.graph.order import canonical_order

                features = conduit_feature_matrix(self.dfc)
                contract = getattr(self.model, "feature_contract", FEATURE_CONTRACT_V2)
                model_input = features.loc[:, list(contract.columns)].copy()
                if "Name" in self.dfc.columns:
                    model_input.index = self.dfc["Name"].astype(str)
                order = self.conduit_order or canonical_order(self.dfc, "inp")

                proba = predict_gnn(self.model, model_input, self.adjacency, order, contract=contract)
                self._assign_aligned_probabilities(proba)

                logger.info(f"{self.model_name} (B4) recommendations generated for {len(self.dfc)} conduits")
                return self.dfc

            input_data = self.dfc.reindex(columns=FEATURE_COLUMNS, fill_value=0)
            logger.info(f"{self.model_name} input shape: {input_data.shape}")
            features = np.nan_to_num(input_data.to_numpy(dtype=np.float32))
            adj_dense = np.asarray(
                self.adjacency.todense() if hasattr(self.adjacency, "todense") else self.adjacency,
                dtype=np.float32,
            )
            preds = self.model([features, adj_dense], training=False).numpy()
        else:
            contract = getattr(self.model, "feature_contract", None)
            cols = list(contract.columns) if contract is not None else FEATURE_COLUMNS
            input_data = self.dfc.reindex(columns=cols, fill_value=0)
            logger.info(f"{self.model_name} input shape: {input_data.shape}")
            preds = self.model.predict(input_data, verbose=0)

        preds_cls = preds.argmax(axis=-1)
        labels = [category.value for category in RecommendationCategory]
        self.dfc["recommendation"] = [labels[i] for i in preds_cls]

        for i, category in enumerate(RecommendationCategory):
            self.dfc[f"confidence_{category.value}"] = [round(float(val), 3) for val in preds[:, i]]

        logger.info(f"{self.model_name} recommendations generated for {len(self.dfc)} conduits")
        return self.dfc

    def _assign_aligned_probabilities(self, proba: pd.DataFrame) -> None:
        """Align GNN probability DataFrame to self.dfc row order and assign recommendation + confidence columns."""
        if "Name" in self.dfc.columns:
            keys = self.dfc["Name"].astype(str)
            aligned_proba = proba.loc[keys].reset_index(drop=True)
        else:
            idx_keys = self.dfc.index.astype(str)
            if idx_keys.isin(proba.index.astype(str)).all():
                proba_str = proba.copy()
                proba_str.index = proba_str.index.astype(str)
                aligned_proba = proba_str.loc[idx_keys].reset_index(drop=True)
            else:
                aligned_proba = proba.reset_index(drop=True)

        pred_categories = aligned_proba.idxmax(axis=1).str.replace("p_", "", regex=False)
        self.dfc["recommendation"] = pred_categories.values

        for category in RecommendationCategory:
            col_name = f"p_{category.value}"
            if col_name in aligned_proba.columns:
                self.dfc[f"confidence_{category.value}"] = aligned_proba[col_name].round(3).values
            else:
                self.dfc[f"confidence_{category.value}"] = 0.0
