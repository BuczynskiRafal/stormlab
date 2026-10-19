"""Unit tests for the individual conduit feature engineering steps.

Migrated from ``app/sa/core/tests/test_conduits.py``. The assertions are
unchanged; what changed is where the frames come from. The original built a live
``DataManager``, which ran SWMM against a tracked INP file on every test; here the
steps are handed the frozen frames from ``stormlab/tests/fixtures``. The tests
that exercised the ``DataManager`` lifecycle (construction, the ``frost_zone``
setter, the ``with`` protocol) are gone with the class itself.
"""

import numpy as np
import pandas as pd
import pytest

from stormlab.features.conduits import (
    SEARCH_OUTCOME_MESSAGES,
    ConduitFeatureError,
    ConduitFeatureSteps,
    _reraise_unless_search_outcome,
)
from stormlab.hydraulics import formulas
from stormlab.hydraulics.validators import validate_filling

FROST_ZONE = 1.2


@pytest.fixture
def steps(frozen_dfc, frozen_dfn):
    """Feature steps over a fresh copy of the frozen network frames."""
    return ConduitFeatureSteps(frozen_dfc.copy(), frozen_dfn.copy(), FROST_ZONE)


class TestConduitSteps:
    """
    A test class for the ConduitsData class, containing various test cases to ensure the correct
    functionality of the methods in ConduitsData.
    """

    def test_init(self, steps):
        """
        Test the initialization of the ConduitsData object and ensure that the 'conduits' attribute
        is a pandas DataFrame.
        """
        assert isinstance(steps.dfc, pd.DataFrame)

    def test_calculate_filling(self, steps):
        """
        Test the 'calculate_filling' method of the ConduitsData class to ensure that it
        correctly calculates and adds the 'Filling' column to the 'conduits' DataFrame.
        """
        steps.calculate_filling()
        assert "Filling" in steps.dfc.columns
        assert all(steps.dfc["Filling"] >= 0)

    def test_filling_is_valid(self, steps):
        """
        Test the 'filling_is_valid' method of the ConduitsData class to ensure that it correctly
        validates the conduit filling and adds the 'ValMaxFill' column to the 'conduits' DataFrame.
        """
        steps.calculate_filling()
        steps.filling_is_valid()
        assert "ValMaxFill" in steps.dfc.columns
        assert all(steps.dfc["ValMaxFill"].isin([0, 1]))

    def test_validate_filling(self):
        """
        Test the 'validate_filling' function to ensure it correctly validates the filling value based
        on the provided diameter.
        """
        assert validate_filling(0, 1)
        assert validate_filling(0.5, 1)
        assert not validate_filling(1, 1)
        assert not validate_filling(1.1, 1)

    def test_velocity_is_valid_column(self, steps):
        """
        Test the 'velocity_is_valid' method of the ConduitsData class to ensure that it correctly
        adds the 'ValMaxV' column to the 'conduits' DataFrame.
        """
        assert "ValMaxV" not in steps.dfc.columns
        steps.velocity_is_valid()
        assert "ValMaxV" in steps.dfc.columns

    def test_velocity_is_valid(self, steps):
        """
        Test the 'velocity_is_valid' method of the ConduitsData class to ensure that it correctly
        validates the conduit velocities and updates the 'ValMaxV' and 'ValMinV' columns in the
        'conduits' DataFrame.
        """
        steps.velocity_is_valid()
        expected_values = [1, 1]
        assert list(steps.dfc["ValMaxV"])[:2] == expected_values
        assert list(steps.dfc["ValMinV"])[:2] == expected_values

    def test_slope_per_mile_column_added(self, steps):
        """
        Test if the 'SlopePerMile' column is added to the conduits dataframe
        after calling the slope_per_mile() method.
        """
        steps.slope_per_mile()
        assert "SlopePerMile" in steps.dfc.columns

    def test_slope_per_mile_calculation(self, steps):
        """
        Test if the calculated values in the 'SlopePerMile' column are correct after
        calling the slope_per_mile() method.
        """
        steps.slope_per_mile()
        expected_values = [1.80, 6.40]  # SlopeFtPerFt * 1000
        assert list(steps.dfc["SlopePerMile"])[:2] == pytest.approx(expected_values, abs=1e-9)

    def test_slopes_is_valid_columns_added(self, steps):
        """Test if slope validation columns are added after slopes_is_valid()."""
        steps.calculate_filling()
        steps.slope_per_mile()
        steps.normalize_slope()
        steps.slopes_is_valid()
        assert "ValMaxSlope" in steps.dfc.columns
        assert "ValMinSlope" in steps.dfc.columns

    def test_slopes_is_valid_max_slope(self, steps):
        """Test if maximum slope validation is correct."""
        steps.calculate_filling()
        steps.slope_per_mile()
        steps.normalize_slope()
        steps.slopes_is_valid()
        expected_values = [1, 1]
        assert list(steps.dfc["ValMaxSlope"])[:2] == expected_values

    def test_slopes_is_valid_min_slope(self, steps):
        """Test if minimum slope validation is correct."""
        steps.calculate_filling()
        steps.slope_per_mile()
        steps.normalize_slope()
        steps.slopes_is_valid()
        expected_values = [1, 1]
        assert list(steps.dfc["ValMinSlope"])[:2] == expected_values

    def test_max_depth_columns_added(self, steps):
        """
        Test if the 'InletMaxDepth' and 'OutletMaxDepth' columns are added to the conduits DataFrame after
        calling the `max_depth()` method.
        """
        steps.max_depth()
        assert "InletMaxDepth" in steps.dfc.columns
        assert "OutletMaxDepth" in steps.dfc.columns

    def test_max_depth_inlet_values_match(self, steps, frozen_dfn):
        """
        Test if the 'InletMaxDepth' values in the conduits DataFrame match the corresponding 'MaxDepth' values
        in the nodes DataFrame, using the 'InletNode' as a reference.
        """
        steps.max_depth()
        nodes_data = frozen_dfn
        for _, conduit in steps.dfc.iterrows():
            inlet_node = conduit["InletNode"]
            node_max_depth = nodes_data.loc[inlet_node, "MaxDepth"]
            conduit_inlet_max_depth = conduit["InletMaxDepth"]
            assert conduit_inlet_max_depth == node_max_depth

    def test_max_depth_outlet_values_match(self, steps, frozen_dfn):
        """
        Test if the 'OutletMaxDepth' values in the conduits DataFrame match the corresponding 'MaxDepth' values
        in the nodes DataFrame, using the 'OutletNode' as a reference.
        """
        steps.max_depth()
        steps.calculate_max_depth()
        nodes_data = frozen_dfn
        for _, conduit in steps.dfc.iterrows():
            outlet_node = conduit["OutletNode"]
            node_max_depth = nodes_data.loc[outlet_node, "MaxDepth"]
            conduit_outlet_max_depth = conduit["OutletMaxDepth"]
            if not pd.isna(conduit_outlet_max_depth) and not pd.isna(node_max_depth):
                assert conduit_outlet_max_depth == node_max_depth

    def test_max_ground_cover_valid_depths(self, steps):
        """Test max_ground_cover_is_valid with valid depths for both inlet and outlet."""
        steps.dfc.at[0, "InletNodeInvert"] = 10
        steps.dfc.at[0, "OutletNodeInvert"] = 8
        steps.dfc.at[0, "InletGroundElevation"] = 15
        steps.dfc.at[0, "OutletGroundElevation"] = 13
        steps.dfc.at[0, "ValDepth"] = np.nan

        steps.max_ground_cover_is_valid()
        assert steps.dfc["ValDepth"].loc[0] == 1

    def test_max_ground_cover_invalid_depths(self, steps):
        """Test max_ground_cover_is_valid with invalid depths for both inlet and outlet."""
        steps.dfc.at[0, "InletNodeInvert"] = 15
        steps.dfc.at[0, "OutletNodeInvert"] = 13
        steps.dfc.at[0, "InletGroundElevation"] = 35
        steps.dfc.at[0, "OutletGroundElevation"] = 32
        steps.dfc.at[0, "ValDepth"] = np.nan

        steps.max_ground_cover_is_valid()
        assert steps.dfc["ValDepth"].loc[0] == 0

    def test_calculate_max_depth(self, steps):
        """
        Test the 'calculate_max_depth' method of the ConduitsData class to ensure that it correctly
        calculates the maximum depth of each conduit's outlet, based on its inlet depth, length, and slope.
        """
        steps.max_depth()
        steps.dfc.at[0, "InletMaxDepth"] = 10
        steps.dfc.at[0, "Length"] = 100
        steps.dfc.at[0, "OutletMaxDepth"] = 5

        steps.dfc.at[1, "InletMaxDepth"] = 20
        steps.dfc.at[1, "Length"] = 200
        steps.dfc.at[1, "OutletMaxDepth"] = 10
        steps.calculate_max_depth()
        assert steps.dfc["OutletMaxDepth"].loc[0] == 5
        assert steps.dfc["OutletMaxDepth"].loc[1] == 10

    def test_calculate_maximum_depth(self, steps):
        """
        Test the 'calculate_max_depth' method of the ConduitsData class to ensure that it correctly
        calculates the maximum depth of each conduit's outlet, based on its inlet depth, length, and slope.
        """
        steps.max_depth()
        # Set up some test data
        test_data = [
            {
                "InletMaxDepth": 10,
                "Length": 100,
                "SlopeFtPerFt": 0.01,
                "OutletMaxDepth": np.nan,
            },
            {
                "InletMaxDepth": 20,
                "Length": 200,
                "SlopeFtPerFt": 0.02,
                "OutletMaxDepth": np.nan,
            },
            {
                "InletMaxDepth": 30,
                "Length": 300,
                "SlopeFtPerFt": 0.03,
                "OutletMaxDepth": np.nan,
            },
        ]
        steps.dfc = pd.DataFrame(test_data)
        steps.dfc = steps.dfc
        steps.calculate_max_depth()

        assert all(~pd.isna(steps.dfc["OutletMaxDepth"]))
        assert steps.dfc.loc[0, "OutletMaxDepth"] == pytest.approx(9, abs=1e-9)
        assert steps.dfc.loc[1, "OutletMaxDepth"] == pytest.approx(16, abs=1e-9)
        assert steps.dfc.loc[2, "OutletMaxDepth"] == pytest.approx(21, abs=1e-9)

    def test_min_ground_cover_is_valid(self, steps):
        """
        Test the 'min_ground_cover_is_valid' method to ensure it correctly checks
        if the ground cover is above frost zone.
        """
        # Set up test data
        test_data = [
            {
                "InletNodeInvert": 10,
                "OutletNodeInvert": 20,
                "InletGroundElevation": 15,
                "OutletGroundElevation": 25,
                "InletGroundCover": 5,
                "OutletGroundCover": 5,
            },
            {
                "InletNodeInvert": 5,
                "OutletNodeInvert": 2,
                "InletGroundElevation": 4,
                "OutletGroundElevation": 1,
                "InletGroundCover": -1,
                "OutletGroundCover": -1,
            },
            {
                "InletNodeInvert": 30,
                "OutletNodeInvert": 40,
                "InletGroundElevation": 35,
                "OutletGroundElevation": 45,
                "InletGroundCover": 5,
                "OutletGroundCover": 5,
            },
        ]
        steps.dfc = pd.DataFrame(test_data)
        steps.dfc = steps.dfc
        steps.frost_zone = 1.0

        # Call the method
        steps.min_ground_cover_is_valid()

        # Assertions
        assert steps.dfc.loc[0, "ValCoverage"] == 1
        assert steps.dfc.loc[1, "ValCoverage"] == 0
        assert steps.dfc.loc[2, "ValCoverage"] == 1

    def test_min_ground_cover_valid(self, steps):
        """Test when both inlet and outlet ground covers are above frost zone."""
        steps.frost_zone = 1.2
        steps.dfc.at[0, "InletGroundCover"] = 1.5
        steps.dfc.at[0, "OutletGroundCover"] = 1.4
        steps.dfc.at[0, "ValCoverage"] = np.nan

        steps.min_ground_cover_is_valid()
        assert steps.dfc["ValCoverage"].loc[0] == 1

    def test_min_ground_cover_invalid(self, steps):
        """Test when both inlet and outlet ground covers are in frost zone."""
        steps.frost_zone = 1.2
        steps.dfc.at[0, "InletGroundCover"] = 0.8
        steps.dfc.at[0, "OutletGroundCover"] = 0.9
        steps.dfc.at[0, "ValCoverage"] = np.nan

        steps.min_ground_cover_is_valid()
        assert steps.dfc["ValCoverage"].loc[0] == 0

    def test_min_ground_cover_mixed(self, steps):
        """Test when inlet is valid but outlet is in frost zone."""
        steps.frost_zone = 1.2
        steps.dfc.at[0, "InletGroundCover"] = 1.5
        steps.dfc.at[0, "OutletGroundCover"] = 0.9
        steps.dfc.at[0, "ValCoverage"] = np.nan

        steps.min_ground_cover_is_valid()
        assert steps.dfc["ValCoverage"].loc[0] == 0

    def test_min_ground_cover_edge_case(self, steps):
        """Test when ground covers are exactly at frost zone depth."""
        steps.frost_zone = 1.2
        steps.dfc.at[0, "InletGroundCover"] = 1.2
        steps.dfc.at[0, "OutletGroundCover"] = 1.2
        steps.dfc.at[0, "ValCoverage"] = np.nan

        steps.min_ground_cover_is_valid()
        assert steps.dfc["ValCoverage"].loc[0] == 1

    def test_ground_elevation(self, steps):
        """
        Tests the 'ground_elevation' method of the ConduitsData class
        to ensure it correctly calculates ground cover over each conduit's inlet and outlet.
        """
        # Prepare test data
        test_data = [
            {"InletNodeInvert": 10, "InletMaxDepth": 2, "OutletNodeInvert": 40, "OutletMaxDepth": 8},
            {"InletNodeInvert": 20, "InletMaxDepth": 4, "OutletNodeInvert": 50, "OutletMaxDepth": 10},
            {"InletNodeInvert": 30, "InletMaxDepth": 6, "OutletNodeInvert": 60, "OutletMaxDepth": 12},
        ]
        steps.dfc = pd.DataFrame(test_data)
        steps.dfc = steps.dfc

        # Call the method
        steps.calculate_ground_elevation()

        # Assertions for InletGroundElevation
        assert steps.dfc.loc[0, "InletGroundElevation"] == pytest.approx(12, abs=1e-9), "Incorrect result for row 0 (Inlet)"
        assert steps.dfc.loc[1, "InletGroundElevation"] == pytest.approx(24, abs=1e-9), "Incorrect result for row 1 (Inlet)"
        assert steps.dfc.loc[2, "InletGroundElevation"] == pytest.approx(36, abs=1e-9), "Incorrect result for row 2 (Inlet)"

        # Assertions for OutletGroundElevation
        assert steps.dfc.loc[0, "OutletGroundElevation"] == pytest.approx(48, abs=1e-9), "Incorrect result for row 0 (Outlet)"
        assert steps.dfc.loc[1, "OutletGroundElevation"] == pytest.approx(60, abs=1e-9), "Incorrect result for row 1 (Outlet)"
        assert steps.dfc.loc[2, "OutletGroundElevation"] == pytest.approx(72, abs=1e-9), "Incorrect result for row 2 (Outlet)"


class TestGroundCover:
    """Tests for the ground_cover method."""

    def test_ground_cover_calculation(self, steps):
        """Test ground cover calculation for inlet and outlet."""
        # Prepare test data
        steps.dfc = pd.DataFrame(
            [
                {
                    "InletGroundElevation": 15,
                    "InletNodeInvert": 10,
                    "Geom1": 0.5,
                    "OutletGroundElevation": 13,
                    "OutletNodeInvert": 8,
                },
                {
                    "InletGroundElevation": 20,
                    "InletNodeInvert": 12,
                    "Geom1": 1.0,
                    "OutletGroundElevation": 18,
                    "OutletNodeInvert": 10,
                },
            ]
        )
        steps.dfc = steps.dfc

        steps.ground_cover()

        # InletGroundCover = InletGroundElevation - InletNodeInvert - Geom1
        assert steps.dfc.loc[0, "InletGroundCover"] == pytest.approx(4.5, abs=1e-9)
        assert steps.dfc.loc[1, "InletGroundCover"] == pytest.approx(7.0, abs=1e-9)
        # OutletGroundCover = OutletGroundElevation - OutletNodeInvert - Geom1
        assert steps.dfc.loc[0, "OutletGroundCover"] == pytest.approx(4.5, abs=1e-9)
        assert steps.dfc.loc[1, "OutletGroundCover"] == pytest.approx(7.0, abs=1e-9)

    def test_ground_cover_with_none_dfc(self, steps):
        """Test ground_cover returns early when dfc is None."""
        steps.dfc = None
        # Should not raise, just return
        steps.ground_cover()

    def test_ground_cover_missing_columns_raises(self, steps):
        """Test ground_cover raises ValueError when required columns are missing."""
        steps.dfc = pd.DataFrame([{"SomeColumn": 1}])
        steps.dfc = steps.dfc

        with pytest.raises(ValueError, match="Missing required columns"):
            steps.ground_cover()


class TestMinGroundCoverAutoCall:
    """Tests for min_ground_cover_is_valid auto-calling ground_cover."""

    def test_min_ground_cover_calls_ground_cover_when_column_missing(self, steps):
        """Test that min_ground_cover_is_valid calls ground_cover if InletGroundCover is missing."""
        steps.dfc = pd.DataFrame(
            [
                {
                    "InletGroundElevation": 15,
                    "InletNodeInvert": 10,
                    "Geom1": 0.5,
                    "OutletGroundElevation": 13,
                    "OutletNodeInvert": 8,
                }
            ]
        )
        steps.dfc = steps.dfc
        steps.frost_zone = 1.0

        # InletGroundCover column doesn't exist - method should call ground_cover()
        assert "InletGroundCover" not in steps.dfc.columns

        steps.min_ground_cover_is_valid()

        # Now the columns should exist and ValCoverage should be calculated
        assert "InletGroundCover" in steps.dfc.columns
        assert "ValCoverage" in steps.dfc.columns


class TestDiameterFeatures:
    """Tests for the diameter_features method."""

    def test_diameter_features_all_equal(self, steps):
        """Test when current diameter equals minimum diameter."""
        steps.dfc = pd.DataFrame(
            [
                {"Geom1": 0.5, "MinDiameter": 0.5},
                {"Geom1": 1.0, "MinDiameter": 1.0},
            ]
        )
        steps.dfc = steps.dfc

        steps.diameter_features()

        assert list(steps.dfc["isMinDiameter"]) == [1, 1]
        assert list(steps.dfc["IncreaseDia"]) == [0, 0]
        assert list(steps.dfc["ReduceDia"]) == [0, 0]

    def test_diameter_features_increase_needed(self, steps):
        """Test when current diameter is smaller than minimum (needs increase)."""
        steps.dfc = pd.DataFrame(
            [
                {"Geom1": 0.3, "MinDiameter": 0.5},
                {"Geom1": 0.8, "MinDiameter": 1.2},
            ]
        )
        steps.dfc = steps.dfc

        steps.diameter_features()

        assert list(steps.dfc["isMinDiameter"]) == [0, 0]
        assert list(steps.dfc["IncreaseDia"]) == [1, 1]
        assert list(steps.dfc["ReduceDia"]) == [0, 0]

    def test_diameter_features_reduce_possible(self, steps):
        """Test when current diameter is larger than minimum (can be reduced)."""
        steps.dfc = pd.DataFrame(
            [
                {"Geom1": 1.0, "MinDiameter": 0.5},
                {"Geom1": 1.5, "MinDiameter": 0.8},
            ]
        )
        steps.dfc = steps.dfc

        steps.diameter_features()

        assert list(steps.dfc["isMinDiameter"]) == [0, 0]
        assert list(steps.dfc["IncreaseDia"]) == [0, 0]
        assert list(steps.dfc["ReduceDia"]) == [1, 1]

    def test_diameter_features_mixed(self, steps):
        """Test with mixed diameter relationships."""
        steps.dfc = pd.DataFrame(
            [
                {"Geom1": 0.5, "MinDiameter": 0.5},  # Equal
                {"Geom1": 0.3, "MinDiameter": 0.5},  # Needs increase
                {"Geom1": 1.0, "MinDiameter": 0.5},  # Can reduce
            ]
        )
        steps.dfc = steps.dfc

        steps.diameter_features()

        assert list(steps.dfc["isMinDiameter"]) == [1, 0, 0]
        assert list(steps.dfc["IncreaseDia"]) == [0, 1, 0]
        assert list(steps.dfc["ReduceDia"]) == [0, 0, 1]


class TestNormalizeRoughness:
    """Tests for the normalize_roughness method."""

    def test_normalize_roughness_typical_values(self, steps):
        """Test normalization with typical roughness values."""
        steps.dfc = pd.DataFrame(
            [
                {"Roughness": 0.009},  # Minimum (smoothest) -> 0.0
                {"Roughness": 0.020},  # Maximum (roughest) -> 1.0
                {"Roughness": 0.0145},  # Midpoint -> 0.5
            ]
        )
        steps.dfc = steps.dfc

        steps.normalize_roughness()

        assert "NRoughness" in steps.dfc.columns
        assert steps.dfc.loc[0, "NRoughness"] == pytest.approx(0.0, abs=1e-9)
        assert steps.dfc.loc[1, "NRoughness"] == pytest.approx(1.0, abs=1e-9)
        assert steps.dfc.loc[2, "NRoughness"] == pytest.approx(0.5, abs=1e-9)

    def test_normalize_roughness_clipping(self, steps):
        """Test that values outside range are clipped."""
        steps.dfc = pd.DataFrame(
            [
                {"Roughness": 0.005},  # Below min, should clip to 0.0
                {"Roughness": 0.030},  # Above max, should clip to 1.0
            ]
        )
        steps.dfc = steps.dfc

        steps.normalize_roughness()

        assert steps.dfc.loc[0, "NRoughness"] == pytest.approx(0.0, abs=1e-9)
        assert steps.dfc.loc[1, "NRoughness"] == pytest.approx(1.0, abs=1e-9)

    def test_normalize_roughness_with_none_dfc(self, steps):
        """Test normalize_roughness returns early when dfc is None."""
        steps.dfc = None
        steps.normalize_roughness()  # Should not raise


class TestNormalizeMaxVelocity:
    """Tests for the normalize_max_velocity method."""

    def test_normalize_max_velocity_typical_values(self, steps):
        """Test normalization with typical velocity values."""
        steps.dfc = pd.DataFrame(
            [
                {"MaxV": 0.0},  # Zero velocity
                {"MaxV": 2.5},  # Half of max (assuming max=5.0)
                {"MaxV": 5.0},  # Max velocity
            ]
        )
        steps.dfc = steps.dfc

        steps.normalize_max_velocity()

        assert "NMaxV" in steps.dfc.columns
        assert steps.dfc.loc[0, "NMaxV"] == pytest.approx(0.0, abs=1e-9)
        assert steps.dfc.loc[1, "NMaxV"] == pytest.approx(0.5, abs=1e-9)
        assert steps.dfc.loc[2, "NMaxV"] == pytest.approx(1.0, abs=1e-9)

    def test_normalize_max_velocity_clipping(self, steps):
        """Test that values above max are clipped."""
        steps.dfc = pd.DataFrame(
            [
                {"MaxV": 10.0},  # Above max, should clip to 1.0
            ]
        )
        steps.dfc = steps.dfc

        steps.normalize_max_velocity()

        assert steps.dfc.loc[0, "NMaxV"] == pytest.approx(1.0, abs=1e-9)

    def test_normalize_max_velocity_with_none_dfc(self, steps):
        """Test normalize_max_velocity returns early when dfc is None."""
        steps.dfc = None
        steps.normalize_max_velocity()  # Should not raise


class TestNormalizeDepth:
    """Tests for the normalize_depth method."""

    def test_normalize_depth_typical_values(self, steps):
        """Test depth normalization with typical values."""
        steps.frost_zone = 1.0
        steps.dfc = pd.DataFrame(
            [
                {
                    "Geom1": 0.5,
                    "InletMaxDepth": 1.5,  # min_depth = 1.0 + 0.5 = 1.5
                    "OutletMaxDepth": 1.5,
                },
                {
                    "Geom1": 0.5,
                    "InletMaxDepth": 4.5,  # Some middle value
                    "OutletMaxDepth": 4.5,
                },
            ]
        )
        steps.dfc = steps.dfc

        steps.normalize_depth()

        assert "NInletDepth" in steps.dfc.columns
        assert "NOutletDepth" in steps.dfc.columns
        # First row: depth at minimum -> normalized to 0.0
        assert steps.dfc.loc[0, "NInletDepth"] == pytest.approx(0.0, abs=1e-9)
        assert steps.dfc.loc[0, "NOutletDepth"] == pytest.approx(0.0, abs=1e-9)

    def test_normalize_depth_with_none_dfc(self, steps):
        """Test normalize_depth returns early when dfc is None."""
        steps.dfc = None
        steps.normalize_depth()  # Should not raise


class TestNormalizeFilling:
    """Tests for the normalize_filling method."""

    def test_normalize_filling_typical_values(self, steps):
        """Test filling normalization with typical values."""
        steps.dfc = pd.DataFrame(
            [
                {"Filling": 0.0, "Geom1": 1.0},  # Zero filling -> 0.0
                {"Filling": 0.4135, "Geom1": 1.0},  # Half of max (0.827*1.0) -> 0.5
                {"Filling": 0.827, "Geom1": 1.0},  # Max filling -> 1.0
            ]
        )
        steps.dfc = steps.dfc

        steps.normalize_filling()

        assert "NFilling" in steps.dfc.columns
        assert steps.dfc.loc[0, "NFilling"] == pytest.approx(0.0, abs=1e-9)
        assert steps.dfc.loc[1, "NFilling"] == pytest.approx(0.5, abs=1e-9)
        assert steps.dfc.loc[2, "NFilling"] == pytest.approx(1.0, abs=1e-9)

    def test_normalize_filling_with_none_dfc(self, steps):
        """Test normalize_filling returns early when dfc is None."""
        steps.dfc = None
        steps.normalize_filling()  # Should not raise


class TestConduitsSubcatchmentInfo:
    """Tests for the conduits_subcatchment_info method."""

    def test_conduits_subcatchment_from_outlet_node(self, steps):
        """Test subcatchment info is mapped from outlet node when available."""
        steps.dfn = pd.DataFrame(
            {
                "Subcatchment": ["SC1", "SC2"],
                "SbcCategory": ["urban_highly_impervious", "rural"],
            },
            index=["N1", "N2"],
        )
        steps.dfc = pd.DataFrame(
            [
                {"InletNode": "N1", "OutletNode": "N2"},
            ]
        )
        steps.dfc = steps.dfc
        steps.dfn = steps.dfn

        steps.conduits_subcatchment_info()

        assert steps.dfc.loc[0, "Subcatchment"] == "SC2"
        assert steps.dfc.loc[0, "SbcCategory"] == "rural"

    def test_conduits_subcatchment_from_inlet_node(self, steps):
        """Test subcatchment info is mapped from inlet node when outlet has no info."""
        steps.dfn = pd.DataFrame(
            {
                "Subcatchment": ["SC1", "-"],
                "SbcCategory": ["urban_highly_impervious", "-"],
            },
            index=["N1", "N2"],
        )
        steps.dfc = pd.DataFrame(
            [
                {"InletNode": "N1", "OutletNode": "N2"},
            ]
        )
        steps.dfc = steps.dfc
        steps.dfn = steps.dfn

        steps.conduits_subcatchment_info()

        assert steps.dfc.loc[0, "Subcatchment"] == "SC1"
        assert steps.dfc.loc[0, "SbcCategory"] == "urban_highly_impervious"

    def test_conduits_subcatchment_no_info(self, steps):
        """Test subcatchment stays as '-' when neither node has info."""
        steps.dfn = pd.DataFrame(
            {
                "Subcatchment": ["-", "-"],
                "SbcCategory": ["-", "-"],
            },
            index=["N1", "N2"],
        )
        steps.dfc = pd.DataFrame(
            [
                {"InletNode": "N1", "OutletNode": "N2"},
            ]
        )
        steps.dfc = steps.dfc
        steps.dfn = steps.dfn

        steps.conduits_subcatchment_info()

        assert steps.dfc.loc[0, "Subcatchment"] == "-"
        assert steps.dfc.loc[0, "SbcCategory"] == "-"

    def test_conduits_subcatchment_with_none_dfc(self, steps):
        """Test conduits_subcatchment_info returns early when dfc is None."""
        steps.dfc = None
        steps.conduits_subcatchment_info()  # Should not raise

    def test_conduits_subcatchment_missing_columns(self, steps):
        """Test conduits_subcatchment_info returns early when required columns missing in dfn."""
        steps.dfn = pd.DataFrame({"SomeColumn": [1]}, index=["N1"])
        steps.dfc = pd.DataFrame([{"InletNode": "N1", "OutletNode": "N2"}])
        steps.dfc = steps.dfc
        steps.dfn = steps.dfn

        # Should return early without error
        steps.conduits_subcatchment_info()


class TestPropagateSubcatchmentInfo:
    """Tests for the propagate_subcatchment_info method."""

    def test_propagate_subcatchment_info_basic(self, steps):
        """Test that subcatchment info propagates through the network."""
        # Node N1 has subcatchment info, N2 doesn't (gets it from N1)
        steps.dfn = pd.DataFrame(
            {
                "Subcatchment": ["SC1", "-"],
                "SbcCategory": ["rural", "-"],
            },
            index=["N1", "N2"],
        )
        steps.dfc = pd.DataFrame(
            [
                {"InletNode": "N1", "OutletNode": "N2", "Subcatchment": "-", "SbcCategory": "-"},
            ]
        )
        steps.dfc = steps.dfc
        steps.dfn = steps.dfn

        steps.propagate_subcatchment_info()

        # Node N2 should now have subcatchment info from N1
        assert steps.dfn.loc["N2", "Subcatchment"] == "SC1"
        assert steps.dfn.loc["N2", "SbcCategory"] == "rural"
        # Conduit should also have subcatchment info
        assert steps.dfc.loc[0, "Subcatchment"] == "SC1"
        assert steps.dfc.loc[0, "SbcCategory"] == "rural"

    def test_propagate_subcatchment_with_none_dfc(self, steps):
        """Test propagate_subcatchment_info returns early when dfc is None."""
        steps.dfc = None
        steps.propagate_subcatchment_info()  # Should not raise


class TestEncodeSbcCategory:
    """Tests for the encode_sbc_category method."""

    def test_encode_sbc_category_basic(self, steps):
        """Test one-hot encoding of subcatchment categories."""
        steps.dfc = pd.DataFrame(
            [
                {"SbcCategory": "rural"},
                {"SbcCategory": "urban_highly_impervious"},
                {"SbcCategory": "forests"},
            ]
        )
        steps.dfc = steps.dfc

        steps.encode_sbc_category()

        # Check that all category columns exist
        from stormlab.contracts.features import FEATURE_ONEHOT_ORDER

        for category in FEATURE_ONEHOT_ORDER:
            assert category in steps.dfc.columns

        # Check values
        assert steps.dfc.loc[0, "rural"] == 1
        assert steps.dfc.loc[0, "urban_highly_impervious"] == 0
        assert steps.dfc.loc[1, "urban_highly_impervious"] == 1
        assert steps.dfc.loc[1, "rural"] == 0
        assert steps.dfc.loc[2, "forests"] == 1

    def test_encode_sbc_category_with_none_dfc(self, steps):
        """Test encode_sbc_category returns early when dfc is None."""
        steps.dfc = None
        steps.encode_sbc_category()  # Should not raise

    def test_encode_sbc_category_empty_df(self, steps):
        """Test encode_sbc_category returns early when dfc is empty."""
        steps.dfc = pd.DataFrame()
        steps.dfc = steps.dfc
        steps.encode_sbc_category()  # Should not raise

    def test_encode_sbc_category_missing_column(self, steps):
        """Test encode_sbc_category returns early when SbcCategory column is missing."""
        steps.dfc = pd.DataFrame([{"SomeColumn": 1}])
        steps.dfc = steps.dfc
        steps.encode_sbc_category()  # Should not raise


class TestSearchOutcomeMessages:
    """The minimum diameter search may fall through on outcomes, never on defects."""

    def test_the_fallthrough_set_is_exactly_the_three_named_formula_outcomes(self):
        """A fourth message added to `formulas` must force a decision, not fall through."""
        assert SEARCH_OUTCOME_MESSAGES == {
            formulas.SLOPE_TOO_SMALL_MESSAGE,
            formulas.SLOPE_TOO_LARGE_MESSAGE,
            formulas.FILLING_EXCEEDS_DIAMETER_MESSAGE,
        }

    @pytest.mark.parametrize(
        "message",
        [
            formulas.SLOPE_TOO_SMALL_MESSAGE,
            formulas.SLOPE_TOO_LARGE_MESSAGE,
            formulas.FILLING_EXCEEDS_DIAMETER_MESSAGE,
        ],
    )
    def test_a_search_outcome_passes_through_so_the_search_can_continue(self, message):
        _reraise_unless_search_outcome(ValueError(message), "C1", 0.3, 0.005)

    def test_any_other_value_error_is_raised_with_the_row_that_produced_it(self):
        """The legacy `except ValueError: pass` hid a bad diameter as a shrug."""
        with pytest.raises(ConduitFeatureError) as raised:
            _reraise_unless_search_outcome(ValueError("Diameter out of range for common diameters"), "C7", 3.5, 0.002)

        message = str(raised.value)
        assert "C7" in message
        assert "3.5" in message
        assert "0.002" in message
        assert "Diameter out of range" in message

    def test_an_already_wrapped_failure_is_not_wrapped_twice(self):
        """`ConduitFeatureError` is a `ValueError`, so the enclosing handler sees it again."""
        original = ConduitFeatureError("Minimum diameter search failed on conduit 'C7'")

        with pytest.raises(ConduitFeatureError) as raised:
            _reraise_unless_search_outcome(original, "C1", 0.3, 0.005)

        assert raised.value is original
