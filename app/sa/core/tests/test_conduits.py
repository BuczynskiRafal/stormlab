import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
import swmmio as sw

from sa.core.data import DataManager
from sa.core.predictor import recommendations_classifier_path
from sa.core.tests import TEST_FILE
from sa.core.valid_round import validate_filling

# Check if TensorFlow is available for tests that require model predictions
TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

# Tests that run the full DataManager pipeline need the MLP recommendation
# model, whose weights are git-ignored and exist only locally.
MLP_MODEL_AVAILABLE = os.path.exists(recommendations_classifier_path)
requires_recommendation_model = pytest.mark.skipif(
    not (TF_AVAILABLE and MLP_MODEL_AVAILABLE),
    reason="TensorFlow or MLP recommendation model artifact not available",
)

desired_width = 500
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 30)


@pytest.fixture(scope="module")
def model():
    yield sw.Model(TEST_FILE, include_rpt=True)


@pytest.fixture(scope="function")
def data_manager(model):
    """Fixture to initialize the DataManager."""
    yield DataManager(model.inp.path)


class TestConduitsData:
    """
    A test class for the ConduitsData class, containing various test cases to ensure the correct
    functionality of the methods in ConduitsData.
    """

    def test_init(self, data_manager: DataManager):
        """
        Test the initialization of the ConduitsData object and ensure that the 'conduits' attribute
        is a pandas DataFrame.
        """
        assert isinstance(data_manager.dfc, pd.DataFrame)

    def test_calculate_filling(self, data_manager: DataManager):
        """
        Test the 'calculate_filling' method of the ConduitsData class to ensure that it
        correctly calculates and adds the 'Filling' column to the 'conduits' DataFrame.
        """
        data_manager.conduit_service.calculate_filling()
        assert "Filling" in data_manager.dfc.columns
        assert all(data_manager.dfc["Filling"] >= 0)

    def test_filling_is_valid(self, data_manager: DataManager):
        """
        Test the 'filling_is_valid' method of the ConduitsData class to ensure that it correctly
        validates the conduit filling and adds the 'ValMaxFill' column to the 'conduits' DataFrame.
        """
        data_manager.conduit_service.calculate_filling()
        data_manager.conduit_service.filling_is_valid()
        assert "ValMaxFill" in data_manager.dfc.columns
        assert all(data_manager.dfc["ValMaxFill"].isin([0, 1]))

    def test_validate_filling(self):
        """
        Test the 'validate_filling' function to ensure it correctly validates the filling value based
        on the provided diameter.
        """
        assert validate_filling(0, 1)
        assert validate_filling(0.5, 1)
        assert not validate_filling(1, 1)
        assert not validate_filling(1.1, 1)

    def test_velocity_is_valid_column(self, data_manager: DataManager):
        """
        Test the 'velocity_is_valid' method of the ConduitsData class to ensure that it correctly
        adds the 'ValMaxV' column to the 'conduits' DataFrame.
        """
        assert "ValMaxV" not in data_manager.dfc.columns
        data_manager.conduit_service.velocity_is_valid()
        assert "ValMaxV" in data_manager.dfc.columns

    def test_velocity_is_valid(self, data_manager: DataManager):
        """
        Test the 'velocity_is_valid' method of the ConduitsData class to ensure that it correctly
        validates the conduit velocities and updates the 'ValMaxV' and 'ValMinV' columns in the
        'conduits' DataFrame.
        """
        data_manager.conduit_service.velocity_is_valid()
        expected_values = [1, 1]
        assert list(data_manager.dfc["ValMaxV"])[:2] == expected_values
        assert list(data_manager.dfc["ValMinV"])[:2] == expected_values

    def test_slope_per_mile_column_added(self, data_manager: DataManager):
        """
        Test if the 'SlopePerMile' column is added to the conduits dataframe
        after calling the slope_per_mile() method.
        """
        data_manager.conduit_service.slope_per_mile()
        assert "SlopePerMile" in data_manager.dfc.columns

    def test_slope_per_mile_calculation(self, data_manager: DataManager):
        """
        Test if the calculated values in the 'SlopePerMile' column are correct after
        calling the slope_per_mile() method.
        """
        data_manager.conduit_service.slope_per_mile()
        expected_values = [1.80, 6.40]  # SlopeFtPerFt * 1000
        assert list(data_manager.dfc["SlopePerMile"])[:2] == pytest.approx(expected_values, abs=1e-9)

    def test_slopes_is_valid_columns_added(self, data_manager: DataManager):
        """Test if slope validation columns are added after slopes_is_valid()."""
        data_manager.conduit_service.calculate_filling()
        data_manager.conduit_service.slope_per_mile()
        data_manager.conduit_service.normalize_slope()
        data_manager.conduit_service.slopes_is_valid()
        assert "ValMaxSlope" in data_manager.dfc.columns
        assert "ValMinSlope" in data_manager.dfc.columns

    def test_slopes_is_valid_max_slope(self, data_manager: DataManager):
        """Test if maximum slope validation is correct."""
        data_manager.conduit_service.calculate_filling()
        data_manager.conduit_service.slope_per_mile()
        data_manager.conduit_service.normalize_slope()
        data_manager.conduit_service.slopes_is_valid()
        expected_values = [1, 1]
        assert list(data_manager.dfc["ValMaxSlope"])[:2] == expected_values

    def test_slopes_is_valid_min_slope(self, data_manager: DataManager):
        """Test if minimum slope validation is correct."""
        data_manager.conduit_service.calculate_filling()
        data_manager.conduit_service.slope_per_mile()
        data_manager.conduit_service.normalize_slope()
        data_manager.conduit_service.slopes_is_valid()
        expected_values = [1, 1]
        assert list(data_manager.dfc["ValMinSlope"])[:2] == expected_values

    def test_max_depth_columns_added(self, data_manager: DataManager):
        """
        Test if the 'InletMaxDepth' and 'OutletMaxDepth' columns are added to the conduits DataFrame after
        calling the `max_depth()` method.
        """
        data_manager.conduit_service.max_depth()
        assert "InletMaxDepth" in data_manager.dfc.columns
        assert "OutletMaxDepth" in data_manager.dfc.columns

    def test_max_depth_inlet_values_match(self, data_manager, model):
        """
        Test if the 'InletMaxDepth' values in the conduits DataFrame match the corresponding 'MaxDepth' values
        in the nodes DataFrame, using the 'InletNode' as a reference.
        """
        data_manager.conduit_service.max_depth()
        nodes_data = model.nodes.dataframe
        for _, conduit in data_manager.dfc.iterrows():
            inlet_node = conduit["InletNode"]
            node_max_depth = nodes_data.loc[inlet_node, "MaxDepth"]
            conduit_inlet_max_depth = conduit["InletMaxDepth"]
            assert conduit_inlet_max_depth == node_max_depth

    def test_max_depth_outlet_values_match(self, data_manager, model):
        """
        Test if the 'OutletMaxDepth' values in the conduits DataFrame match the corresponding 'MaxDepth' values
        in the nodes DataFrame, using the 'OutletNode' as a reference.
        """
        data_manager.conduit_service.max_depth()
        data_manager.conduit_service.calculate_max_depth()
        nodes_data = model.nodes.dataframe
        for _, conduit in data_manager.dfc.iterrows():
            outlet_node = conduit["OutletNode"]
            node_max_depth = nodes_data.loc[outlet_node, "MaxDepth"]
            conduit_outlet_max_depth = conduit["OutletMaxDepth"]
            if not pd.isna(conduit_outlet_max_depth) and not pd.isna(node_max_depth):
                assert conduit_outlet_max_depth == node_max_depth

    @pytest.mark.parametrize("value", [1.0, 1.2, 1.4, 1.6])
    def test_frost_zone_valid_values(self, data_manager: DataManager, value):
        """Test setting frost_zone attribute with valid values."""
        data_manager.frost_zone = value
        assert data_manager.frost_zone == value

    @pytest.mark.parametrize("value", [0.5, 1.7, -1.0, 2.0])
    def test_frost_zone_invalid_values(self, data_manager: DataManager, value):
        """Test setting frost_zone attribute with invalid values."""
        with pytest.raises(ValueError):
            data_manager.frost_zone = value

    def test_max_ground_cover_valid_depths(self, data_manager: DataManager):
        """Test max_ground_cover_is_valid with valid depths for both inlet and outlet."""
        data_manager.dfc.at[0, "InletNodeInvert"] = 10
        data_manager.dfc.at[0, "OutletNodeInvert"] = 8
        data_manager.dfc.at[0, "InletGroundElevation"] = 15
        data_manager.dfc.at[0, "OutletGroundElevation"] = 13
        data_manager.dfc.at[0, "ValDepth"] = np.nan

        data_manager.conduit_service.max_ground_cover_is_valid()
        assert data_manager.dfc["ValDepth"].loc[0] == 1

    def test_max_ground_cover_invalid_depths(self, data_manager: DataManager):
        """Test max_ground_cover_is_valid with invalid depths for both inlet and outlet."""
        data_manager.dfc.at[0, "InletNodeInvert"] = 15
        data_manager.dfc.at[0, "OutletNodeInvert"] = 13
        data_manager.dfc.at[0, "InletGroundElevation"] = 35
        data_manager.dfc.at[0, "OutletGroundElevation"] = 32
        data_manager.dfc.at[0, "ValDepth"] = np.nan

        data_manager.conduit_service.max_ground_cover_is_valid()
        assert data_manager.dfc["ValDepth"].loc[0] == 0

    def test_calculate_max_depth(self, data_manager: DataManager):
        """
        Test the 'calculate_max_depth' method of the ConduitsData class to ensure that it correctly
        calculates the maximum depth of each conduit's outlet, based on its inlet depth, length, and slope.
        """
        data_manager.conduit_service.max_depth()
        data_manager.dfc.at[0, "InletMaxDepth"] = 10
        data_manager.dfc.at[0, "Length"] = 100
        data_manager.dfc.at[0, "OutletMaxDepth"] = 5

        data_manager.dfc.at[1, "InletMaxDepth"] = 20
        data_manager.dfc.at[1, "Length"] = 200
        data_manager.dfc.at[1, "OutletMaxDepth"] = 10
        data_manager.conduit_service.calculate_max_depth()
        assert data_manager.dfc["OutletMaxDepth"].loc[0] == 5
        assert data_manager.dfc["OutletMaxDepth"].loc[1] == 10

    def test_calculate_maximum_depth(self, data_manager: DataManager):
        """
        Test the 'calculate_max_depth' method of the ConduitsData class to ensure that it correctly
        calculates the maximum depth of each conduit's outlet, based on its inlet depth, length, and slope.
        """
        data_manager.conduit_service.max_depth()
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
        data_manager.dfc = pd.DataFrame(test_data)
        data_manager.conduit_service.dfc = data_manager.dfc
        data_manager.conduit_service.calculate_max_depth()

        assert all(~pd.isna(data_manager.dfc["OutletMaxDepth"]))
        assert data_manager.dfc.loc[0, "OutletMaxDepth"] == pytest.approx(9, abs=1e-9)
        assert data_manager.dfc.loc[1, "OutletMaxDepth"] == pytest.approx(16, abs=1e-9)
        assert data_manager.dfc.loc[2, "OutletMaxDepth"] == pytest.approx(21, abs=1e-9)

    def test_min_ground_cover_is_valid(self, data_manager: DataManager):
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
        data_manager.dfc = pd.DataFrame(test_data)
        data_manager.conduit_service.dfc = data_manager.dfc
        data_manager.frost_zone = 1.0

        # Call the method
        data_manager.conduit_service.min_ground_cover_is_valid()

        # Assertions
        assert data_manager.dfc.loc[0, "ValCoverage"] == 1
        assert data_manager.dfc.loc[1, "ValCoverage"] == 0
        assert data_manager.dfc.loc[2, "ValCoverage"] == 1

    def test_min_ground_cover_valid(self, data_manager: DataManager):
        """Test when both inlet and outlet ground covers are above frost zone."""
        data_manager.frost_zone = 1.2
        data_manager.dfc.at[0, "InletGroundCover"] = 1.5
        data_manager.dfc.at[0, "OutletGroundCover"] = 1.4
        data_manager.dfc.at[0, "ValCoverage"] = np.nan

        data_manager.conduit_service.min_ground_cover_is_valid()
        assert data_manager.dfc["ValCoverage"].loc[0] == 1

    def test_min_ground_cover_invalid(self, data_manager: DataManager):
        """Test when both inlet and outlet ground covers are in frost zone."""
        data_manager.frost_zone = 1.2
        data_manager.dfc.at[0, "InletGroundCover"] = 0.8
        data_manager.dfc.at[0, "OutletGroundCover"] = 0.9
        data_manager.dfc.at[0, "ValCoverage"] = np.nan

        data_manager.conduit_service.min_ground_cover_is_valid()
        assert data_manager.dfc["ValCoverage"].loc[0] == 0

    def test_min_ground_cover_mixed(self, data_manager: DataManager):
        """Test when inlet is valid but outlet is in frost zone."""
        data_manager.frost_zone = 1.2
        data_manager.dfc.at[0, "InletGroundCover"] = 1.5
        data_manager.dfc.at[0, "OutletGroundCover"] = 0.9
        data_manager.dfc.at[0, "ValCoverage"] = np.nan

        data_manager.conduit_service.min_ground_cover_is_valid()
        assert data_manager.dfc["ValCoverage"].loc[0] == 0

    def test_min_ground_cover_edge_case(self, data_manager: DataManager):
        """Test when ground covers are exactly at frost zone depth."""
        data_manager.frost_zone = 1.2
        data_manager.dfc.at[0, "InletGroundCover"] = 1.2
        data_manager.dfc.at[0, "OutletGroundCover"] = 1.2
        data_manager.dfc.at[0, "ValCoverage"] = np.nan

        data_manager.conduit_service.min_ground_cover_is_valid()
        assert data_manager.dfc["ValCoverage"].loc[0] == 1

    def test_ground_elevation(self, data_manager: DataManager):
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
        data_manager.dfc = pd.DataFrame(test_data)
        data_manager.conduit_service.dfc = data_manager.dfc

        # Call the method
        data_manager.conduit_service.calculate_ground_elevation()

        # Assertions for InletGroundElevation
        assert data_manager.dfc.loc[0, "InletGroundElevation"] == pytest.approx(12, abs=1e-9), (
            "Incorrect result for row 0 (Inlet)"
        )
        assert data_manager.dfc.loc[1, "InletGroundElevation"] == pytest.approx(24, abs=1e-9), (
            "Incorrect result for row 1 (Inlet)"
        )
        assert data_manager.dfc.loc[2, "InletGroundElevation"] == pytest.approx(36, abs=1e-9), (
            "Incorrect result for row 2 (Inlet)"
        )

        # Assertions for OutletGroundElevation
        assert data_manager.dfc.loc[0, "OutletGroundElevation"] == pytest.approx(48, abs=1e-9), (
            "Incorrect result for row 0 (Outlet)"
        )
        assert data_manager.dfc.loc[1, "OutletGroundElevation"] == pytest.approx(60, abs=1e-9), (
            "Incorrect result for row 1 (Outlet)"
        )
        assert data_manager.dfc.loc[2, "OutletGroundElevation"] == pytest.approx(72, abs=1e-9), (
            "Incorrect result for row 2 (Outlet)"
        )


class TestGroundCover:
    """Tests for the ground_cover method."""

    def test_ground_cover_calculation(self, data_manager: DataManager):
        """Test ground cover calculation for inlet and outlet."""
        # Prepare test data
        data_manager.dfc = pd.DataFrame(
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
        data_manager.conduit_service.dfc = data_manager.dfc

        data_manager.conduit_service.ground_cover()

        # InletGroundCover = InletGroundElevation - InletNodeInvert - Geom1
        assert data_manager.dfc.loc[0, "InletGroundCover"] == pytest.approx(4.5, abs=1e-9)
        assert data_manager.dfc.loc[1, "InletGroundCover"] == pytest.approx(7.0, abs=1e-9)
        # OutletGroundCover = OutletGroundElevation - OutletNodeInvert - Geom1
        assert data_manager.dfc.loc[0, "OutletGroundCover"] == pytest.approx(4.5, abs=1e-9)
        assert data_manager.dfc.loc[1, "OutletGroundCover"] == pytest.approx(7.0, abs=1e-9)

    def test_ground_cover_with_none_dfc(self, data_manager: DataManager):
        """Test ground_cover returns early when dfc is None."""
        data_manager.conduit_service.dfc = None
        # Should not raise, just return
        data_manager.conduit_service.ground_cover()

    def test_ground_cover_missing_columns_raises(self, data_manager: DataManager):
        """Test ground_cover raises ValueError when required columns are missing."""
        data_manager.dfc = pd.DataFrame([{"SomeColumn": 1}])
        data_manager.conduit_service.dfc = data_manager.dfc

        with pytest.raises(ValueError, match="Missing required columns"):
            data_manager.conduit_service.ground_cover()


class TestMinGroundCoverAutoCall:
    """Tests for min_ground_cover_is_valid auto-calling ground_cover."""

    def test_min_ground_cover_calls_ground_cover_when_column_missing(self, data_manager: DataManager):
        """Test that min_ground_cover_is_valid calls ground_cover if InletGroundCover is missing."""
        data_manager.dfc = pd.DataFrame(
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
        data_manager.conduit_service.dfc = data_manager.dfc
        data_manager.frost_zone = 1.0

        # InletGroundCover column doesn't exist - method should call ground_cover()
        assert "InletGroundCover" not in data_manager.dfc.columns

        data_manager.conduit_service.min_ground_cover_is_valid()

        # Now the columns should exist and ValCoverage should be calculated
        assert "InletGroundCover" in data_manager.dfc.columns
        assert "ValCoverage" in data_manager.dfc.columns


class TestDiameterFeatures:
    """Tests for the diameter_features method."""

    def test_diameter_features_all_equal(self, data_manager: DataManager):
        """Test when current diameter equals minimum diameter."""
        data_manager.dfc = pd.DataFrame(
            [
                {"Geom1": 0.5, "MinDiameter": 0.5},
                {"Geom1": 1.0, "MinDiameter": 1.0},
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc

        data_manager.conduit_service.diameter_features()

        assert list(data_manager.dfc["isMinDiameter"]) == [1, 1]
        assert list(data_manager.dfc["IncreaseDia"]) == [0, 0]
        assert list(data_manager.dfc["ReduceDia"]) == [0, 0]

    def test_diameter_features_increase_needed(self, data_manager: DataManager):
        """Test when current diameter is smaller than minimum (needs increase)."""
        data_manager.dfc = pd.DataFrame(
            [
                {"Geom1": 0.3, "MinDiameter": 0.5},
                {"Geom1": 0.8, "MinDiameter": 1.2},
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc

        data_manager.conduit_service.diameter_features()

        assert list(data_manager.dfc["isMinDiameter"]) == [0, 0]
        assert list(data_manager.dfc["IncreaseDia"]) == [1, 1]
        assert list(data_manager.dfc["ReduceDia"]) == [0, 0]

    def test_diameter_features_reduce_possible(self, data_manager: DataManager):
        """Test when current diameter is larger than minimum (can be reduced)."""
        data_manager.dfc = pd.DataFrame(
            [
                {"Geom1": 1.0, "MinDiameter": 0.5},
                {"Geom1": 1.5, "MinDiameter": 0.8},
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc

        data_manager.conduit_service.diameter_features()

        assert list(data_manager.dfc["isMinDiameter"]) == [0, 0]
        assert list(data_manager.dfc["IncreaseDia"]) == [0, 0]
        assert list(data_manager.dfc["ReduceDia"]) == [1, 1]

    def test_diameter_features_mixed(self, data_manager: DataManager):
        """Test with mixed diameter relationships."""
        data_manager.dfc = pd.DataFrame(
            [
                {"Geom1": 0.5, "MinDiameter": 0.5},  # Equal
                {"Geom1": 0.3, "MinDiameter": 0.5},  # Needs increase
                {"Geom1": 1.0, "MinDiameter": 0.5},  # Can reduce
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc

        data_manager.conduit_service.diameter_features()

        assert list(data_manager.dfc["isMinDiameter"]) == [1, 0, 0]
        assert list(data_manager.dfc["IncreaseDia"]) == [0, 1, 0]
        assert list(data_manager.dfc["ReduceDia"]) == [0, 0, 1]


class TestNormalizeRoughness:
    """Tests for the normalize_roughness method."""

    def test_normalize_roughness_typical_values(self, data_manager: DataManager):
        """Test normalization with typical roughness values."""
        data_manager.dfc = pd.DataFrame(
            [
                {"Roughness": 0.009},  # Minimum (smoothest) -> 0.0
                {"Roughness": 0.020},  # Maximum (roughest) -> 1.0
                {"Roughness": 0.0145},  # Midpoint -> 0.5
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc

        data_manager.conduit_service.normalize_roughness()

        assert "NRoughness" in data_manager.dfc.columns
        assert data_manager.dfc.loc[0, "NRoughness"] == pytest.approx(0.0, abs=1e-9)
        assert data_manager.dfc.loc[1, "NRoughness"] == pytest.approx(1.0, abs=1e-9)
        assert data_manager.dfc.loc[2, "NRoughness"] == pytest.approx(0.5, abs=1e-9)

    def test_normalize_roughness_clipping(self, data_manager: DataManager):
        """Test that values outside range are clipped."""
        data_manager.dfc = pd.DataFrame(
            [
                {"Roughness": 0.005},  # Below min, should clip to 0.0
                {"Roughness": 0.030},  # Above max, should clip to 1.0
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc

        data_manager.conduit_service.normalize_roughness()

        assert data_manager.dfc.loc[0, "NRoughness"] == pytest.approx(0.0, abs=1e-9)
        assert data_manager.dfc.loc[1, "NRoughness"] == pytest.approx(1.0, abs=1e-9)

    def test_normalize_roughness_with_none_dfc(self, data_manager: DataManager):
        """Test normalize_roughness returns early when dfc is None."""
        data_manager.conduit_service.dfc = None
        data_manager.conduit_service.normalize_roughness()  # Should not raise


class TestNormalizeMaxVelocity:
    """Tests for the normalize_max_velocity method."""

    def test_normalize_max_velocity_typical_values(self, data_manager: DataManager):
        """Test normalization with typical velocity values."""
        data_manager.dfc = pd.DataFrame(
            [
                {"MaxV": 0.0},  # Zero velocity
                {"MaxV": 2.5},  # Half of max (assuming max=5.0)
                {"MaxV": 5.0},  # Max velocity
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc

        data_manager.conduit_service.normalize_max_velocity()

        assert "NMaxV" in data_manager.dfc.columns
        assert data_manager.dfc.loc[0, "NMaxV"] == pytest.approx(0.0, abs=1e-9)
        assert data_manager.dfc.loc[1, "NMaxV"] == pytest.approx(0.5, abs=1e-9)
        assert data_manager.dfc.loc[2, "NMaxV"] == pytest.approx(1.0, abs=1e-9)

    def test_normalize_max_velocity_clipping(self, data_manager: DataManager):
        """Test that values above max are clipped."""
        data_manager.dfc = pd.DataFrame(
            [
                {"MaxV": 10.0},  # Above max, should clip to 1.0
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc

        data_manager.conduit_service.normalize_max_velocity()

        assert data_manager.dfc.loc[0, "NMaxV"] == pytest.approx(1.0, abs=1e-9)

    def test_normalize_max_velocity_with_none_dfc(self, data_manager: DataManager):
        """Test normalize_max_velocity returns early when dfc is None."""
        data_manager.conduit_service.dfc = None
        data_manager.conduit_service.normalize_max_velocity()  # Should not raise


class TestNormalizeDepth:
    """Tests for the normalize_depth method."""

    def test_normalize_depth_typical_values(self, data_manager: DataManager):
        """Test depth normalization with typical values."""
        data_manager.frost_zone = 1.0
        data_manager.dfc = pd.DataFrame(
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
        data_manager.conduit_service.dfc = data_manager.dfc

        data_manager.conduit_service.normalize_depth()

        assert "NInletDepth" in data_manager.dfc.columns
        assert "NOutletDepth" in data_manager.dfc.columns
        # First row: depth at minimum -> normalized to 0.0
        assert data_manager.dfc.loc[0, "NInletDepth"] == pytest.approx(0.0, abs=1e-9)
        assert data_manager.dfc.loc[0, "NOutletDepth"] == pytest.approx(0.0, abs=1e-9)

    def test_normalize_depth_with_none_dfc(self, data_manager: DataManager):
        """Test normalize_depth returns early when dfc is None."""
        data_manager.conduit_service.dfc = None
        data_manager.conduit_service.normalize_depth()  # Should not raise


class TestNormalizeFilling:
    """Tests for the normalize_filling method."""

    def test_normalize_filling_typical_values(self, data_manager: DataManager):
        """Test filling normalization with typical values."""
        data_manager.dfc = pd.DataFrame(
            [
                {"Filling": 0.0, "Geom1": 1.0},  # Zero filling -> 0.0
                {"Filling": 0.4135, "Geom1": 1.0},  # Half of max (0.827*1.0) -> 0.5
                {"Filling": 0.827, "Geom1": 1.0},  # Max filling -> 1.0
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc

        data_manager.conduit_service.normalize_filling()

        assert "NFilling" in data_manager.dfc.columns
        assert data_manager.dfc.loc[0, "NFilling"] == pytest.approx(0.0, abs=1e-9)
        assert data_manager.dfc.loc[1, "NFilling"] == pytest.approx(0.5, abs=1e-9)
        assert data_manager.dfc.loc[2, "NFilling"] == pytest.approx(1.0, abs=1e-9)

    def test_normalize_filling_with_none_dfc(self, data_manager: DataManager):
        """Test normalize_filling returns early when dfc is None."""
        data_manager.conduit_service.dfc = None
        data_manager.conduit_service.normalize_filling()  # Should not raise


class TestConduitsSubcatchmentInfo:
    """Tests for the conduits_subcatchment_info method."""

    def test_conduits_subcatchment_from_outlet_node(self, data_manager: DataManager):
        """Test subcatchment info is mapped from outlet node when available."""
        data_manager.dfn = pd.DataFrame(
            {
                "Subcatchment": ["SC1", "SC2"],
                "SbcCategory": ["urban_highly_impervious", "rural"],
            },
            index=["N1", "N2"],
        )
        data_manager.dfc = pd.DataFrame(
            [
                {"InletNode": "N1", "OutletNode": "N2"},
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc
        data_manager.conduit_service.dfn = data_manager.dfn

        data_manager.conduit_service.conduits_subcatchment_info()

        assert data_manager.dfc.loc[0, "Subcatchment"] == "SC2"
        assert data_manager.dfc.loc[0, "SbcCategory"] == "rural"

    def test_conduits_subcatchment_from_inlet_node(self, data_manager: DataManager):
        """Test subcatchment info is mapped from inlet node when outlet has no info."""
        data_manager.dfn = pd.DataFrame(
            {
                "Subcatchment": ["SC1", "-"],
                "SbcCategory": ["urban_highly_impervious", "-"],
            },
            index=["N1", "N2"],
        )
        data_manager.dfc = pd.DataFrame(
            [
                {"InletNode": "N1", "OutletNode": "N2"},
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc
        data_manager.conduit_service.dfn = data_manager.dfn

        data_manager.conduit_service.conduits_subcatchment_info()

        assert data_manager.dfc.loc[0, "Subcatchment"] == "SC1"
        assert data_manager.dfc.loc[0, "SbcCategory"] == "urban_highly_impervious"

    def test_conduits_subcatchment_no_info(self, data_manager: DataManager):
        """Test subcatchment stays as '-' when neither node has info."""
        data_manager.dfn = pd.DataFrame(
            {
                "Subcatchment": ["-", "-"],
                "SbcCategory": ["-", "-"],
            },
            index=["N1", "N2"],
        )
        data_manager.dfc = pd.DataFrame(
            [
                {"InletNode": "N1", "OutletNode": "N2"},
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc
        data_manager.conduit_service.dfn = data_manager.dfn

        data_manager.conduit_service.conduits_subcatchment_info()

        assert data_manager.dfc.loc[0, "Subcatchment"] == "-"
        assert data_manager.dfc.loc[0, "SbcCategory"] == "-"

    def test_conduits_subcatchment_with_none_dfc(self, data_manager: DataManager):
        """Test conduits_subcatchment_info returns early when dfc is None."""
        data_manager.conduit_service.dfc = None
        data_manager.conduit_service.conduits_subcatchment_info()  # Should not raise

    def test_conduits_subcatchment_missing_columns(self, data_manager: DataManager):
        """Test conduits_subcatchment_info returns early when required columns missing in dfn."""
        data_manager.dfn = pd.DataFrame({"SomeColumn": [1]}, index=["N1"])
        data_manager.dfc = pd.DataFrame([{"InletNode": "N1", "OutletNode": "N2"}])
        data_manager.conduit_service.dfc = data_manager.dfc
        data_manager.conduit_service.dfn = data_manager.dfn

        # Should return early without error
        data_manager.conduit_service.conduits_subcatchment_info()


class TestPropagateSubcatchmentInfo:
    """Tests for the propagate_subcatchment_info method."""

    def test_propagate_subcatchment_info_basic(self, data_manager: DataManager):
        """Test that subcatchment info propagates through the network."""
        # Node N1 has subcatchment info, N2 doesn't (gets it from N1)
        data_manager.dfn = pd.DataFrame(
            {
                "Subcatchment": ["SC1", "-"],
                "SbcCategory": ["rural", "-"],
            },
            index=["N1", "N2"],
        )
        data_manager.dfc = pd.DataFrame(
            [
                {"InletNode": "N1", "OutletNode": "N2", "Subcatchment": "-", "SbcCategory": "-"},
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc
        data_manager.conduit_service.dfn = data_manager.dfn

        data_manager.conduit_service.propagate_subcatchment_info()

        # Node N2 should now have subcatchment info from N1
        assert data_manager.dfn.loc["N2", "Subcatchment"] == "SC1"
        assert data_manager.dfn.loc["N2", "SbcCategory"] == "rural"
        # Conduit should also have subcatchment info
        assert data_manager.dfc.loc[0, "Subcatchment"] == "SC1"
        assert data_manager.dfc.loc[0, "SbcCategory"] == "rural"

    def test_propagate_subcatchment_with_none_dfc(self, data_manager: DataManager):
        """Test propagate_subcatchment_info returns early when dfc is None."""
        data_manager.conduit_service.dfc = None
        data_manager.conduit_service.propagate_subcatchment_info()  # Should not raise


class TestEncodeSbcCategory:
    """Tests for the encode_sbc_category method."""

    def test_encode_sbc_category_basic(self, data_manager: DataManager):
        """Test one-hot encoding of subcatchment categories."""
        data_manager.dfc = pd.DataFrame(
            [
                {"SbcCategory": "rural"},
                {"SbcCategory": "urban_highly_impervious"},
                {"SbcCategory": "forests"},
            ]
        )
        data_manager.conduit_service.dfc = data_manager.dfc

        data_manager.conduit_service.encode_sbc_category()

        # Check that all category columns exist
        from sa.core.constants import SUBCATCHMENT_CATEGORIES

        for category in SUBCATCHMENT_CATEGORIES:
            assert category in data_manager.dfc.columns

        # Check values
        assert data_manager.dfc.loc[0, "rural"] == 1
        assert data_manager.dfc.loc[0, "urban_highly_impervious"] == 0
        assert data_manager.dfc.loc[1, "urban_highly_impervious"] == 1
        assert data_manager.dfc.loc[1, "rural"] == 0
        assert data_manager.dfc.loc[2, "forests"] == 1

    def test_encode_sbc_category_with_none_dfc(self, data_manager: DataManager):
        """Test encode_sbc_category returns early when dfc is None."""
        data_manager.conduit_service.dfc = None
        data_manager.conduit_service.encode_sbc_category()  # Should not raise

    def test_encode_sbc_category_empty_df(self, data_manager: DataManager):
        """Test encode_sbc_category returns early when dfc is empty."""
        data_manager.dfc = pd.DataFrame()
        data_manager.conduit_service.dfc = data_manager.dfc
        data_manager.conduit_service.encode_sbc_category()  # Should not raise

    def test_encode_sbc_category_missing_column(self, data_manager: DataManager):
        """Test encode_sbc_category returns early when SbcCategory column is missing."""
        data_manager.dfc = pd.DataFrame([{"SomeColumn": 1}])
        data_manager.conduit_service.dfc = data_manager.dfc
        data_manager.conduit_service.encode_sbc_category()  # Should not raise
