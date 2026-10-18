"""Unit tests for services.py module."""

import logging
import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sa.core.services import (
    HydraulicCalculationsService,
    SimulationRunnerService,
    TraceAnalysisService,
)


class TestHydraulicCalculationsServiceCalcArea:
    """Tests for calc_area method."""

    def test_half_filling(self):
        """Test area calculation when filling equals half of diameter."""
        diameter = 0.6
        filling = diameter / 2.0
        expected_area = 0.5 * math.pi * (diameter / 2.0) ** 2
        result = HydraulicCalculationsService.calc_area(filling, diameter)
        assert result == pytest.approx(expected_area, rel=1e-6)

    def test_partial_filling_below_half(self):
        """Test area calculation when filling is below half of diameter."""
        diameter = 0.5
        filling = 0.15  # Below half
        result = HydraulicCalculationsService.calc_area(filling, diameter)
        assert result > 0.0

    def test_partial_filling_above_half(self):
        """Test area calculation when filling is above half of diameter."""
        diameter = 0.5
        filling = 0.35  # Above half but within max filling (0.827 * 0.5 = 0.4135)
        result = HydraulicCalculationsService.calc_area(filling, diameter)
        assert result > 0.0

    def test_zero_filling(self):
        """Test area calculation when filling is zero."""
        result = HydraulicCalculationsService.calc_area(0.0, 0.5)
        assert result == 0.0

    def test_invalid_diameter(self):
        """Test area calculation with invalid diameter."""
        result = HydraulicCalculationsService.calc_area(0.2, 0.0)
        assert result == 0.0

    def test_filling_exceeds_diameter(self):
        """Test area calculation when filling exceeds diameter."""
        result = HydraulicCalculationsService.calc_area(0.8, 0.5)
        assert result == 0.0

    def test_filling_equals_diameter_returns_zero(self):
        """Test that filling == diameter returns 0.0 (validated out by decorator)."""
        result = HydraulicCalculationsService.calc_area(0.5, 0.5)
        assert result == 0.0


class TestHydraulicCalculationsServiceCalcU:
    """Tests for calc_u (wetted perimeter) method."""

    def test_half_filling(self):
        """Test wetted perimeter when filling equals half of diameter."""
        diameter = 0.6
        filling = diameter / 2.0
        radius = diameter / 2.0
        expected_perimeter = math.pi * radius
        result = HydraulicCalculationsService.calc_u(filling, diameter)
        assert result == pytest.approx(expected_perimeter, rel=1e-6)

    def test_partial_filling_below_half(self):
        """Test wetted perimeter when filling is below half of diameter."""
        diameter = 0.5
        filling = 0.15
        result = HydraulicCalculationsService.calc_u(filling, diameter)
        assert result > 0.0

    def test_partial_filling_above_half(self):
        """Test wetted perimeter when filling is above half of diameter."""
        diameter = 0.5
        filling = 0.35
        result = HydraulicCalculationsService.calc_u(filling, diameter)
        assert result > 0.0

    def test_zero_filling(self):
        """Test wetted perimeter when filling is zero."""
        result = HydraulicCalculationsService.calc_u(0.0, 0.5)
        assert result == 0.0

    def test_filling_equals_diameter_returns_zero(self):
        """Test that filling == diameter returns 0.0 (validated out by decorator)."""
        result = HydraulicCalculationsService.calc_u(0.5, 0.5)
        assert result == 0.0


class TestHydraulicCalculationsServiceCalcVelocity:
    """Tests for calc_velocity method."""

    def test_velocity_with_zero_rh(self):
        """Test velocity calculation when hydraulic radius is zero (line 166)."""
        result = HydraulicCalculationsService.calc_velocity(0.0, 0.5, 0.01)
        assert result == 0.0

    def test_valid_velocity_calculation(self):
        """Test valid velocity calculation."""
        result = HydraulicCalculationsService.calc_velocity(0.2, 0.5, 0.01)
        assert result > 0.0

    def test_slope_exceeds_maximum_raises(self):
        """Test velocity calculation raises when slope exceeds maximum (line 159)."""
        # For diameter 0.2, max slope is around 129 according to max_slopes
        # A slope of 500 should exceed any max slope
        with pytest.raises(ValueError, match="Slope exceeds maximum allowed value"):
            HydraulicCalculationsService.calc_velocity(0.1, 0.2, 500)

    def test_slope_too_small_raises(self):
        """Test velocity calculation raises when slope is too small."""
        with pytest.raises(ValueError, match="Slope is too small"):
            HydraulicCalculationsService.calc_velocity(0.2, 0.5, 0.00001)


class TestHydraulicCalculationsServiceCalcFillingPercentage:
    """Tests for calc_filling_percentage method."""

    def test_filling_percentage_valid(self):
        """Test filling percentage with valid inputs."""
        result = HydraulicCalculationsService.calc_filling_percentage(0.25, 0.5)
        assert result == pytest.approx(50.0, rel=1e-6)

    def test_filling_percentage_zero(self):
        """Test filling percentage when filling is zero."""
        result = HydraulicCalculationsService.calc_filling_percentage(0.0, 0.5)
        assert result == 0.0

    def test_filling_percentage_invalid(self):
        """Test filling percentage with invalid filling (exceeds max)."""
        result = HydraulicCalculationsService.calc_filling_percentage(0.6, 0.5)
        assert result == 0.0


class TestHydraulicCalculationsServiceCalcRh:
    """Tests for calc_rh (hydraulic radius) method."""

    def test_hydraulic_radius_valid(self):
        """Test hydraulic radius with valid inputs."""
        result = HydraulicCalculationsService.calc_rh(0.25, 0.5)
        assert result > 0.0

    def test_hydraulic_radius_zero_filling(self):
        """Test hydraulic radius when filling is zero."""
        result = HydraulicCalculationsService.calc_rh(0.0, 0.5)
        assert result == 0.0


class TestHydraulicCalculationsServiceCalcFlow:
    """Tests for calc_flow method."""

    def test_flow_valid_inputs(self):
        """Test flow calculation with valid inputs."""
        result = HydraulicCalculationsService.calc_flow(0.2, 0.5, 0.01)
        assert result > 0.0

    def test_flow_zero_filling(self):
        """Test flow calculation when filling is zero."""
        result = HydraulicCalculationsService.calc_flow(0.0, 0.5, 0.01)
        assert result == 0.0

    def test_flow_with_slope_exceeds_max_warns_then_raises(self, caplog):
        """Test that flow calculation logs a warning when slope exceeds max.

        Note: calc_flow logs the warning but then calls calc_velocity which raises.
        """
        with caplog.at_level(logging.WARNING, logger="sa.core.services"):
            with pytest.raises(ValueError, match="Slope exceeds maximum allowed value"):
                HydraulicCalculationsService.calc_flow(0.1, 0.2, 500)
        assert "Slope exceeds maximum allowed value" in caplog.text

    def test_flow_with_slope_too_small_warns_then_raises(self, caplog):
        """Test that flow calculation logs a warning when slope is too small.

        Note: calc_flow logs the warning but then calls calc_velocity which raises.
        """
        with caplog.at_level(logging.WARNING, logger="sa.core.services"):
            with pytest.raises(ValueError, match="Slope is too small"):
                HydraulicCalculationsService.calc_flow(0.2, 0.5, 0.0001)
        assert "Slope is too small" in caplog.text


class TestHydraulicCalculationsServiceCalcFilling:
    """Tests for calc_filling method."""

    def test_filling_valid_flow(self):
        """Test filling calculation with valid flow."""
        result = HydraulicCalculationsService.calc_filling(0.01, 0.5, 0.01)
        assert result >= 0.0

    def test_filling_zero_flow(self):
        """Test filling calculation when flow is zero."""
        result = HydraulicCalculationsService.calc_filling(0.0, 0.5, 0.01)
        assert result == 0.0

    def test_filling_negative_flow(self):
        """Test filling calculation with negative flow."""
        with pytest.raises(ValueError, match="Invalid flow rate"):
            HydraulicCalculationsService.calc_filling(-0.01, 0.5, 0.01)

    def test_filling_diameter_out_of_range_low(self):
        """Test filling calculation with diameter below range."""
        with pytest.raises(ValueError, match="Diameter out of range"):
            HydraulicCalculationsService.calc_filling(0.01, 0.05, 0.01)

    def test_filling_diameter_out_of_range_high(self):
        """Test filling calculation with diameter above range."""
        with pytest.raises(ValueError, match="Diameter out of range"):
            HydraulicCalculationsService.calc_filling(0.01, 5.0, 0.01)

    def test_filling_exceeds_diameter_without_achieving_flow(self):
        """Test filling calculation when filling exceeds diameter (line 234)."""
        # Use a very large flow that cannot be achieved
        with pytest.raises(ValueError, match="Filling exceeds diameter"):
            HydraulicCalculationsService.calc_filling(100.0, 0.2, 0.1)

    def test_filling_iteration_breaks_when_flow_achieved(self):
        """Test that iteration breaks when desired flow is achieved (line 227-230)."""
        # Use a small flow that should be achieved quickly
        result = HydraulicCalculationsService.calc_filling(0.001, 0.5, 0.01)
        assert 0.0 < result < 0.5


class TestSimulationRunnerService:
    """Tests for SimulationRunnerService class."""

    def test_init(self):
        """Test service initialization."""
        service = SimulationRunnerService("/path/to/file.inp")
        assert service.inp_path == "/path/to/file.inp"


class TestTraceAnalysisService:
    """Tests for TraceAnalysisService class - covering lines 263-305."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock DataManager model."""
        model = MagicMock()

        # Mock inp.outfalls with index
        outfalls_index = pd.Index(["outfall_1", "outfall_2"])
        model.inp.outfalls.index = outfalls_index

        # Mock conduits dataframe
        conduits_df = pd.DataFrame(
            {
                "Name": ["C1", "C2", "C3"],
                "InletNode": ["N1", "N2", "N3"],
                "OutletNode": ["N2", "N3", "outfall_1"],
            }
        )
        model.conduits = conduits_df

        # Mock dfc dataframe with ValMaxFill column
        dfc_df = pd.DataFrame(
            {
                "Name": ["C1", "C2", "C3"],
                "InletNode": ["N1", "N2", "N3"],
                "OutletNode": ["N2", "N3", "outfall_1"],
                "ValMaxFill": [1, 0, 1],
            },
            index=["C1", "C2", "C3"],
        )
        model.dfc = dfc_df

        return model

    def test_all_traces(self, mock_model):
        """Test all_traces method (lines 263-264)."""
        with patch("sa.core.services.trace_from_node") as mock_trace:
            mock_trace.return_value = {"conduits": ["C1", "C2"], "nodes": ["N1", "N2"]}

            service = TraceAnalysisService(mock_model)
            result = service.all_traces()

            assert "outfall_1" in result
            assert "outfall_2" in result
            assert mock_trace.call_count == 2

    def test_overflowing_pipes(self, mock_model):
        """Test overflowing_pipes method (line 270)."""
        service = TraceAnalysisService(mock_model)
        result = service.overflowing_pipes()

        assert len(result) == 1
        assert "C2" in result.index

    def test_overflowing_pipes_no_overflow(self, mock_model):
        """Test overflowing_pipes method when there are no overflowing pipes."""
        mock_model.dfc["ValMaxFill"] = [1, 1, 1]
        service = TraceAnalysisService(mock_model)
        result = service.overflowing_pipes()

        assert len(result) == 0

    def test_overflowing_traces(self, mock_model):
        """Test overflowing_traces method (lines 276-286)."""
        with patch("sa.core.services.trace_from_node") as mock_trace:
            mock_trace.side_effect = [
                {"conduits": ["C1", "C2", "C3"], "nodes": ["N1", "N2", "N3"]},
                {"conduits": ["C4", "C5"], "nodes": ["N4", "N5"]},
                {"conduits": ["C2"], "nodes": ["N2", "N3"]},
            ]

            service = TraceAnalysisService(mock_model)
            result = service.overflowing_traces()

            assert "outfall_1" in result

    def test_overflowing_traces_no_overlap(self, mock_model):
        """Test overflowing_traces method when there's no overlap."""
        mock_model.dfc["ValMaxFill"] = [1, 1, 1]

        with patch("sa.core.services.trace_from_node") as mock_trace:
            mock_trace.return_value = {"conduits": ["C99"], "nodes": ["N99"]}

            service = TraceAnalysisService(mock_model)
            result = service.overflowing_traces()

            assert len(result) == 0

    def test_place_to_change(self, mock_model):
        """Test place_to_change method (lines 300-305)."""
        with patch("sa.core.services.trace_from_node") as mock_trace:
            mock_trace.side_effect = [
                {"conduits": ["C1", "C2", "C3"], "nodes": ["N1", "N2", "N3"]},
                {"conduits": ["C4", "C5"], "nodes": ["N4", "N5"]},
                {"conduits": ["C2"], "nodes": ["N2", "N3"]},
            ]

            service = TraceAnalysisService(mock_model)
            result = service.place_to_change()

            assert isinstance(result, list)
            assert "N2" in result

    def test_place_to_change_no_traces(self, mock_model):
        """Test place_to_change when overflowing_traces returns empty."""
        mock_model.dfc["ValMaxFill"] = [1, 1, 1]

        with patch("sa.core.services.trace_from_node") as mock_trace:
            mock_trace.return_value = {"conduits": [], "nodes": []}

            service = TraceAnalysisService(mock_model)
            result = service.place_to_change()

            assert result == []

    def test_place_to_change_no_nodes_key(self, mock_model):
        """Test place_to_change when trace data has no 'nodes' key."""
        with patch("sa.core.services.trace_from_node") as mock_trace:
            mock_trace.side_effect = [
                {"conduits": ["C1", "C2", "C3"], "nodes": ["N1", "N2", "N3"]},
                {"conduits": ["C4", "C5"], "nodes": ["N4", "N5"]},
                {"conduits": ["C2"]},
            ]

            service = TraceAnalysisService(mock_model)
            result = service.place_to_change()

            assert isinstance(result, list)
