"""Unit tests for the Manning-family hydraulic formulas.

Migrated from ``app/sa/core/tests/test_services.py``; the assertions are
unchanged, only the static-method calls became module-level function calls. The
``SimulationRunnerService`` and ``TraceAnalysisService`` test classes are gone
with the services themselves.
"""

import logging
import math

import pytest

from stormlab.hydraulics.formulas import (
    calc_area,
    calc_filling,
    calc_filling_percentage,
    calc_flow,
    calc_rh,
    calc_u,
    calc_velocity,
)


class TestFormulasCalcArea:
    """Tests for calc_area method."""

    def test_half_filling(self):
        """Test area calculation when filling equals half of diameter."""
        diameter = 0.6
        filling = diameter / 2.0
        expected_area = 0.5 * math.pi * (diameter / 2.0) ** 2
        result = calc_area(filling, diameter)
        assert result == pytest.approx(expected_area, rel=1e-6)

    def test_partial_filling_below_half(self):
        """Test area calculation when filling is below half of diameter."""
        diameter = 0.5
        filling = 0.15  # Below half
        result = calc_area(filling, diameter)
        assert result > 0.0

    def test_partial_filling_above_half(self):
        """Test area calculation when filling is above half of diameter."""
        diameter = 0.5
        filling = 0.35  # Above half but within max filling (0.827 * 0.5 = 0.4135)
        result = calc_area(filling, diameter)
        assert result > 0.0

    def test_zero_filling(self):
        """Test area calculation when filling is zero."""
        result = calc_area(0.0, 0.5)
        assert result == 0.0

    def test_invalid_diameter(self):
        """Test area calculation with invalid diameter."""
        result = calc_area(0.2, 0.0)
        assert result == 0.0

    def test_filling_exceeds_diameter(self):
        """Test area calculation when filling exceeds diameter."""
        result = calc_area(0.8, 0.5)
        assert result == 0.0

    def test_filling_equals_diameter_returns_zero(self):
        """Test that filling == diameter returns 0.0 (validated out by decorator)."""
        result = calc_area(0.5, 0.5)
        assert result == 0.0


class TestFormulasCalcU:
    """Tests for calc_u (wetted perimeter) method."""

    def test_half_filling(self):
        """Test wetted perimeter when filling equals half of diameter."""
        diameter = 0.6
        filling = diameter / 2.0
        radius = diameter / 2.0
        expected_perimeter = math.pi * radius
        result = calc_u(filling, diameter)
        assert result == pytest.approx(expected_perimeter, rel=1e-6)

    def test_partial_filling_below_half(self):
        """Test wetted perimeter when filling is below half of diameter."""
        diameter = 0.5
        filling = 0.15
        result = calc_u(filling, diameter)
        assert result > 0.0

    def test_partial_filling_above_half(self):
        """Test wetted perimeter when filling is above half of diameter."""
        diameter = 0.5
        filling = 0.35
        result = calc_u(filling, diameter)
        assert result > 0.0

    def test_zero_filling(self):
        """Test wetted perimeter when filling is zero."""
        result = calc_u(0.0, 0.5)
        assert result == 0.0

    def test_filling_equals_diameter_returns_zero(self):
        """Test that filling == diameter returns 0.0 (validated out by decorator)."""
        result = calc_u(0.5, 0.5)
        assert result == 0.0


class TestFormulasCalcVelocity:
    """Tests for calc_velocity method."""

    def test_velocity_with_zero_rh(self):
        """Test velocity calculation when hydraulic radius is zero (line 166)."""
        result = calc_velocity(0.0, 0.5, 0.01)
        assert result == 0.0

    def test_valid_velocity_calculation(self):
        """Test valid velocity calculation."""
        result = calc_velocity(0.2, 0.5, 0.01)
        assert result > 0.0

    def test_slope_exceeds_maximum_raises(self):
        """Test velocity calculation raises when slope exceeds maximum (line 159)."""
        # For diameter 0.2, max slope is around 129 according to max_slopes
        # A slope of 500 should exceed any max slope
        with pytest.raises(ValueError, match="Slope exceeds maximum allowed value"):
            calc_velocity(0.1, 0.2, 500)

    def test_slope_too_small_raises(self):
        """Test velocity calculation raises when slope is too small."""
        with pytest.raises(ValueError, match="Slope is too small"):
            calc_velocity(0.2, 0.5, 0.00001)


class TestFormulasCalcFillingPercentage:
    """Tests for calc_filling_percentage method."""

    def test_filling_percentage_valid(self):
        """Test filling percentage with valid inputs."""
        result = calc_filling_percentage(0.25, 0.5)
        assert result == pytest.approx(50.0, rel=1e-6)

    def test_filling_percentage_zero(self):
        """Test filling percentage when filling is zero."""
        result = calc_filling_percentage(0.0, 0.5)
        assert result == 0.0

    def test_filling_percentage_invalid(self):
        """Test filling percentage with invalid filling (exceeds max)."""
        result = calc_filling_percentage(0.6, 0.5)
        assert result == 0.0


class TestFormulasCalcRh:
    """Tests for calc_rh (hydraulic radius) method."""

    def test_hydraulic_radius_valid(self):
        """Test hydraulic radius with valid inputs."""
        result = calc_rh(0.25, 0.5)
        assert result > 0.0

    def test_hydraulic_radius_zero_filling(self):
        """Test hydraulic radius when filling is zero."""
        result = calc_rh(0.0, 0.5)
        assert result == 0.0


class TestFormulasCalcFlow:
    """Tests for calc_flow method."""

    def test_flow_valid_inputs(self):
        """Test flow calculation with valid inputs."""
        result = calc_flow(0.2, 0.5, 0.01)
        assert result > 0.0

    def test_flow_zero_filling(self):
        """Test flow calculation when filling is zero."""
        result = calc_flow(0.0, 0.5, 0.01)
        assert result == 0.0

    def test_flow_with_slope_exceeds_max_warns_then_raises(self, caplog):
        """Test that flow calculation logs a warning when slope exceeds max.

        Note: calc_flow logs the warning but then calls calc_velocity which raises.
        """
        with caplog.at_level(logging.WARNING, logger="stormlab.hydraulics.formulas"):
            with pytest.raises(ValueError, match="Slope exceeds maximum allowed value"):
                calc_flow(0.1, 0.2, 500)
        assert "Slope exceeds maximum allowed value" in caplog.text

    def test_flow_with_slope_too_small_warns_then_raises(self, caplog):
        """Test that flow calculation logs a warning when slope is too small.

        Note: calc_flow logs the warning but then calls calc_velocity which raises.
        """
        with caplog.at_level(logging.WARNING, logger="stormlab.hydraulics.formulas"):
            with pytest.raises(ValueError, match="Slope is too small"):
                calc_flow(0.2, 0.5, 0.0001)
        assert "Slope is too small" in caplog.text


class TestFormulasCalcFilling:
    """Tests for calc_filling method."""

    def test_filling_valid_flow(self):
        """Test filling calculation with valid flow."""
        result = calc_filling(0.01, 0.5, 0.01)
        assert result >= 0.0

    def test_filling_zero_flow(self):
        """Test filling calculation when flow is zero."""
        result = calc_filling(0.0, 0.5, 0.01)
        assert result == 0.0

    def test_filling_negative_flow(self):
        """Test filling calculation with negative flow."""
        with pytest.raises(ValueError, match="Invalid flow rate"):
            calc_filling(-0.01, 0.5, 0.01)

    def test_filling_diameter_out_of_range_low(self):
        """Test filling calculation with diameter below range."""
        with pytest.raises(ValueError, match="Diameter out of range"):
            calc_filling(0.01, 0.05, 0.01)

    def test_filling_diameter_out_of_range_high(self):
        """Test filling calculation with diameter above range."""
        with pytest.raises(ValueError, match="Diameter out of range"):
            calc_filling(0.01, 5.0, 0.01)

    def test_filling_exceeds_diameter_without_achieving_flow(self):
        """Test filling calculation when filling exceeds diameter (line 234)."""
        # Use a very large flow that cannot be achieved
        with pytest.raises(ValueError, match="Filling exceeds diameter"):
            calc_filling(100.0, 0.2, 0.1)

    def test_filling_iteration_breaks_when_flow_achieved(self):
        """Test that iteration breaks when desired flow is achieved (line 227-230)."""
        # Use a small flow that should be achieved quickly
        result = calc_filling(0.001, 0.5, 0.01)
        assert 0.0 < result < 0.5
