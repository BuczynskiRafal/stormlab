import numpy as np
import pytest

from stormlab.hydraulics.validators import (
    HYDROMECHANIC_COEFFICIENT,
    check_slope,
    max_velocity_value,
    min_slope_hydromechanic,
    validate_filling,
    validate_max_slope,
    validate_max_velocity,
    validate_min_slope,
    validate_min_velocity,
)


class TestValidateMaxFilling:
    def test_max_filling_valid_values(self):
        for fill, dia in zip(
            [0.16, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80],
            np.arange(0.2, 1.1, 0.1),
        ):
            assert validate_filling(fill, dia)

    def test_max_filling_invalid_values(self):
        for fill, dia in zip(
            [0.19, 0.26, 0.35, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90],
            np.arange(0.2, 1.1, 0.1),
        ):
            assert not validate_filling(fill, dia)

    def test_max_filling_zero(self):
        for dia in np.arange(0.2, 1.1, 0.1):
            assert validate_filling(0, dia)

    def test_max_filling_invalid_diameter(self):
        assert not validate_filling(2, 3.0)

    def test_max_filling_invalid_value(self):
        assert not validate_filling(1, -2)

    def test_max_filling_invalid_types(self):
        assert not validate_filling("", 1)
        assert not validate_filling("four", 1)
        assert not validate_filling([], 1)
        assert not validate_filling({}, 1)


class TestValidateMaxVelocity:
    def test_velocity_below_max_velocity(self):
        assert validate_max_velocity(3.0)

    def test_velocity_equal_to_max_velocity(self):
        assert validate_max_velocity(5.0)

    def test_velocity_above_max_velocity(self):
        assert not validate_max_velocity(15.0)

    def test_velocity_is_float(self):
        with pytest.raises(TypeError):
            validate_max_velocity("10.0")  # type: ignore

    def test_velocity_is_not_none(self):
        with pytest.raises(TypeError):
            validate_max_velocity(None)  # type: ignore


class TestValidateMinVelocity:
    def test_validate_min_velocity_equals_minimum_velocity(self):
        assert validate_min_velocity(0.7)

    def test_validate_min_velocity_below_minimum_velocity(self):
        assert not validate_min_velocity(0.6)

    def test_validate_min_velocity_above_minimum_velocity(self):
        assert validate_min_velocity(0.8)

    def test_validate_min_velocity_negative_velocity(self):
        assert not validate_min_velocity(-0.7)

    def test_validate_min_velocity_zero(self):
        assert not validate_min_velocity(0)

    def test_validate_min_velocity_maximum_velocity(self):
        assert validate_min_velocity(max_velocity_value)

    def test_validate_min_velocity_slightly_above_minimum_velocity(self):
        assert validate_min_velocity(0.701)

    def test_validate_min_velocity_slightly_bellow_minimum_velocity(self):
        assert not validate_min_velocity(0.699)

    def test_validate_min_velocity_string_value(self):
        with pytest.raises(TypeError):
            validate_min_velocity("10.0")  # type: ignore


class TestValidateMinSlope:
    @pytest.mark.parametrize(
        ("slope", "filling", "diameter"),
        [
            (5.0, 0.10, 0.2),
            (2.5, 0.20, 0.3),
            (2.5, 0.30, 0.4),
            (2.5, 0.40, 0.5),
            (2.5, 0.45, 0.6),
            (2.5, 0.50, 0.7),
            (2.5, 0.60, 0.8),
            (2.5, 0.70, 0.9),
            (2.5, 0.80, 1.0),
        ],
    )
    def test_valid_slope(self, slope, filling, diameter):
        assert validate_min_slope(slope, filling, diameter)

    @pytest.mark.parametrize(
        ("slope", "filling", "diameter"),
        [
            (-5.0, 0.10, 0.2),
            (-2.5, 0.20, 0.3),
            (-2.5, 0.30, 0.4),
            (-2.5, 0.40, 0.5),
            (-2.5, 0.45, 0.6),
            (-2.5, 0.50, 0.7),
            (-2.5, 0.60, 0.8),
            (-2.5, 0.70, 0.9),
            (-2.5, 0.80, 1.0),
        ],
    )
    def test_valid_negative_slope(self, slope, filling, diameter):
        assert not validate_min_slope(slope, filling, diameter)

    def test_invalid_slope(self):
        assert not validate_min_slope(0.0005, 0.3, 0.5)

    def test_valid_slope_with_different_theta(self):
        assert validate_min_slope(2.5, 0.2, 0.3)

    def test_valid_slope_with_different_diameter(self):
        assert validate_min_slope(1.2, 0.5, 0.6)


class TestValidMaxSlope:
    """
    Tests for validate_max_slope function.
    """

    def test_validate_max_slope_valid(self):
        """
        Test the `validate_max_slope` function with valid input.
        """
        assert validate_max_slope(0.1, 0.3)

    def test_validate_max_slope_invalid(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        assert not validate_max_slope(250, 0.2)

    def test_validate_max_slope_equal(self):
        """
        Test the `validate_max_slope` function with valid input.
        """
        assert validate_max_slope(0.7, 0.7)

    def test_validate_max_slope_invalid_diameter(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        with pytest.raises(ValueError):
            validate_max_slope(0.5, 0.1)

    def test_validate_max_slope_rejects_a_nonstandard_diameter_inside_the_range(self):
        with pytest.raises(ValueError, match="No maximum slope is defined"):
            validate_max_slope(0.5, 0.25)

    def test_validate_max_slope_rejects_500_percent_expressed_as_per_mille(self):
        with pytest.raises(ValueError, match="expected per mille"):
            validate_max_slope(5000, 0.4)

    def test_validate_max_slope_string_diameter(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        with pytest.raises(TypeError):
            validate_max_slope(0.5, "0.8")  # type: ignore

    def test_validate_max_slope_string_slope(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        with pytest.raises(TypeError):
            validate_max_slope("0.5", 0.8)  # type: ignore

    def test_validate_max_slope_negative_slope(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        with pytest.raises(ValueError):
            validate_max_slope(-0.5, 1.0)

    def test_validate_max_slope_zero_slope(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        with pytest.raises(ValueError):
            validate_max_slope(0, 0.6)

    def test_validate_max_slope_greater_than_one(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        assert validate_max_slope(1.5, 1.2)

    def test_validate_max_slope_equal_to_one(self):
        """
        Test the `validate_max_slope` function with invalid input.
        """
        assert validate_max_slope(1, 1.5)


class TestCheckSlope:
    """
    Tests for the `check_slope` function.
    """

    def test_check_slope_positive(self):
        """
        Test the `check_slope` function with positive slope.
        """
        assert check_slope(0.1)

    def test_check_slope_zero(self):
        """
        Test the `check_slope` function with zero slope.
        """
        with pytest.raises(ValueError):
            check_slope(0)

    def test_check_slope_negative(self):
        """
        Test the `check_slope` function with negative slope.
        """
        with pytest.raises(ValueError):
            check_slope(-0.1)

    def test_check_slope_not_float_or_int(self):
        """
        Test the `check_slope` function with non-float or non-int slope.
        """
        with pytest.raises(TypeError):
            check_slope("0.1")  # type: ignore

    def test_check_slope_int(self):
        """
        Test the `check_slope` function with int slope.
        """
        assert check_slope(1)

    def test_check_slope_max(self):
        """
        Test the `check_slope` function with maximum slope.
        """
        assert check_slope(900)

    def test_check_slope_string_value(self):
        """
        Test the `check_slope` function with string value.
        """
        with pytest.raises(TypeError):
            check_slope("nan")  # type: ignore


class TestMinSlopeHydromechanic:
    """Tests for min_slope_hydromechanic function."""

    def test_partial_flow_returns_positive_slope(self):
        result = min_slope_hydromechanic(0.3, 0.5)
        assert result > 0
        assert result < 1

    def test_zero_perimeter_returns_infinity(self):
        result = min_slope_hydromechanic(0.0, 0.5)
        assert result == float("inf")

    def test_it_does_not_accept_theta_because_the_coefficient_is_a_frozen_literal(self):
        with pytest.raises(TypeError):
            min_slope_hydromechanic(0.3, 0.5, theta=2.0)  # type: ignore[call-arg]

    def test_the_coefficient_is_the_rounded_literal_of_four_theta_over_g(self):
        # Documented, not derived: recomputing it from theta/g would move every
        # slope this criterion gates by a part in a thousand.
        assert HYDROMECHANIC_COEFFICIENT == pytest.approx(4 * 1.5 / 9.81 / 1000, rel=1e-3)
        assert HYDROMECHANIC_COEFFICIENT != 4 * 1.5 / 9.81 / 1000

    def test_larger_diameter_gives_smaller_slope(self):
        result_small = min_slope_hydromechanic(0.3, 0.3)
        result_large = min_slope_hydromechanic(0.3, 0.6)
        assert result_small > result_large

    def test_larger_filling_changes_result(self):
        result_small = min_slope_hydromechanic(0.2, 0.5)
        result_large = min_slope_hydromechanic(0.4, 0.5)
        assert result_small != result_large
