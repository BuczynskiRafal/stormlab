import math

import pytest

from sa.core.data import HydraulicCalculationsService as hcs


@pytest.mark.parametrize(
    "filling, diameter, expected",
    [
        (0.0, 1.0, 0.0),  # Filling equals 0
        (0.5, 1.0, 50.0),  # Filling equals half the diameter
        (0.827, 1.0, 82.7),  # Filling equals the maximum filling
        (0.0001, 1.0, 0.01),  # Very small filling
    ],
)
def test_calc_filling_percentage(filling, diameter, expected):
    result = hcs.calc_filling_percentage(filling, diameter)
    assert result == pytest.approx(expected, rel=1e-2), f"Failed for filling={filling}, diameter={diameter}"


@pytest.mark.parametrize(
    "filling, diameter, expected",
    [
        (-0.5, 1.0, 0.0),  # Negative filling
        (0.5, -1.0, 0.0),  # Negative diameter
        (-0.5, -1.0, 0.0),  # Negative filling and diameter
        (1.0, 0.1, 0.0),  # Diameter below the valid range
        (1.0, 3.0, 0.0),  # Diameter above the valid range
        (1.0, 0.0, 0.0),  # Diameter equals 0
        (1e-6, 1e-3, 0.0),  # Very small values
        (2.5, 4.3, 0.0),  # Random values
        (0.9999, 1.0, 0.0),  # Filling close to the diameter
    ],
)
def test_filling_percentage_invalid_values(filling, diameter, expected):
    result = hcs.calc_filling_percentage(filling, diameter)
    assert result == expected


@pytest.mark.parametrize(
    "filling, diameter, expected",
    [
        ("0.5", 1.0, 0.0),  # Invalid filling type (string)
        (0.5, "1.0", 0.0),  # Invalid diameter type (string)
        ("0.5", "1.0", 0.0),  # Invalid types for both arguments
        (None, 1.0, 0.0),  # Filling equals None
        (0.5, None, 0.0),  # Diameter equals None
    ],
)
def test_filling_percentage_invalid_types(filling, diameter, expected):
    result = hcs.calc_filling_percentage(filling, diameter)
    assert result == expected


# ----------------------------
# Tests for calc_area
# ----------------------------
@pytest.mark.parametrize(
    "filling, diameter, expected",
    [
        (0.0, 1.0, 0.0),  # Empty pipe
        (0.5, 1.0, 0.3927),  # Half filled
        (0.3, 1.0, 0.1981),  # Below radius
        (0.7, 1.0, 0.5872),  # Above radius
        (0.827, 1.0, 0.6946),  # Maximum allowed filling
        (0.0001, 1.0, 1.333e-6),  # Very small filling
    ],
)
def test_calc_area_valid_values(filling, diameter, expected):
    result = hcs.calc_area(filling, diameter)
    assert result == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    "filling, diameter, expected",
    [
        (-0.5, 1.0, 0.0),  # Negative filling
        (0.5, -1.0, 0.0),  # Negative diameter
        (-0.5, -1.0, 0.0),  # Both negative
        (1.5, 1.0, 0.0),  # Filling > diameter
        (1.0, 0.1, 0.0),  # Diameter below range
        (1.0, 3.0, 0.0),  # Diameter above range
        (1.0, 0.0, 0.0),  # Zero diameter
        (1.0, 1.0, 0.0),  # Fully filled
        (0.9999, 1.0, 0.0),  # Almost full
    ],
)
def test_calc_area_invalid_values(filling, diameter, expected):
    result = hcs.calc_area(filling, diameter)
    assert result == expected


@pytest.mark.parametrize(
    "filling, diameter, expected",
    [
        ("0.5", 1.0, 0.0),  # String filling
        (0.5, "1.0", 0.0),  # String diameter
        (None, 1.0, 0.0),  # None filling
        (0.5, None, 0.0),  # None diameter
        ([0.5], 1.0, 0.0),  # List filling
        (0.5, [1.0], 0.0),  # List diameter
    ],
)
def test_calc_area_invalid_types(filling, diameter, expected):
    result = hcs.calc_area(filling, diameter)
    assert result == expected


def test_calc_area_special_cases():
    # Test half-filled
    assert hcs.calc_area(0.5, 1.0) == pytest.approx(0.5 * math.pi * 0.5**2)

    # Test empty pipe
    assert hcs.calc_area(0.0, 1.0) == 0.0


def test_calc_area_invalid_filling():
    assert hcs.calc_area(-0.5, 1.0) == 0.0


def test_calc_area_invalid_diameter():
    assert hcs.calc_area(0.5, -1.0) == 0.0
    assert hcs.calc_area(0.5, 0.0) == 0.0
    assert hcs.calc_area(0.5, 3.0) == 0.0


def test_calc_area_invalid_filling_type():
    assert hcs.calc_area("0.5", 1.0) == 0.0
    assert hcs.calc_area(None, 1.0) == 0.0
    assert hcs.calc_area([0.5], 1.0) == 0.0


# ----------------------------
# Tests for calc_u
# ----------------------------
@pytest.mark.parametrize(
    "filling, diameter, expected",
    [
        (0.0, 1.0, 0.0),  # Filling equals 0
        (0.5, 1.0, math.pi / 2),  # Filling equals half the diameter
        (0.827, 1.0, 2.2836),  # Filling equals the maximum filling
        (0.0001, 1.0, 0.02),  # Very small filling
    ],
)
def test_calc_u_valid_values(filling, diameter, expected):
    result = hcs.calc_u(filling, diameter)
    assert result == pytest.approx(expected, rel=1e-2), f"Failed for filling={filling}, diameter={diameter}"


@pytest.mark.parametrize(
    "filling, diameter, expected",
    [
        (-0.5, 1.0, 0.0),  # Negative filling
        (0.5, -1.0, 0.0),  # Negative diameter
        (1.5, 1.0, 0.0),  # Filling > diameter
    ],
)
def test_calc_u_invalid_values(filling, diameter, expected):
    result = hcs.calc_u(filling, diameter)
    assert result == expected


@pytest.mark.parametrize(
    "filling, diameter, expected",
    [
        ("0.5", 1.0, 0.0),  # Invalid filling type (string)
        (0.5, "1.0", 0.0),  # Invalid diameter type (string)
        ("0.5", "1.0", 0.0),  # Invalid types for both arguments
        (None, 1.0, 0.0),  # Filling equals None
        (0.5, None, 0.0),  # Diameter equals None
    ],
)
def test_calc_u_invalid_types(filling, diameter, expected):
    result = hcs.calc_u(filling, diameter)
    assert result == expected


# ----------------------------
# Tests for calc_rh
# ----------------------------
@pytest.mark.parametrize(
    "filling, diameter, expected",
    [
        (0.0, 1.0, 0.0),  # Filling equals 0
        (0.5, 1.0, (hcs.calc_area(0.5, 1.0) / hcs.calc_u(0.5, 1.0))),  # Filling equals half the diameter
        (0.827, 1.0, (hcs.calc_area(0.827, 1.0) / hcs.calc_u(0.827, 1.0))),  # Filling equals the maximum filling
        (0.0001, 1.0, (hcs.calc_area(0.0001, 1.0) / hcs.calc_u(0.0001, 1.0))),  # Very small filling
    ],
)
def test_calc_rh_valid_values(filling, diameter, expected):
    result = hcs.calc_rh(filling, diameter)
    assert result == pytest.approx(expected, rel=1e-2), f"Failed for filling={filling}, diameter={diameter}"


@pytest.mark.parametrize(
    "filling, diameter, expected",
    [
        (-0.5, 1.0, 0.0),  # Negative filling
        (0.5, -1.0, 0.0),  # Negative diameter
        (1.5, 1.0, 0.0),  # Filling > diameter
    ],
)
def test_calc_rh_invalid_values(filling, diameter, expected):
    result = hcs.calc_rh(filling, diameter)
    assert result == expected


@pytest.mark.parametrize(
    "filling, diameter, expected",
    [
        ("0.5", 1.0, 0.0),  # Invalid filling type (string)
        (0.5, "1.0", 0.0),  # Invalid diameter type (string)
        ("0.5", "1.0", 0.0),  # Invalid types for both arguments
        (None, 1.0, 0.0),  # Filling equals None
        (0.5, None, 0.0),  # Diameter equals None
    ],
)
def test_calc_rh_invalid_types(filling, diameter, expected):
    result = hcs.calc_rh(filling, diameter)
    assert result == expected


# ----------------------------
# Tests for calc_velocity
# ----------------------------
@pytest.mark.parametrize(
    "filling, diameter, slope, expected",
    [
        (0.2, 1.0, 0.010, 1.878),  # Normal case
        (0.4135, 0.5, 0.010, 2.192),  # Max allowed filling
    ],
)
def test_calc_velocity_valid(filling, diameter, slope, expected):
    result = hcs.calc_velocity(filling, diameter, slope)
    assert round(result, 3) == expected, f"Failed for filling={filling}, diameter={diameter}, slope={slope}"


@pytest.mark.parametrize(
    "filling, diameter, slope, expected",
    [
        (0.5, 0.1, 10.0, 0.0),  # Invalid diameter (too small)
        (0.5, 2.5, 10.0, 0.0),  # Invalid diameter (too large)
    ],
)
def test_calc_velocity_invalid_diameter(filling, diameter, slope, expected):
    result = hcs.calc_velocity(filling, diameter, slope)
    assert result == expected


@pytest.mark.parametrize(
    "filling, diameter, slope",
    [
        (0.5, 1.0, 0.0),  # Zero slope
        (0.5, 1.0, -1.0),  # Negative slope
    ],
)
def test_calc_velocity_invalid_slope(filling, diameter, slope):
    with pytest.raises(ValueError):
        hcs.calc_velocity(filling, diameter, slope)


# ----------------------------
# Tests for calc_flow
# ----------------------------
@pytest.mark.parametrize(
    "filling, diameter, slope, expected",
    [
        (0.0, 1.0, 0.05, 0),  # Zero filling
        (0.5, 1.0, 0.05, 2.68),  # Normal case (converted from dm³/s to m³/s)
        (0.827, 1.0, 0.05, 5.40),  # Max allowed filling (converted from dm³/s to m³/s)onverted from dm³/s to m³/s)
    ],
)
def test_calc_flow_valid(filling, diameter, slope, expected):
    result = hcs.calc_flow(filling, diameter, slope)
    assert round(result, 2) == expected, f"Failed for filling={filling}, diameter={diameter}, slope={slope}"


@pytest.mark.parametrize(
    "filling, diameter, slope",
    [
        (0.5, 1.0, -1.0),  # Negative slope
        (0.5, 1.0, 0.0),  # Zero slope
        (0.5, 1.0, 100000.0),  # Slope too high
    ],
)
def test_calc_flow_invalid_slope(filling, diameter, slope):
    with pytest.raises(ValueError):
        hcs.calc_flow(filling, diameter, slope)


@pytest.mark.parametrize(
    "filling, diameter, slope, expected",
    [
        (1.2, 1.0, 10.0, 0.0),  # Filling exceeds diameter
        (-0.3, 1.0, 10.0, 0.0),  # Negative filling
        (0.5, 0.0, 10.0, 0.0),  # Zero diameter
        (-0.5, 1.0, 10.0, 0.0),  # Negative filling
        (0.5, -1.0, 10.0, 0.0),  # Invalid diameter
        (1.0, 1.0, 0.01, 0.0),  # Filling equals diameter
    ],
)
def test_calc_flow_invalid_inputs(filling, diameter, slope, expected):
    result = hcs.calc_flow(filling, diameter, slope)
    assert result == expected


# ----------------------------
# Tests for calc_filling
# ----------------------------
@pytest.mark.parametrize(
    "q, diameter, slope, expected",
    [
        (0.0, 1.0, 0.05, 0.0),  # Flow equals 0
        (0.001, 1.0, 0.05, 0.012),  # Small flow (converted from dmu00b3/s to mu00b3/s)
        (0.1, 1.0, 0.05, 0.095),  # Medium flow (converted from dmu00b3/s to mu00b3/s)
        (0.5, 1.0, 0.05, 0.207),  # Medium flow (converted from dmu00b3/s to mu00b3/s)
        (5.397, 1.0, 0.05, 0.827),  # Maximum flow (converted from dmu00b3/s to mu00b3/s)
    ],
)
def test_calc_filling_valid_values(q, diameter, slope, expected):
    result = hcs.calc_filling(q, diameter, slope)
    assert pytest.approx(result, rel=5e-3) == expected, f"Failed for q={q}, diameter={diameter}, slope={slope}"


@pytest.mark.parametrize(
    "q, diameter, slope",
    [
        (-10.0, 1.0, 0.01),  # Negative flow
        (10.0, -1.0, 0.01),  # Negative diameter
        (10.0, 0.1, 0.01),  # Diameter below the valid range
        (10.0, 3.0, 0.01),  # Diameter above the valid range
        (10.0, 0.0, 0.01),  # Diameter equals 0
        (10.0, 1.0, -0.01),  # Negative slope
    ],
)
def test_calc_filling_invalid_values(q, diameter, slope):
    with pytest.raises(ValueError):
        hcs.calc_filling(q, diameter, slope)


@pytest.mark.parametrize(
    "q, diameter, slope",
    [
        ("10.0", 1.0, 0.01),  # Invalid flow type (string)
        (10.0, "1.0", 0.01),  # Invalid diameter type (string)
        (10.0, 1.0, "0.01"),  # Invalid slope type (string)
        (None, 1.0, 0.01),  # Flow equals None
        (10.0, None, 0.01),  # Diameter equals None
        (10.0, 1.0, None),  # Slope equals None
    ],
)
def test_calc_filling_invalid_types(q, diameter, slope):
    with pytest.raises(TypeError):
        hcs.calc_filling(q, diameter, slope)
