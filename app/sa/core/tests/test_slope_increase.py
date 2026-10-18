import pytest
import pandas as pd


from sa.core.data import ConduitFeatureEngineeringService
from sa.core.round import common_diameters, min_slope


DEFAULT_FILLING_RATIO = 0.5


def create_test_df(diameter, slope, filling=None):
    if filling is None:
        filling = diameter * DEFAULT_FILLING_RATIO

    return pd.DataFrame({"Geom1": [diameter], "SlopePerMile": [slope], "Filling": [filling]})


def test_slope_increase_none_df():
    service = ConduitFeatureEngineeringService(None, None, 1.0)

    service.slope_increase()

    assert service.dfc is None


def test_slope_increase_existing_column():
    df = create_test_df(1.0, 5.0)
    df["MinRequiredSlope"] = 10.0

    service = ConduitFeatureEngineeringService(df, None, 1.0)

    service.slope_increase()

    assert "IncreaseSlope" in service.dfc.columns
    assert service.dfc["IncreaseSlope"].iloc[0] == 1


@pytest.mark.parametrize("diameter", common_diameters)
def test_slope_increase_standard_diameters(diameter):
    filling = diameter * DEFAULT_FILLING_RATIO

    min_required_slope = min_slope(filling, diameter)

    test_cases = [(min_required_slope * 0.9, 1), (min_required_slope, 0), (min_required_slope * 1.1, 0)]

    for slope, expected in test_cases:
        df = create_test_df(diameter, slope, filling)

        service = ConduitFeatureEngineeringService(df, None, 1.0)

        service.slope_increase()

        assert "MinRequiredSlope" in service.dfc.columns
        assert service.dfc["MinRequiredSlope"].iloc[0] == pytest.approx(min_required_slope, rel=1e-3)

        assert "IncreaseSlope" in service.dfc.columns
        assert service.dfc["IncreaseSlope"].iloc[0] == expected, (
            f"Failed for diameter={diameter}, filling={filling}, slope={slope}, min_slope={min_required_slope}"
        )


@pytest.mark.parametrize("diameter", [0.5, 1.0])
@pytest.mark.parametrize("filling_ratio", [0.1, 0.3, 0.5, 0.7, 0.827])
def test_slope_increase_filling_levels(diameter, filling_ratio):
    filling = diameter * filling_ratio

    min_required_slope = min_slope(filling, diameter)

    slope = min_required_slope * 0.9

    df = create_test_df(diameter, slope, filling)

    service = ConduitFeatureEngineeringService(df, None, 1.0)

    service.slope_increase()

    assert service.dfc["IncreaseSlope"].iloc[0] == 1, (
        f"Failed for diameter={diameter}, filling_ratio={filling_ratio}, slope={slope}, min_slope={min_required_slope}"
    )


@pytest.mark.parametrize("diameter", [0.5, 1.0])
def test_slope_increase_boundary_values(diameter):
    filling = diameter * DEFAULT_FILLING_RATIO

    min_required_slope = min_slope(filling, diameter)

    test_cases = [
        (0.0, 1),
        (min_required_slope - 0.001, 1),
        (min_required_slope + 0.001, 0),
        (min_required_slope * 10, 0),
        (float("inf"), 0),
    ]

    for slope, expected in test_cases:
        df = create_test_df(diameter, slope, filling)

        service = ConduitFeatureEngineeringService(df, None, 1.0)

        service.slope_increase()

        assert service.dfc["IncreaseSlope"].iloc[0] == expected, (
            f"Failed for diameter={diameter}, slope={slope}, min_slope={min_required_slope}"
        )


@pytest.mark.parametrize("diameter", [0.5, 1.0])
def test_slope_increase_negative_slopes(diameter):
    filling = diameter * DEFAULT_FILLING_RATIO

    min_required_slope = min_slope(filling, diameter)

    test_cases = [(-0.001, 1), (-min_required_slope, 1), (-1000, 1)]

    for slope, expected in test_cases:
        df = create_test_df(diameter, slope, filling)

        service = ConduitFeatureEngineeringService(df, None, 1.0)

        service.slope_increase()

        assert service.dfc["IncreaseSlope"].iloc[0] == expected, (
            f"Failed for diameter={diameter}, slope={slope}, min_slope={min_required_slope}"
        )


@pytest.mark.parametrize("diameter", [0.25, 0.35, 1.1, 1.8])
def test_slope_increase_nonstandard_diameters(diameter):
    filling = diameter * DEFAULT_FILLING_RATIO

    try:
        min_required_slope = min_slope(filling, diameter)

        df = create_test_df(diameter, min_required_slope * 0.5, filling)
        service = ConduitFeatureEngineeringService(df, None, 1.0)
        service.slope_increase()

        assert service.dfc["IncreaseSlope"].iloc[0] == 1
    except ValueError:
        df = create_test_df(diameter, 1.0, filling)
        service = ConduitFeatureEngineeringService(df, None, 1.0)

        service.slope_increase()
