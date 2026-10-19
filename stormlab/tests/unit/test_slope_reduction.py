import pytest
import pandas as pd
from unittest.mock import patch

from stormlab.features.conduits import ConduitFeatureSteps
from stormlab.hydraulics.round import common_diameters, max_slope


def create_test_df(diameter, slope):
    filling = diameter * 0.5 if diameter > 0 else 0.5

    return pd.DataFrame({"Geom1": [diameter], "SlopePerMile": [slope], "Filling": [filling]})


def test_slope_reduction_none_df():
    service = ConduitFeatureSteps(None, None, 1.0)
    service.slope_reduction()


def test_slope_reduction_existing_column():
    df = pd.DataFrame({"Geom1": [1.0], "SlopePerMile": [50.0], "Filling": [0.5], "MaxAllowableSlope": [40.0]})

    service = ConduitFeatureSteps(df, None, 1.0)
    service.slope_reduction()

    assert "ReduceSlope" in service.dfc.columns
    assert service.dfc["ReduceSlope"].iloc[0] == 1


@pytest.mark.parametrize("diameter", common_diameters)
def test_slope_reduction_standard_diameters(diameter):
    """Every standard diameter has a slope envelope, so none of these inputs may raise."""
    max_allowable_slope = max_slope(diameter)

    test_cases = [(max_allowable_slope * 0.9, 0), (max_allowable_slope, 0), (max_allowable_slope * 1.1, 1)]

    for slope, expected in test_cases:
        df = pd.DataFrame({"Geom1": [diameter], "SlopePerMile": [slope], "Filling": [diameter * 0.5]})

        service = ConduitFeatureSteps(df, None, 1.0)
        service.slope_reduction()

        assert "MaxAllowableSlope" in service.dfc.columns
        assert service.dfc["MaxAllowableSlope"].iloc[0] == pytest.approx(max_allowable_slope, rel=1e-3)

        assert "ReduceSlope" in service.dfc.columns
        assert service.dfc["ReduceSlope"].iloc[0] == expected, (
            f"Failed for diameter={diameter}, slope={slope}, max_slope={max_allowable_slope}"
        )


@pytest.mark.parametrize("diameter", [0.5, 1.0])
def test_slope_reduction_boundary_values(diameter):
    max_allowable_slope = max_slope(diameter)

    test_cases = [
        (0.0, 0),
        (max_allowable_slope - 0.001, 0),
        (max_allowable_slope + 0.001, 1),
        (max_allowable_slope * 10, 1),
        (float("inf"), 1),
    ]

    for slope, expected in test_cases:
        df = pd.DataFrame({"Geom1": [diameter], "SlopePerMile": [slope], "Filling": [diameter * 0.5]})

        service = ConduitFeatureSteps(df, None, 1.0)
        service.slope_reduction()

        assert service.dfc["ReduceSlope"].iloc[0] == expected, (
            f"Failed for diameter={diameter}, slope={slope}, max_slope={max_allowable_slope}"
        )


@pytest.mark.parametrize("diameter", [0.5, 1.0])
def test_slope_reduction_negative_slopes(diameter):
    max_allowable_slope = max_slope(diameter)

    test_cases = [(-0.001, 0), (-max_allowable_slope, 0), (-1000, 0)]

    for slope, expected in test_cases:
        df = pd.DataFrame({"Geom1": [diameter], "SlopePerMile": [slope], "Filling": [diameter * 0.5]})

        service = ConduitFeatureSteps(df, None, 1.0)
        service.slope_reduction()

        assert service.dfc["ReduceSlope"].iloc[0] == expected, (
            f"Failed for diameter={diameter}, slope={slope}, max_slope={max_allowable_slope}"
        )


@pytest.mark.parametrize("diameter", [0.25, 0.35, 1.1, 1.8])
def test_slope_reduction_nonstandard_diameters(diameter):
    """``max_slope`` accepts the whole 0.2-2.0 m range, not only the standard diameters."""
    max_allowable_slope = max_slope(diameter)

    df = pd.DataFrame({"Geom1": [diameter], "SlopePerMile": [max_allowable_slope * 1.5], "Filling": [diameter * 0.5]})
    service = ConduitFeatureSteps(df, None, 1.0)
    service.slope_reduction()

    assert service.dfc["ReduceSlope"].iloc[0] == 1


@pytest.mark.parametrize("diameter", [0.0, -0.5, 2.5, float("nan")])
def test_slope_reduction_invalid_diameters(diameter):
    """A diameter outside 0.2-2.0 m has no slope envelope; refusing it is the contract.

    The anchor is short on purpose: the message of ``check_dimensions`` is wrapped
    across two indented lines, so a longer pattern would not survive ``re.search``.
    """
    df = pd.DataFrame({"Geom1": [diameter], "SlopePerMile": [50.0], "Filling": [0.5]})
    service = ConduitFeatureSteps(df, None, 1.0)

    with pytest.raises(ValueError, match="Value out of bounds"):
        service.slope_reduction()


def test_slope_reduction_multiple_pipes():
    df = pd.DataFrame(
        {"Geom1": [0.3, 0.5, 0.8, 1.0], "SlopePerMile": [20.0, 30.0, 40.0, 50.0], "Filling": [0.15, 0.25, 0.4, 0.5]}
    )

    service = ConduitFeatureSteps(df, None, 1.0)
    service.slope_reduction()

    assert "MaxAllowableSlope" in service.dfc.columns

    assert "ReduceSlope" in service.dfc.columns

    for i in range(len(df)):
        diameter = df["Geom1"].iloc[i]
        slope = df["SlopePerMile"].iloc[i]
        max_allowable = service.dfc["MaxAllowableSlope"].iloc[i]
        slope_reduction = service.dfc["ReduceSlope"].iloc[i]

        expected = 1 if slope > max_allowable else 0
        assert slope_reduction == expected, (
            f"Failed for pipe {i}: diameter={diameter}, slope={slope}, max_allowable={max_allowable}"
        )


def test_slope_reduction_calls_max_slope():
    df = create_test_df(1.0, 50.0)
    service = ConduitFeatureSteps(df, None, 1.0)

    with patch("stormlab.features.conduits.max_slope") as mock_max_slope:
        mock_max_slope.return_value = 30.0

        service.slope_reduction()

        mock_max_slope.assert_called_once_with(1.0, theta=1.5, g=9.81, manning=0.013)

        assert service.dfc["ReduceSlope"].iloc[0] == 1


def test_slope_reduction_empty_dataframe():
    df = pd.DataFrame(columns=["Geom1", "SlopePerMile", "Filling"])
    service = ConduitFeatureSteps(df, None, 1.0)

    service.slope_reduction()

    assert service.dfc is not None
    assert len(service.dfc) == 0
    assert list(service.dfc.columns) == ["Geom1", "SlopePerMile", "Filling"]


def test_configured_constants_reach_the_maximum_slope_column():
    """``theta``/``g``/``manning`` are forwarded, not read off a module-level table.

    ``round.max_slopes`` is built at import with the defaults, so a step that
    looked the limit up there would answer with them however the run is
    configured — and the configured value would silently do nothing.
    """
    df = create_test_df(1.0, 50.0)

    default = ConduitFeatureSteps(df.copy(), None, 1.0)
    default.slope_reduction()
    configured = ConduitFeatureSteps(df.copy(), None, 1.0, theta=3.0)
    configured.slope_reduction()

    assert default.dfc["MaxAllowableSlope"].iloc[0] != configured.dfc["MaxAllowableSlope"].iloc[0]
    assert default.dfc["MaxAllowableSlope"].iloc[0] == pytest.approx(max_slope(1.0))
    assert configured.dfc["MaxAllowableSlope"].iloc[0] == pytest.approx(max_slope(1.0, theta=3.0))
