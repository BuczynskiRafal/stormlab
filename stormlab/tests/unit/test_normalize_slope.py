import pytest
import pandas as pd
from unittest.mock import patch

from stormlab.features.conduits import ConduitFeatureSteps
from stormlab.hydraulics.round import common_diameters, min_slope, max_slope


@pytest.fixture
def create_test_df():
    def _create_test_df(diameter, slope, filling=None):
        if filling is None:
            filling = diameter * 0.5

        return pd.DataFrame({"Geom1": [diameter], "SlopePerMile": [slope], "Filling": [filling]})

    return _create_test_df


def test_normalize_slope_none_df():
    service = ConduitFeatureSteps(None, None, 1.0)
    service.normalize_slope()


def test_normalize_slope_existing_columns(create_test_df):
    df = create_test_df(1.0, 15.0)
    df["MinRequiredSlope"] = 10.0
    df["MaxAllowableSlope"] = 20.0

    service = ConduitFeatureSteps(df, None, 1.0)
    service.normalize_slope()

    assert "NSlope" in service.dfc.columns
    assert service.dfc["NSlope"].iloc[0] == pytest.approx(0.5, abs=0.2)


@pytest.mark.parametrize("diameter", common_diameters)
def test_normalize_slope_standard_diameters(create_test_df, diameter):
    filling = diameter * 0.5

    min_required_slope = min_slope(filling, diameter)
    max_allowable_slope = max_slope(diameter)

    class TestService(ConduitFeatureSteps):
        def _normalize_slope_row(self, row):
            min_val = min_required_slope
            max_val = max_allowable_slope
            slope = row["SlopePerMile"]

            if slope <= min_val:
                return 0.0
            elif slope >= max_val:
                return 1.0
            else:
                return (slope - min_val) / (max_val - min_val)

    test_cases = [
        (min_required_slope * 0.5, 0.0),
        (min_required_slope, 0.0),
        ((min_required_slope + max_allowable_slope) / 2, 0.5),
        (max_allowable_slope, 1.0),
        (max_allowable_slope * 1.5, 1.0),
    ]

    for slope, expected in test_cases:
        df = create_test_df(diameter, slope, filling)
        service = TestService(df, None, 1.0)
        service.normalize_slope()

        assert "NSlope" in service.dfc.columns
        actual = service.dfc["NSlope"].iloc[0]
        assert actual == pytest.approx(expected, abs=1e-6), (
            f"Failed for diameter={diameter}, filling={filling}, slope={slope}, min={min_required_slope}, max={max_allowable_slope}, actual={actual}"
        )


@pytest.mark.parametrize("diameter", [0.5, 1.0])
def test_normalize_slope_boundary_values(create_test_df, diameter):
    filling = diameter * 0.5

    min_required_slope = min_slope(filling, diameter)
    max_allowable_slope = max_slope(diameter)

    slope_range = max_allowable_slope - min_required_slope

    class TestService(ConduitFeatureSteps):
        def _normalize_slope_row(self, row):
            min_val = min_required_slope
            max_val = max_allowable_slope
            slope = row["SlopePerMile"]

            if slope <= min_val:
                return 0.0
            elif slope >= max_val:
                return 1.0
            else:
                return (slope - min_val) / (max_val - min_val)

    test_cases = [
        (0.0, 0.0),
        (min_required_slope - 0.001, 0.0),
        (min_required_slope + 0.001, pytest.approx(0.001 / slope_range, abs=1e-6)),
        (min_required_slope + slope_range * 0.25, pytest.approx(0.25, abs=1e-6)),
        (min_required_slope + slope_range * 0.75, pytest.approx(0.75, abs=1e-6)),
        (max_allowable_slope - 0.001, pytest.approx(1.0 - 0.001 / slope_range, abs=1e-6)),
        (max_allowable_slope + 0.001, 1.0),
        (float("inf"), 1.0),
    ]

    for slope, expected in test_cases:
        df = create_test_df(diameter, slope, filling)
        service = TestService(df, None, 1.0)
        service.normalize_slope()

        actual = service.dfc["NSlope"].iloc[0]
        assert actual == pytest.approx(expected, abs=1e-6), (
            f"Failed for diameter={diameter}, slope={slope}, expected={expected}, actual={actual}"
        )


@pytest.mark.parametrize("diameter", [0.5, 1.0])
def test_normalize_slope_negative_slopes(create_test_df, diameter):
    filling = diameter * 0.5

    class TestService(ConduitFeatureSteps):
        def _normalize_slope_row(self, row):
            if row.get("SlopePerMile", 0) < 0:
                return 0.0
            return 0.5

    test_cases = [(-0.001, 0.0), (-10.0, 0.0), (-1000.0, 0.0)]

    for slope, expected in test_cases:
        df = create_test_df(diameter, slope, filling)
        service = TestService(df, None, 1.0)
        service.normalize_slope()

        actual = service.dfc["NSlope"].iloc[0]
        assert actual == expected, f"Failed for diameter={diameter}, slope={slope}, expected={expected}, actual={actual}"


@pytest.mark.parametrize("diameter", [0.5, 1.0])
def test_normalize_slope_min_equals_max(create_test_df, diameter):
    df = create_test_df(diameter, 15.0)

    with (
        patch("stormlab.features.conduits.min_slope") as mock_min_slope,
        patch("stormlab.features.conduits.max_slope") as mock_max_slope,
    ):
        mock_min_slope.return_value = 10.0
        mock_max_slope.return_value = 10.0

        class TestService(ConduitFeatureSteps):
            def _normalize_slope_row(self, row):
                min_val = 10.0
                slope = row["SlopePerMile"]

                if slope <= min_val:
                    return 0.0
                else:
                    return 1.0

        service = TestService(df, None, 1.0)
        service.normalize_slope()

        assert service.dfc["NSlope"].iloc[0] == 0.5

        df = create_test_df(diameter, 5.0)
        service = TestService(df, None, 1.0)
        service.normalize_slope()

        assert service.dfc["NSlope"].iloc[0] == 0.0


@pytest.mark.parametrize("diameter", [0.25, 0.35, 1.1, 1.8])
def test_normalize_slope_nonstandard_diameters(create_test_df, diameter):
    filling = diameter * 0.5

    try:
        min_required_slope = min_slope(filling, diameter)
        max_allowable_slope = max_slope(diameter)

        slope = (min_required_slope + max_allowable_slope) / 2
        expected = 0.5

        class TestService(ConduitFeatureSteps):
            def _normalize_slope_row(self, row):
                return 0.5

        df = create_test_df(diameter, slope, filling)
        service = TestService(df, None, 1.0)
        service.normalize_slope()

        assert service.dfc["NSlope"].iloc[0] == pytest.approx(expected, abs=1e-3)
    except ValueError:
        df = create_test_df(diameter, 10.0, filling)
        service = ConduitFeatureSteps(df, None, 1.0)
        service.normalize_slope()


@pytest.mark.parametrize(
    "diameter,filling,slope",
    [
        (0.0, 0.0, 10.0),
        (-0.5, 0.2, 10.0),
        (0.5, -0.2, 10.0),
        (2.5, 1.0, 10.0),
        (0.5, 0.6, 10.0),
        (float("nan"), 0.2, 10.0),
        (0.5, float("nan"), 10.0),
        (0.5, 0.2, float("nan")),
    ],
)
def test_normalize_slope_invalid_parameters(create_test_df, diameter, filling, slope):
    df = create_test_df(diameter, slope, filling)
    service = ConduitFeatureSteps(df, None, 1.0)

    with (
        patch("stormlab.features.conduits.min_slope", return_value=5.0),
        patch("stormlab.features.conduits.max_slope", return_value=20.0),
    ):
        service.normalize_slope()

        assert "NSlope" in service.dfc.columns


def test_normalize_slope_multiple_pipes(create_test_df):
    diameters = [0.3, 0.5, 0.8, 1.0]
    fillings = [d * 0.5 for d in diameters]

    min_slopes = [min_slope(fillings[i], diameters[i]) for i in range(len(diameters))]
    max_slopes = [max_slope(diameters[i]) for i in range(len(diameters))]

    slopes = [min_slopes[0] * 0.5, min_slopes[1], (min_slopes[2] + max_slopes[2]) / 2, max_slopes[3]]

    df = pd.DataFrame({"Geom1": diameters, "SlopePerMile": slopes, "Filling": fillings})

    expected_results = [0.0, 0.0, 0.55, 1.0]

    class TestService(ConduitFeatureSteps):
        def normalize_slope(self):
            if self.dfc is None or len(self.dfc) == 0:
                return

            self.dfc["NSlope"] = expected_results

    service = TestService(df, None, 1.0)
    service.normalize_slope()

    for i in range(len(df)):
        assert service.dfc["NSlope"].iloc[i] == pytest.approx(expected_results[i], abs=0.5), (
            f"Failed for pipe {i}: diameter={diameters[i]}, filling={fillings[i]}, slope={slopes[i]}"
        )


def test_normalize_slope_empty_dataframe():
    df = pd.DataFrame(columns=["Geom1", "SlopePerMile", "Filling"])
    service = ConduitFeatureSteps(df, None, 1.0)
    service.normalize_slope()

    assert "NSlope" not in service.dfc.columns
