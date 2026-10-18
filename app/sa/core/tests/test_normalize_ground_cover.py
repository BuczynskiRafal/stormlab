import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from sa.core.data import ConduitFeatureEngineeringService
from sa.core.round import max_depth_value


def create_test_df(diameters, inlet_ground_covers, outlet_ground_covers):
    """Create a test dataframe with the specified parameters.

    Args:
        diameters: List of pipe diameters
        inlet_ground_covers: List of inlet ground cover values
        outlet_ground_covers: List of outlet ground cover values

    Returns:
        DataFrame with the specified values
    """
    return pd.DataFrame({"Geom1": diameters, "InletGroundCover": inlet_ground_covers, "OutletGroundCover": outlet_ground_covers})


def test_normalize_ground_cover_none_df():
    service = ConduitFeatureEngineeringService(None, None, 1.0)

    service.normalize_ground_cover()

    assert service.dfc is None


def test_normalize_ground_cover_missing_columns():
    df = pd.DataFrame(
        {
            "Geom1": [0.5, 1.0],
            "InletGroundElevation": [10.0, 12.0],
            "InletNodeInvert": [5.0, 6.0],
            "OutletGroundElevation": [9.0, 11.0],
            "OutletNodeInvert": [4.5, 5.5],
        }
    )

    service = ConduitFeatureEngineeringService(df, None, 1.0)

    original_ground_cover = service.ground_cover
    service.ground_cover = MagicMock()
    service.ground_cover.side_effect = lambda: (
        df.__setitem__("InletGroundCover", [4.5, 5.0]),
        df.__setitem__("OutletGroundCover", [4.0, 4.5]),
    )

    service.normalize_ground_cover()

    service.ground_cover.assert_called_once()

    assert "NInletGroundCover" in service.dfc.columns
    assert "NOutletGroundCover" in service.dfc.columns

    service.ground_cover = original_ground_cover


def test_normalize_ground_cover_normal_values():
    frost_zone = 1.0
    diameters = [0.5, 1.0, 1.5]

    min_covers = [frost_zone] * 3
    max_covers = [max_depth_value - d for d in diameters]

    inlet_covers = [min_covers[0] + 0.5, (min_covers[1] + max_covers[1]) / 2, max_covers[2] - 0.5]

    outlet_covers = [min_covers[0] + 0.2, (min_covers[1] + max_covers[1]) * 0.75, max_covers[2]]

    df = create_test_df(diameters, inlet_covers, outlet_covers)

    service = ConduitFeatureEngineeringService(df, None, frost_zone)

    service.normalize_ground_cover()

    for i in range(3):
        range_size = max_covers[i] - min_covers[i]
        expected_inlet_norm = (inlet_covers[i] - min_covers[i]) / range_size
        expected_outlet_norm = (outlet_covers[i] - min_covers[i]) / range_size

        assert service.dfc["NInletGroundCover"].iloc[i] == pytest.approx(expected_inlet_norm, rel=1e-5)
        assert service.dfc["NOutletGroundCover"].iloc[i] == pytest.approx(expected_outlet_norm, rel=1e-5)


def test_normalize_ground_cover_clipping():
    frost_zone = 1.0
    diameters = [0.5, 1.0, 1.5]

    min_covers = [frost_zone] * 3
    max_covers = [max_depth_value - d for d in diameters]

    inlet_covers = [min_covers[0] - 0.5, (min_covers[1] + max_covers[1]) / 2, max_covers[2] + 2.0]

    outlet_covers = [min_covers[0] - 0.2, max_covers[1] + 1.0, min_covers[2] - 1.0]

    df = create_test_df(diameters, inlet_covers, outlet_covers)

    service = ConduitFeatureEngineeringService(df, None, frost_zone)

    service.normalize_ground_cover()

    expected_inlet_clipped = [min_covers[0], inlet_covers[1], max_covers[2]]

    expected_outlet_clipped = [min_covers[0], max_covers[1], min_covers[2]]

    for i in range(3):
        range_size = max_covers[i] - min_covers[i]
        expected_inlet_norm = (expected_inlet_clipped[i] - min_covers[i]) / range_size
        expected_outlet_norm = (expected_outlet_clipped[i] - min_covers[i]) / range_size

        assert service.dfc["NInletGroundCover"].iloc[i] == pytest.approx(expected_inlet_norm, rel=1e-5)
        assert service.dfc["NOutletGroundCover"].iloc[i] == pytest.approx(expected_outlet_norm, rel=1e-5)


def test_normalize_ground_cover_zero_range():
    frost_zone = 1.0

    diameter = max_depth_value - frost_zone

    df = create_test_df([diameter], [frost_zone + 0.5], [frost_zone + 0.5])

    service = ConduitFeatureEngineeringService(df, None, frost_zone)

    service.normalize_ground_cover()

    assert service.dfc["NInletGroundCover"].iloc[0] == 0.0
    assert service.dfc["NOutletGroundCover"].iloc[0] == 0.0


def test_normalize_ground_cover_mixed_cases():
    frost_zone = 1.0

    df = pd.DataFrame(
        {
            "Geom1": [0.5, max_depth_value - frost_zone, 1.0, 2.0],
            "InletGroundCover": [2.0, frost_zone, max_depth_value, frost_zone - 0.5],
            "OutletGroundCover": [3.0, frost_zone + 0.1, frost_zone, max_depth_value],
        }
    )

    service = ConduitFeatureEngineeringService(df, None, frost_zone)

    service.normalize_ground_cover()

    max_cover_0 = max_depth_value - df["Geom1"].iloc[0]
    range_0 = max_cover_0 - frost_zone
    assert service.dfc["NInletGroundCover"].iloc[0] == pytest.approx((2.0 - frost_zone) / range_0, rel=1e-5)
    assert service.dfc["NOutletGroundCover"].iloc[0] == pytest.approx((3.0 - frost_zone) / range_0, rel=1e-5)

    assert service.dfc["NInletGroundCover"].iloc[1] == 0.0
    assert service.dfc["NOutletGroundCover"].iloc[1] == 0.0

    max_cover_2 = max_depth_value - df["Geom1"].iloc[2]
    range_2 = max_cover_2 - frost_zone
    assert service.dfc["NInletGroundCover"].iloc[2] == pytest.approx((max_cover_2 - frost_zone) / range_2, rel=1e-5)
    assert service.dfc["NOutletGroundCover"].iloc[2] == pytest.approx((frost_zone - frost_zone) / range_2, rel=1e-5)

    max_cover_3 = max_depth_value - df["Geom1"].iloc[3]
    range_3 = max_cover_3 - frost_zone
    assert service.dfc["NInletGroundCover"].iloc[3] == pytest.approx((frost_zone - frost_zone) / range_3, rel=1e-5)
    assert service.dfc["NOutletGroundCover"].iloc[3] == pytest.approx((max_cover_3 - frost_zone) / range_3, rel=1e-5)


def test_normalize_ground_cover_nan_values():
    frost_zone = 1.0

    df = pd.DataFrame({"Geom1": [0.5, 1.0, 1.5], "InletGroundCover": [2.0, np.nan, 3.0], "OutletGroundCover": [np.nan, 2.5, 3.5]})

    service = ConduitFeatureEngineeringService(df, None, frost_zone)

    service.normalize_ground_cover()

    max_cover_0 = max_depth_value - df["Geom1"].iloc[0]
    range_0 = max_cover_0 - frost_zone
    assert service.dfc["NInletGroundCover"].iloc[0] == pytest.approx((2.0 - frost_zone) / range_0, rel=1e-5)
    assert np.isnan(service.dfc["NOutletGroundCover"].iloc[0]) or service.dfc["NOutletGroundCover"].iloc[0] == 0.0

    max_cover_1 = max_depth_value - df["Geom1"].iloc[1]
    range_1 = max_cover_1 - frost_zone
    assert np.isnan(service.dfc["NInletGroundCover"].iloc[1]) or service.dfc["NInletGroundCover"].iloc[1] == 0.0
    assert service.dfc["NOutletGroundCover"].iloc[1] == pytest.approx((2.5 - frost_zone) / range_1, rel=1e-5)

    max_cover_2 = max_depth_value - df["Geom1"].iloc[2]
    range_2 = max_cover_2 - frost_zone
    assert service.dfc["NInletGroundCover"].iloc[2] == pytest.approx((3.0 - frost_zone) / range_2, rel=1e-5)
    assert service.dfc["NOutletGroundCover"].iloc[2] == pytest.approx((3.5 - frost_zone) / range_2, rel=1e-5)


def test_normalize_ground_cover_zero_frost_zone():
    frost_zone = 0.0

    df = pd.DataFrame({"Geom1": [0.5, 1.0], "InletGroundCover": [0.2, 2.0], "OutletGroundCover": [0.0, 3.0]})

    service = ConduitFeatureEngineeringService(df, None, frost_zone)

    service.normalize_ground_cover()

    for i in range(2):
        max_cover = max_depth_value - df["Geom1"].iloc[i]
        range_size = max_cover

        expected_inlet_norm = df["InletGroundCover"].iloc[i] / range_size
        expected_outlet_norm = df["OutletGroundCover"].iloc[i] / range_size

        assert service.dfc["NInletGroundCover"].iloc[i] == pytest.approx(expected_inlet_norm, rel=1e-5)
        assert service.dfc["NOutletGroundCover"].iloc[i] == pytest.approx(expected_outlet_norm, rel=1e-5)


def test_normalize_ground_cover_negative_frost_zone():
    frost_zone = -1.0

    df = pd.DataFrame({"Geom1": [0.5, 1.0], "InletGroundCover": [-0.5, 2.0], "OutletGroundCover": [0.0, 3.0]})

    service = ConduitFeatureEngineeringService(df, None, frost_zone)

    service.normalize_ground_cover()

    for i in range(2):
        max_cover = max_depth_value - df["Geom1"].iloc[i]
        range_size = max_cover - frost_zone

        inlet_clipped = max(df["InletGroundCover"].iloc[i], frost_zone)
        outlet_clipped = max(df["OutletGroundCover"].iloc[i], frost_zone)

        expected_inlet_norm = (inlet_clipped - frost_zone) / range_size
        expected_outlet_norm = (outlet_clipped - frost_zone) / range_size

        assert service.dfc["NInletGroundCover"].iloc[i] == pytest.approx(expected_inlet_norm, rel=1e-5)
        assert service.dfc["NOutletGroundCover"].iloc[i] == pytest.approx(expected_outlet_norm, rel=1e-5)


def test_normalize_ground_cover_invalid_diameters():
    frost_zone = 1.0

    df = pd.DataFrame(
        {"Geom1": [0.0, -0.5, max_depth_value + 1], "InletGroundCover": [2.0, 3.0, 4.0], "OutletGroundCover": [2.5, 3.5, 4.5]}
    )

    service = ConduitFeatureEngineeringService(df, None, frost_zone)

    service.normalize_ground_cover()

    max_cover_0 = max_depth_value
    range_0 = max_cover_0 - frost_zone
    assert service.dfc["NInletGroundCover"].iloc[0] == pytest.approx((min(2.0, max_cover_0) - frost_zone) / range_0, rel=1e-5)
    assert service.dfc["NOutletGroundCover"].iloc[0] == pytest.approx((min(2.5, max_cover_0) - frost_zone) / range_0, rel=1e-5)

    assert "NInletGroundCover" in service.dfc.columns
    assert "NOutletGroundCover" in service.dfc.columns

    assert service.dfc["NInletGroundCover"].iloc[2] == 0.0
    assert service.dfc["NOutletGroundCover"].iloc[2] == 0.0


def test_normalize_ground_cover_ground_cover_exception():
    df = pd.DataFrame(
        {
            "Geom1": [0.5, 1.0],
        }
    )

    service = ConduitFeatureEngineeringService(df, None, 1.0)

    original_ground_cover = service.ground_cover
    service.ground_cover = MagicMock(side_effect=ValueError("Missing required columns"))

    try:
        service.normalize_ground_cover()

        assert "NInletGroundCover" in service.dfc.columns
        assert "NOutletGroundCover" in service.dfc.columns
        assert (service.dfc["NInletGroundCover"] == 0.0).all()
        assert (service.dfc["NOutletGroundCover"] == 0.0).all()
    except ValueError:
        pass
    finally:
        service.ground_cover = original_ground_cover
