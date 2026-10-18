import pytest
import pandas as pd
from sa.core.data import ConduitFeatureEngineeringService


@pytest.fixture
def service():
    dfc = pd.DataFrame(
        {
            "Geom1": [0.3, 0.5, 0.8, 1.0, 1.2, 0.0, None],
            "SlopePerMile": [5.0, 10.0, 15.0, 20.0, 25.0, 5.0, 10.0],
            "MaxQ": [0.1, 0.5, 1.5, 3.0, 5.0, 0.1, 0.5],
        }
    )

    dfn = None

    return ConduitFeatureEngineeringService(dfc, dfn, 1.0)


def test_normalize_max_q_normal_values(service):
    """Test normalization with normal values within expected ranges."""
    service.normalize_max_q()

    assert "NMaxQ" in service.dfc.columns

    assert all(0 <= val <= 1 for val in service.dfc["NMaxQ"].dropna())

    k = 0.312

    for i in range(3):
        diameter = service.dfc.loc[i, "Geom1"]
        slope = service.dfc.loc[i, "SlopePerMile"]
        max_q = service.dfc.loc[i, "MaxQ"]
        theoretical_max = k * (diameter**2.5) * (slope**0.5)
        expected = min(max_q / theoretical_max, 1.0)

        assert pytest.approx(service.dfc.loc[i, "NMaxQ"], abs=1e-6) == expected, f"Row {i} normalization incorrect"


def test_normalize_max_q_zero_diameter(service):
    """Test with zero diameter which should result in NaN or inf values that get clipped."""
    service.normalize_max_q()

    row_idx = 5

    result = service.dfc.loc[row_idx, "NMaxQ"]
    assert pd.isna(result) or result in (0.0, 1.0), f"Expected 0, 1, or NaN for zero diameter, got {result}"


def test_normalize_max_q_none_diameter(service):
    """Test with None/NaN diameter which should result in NaN values."""
    service.normalize_max_q()

    row_idx = 6
    assert pd.isna(service.dfc.loc[row_idx, "NMaxQ"]), f"Expected NaN for None diameter, got {service.dfc.loc[row_idx, 'NMaxQ']}"


def test_normalize_max_q_zero_slope(service):
    """Test with zero slope which should result in division by zero."""

    row_idx = 7
    service.dfc.loc[row_idx] = {"Geom1": 0.5, "SlopePerMile": 0.0, "MaxQ": 0.5}

    service.normalize_max_q()

    result = service.dfc.loc[row_idx, "NMaxQ"]
    assert pd.isna(result) or result in (0.0, 1.0), f"Expected 0, 1, or NaN for zero slope, got {result}"


def test_normalize_max_q_negative_slope(service):
    """Test with negative slope which is physically impossible."""

    row_idx = 8
    service.dfc.loc[row_idx] = {"Geom1": 0.5, "SlopePerMile": -5.0, "MaxQ": 0.5}

    service.normalize_max_q()

    result = service.dfc.loc[row_idx, "NMaxQ"]
    assert pd.isna(result) or result in (0.0, 1.0), f"Expected 0, 1, or NaN for negative slope, got {result}"


def test_normalize_max_q_zero_flow(service):
    """Test with zero flow which should result in zero normalized flow."""

    row_idx = 9
    service.dfc.loc[row_idx] = {"Geom1": 0.5, "SlopePerMile": 10.0, "MaxQ": 0.0}

    service.normalize_max_q()

    assert service.dfc.loc[row_idx, "NMaxQ"] == 0.0, f"Expected 0.0 for zero flow, got {service.dfc.loc[row_idx, 'NMaxQ']}"


def test_normalize_max_q_negative_flow(service):
    """Test with negative flow which is physically impossible."""

    row_idx = 10
    service.dfc.loc[row_idx] = {"Geom1": 0.5, "SlopePerMile": 10.0, "MaxQ": -0.5}

    service.normalize_max_q()

    assert service.dfc.loc[row_idx, "NMaxQ"] == 0.0, f"Expected 0.0 for negative flow, got {service.dfc.loc[row_idx, 'NMaxQ']}"


def test_normalize_max_q_very_large_flow(service):
    """Test with very large flow that exceeds theoretical maximum."""

    row_idx = 11
    service.dfc.loc[row_idx] = {"Geom1": 0.5, "SlopePerMile": 10.0, "MaxQ": 1000.0}

    service.normalize_max_q()

    assert service.dfc.loc[row_idx, "NMaxQ"] == 1.0, f"Expected 1.0 for very large flow, got {service.dfc.loc[row_idx, 'NMaxQ']}"


def test_normalize_max_q_very_large_diameter(service):
    """Test with very large diameter."""

    row_idx = 12
    service.dfc.loc[row_idx] = {"Geom1": 10.0, "SlopePerMile": 10.0, "MaxQ": 5.0}

    service.normalize_max_q()

    result = service.dfc.loc[row_idx, "NMaxQ"]
    assert 0 <= result < 0.1, f"Expected small value (<0.1) for large diameter, got {result}"


def test_normalize_max_q_very_small_diameter(service):
    """Test with very small but non-zero diameter."""

    row_idx = 13
    service.dfc.loc[row_idx] = {"Geom1": 0.01, "SlopePerMile": 10.0, "MaxQ": 0.001}

    service.normalize_max_q()

    result = service.dfc.loc[row_idx, "NMaxQ"]
    assert result > 0.5, f"Expected large value (>0.5) for small diameter, got {result}"


def test_normalize_max_q_none_dataframe():
    """Test with None dataframe which should return without error."""
    service = ConduitFeatureEngineeringService(None, None, 1.0)

    service.normalize_max_q()

    assert service.dfc is None


def test_normalize_max_q_empty_dataframe():
    """Test with empty dataframe which should raise KeyError."""
    service = ConduitFeatureEngineeringService(pd.DataFrame(), None, 1.0)

    with pytest.raises(KeyError):
        service.normalize_max_q()

    assert service.dfc.empty


def test_normalize_max_q_missing_columns():
    """Test with dataframe missing required columns."""

    dfc_missing_slope = pd.DataFrame({"Geom1": [0.3, 0.5], "MaxQ": [0.1, 0.5]})
    service = ConduitFeatureEngineeringService(dfc_missing_slope, None, 1.0)

    with pytest.raises(KeyError):
        service.normalize_max_q()

    dfc_missing_geom = pd.DataFrame({"SlopePerMile": [5.0, 10.0], "MaxQ": [0.1, 0.5]})
    service = ConduitFeatureEngineeringService(dfc_missing_geom, None, 1.0)

    with pytest.raises(KeyError):
        service.normalize_max_q()

    dfc_missing_maxq = pd.DataFrame({"Geom1": [0.3, 0.5], "SlopePerMile": [5.0, 10.0]})
    service = ConduitFeatureEngineeringService(dfc_missing_maxq, None, 1.0)

    with pytest.raises(KeyError):
        service.normalize_max_q()


def test_normalize_max_q_formula_correctness():
    """Test that the normalization formula is correctly implemented."""

    test_df = pd.DataFrame({"Geom1": [0.4], "SlopePerMile": [8.0], "MaxQ": [0.2]})
    service = ConduitFeatureEngineeringService(test_df, None, 1.0)

    service.normalize_max_q()

    k = 0.312
    diameter = 0.4
    slope = 8.0
    max_q = 0.2
    theoretical_max = k * (diameter**2.5) * (slope**0.5)
    expected = min(max_q / theoretical_max, 1.0)

    assert service.dfc.loc[0, "NMaxQ"] == pytest.approx(expected, abs=1e-6), "Normalization formula not correctly implemented"
