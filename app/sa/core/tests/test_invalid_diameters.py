import pytest
import pandas as pd

from sa.core.data import ConduitFeatureEngineeringService


def create_test_df(diameter: float, slope: float) -> pd.DataFrame:
    filling = diameter * 0.5 if diameter > 0 else 0
    return pd.DataFrame({"Geom1": [diameter], "SlopePerMile": [slope], "Filling": [filling]})


@pytest.mark.parametrize(
    "diameter",
    [0.0, -0.5, 2.5, float("nan")],
    ids=["zero", "negative", "oversized", "nan"],
)
def test_slope_reduction_raises_on_invalid_diameter(diameter: float) -> None:
    df = create_test_df(diameter, 50.0)
    service = ConduitFeatureEngineeringService(df, None, 1.0)

    with pytest.raises(ValueError):
        service.slope_reduction()


def test_slope_reduction_succeeds_with_valid_diameter() -> None:
    df = create_test_df(0.5, 50.0)
    service = ConduitFeatureEngineeringService(df, None, 1.0)

    service.slope_reduction()

    assert "ReduceSlope" in service.dfc.columns
