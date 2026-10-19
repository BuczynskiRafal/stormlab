import pandas as pd
from unittest.mock import patch

from stormlab.features.conduits import ConduitFeatureSteps
from stormlab.hydraulics import formulas


def test_min_conduit_diameter_valid_filling():
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.3], "MaxQ": [0.05], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureSteps(df, None, 1.0)

    with patch("stormlab.features.conduits.common_diameters", [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]):
        with patch("stormlab.features.conduits.validate_filling", return_value=True):
            with patch.object(formulas, "calc_filling", return_value=0.3):
                service.min_conduit_diameter()
                assert service.dfc.loc[0, "MinDiameter"] == 0.3


def test_min_conduit_diameter_invalid_filling():
    df = pd.DataFrame({"Geom1": [0.3], "Filling": [0.25], "MaxQ": [0.05], "SlopeFtPerFt": [0.02], "ValMaxFill": [0]})
    service = ConduitFeatureSteps(df, None, 1.0)

    with patch("stormlab.features.conduits.common_diameters", [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]):
        with patch("stormlab.features.conduits.validate_filling") as mock_validate:
            mock_validate.side_effect = lambda f, d, ratio=None: d >= 0.4
            service.min_conduit_diameter()
            assert service.dfc.loc[0, "MinDiameter"] == 0.4


def test_min_conduit_diameter_very_small_flow():
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.1], "MaxQ": [0.005], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureSteps(df, None, 1.0)

    with patch("stormlab.features.conduits.common_diameters", [0.2, 0.3, 0.4, 0.5, 0.6]):
        service.min_conduit_diameter()
        assert service.dfc.loc[0, "MinDiameter"] == 0.3


def test_min_conduit_diameter_non_standard():
    df = pd.DataFrame({"Geom1": [0.35], "Filling": [0.2], "MaxQ": [0.05], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureSteps(df, None, 1.0)

    with patch("stormlab.features.conduits.common_diameters", [0.3, 0.4, 0.5, 0.6]):
        service.min_conduit_diameter()
        assert service.dfc.loc[0, "MinDiameter"] == 0.35


def test_min_conduit_diameter_low_filling_ratio():
    df = pd.DataFrame({"Geom1": [0.6], "Filling": [0.15], "MaxQ": [0.03], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureSteps(df, None, 1.0)

    with patch("stormlab.features.conduits.common_diameters", [0.3, 0.4, 0.5, 0.6]):
        with patch.object(formulas, "calc_filling", return_value=0.15):
            with patch("stormlab.features.conduits.validate_filling", return_value=True):
                service.min_conduit_diameter()
                assert service.dfc.loc[0, "MinDiameter"] == 0.3


def test_min_conduit_diameter_max_filling():
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.33], "MaxQ": [0.1], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureSteps(df, None, 1.0)

    with patch("stormlab.features.conduits.common_diameters", [0.3, 0.4, 0.5, 0.6]):
        with patch("stormlab.features.conduits.max_filling", return_value=0.827):
            with patch.object(formulas, "calc_filling", return_value=0.33):
                with patch("stormlab.features.conduits.validate_filling", return_value=True):
                    service.min_conduit_diameter()
                    assert service.dfc.loc[0, "MinDiameter"] == 0.3


def test_min_conduit_diameter_exceeds_max_filling():
    """Test that when validate_filling fails for current diameter, function finds larger diameter."""
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.34], "MaxQ": [0.1], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureSteps(df, None, 1.0)

    with patch("stormlab.features.conduits.common_diameters", [0.3, 0.4, 0.5, 0.6]):
        with patch("stormlab.features.conduits.max_filling", return_value=0.827):
            with patch("stormlab.features.conduits.validate_filling") as mock_validate:
                # validate_filling returns True only for d > 0.4, so 0.4 fails but 0.5 passes
                mock_validate.side_effect = lambda f, d, ratio=None: d > 0.4
                service.min_conduit_diameter()
                # MinDiameter should be 0.5 (smallest diameter that passes validation)
                assert service.dfc.loc[0, "MinDiameter"] == 0.5


def test_min_conduit_diameter_zero_filling():
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.0], "MaxQ": [0.0], "SlopeFtPerFt": [0.02], "ValMaxFill": [1]})
    service = ConduitFeatureSteps(df, None, 1.0)

    with patch("stormlab.features.conduits.common_diameters", [0.3, 0.4, 0.5, 0.6]):
        service.min_conduit_diameter()
        assert service.dfc.loc[0, "MinDiameter"] == 0.3


def test_min_conduit_diameter_slope_too_small():
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.2], "MaxQ": [0.1], "SlopeFtPerFt": [0.001], "ValMaxFill": [1]})
    service = ConduitFeatureSteps(df, None, 1.0)

    with patch("stormlab.features.conduits.common_diameters", [0.3, 0.4, 0.5, 0.6]):
        with patch("stormlab.features.conduits.validate_filling", return_value=True):
            with patch.object(formulas, "calc_filling") as mock_calc:
                mock_calc.side_effect = ValueError("Slope is too small")
                service.min_conduit_diameter()
                assert service.dfc.loc[0, "MinDiameter"] == 0.4


def test_min_conduit_diameter_forwards_configured_theta_and_g():
    df = pd.DataFrame({"Geom1": [0.4], "Filling": [0.2], "MaxQ": [0.1], "SlopeFtPerFt": [0.01], "ValMaxFill": [1]})
    service = ConduitFeatureSteps(
        df,
        None,
        1.0,
        hydraulic_validation_min_slope_rule="hydromechanical_canonical",
        theta=3.0,
        g=10.0,
    )

    with patch("stormlab.features.conduits.common_diameters", [0.3, 0.4]):
        with patch("stormlab.features.conduits.validate_filling", return_value=True):
            with patch.object(formulas, "calc_filling", return_value=0.2) as mock_calc:
                service.min_conduit_diameter()

    assert mock_calc.call_count > 0
    for call in mock_calc.call_args_list:
        assert call.kwargs["theta"] == 3.0
        assert call.kwargs["g"] == 10.0


def test_min_conduit_diameter_none_dataframe():
    service = ConduitFeatureSteps(None, None, 1.0)
    service.min_conduit_diameter()
    assert service.dfc is None


def test_min_conduit_diameter_empty_dataframe():
    df = pd.DataFrame(columns=["Geom1", "Filling", "MaxQ", "SlopeFtPerFt", "ValMaxFill"])
    service = ConduitFeatureSteps(df, None, 1.0)
    service.min_conduit_diameter()
    assert len(service.dfc) == 0
