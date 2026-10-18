"""Edge case tests for data.py to improve coverage."""

import pytest
import pandas as pd
import math

from sa.core.data import ConduitFeatureEngineeringService
from sa.core.round import calc_area, calc_u


class TestHydraulicCalculationsFullPipe:
    """Tests for edge cases when pipe is completely full (filling == diameter)."""

    def test_calc_area_full_pipe(self):
        diameter = 1.0
        result = calc_area(diameter, diameter)
        expected = math.pi * (diameter / 2) ** 2
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_calc_u_full_pipe(self):
        diameter = 1.0
        result = calc_u(diameter, diameter)
        expected = math.pi * diameter
        assert math.isclose(result, expected, rel_tol=1e-9)


class TestConduitFeatureEngineeringServiceNoneDataframe:
    """Tests for ConduitFeatureEngineeringService with None dataframes."""

    def test_normalize_roughness_none_df(self):
        service = ConduitFeatureEngineeringService(None, None, 1.0)
        service.normalize_roughness()
        assert service.dfc is None

    def test_normalize_max_velocity_none_df(self):
        service = ConduitFeatureEngineeringService(None, None, 1.0)
        service.normalize_max_velocity()
        assert service.dfc is None

    def test_normalize_depth_none_df(self):
        service = ConduitFeatureEngineeringService(None, None, 1.0)
        service.normalize_depth()
        assert service.dfc is None

    def test_normalize_filling_none_df(self):
        service = ConduitFeatureEngineeringService(None, None, 1.0)
        service.normalize_filling()
        assert service.dfc is None


class TestConduitFeatureEngineeringMissingColumns:
    """Tests for handling of missing required columns."""

    def test_ground_cover_missing_columns(self):
        df = pd.DataFrame({"Geom1": [0.5]})
        service = ConduitFeatureEngineeringService(df, None, 1.0)
        with pytest.raises(ValueError, match="Missing required columns"):
            service.ground_cover()


class TestConduitFeatureEngineeringMinGroundCover:
    """Tests for min_ground_cover_is_valid edge cases."""

    def test_min_ground_cover_none_df(self):
        service = ConduitFeatureEngineeringService(None, None, 1.0)
        service.min_ground_cover_is_valid()  # Should not raise
        assert service.dfc is None


class TestConduitSubcatchmentInfo:
    """Tests for conduits_subcatchment_info edge cases."""

    def test_conduits_subcatchment_info_none_dfc(self):
        service = ConduitFeatureEngineeringService(None, None, 1.0)
        service.conduits_subcatchment_info()
        assert service.dfc is None

    def test_conduits_subcatchment_info_none_dfn(self):
        df = pd.DataFrame({"Name": ["C1"], "InletNode": ["N1"]})
        service = ConduitFeatureEngineeringService(df, None, 1.0)
        service.conduits_subcatchment_info()  # Should not raise

    def test_propagate_subcatchment_info_none_dfc(self):
        service = ConduitFeatureEngineeringService(None, None, 1.0)
        service.propagate_subcatchment_info()
        assert service.dfc is None


class TestEncodeSbcCategory:
    """Tests for encode_sbc_category edge cases."""

    def test_encode_sbc_category_none_df(self):
        service = ConduitFeatureEngineeringService(None, None, 1.0)
        service.encode_sbc_category()
        assert service.dfc is None

    def test_encode_sbc_category_empty_df(self):
        df = pd.DataFrame()
        service = ConduitFeatureEngineeringService(df, None, 1.0)
        service.encode_sbc_category()  # Should not raise

    def test_encode_sbc_category_missing_column(self):
        df = pd.DataFrame({"Name": ["C1"]})
        service = ConduitFeatureEngineeringService(df, None, 1.0)
        service.encode_sbc_category()  # Should not raise - just returns early


class TestNormalizeMaxQ:
    """Additional tests for normalize_max_q edge cases."""

    def test_normalize_max_q_zero_diameter_handling(self):
        df = pd.DataFrame(
            {
                "Geom1": [0.0],
                "SlopePerMile": [10.0],
                "MaxQ": [0.5],
            }
        )
        service = ConduitFeatureEngineeringService(df, None, 1.0)
        service.normalize_max_q()
        result = service.dfc.loc[0, "NMaxQ"]
        assert pd.isna(result) or result in (0.0, 1.0)

    def test_normalize_max_q_negative_max_q_clipped_to_zero(self):
        df = pd.DataFrame(
            {
                "Geom1": [0.5],
                "SlopePerMile": [10.0],
                "MaxQ": [-0.5],
            }
        )
        service = ConduitFeatureEngineeringService(df, None, 1.0)
        service.normalize_max_q()
        assert service.dfc.loc[0, "NMaxQ"] == 0.0
