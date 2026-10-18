"""Unit tests for DataManager class in data.py to improve coverage."""

import importlib.util
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sa.core.data import DataManager
from sa.core.enums import RecommendationCategory
from sa.core.tests import TEST_FILE

TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
requires_tensorflow = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")


@pytest.fixture(scope="function")
def data_manager():
    """Fixture to initialize the DataManager."""
    return DataManager(TEST_FILE)


class TestGetDfSafe:
    """Tests for the _get_df_safe helper method."""

    def test_get_df_safe_with_none_raises_value_error(self, data_manager: DataManager):
        """Test that _get_df_safe raises ValueError when source is None."""
        with pytest.raises(ValueError, match="DataFrame source is None"):
            data_manager._get_df_safe(None)

    def test_get_df_safe_with_dataframe(self, data_manager: DataManager):
        """Test that _get_df_safe returns a copy of the DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        result = data_manager._get_df_safe(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result["A"]) == [1, 2, 3]
        # Verify it's a copy
        result["A"] = [4, 5, 6]
        assert list(df["A"]) == [1, 2, 3]

    def test_get_df_safe_with_object_having_dataframe_attr(self, data_manager: DataManager):
        """Test that _get_df_safe handles objects with dataframe attribute."""
        mock_obj = MagicMock()
        mock_obj.dataframe = pd.DataFrame({"B": [10, 20]})
        result = data_manager._get_df_safe(mock_obj)
        assert isinstance(result, pd.DataFrame)
        assert list(result["B"]) == [10, 20]


class TestRoundFloatColumns:
    """Tests for the _round_float_columns method."""

    def test_round_float_columns_rounds_to_2_decimals(self, data_manager: DataManager):
        """Test that float columns are rounded to 2 decimal places."""
        data_manager.dfc = pd.DataFrame({"FloatCol": [1.23456, 2.98765]})
        data_manager.dfn = None
        data_manager.dfs = None
        data_manager._round_float_columns()
        assert list(data_manager.dfc["FloatCol"]) == [1.23, 2.99]

    def test_round_float_columns_excludes_confidence_columns(self, data_manager: DataManager):
        """Test that confidence columns are excluded from rounding."""
        confidence_cols = {f"confidence_{cat.value}": [0.123456789] for cat in RecommendationCategory}
        data_manager.dfc = pd.DataFrame(confidence_cols)
        original_values = {col: data_manager.dfc[col].iloc[0] for col in confidence_cols}
        data_manager.dfn = None
        data_manager.dfs = None
        data_manager._round_float_columns()
        # Confidence columns should NOT be rounded
        for col in confidence_cols:
            assert data_manager.dfc[col].iloc[0] == original_values[col]

    def test_round_float_columns_preserves_node_report_precision(self, data_manager: DataManager):
        data_manager.dfc = None
        data_manager.dfs = None
        data_manager.dfn = pd.DataFrame(
            {
                "MaxNodeDepth": [1.234],
                "HoursFlooded": [0.016],
                "TotalFloodVol": [0.003],
                "Other": [2.345],
            }
        )

        data_manager._round_float_columns()

        assert data_manager.dfn.loc[0, "MaxNodeDepth"] == 1.234
        assert data_manager.dfn.loc[0, "HoursFlooded"] == 0.016
        assert data_manager.dfn.loc[0, "TotalFloodVol"] == 0.003
        assert data_manager.dfn.loc[0, "Other"] == 2.35

    def test_round_float_columns_handles_none_dataframes(self, data_manager: DataManager):
        """Test that _round_float_columns handles None dataframes gracefully."""
        data_manager.dfc = None
        data_manager.dfn = None
        data_manager.dfs = None
        # Should not raise
        data_manager._round_float_columns()

    def test_round_float_columns_all_dataframes(self, data_manager: DataManager):
        """Test rounding works for all three dataframes."""
        data_manager.dfc = pd.DataFrame({"Val1": [1.12345]})
        data_manager.dfn = pd.DataFrame({"Val2": [2.98765]})
        data_manager.dfs = pd.DataFrame({"Val3": [3.45678]})
        data_manager._round_float_columns()
        assert data_manager.dfc["Val1"].iloc[0] == 1.12
        assert data_manager.dfn["Val2"].iloc[0] == 2.99
        assert data_manager.dfs["Val3"].iloc[0] == 3.46


class TestDropUnused:
    """Tests for the _drop_unused method."""

    def test_drop_unused_removes_conduit_columns(self, data_manager: DataManager):
        """Test that unused conduit columns are removed."""
        unused_cols = ["OutOffset", "InitFlow", "Barrels", "Shape", "InOffset", "coords", "Geom2", "SlopeFtPerFt", "Type"]
        data_manager.dfc = pd.DataFrame({col: [1] for col in unused_cols})
        data_manager.dfc["KeepMe"] = [2]
        data_manager.dfn = None
        data_manager.dfs = None
        data_manager._drop_unused()
        for col in unused_cols:
            assert col not in data_manager.dfc.columns
        assert "KeepMe" in data_manager.dfc.columns

    def test_drop_unused_removes_node_columns(self, data_manager: DataManager):
        """Test that unused node columns are removed."""
        unused_cols = ["coords", "StageOrTimeseries"]
        data_manager.dfn = pd.DataFrame({col: [1] for col in unused_cols})
        data_manager.dfn["KeepMe"] = [2]
        data_manager.dfc = None
        data_manager.dfs = None
        data_manager._drop_unused()
        for col in unused_cols:
            assert col not in data_manager.dfn.columns
        assert "KeepMe" in data_manager.dfn.columns

    def test_drop_unused_removes_subcatchment_columns(self, data_manager: DataManager):
        """Test that unused subcatchment columns are removed."""
        data_manager.dfs = pd.DataFrame({"coords": [1], "KeepMe": [2]})
        data_manager.dfc = None
        data_manager.dfn = None
        data_manager._drop_unused()
        assert "coords" not in data_manager.dfs.columns
        assert "KeepMe" in data_manager.dfs.columns

    def test_drop_unused_handles_none_dataframes(self, data_manager: DataManager):
        """Test that _drop_unused handles None dataframes gracefully."""
        data_manager.dfc = None
        data_manager.dfn = None
        data_manager.dfs = None
        # Should not raise
        data_manager._drop_unused()

    def test_drop_unused_ignores_missing_columns(self, data_manager: DataManager):
        """Test that _drop_unused doesn't error on missing columns."""
        data_manager.dfc = pd.DataFrame({"SomeColumn": [1]})
        data_manager.dfn = pd.DataFrame({"OtherColumn": [2]})
        data_manager.dfs = pd.DataFrame({"AnotherColumn": [3]})
        # Should not raise even though unused columns don't exist
        data_manager._drop_unused()


class TestFrostZoneValidation:
    """Tests for frost_zone property validation."""

    def test_frost_zone_valid_boundary_low(self, data_manager: DataManager):
        """Test frost_zone accepts minimum valid value."""
        data_manager.frost_zone = 0.8
        assert data_manager.frost_zone == 0.8

    def test_frost_zone_valid_boundary_high(self, data_manager: DataManager):
        """Test frost_zone accepts maximum valid value."""
        data_manager.frost_zone = 1.6
        assert data_manager.frost_zone == 1.6

    def test_frost_zone_invalid_below_minimum(self, data_manager: DataManager):
        """Test frost_zone raises ValueError for values below 0.8."""
        with pytest.raises(ValueError, match="Frost zone must be between 0.8 and 1.6 meters"):
            data_manager.frost_zone = 0.7

    def test_frost_zone_invalid_above_maximum(self, data_manager: DataManager):
        """Test frost_zone raises ValueError for values above 1.6."""
        with pytest.raises(ValueError, match="Frost zone must be between 0.8 and 1.6 meters"):
            data_manager.frost_zone = 1.7


class TestContextManager:
    """Tests for __enter__ and __exit__ context manager methods."""

    @requires_tensorflow
    def test_enter_calls_all_processing_methods(self):
        """Test that __enter__ calls calculate, feature_engineering, recommendations, etc."""
        with (
            patch.object(DataManager, "calculate") as mock_calc,
            patch.object(DataManager, "feature_engineering") as mock_fe,
            patch.object(DataManager, "recommendations") as mock_rec,
            patch.object(DataManager, "_round_float_columns") as mock_round,
            patch.object(DataManager, "_drop_unused") as mock_drop,
        ):
            dm = DataManager(TEST_FILE)
            result = dm.__enter__()
            assert result is dm
            mock_calc.assert_called_once()
            mock_fe.assert_called_once()
            mock_rec.assert_called_once()
            mock_round.assert_called_once()
            mock_drop.assert_called_once()

    def test_exit_without_exception(self, data_manager: DataManager):
        """Test __exit__ returns False when no exception occurs."""
        result = data_manager.__exit__(None, None, None)
        assert result is False

    def test_exit_with_exception(self, data_manager: DataManager, caplog):
        """Test __exit__ logs the exception and returns False."""
        with caplog.at_level(logging.ERROR, logger="sa.core.data"):
            result = data_manager.__exit__(ValueError, ValueError("test error"), None)
        assert result is False
        assert "Exception occurred: test error" in caplog.text


class TestDelegationMethods:
    """Tests for methods that delegate to services."""

    def test_all_traces_delegates_to_service(self, data_manager: DataManager):
        """Test all_traces delegates to trace_analysis_service."""
        mock_result = {"trace1": ["C1", "C2"]}
        data_manager.trace_analysis_service = MagicMock()
        data_manager.trace_analysis_service.all_traces.return_value = mock_result
        result = data_manager.all_traces()
        assert result == mock_result
        data_manager.trace_analysis_service.all_traces.assert_called_once()

    def test_overflowing_pipes_delegates_to_service(self, data_manager: DataManager):
        """Test overflowing_pipes delegates to trace_analysis_service."""
        mock_result = pd.DataFrame({"Name": ["C1"]})
        data_manager.trace_analysis_service = MagicMock()
        data_manager.trace_analysis_service.overflowing_pipes.return_value = mock_result
        result = data_manager.overflowing_pipes()
        assert isinstance(result, pd.DataFrame)
        data_manager.trace_analysis_service.overflowing_pipes.assert_called_once()

    def test_overflowing_traces_delegates_to_service(self, data_manager: DataManager):
        """Test overflowing_traces delegates to trace_analysis_service."""
        mock_result = {"pipe1": {"upstream": ["C1"], "downstream": ["C2"]}}
        data_manager.trace_analysis_service = MagicMock()
        data_manager.trace_analysis_service.overflowing_traces.return_value = mock_result
        result = data_manager.overflowing_traces()
        assert result == mock_result
        data_manager.trace_analysis_service.overflowing_traces.assert_called_once()

    def test_place_to_change_delegates_to_service(self, data_manager: DataManager):
        """Test place_to_change delegates to trace_analysis_service."""
        mock_result = ["N1", "N2"]
        data_manager.trace_analysis_service = MagicMock()
        data_manager.trace_analysis_service.place_to_change.return_value = mock_result
        result = data_manager.place_to_change()
        assert result == mock_result
        data_manager.trace_analysis_service.place_to_change.assert_called_once()


class TestGNNModelAvailability:
    """Tests for GNN model availability logging."""

    def test_gnn_model_available_logs_info(self):
        """Test that logger.info is called when GNN model is available."""
        with (
            patch("sa.core.data.gnn_enabled", return_value=True),
            patch("sa.core.data.get_gnn_recommendation", return_value=MagicMock()),
            patch("sa.core.data.logger") as mock_logger,
        ):
            DataManager(TEST_FILE)
            # Check that the "GNN model is available" log was called
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("GNN model is available" in str(call) for call in info_calls)

    def test_gnn_model_not_available_logs_info(self):
        """Test that logger.info is called when GNN model is not available."""
        with (
            patch("sa.core.data.gnn_enabled", return_value=True),
            patch("sa.core.data.get_gnn_recommendation", return_value=None),
            patch("sa.core.data.logger") as mock_logger,
        ):
            DataManager(TEST_FILE)
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("GNN model is not available" in str(call) for call in info_calls)


class TestCalculateMethod:
    """Tests for the calculate method."""

    @requires_tensorflow
    def test_calculate_runs_simulation(self, data_manager: DataManager):
        """Test calculate method runs simulation service."""
        with (
            patch.object(data_manager.simulation_service, "run_simulation") as mock_sim,
            patch("sa.core.data.sw.Model.__init__"),
            patch.object(type(data_manager), "conduits", return_value=pd.DataFrame()),
            patch.object(type(data_manager), "nodes", return_value=pd.DataFrame()),
            patch.object(type(data_manager), "subcatchments", return_value=pd.DataFrame()),
            patch("sa.core.data.SubcatchmentFeatureEngineeringService"),
            patch("sa.core.data.NodeFeatureEngineeringService"),
            patch("sa.core.data.ConduitFeatureEngineeringService"),
            patch("sa.core.data.RecommendationService"),
        ):
            data_manager.calculate()
            mock_sim.assert_called_once()

    def test_calculate_handles_reinit_exception(self, data_manager: DataManager):
        """Test calculate handles exception during reinitialization gracefully."""
        with (
            patch.object(data_manager.simulation_service, "run_simulation"),
            patch("sa.core.data.sw.Model.__init__", side_effect=Exception("Reinit failed")),
            patch("sa.core.data.logger") as mock_logger,
        ):
            data_manager.calculate()
            # Verify warning was logged
            mock_logger.warning.assert_called()
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("Could not reinitialize" in str(call) for call in warning_calls)

    def test_calculate_uses_mlp_when_gnn_not_available(self, data_manager: DataManager):
        """Test calculate uses MLP model when GNN is not available."""
        mock_conduits_df = pd.DataFrame({"Name": ["C1"], "InletNode": ["N1"], "OutletNode": ["N2"]})
        mock_nodes_df = pd.DataFrame({"Name": ["N1"]})
        mock_subcatch_df = pd.DataFrame({"Name": ["S1"]})

        with (
            patch.object(data_manager.simulation_service, "run_simulation"),
            patch("sa.core.data.sw.Model.__init__"),
            patch.object(type(data_manager), "conduits", return_value=mock_conduits_df),
            patch.object(type(data_manager), "nodes", return_value=mock_nodes_df),
            patch.object(type(data_manager), "subcatchments", return_value=mock_subcatch_df),
            patch("sa.core.data.gnn_enabled", return_value=False),
            patch("sa.core.data.logger") as mock_logger,
            patch("sa.core.data.SubcatchmentFeatureEngineeringService"),
            patch("sa.core.data.NodeFeatureEngineeringService"),
            patch("sa.core.data.ConduitFeatureEngineeringService"),
            patch("sa.core.data.RecommendationService") as mock_rec_service,
        ):
            data_manager.calculate()
            # Verify MLP was chosen
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("MLP model" in str(call) for call in info_calls)
            # Verify RecommendationService was created with MLP
            mock_rec_service.assert_called()
            call_args = mock_rec_service.call_args
            assert call_args[1].get("model_name") == "MLP"

    def test_calculate_uses_gnn_when_available(self, data_manager: DataManager):
        """Test calculate uses GNN model when available."""
        mock_conduits_df = pd.DataFrame({"Name": ["C1"], "InletNode": ["N1"], "OutletNode": ["N2"]})
        mock_nodes_df = pd.DataFrame({"Name": ["N1"]})
        mock_subcatch_df = pd.DataFrame({"Name": ["S1"]})
        mock_gnn = MagicMock()

        with (
            patch.object(data_manager.simulation_service, "run_simulation"),
            patch("sa.core.data.sw.Model.__init__"),
            patch.object(type(data_manager), "conduits", return_value=mock_conduits_df),
            patch.object(type(data_manager), "nodes", return_value=mock_nodes_df),
            patch.object(type(data_manager), "subcatchments", return_value=mock_subcatch_df),
            patch("sa.core.data.gnn_enabled", return_value=True),
            patch("sa.core.data.get_gnn_recommendation", return_value=mock_gnn),
            patch("sa.core.data.logger") as mock_logger,
            patch("sa.core.data.SubcatchmentFeatureEngineeringService"),
            patch("sa.core.data.NodeFeatureEngineeringService"),
            patch("sa.core.data.ConduitFeatureEngineeringService"),
            patch("sa.core.data.RecommendationService") as mock_rec_service,
        ):
            data_manager.calculate()
            # Verify GNN was chosen
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("GNN model" in str(call) for call in info_calls)
            # Verify RecommendationService was created with GNN
            mock_rec_service.assert_called()
            call_args = mock_rec_service.call_args
            assert call_args[1].get("model_name") == "GNN"


class TestNodeReportResults:
    """Node result summaries are joined without running a simulation."""

    def test_joins_depth_and_flooding_summaries_by_node_name(self, data_manager: DataManager):
        data_manager._join_node_report_results()

        assert data_manager.dfn.loc["J2", "MaxNodeDepth"] == pytest.approx(2.71)
        assert data_manager.dfn.loc["J2", "HoursFlooded"] == pytest.approx(0.02)
        assert data_manager.dfn.loc["J2", "TotalFloodVol"] == pytest.approx(0.003)
        assert data_manager.dfn.loc["J3", "HoursFlooded"] == 0.0
        assert data_manager.dfn.loc["J3", "TotalFloodVol"] == 0.0

    def test_empty_flooding_summary_sets_zero_for_every_node(self, data_manager: DataManager, tmp_path: Path):
        source = Path(TEST_FILE).with_suffix(".rpt")
        report = source.read_text()
        section_start = report.index("  Node Flooding Summary")
        next_section = report.index("  Outfall Loading Summary")
        empty_section = """  Node Flooding Summary
  *********************

  No nodes were flooded.


  ***********************
"""
        report_path = tmp_path / "empty-flooding.rpt"
        report_path.write_text(report[:section_start] + empty_section + report[next_section:])

        data_manager.rpt.path = str(report_path)
        data_manager._join_node_report_results()

        assert data_manager.dfn["HoursFlooded"].eq(0.0).all()
        assert data_manager.dfn["TotalFloodVol"].eq(0.0).all()
        assert data_manager.dfn.loc["J2", "MaxNodeDepth"] == pytest.approx(2.71)

    def test_missing_report_leaves_result_columns_absent(self, data_manager: DataManager, tmp_path: Path):
        data_manager.dfn = data_manager.dfn.drop(
            columns=["MaxNodeDepth", "HoursFlooded", "TotalFloodVol"],
            errors="ignore",
        )
        data_manager.rpt.path = str(tmp_path / "missing.rpt")

        data_manager._join_node_report_results()

        assert "MaxNodeDepth" not in data_manager.dfn.columns
        assert "HoursFlooded" not in data_manager.dfn.columns
        assert "TotalFloodVol" not in data_manager.dfn.columns


class TestFeatureEngineeringMethod:
    """Tests for the feature_engineering method.

    Note: This is an orchestration method that delegates to services.
    We verify that services are invoked rather than checking every method,
    to reduce brittleness while maintaining coverage.
    """

    def test_feature_engineering_invokes_all_services(self, data_manager: DataManager):
        """Test feature_engineering invokes subcatchment, node, and conduit services."""
        data_manager.subcatchment_service = MagicMock()
        data_manager.node_service = MagicMock()
        data_manager.conduit_service = MagicMock()

        data_manager.feature_engineering()

        # Verify each service was used (at least one method called)
        assert data_manager.subcatchment_service.subcatchments_classify.called
        assert data_manager.node_service.nodes_subcatchment_info.called
        # Conduit service should have multiple calls for feature engineering
        assert data_manager.conduit_service.calculate_filling.called
        assert data_manager.conduit_service.encode_sbc_category.called

    def test_feature_engineering_subcatchments_classify_with_categories(self, data_manager: DataManager):
        """Test subcatchments_classify is called with categories=True."""
        data_manager.subcatchment_service = MagicMock()
        data_manager.node_service = MagicMock()
        data_manager.conduit_service = MagicMock()

        data_manager.feature_engineering()

        data_manager.subcatchment_service.subcatchments_classify.assert_called_once_with(categories=True)


class TestRecommendationsMethod:
    """Tests for the recommendations method."""

    def test_recommendations_delegates_to_service(self, data_manager: DataManager):
        """Test recommendations method delegates to recommendation_service."""
        data_manager.recommendation_service = MagicMock()
        data_manager.recommendations()
        data_manager.recommendation_service.recommendations.assert_called_once()

    def test_recommendations_populates_empty_columns_when_model_is_none(self, data_manager: DataManager):
        """Test recommendations populates '-' and 0.0 confidence columns when no ML model is loaded."""
        from sa.core.conduits import RecommendationService
        from sa.core.predictor import _LazyModelProxy

        data_manager.dfc = pd.DataFrame({"Name": ["C1", "C2"]})
        data_manager.recommendation_service = RecommendationService(
            data_manager.dfc,
            model=_LazyModelProxy(lambda: None),
            model_name="MLP",
        )
        data_manager.recommendations()

        assert list(data_manager.dfc["recommendation"]) == ["-", "-"]
        for category in RecommendationCategory:
            col = f"confidence_{category.value}"
            assert col in data_manager.dfc.columns
            assert (data_manager.dfc[col] == 0.0).all()
