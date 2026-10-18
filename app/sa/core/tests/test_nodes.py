"""Unit tests for NodeFeatureEngineeringService."""

import pandas as pd

from sa.core.nodes import NodeFeatureEngineeringService


class TestNodeFeatureEngineeringService:
    """Tests for NodeFeatureEngineeringService class."""

    def test_init(self):
        """Test initialization stores dataframes correctly."""
        dfn = pd.DataFrame({"col": [1]}, index=["N1"])
        dfs = pd.DataFrame({"col": [2]}, index=["S1"])

        service = NodeFeatureEngineeringService(dfn, dfs)

        assert service.dfn is dfn
        assert service.dfs is dfs

    def test_nodes_subcatchment_name_calls_nodes_subcatchment_info(self):
        """Test that nodes_subcatchment_name() calls nodes_subcatchment_info() for backward compatibility."""
        dfn = pd.DataFrame({"col": [1]}, index=["N1"])
        dfs = pd.DataFrame({"Name": ["SC1"], "Outlet": ["N1"]}, index=["S1"])

        service = NodeFeatureEngineeringService(dfn, dfs)
        service.nodes_subcatchment_name()

        assert "Subcatchment" in service.dfn.columns
        assert "SbcCategory" in service.dfn.columns


class TestNodesSubcatchmentInfo:
    """Tests for nodes_subcatchment_info method."""

    def test_returns_early_when_dfs_is_none(self):
        """Test method returns early when dfs is None."""
        dfn = pd.DataFrame({"col": [1]}, index=["N1"])

        service = NodeFeatureEngineeringService(dfn, None)
        service.nodes_subcatchment_info()

        assert "Subcatchment" not in service.dfn.columns

    def test_returns_early_when_dfn_is_none(self):
        """Test method returns early when dfn is None."""
        dfs = pd.DataFrame({"col": [1]}, index=["S1"])

        service = NodeFeatureEngineeringService(None, dfs)
        service.nodes_subcatchment_info()

    def test_returns_early_when_both_none(self):
        """Test method returns early when both dataframes are None."""
        service = NodeFeatureEngineeringService(None, None)
        service.nodes_subcatchment_info()

    def test_initializes_columns_with_defaults(self):
        """Test that Subcatchment and SbcCategory columns are initialized with '-'."""
        dfn = pd.DataFrame({"col": [1, 2]}, index=["N1", "N2"])
        dfs = pd.DataFrame({"col": [1]}, index=["S1"])  # No Outlet column

        service = NodeFeatureEngineeringService(dfn, dfs)
        service.nodes_subcatchment_info()

        assert "Subcatchment" in service.dfn.columns
        assert "SbcCategory" in service.dfn.columns
        assert list(service.dfn["Subcatchment"]) == ["-", "-"]
        assert list(service.dfn["SbcCategory"]) == ["-", "-"]

    def test_maps_subcatchment_name_from_outlet(self):
        """Test that subcatchment name is mapped to nodes based on Outlet column."""
        dfn = pd.DataFrame({"col": [1, 2, 3]}, index=["N1", "N2", "N3"])
        dfs = pd.DataFrame(
            {"Name": ["SC1", "SC2"], "Outlet": ["N1", "N3"]},
            index=["S1", "S2"],
        )

        service = NodeFeatureEngineeringService(dfn, dfs)
        service.nodes_subcatchment_info()

        assert service.dfn.loc["N1", "Subcatchment"] == "SC1"
        assert service.dfn.loc["N2", "Subcatchment"] == "-"
        assert service.dfn.loc["N3", "Subcatchment"] == "SC2"

    def test_maps_subcatchment_category_when_column_exists(self):
        """Test that subcatchment category is mapped when category column exists."""
        dfn = pd.DataFrame({"col": [1, 2]}, index=["N1", "N2"])
        dfs = pd.DataFrame(
            {"Name": ["SC1"], "Outlet": ["N1"], "category": ["rural"]},
            index=["S1"],
        )

        service = NodeFeatureEngineeringService(dfn, dfs)
        service.nodes_subcatchment_info()

        assert service.dfn.loc["N1", "Subcatchment"] == "SC1"
        assert service.dfn.loc["N1", "SbcCategory"] == "rural"
        assert service.dfn.loc["N2", "Subcatchment"] == "-"
        assert service.dfn.loc["N2", "SbcCategory"] == "-"

    def test_no_category_mapping_when_column_missing(self):
        """Test that SbcCategory stays '-' when category column is missing."""
        dfn = pd.DataFrame({"col": [1]}, index=["N1"])
        dfs = pd.DataFrame(
            {"Name": ["SC1"], "Outlet": ["N1"]},
            index=["S1"],
        )

        service = NodeFeatureEngineeringService(dfn, dfs)
        service.nodes_subcatchment_info()

        assert service.dfn.loc["N1", "Subcatchment"] == "SC1"
        assert service.dfn.loc["N1", "SbcCategory"] == "-"

    def test_multiple_subcatchments_mapped_correctly(self):
        """Test mapping with multiple subcatchments to different nodes."""
        dfn = pd.DataFrame({"col": [1, 2, 3, 4]}, index=["N1", "N2", "N3", "N4"])
        dfs = pd.DataFrame(
            {
                "Name": ["SC1", "SC2", "SC3"],
                "Outlet": ["N1", "N2", "N4"],
                "category": ["rural", "urban_highly_impervious", "forests"],
            },
            index=["S1", "S2", "S3"],
        )

        service = NodeFeatureEngineeringService(dfn, dfs)
        service.nodes_subcatchment_info()

        assert service.dfn.loc["N1", "Subcatchment"] == "SC1"
        assert service.dfn.loc["N1", "SbcCategory"] == "rural"
        assert service.dfn.loc["N2", "Subcatchment"] == "SC2"
        assert service.dfn.loc["N2", "SbcCategory"] == "urban_highly_impervious"
        assert service.dfn.loc["N3", "Subcatchment"] == "-"
        assert service.dfn.loc["N3", "SbcCategory"] == "-"
        assert service.dfn.loc["N4", "Subcatchment"] == "SC3"
        assert service.dfn.loc["N4", "SbcCategory"] == "forests"

    def test_outlet_not_in_nodes_ignored(self):
        """Test that outlets pointing to non-existent nodes are ignored."""
        dfn = pd.DataFrame({"col": [1]}, index=["N1"])
        dfs = pd.DataFrame(
            {"Name": ["SC1", "SC2"], "Outlet": ["N1", "N_NONEXISTENT"]},
            index=["S1", "S2"],
        )

        service = NodeFeatureEngineeringService(dfn, dfs)
        service.nodes_subcatchment_info()

        assert service.dfn.loc["N1", "Subcatchment"] == "SC1"
