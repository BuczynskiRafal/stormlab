"""Unit tests for the node-to-subcatchment mapping.

Migrated from ``app/sa/core/tests/test_nodes.py``. The assertions are unchanged;
the ``None``-input tests are gone because the function now takes frames rather
than optional attributes, and the "returns a copy" behaviour is new and tested.
"""

import pandas as pd
import pytest

from stormlab.features.nodes import NO_SUBCATCHMENT, assign_subcatchment_info


class TestAssignSubcatchmentInfo:
    def test_initializes_columns_with_defaults(self):
        dfn = pd.DataFrame({"col": [1, 2]}, index=["N1", "N2"])
        dfs = pd.DataFrame({"col": [1]}, index=["S1"])  # No Outlet column

        result = assign_subcatchment_info(dfn, dfs)

        assert "Subcatchment" in result.columns
        assert "SbcCategory" in result.columns
        assert list(result["Subcatchment"]) == [NO_SUBCATCHMENT, NO_SUBCATCHMENT]
        assert list(result["SbcCategory"]) == [NO_SUBCATCHMENT, NO_SUBCATCHMENT]

    def test_does_not_mutate_the_input_frame(self):
        dfn = pd.DataFrame({"col": [1]}, index=["N1"])
        dfs = pd.DataFrame({"Name": ["SC1"], "Outlet": ["N1"]}, index=["S1"])

        assign_subcatchment_info(dfn, dfs)

        assert list(dfn.columns) == ["col"]

    def test_maps_subcatchment_name_from_outlet(self):
        dfn = pd.DataFrame({"col": [1, 2, 3]}, index=["N1", "N2", "N3"])
        dfs = pd.DataFrame({"Name": ["SC1", "SC2"], "Outlet": ["N1", "N3"]}, index=["S1", "S2"])

        result = assign_subcatchment_info(dfn, dfs)

        assert result.loc["N1", "Subcatchment"] == "SC1"
        assert result.loc["N2", "Subcatchment"] == NO_SUBCATCHMENT
        assert result.loc["N3", "Subcatchment"] == "SC2"

    def test_maps_subcatchment_category_when_column_exists(self):
        dfn = pd.DataFrame({"col": [1, 2]}, index=["N1", "N2"])
        dfs = pd.DataFrame({"Name": ["SC1"], "Outlet": ["N1"], "category": ["rural"]}, index=["S1"])

        result = assign_subcatchment_info(dfn, dfs)

        assert result.loc["N1", "Subcatchment"] == "SC1"
        assert result.loc["N1", "SbcCategory"] == "rural"
        assert result.loc["N2", "Subcatchment"] == NO_SUBCATCHMENT
        assert result.loc["N2", "SbcCategory"] == NO_SUBCATCHMENT

    def test_no_category_mapping_when_column_missing(self):
        dfn = pd.DataFrame({"col": [1]}, index=["N1"])
        dfs = pd.DataFrame({"Name": ["SC1"], "Outlet": ["N1"]}, index=["S1"])

        result = assign_subcatchment_info(dfn, dfs)

        assert result.loc["N1", "Subcatchment"] == "SC1"
        assert result.loc["N1", "SbcCategory"] == NO_SUBCATCHMENT

    def test_untagged_subcatchment_writes_nan_as_the_legacy_service_did(self):
        """A subcatchment whose land cover is unknown puts NaN into SbcCategory.

        ``app/sa/core/nodes.py`` assigns ``category_mapping[node_id]`` whenever the
        node is in the mapping, without inspecting the value, so an untagged
        subcatchment leaves ``NaN`` there. Substituting the sentinel would be a
        semantic change: both encode to an all-zero one-hot block, but only one of
        them is what the frozen golden was built from.
        """
        dfn = pd.DataFrame({"col": [1]}, index=["N1"])
        dfs = pd.DataFrame({"Name": ["SC1"], "Outlet": ["N1"], "category": [None]}, index=["S1"])

        result = assign_subcatchment_info(dfn, dfs)

        assert result.loc["N1", "Subcatchment"] == "SC1"
        assert pd.isna(result.loc["N1", "SbcCategory"])

    def test_multiple_subcatchments_mapped_correctly(self):
        dfn = pd.DataFrame({"col": [1, 2, 3, 4]}, index=["N1", "N2", "N3", "N4"])
        dfs = pd.DataFrame(
            {
                "Name": ["SC1", "SC2", "SC3"],
                "Outlet": ["N1", "N2", "N4"],
                "category": ["rural", "urban_highly_impervious", "forests"],
            },
            index=["S1", "S2", "S3"],
        )

        result = assign_subcatchment_info(dfn, dfs)

        assert result.loc["N1", "Subcatchment"] == "SC1"
        assert result.loc["N1", "SbcCategory"] == "rural"
        assert result.loc["N2", "Subcatchment"] == "SC2"
        assert result.loc["N2", "SbcCategory"] == "urban_highly_impervious"
        assert result.loc["N3", "Subcatchment"] == NO_SUBCATCHMENT
        assert result.loc["N3", "SbcCategory"] == NO_SUBCATCHMENT
        assert result.loc["N4", "Subcatchment"] == "SC3"
        assert result.loc["N4", "SbcCategory"] == "forests"

    def test_outlet_not_in_nodes_ignored(self):
        dfn = pd.DataFrame({"col": [1]}, index=["N1"])
        dfs = pd.DataFrame({"Name": ["SC1", "SC2"], "Outlet": ["N1", "N_NONEXISTENT"]}, index=["S1", "S2"])

        result = assign_subcatchment_info(dfn, dfs)

        assert result.loc["N1", "Subcatchment"] == "SC1"

    def test_empty_node_frame_still_gets_the_columns(self):
        dfn = pd.DataFrame({"col": []})
        dfs = pd.DataFrame({"Name": ["SC1"], "Outlet": ["N1"]}, index=["S1"])

        result = assign_subcatchment_info(dfn, dfs)

        assert "Subcatchment" in result.columns
        assert "SbcCategory" in result.columns
        assert len(result) == 0

    def test_missing_outlet_column_is_not_an_error(self):
        dfn = pd.DataFrame({"col": [1]}, index=["N1"])
        dfs = pd.DataFrame({"Name": ["SC1"]}, index=["S1"])

        result = assign_subcatchment_info(dfn, dfs)

        assert result.loc["N1", "Subcatchment"] == NO_SUBCATCHMENT

    def test_missing_name_column_is_reported(self):
        dfn = pd.DataFrame({"col": [1]}, index=["N1"])
        dfs = pd.DataFrame({"Outlet": ["N1"]}, index=["S1"])

        with pytest.raises(KeyError):
            assign_subcatchment_info(dfn, dfs)
