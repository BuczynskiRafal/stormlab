"""Unit tests for subcatchment categories and their one-hot encoding.

Migrated from ``app/sa/core/tests/test_subcatchments.py``. The assertions carry
over; the fixtures do not, because the tags now arrive as a Series from
:func:`stormlab.io.inp.read_tags` instead of through a ``MagicMock`` model, so
there is nothing left to mock.
"""

import logging

import pandas as pd
import pytest

from stormlab.contracts.features import FEATURE_ONEHOT_ORDER
from stormlab.features.subcatchments import (
    SUBCATCHMENT_TAG_PREFIX,
    SubcatchmentFeatureError,
    assign_categories,
    one_hot_categories,
)


def tag_series(pairs: list[tuple[str, str, str]]) -> pd.Series:
    """Build a ``[TAGS]`` Series of ``(element_type, name, tag)`` triples."""
    index = pd.MultiIndex.from_arrays(
        [[element for element, _, _ in pairs], [name for _, name, _ in pairs]],
        names=["ElementType", "Name"],
    )
    return pd.Series([tag for _, _, tag in pairs], index=index, name="Tag")


class TestAssignCategories:
    def test_assigns_categories_from_tags_section(self):
        tags = tag_series([("Subcatch", "SC1", "rural"), ("Subcatch", "SC2", "forests")])
        dfs = pd.DataFrame({"Area": [100, 200]}, index=["SC1", "SC2"])

        result = assign_categories(dfs, tags)

        assert result.loc["SC1", "category"] == "rural"
        assert result.loc["SC2", "category"] == "forests"

    def test_does_not_mutate_the_input_frame(self):
        tags = tag_series([("Subcatch", "SC1", "rural")])
        dfs = pd.DataFrame({"Area": [100]}, index=["SC1"])

        assign_categories(dfs, tags)

        assert "category" not in dfs.columns

    def test_ignores_tags_of_other_element_types(self):
        tags = tag_series([("Link", "C1", "depth_increase"), ("Node", "J1", "x")])
        dfs = pd.DataFrame({"Area": [100]}, index=["SC1"])

        result = assign_categories(dfs, tags)

        assert pd.isna(result.loc["SC1", "category"])

    def test_no_tags_at_all_gives_nan_and_a_warning(self, caplog):
        """A synthetic untagged network: no tags is data, not an error.

        Not the Wroclaw model — that one tags all 37 of its subcatchments; see
        ``TestWroclawTagProfile`` below for what it actually carries.
        """
        dfs = pd.DataFrame({"Area": [100, 200]}, index=["SC1", "SC2"])
        empty = tag_series([])

        with caplog.at_level(logging.WARNING, logger="stormlab.features.subcatchments"):
            result = assign_categories(dfs, empty)

        assert "category" in result.columns
        assert result["category"].isna().all()
        assert "No subcatchment tags found" in caplog.text

    def test_partial_tags_leave_the_untagged_rows_nan(self, caplog):
        tags = tag_series([("Subcatch", "SC1", "rural")])
        dfs = pd.DataFrame({"Area": [100, 200]}, index=["SC1", "SC2"])

        with caplog.at_level(logging.WARNING, logger="stormlab.features.subcatchments"):
            result = assign_categories(dfs, tags)

        assert result.loc["SC1", "category"] == "rural"
        assert pd.isna(result.loc["SC2", "category"])
        assert "1 of 2 subcatchments have no tag" in caplog.text

    def test_logs_the_categorized_count(self, caplog):
        tags = tag_series([("Subcatch", f"SC{i}", "rural") for i in (1, 2, 3)])
        dfs = pd.DataFrame({"Area": [100, 200, 300]}, index=["SC1", "SC2", "SC3"])

        with caplog.at_level(logging.INFO, logger="stormlab.features.subcatchments"):
            assign_categories(dfs, tags)

        assert "Assigned categories to 3 subcatchments" in caplog.text

    def test_empty_frame_still_gets_the_column(self):
        result = assign_categories(pd.DataFrame(), tag_series([]))

        assert "category" in result.columns
        assert len(result) == 0

    def test_accepts_a_flat_indexed_tag_series(self):
        tags = pd.Series(["rural"], index=["SC1"], name="Tag")
        dfs = pd.DataFrame({"Area": [100]}, index=["SC1"])

        result = assign_categories(dfs, tags)

        assert result.loc["SC1", "category"] == "rural"


class TestOneHotCategories:
    def test_encodes_single_category_across_the_whole_order(self):
        dfs = pd.DataFrame({"category": ["rural"]}, index=["S1"])

        result = one_hot_categories(dfs, FEATURE_ONEHOT_ORDER)

        for category in FEATURE_ONEHOT_ORDER:
            assert category in result.columns
        assert result.loc["S1", "rural"] == 1
        assert result.loc["S1", "urban_highly_impervious"] == 0

    def test_encodes_multiple_categories(self):
        dfs = pd.DataFrame(
            {"category": ["rural", "forests", "urban_highly_impervious"]},
            index=["S1", "S2", "S3"],
        )

        result = one_hot_categories(dfs, FEATURE_ONEHOT_ORDER)

        assert result.loc["S1", "rural"] == 1
        assert result.loc["S1", "forests"] == 0
        assert result.loc["S2", "forests"] == 1
        assert result.loc["S2", "rural"] == 0
        assert result.loc["S3", "urban_highly_impervious"] == 1

    def test_preserves_original_columns(self):
        dfs = pd.DataFrame({"category": ["rural"], "Area": [100.0], "PercImperv": [50.0]}, index=["S1"])

        result = one_hot_categories(dfs, FEATURE_ONEHOT_ORDER)

        assert "category" in result.columns
        assert "Area" in result.columns
        assert result.loc["S1", "Area"] == 100.0

    def test_all_indicator_columns_are_integers(self):
        dfs = pd.DataFrame({"category": ["rural", "arable"]}, index=["S1", "S2"])

        result = one_hot_categories(dfs, FEATURE_ONEHOT_ORDER)

        for category in FEATURE_ONEHOT_ORDER:
            assert result[category].dtype.kind == "i"

    def test_emits_the_block_in_the_requested_order(self):
        dfs = pd.DataFrame({"category": ["rural"]}, index=["S1"])

        result = one_hot_categories(dfs, FEATURE_ONEHOT_ORDER)

        assert list(result.columns[-len(FEATURE_ONEHOT_ORDER) :]) == list(FEATURE_ONEHOT_ORDER)

    def test_custom_category_column_name(self):
        dfs = pd.DataFrame({"land_use": ["forests"]}, index=["S1"])

        result = one_hot_categories(dfs, FEATURE_ONEHOT_ORDER, category_column="land_use")

        assert result.loc["S1", "forests"] == 1
        assert result.loc["S1", "rural"] == 0

    def test_all_nan_categories_encode_to_zeros(self):
        dfs = pd.DataFrame({"category": [None, None]}, index=["S1", "S2"])

        result = one_hot_categories(dfs, FEATURE_ONEHOT_ORDER)

        assert result[list(FEATURE_ONEHOT_ORDER)].to_numpy().sum() == 0

    @pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, pd.NaT, "-"], ids=repr)
    def test_every_spelling_of_a_missing_category_encodes_to_zeros(self, missing):
        """``assign_categories`` writes ``pd.NA``, whose ``== '-'`` is itself ``pd.NA``."""
        dfs = pd.DataFrame({"category": pd.Series([missing], index=["S1"], dtype="object")})

        result = one_hot_categories(dfs, FEATURE_ONEHOT_ORDER)

        assert result[list(FEATURE_ONEHOT_ORDER)].to_numpy().sum() == 0

    def test_the_untagged_output_of_assign_categories_encodes_to_zeros(self):
        """The two functions have to compose: this is the frozen fixture's own path."""
        dfs = assign_categories(pd.DataFrame({"Area": [100, 200]}, index=["SC1", "SC2"]), tag_series([]))

        result = one_hot_categories(dfs, FEATURE_ONEHOT_ORDER)

        assert result[list(FEATURE_ONEHOT_ORDER)].to_numpy().sum() == 0

    def test_raises_when_the_category_column_is_missing(self):
        dfs = pd.DataFrame({"other_col": [1, 2]}, index=["S1", "S2"])

        with pytest.raises(SubcatchmentFeatureError, match="Column 'category' not found"):
            one_hot_categories(dfs, FEATURE_ONEHOT_ORDER)

    def test_raises_for_a_custom_missing_column_name(self):
        dfs = pd.DataFrame({"category": ["rural"]}, index=["S1"])

        with pytest.raises(SubcatchmentFeatureError, match="Column 'custom_cat' not found"):
            one_hot_categories(dfs, FEATURE_ONEHOT_ORDER, category_column="custom_cat")

    def test_raises_on_a_category_outside_the_requested_order(self):
        """A class the order does not cover cannot be encoded; zeros would hide it."""
        dfs = pd.DataFrame({"category": ["swamp_of_sadness"]}, index=["S1"])

        with pytest.raises(SubcatchmentFeatureError, match="swamp_of_sadness"):
            one_hot_categories(dfs, FEATURE_ONEHOT_ORDER)

    def test_empty_frame_is_returned_untouched(self):
        dfs = pd.DataFrame()

        assert len(one_hot_categories(dfs, FEATURE_ONEHOT_ORDER)) == 0


class TestWroclawTagProfile:
    """What the real Wroclaw INP carries, read from the file rather than assumed.

    Read-only: ``swmmio`` parses the INP text, so nothing runs SWMM and nothing is
    written next to the tracked network.
    """

    @pytest.fixture(scope="class")
    def wroclaw_tags(self) -> pd.Series:
        from stormlab.config import repo_root
        from stormlab.io.inp import load_model, read_tags

        inp = repo_root() / "models" / "recomendations" / "gnn" / "wroclaw" / "wroclaw_pipes.inp"
        if not inp.is_file():
            pytest.skip(f"the Wroclaw network is not available at {inp}")
        return read_tags(load_model(inp, include_rpt=False))

    def test_every_subcatchment_is_tagged(self, wroclaw_tags):
        element_types = wroclaw_tags.index.get_level_values(0).astype(str)
        subcatchment_tags = wroclaw_tags[element_types.str.startswith(SUBCATCHMENT_TAG_PREFIX)]

        assert len(subcatchment_tags) == 37

    def test_the_tags_are_all_contract_categories(self, wroclaw_tags):
        element_types = wroclaw_tags.index.get_level_values(0).astype(str)
        subcatchment_tags = wroclaw_tags[element_types.str.startswith(SUBCATCHMENT_TAG_PREFIX)]

        assert set(subcatchment_tags) <= set(FEATURE_ONEHOT_ORDER)

    def test_no_conduit_carries_a_tag(self, wroclaw_tags):
        """It is the *link* tags Wroclaw lacks, which is why its one-hot block is not empty."""
        element_types = wroclaw_tags.index.get_level_values(0).astype(str)

        assert not element_types.str.startswith("Link").any()

    def test_the_encoded_categories_are_not_all_zero(self, wroclaw_tags):
        from stormlab.config import repo_root
        from stormlab.io.inp import load_model, read_frames

        inp = repo_root() / "models" / "recomendations" / "gnn" / "wroclaw" / "wroclaw_pipes.inp"
        _, _, dfs = read_frames(load_model(inp, include_rpt=False), include_rpt=False)
        encoded = one_hot_categories(assign_categories(dfs, wroclaw_tags), FEATURE_ONEHOT_ORDER)

        assert encoded[list(FEATURE_ONEHOT_ORDER)].to_numpy().sum() == 37
