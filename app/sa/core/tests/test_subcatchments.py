"""Unit tests for SubcatchmentFeatureEngineeringService."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from sa.core.constants import SUBCATCHMENT_CATEGORIES
from sa.core.subcatchments import SubcatchmentFeatureEngineeringService


class TestSubcatchmentFeatureEngineeringService:
    """Tests for SubcatchmentFeatureEngineeringService class."""

    def test_init(self):
        """Test initialization stores dataframe and model correctly."""
        dfs = pd.DataFrame({"col": [1]}, index=["S1"])
        model = MagicMock()

        service = SubcatchmentFeatureEngineeringService(dfs, model)

        assert service.dfs is dfs
        assert service.model is model


class TestEncodeCategoryColumn:
    """Tests for encode_category_column method."""

    def test_returns_none_when_dfs_is_none(self):
        """Test method returns None when dfs is None."""
        model = MagicMock()
        service = SubcatchmentFeatureEngineeringService(None, model)

        result = service.encode_category_column()

        assert result is None

    def test_returns_empty_df_when_dfs_is_empty(self):
        """Test method returns empty DataFrame when dfs is empty."""
        model = MagicMock()
        dfs = pd.DataFrame()
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        result = service.encode_category_column()

        assert result is not None
        assert len(result) == 0

    def test_raises_error_when_category_column_missing(self):
        """Test method raises ValueError when category column is not found."""
        model = MagicMock()
        dfs = pd.DataFrame({"other_col": [1, 2]}, index=["S1", "S2"])
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        with pytest.raises(ValueError, match="Column 'category' not found"):
            service.encode_category_column()

    def test_raises_error_for_custom_column_name(self):
        """Test method raises ValueError for custom missing column name."""
        model = MagicMock()
        dfs = pd.DataFrame({"category": ["rural"]}, index=["S1"])
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        with pytest.raises(ValueError, match="Column 'custom_cat' not found"):
            service.encode_category_column(category_column="custom_cat")

    def test_encodes_single_category(self):
        """Test one-hot encoding with a single category."""
        model = MagicMock()
        dfs = pd.DataFrame({"category": ["rural"]}, index=["S1"])
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        result = service.encode_category_column()

        # Check all SUBCATCHMENT_CATEGORIES are present
        for cat in SUBCATCHMENT_CATEGORIES:
            assert cat in result.columns

        # Check correct encoding
        assert result.loc["S1", "rural"] == 1
        assert result.loc["S1", "urban_highly_impervious"] == 0

    def test_encodes_multiple_categories(self):
        """Test one-hot encoding with multiple categories."""
        model = MagicMock()
        dfs = pd.DataFrame(
            {"category": ["rural", "forests", "urban_highly_impervious"]},
            index=["S1", "S2", "S3"],
        )
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        result = service.encode_category_column()

        # Check encodings
        assert result.loc["S1", "rural"] == 1
        assert result.loc["S1", "forests"] == 0
        assert result.loc["S2", "forests"] == 1
        assert result.loc["S2", "rural"] == 0
        assert result.loc["S3", "urban_highly_impervious"] == 1

    def test_preserves_original_columns(self):
        """Test that original columns are preserved after encoding."""
        model = MagicMock()
        dfs = pd.DataFrame(
            {"category": ["rural"], "Area": [100.0], "PercImperv": [50.0]},
            index=["S1"],
        )
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        result = service.encode_category_column()

        assert "category" in result.columns
        assert "Area" in result.columns
        assert "PercImperv" in result.columns
        assert result.loc["S1", "Area"] == 100.0

    def test_all_categories_have_int_type(self):
        """Test that all category columns are integer type."""
        model = MagicMock()
        dfs = pd.DataFrame({"category": ["rural", "arable"]}, index=["S1", "S2"])
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        result = service.encode_category_column()

        for cat in SUBCATCHMENT_CATEGORIES:
            assert result[cat].dtype in [int, "int64", "int32"]

    def test_custom_category_column_name(self):
        """Test encoding with custom category column name."""
        model = MagicMock()
        dfs = pd.DataFrame({"land_use": ["forests"]}, index=["S1"])
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        result = service.encode_category_column(category_column="land_use")

        assert result.loc["S1", "forests"] == 1
        assert result.loc["S1", "rural"] == 0


class TestSubcatchmentsClassify:
    """Tests for subcatchments_classify method."""

    def test_returns_early_when_dfs_is_none(self):
        """Test method returns early when dfs is None."""
        model = MagicMock()
        service = SubcatchmentFeatureEngineeringService(None, model)

        result = service.subcatchments_classify()

        assert result is None

    def test_returns_early_when_dfs_is_empty(self):
        """Test method returns early when dfs is empty."""
        model = MagicMock()
        dfs = pd.DataFrame()
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        result = service.subcatchments_classify()

        assert result is None

    def test_assigns_categories_from_tags_section(self):
        """Test that categories are assigned from model TAGS section."""
        model = MagicMock()

        # Create tags DataFrame with proper structure
        tags_df = pd.DataFrame(
            {"Name": ["SC1", "SC2"], "Tag": ["rural", "forests"]},
            index=["Subcatch SC1", "Subcatch SC2"],
        )
        model.inp.tags = tags_df

        dfs = pd.DataFrame({"Area": [100, 200]}, index=["SC1", "SC2"])
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        service.subcatchments_classify()

        assert "category" in service.dfs.columns
        assert service.dfs.loc["SC1", "category"] == "rural"
        assert service.dfs.loc["SC2", "category"] == "forests"

    def test_logs_warning_when_no_subcatchment_tags(self, caplog):
        """Test warning is logged when subcatch tags section is empty."""
        model = MagicMock()

        # Tags exist but no subcatchment tags (different prefix)
        tags_df = pd.DataFrame(
            {"Name": ["C1"], "Tag": ["pipe"]},
            index=["Conduit C1"],
        )
        model.inp.tags = tags_df

        dfs = pd.DataFrame({"Area": [100]}, index=["SC1"])
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        service.subcatchments_classify()

        assert "No subcatchment tags found" in caplog.text

    def test_logs_warning_when_no_tags_section(self, caplog):
        """Test warning is logged when no TAGS section exists."""
        model = MagicMock()
        model.inp.tags = None

        dfs = pd.DataFrame({"Area": [100]}, index=["SC1"])
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        service.subcatchments_classify()

        assert "No TAGS section found" in caplog.text

    def test_logs_warning_when_no_tags_attribute(self, caplog):
        """Test warning is logged when model.inp has no tags attribute."""
        model = MagicMock(spec=[])
        model.inp = MagicMock(spec=[])  # No tags attribute

        dfs = pd.DataFrame({"Area": [100]}, index=["SC1"])
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        service.subcatchments_classify()

        assert "No TAGS section found" in caplog.text

    def test_partial_subcatchment_tags(self):
        """Test that only subcatchments with tags get categories assigned."""
        model = MagicMock()

        # Only SC1 has a tag
        tags_df = pd.DataFrame(
            {"Name": ["SC1"], "Tag": ["rural"]},
            index=["Subcatch SC1"],
        )
        model.inp.tags = tags_df

        dfs = pd.DataFrame({"Area": [100, 200]}, index=["SC1", "SC2"])
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        service.subcatchments_classify()

        assert service.dfs.loc["SC1", "category"] == "rural"
        assert pd.isna(service.dfs.loc["SC2", "category"])

    def test_logs_categorized_count(self, caplog):
        """Test that the number of categorized subcatchments is logged."""
        import logging

        caplog.set_level(logging.INFO)
        model = MagicMock()

        tags_df = pd.DataFrame(
            {"Name": ["SC1", "SC2", "SC3"], "Tag": ["rural", "forests", "arable"]},
            index=["Subcatch SC1", "Subcatch SC2", "Subcatch SC3"],
        )
        model.inp.tags = tags_df

        dfs = pd.DataFrame({"Area": [100, 200, 300]}, index=["SC1", "SC2", "SC3"])
        service = SubcatchmentFeatureEngineeringService(dfs, model)

        service.subcatchments_classify()

        assert "Assigned categories to 3 subcatchments" in caplog.text

    def test_subcatchment_categories_alphabetical_order(self):
        """SUBCATCHMENT_CATEGORIES must match the alphabetical pd.get_dummies order from classifier training."""
        assert len(SUBCATCHMENT_CATEGORIES) == 12
        assert SUBCATCHMENT_CATEGORIES == sorted(SUBCATCHMENT_CATEGORIES)
        assert SUBCATCHMENT_CATEGORIES[0] == "arable"
        assert SUBCATCHMENT_CATEGORIES[1] == "forests"

    def test_ml_classifier_fallback_when_no_tags(self):
        """When [TAGS] is missing and all 12 feature columns + Area are present, use ML classifier fallback."""
        import numpy as np
        from unittest.mock import patch

        model = MagicMock()
        model.inp.tags = None

        dfs = pd.DataFrame(
            {
                "PercImperv": [25.0, 80.0],
                "PercSlope": [1.5, 2.0],
                "N-Imperv": [0.015, 0.012],
                "N-Perv": [0.15, 0.10],
                "S-Imperv": [1.5, 1.2],
                "S-Perv": [4.0, 3.0],
                "PctZero": [25.0, 25.0],
                "RunoffCoeff": [0.3, 0.8],
                "TotalInfil": [10.0, 2.0],
                "ImpervRunoff": [5.0, 20.0],
                "TotalRunoffMG": [0.5, 2.0],
                "PeakRunoff": [1.0, 4.0],
                "Area": [2.0, 4.0],
            },
            index=["SC1", "SC2"],
        )

        probs = np.zeros((2, 12), dtype=float)
        probs[0, 1] = 0.95  # index 1 -> "forests"
        probs[1, 9] = 0.90  # index 9 -> "urban_highly_impervious"

        fake_clf = MagicMock()
        fake_clf.predict.return_value = probs

        with patch("sa.core.predictor.get_classifier", return_value=fake_clf):
            service = SubcatchmentFeatureEngineeringService(dfs, model)
            service.subcatchments_classify(categories=True)

        assert list(service.dfs["category"]) == ["forests", "urban_highly_impervious"]
        passed_features = fake_clf.predict.call_args[0][0]
        assert passed_features.loc["SC1", "TotalInfil"] == pytest.approx(5.0)
        assert passed_features.loc["SC2", "PeakRunoff"] == pytest.approx(1.0)

    def test_partial_subcatchment_tags_fills_uncategorized_with_ml_fallback(self):
        """When [TAGS] covers only SC1 and all 12 feature columns + Area are present, SC2 is classified via ML."""
        import numpy as np
        from unittest.mock import patch

        model = MagicMock()
        model.inp.tags = pd.DataFrame(
            {"Name": ["SC1"], "Tag": ["rural"]},
            index=["Subcatch SC1"],
        )

        dfs = pd.DataFrame(
            {
                "PercImperv": [25.0, 80.0],
                "PercSlope": [1.5, 2.0],
                "N-Imperv": [0.015, 0.012],
                "N-Perv": [0.15, 0.10],
                "S-Imperv": [1.5, 1.2],
                "S-Perv": [4.0, 3.0],
                "PctZero": [25.0, 25.0],
                "RunoffCoeff": [0.3, 0.8],
                "TotalInfil": [10.0, float("nan")],
                "ImpervRunoff": [5.0, 20.0],
                "TotalRunoffMG": [0.5, 2.0],
                "PeakRunoff": [1.0, 4.0],
                "Area": [2.0, 4.0],
            },
            index=["SC1", "SC2"],
        )

        probs = np.zeros((2, 12), dtype=float)
        probs[0, 1] = 0.95  # would be "forests", but SC1 already has "rural" from [TAGS]
        probs[1, 9] = 0.90  # index 9 -> "urban_highly_impervious"

        fake_clf = MagicMock()
        fake_clf.predict.return_value = probs

        with patch("sa.core.predictor.get_classifier", return_value=fake_clf):
            service = SubcatchmentFeatureEngineeringService(dfs, model)
            service.subcatchments_classify(categories=True)

        assert service.dfs.loc["SC1", "category"] == "rural"
        assert service.dfs.loc["SC2", "category"] == "urban_highly_impervious"

