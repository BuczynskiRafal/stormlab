"""
Feature engineering services for subcatchments.
"""

import logging

import numpy as np
import pandas as pd
import swmmio as sw

from .constants import SUBCATCHMENT_CATEGORIES

logger = logging.getLogger(__name__)

AREA_NORMALIZED_COLS = ("TotalInfil", "ImpervRunoff", "TotalRunoffMG", "PeakRunoff")
CATCHMENT_FEATURE_COLS = (
    "PercImperv",
    "PercSlope",
    "N-Imperv",
    "N-Perv",
    "S-Imperv",
    "S-Perv",
    "PctZero",
    "RunoffCoeff",
    *AREA_NORMALIZED_COLS,
)


class SubcatchmentFeatureEngineeringService:
    def __init__(self, dfs: pd.DataFrame, model: sw.Model):
        """Initialize the service with subcatchment dataframe and SWMM model.

        Args:
            dfs (pd.DataFrame): DataFrame containing subcatchment data.
            model (sw.Model): SWMM model instance.
        """
        self.dfs = dfs
        self.model = model

    def encode_category_column(self, category_column: str = "category") -> pd.DataFrame:
        """One-hot encodes the category column for use in machine learning models.

        Creates columns for all possible categories, not just those present in the data,
        ensuring consistent feature sets across different datasets.

        Args:
            category_column (str, optional): Name of the category column to encode.
                                            Defaults to "category".

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded category columns added
        """
        if self.dfs is None or len(self.dfs) == 0:
            return self.dfs

        if category_column not in self.dfs.columns:
            raise ValueError(f"Column '{category_column}' not found in dataframe")

        encoded_categories = pd.get_dummies(self.dfs[category_column], prefix=None, drop_first=False)

        for category in SUBCATCHMENT_CATEGORIES:
            if category not in encoded_categories.columns:
                encoded_categories[category] = 0
            else:
                encoded_categories[category] = encoded_categories[category].astype(int)

        result_df = pd.concat([self.dfs, encoded_categories], axis=1)
        return result_df

    def subcatchments_classify(self, categories: bool = True) -> None:
        """Assign subcatchment categories from ``[TAGS]`` with an ML classifier fallback.

        Args:
            categories (bool, optional): When True, writes string category names from
                ``SUBCATCHMENT_CATEGORIES``; when False, writes integer class indices.
                Defaults to True.

        Returns:
            None: Adds or updates the ``category`` column on ``self.dfs`` in-place.
        """
        if self.dfs is None or len(self.dfs) == 0:
            return

        if hasattr(self.model.inp, "tags") and self.model.inp.tags is not None:
            subcatch_tags = self.model.inp.tags[self.model.inp.tags.index.str.startswith("Subcatch")]

            if not subcatch_tags.empty:
                category_map = dict(zip(subcatch_tags["Name"], subcatch_tags["Tag"]))
                self.dfs["category"] = self.dfs.index.map(category_map)
                categorized_count = int(self.dfs["category"].notna().sum())
                logger.info(f"Assigned categories to {categorized_count} subcatchments from TAGS section")
                if categorized_count == len(self.dfs):
                    return
            else:
                logger.warning("No subcatchment tags found in TAGS section")
        else:
            logger.warning("No TAGS section found in the model")

        from .predictor import get_classifier

        required_cols = [*CATCHMENT_FEATURE_COLS, "Area"]
        missing_cols = [col for col in required_cols if col not in self.dfs.columns]
        if missing_cols:
            logger.warning(f"Cannot run ML subcatchment classifier; missing columns: {missing_cols}")
            return

        clf = get_classifier()
        if clf is None:
            logger.warning("ML subcatchment classifier is not available; skipping classification fallback")
            return

        df = self.dfs[required_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        area = df["Area"].where(df["Area"] > 0.0, np.nan)
        area_cols = list(AREA_NORMALIZED_COLS)
        df[area_cols] = df[area_cols].div(area, axis=0).fillna(0.0)

        features = df[list(CATCHMENT_FEATURE_COLS)].astype("float64")
        preds = clf.predict(features)
        preds_cls = preds.argmax(axis=-1)
        predicted_values = [SUBCATCHMENT_CATEGORIES[int(i)] for i in preds_cls] if categories else list(preds_cls)

        if "category" in self.dfs.columns:
            missing_mask = self.dfs["category"].isna()
            pred_series = pd.Series(predicted_values, index=self.dfs.index)
            self.dfs.loc[missing_mask, "category"] = pred_series.loc[missing_mask]
            logger.info(f"Classified {int(missing_mask.sum())} uncategorized subcatchments using ML classifier fallback")
        else:
            self.dfs["category"] = predicted_values
            logger.info(f"Classified {len(self.dfs)} subcatchments using ML classifier fallback")

