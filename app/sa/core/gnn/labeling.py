"""Canonical in-code GNN recommendation labeler."""

import logging

import numpy as np
import pandas as pd

from ..enums import RecommendationCategory

logger = logging.getLogger(__name__)

LABELER_VERSION = "conduit_classifier_v1"

HIGHLY_IMPERVIOUS_CATEGORIES = {
    "urban_highly_impervious",
    "urban_moderately_impervious",
    "suburban_highly_impervious",
}
MODERATELY_PERVIOUS_CATEGORIES = {
    "urban_weakly_impervious",
    "suburban_weakly_impervious",
    "rural",
}
HIGHLY_PERVIOUS_CATEGORIES = {
    "arable",
    "meadows",
    "forests",
    "marshes",
    "mountains_vegetated",
    "mountains_rocky",
}


def _val(row: pd.Series, key: str, default: float) -> float:
    raw = row.get(key, default)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _is_shallow_pervious(row: pd.Series, sbc_category: object) -> bool:
    return (
        sbc_category in (HIGHLY_PERVIOUS_CATEGORIES | MODERATELY_PERVIOUS_CATEGORIES)
        and _val(row, "NInletDepth", 1) <= 0.25
        and _val(row, "NOutletDepth", 1) <= 0.25
    )


def classify_conduit(row: pd.Series) -> RecommendationCategory:
    """Classify one conduit row using the original ConduitClassifier rules."""
    conduit_name = row.name
    sbc_category = row.get("SbcCategory", "unknown")
    logger.debug("Classifying conduit %s in SbcCategory=%s", conduit_name, sbc_category)

    if _val(row, "ValDepth", 1) == 0:
        logger.debug("Invalid depth for conduit %s", conduit_name)
        if sbc_category in HIGHLY_IMPERVIOUS_CATEGORIES:
            return RecommendationCategory.PUMP
        elif _val(row, "NSlope", 1) < 0.25:
            if _val(row, "ValMinSlope", 1) == 0 or _val(row, "IncreaseSlope", 0) == 1:
                return RecommendationCategory.PUMP
            return RecommendationCategory.SLOPE_INCREASE
        elif _val(row, "IncreaseDia", 0) == 1:
            return RecommendationCategory.DIAMETER_INCREASE
        return RecommendationCategory.PUMP

    if _val(row, "ValCoverage", 1) == 0:
        logger.debug("Invalid coverage for conduit %s", conduit_name)
        can_deepen = _val(row, "NInletDepth", 0) < 0.95 and _val(row, "NOutletDepth", 0) < 0.95
        if can_deepen:
            return RecommendationCategory.DEPTH_INCREASE
        if _is_shallow_pervious(row, sbc_category):
            return RecommendationCategory.SEEPAGE_BOXES
        return RecommendationCategory.TANK

    if _val(row, "ValMaxFill", 1) == 0:
        logger.debug("Overflow for conduit %s", conduit_name)
        if sbc_category in HIGHLY_IMPERVIOUS_CATEGORIES:
            return RecommendationCategory.TANK
        if (
            _val(row, "ValMaxSlope", 1) == 0
            or _val(row, "ReduceSlope", 0) == 1
            or (_val(row, "NFilling", 0) > 1.15 and (_val(row, "ValMinSlope", 1) == 0 or _val(row, "IncreaseSlope", 0) == 1))
            or _val(row, "IncreaseDia", 0) == 1
        ):
            return RecommendationCategory.DIAMETER_INCREASE
        if _is_shallow_pervious(row, sbc_category):
            return RecommendationCategory.SEEPAGE_BOXES
        return RecommendationCategory.DIAMETER_INCREASE

    if _val(row, "ValMinSlope", 1) == 0 or _val(row, "IncreaseSlope", 0) == 1:
        logger.debug("Slope too low for conduit %s", conduit_name)
        if sbc_category in HIGHLY_IMPERVIOUS_CATEGORIES:
            if _val(row, "ValMinV", 1) == 0:
                return RecommendationCategory.TANK
            return RecommendationCategory.VALID
        return RecommendationCategory.SLOPE_INCREASE

    if _val(row, "ValMaxSlope", 1) == 0 or _val(row, "ReduceSlope", 0) == 1:
        logger.debug("Slope too high for conduit %s", conduit_name)
        if _val(row, "ValMaxV", 1) == 0:
            return RecommendationCategory.SLOPE_REDUCTION
        if _val(row, "ReduceDia", 0) == 1 and sbc_category not in HIGHLY_IMPERVIOUS_CATEGORIES:
            return RecommendationCategory.DIAMETER_REDUCTION
        return RecommendationCategory.SLOPE_REDUCTION

    if _val(row, "IncreaseDia", 0) == 1:
        logger.debug("Diameter increase flag for conduit %s", conduit_name)
        if sbc_category in HIGHLY_IMPERVIOUS_CATEGORIES:
            return RecommendationCategory.TANK
        return RecommendationCategory.DIAMETER_INCREASE

    if _val(row, "ValMaxV", 1) == 0:
        logger.debug("Max velocity exceeded for conduit %s", conduit_name)
        if sbc_category in HIGHLY_IMPERVIOUS_CATEGORIES:
            return RecommendationCategory.TANK
        if _val(row, "NSlope", 0) > 0.75 and not (_val(row, "ValMaxSlope", 1) == 0 or _val(row, "ReduceSlope", 0) == 1):
            return RecommendationCategory.SLOPE_REDUCTION
        if _val(row, "IncreaseDia", 0) == 1:
            return RecommendationCategory.DIAMETER_INCREASE
        if _is_shallow_pervious(row, sbc_category):
            return RecommendationCategory.SEEPAGE_BOXES
        return RecommendationCategory.TANK

    if _val(row, "ValMinV", 1) == 0:
        logger.debug("Min velocity not met for conduit %s", conduit_name)
        if sbc_category in HIGHLY_IMPERVIOUS_CATEGORIES:
            return RecommendationCategory.VALID
        if _val(row, "NSlope", 1) < 0.25 and not (_val(row, "ValMinSlope", 1) == 0 or _val(row, "IncreaseSlope", 0) == 1):
            return RecommendationCategory.SLOPE_INCREASE
        if _val(row, "ReduceDia", 0) == 1:
            return RecommendationCategory.DIAMETER_REDUCTION
        return RecommendationCategory.SLOPE_INCREASE

    if _val(row, "ReduceDia", 0) == 1:
        logger.debug("Diameter reduction flag for conduit %s", conduit_name)
        if sbc_category in HIGHLY_IMPERVIOUS_CATEGORIES:
            return RecommendationCategory.VALID
        return RecommendationCategory.DIAMETER_REDUCTION

    if (
        _is_shallow_pervious(row, sbc_category)
        and _val(row, "ValMaxFill", 1) == 1
        and _val(row, "ValMinV", 1) == 1
        and _val(row, "NMaxQ", 0) > 0.6
    ):
        return RecommendationCategory.SEEPAGE_BOXES

    return RecommendationCategory.VALID


def classify_conduits(dfc: pd.DataFrame) -> pd.Series:
    """Classify all conduits in a dataframe, preserving the dataframe index."""
    result = dfc.apply(classify_conduit, axis=1)
    return pd.Series(result, index=dfc.index)


def _coerce_recommendation_label(label, row_name) -> RecommendationCategory:
    if isinstance(label, RecommendationCategory):
        return label

    if pd.isna(label):
        raise ValueError(f"Missing recommendation label for conduit '{row_name}': {label!r}")

    if isinstance(label, str):
        try:
            return RecommendationCategory(label)
        except ValueError as exc:
            valid_labels = ", ".join(cat.value for cat in RecommendationCategory)
            raise ValueError(
                f"Unknown recommendation label for conduit '{row_name}': {label!r}. Expected one of: {valid_labels}"
            ) from exc

    valid_labels = ", ".join(cat.value for cat in RecommendationCategory)
    raise ValueError(
        f"Unknown recommendation label for conduit '{row_name}': {label!r} "
        f"({type(label).__name__}). Expected one of: {valid_labels}"
    )


def prepare_swmm_labels(labels, one_hot=True):
    """Prepare SWMM recommendation labels, failing on missing or unknown labels."""
    all_classes = [cat.value for cat in RecommendationCategory]

    if isinstance(labels, pd.DataFrame):
        return labels.values.astype(np.float32)

    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)

    categories = [_coerce_recommendation_label(label, row_name) for row_name, label in labels.items()]
    label_to_idx = {category: idx for idx, category in enumerate(RecommendationCategory)}
    indices = np.array([label_to_idx[category] for category in categories], dtype=np.int64)

    if not one_hot:
        return indices

    y_onehot = np.zeros((len(indices), len(all_classes)), dtype=np.float32)
    if len(indices) > 0:
        y_onehot[np.arange(len(indices)), indices] = 1.0

    row_sums = y_onehot.sum(axis=1)
    if not np.all(row_sums == 1.0):
        bad_rows = labels.index[row_sums != 1.0].tolist()
        raise ValueError(f"One-hot encoding failed for conduit labels at rows: {bad_rows}")

    return y_onehot
