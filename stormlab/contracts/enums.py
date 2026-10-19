"""Canonical class vocabularies.

``RecommendationCategory`` mirrors ``app/sa/core/enums.py`` value-for-value and
in declaration order, because the saved recommendation models were trained with
that order as the softmax layout. ``SubcatchmentCategory`` names the twelve land
cover classes used by the catchment classifier and by the one-hot block of the
conduit feature contract.

The declaration order of ``SubcatchmentCategory`` is deliberately NOT a source
of truth for any model: both canonical orders live as explicit literals in
``features.py`` / ``catchment_features.py``. Deriving an order from this enum is
the exact mistake that produced the historical mis-decoding of the classifier.
"""

from __future__ import annotations

from enum import Enum


class RecommendationCategory(Enum):
    """Nine conduit intervention classes; index == declaration order."""

    PUMP = "pump"
    TANK = "tank"
    SEEPAGE_BOXES = "seepage_boxes"
    DIAMETER_INCREASE = "diameter_increase"
    DIAMETER_REDUCTION = "diameter_reduction"
    SLOPE_INCREASE = "slope_increase"
    SLOPE_REDUCTION = "slope_reduction"
    DEPTH_INCREASE = "depth_increase"
    VALID = "valid"


class SubcatchmentCategory(Enum):
    """Twelve land cover classes as written in the ``[TAGS]`` section of the INP files."""

    MARSHES = "marshes"
    ARABLE = "arable"
    MEADOWS = "meadows"
    FORESTS = "forests"
    RURAL = "rural"
    SUBURBAN_WEAKLY_IMPERVIOUS = "suburban_weakly_impervious"
    SUBURBAN_HIGHLY_IMPERVIOUS = "suburban_highly_impervious"
    URBAN_WEAKLY_IMPERVIOUS = "urban_weakly_impervious"
    URBAN_MODERATELY_IMPERVIOUS = "urban_moderately_impervious"
    URBAN_HIGHLY_IMPERVIOUS = "urban_highly_impervious"
    MOUNTAINS_ROCKY = "mountains_rocky"
    MOUNTAINS_VEGETATED = "mountains_vegetated"


RECOMMENDATION_CLASS_ORDER: tuple[str, ...] = tuple(category.value for category in RecommendationCategory)
"""Softmax column order of the recommendation models (== enum declaration order)."""
