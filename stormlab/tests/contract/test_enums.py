"""The class vocabularies must not drift from the ones the models were trained on."""

from __future__ import annotations

import pytest

from stormlab.contracts.catchment_features import CLASSIFIER_CLASS_ORDER
from stormlab.contracts.enums import RECOMMENDATION_CLASS_ORDER, RecommendationCategory, SubcatchmentCategory

LEGACY_RECOMMENDATION_VALUES = (
    "pump",
    "tank",
    "seepage_boxes",
    "diameter_increase",
    "diameter_reduction",
    "slope_increase",
    "slope_reduction",
    "depth_increase",
    "valid",
)


def test_recommendation_categories_match_the_legacy_enum_value_and_order() -> None:
    assert tuple(category.value for category in RecommendationCategory) == LEGACY_RECOMMENDATION_VALUES


def test_recommendation_class_order_is_the_softmax_layout_the_models_were_trained_on() -> None:
    """Pinned against the literal above, never against its own derivation.

    ``RECOMMENDATION_CLASS_ORDER`` is defined as a comprehension over
    ``RecommendationCategory``, so comparing it with that same comprehension can
    never fail — a rename or a reordering of the enum would move both sides
    together and silently relabel every saved model's softmax column.
    """
    assert RECOMMENDATION_CLASS_ORDER == LEGACY_RECOMMENDATION_VALUES


def test_subcatchment_category_declares_twelve_unique_classes() -> None:
    values = [category.value for category in SubcatchmentCategory]
    assert len(values) == 12
    assert len(set(values)) == 12


def test_classifier_class_order_is_the_alphabetical_ordering_of_the_tag_vocabulary() -> None:
    assert CLASSIFIER_CLASS_ORDER == tuple(sorted(category.value for category in SubcatchmentCategory))


@pytest.mark.parametrize("category", list(SubcatchmentCategory))
def test_every_subcatchment_category_appears_in_the_classifier_order(category: SubcatchmentCategory) -> None:
    assert category.value in CLASSIFIER_CLASS_ORDER


def test_declaration_order_of_subcatchment_category_is_not_a_usable_class_order() -> None:
    """Neither model was trained on the enum's declaration order.

    Guards against a future ``[c.value for c in SubcatchmentCategory]`` — the
    derivation that historically mis-decoded 11 of the 12 classifier outputs.
    """
    from stormlab.contracts.features import FEATURE_ONEHOT_ORDER

    declared = tuple(category.value for category in SubcatchmentCategory)
    assert declared != CLASSIFIER_CLASS_ORDER
    assert declared != FEATURE_ONEHOT_ORDER
