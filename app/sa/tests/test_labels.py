"""The display labels of recommendation codes and land-cover codes (``sa.labels``)."""

from __future__ import annotations

from django.test import SimpleTestCase

from sa.core.constants import SUBCATCHMENT_CATEGORIES
from sa.core.enums import RecommendationCategory
from sa.labels import ACTION_LABELS, CATEGORY_LABELS, MISSING, action_label, category_label
from stormlab.contracts.enums import RecommendationCategory as ContractRecommendationCategory
from stormlab.contracts.enums import SubcatchmentCategory


class LabelCoverageTests(SimpleTestCase):
    def test_every_recommendation_class_has_a_label(self):
        for enum in (RecommendationCategory, ContractRecommendationCategory):
            codes = {member.value for member in enum}
            self.assertEqual(codes, set(ACTION_LABELS), enum)

    def test_every_land_cover_class_has_a_label(self):
        self.assertEqual(set(SUBCATCHMENT_CATEGORIES), set(CATEGORY_LABELS))
        self.assertEqual({member.value for member in SubcatchmentCategory}, set(CATEGORY_LABELS))

    def test_labels_are_sentence_case_words_without_underscores(self):
        for label in [*ACTION_LABELS.values(), *CATEGORY_LABELS.values()]:
            self.assertNotIn("_", label)
            self.assertEqual(label[0], label[0].upper())
            self.assertEqual(label[1:], label[1:].lower(), label)


class LabelLookupTests(SimpleTestCase):
    def test_shared_vocabulary(self):
        self.assertEqual(action_label("valid"), "No changes")
        self.assertEqual(action_label("slope_increase"), "Increase slope")
        self.assertEqual(action_label("seepage_boxes"), "Seepage boxes")
        self.assertEqual(category_label("urban_weakly_impervious"), "Urban, weakly impervious")
        self.assertEqual(category_label("arable"), "Arable land")

    def test_lookup_ignores_case_and_surrounding_space(self):
        self.assertEqual(action_label("  VALID "), "No changes")
        self.assertEqual(category_label("Forests"), "Forests")

    def test_missing_values_read_as_a_dash(self):
        for value in (None, "", "  ", "-", MISSING):
            self.assertEqual(action_label(value), MISSING)
            self.assertEqual(category_label(value), MISSING)

    def test_unknown_codes_are_spelled_out_as_words(self):
        self.assertEqual(action_label("pipe_relining"), "Pipe relining")
        self.assertEqual(category_label("industrial_zone"), "Industrial zone")
