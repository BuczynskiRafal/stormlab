"""Frost zones are named "Zone I (0.8 m)" ... "Zone IV (1.4 m)" everywhere; their values and depths do not change."""

from __future__ import annotations

from django.test import TestCase
from django.urls import reverse

from sa.forms import SWMMModelForm
from sa.models import SWMMModel

from .factories import make_session, make_swmm_model, make_user

LABELS = {1: "Zone I (0.8 m)", 2: "Zone II (1.0 m)", 3: "Zone III (1.2 m)", 4: "Zone IV (1.4 m)"}


class ZoneLabelTests(TestCase):
    def setUp(self) -> None:
        self.user = make_user()

    def test_values_and_depths_used_by_the_analysis_are_unchanged(self):
        self.assertEqual(SWMMModel.ZONE_CHOICES, ((1, 0.8), (2, 1.0), (3, 1.2), (4, 1.4)))
        for zone, depth in SWMMModel.ZONE_CHOICES:
            self.assertEqual(make_swmm_model(self.user, zone=zone).get_frost_zone_value(), depth)

    def test_the_display_name_uses_roman_numerals_and_metres(self):
        for zone, label in LABELS.items():
            with self.subTest(zone=zone):
                self.assertEqual(make_swmm_model(self.user, zone=zone).get_zone_display_name(), label)

    def test_the_form_offers_the_same_labels_and_plain_field_labels(self):
        form = SWMMModelForm()

        self.assertEqual(list(form.fields["zone"].choices), list(LABELS.items()))
        self.assertEqual((form.fields["file"].label, form.fields["zone"].label), ("SWMM model file", "Frost zone"))

    def test_history_and_the_info_tab_print_the_zone_the_same_way(self):
        self.client.force_login(self.user)
        session = make_session(self.user, swmm_model=make_swmm_model(self.user, zone=3), n_conduits=1)

        self.assertContains(self.client.get(reverse("sa:history")), "Zone III (1.2 m)")
        self.assertContains(self.client.get(reverse("sa:analysis_results", args=[session.id])), "Zone III (1.2 m)")
