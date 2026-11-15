"""Uploads shaped like the models designers actually have, run end to end through ``POST /analysis/``.

Each test edits a copy of the demo network the way a real file differs from it and
checks what the user gets: the results page, or a plain sentence above the form.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from sa.core.constants import SUBCATCHMENT_CATEGORIES
from sa.models import CalculationSession

from .factories import make_user

DEMO_NETWORK = Path(__file__).resolve().parents[3] / "examples" / "demo_network_coordinates.inp"


def demo_text() -> str:
    return DEMO_NETWORK.read_text(encoding="utf-8")


def without_section(text: str, name: str) -> str:
    """``text`` without the section ``[name]``, its header included."""
    pattern = re.compile(r"^\[" + re.escape(name) + r"\][^\[]*", flags=re.MULTILINE)
    assert pattern.search(text), name
    return pattern.sub("", text)


class RealWorldUploadTestCase(TestCase):
    def setUp(self) -> None:
        self.user = make_user()
        self.client.force_login(self.user)

    def upload(self, content: str | bytes, name: str = "network.inp"):
        payload = content.encode("utf-8") if isinstance(content, str) else content
        upload = SimpleUploadedFile(name, payload, content_type="application/octet-stream")
        return self.client.post(reverse("sa:analysis"), {"file": upload, "zone": 1})

    def assert_completed(self, response) -> CalculationSession:
        session = CalculationSession.objects.get(user=self.user)
        self.assertEqual(session.status, "completed", session.error_message)
        self.assertRedirects(response, reverse("sa:analysis_results", args=[session.id]), fetch_redirect_response=False)
        return session


class ModelWithoutTagsTests(RealWorldUploadTestCase):
    def test_model_without_a_tags_section_is_analysed_and_classified(self):
        response = self.upload(without_section(demo_text(), "TAGS"))

        session = self.assert_completed(response)
        categories = set(session.subcatchments.values_list("category", flat=True))
        self.assertEqual(session.subcatchments.count(), 7)
        self.assertTrue(categories <= set(SUBCATCHMENT_CATEGORIES), categories)


class ForeignTagTests(RealWorldUploadTestCase):
    def test_tags_outside_the_twelve_categories_do_not_stop_the_analysis(self):
        text = demo_text()
        text = text.replace("Subcatch  S1               urban_moderately_impervious", "Subcatch  S1               Osiedle_A")
        text = text.replace(
            "Subcatch  S2               urban_weakly_impervious", "Subcatch  S2               URBAN_WEAKLY_IMPERVIOUS"
        )
        text = text.replace("[TAGS]\n", "[TAGS]\nNode      J0               manhole_type_B\nLink      K1               forests\n")

        session = self.assert_completed(self.upload(text))

        categories = dict(session.subcatchments.values_list("subcatchment_name", "category"))
        self.assertIn(categories["S1"], SUBCATCHMENT_CATEGORIES)
        self.assertEqual(categories["S2"], "urban_weakly_impervious")
        self.assertEqual(categories["S3"], "urban_weakly_impervious")
        recommendations = set(session.conduits.values_list("recommendation", flat=True))
        self.assertNotIn("-", recommendations)


def with_diameters(text: str, diameters: dict[str, str]) -> str:
    """``text`` with the CIRCULAR diameter of each named conduit replaced."""
    for name, diameter in diameters.items():
        pattern = re.compile(r"^(" + re.escape(name) + r"\s+CIRCULAR\s+)\S+", flags=re.MULTILINE)
        text, count = pattern.subn(lambda match: match.group(1) + diameter, text)
        assert count == 1, name
    return text


class DiameterOutsideTheCriteriaTests(RealWorldUploadTestCase):
    """Conduits of 0.15 m and 2.5 m lie outside the 0.2-2.0 m the design criteria cover."""

    NOT_ASSESSED = ("min_diameter", "min_required_slope", "max_allowable_slope")
    SUGGESTIONS = ("is_min_diameter", "increase_dia", "reduce_dia", "increase_slope", "reduce_slope")

    def setUp(self) -> None:
        super().setUp()
        response = self.upload(with_diameters(demo_text(), {"K1": "0.15", "K4": "2.5"}))
        self.session = self.assert_completed(response)

    def test_such_conduits_are_stored_as_not_assessed_and_fail_the_checks_that_need_the_limits(self):
        for name in ("K1", "K4"):
            conduit = self.session.conduits.get(conduit_name=name)
            with self.subTest(conduit=name):
                for field in (*self.NOT_ASSESSED, *self.SUGGESTIONS):
                    self.assertIsNone(getattr(conduit, field), field)
                self.assertEqual((conduit.val_max_fill, conduit.val_max_slope, conduit.val_min_slope), (0, 0, 0))

        other = self.session.conduits.get(conduit_name="K2")
        self.assertIsNotNone(other.min_required_slope)
        self.assertIsNotNone(other.increase_slope)

    def test_results_page_and_export_render_them(self):
        page = self.client.get(reverse("sa:analysis_results", args=[self.session.id]))
        self.assertEqual(page.status_code, 200)
        conduits = {row["id"]: row for row in page.context["results_v2_data"]["conduits"]}
        self.assertIs(conduits["K1"]["limits_assessed"], False)
        self.assertIsNone(conduits["K1"]["min_required_slope"])
        self.assertIsNone(conduits["K1"]["increase_slope"])
        self.assertIs(conduits["K2"]["limits_assessed"], True)

        export = self.client.get(reverse("sa:export_session_csv", args=[self.session.id, "conduits"]))
        self.assertEqual(export.status_code, 200)

    def test_conduit_panel_says_the_limits_were_not_assessed(self):
        for name in ("K1", "K4"):
            with self.subTest(conduit=name):
                page = self.client.get(reverse("sa:conduit_detail", args=[self.session.id, name]))
                self.assertEqual(page.status_code, 200)
                self.assertContains(page, "not assessed")
                self.assertContains(page, "Ø outside 0.20–2.00 m")
                self.assertNotContains(page, "No change required by the design rules")


class FailureMessageTests(RealWorldUploadTestCase):
    """A failed run explains itself in plain words: no exception text, no paths, no placeholders."""

    def assert_failed_with(self, response, expected: str) -> CalculationSession:
        session = CalculationSession.objects.get(user=self.user)
        self.assertEqual(session.status, "failed")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(session.error_message, expected)
        self.assertContains(response, "The analysis failed: ")
        page = response.content.decode()
        for leak in ("/tmp", ".rpt", "%s", "Traceback", "KeyError"):
            self.assertNotIn(leak, page)
            self.assertNotIn(leak, session.error_message)
        return session

    def test_a_conduit_to_an_undefined_node_is_named(self):
        text = demo_text().replace("K21              C7               C1", "K21              C7               JX")

        with self.assertLogs("sa.services", level="WARNING"):
            response = self.upload(text)

        self.assert_failed_with(response, "These links refer to a node the model does not define: K21 → JX.")

    def test_a_file_with_options_only_says_it_has_no_conduits(self):
        with self.assertLogs("sa.services", level="WARNING"):
            response = self.upload(demo_text().split("[EVAPORATION]")[0])

        self.assert_failed_with(response, "The model has no conduits; StormLab assesses the conduits of a storm sewer network.")

    def test_a_model_without_subcatchments_says_what_is_required(self):
        text = demo_text()
        for section in ("SUBCATCHMENTS", "SUBAREAS", "INFILTRATION", "TAGS", "Polygons"):
            text = without_section(text, section)

        with self.assertLogs("sa.services", level="WARNING"):
            response = self.upload(text)

        self.assert_failed_with(
            response,
            "The model has no subcatchments; StormLab analyses networks loaded by rainfall runoff "
            "and needs at least one subcatchment.",
        )

    def test_an_input_error_of_the_engine_is_shown_with_the_elements_it_names(self):
        weir = "[WEIRS]\nW1  J3  O1  TRANSVERSE  0.5  3.33  NO  0  0\n\n[XSECTIONS]\nW1  RECT_OPEN  1  1  0  0\n"
        text = demo_text().replace("[XSECTIONS]\n", weir, 1)

        with self.assertLogs("sa.services", level="WARNING"):
            response = self.upload(text)

        self.assert_failed_with(
            response,
            "SWMM could not run the model: ERROR 133: Node J3 has more than one outlet link; "
            "ERROR 139: Regulator W1 is the outlet of a non-storage node.",
        )

    def test_an_unexpected_error_gives_the_analysis_number_and_logs_the_traceback(self):
        with (
            mock.patch("sa.core.data.DataManager.feature_engineering", side_effect=KeyError("MaxDPerc")),
            self.assertLogs("sa.services", level="ERROR") as logs,
        ):
            response = self.upload(demo_text())

        session = CalculationSession.objects.get(user=self.user)
        self.assert_failed_with(
            response,
            "The analysis stopped because of an unexpected error in the application. "
            f"Please report it, quoting analysis number {session.id}.",
        )
        self.assertNotIn("MaxDPerc", response.content.decode())
        self.assertIn("MaxDPerc", "\n".join(logs.output))
        self.assertIn("Traceback", "\n".join(logs.output))

    def test_a_report_that_cannot_be_read_back_is_an_unexpected_error(self):
        with (
            mock.patch("sa.core.data.DataManager._join_node_report_results", side_effect=ValueError("bad report")),
            self.assertLogs("sa.services", level="ERROR") as logs,
        ):
            response = self.upload(demo_text())

        session = CalculationSession.objects.get(user=self.user)
        self.assertIn(f"analysis number {session.id}", session.error_message)
        self.assertNotIn("bad report", response.content.decode())
        self.assertIn("SimulationReportError", "\n".join(logs.output))


class NonCircularConduitTests(RealWorldUploadTestCase):
    def test_a_rectangular_conduit_is_refused_before_anything_runs(self):
        text = re.sub(r"^(K5\s+)CIRCULAR(\s+)0\.3", r"\1RECT_CLOSED\g<2>0.6  0.4", demo_text(), flags=re.MULTILINE)

        response = self.upload(text)

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response, "StormLab assesses circular conduits only; these conduits have another shape: K5 (RECT_CLOSED)."
        )
        self.assertFalse(CalculationSession.objects.exists())


class WindowsEncodedModelTests(RealWorldUploadTestCase):
    def test_a_windows_1250_model_with_polish_letters_is_analysed(self):
        payload = (Path(__file__).resolve().parent / "data" / "polish_cp1250.inp").read_bytes()
        with self.assertRaises(UnicodeDecodeError):
            payload.decode("utf-8")

        session = self.assert_completed(self.upload(payload))

        names = set(session.subcatchments.values_list("subcatchment_name", flat=True))
        self.assertIn("\u015a1", names)  # "Ś1", spelled as in the file
        self.assertEqual(session.conduits.count(), 21)
