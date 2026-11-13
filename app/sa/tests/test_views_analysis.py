"""``POST /analysis/``: a finished run opens the results page, a failed one explains itself."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from django.contrib.messages import get_messages
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from sa.models import CalculationSession

from .factories import make_swmm_model, make_user

UNAVAILABLE_MESSAGE = "The recommendation model could not be used"
DEMO_NETWORK = Path(__file__).resolve().parents[3] / "examples" / "demo_network_coordinates.inp"


class DemoAnalysisTests(TestCase):
    """Runs the demo network through the view, models included."""

    def setUp(self) -> None:
        self.user = make_user()
        self.client.force_login(self.user)

    def test_demo_run_redirects_to_its_results_page_with_the_map(self):
        response = self.client.post(reverse("sa:analysis"), {"use_test_file": "true", "zone": 1})

        session = CalculationSession.objects.get(user=self.user)
        results_url = reverse("sa:analysis_results", args=[session.id])
        self.assertEqual(session.status, "completed")
        self.assertRedirects(response, results_url, fetch_redirect_response=False)

        page = self.client.get(results_url)

        self.assertEqual(page.status_code, 200)
        self.assertTrue(page.context["network_map_data"]["available"])
        self.assertContains(page, '<section id="r-map-panel" role="tabpanel" aria-labelledby="r-tab-map">', html=False)
        self.assertContains(page, 'id="nm-canvas"')

    def test_uploaded_file_redirects_to_its_results_page(self):
        upload = SimpleUploadedFile("network.inp", DEMO_NETWORK.read_bytes(), content_type="application/octet-stream")

        response = self.client.post(reverse("sa:analysis"), {"file": upload, "zone": 1})

        session = CalculationSession.objects.get(user=self.user)
        self.assertRedirects(response, reverse("sa:analysis_results", args=[session.id]), fetch_redirect_response=False)


class FailedAnalysisTests(TestCase):
    """The reason a run failed is on the page the user is returned to."""

    def setUp(self) -> None:
        self.user = make_user()
        self.client.force_login(self.user)

    def _post_demo_without_model(self):
        with (
            mock.patch("sa.core.data.gnn_enabled", return_value=True),
            mock.patch("sa.core.data.get_gnn_recommendation", return_value=None),
            mock.patch("sa.core.data.get_recommendation", return_value=None),
            self.assertLogs("sa.core.data", level="ERROR"),
        ):
            return self.client.post(reverse("sa:analysis"), {"use_test_file": "true", "zone": 1})

    def test_unavailable_model_is_explained_above_the_upload_form(self):
        response = self._post_demo_without_model()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f"The analysis failed: {UNAVAILABLE_MESSAGE}.")
        self.assertContains(response, 'class="alert alert-danger" role="alert"')
        self.assertContains(response, 'id="uploadForm"')
        self.assertEqual(CalculationSession.objects.get(user=self.user).status, "failed")
        self.assertEqual(list(get_messages(response.wsgi_request)), [])

    def _post_demo_with_failed_session(self, error_message: str):
        session = CalculationSession.objects.create(
            user=self.user, swmm_model=make_swmm_model(self.user), frost_zone=0.8, status="failed", error_message=error_message
        )
        with mock.patch("sa.views.AnalysisService.analyze_file", return_value=(session, False)):
            return self.client.post(reverse("sa:analysis"), {"use_test_file": "true", "zone": 1})

    def test_error_text_is_escaped(self):
        response = self._post_demo_with_failed_session("<script>alert('x')</script>")

        self.assertContains(response, "The analysis failed: &lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;.")
        self.assertNotContains(response, "<script>alert('x')")

    def test_failed_run_page_does_not_serve_jquery_and_datatables(self):
        response = self._post_demo_with_failed_session("The network has no outfall.")

        self.assertContains(response, 'id="uploadForm"')
        self.assertNotContains(response, "code.jquery.com")
        self.assertNotContains(response, "cdn.datatables.net")

    def test_message_that_ends_with_a_full_stop_gets_no_second_one(self):
        response = self._post_demo_with_failed_session("The network has no outfall.")

        self.assertContains(response, "The analysis failed: The network has no outfall.</div>")

    def test_failure_without_a_message_is_still_announced(self):
        response = self._post_demo_with_failed_session("")

        self.assertContains(response, "The analysis failed.</div>")

    def test_unexpected_error_shows_a_fixed_sentence_and_no_exception_text(self):
        failure = FileNotFoundError("/srv/stormlab/private/network.inp")
        with (
            mock.patch("sa.views.AnalysisService.analyze_file", side_effect=failure),
            self.assertLogs("sa.views", level="ERROR") as logs,
        ):
            response = self.client.post(reverse("sa:analysis"), {"use_test_file": "true", "zone": 1})

        self.assertContains(response, "The demo network could not be analysed.</div>")
        self.assertNotContains(response, "/srv/stormlab")
        self.assertIn("/srv/stormlab/private/network.inp", "\n".join(logs.output))

    def test_plain_form_has_no_alert(self):
        response = self.client.get(reverse("sa:analysis"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'role="alert"')
        self.assertContains(response, 'id="uploadForm"')
