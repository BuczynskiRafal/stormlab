"""``POST /analysis/``: a finished run opens the results page, a failed one explains itself."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from django.contrib.messages import get_messages
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from sa.models import CalculationSession

from .factories import make_user

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
        self.assertContains(response, UNAVAILABLE_MESSAGE)
        self.assertContains(response, 'class="alert alert-danger" role="alert"')
        self.assertContains(response, 'id="uploadForm"')
        self.assertEqual(CalculationSession.objects.get(user=self.user).status, "failed")
        self.assertEqual(list(get_messages(response.wsgi_request)), [])

    def test_error_text_is_escaped(self):
        failure = RuntimeError("<script>alert('x')</script>")
        with (
            mock.patch("sa.views.AnalysisService.analyze_file", side_effect=failure),
            self.assertLogs("sa.views", level="ERROR"),
        ):
            response = self.client.post(reverse("sa:analysis"), {"use_test_file": "true", "zone": 1})

        self.assertContains(response, "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;")
        self.assertNotContains(response, "<script>alert('x')")

    def test_plain_form_has_no_alert(self):
        response = self.client.get(reverse("sa:analysis"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'role="alert"')
        self.assertContains(response, 'id="uploadForm"')
