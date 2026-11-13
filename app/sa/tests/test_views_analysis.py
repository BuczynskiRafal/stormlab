"""``POST /analysis/``: a finished run opens the results page."""

from __future__ import annotations

from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from sa.models import CalculationSession

from .factories import make_user

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
