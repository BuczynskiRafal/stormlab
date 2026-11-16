"""A session deleted while its analysis runs stays deleted, whatever the analysis does next."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.urls import reverse

from sa.core.data import DataManager
from sa.models import CalculationSession, ConduitData, NodeData, SessionGeometry, SubcatchmentData

from .factories import make_user

DEMO_NETWORK = Path(__file__).resolve().parents[3] / "examples" / "demo_network_coordinates.inp"


class DeletedWhileRunningTests(TestCase):
    def setUp(self) -> None:
        self.user = make_user()
        self.client.force_login(self.user)
        # A second browser tab of the same user, which deletes the session mid-run.
        self.other_tab = Client()
        self.other_tab.force_login(self.user)

    def _delete_running_session(self) -> None:
        session = CalculationSession.objects.get(user=self.user, status="processing")
        response = self.other_tab.post(reverse("sa:delete_session", args=[session.id]))
        self.assertEqual(response.status_code, 200)

    def _upload(self):
        upload = SimpleUploadedFile("network.inp", DEMO_NETWORK.read_bytes(), content_type="application/octet-stream")
        return self.client.post(reverse("sa:analysis"), {"file": upload, "zone": 1})

    def assert_nothing_left(self) -> None:
        self.assertFalse(CalculationSession.objects.exists())
        for model in (ConduitData, NodeData, SubcatchmentData, SessionGeometry):
            self.assertFalse(model.objects.exists(), model.__name__)

    def test_a_session_deleted_before_its_results_are_saved_stays_deleted(self):
        original = DataManager.recommendations

        def delete_then_recommend(manager):
            self._delete_running_session()
            return original(manager)

        with mock.patch.object(DataManager, "recommendations", delete_then_recommend):
            response = self._upload()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "This analysis was deleted while it was running.")
        self.assert_nothing_left()

    def test_a_session_deleted_before_its_analysis_fails_stays_deleted(self):
        def delete_then_fail(manager):
            self._delete_running_session()
            raise RuntimeError("pipeline broke")

        with (
            mock.patch.object(DataManager, "recommendations", delete_then_fail),
            self.assertLogs("sa.services", level="ERROR"),
        ):
            self._upload()

        self.assert_nothing_left()
