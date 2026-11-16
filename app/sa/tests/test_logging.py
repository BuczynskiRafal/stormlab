"""A handled exception reaches the log with its traceback, once."""

from __future__ import annotations

import json
from unittest import mock

from django.test import TestCase
from django.urls import reverse

from sa.core.data import DataManager
from sa.models import CalculationSession
from sa.services import AnalysisService, CalculationPersistenceService

from .factories import make_session, make_swmm_model, make_user


class TracebackLoggingTests(TestCase):
    def setUp(self) -> None:
        self.user = make_user()
        self.client.force_login(self.user)

    def assert_logged_with_traceback(self, logs, error: str) -> None:
        records = [record for record in logs.records if record.exc_info]
        self.assertTrue(records, "no record carries the exception")
        self.assertIn(error, "\n".join(logs.output))
        self.assertIn("Traceback", "\n".join(logs.output))

    def test_a_failed_analysis_is_logged_with_its_traceback(self):
        with (
            mock.patch("sa.services.DataManager", side_effect=RuntimeError("engine exploded")),
            self.assertLogs("sa.services", level="ERROR") as logs,
        ):
            AnalysisService.analyze_file(
                user=self.user, file_path="/nowhere.inp", frost_zone=0.8, swmm_model=make_swmm_model(self.user)
            )

        self.assert_logged_with_traceback(logs, "engine exploded")

    def test_a_failed_save_is_logged_with_its_traceback(self):
        session = make_session(self.user, status="processing")
        with (
            mock.patch.object(CalculationPersistenceService, "_save_conduit_data", side_effect=ValueError("bad row")),
            self.assertLogs("sa.services", level="ERROR") as logs,
        ):
            CalculationPersistenceService.save_calculation_results(session, mock.MagicMock())

        self.assert_logged_with_traceback(logs, "bad row")

    def test_a_failed_geometry_extraction_is_logged_with_its_traceback(self):
        session = make_session(self.user)
        with (
            mock.patch("sa.services.extract_geometry", side_effect=KeyError("coords")),
            self.assertLogs("sa.services", level="ERROR") as logs,
        ):
            CalculationPersistenceService._save_geometry(session, mock.MagicMock())

        self.assert_logged_with_traceback(logs, "coords")

    def test_a_failed_demo_run_is_logged_with_its_traceback(self):
        with (
            mock.patch("sa.views.AnalysisService.analyze_file", side_effect=OSError("disk full")),
            self.assertLogs("sa.views", level="ERROR") as logs,
        ):
            self.client.post(reverse("sa:analysis"), {"use_test_file": "true", "zone": 1})

        self.assert_logged_with_traceback(logs, "disk full")

    def test_failed_deletes_are_logged_with_their_traceback(self):
        session = make_session(self.user)
        with (
            mock.patch.object(CalculationSession, "delete", side_effect=RuntimeError("locked")),
            mock.patch("django.db.models.query.QuerySet.delete", side_effect=RuntimeError("locked")),
            self.assertLogs("sa.views", level="ERROR") as logs,
        ):
            single = self.client.post(reverse("sa:delete_session", args=[session.id]))
            bulk = self.client.post(
                reverse("sa:bulk_delete_sessions"), json.dumps({"session_ids": [session.id]}), content_type="application/json"
            )

        self.assertEqual((single.status_code, bulk.status_code), (500, 500))
        self.assertEqual(len([record for record in logs.records if record.exc_info]), 2)
        self.assertNotIn("locked", single.content.decode() + bulk.content.decode())

    def test_the_data_manager_does_not_log_the_exception_its_caller_handles(self):
        error = RuntimeError("raised inside the pipeline")
        with self.assertNoLogs("sa.core.data", level="ERROR"):
            self.assertFalse(DataManager.__exit__(mock.MagicMock(), RuntimeError, error, None))
