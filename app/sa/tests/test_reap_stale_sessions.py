"""``reap_stale_sessions``: stuck ``processing`` sessions become ``failed``; nothing else changes."""

from __future__ import annotations

from datetime import timedelta
from io import StringIO

from django.core.management import CommandError, call_command
from django.test import TestCase
from django.utils import timezone

from sa.management.commands.reap_stale_sessions import STALE_MESSAGE
from sa.models import CalculationSession
from sa.tests.factories import make_session, make_user


class ReapStaleSessionsTests(TestCase):
    def setUp(self) -> None:
        self.user = make_user()

    def _session(self, status: str, minutes_ago: int) -> CalculationSession:
        session = make_session(self.user, status=status, n_conduits=1)
        # ``auto_now`` cannot be set through save(); update() bypasses it.
        CalculationSession.objects.filter(pk=session.pk).update(updated_at=timezone.now() - timedelta(minutes=minutes_ago))
        return session

    def _run(self, *args: str) -> str:
        out = StringIO()
        call_command("reap_stale_sessions", *args, stdout=out)
        return out.getvalue()

    def test_marks_only_old_processing_sessions_as_failed(self) -> None:
        stuck = self._session("processing", 45)
        running = self._session("processing", 5)
        done = self._session("completed", 600)

        output = self._run()

        stuck.refresh_from_db()
        self.assertEqual((stuck.status, stuck.error_message), ("failed", STALE_MESSAGE))
        self.assertEqual(stuck.conduits.count(), 1)
        running.refresh_from_db()
        done.refresh_from_db()
        self.assertEqual((running.status, done.status), ("processing", "completed"))
        self.assertIn("Marked 1 stale session(s)", output)

    def test_minutes_option_sets_the_threshold(self) -> None:
        session = self._session("processing", 5)

        self._run("--minutes", "2")

        session.refresh_from_db()
        self.assertEqual(session.status, "failed")

    def test_dry_run_changes_nothing(self) -> None:
        session = self._session("processing", 45)

        output = self._run("--dry-run")

        session.refresh_from_db()
        self.assertEqual(session.status, "processing")
        self.assertIn(f"Would mark 1 stale session(s) as failed: [{session.pk}]", output)

    def test_rejects_a_non_positive_threshold(self) -> None:
        with self.assertRaises(CommandError):
            self._run("--minutes", "0")
