"""What a killed worker leaves behind is cleared without waiting for a restart, and nothing else is touched."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import time
from datetime import timedelta
from io import StringIO
from pathlib import Path
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from sa.housekeeping import (
    STALE_MESSAGE,
    TEMP_DIR_NAME,
    analysis_temp_dir,
    new_analysis_file,
    reap_stale_sessions,
    remove_analysis_files,
    remove_stale_analysis_files,
)
from sa.models import CalculationSession

from .factories import make_session, make_user

DEMO_NETWORK = Path(__file__).resolve().parents[3] / "examples" / "demo_network_coordinates.inp"


class PrivateTempDirTestCase(TestCase):
    """Points the system temporary directory at a fresh one, so the tests see only their own files."""

    def setUp(self) -> None:
        self.system_tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.system_tmp, ignore_errors=True)
        patcher = mock.patch("tempfile.tempdir", str(self.system_tmp))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.user = make_user()

    def age(self, path: Path, minutes: int) -> None:
        then = time.time() - minutes * 60
        os.utime(path, (then, then))

    def stale_session(self, minutes_ago: int, status: str = "processing") -> CalculationSession:
        session = make_session(self.user, status=status)
        CalculationSession.objects.filter(pk=session.pk).update(updated_at=timezone.now() - timedelta(minutes=minutes_ago))
        return session


class AnalysisFilesTests(PrivateTempDirTestCase):
    def test_analysis_files_live_in_one_private_directory(self):
        path = Path(new_analysis_file())

        self.assertEqual(path.parent, self.system_tmp / TEMP_DIR_NAME)
        self.assertEqual(path.suffix, ".inp")
        self.assertEqual(stat.S_IMODE(os.stat(path.parent).st_mode), 0o700)

    def test_a_symbolic_link_in_place_of_the_directory_is_refused(self):
        elsewhere = self.system_tmp / "elsewhere"
        elsewhere.mkdir()
        (self.system_tmp / TEMP_DIR_NAME).symlink_to(elsewhere)

        with self.assertRaises(OSError):
            analysis_temp_dir()

    def test_removing_an_analysis_removes_the_files_the_engine_wrote_beside_it(self):
        inp = Path(new_analysis_file())
        siblings = [inp.with_suffix(suffix) for suffix in (".rpt", ".out", ".bak")]
        for sibling in siblings:
            sibling.write_text("x")
        unrelated = inp.parent / "other.inp"
        unrelated.write_text("x")

        remove_analysis_files(str(inp))

        self.assertFalse(any(path.exists() for path in (inp, *siblings)))
        self.assertTrue(unrelated.exists())

    def test_only_old_files_of_the_directory_are_removed_unless_all_are_asked_for(self):
        old, young = Path(new_analysis_file()), Path(new_analysis_file())
        self.age(old, 45)
        subdirectory = analysis_temp_dir() / "keep"
        subdirectory.mkdir()
        outside = self.system_tmp / "not-ours.out"
        outside.write_text("x")
        self.age(outside, 600)

        self.assertEqual(remove_stale_analysis_files(timedelta(minutes=30)), 1)
        self.assertEqual((old.exists(), young.exists()), (False, True))

        self.assertEqual(remove_stale_analysis_files(None), 1)
        self.assertFalse(young.exists())
        self.assertTrue(subdirectory.is_dir())
        self.assertTrue(outside.exists())


class ReapingTests(PrivateTempDirTestCase):
    def test_reaping_is_one_update(self):
        stuck = self.stale_session(45)

        with self.assertNumQueries(1):
            self.assertEqual(reap_stale_sessions(), 1)

        stuck.refresh_from_db()
        self.assertEqual((stuck.status, stuck.error_message), ("failed", STALE_MESSAGE))

    def test_opening_the_analysis_form_or_the_history_reaps_stale_sessions_and_files(self):
        self.client.force_login(self.user)
        for url in (reverse("sa:analysis"), reverse("sa:history")):
            with self.subTest(url=url):
                stuck, running = self.stale_session(45), self.stale_session(5)
                old_file = Path(new_analysis_file())
                self.age(old_file, 45)

                self.assertEqual(self.client.get(url).status_code, 200)

                stuck.refresh_from_db()
                running.refresh_from_db()
                self.assertEqual((stuck.status, running.status), ("failed", "processing"))
                self.assertFalse(old_file.exists())

    def test_the_command_at_container_start_removes_every_leftover_file(self):
        young = Path(new_analysis_file())
        out = StringIO()

        call_command("reap_stale_sessions", "--all-temp-files", stdout=out)

        self.assertFalse(young.exists())
        self.assertIn("Removed 1 leftover analysis file(s).", out.getvalue())


class UploadTempFileTests(PrivateTempDirTestCase):
    def test_an_upload_is_analysed_from_the_analysis_directory_and_cleaned_up(self):
        self.client.force_login(self.user)
        seen: list[Path] = []

        def analyze_file(*, user, file_path, frost_zone, swmm_model):
            seen.append(Path(file_path))
            Path(file_path).with_suffix(".out").write_text("engine output")
            session = make_session(user, swmm_model=swmm_model, status="failed", error_message="stop")
            return session, False

        upload = SimpleUploadedFile("network.inp", DEMO_NETWORK.read_bytes(), content_type="application/octet-stream")
        with mock.patch("sa.views.AnalysisService.analyze_file", side_effect=analyze_file):
            self.client.post(reverse("sa:analysis"), {"file": upload, "zone": 1})

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].parent, self.system_tmp / TEMP_DIR_NAME)
        self.assertEqual(list((self.system_tmp / TEMP_DIR_NAME).iterdir()), [])
