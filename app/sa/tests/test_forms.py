"""Tests for ``SWMMModelForm.clean_file`` — FLOW_UNITS CMS gate.

Covers both ``SimpleUploadedFile`` (in-memory) and ``TemporaryUploadedFile``
(on-disk) to verify that the validator reads the file content without consuming
the buffer, so ``chunks()`` still returns the full content after validation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile, TemporaryUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from accounts.models import User
from sa.core.inp_safety import find_unsafe_input
from sa.forms import SWMMModelForm
from sa.models import CalculationSession

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _read_fixture(name: str) -> bytes:
    with open(os.path.join(DATA_DIR, name), "rb") as fh:
        return fh.read()


def _make_simple(name: str) -> SimpleUploadedFile:
    return SimpleUploadedFile(name, _read_fixture(name), content_type="application/octet-stream")


def _make_temporary(name: str) -> TemporaryUploadedFile:
    content = _read_fixture(name)
    tmp = TemporaryUploadedFile(
        name=name,
        content_type="application/octet-stream",
        size=len(content),
        charset=None,
    )
    tmp.write(content)
    tmp.seek(0)
    return tmp


class FormFlowUnitsGateTests(SimpleTestCase):
    """SWMMModelForm rejects files that are not FLOW_UNITS CMS."""

    def _get_form(self, uploaded):
        return SWMMModelForm(
            data={"zone": 1},
            files={"file": uploaded},
        )

    # --- CMS accepted ---

    def test_cms_file_accepted(self):
        form = self._get_form(_make_simple("flow_units_cms.inp"))
        self.assertTrue(form.is_valid(), form.errors)

    # --- CFS rejected ---

    def test_cfs_file_rejected(self):
        form = self._get_form(_make_simple("flow_units_cfs.inp"))
        self.assertFalse(form.is_valid())
        self.assertIn("file", form.errors)
        self.assertIn("CFS", form.errors["file"][0])

    def test_cfs_error_code(self):
        form = self._get_form(_make_simple("flow_units_cfs.inp"))
        form.is_valid()
        codes = [e.code for e in form.errors.as_data()["file"]]
        self.assertIn("flow_units", codes)

    # --- Missing FLOW_UNITS rejected ---

    def test_missing_flow_units_rejected(self):
        form = self._get_form(_make_simple("flow_units_missing.inp"))
        self.assertFalse(form.is_valid())
        self.assertIn("file", form.errors)

    def test_missing_flow_units_error_code(self):
        form = self._get_form(_make_simple("flow_units_missing.inp"))
        form.is_valid()
        codes = [e.code for e in form.errors.as_data()["file"]]
        self.assertIn("flow_units_missing", codes)

    # --- Buffer integrity after validation ---

    def _assert_full_content_via_chunks(self, uploaded, name):
        """After form validation, ``chunks()`` must still yield the full file, byte for byte."""
        form = self._get_form(uploaded)
        self.assertTrue(form.is_valid(), form.errors)
        # Re-read the file via chunks() — this is how the view consumes it.
        reassembled = b"".join(uploaded.chunks())
        self.assertEqual(reassembled, _read_fixture(name))

    def test_simple_upload_file_readable_after_validation(self):
        uploaded = _make_simple("flow_units_cms.inp")
        self._assert_full_content_via_chunks(uploaded, "flow_units_cms.inp")

    def test_temporary_upload_file_readable_after_validation(self):
        uploaded = _make_temporary("flow_units_cms.inp")
        self._assert_full_content_via_chunks(uploaded, "flow_units_cms.inp")


class FormSizeLimitTests(SimpleTestCase):
    def _form(self, size: int):
        uploaded = SimpleUploadedFile("huge.inp", b"[OPTIONS]\nFLOW_UNITS CMS\n", content_type="application/octet-stream")
        uploaded.size = size
        return SWMMModelForm(data={"zone": 1}, files={"file": uploaded})

    def test_oversized_file_is_rejected_before_it_is_read(self):
        form = self._form(settings.MAX_INP_BYTES + 1)

        self.assertFalse(form.is_valid())
        self.assertEqual(form.errors["file"].as_data()[0].code, "too_large")

    @override_settings(MAX_INP_BYTES=100)
    def test_limit_follows_the_setting(self):
        self.assertEqual(self._form(101).errors["file"].as_data()[0].code, "too_large")
        self.assertTrue(self._form(100).is_valid())


class FormHostileContentTests(SimpleTestCase):
    def _form(self, payload: bytes):
        uploaded = SimpleUploadedFile("x.inp", payload, content_type="application/octet-stream")
        return SWMMModelForm(data={"zone": 1}, files={"file": uploaded})

    def test_binary_payload_with_a_planted_directive_is_rejected(self):
        form = self._form(b"[OPTIONS]\nFLOW_UNITS CMS\x00\xff\xfe\n[JUNCTIONS]\n")

        self.assertFalse(form.is_valid())
        self.assertEqual(form.errors["file"].as_data()[0].code, "flow_units_missing")

    def test_one_long_line_is_read_only_up_to_the_byte_cap(self):
        from sa.forms import MAX_SCANNED_BYTES

        payload = b"x" * (MAX_SCANNED_BYTES + 10) + b"\n[OPTIONS]\nFLOW_UNITS CMS\n"
        form = self._form(payload)

        self.assertFalse(form.is_valid())
        self.assertEqual(form.errors["file"].as_data()[0].code, "flow_units_missing")


class FormExternalFileTests(SimpleTestCase):
    """Uploads that would make SWMM read or write another file are refused."""

    BASE = _read_fixture("flow_units_cms.inp")

    def _errors(self, extra: bytes, *, prefix: bytes = b"") -> list:
        uploaded = SimpleUploadedFile("x.inp", prefix + self.BASE + b"\n" + extra, content_type="application/octet-stream")
        form = SWMMModelForm(data={"zone": 1}, files={"file": uploaded})
        return [] if form.is_valid() else form.errors.as_data().get("file", [])

    def assertRejected(self, extra: bytes, *, prefix: bytes = b"") -> None:
        errors = self._errors(extra, prefix=prefix)
        self.assertEqual([e.code for e in errors], ["unsafe_input"], extra)

    def assertAccepted(self, extra: bytes) -> None:
        self.assertEqual(self._errors(extra), [], extra)

    def test_files_section_is_rejected_in_any_case_and_spelling_swmm_accepts(self):
        for header in (b"[FILES]", b"[files]", b"  [Files] ; note", b"[FILE]", b'"[FILES]"'):
            with self.subTest(header=header):
                self.assertRejected(header + b'\nSAVE HOTSTART "/tmp/x.hsf"\n')

    def test_empty_files_section_is_rejected(self):
        self.assertRejected(b"[FILES]\n")

    def test_files_section_anywhere_in_a_large_file_is_found(self):
        filler = b";;" + b"x" * 100 + b"\n"
        self.assertRejected(filler * 20000 + b"[FILES]\nUSE RAINFALL rain.dat\n")

    def test_rain_gage_reading_a_file_is_rejected(self):
        self.assertRejected(b'[RAINGAGES]\nRG1 INTENSITY 0:15 1.0 FILE "rain.dat" STA1 MM\n')
        self.assertRejected(b"[raingages]\nRG1 INTENSITY 0:15 1.0 file rain.dat STA1 MM\n")

    def test_time_series_reading_a_file_is_rejected(self):
        self.assertRejected(b'[TIMESERIES]\nTS1 FILE "C:\\data\\ts.dat"\n')
        self.assertRejected(b'[TIMESERIES]\nTS1 "  file" ts.dat\n')

    def test_climate_files_are_rejected(self):
        self.assertRejected(b'[TEMPERATURE]\nFILE "climate.dat" 01/01/2020\n')
        self.assertRejected(b"[TEMPERATURE]\nWINDSPEED FILE\n")
        self.assertRejected(b"[EVAPORATION]\nFILE 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7\n")

    def test_lid_report_file_is_rejected(self):
        self.assertRejected(b'[LID_USAGE]\nS1 BC1 1 100 10 0 0 0 "report.txt"\n')
        self.assertRejected(b'[LID_USAGE]\nS1 BC1 1 100 10 0 0 0 ""\n')

    def test_tempdir_option_is_rejected(self):
        self.assertRejected(b"[OPTIONS]\nTEMPDIR /tmp/elsewhere\n")

    def test_line_the_engine_would_split_is_rejected(self):
        # SWMM reads 1023 bytes at a time, so the header below would start a line of its own.
        self.assertRejected(b";" + b"x" * 1022 + b"[FILES]\nSAVE HOTSTART x.hsf\n")

    def test_header_hidden_behind_a_nul_byte_does_not_change_the_section(self):
        # The engine stops reading a line at NUL, so it stays in [TIMESERIES] here.
        self.assertRejected(b"[TIMESERIES]\n\x00[TITLE]\nTS1 FILE ts.dat\n")

    def test_look_alikes_are_accepted(self):
        accepted = [
            b"[TITLE]\nThe FILE keyword in a title, and [FILES] in prose: FILE x\n",
            b"[TIMESERIES]\nFILE1 0:00 0.0\nFILE1 1:00 2.5\n",
            b"[TIMESERIES]\nTS1 0:00 0.0 ; read from FILE later\n",
            b"[RAINGAGES]\nRG1 INTENSITY 0:15 1.0 TIMESERIES FILE1\n",
            b"[TEMPERATURE]\nTIMESERIES FILE1\nWINDSPEED MONTHLY 1 1 1 1 1 1 1 1 1 1 1 1\n",
            b"[EVAPORATION]\nCONSTANT 0.0\nDRY_ONLY NO\n",
            b"[LID_USAGE]\nS1 BC1 1 100 10 0 0 0\nS2 BC1 1 100 10 0 0 0 * J1\n",
            b";[FILES]\n; [FILES] commented out\n",
            b'[BACKDROP]\nFILE "map.png"\nDIMENSIONS 0 0 100 100\n',
            b"[OPTIONS]\nTEMP_UNITS C\n",
            b";" + b"x" * 1022 + b"\n",
        ]
        for extra in accepted:
            with self.subTest(extra=extra[:60]):
                self.assertAccepted(extra)

    def test_crlf_line_endings_are_handled(self):
        self.assertRejected(b"[FILES]\r\nSAVE HOTSTART x.hsf\r\n")
        self.assertAccepted(b"[TIMESERIES]\r\nTS1 0:00 0.0\r\n")

    def test_shipped_and_test_models_pass(self):
        repo = Path(settings.BASE_DIR).parent
        paths = [*repo.glob("examples/*.inp"), *repo.glob("app/sa/tests/data/*.inp"), *repo.glob("app/sa/core/tests/*.inp")]
        self.assertGreater(len(paths), 10)
        for path in paths:
            with self.subTest(path=path.name), path.open("rb") as fh:
                self.assertIsNone(find_unsafe_input(iter(lambda: fh.read(4096), b"")))


class UploadProofOfConceptTests(TestCase):
    """The upload that used to make SWMM write a file at a path chosen by the uploader."""

    def test_files_section_upload_runs_nothing_and_writes_nothing(self):
        user = User.objects.create_user(email="poc@example.com", password="poc-pass-123")
        self.client.force_login(user)
        demo = (Path(settings.BASE_DIR).parent / "examples" / "demo_network_coordinates.inp").read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "written_by_swmm.hsf"
            payload = f'[FILES]\nSAVE HOTSTART "{target}"\n\n'.encode() + demo
            upload = SimpleUploadedFile("demo.inp", payload, content_type="application/octet-stream")

            response = self.client.post(reverse("sa:analysis"), {"file": upload, "zone": 1})

            self.assertFalse(target.exists())
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, "must be self-contained")
        self.assertFalse(CalculationSession.objects.exists())
