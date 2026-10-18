"""Tests for ``SWMMModelForm.clean_file`` — FLOW_UNITS CMS gate.

Covers both ``SimpleUploadedFile`` (in-memory) and ``TemporaryUploadedFile``
(on-disk) to verify that the validator reads the file content without consuming
the buffer, so ``chunks()`` still returns the full content after validation.
"""

from __future__ import annotations

import os

from django.core.files.uploadedfile import SimpleUploadedFile, TemporaryUploadedFile
from django.test import SimpleTestCase

from sa.forms import SWMMModelForm

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
    def test_oversized_file_is_rejected_before_it_is_read(self):
        from sa.forms import MAX_INP_BYTES

        uploaded = SimpleUploadedFile("huge.inp", b"[OPTIONS]\nFLOW_UNITS CMS\n", content_type="application/octet-stream")
        uploaded.size = MAX_INP_BYTES + 1
        form = SWMMModelForm(data={"zone": 1}, files={"file": uploaded})

        self.assertFalse(form.is_valid())
        self.assertEqual(form.errors["file"].as_data()[0].code, "too_large")


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
