"""Tests for ``SWMMModelForm.clean_file`` — FLOW_UNITS CMS gate.

Covers both ``SimpleUploadedFile`` (in-memory) and ``TemporaryUploadedFile``
(on-disk) to verify that the validator reads the file content without consuming
the buffer, so ``chunks()`` still returns the full content after validation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

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


class FormFileNameTests(SimpleTestCase):
    """The file name is checked before anything is read from the file."""

    def _errors(self, name: str) -> list:
        uploaded = SimpleUploadedFile(name, _read_fixture("flow_units_cms.inp"), content_type="application/octet-stream")
        form = SWMMModelForm(data={"zone": 1}, files={"file": uploaded})
        return [] if form.is_valid() else form.errors.as_data().get("file", [])

    def test_a_file_that_is_not_inp_is_asked_for_an_inp_file(self):
        for name in ("notes.txt", "model.inp.txt", "model", "model.rpt"):
            with self.subTest(name=name):
                errors = self._errors(name)
                self.assertEqual([e.code for e in errors], ["extension"])
                self.assertEqual(errors[0].messages, ["Choose a SWMM input file (.inp)."])

    def test_the_suffix_is_checked_in_any_letter_case(self):
        for name in ("MODEL.INP", "Model.Inp", "model.inp"):
            with self.subTest(name=name):
                self.assertEqual(self._errors(name), [])

    def test_a_wrong_name_wins_over_the_size_limit(self):
        uploaded = SimpleUploadedFile("notes.txt", b"x", content_type="text/plain")
        uploaded.size = settings.MAX_INP_BYTES + 1
        form = SWMMModelForm(data={"zone": 1}, files={"file": uploaded})

        self.assertEqual([e.code for e in form.errors.as_data()["file"]], ["extension"])


class FormConduitShapeTests(SimpleTestCase):
    """The checks treat Geom1 as a diameter, so other cross-sections are refused at upload."""

    DEMO = (Path(settings.BASE_DIR).parent / "examples" / "demo_network_coordinates.inp").read_text(encoding="utf-8")

    def _errors(self, text: str) -> list:
        uploaded = SimpleUploadedFile("x.inp", text.encode("utf-8"), content_type="application/octet-stream")
        form = SWMMModelForm(data={"zone": 1}, files={"file": uploaded})
        return [] if form.is_valid() else form.errors.as_data().get("file", [])

    def _with_shapes(self, shapes: dict[str, str]) -> str:
        text = self.DEMO
        for name, shape in shapes.items():
            text = text.replace(f"{name:<17}CIRCULAR    ", f"{name:<17}{shape:<12}", 1)
            self.assertIn(f"{name:<17}{shape}", text)
        return text

    def test_conduits_of_another_shape_are_named_with_their_shape(self):
        errors = self._errors(self._with_shapes({"K5": "RECT_CLOSED", "K6": "EGG", "K7": "RECT_OPEN", "K8": "EGG", "K9": "ARCH"}))

        self.assertEqual([e.code for e in errors], ["conduit_shape"])
        self.assertEqual(
            errors[0].messages,
            [
                "StormLab assesses circular conduits only; these conduits have another shape: "
                "K5 (RECT_CLOSED), K6 (EGG), K7 (RECT_OPEN) and 2 more. "
                "To analyse the network here, give them a CIRCULAR cross-section in [XSECTIONS]."
            ],
        )

    def test_force_main_and_lower_case_circular_conduits_are_accepted(self):
        self.assertEqual(self._errors(self._with_shapes({"K5": "FORCE_MAIN", "K6": "circular"})), [])

    def test_a_weir_with_a_rectangular_opening_is_not_a_conduit(self):
        weir = "[WEIRS]\nW1  J3  O1  TRANSVERSE  0.5  3.33  NO  0  0\n\n[XSECTIONS]\nW1  RECT_OPEN  1  1  0  0\n"
        self.assertEqual(self._errors(self.DEMO.replace("[XSECTIONS]\n", weir, 1)), [])

    def test_crlf_files_are_read_too(self):
        text = self._with_shapes({"K5": "TRAPEZOIDAL"}).replace("\n", "\r\n")
        self.assertEqual([e.code for e in self._errors(text)], ["conduit_shape"])


class FormEncodingTests(SimpleTestCase):
    """A Windows-1250 file is scanned as the UTF-8 copy the engine will read."""

    def _errors(self, payload: bytes) -> list:
        uploaded = SimpleUploadedFile("x.inp", payload, content_type="application/octet-stream")
        form = SWMMModelForm(data={"zone": 1}, files={"file": uploaded})
        return [] if form.is_valid() else form.errors.as_data().get("file", [])

    def test_a_windows_1250_model_with_polish_letters_is_accepted(self):
        self.assertEqual(self._errors(_read_fixture("polish_cp1250.inp")), [])

    def test_the_line_limit_applies_to_the_transcoded_line(self):
        # 600 letters "ż" are 601 bytes in Windows-1250 but 1201 in the UTF-8 copy the
        # engine reads, which it would split into two lines.
        comment = (";" + "ż" * 600 + "\n").encode("cp1250")
        self.assertLessEqual(len(comment), 1023)

        errors = self._errors(_read_fixture("flow_units_cms.inp") + comment)

        self.assertEqual([e.code for e in errors], ["unsafe_input"])


class FormCostLimitTests(SimpleTestCase):
    """A run is bounded by the request timeout and the temporary disk, so the form bounds what drives them."""

    DEMO = (Path(settings.BASE_DIR).parent / "examples" / "demo_network_coordinates.inp").read_text(encoding="utf-8")

    def _errors(self, text: str) -> list:
        uploaded = SimpleUploadedFile("x.inp", text.encode("utf-8"), content_type="application/octet-stream")
        form = SWMMModelForm(data={"zone": 1}, files={"file": uploaded})
        return [] if form.is_valid() else form.errors.as_data().get("file", [])

    def _option(self, name: str, value: str, text: str | None = None) -> str:
        text = self.DEMO if text is None else text
        lines = [f"{name:<21}{value}" if line.startswith(name + " ") else line for line in text.split("\n")]
        return "\n".join(lines)

    def test_the_audited_eight_day_one_second_run_is_refused_for_its_output(self):
        # Six and a half days are within the week; 561,600 one-second report steps of 51 elements are not.
        text = self._option("REPORT_STEP", "00:00:01", self._option("END_DATE", "06/23/2022"))

        errors = self._errors(text)

        self.assertEqual([e.code for e in errors], ["output_too_large"])
        self.assertEqual(
            errors[0].messages,
            [
                "The SWMM file would write an estimated 688 MB of results (elements × report steps × about 24 bytes); "
                "the limit is 200 MB. Lengthen REPORT_STEP or shorten the simulation."
            ],
        )

    def test_a_day_reported_every_minute_is_accepted(self):
        text = self._option("REPORT_STEP", "00:01:00", self._option("END_DATE", "06/18/2022"))
        self.assertEqual(self._errors(text), [])

    def test_more_than_a_week_is_refused(self):
        errors = self._errors(self._option("END_DATE", "06/25/2022"))

        self.assertEqual([e.code for e in errors], ["simulation_too_long"])
        self.assertEqual(
            errors[0].messages, ["The SWMM file simulates 204 hours (START to END in [OPTIONS]); the limit is 168 hours."]
        )

    def test_a_fraction_of_an_hour_is_rounded_up_to_one_decimal(self):
        from sa.forms import _hours

        self.assertEqual(_hours(204), "204 hours")
        self.assertEqual(_hours(204.0), "204 hours")
        self.assertEqual(_hours(168 + 1 / 60), "168.1 hours")  # one minute over a week
        self.assertEqual(_hours(204.5), "204.5 hours")
        self.assertEqual(_hours(204.25), "204.3 hours")
        self.assertEqual(_hours(1), "1 hour")
        self.assertEqual(_hours(1200000), "1,200,000 hours")

        errors = self._errors(self._option("END_TIME", "00:01:00", self._option("END_DATE", "06/24/2022")))
        self.assertEqual(
            errors[0].messages, ["The SWMM file simulates 168.1 hours (START to END in [OPTIONS]); the limit is 168 hours."]
        )

    def test_an_end_date_hidden_in_a_second_options_section_counts(self):
        errors = self._errors(self.DEMO + "\n[OPTIONS]\nEND_DATE 06/30/2022\n")
        self.assertEqual([e.code for e in errors], ["simulation_too_long"])

    def test_a_file_without_start_date_runs_from_2004_and_is_refused(self):
        text = "\n".join(line for line in self.DEMO.split("\n") if not line.startswith(("START_DATE", "REPORT_START")))
        self.assertEqual([e.code for e in self._errors(text)], ["simulation_too_long"])

    def test_a_tiny_routing_step_is_refused_with_the_estimate_and_the_limit(self):
        # 12 hours at 0.01 s are 4.32 million steps, counted as 51 elements each.
        errors = self._errors(self._option("ROUTING_STEP", "0.01"))

        self.assertEqual([e.code for e in errors], ["too_much_routing"])
        self.assertEqual(
            errors[0].messages,
            [
                "The SWMM file asks for an estimated 221 million element routing steps (elements × simulated time "
                "÷ ROUTING_STEP); the limit is 100 million. Lengthen ROUTING_STEP or shorten the simulation."
            ],
        )

    def test_a_tiny_model_counts_as_fifty_elements_per_routing_step(self):
        from sa.forms import routing_work

        period = mock.Mock(routing_steps=1_000_000)
        self.assertEqual(routing_work(4, period), 50_000_000)
        self.assertEqual(routing_work(5000, period), 5_000_000_000)

    def test_an_unreadable_period_option_is_named(self):
        errors = self._errors(self._option("END_DATE", "2022-06-18"))

        self.assertEqual([e.code for e in errors], ["option_value"])
        self.assertIn("END_DATE 2022-06-18", errors[0].messages[0])

    @override_settings(MAX_MODEL_ELEMENTS=50)
    def test_too_many_elements_are_refused_with_the_count_and_the_limit(self):
        errors = self._errors(self.DEMO)

        self.assertEqual([e.code for e in errors], ["too_many_elements"])
        self.assertEqual(
            errors[0].messages,
            ["The SWMM file has 51 elements (conduits, nodes and subcatchments); the limit is 50."],
        )

    @override_settings(MAX_SIMULATION_HOURS=12, MAX_MODEL_ELEMENTS=51, MAX_OUTPUT_MB=1, MAX_ROUTING_MILLIONS=1)
    def test_limits_follow_the_settings_and_a_file_within_them_passes(self):
        # The demo: 12 hours, 51 x 48 x 24 B = 0.06 MB, 51 x 720 = 0.04 million routing steps.
        self.assertEqual(self._errors(self.DEMO), [])
        with self.settings(MAX_SIMULATION_HOURS=11):
            self.assertEqual([e.code for e in self._errors(self.DEMO)], ["simulation_too_long"])

    def test_the_shipped_examples_pass(self):
        for path in (Path(settings.BASE_DIR).parent / "examples").glob("*.inp"):
            with self.subTest(path=path.name):
                self.assertEqual(self._errors(path.read_text(encoding="utf-8")), [])


class FormSizeLimitTests(SimpleTestCase):
    def _form(self, size: int):
        uploaded = SimpleUploadedFile("huge.inp", b"[OPTIONS]\nFLOW_UNITS CMS\n", content_type="application/octet-stream")
        uploaded.size = size
        return SWMMModelForm(data={"zone": 1}, files={"file": uploaded})

    def test_oversized_file_is_rejected_before_it_is_read(self):
        form = self._form(settings.MAX_INP_BYTES + 1)

        self.assertFalse(form.is_valid())
        self.assertEqual(form.errors["file"].as_data()[0].code, "too_large")

    @override_settings(MAX_INP_BYTES=50 * 1024 * 1024)
    def test_the_message_gives_both_sizes_in_megabytes(self):
        self.assertEqual(
            self._form(61 * 1024 * 1024 + 300000).errors["file"].as_data()[0].messages,
            ["The SWMM file is too large (61.3 MB; the limit is 50.0 MB)."],
        )
        # One byte over still reads as over.
        self.assertIn("(50.1 MB; the limit is 50.0 MB)", self._form(50 * 1024 * 1024 + 1).errors["file"].as_data()[0].messages[0])

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
