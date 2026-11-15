"""Tests for ``sa.core.inp_options``: the FLOW_UNITS gate and the simulated period."""

from __future__ import annotations

from datetime import datetime

from django.test import SimpleTestCase

from sa.core.inp_options import OptionValueError, read_flow_units, simulation_period


class ReadFlowUnitsTests(SimpleTestCase):
    """Parsing of the FLOW_UNITS directive from INP [OPTIONS]."""

    def test_cms_returned_from_standard_options(self):
        lines = [
            "[OPTIONS]",
            "FLOW_UNITS           CMS",
            "",
            "[EVAPORATION]",
        ]
        self.assertEqual(read_flow_units(lines), "CMS")

    def test_cms_with_trailing_comment(self):
        lines = [
            "[OPTIONS]",
            "FLOW_UNITS  cms ; metric units",
            "[EVAPORATION]",
        ]
        self.assertEqual(read_flow_units(lines), "CMS")

    def test_none_when_flow_units_line_missing(self):
        lines = [
            "[OPTIONS]",
            "INFILTRATION   MODIFIED_GREEN_AMPT",
            "[EVAPORATION]",
        ]
        self.assertIsNone(read_flow_units(lines))

    def test_none_when_options_section_missing(self):
        lines = [
            "[TITLE]",
            "Some model",
            "[EVAPORATION]",
            "CONSTANT  0.0",
        ]
        self.assertIsNone(read_flow_units(lines))

    def test_flow_units_outside_options_ignored(self):
        """FLOW_UNITS appearing after [OPTIONS] ends must not be picked up."""
        lines = [
            "[OPTIONS]",
            "INFILTRATION   MODIFIED_GREEN_AMPT",
            "[EVAPORATION]",
            "FLOW_UNITS   CFS",
        ]
        self.assertIsNone(read_flow_units(lines))

    def test_cfs_returned_verbatim_upper(self):
        lines = [
            "[OPTIONS]",
            "flow_units   cfs",
            "[EVAPORATION]",
        ]
        self.assertEqual(read_flow_units(lines), "CFS")

    def test_comment_only_lines_skipped(self):
        lines = [
            "[OPTIONS]",
            ";; this is a comment",
            "FLOW_UNITS   LPS",
            "[EVAPORATION]",
        ]
        self.assertEqual(read_flow_units(lines), "LPS")

    def test_empty_input_returns_none(self):
        self.assertIsNone(read_flow_units([]))

    def test_scan_is_bounded_when_options_never_appears(self):
        from itertools import count

        from sa.core.inp_options import MAX_SCANNED_LINES

        seen = []

        def endless():
            for n in count():
                seen.append(n)
                yield "junk line\n"

        self.assertIsNone(read_flow_units(endless()))
        self.assertEqual(len(seen), MAX_SCANNED_LINES)


class SimulationPeriodTests(SimpleTestCase):
    """The simulated time and the report steps, read with the SWMM 5.2 semantics checked against the engine."""

    DEMO = [
        ("START_DATE", "06/17/2022"),
        ("START_TIME", "00:00:00"),
        ("REPORT_START_DATE", "06/17/2022"),
        ("REPORT_START_TIME", "00:00:00"),
        ("END_DATE", "06/17/2022"),
        ("END_TIME", "12:00:00"),
        ("REPORT_STEP", "00:15:00"),
    ]

    def period(self, changes=(), drop=()):
        lines = [(key, value) for key, value in self.DEMO if key not in drop]
        return simulation_period([*lines, *changes])

    def test_the_demo_network_simulates_twelve_hours_in_48_steps(self):
        period = self.period()
        self.assertEqual((period.duration_hours, period.report_steps), (12.0, 48))

    def test_the_shipped_examples_are_far_inside_the_default_limits(self):
        from pathlib import Path

        from django.conf import settings

        from sa.core.inp_structure import survey_inp

        for path in (Path(settings.BASE_DIR).parent / "examples").glob("*.inp"):
            survey = survey_inp([path.read_bytes()])
            period = simulation_period(survey.options)
            with self.subTest(path=path.name):
                self.assertLessEqual(period.duration_hours * 6, settings.MAX_SIMULATION_HOURS)
                self.assertLessEqual(period.report_steps * 4, settings.MAX_REPORT_STEPS)
                self.assertLessEqual(survey.elements * 30, settings.MAX_MODEL_ELEMENTS)

    def test_a_later_line_wins_and_keywords_match_by_prefix_in_any_case(self):
        period = self.period([("end_date_extra", "06/19/2022")])
        self.assertEqual(period.duration_hours, 60.0)

    def test_dates_with_dashes_and_month_names_are_read(self):
        self.assertEqual(self.period([("END_DATE", "JUN-18-2022")]).duration_hours, 36.0)
        self.assertEqual(self.period([("END_DATE", "06-18-2022")]).duration_hours, 36.0)

    def test_times_and_steps_in_decimal_hours_and_beyond_24_hours(self):
        self.assertEqual(self.period([("END_TIME", "6.5")]).duration_hours, 6.5)
        self.assertEqual(self.period([("END_TIME", "30:00:00")]).duration_hours, 30.0)
        self.assertEqual(self.period([("END_TIME", "12:30")]).duration_hours, 12.5)
        # A bare number is hours, not seconds: "0.5" is a 30-minute step.
        self.assertEqual(self.period([("REPORT_STEP", "0.5")]).report_steps, 24)
        self.assertEqual(self.period([("REPORT_STEP", "0:00:01")]).report_steps, 43200)

    def test_defaults_follow_the_engine(self):
        # No START_DATE: the run starts on 01/01/2004 and lasts until the END_DATE.
        period = self.period(drop=("START_DATE", "START_TIME", "REPORT_START_DATE", "REPORT_START_TIME"))
        self.assertEqual(period.duration_hours, (datetime(2022, 6, 17, 12) - datetime(2004, 1, 1)).total_seconds() / 3600)
        # No REPORT_STEP: 15 minutes.
        self.assertEqual(self.period(drop=("REPORT_STEP",)).report_steps, 48)
        # No END_DATE: 01/01/2004, before the start; the engine refuses it, the limits do not.
        self.assertEqual(self.period(drop=("END_DATE",)).duration_seconds, 0.0)

    def test_reporting_starts_at_the_report_start_only_when_both_parts_are_given(self):
        later = [("REPORT_START_TIME", "06:00:00")]
        self.assertEqual(self.period(later).report_steps, 24)
        self.assertEqual(self.period(drop=("REPORT_START_TIME",)).report_steps, 48)
        earlier = [("REPORT_START_DATE", "06/10/2022")]
        self.assertEqual(self.period(earlier).report_steps, 48)

    def test_an_unreadable_value_is_reported_with_its_option(self):
        cases = (
            ("END_DATE", "2022-06-18"),
            ("END_TIME", "inf"),
            ("END_TIME", "0x10"),
            ("END_TIME", "12:00abc"),
            ("REPORT_STEP", "0"),
            ("START_DATE", "13/01/2022"),
        )
        for option, value in cases:
            with self.subTest(option=option, value=value):
                with self.assertRaises(OptionValueError) as raised:
                    self.period([(option, value)])
                self.assertEqual((raised.exception.option, raised.exception.value), (option, value))

    def test_a_number_in_exponent_form_is_hours_as_the_engine_reads_it(self):
        self.assertEqual(self.period([("END_DATE", "06/17/2022"), ("END_TIME", "1e2")]).duration_hours, 100.0)
