"""Tests for ``sa.core.inp_options.read_flow_units``."""

from __future__ import annotations

from django.test import SimpleTestCase

from sa.core.inp_options import read_flow_units


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
