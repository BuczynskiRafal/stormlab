"""Tests for the markup contract of the conduit panel (``sa/conduit_panel.html``).

The six cards of the "Results and design criteria" section come from the seven
validation booleans of the conduit (velocity and slope each join two of them), and the
count in the section head comes from ``ConduitPanel.checks_failed`` over the same
booleans, so the assertions below are the guard that count and cards never disagree.
"""

from __future__ import annotations

import html as html_lib
import re
import shutil
import subprocess
from pathlib import Path

from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from .factories import DRY_CONDUIT, VALIDATION_FLAGS, make_conduit, make_node, make_session, make_subcatchment, make_user

CARD_RE = re.compile(
    r'<li class="criterion(?: (is-failed|is-passed))?" data-criterion="([\w-]+)">(.*?)</li>',
    re.DOTALL,
)


def _text(fragment: str) -> str:
    """Visible text of a markup fragment, whitespace collapsed."""
    return " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def criteria_cards(page: str) -> dict[str, dict[str, str | None]]:
    """The criteria cards of a rendered page by key: state, value, criterion, status."""
    cards: dict[str, dict[str, str | None]] = {}
    for state, key, body in CARD_RE.findall(page):
        part = {
            name: re.search(rf'<span class="criterion-{name}">(.*?)</span>\s*(?:<span class="criterion-|$)', body, re.DOTALL)
            for name in ("label", "value", "limit")
        }
        status = re.search(r'<span class="criterion-status">(.*)</span>', body, re.DOTALL)
        cards[key] = {
            "state": state or None,
            "label": _text(part["label"].group(1)),
            "value": _text(part["value"].group(1)),
            "limit": _text(part["limit"].group(1)),
            "status": _text(status.group(1)) if status else None,
        }
    return cards


class CriteriaSectionTests(TestCase):
    """The section that replaced the metric bar and the Technical validation card."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="detail@example.com")
        cls.session = make_session(cls.user, n_conduits=1, n_nodes=2, n_subcatchments=1)

    def setUp(self):
        self.client.force_login(self.user)

    def _page(self, name="C1", **flags):
        if name != "C1":
            make_conduit(self.session, conduit_name=name, inlet_node="N1", outlet_node="N2", subcatchment="S1", **flags)
        response = self.client.get(reverse("sa:conduit_detail", args=[self.session.id, name]))
        self.assertEqual(response.status_code, 200)
        return response.content.decode()

    def test_six_cards_in_order_with_values_criteria_and_quiet_passed_states(self):
        page = self._page()
        cards = criteria_cards(page)

        self.assertEqual(list(cards), ["max_q", "val_max_fill", "velocity", "slope", "val_depth", "val_coverage"])
        self.assertEqual(
            {key: (card["label"], card["value"], card["limit"]) for key, card in cards.items()},
            {
                "max_q": ("Max flow", "0.150 m³/s", "SWMM peak"),
                "val_max_fill": ("Filling", "50 %", "≤ 0.33 m (82.7 %)"),
                "velocity": ("Velocity", "1.20 m/s", "0.70 – 5.00 m/s"),
                "slope": ("Slope", "5.00 ‰", "2.00 – 40.00 ‰"),
                "val_depth": ("Depth", "3.00 m", "≤ 8.00 m"),
                # The frost zone of the session is named in the Cover card.
                "val_coverage": ("Cover", "1.50 m", "≥ 0.80 m (frost zone)"),
            },
        )
        # Max flow has no criterion: neutral, no verdict. The others pass quietly.
        self.assertEqual((cards["max_q"]["state"], cards["max_q"]["status"]), (None, None))
        for key in ("val_max_fill", "velocity", "slope", "val_depth", "val_coverage"):
            self.assertEqual((cards[key]["state"], cards[key]["status"]), ("is-passed", "✓ within limits"), key)
        self.assertIn("All 7 checks passed", page)
        self.assertNotIn("checks failed", page)

    def test_every_stored_flag_drives_its_card_and_names_the_failed_side(self):
        cases = {
            "val_max_fill": ("val_max_fill", "exceeds limit"),
            "val_max_v": ("velocity", "above maximum"),
            "val_min_v": ("velocity", "below minimum"),
            "val_max_slope": ("slope", "above maximum"),
            "val_min_slope": ("slope", "below minimum"),
            "val_depth": ("val_depth", "exceeds limit"),
            "val_coverage": ("val_coverage", "below minimum"),
        }
        self.assertEqual(set(cases), set(VALIDATION_FLAGS))
        for flag, (card_key, status) in cases.items():
            with self.subTest(flag=flag):
                page = self._page(f"C-{flag}", **{flag: 0})
                cards = criteria_cards(page)
                failed = {key for key, card in cards.items() if card["state"] == "is-failed"}
                self.assertEqual(failed, {card_key})
                self.assertEqual(cards[card_key]["status"], status)
                self.assertIn("1 of 7 checks failed", page)

    def test_both_sides_of_a_range_failing_report_the_maximum_first(self):
        page = self._page("C-both", val_max_v=0, val_min_v=0)

        card = criteria_cards(page)["velocity"]
        self.assertEqual((card["state"], card["status"]), ("is-failed", "above maximum"))
        self.assertIn("2 of 7 checks failed", page)

    def test_a_diameter_outside_the_standard_range_is_named_in_the_filling_card(self):
        page = self._page("C-odd", geom1=0.15, filling=0.05, val_max_fill=0)

        card = criteria_cards(page)["val_max_fill"]
        self.assertEqual(
            (card["state"], card["value"], card["limit"], card["status"]),
            ("is-failed", "33 %", "Ø outside 0.20–2.00 m", "diameter out of range"),
        )

    def test_a_dry_conduit_with_an_empty_slope_range_says_no_slope_can_pass(self):
        page = self._page("C-dry", **DRY_CONDUIT)

        cards = criteria_cards(page)
        self.assertEqual(
            (cards["slope"]["state"], cards["slope"]["value"], cards["slope"]["limit"], cards["slope"]["status"]),
            ("is-failed", "1.41 ‰", "no feasible slope (min 229.72 > max 134.56 ‰)", "no feasible range"),
        )
        self.assertIn("max 134.56\u00a0‰)", page, "the unit does not wrap away from its number")
        self.assertNotIn("229.72 – 134.56", page)
        self.assertEqual((cards["velocity"]["state"], cards["velocity"]["status"]), ("is-failed", "below minimum"))
        # The stored flags keep counting as they are.
        self.assertIn("2 of 7 checks failed", page)

    def test_the_section_head_counts_failed_checks_and_replaces_the_old_caption_and_badge(self):
        flags = dict.fromkeys(VALIDATION_FLAGS, 1)
        flags.update(val_max_fill=0, val_min_v=0, val_coverage=0)
        page = self._page("C-three", **flags)

        head = re.search(r'<div class="criteria-head">(.*?)</div>', page, re.DOTALL).group(1)
        self.assertIn("Results and design criteria", head)
        self.assertIn('<span class="criteria-count is-warning">3 of 7 checks failed</span>', head)
        self.assertEqual(page.count("checks failed"), 1, "the count is not repeated at the top")
        for gone in (
            "Technical validation",
            "checks passed</span>",
            "Attention required",
            "Minor issues",
            "validation-item",
            "panel-badge",
        ):
            self.assertNotIn(gone, page)


class PageHeadingTests(TestCase):
    """The page opens on the drawing: the profile card head is the page heading."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="heading@example.com")
        cls.session = make_session(cls.user, n_conduits=1, n_nodes=2, n_subcatchments=1)

    def setUp(self):
        self.client.force_login(self.user)
        self.page = self.client.get(reverse("sa:conduit_detail", args=[self.session.id, "C1"])).content.decode()

    def test_exactly_one_h1_and_it_heads_the_profile_card(self):
        self.assertEqual(len(re.findall(r"<h1\b", self.page)), 1)
        card = re.search(r'<section class="panel-surface panel-profile-card">(.*?)</section>', self.page, re.DOTALL).group(1)
        self.assertRegex(card, r'<h1 class="[^"]*">Conduit C1</h1>\s*<p class="panelhead-sub">Longitudinal profile</p>')
        # The Back button moved into that head, with the results URL.
        results_url = reverse("sa:analysis_results", args=[self.session.id])
        self.assertIn(f'<a href="{results_url}" class="panel-back">', card)
        self.assertRegex(self.page, r"<title>\s*Conduit C1 - Session")

    def test_the_profile_card_is_the_first_content_block(self):
        # Nothing stands between the page wrapper and the profile/prediction row: the
        # prev/next pager lives in the profile card head, not above it.
        self.assertRegex(self.page, r'<div class="conduit-page">\s*<div class="panel-layout">')
        self.assertLess(self.page.index("panel-profile-card"), self.page.index('id="criteria"'))
        self.assertLess(self.page.index('id="criteria"'), self.page.index('id="conduit-details"'))

    def test_breadcrumb_subtitle_metric_bar_eyebrow_and_foot_are_gone(self):
        for gone in (
            "panel-crumb",
            'aria-label="Breadcrumb"',
            "panel-heading",
            "Simulation results and model recommendation",
            "Created ",
            "panel-metrics",
            "Key results",
            '<span class="panel-eyebrow">C1</span>',
            "Geometry and pipe filling",
            'class="panel-foot"',
            "recommendation classes",
        ):
            self.assertNotIn(gone, self.page, gone)

    def test_the_caption_sits_in_the_profile_card_under_the_drawing(self):
        card = re.search(r'<section class="panel-surface panel-profile-card">(.*?)</section>', self.page, re.DOTALL).group(1)
        self.assertIn('<p id="conduit-profile-note" class="profile-caption" hidden></p>', card)
        self.assertLess(card.index('id="conduit-diagram"'), card.index('id="conduit-profile-note"'))
        self.assertLess(card.index('id="conduit-profile-note"'), card.index('class="chain-details"'))


# Every property the conduit panel showed before the details card replaced the single
# table; each one still has to be on the page, under one of the three tabs.
PROPERTY_LABELS = (
    "Name",
    "Inlet Node",
    "Outlet Node",
    "Length",
    "Diameter",
    "Max Velocity",
    "Max Flow",
    "Filling",
    "Slope",
    "Min Required Slope",
    "Max Allowable Slope",
    "Inlet Ground Elevation",
    "Outlet Ground Elevation",
    "Inlet Ground Cover",
    "Outlet Ground Cover",
    "Subcatchment (model)",
    "Subcatchment Category",
    "Recommendation",
)


class ConduitPagerTests(TestCase):
    """Previous / next conduit: a segmented pager in the profile card head, before Back."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="pager@example.com")
        cls.session = make_session(cls.user, n_conduits=3, n_nodes=4, n_subcatchments=3)

    def setUp(self):
        self.client.force_login(self.user)

    def _url(self, name: str) -> str:
        return reverse("sa:conduit_detail", args=[self.session.id, name])

    def _pager(self, name: str) -> tuple[str, str]:
        """The rendered page and the markup of its pager."""
        page = self.client.get(self._url(name)).content.decode()
        pager = re.search(r'<nav class="conduit-pager" aria-label="Conduit navigation">(.*?)</nav>', page, re.DOTALL)
        self.assertIsNotNone(pager, "the pager is missing")
        return page, pager.group(1)

    def test_middle_conduit_links_both_neighbours(self):
        _, pager = self._pager("C2")

        self.assertIn(
            f'<a href="{self._url("C1")}" class="conduit-pager-link" rel="prev" '
            'aria-label="Previous conduit: C1" title="Previous conduit C1 (&larr;)">',
            pager,
        )
        self.assertIn(
            f'<a href="{self._url("C3")}" class="conduit-pager-link" rel="next" '
            'aria-label="Next conduit: C3" title="Next conduit C3 (&rarr;)">',
            pager,
        )
        self.assertIn('<span class="conduit-pager-name">C1</span>', pager)
        self.assertIn('<span class="conduit-pager-name">C3</span>', pager)
        self.assertNotIn("aria-disabled", pager)
        # Previous comes first, and each half carries a drawn chevron.
        self.assertLess(pager.index('rel="prev"'), pager.index('rel="next"'))
        self.assertEqual(pager.count('<svg class="conduit-pager-icon"'), 2)
        self.assertEqual(pager.count('stroke="currentColor" stroke-width="2"'), 2)

    def test_first_conduit_renders_a_disabled_previous_half(self):
        _, pager = self._pager("C1")

        self.assertNotIn('rel="prev"', pager)
        self.assertIn('<span class="conduit-pager-link is-disabled" aria-disabled="true" title="No previous conduit">', pager)
        self.assertIn('<span class="visually-hidden">No previous conduit</span>', pager)
        self.assertIn(f'<a href="{self._url("C2")}" class="conduit-pager-link" rel="next"', pager)
        # The disabled half keeps its place and width: still two halves, previous first, and
        # a name slot (a dash) so it is as wide as a named half.
        self.assertEqual(pager.count('<span class="conduit-pager-name" aria-hidden="true">&mdash;</span>'), 1)
        self.assertEqual(pager.count('class="conduit-pager-link'), 2)
        self.assertLess(pager.index("is-disabled"), pager.index('rel="next"'))

    def test_last_conduit_renders_a_disabled_next_half(self):
        _, pager = self._pager("C3")

        self.assertNotIn('rel="next"', pager)
        self.assertIn('<span class="conduit-pager-link is-disabled" aria-disabled="true" title="No next conduit">', pager)
        self.assertIn('<span class="visually-hidden">No next conduit</span>', pager)
        self.assertEqual(pager.count('<span class="conduit-pager-name" aria-hidden="true">&mdash;</span>'), 1)
        self.assertIn(f'<a href="{self._url("C2")}" class="conduit-pager-link" rel="prev"', pager)
        self.assertEqual(pager.count('class="conduit-pager-link'), 2)
        self.assertLess(pager.index('rel="prev"'), pager.index("is-disabled"))

    def test_pager_sits_in_the_profile_head_before_back(self):
        page, _ = self._pager("C2")
        # The profile head ends where the drawing starts.
        head = re.search(r'<div class="panelhead">(.*?)<div id="conduit-diagram">', page, re.DOTALL).group(1)

        self.assertIn('<nav class="conduit-pager"', head)
        self.assertLess(head.index("</h1>"), head.index('<nav class="conduit-pager"'))
        self.assertLess(head.index('<nav class="conduit-pager"'), head.index('class="panel-back"'))

    def test_the_fixed_edge_handles_are_gone(self):
        page, pager = self._pager("C2")

        for gone in ("navigation-arrows", "nav-arrow", "navArrow", "nav-tooltip", "nav-icon"):
            self.assertNotIn(gone, page, gone)
        # The pager must not reuse the site-nav class: its padding hid the old chevrons.
        self.assertNotIn("nav-link", pager)

    def test_keyboard_shortcut_targets_the_pager_links(self):
        path = Path(__file__).resolve().parent.parent.parent / "static" / "js" / "main.js"
        script = path.read_text(encoding="utf-8")

        self.assertIn("document.querySelector('.conduit-pager')", script)
        self.assertIn('a[rel="${rel}"]', script)
        self.assertNotIn("navArrow", script)


class PropertyDetailsTests(TestCase):
    """The tabbed details card: three panels, every old property row, the numeric hook."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="table@example.com")
        cls.session = make_session(cls.user, n_conduits=1, n_nodes=2, n_subcatchments=1)

    def setUp(self):
        self.client.force_login(self.user)

    def _get(self):
        return self.client.get(reverse("sa:conduit_detail", args=[self.session.id, "C1"]))

    def test_details_card_carries_the_hook_class_and_right_aligned_numbers(self):
        response = self._get()

        self.assertContains(response, 'class="card panel-details" id="conduit-details"')
        # Every numeric property opts into the right-aligned, monospaced column.
        # Counted inside the details card only: the metric bar and the panels around it
        # must not use the class.
        html = response.content.decode()
        details = re.search(r'id="conduit-details"(.*)', html, re.S).group(1)
        self.assertEqual(details.count('class="prop-num"'), 12)
        self.assertEqual(html.count('class="prop-num"'), 12)

    def test_every_property_of_the_old_table_is_still_rendered(self):
        response = self._get()

        for label in PROPERTY_LABELS:
            with self.subTest(label=label):
                self.assertContains(response, f"<span>{label}</span>")

    def test_all_three_detail_panels_are_server_rendered(self):
        response = self._get()
        html = response.content.decode()

        self.assertEqual(html.count('role="tab"'), 3)
        self.assertEqual(html.count('role="tabpanel"'), 3)
        for name in ("geometry", "hydraulics", "catchments"):
            with self.subTest(panel=name):
                self.assertContains(response, f'id="tabpanel-{name}"')
                self.assertContains(response, f'aria-controls="tabpanel-{name}"')
        # Without JavaScript nothing is hidden: the three panels read as one document.
        self.assertNotContains(response, 'panel-tabpanel" hidden')


class ConduitPanelTests(TestCase):
    """The conduit panel: flow chain with both nodes, catchments of the inlet node, notes."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="panel@example.com")
        # C1 runs N1 -> N2 and S1 drains to N1 (factories.make_session wiring).
        cls.session = make_session(cls.user, n_conduits=1, n_nodes=2, n_subcatchments=1)

    def setUp(self):
        self.client.force_login(self.user)

    def _get(self, name="C1"):
        return self.client.get(reverse("sa:conduit_detail", args=[self.session.id, name]))

    def test_flow_chain_renders_both_node_cards_with_role_anchors(self):
        response = self._get()

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="node-inlet"')
        self.assertContains(response, 'id="node-outlet"')
        # Node invert 117.0 + max depth 3.0 = ground 120.0, straight from the factory —
        # asserted inside the inlet card, because "120.00 m" also appears in the property table.
        html = response.content.decode()
        inlet_card = re.search(r'id="node-inlet"(.*?)</dl>', html, re.S).group(1)
        self.assertIn("<dd>117.00 m</dd>", inlet_card)
        self.assertIn("<dd>3.00 m</dd>", inlet_card)
        self.assertIn("<dd>120.00 m</dd>", inlet_card)
        self.assertIn("<dt>Manhole depth</dt>", inlet_card)
        self.assertNotIn("<dt>Max depth</dt>", inlet_card)
        self.assertNotContains(response, "connections-card")

    def test_flooding_row_renders_missing_none_and_flooded_states(self):
        cases = (
            ("missing", None, None, "<dd>—</dd>", False),
            ("none", 0.0, 0.0, "<dd>none</dd>", False),
            ("flooded", 0.42, 0.013, "<dd>0.42 h · 0.013 × 10⁶ l</dd>", True),
            # SWMM reports Hours Flooded with two decimals: a short flood is 0.00 h.
            ("short", 0.0, 0.003, "<dd>&lt; 0.01 h · 0.003 × 10⁶ l</dd>", True),
        )
        for index, (name, hours, volume, expected, flooded) in enumerate(cases, start=10):
            with self.subTest(state=name):
                node_name = f"N{index}"
                make_node(
                    self.session,
                    node_name=node_name,
                    max_water_depth=1.25,
                    hours_flooded=hours,
                    flood_volume=volume,
                )
                make_conduit(
                    self.session,
                    conduit_name=f"C-{name}",
                    inlet_node=node_name,
                    outlet_node="N-unknown",
                    subcatchment="-",
                )

                response = self._get(f"C-{name}")
                inlet_card = re.search(r'id="node-inlet"(.*?)</dl>', response.content.decode(), re.S).group(1)

                self.assertIn("<dt>Max water depth</dt>", inlet_card)
                self.assertIn("<dd>1.25 m</dd>", inlet_card)
                self.assertIn("<dt>Flooding</dt>", inlet_card)
                self.assertIn(expected, inlet_card)
                self.assertEqual('class="node-card is-flooded"' in response.content.decode(), flooded)
                self.assertEqual("Flooded</span>" in inlet_card, flooded)

    def test_node_cards_live_inside_the_open_node_details(self):
        """The node cards are the only content under the drawing, so the disclosure is open."""
        response = self._get()
        html = response.content.decode()

        self.assertContains(response, "<summary>Inlet and outlet nodes</summary>")
        block = re.search(r'<details class="chain-details" open>(.*?)</details>', html, re.S)
        self.assertIsNotNone(block, 'the node cards are not wrapped in an open <details class="chain-details">')
        self.assertIn('id="node-inlet"', block.group(1))
        self.assertIn('id="node-outlet"', block.group(1))
        # The chain bar it replaced is gone; the reader can still collapse the cards.
        self.assertNotContains(response, 'class="flow-chain"')

    def test_catchment_rows_carry_pk_anchors_only_in_the_catchment_section(self):
        s1 = self.session.subcatchments.get(subcatchment_name="S1")
        response = self._get()
        html = response.content.decode()

        self.assertContains(response, f'id="catchment-{s1.pk}"')
        self.assertEqual(html.count('id="catchment-'), 1)
        self.assertContains(response, "Catchments connected directly to inlet node N1")

    def test_model_note_without_differs_when_the_single_direct_catchment_is_the_model_one(self):
        response = self._get()

        self.assertContains(response, "Model feature (Subcatchment (model)): <strong>S1</strong>")
        self.assertNotContains(response, "differs from the direct catchments above")

    def test_model_note_flags_differs_with_two_direct_catchments(self):
        make_subcatchment(self.session, subcatchment_name="S9", outlet="N1")
        response = self._get()
        html = response.content.decode()

        self.assertEqual(html.count('id="catchment-'), 2)
        # S1 is one of the two rows above, so the note must not claim it differs from them.
        self.assertContains(response, "the pipeline kept only this one of the direct catchments above")
        self.assertNotContains(response, "differs from the direct catchments above")

    def test_model_note_flags_differs_when_the_model_catchment_is_not_a_direct_one(self):
        make_conduit(self.session, conduit_name="C-down", inlet_node="N1", outlet_node="N2", subcatchment="S-far")
        response = self._get("C-down")

        self.assertContains(response, "Model feature (Subcatchment (model)): <strong>S-far</strong>")
        self.assertContains(response, "differs from the direct catchments above")

    def test_empty_states_for_a_conduit_between_unknown_nodes(self):
        make_conduit(self.session, conduit_name="C-lone", inlet_node="N8", outlet_node="N9", subcatchment="-")
        response = self._get("C-lone")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "node not in session data", count=2)
        self.assertContains(response, "no catchment connected", count=3)  # two node cards + the section
        self.assertNotContains(response, "Model feature")
        self.assertContains(response, "<dd>none</dd>", count=4)  # incoming/outgoing on both cards

    def test_zero_invert_renders_as_zero_not_as_missing(self):
        make_node(self.session, node_name="N0", invert_elevation=0.0, max_depth=2.5)
        make_conduit(self.session, conduit_name="C-zero", inlet_node="N0", outlet_node="N1", subcatchment="-")
        response = self._get("C-zero")

        self.assertContains(response, "0.00 m")
        self.assertContains(response, "2.50 m")

    def test_incoming_and_outgoing_conduits_are_links_to_their_panels(self):
        make_conduit(self.session, conduit_name="C-up", inlet_node="N0", outlet_node="N1", subcatchment="-")
        response = self._get()

        self.assertContains(response, reverse("sa:conduit_detail", args=[self.session.id, "C-up"]))
        self.assertContains(response, 'class="chain-link">C-up</a>')

    def test_units_and_footnote(self):
        response = self._get()

        self.assertContains(response, "Total runoff [10⁶ l]")
        self.assertContains(response, "Peak runoff [m³/s]")
        self.assertContains(response, "the app accepts FLOW_UNITS CMS only")
        self.assertContains(response, "Max Flow")

    def test_no_template_comment_leaks_into_the_page(self):
        # Django's {# #} is single-line only: a comment wrapped over several lines is not
        # a comment at all and renders as text (it did, inside the technical analysis rows).
        response = self._get()

        self.assertContains(response, "Design rules")
        self.assertNotContains(response, "{#")
        self.assertNotContains(response, "#}")

    def test_diagram_json_carries_node_inverts(self):
        response = self._get()

        self.assertContains(response, 'id="conduit-data"')
        self.assertContains(response, '"inletInvert": 117.0')
        self.assertContains(response, '"outletMaxDepth": 3.0')

    def test_profile_replaces_the_zoomable_svgjs_diagram(self):
        response = self._get()

        self.assertContains(response, "js/conduit-panel.js")
        self.assertContains(response, "Longitudinal profile")
        self.assertContains(response, 'class="conduit-profile"')
        self.assertNotContains(response, "svg.js")
        self.assertNotContains(response, "reset-zoom")
        self.assertNotContains(response, "conduit-diagram.js")

    def test_rule_label_row_survives_for_a_conduit_that_has_one(self):
        # K1 is in WROCLAW_RULE_LABELS, so the panel adds the ground-truth row that the
        # factory conduits (unknown names) never trigger.
        make_conduit(self.session, conduit_name="K1", inlet_node="N1", outlet_node="N2", subcatchment="-")
        response = self._get("K1")

        self.assertContains(response, "<span>Rule Label (Ground Truth)</span>")
        self.assertContains(response, "depth_increase")

    def test_design_rules_list_only_the_changes_the_stored_flags_ask_for(self):
        cases = {
            "C-none": ({}, []),
            "C-inc-dia": ({"increase_dia": 1, "geom1": 0.3, "min_diameter": 0.4}, [("Diameter", "0.30 → 0.40 m")]),
            "C-red-dia": ({"reduce_dia": 1}, [("Diameter", "could be reduced")]),
            "C-inc-slope": (
                {"increase_slope": 1, "slope_per_mile": 0.28, "min_required_slope": 2.04},
                [("Slope", "0.28 → 2.04 ‰")],
            ),
            "C-red-slope": (
                {"reduce_slope": 1, "slope_per_mile": 180.0, "max_allowable_slope": 134.56},
                [("Slope", "180.00 → 134.56 ‰")],
            ),
            # A dry pipe whose stored minimum lies above its maximum: no target proposed.
            "C-dry": (DRY_CONDUIT, [("Slope", "no feasible value — no flow in the run")]),
        }
        for name, (fields, expected) in cases.items():
            with self.subTest(conduit=name):
                make_conduit(self.session, conduit_name=name, inlet_node="N1", outlet_node="N2", **fields)
                page = self._get(name).content.decode()
                block = re.search(r'<div class="design-rules">(.*?)</div>\s*</div>\s*</section>', page, re.DOTALL).group(1)
                rows = re.findall(r'<div class="design-rule"><span>([^<]+)</span><b>([^<]+)</b></div>', block)
                self.assertEqual(rows, expected)
                self.assertEqual("No change required by the design rules" in block, not expected)
                for gone in ("technical-row", "Required min. diameter", "Allowed slope range", "Action needed"):
                    self.assertNotIn(gone, page)

    def test_prediction_card_lists_all_nine_classes_most_confident_first(self):
        response = self._get()
        html = response.content.decode()

        names = re.findall(r'<span class="prediction-name">([^<]+)</span>', html)
        # C1 is recommended "valid", so that class heads the card and the other eight
        # are listed by descending confidence (0.02 ×3, 0.01 ×3, 0.0 last).
        self.assertEqual(len(names), 8)
        # The recommended class is named by its label, above the eight others.
        self.assertContains(response, '<div class="selected-value">No changes needed</div>')
        self.assertEqual(names[:3], ["Increase diameter", "Reduce diameter", "Increase slope"])
        self.assertEqual(names[-1], "Increase depth")
        # Open by default; the summary names the classes instead of an action.
        self.assertContains(response, '<details class="prediction-rest" open>')
        self.assertContains(response, "Remaining 6 classes")

    def test_diagram_json_escapes_hostile_element_names(self):
        evil = "</script><script>alert(1)</script>"
        make_node(self.session, node_name=evil)
        make_conduit(self.session, conduit_name="C-evil", inlet_node=evil, outlet_node="N2", subcatchment="-")
        response = self._get("C-evil")

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "<script>alert(1)</script>")
        self.assertContains(response, "\\u003Cscript\\u003Ealert(1)")

    def test_panel_query_budget(self):
        url = reverse("sa:conduit_detail", args=[self.session.id, "C1"])
        with CaptureQueriesContext(connection) as ctx:
            response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertLessEqual(len(ctx.captured_queries), 10, "\n".join(q["sql"] for q in ctx.captured_queries))


class ProfileScriptFloodingContractTests(TestCase):
    """``conduit-panel.js`` must take the flooding verdict from the server, not from the hours.

    A source-level guard of the contract between ``diagram_data`` and the script that reads
    it; the drawing's geometry is checked by ``harness_conduit_profile.js`` below.
    """

    @staticmethod
    def _script() -> str:
        path = Path(__file__).resolve().parent.parent.parent / "static" / "js" / "conduit-panel.js"
        return path.read_text(encoding="utf-8")

    def test_script_reads_the_server_side_flooded_flags(self):
        script = self._script()

        self.assertIn("d.inletFlooded === true", script)
        self.assertIn("d.outletFlooded === true", script)

    def test_script_does_not_re_derive_flooding_from_the_rounded_hours(self):
        script = self._script()

        self.assertNotIn("d.inletFloodHours > 0", script)
        self.assertNotIn("d.outletFloodHours > 0", script)


class ProfileGeometryHarnessTests(TestCase):
    """Run ``harness_conduit_profile.js``: the joins of the pipe and the manholes, from Node."""

    def test_conduit_profile_harness(self):
        """Skipped only when Node is missing - never when it fails."""
        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("Node.js is not installed; skipping the conduit profile harness")

        harness_path = Path(__file__).resolve().parent / "harness_conduit_profile.js"
        self.assertTrue(harness_path.exists(), f"harness_conduit_profile.js not found at {harness_path}")

        result = subprocess.run(
            [node_bin, str(harness_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"the conduit profile harness failed with exit code {result.returncode}:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.assertIn("ALL CONDUIT PROFILE CASES VERIFIED (11 / 11 PASSED)", result.stdout)
