"""Empirical automated test suite for Results V2 Visual & Interaction Acceptance Criteria.

Verifies:
1. Tabbed workspace (Conduits, Nodes, Subcatchments with dynamic count badges)
2. 3 summary KPI cards with real session metrics
3. Search and filtering across columns
4. Sortable column headers (ascending/descending natural & numeric)
5. Column visibility presets (Overview, Technical/Hydraulics, All columns) & pinned columns
6. Inline validation expansion rows (7-check grid with actual values vs thresholds)
7. CSV export downloads correct data with UTF-8 BOM
8. Clicking conduit name navigates to conduit_detail URL
9. Zero regressions across existing pages (form, history, conduit panel, about, home)
10. Scoped CSS design system isolation (100% rules scoped under #storm-results-v2)
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from django.test import TestCase
from django.urls import reverse

from .factories import make_conduit, make_geometry, make_node, make_session, make_subcatchment, make_user


class ResultsV2VisualTemplateContractTests(TestCase):
    """Verify HTML template structure, component presence, and context contracts."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="visual_tester@example.com")
        cls.session = make_session(cls.user, n_conduits=0, n_nodes=0, n_subcatchments=0)

        # 3 Conduits: C1 (valid), C2 (changes), C3 (changes)
        cls.c1 = make_conduit(cls.session, conduit_name="C1", recommendation="valid", val_max_fill=1)
        cls.c2 = make_conduit(cls.session, conduit_name="C2", recommendation="diameter_increase", val_max_fill=0, val_coverage=0)
        cls.c3 = make_conduit(cls.session, conduit_name="C3", recommendation="slope_increase", val_min_v=0)

        # 3 Nodes: N1 (dry), N2 (flooded), N3 (dry)
        cls.n1 = make_node(cls.session, node_name="N1", flood_volume=0.0, hours_flooded=0.0)
        cls.n2 = make_node(cls.session, node_name="N2", flood_volume=3.5, hours_flooded=1.2)
        cls.n3 = make_node(cls.session, node_name="N3", flood_volume=0.0, hours_flooded=0.0)

        # 2 Subcatchments
        cls.s1 = make_subcatchment(cls.session, subcatchment_name="S1", outlet="N1", category="residential")
        cls.s2 = make_subcatchment(cls.session, subcatchment_name="S2", outlet="N2", category="industrial")

    def setUp(self):
        self.client.force_login(self.user)
        self.response = self.client.get(reverse("sa:analysis_results", args=[self.session.id]))
        self.html = self.response.content.decode("utf-8")

    def test_http_200_and_root_container_rendered(self):
        """Page returns HTTP 200 and root #storm-results-v2 container is present with attributes."""
        self.assertEqual(self.response.status_code, 200)
        self.assertIn('<div id="storm-results-v2"', self.html)
        self.assertIn('data-density="compact"', self.html)
        self.assertIn('data-stripe="false"', self.html)

    def test_scoped_css_and_deferred_js_included(self):
        """results-v2.css and results-v2.js are linked with appropriate scopes and defer attributes."""
        self.assertIn("results-v2.css", self.html)
        self.assertIn("results-v2.js", self.html)
        self.assertIn("defer", self.html)

    def test_info_tab_renders_session_metadata(self):
        """Info tab and panel display simulation metadata (file, date, frost zone)."""
        self.assertIn('id="r-tab-info"', self.html)
        self.assertIn('id="r-info-panel"', self.html)
        self.assertIn("Model file", self.html)
        self.assertIn("Simulation date", self.html)
        self.assertIn("Frost zone", self.html)

    def test_tabbed_workspace_renders_tabs_and_badges(self):
        """Conduits, Nodes, Subcatchments, Info tabs, badges, and header action buttons render correctly."""
        self.assertIn('id="r-tab-conduits"', self.html)
        self.assertIn('id="r-count-conduits">3</span>', self.html)

        self.assertIn('id="r-tab-nodes"', self.html)
        self.assertIn('id="r-count-nodes">3</span>', self.html)

        self.assertIn('id="r-tab-catchments"', self.html)
        self.assertIn('id="r-count-catchments">2</span>', self.html)

        # Header action buttons integrated in tabs
        self.assertIn("New Analysis", self.html)
        self.assertIn("History", self.html)
        self.assertIn('class="r-tabs-actions"', self.html)

        # Breadcrumbs bar removed
        self.assertNotIn('aria-label="breadcrumb"', self.html)

    def test_action_toolbar_and_filter_controls_present(self):
        """Search input, filter dropdown, failed checkbox, columns button, and CSV button present."""
        self.assertIn('id="r-search"', self.html)
        self.assertIn('id="r-filter-select"', self.html)
        self.assertIn('id="r-failed"', self.html)
        self.assertIn('id="r-columns"', self.html)
        self.assertIn('id="r-export"', self.html)
        self.assertIn('id="r-clear-filters"', self.html)

    def test_column_selection_drawer_and_presets_present(self):
        """Column visibility drawer with overview, technical, and all presets present."""
        self.assertIn('id="r-column-selection"', self.html)
        self.assertIn('data-preset="overview"', self.html)
        self.assertIn('data-preset="technical"', self.html)
        self.assertIn('data-preset="all"', self.html)
        self.assertIn('id="r-close-columns"', self.html)

    def test_table_frame_and_footer_present(self):
        """Table frame, scrollable overflow container, table element, and row count footer present."""
        self.assertIn('id="r-tableframe"', self.html)
        self.assertIn('id="r-overflow"', self.html)
        self.assertIn('id="r-table"', self.html)
        self.assertIn('id="r-row-count"', self.html)
        self.assertIn('id="r-foot-columns"', self.html)

    def test_embedded_json_script_payload(self):
        """<script id="results-v2-data" type="application/json"> contains complete serializable data."""
        self.assertIn('<script id="results-v2-data" type="application/json">', self.html)
        v2_data = self.response.context["results_v2_data"]
        self.assertEqual(len(v2_data["conduits"]), 3)
        self.assertEqual(len(v2_data["nodes"]), 3)
        self.assertEqual(len(v2_data["catchments"]), 2)
        self.assertEqual(v2_data["summary"]["conduits_changes_count"], 2)
        self.assertEqual(v2_data["summary"]["nodes_flooding_count"], 1)

    def test_noscript_accessible_fallback(self):
        """Accessible <noscript> fallback provides a JS-disabled message with history link."""
        self.assertIn("<noscript>", self.html)
        self.assertIn("JavaScript is required", self.html)
        history_url = reverse("sa:history")
        self.assertIn(f'href="{history_url}"', self.html)


class ResultsV2ExistingPagesZeroRegressionTests(TestCase):
    """Verify all existing pages continue functioning with HTTP 200 and no style/script leakage."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="regression_tester@example.com")
        cls.session = make_session(cls.user, n_conduits=1, n_nodes=1, n_subcatchments=1)
        cls.conduit = cls.session.conduits.first()

    def setUp(self):
        self.client.force_login(self.user)

    def test_index_page_returns_200(self):
        """Home/landing page returns 200 and does not leak results-v2 assets."""
        response = self.client.get(reverse("sa:index"))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertNotIn("results-v2.css", content)
        self.assertNotIn("results-v2.js", content)

    def test_analysis_form_page_returns_200(self):
        """Analysis creation form returns 200 and preserves existing form controls."""
        response = self.client.get(reverse("sa:analysis"))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertNotIn("results-v2.css", content)
        self.assertNotIn("results-v2.js", content)

    def test_history_page_returns_200(self):
        """History session log returns 200 and lists user sessions."""
        response = self.client.get(reverse("sa:history"))
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn(str(self.session.id), content)
        self.assertNotIn("results-v2.css", content)
        self.assertNotIn("results-v2.js", content)

    def test_conduit_detail_panel_page_returns_200(self):
        """Existing conduit detail panel returns 200 and displays conduit inspection."""
        url = reverse("sa:conduit_detail", args=[self.session.id, self.conduit.conduit_name])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn(self.conduit.conduit_name, content)

    def test_about_page_returns_200(self):
        """About page returns 200 without regressions."""
        response = self.client.get(reverse("sa:about"))
        self.assertEqual(response.status_code, 200)


class ResultsV2CSSIsolationTests(TestCase):
    """Verify 100% of CSS rules in results-v2.css are strictly scoped under #storm-results-v2."""

    def test_all_css_rules_scoped_under_root_id(self):
        """Every CSS rule selector must start with #storm-results-v2 to prevent Bootstrap collisions."""
        css_path = Path(__file__).resolve().parent.parent.parent / "static" / "css" / "results-v2.css"
        self.assertTrue(css_path.exists(), f"results-v2.css not found at {css_path}")

        lines = css_path.read_text(encoding="utf-8").splitlines()
        in_comment = False
        unscoped_selectors = []

        for line_idx, line in enumerate(lines, 1):
            stripped = line.strip()
            if "/*" in stripped and "*/" in stripped:
                continue
            if "/*" in stripped:
                in_comment = True
                continue
            if "*/" in stripped:
                in_comment = False
                continue
            if in_comment or not stripped:
                continue
            # Skip @media, @keyframes, closing braces, keyframe steps
            if (
                stripped.startswith("@")
                or stripped.startswith("}")
                or stripped.startswith("from")
                or stripped.startswith("to")
                or stripped.endswith(";")
                or stripped.startswith("/*")
                or stripped.startswith("*")
            ):
                continue

            if "{" in stripped:
                selector = stripped.split("{")[0].strip()
                # Split multiple selectors separated by comma
                for part in selector.split(","):
                    p = part.strip()
                    if p and not p.startswith("#storm-results-v2"):
                        unscoped_selectors.append((line_idx, p))

        self.assertEqual(
            len(unscoped_selectors),
            0,
            f"Found {len(unscoped_selectors)} unscoped CSS selectors in results-v2.css:\n"
            + "\n".join(f"Line {idx}: {sel}" for idx, sel in unscoped_selectors[:10]),
        )


class ResultsV2JavaScriptEmpiricalHarnessTests(TestCase):
    """Execute the Node.js empirical interaction harness verifying all 8 interactive criteria."""

    def test_node_empirical_interaction_harness(self):
        """Run Node.js harness testing tabs, summary cards, search, sort, presets, validation, CSV export."""
        import shutil

        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("Node.js is not installed; skipping empirical JS harness")

        harness_path = Path(__file__).resolve().parent / "harness_results_v2.js"
        self.assertTrue(harness_path.exists(), f"harness_results_v2.js not found at {harness_path}")

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
            f"Node.js empirical test harness failed with exit code {result.returncode}:\n"
            + f"STDOUT:\n{result.stdout}\n"
            + f"STDERR:\n{result.stderr}",
        )
        self.assertIn("ALL EMPIRICAL ACCEPTANCE CRITERIA VERIFIED (8 / 8 PASSED)", result.stdout)


class ResultsV2NetworkMapTemplateTests(TestCase):
    """The map tab, the map panel and the second JSON channel, both with and without geometry."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="map_visual_tester@example.com")
        cls.mapped = make_session(cls.user, n_conduits=1, n_nodes=1, n_subcatchments=1)
        make_geometry(cls.mapped)
        cls.unmapped = make_session(cls.user, n_conduits=1, n_nodes=1, n_subcatchments=1)

    def setUp(self):
        self.client.force_login(self.user)

    def _html(self, session) -> str:
        response = self.client.get(reverse("sa:analysis_results", args=[session.id]))
        self.assertEqual(response.status_code, 200)
        return response.content.decode("utf-8")

    def test_map_payload_is_json_script(self):
        """The geometry reaches the browser through json_script and through nothing else."""
        html = self._html(self.mapped)
        self.assertIn('<script id="network-map-data" type="application/json">', html)
        self.assertIn('"NM1"', html)

        # No data-* attribute anywhere on the page carries the payload: an attribute is
        # a second, unescaped channel and the element names are user input.
        for attribute in re.findall(r'data-[a-zA-Z-]+="([^"]*)"', html):
            self.assertNotIn("NM1", attribute)
            self.assertNotIn("bbox", attribute)

    def test_css_scoped_under_root(self):
        """Every selector of network-map.css starts with #storm-results-v2.

        An at-rule prelude is not a selector: ``@media (max-width: 520px) {`` opens a
        block whose *contents* are scoped, and the file gained one such block when the
        mode row had to be tightened for 390x844 (end of network-map.css).  Only a
        ``@media`` prelude is skipped, not any ``@`` line: the skip exists for the one
        at-rule this file has, and a broader one would wave through ``@keyframes`` or
        ``@supports`` blocks nobody has reviewed (both land in ``unscoped`` today,
        measured).  The rules inside the block are read on their own lines and still
        have to carry the root id.

        The skip is also narrowed to a *bare* prelude - a line that ends at the brace
        it opens.  Written loosely it read the whole line, so a single-line
        ``@media (max-width: 520px) { .sneaky { color: blue } }`` hid its own contents
        from the guard: the inner selector never reached the check.  Measured on a
        synthetic stylesheet, that line passed before this narrowing and fails after it.
        """
        css_path = Path(__file__).resolve().parent.parent.parent / "static" / "css" / "network-map.css"
        self.assertTrue(css_path.exists(), f"network-map.css not found at {css_path}")

        unscoped = []
        in_comment = False
        for line_idx, line in enumerate(css_path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if "/*" in stripped and "*/" in stripped:
                continue
            if "/*" in stripped:
                in_comment = True
                continue
            if "*/" in stripped:
                in_comment = False
                continue
            if in_comment or not stripped or "{" not in stripped:
                continue
            prelude = stripped.split("{")[0].strip()
            if prelude.startswith("@media") and stripped.endswith("{"):
                continue
            for part in prelude.split(","):
                selector = part.strip()
                if selector and not selector.startswith("#storm-results-v2"):
                    unscoped.append((line_idx, selector))

        self.assertEqual(unscoped, [], f"unscoped selectors in network-map.css: {unscoped}")

    def test_node_symbol_stroke_does_not_scale(self):
        """The node symbol carries vector-effect itself: the property is not inherited.

        A node is ``<g class="nm-node" transform="... scale(r/pxPerUnit)">`` with the symbol
        inside, and fill/stroke/stroke-width reach that symbol by inheritance.  ``vector-effect``
        does not inherit, so a rule placed only on the group leaves the symbol scaling its
        outline with the group and drawing it 2-3.5 px wide instead of the contracted 1 px.
        """
        css_path = Path(__file__).resolve().parent.parent.parent / "static" / "css" / "network-map.css"
        css = re.sub(r"/\*.*?\*/", "", css_path.read_text(encoding="utf-8"), flags=re.DOTALL)

        guarded = False
        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            if "vector-effect" not in body or "non-scaling-stroke" not in body:
                continue
            for selector in selectors.split(","):
                # A descendant of .nm-node, not the group itself: ".nm-node > *", ".nm-node circle".
                if re.search(r"\.nm-node\s*(>|\s)\s*\S", selector.strip()):
                    guarded = True

        self.assertTrue(
            guarded,
            "network-map.css must give the node symbol (a child of g.nm-node) "
            "vector-effect: non-scaling-stroke; the rule on the group alone does not inherit",
        )

    def test_versioned_static_used_for_map_assets(self):
        """Both map assets are linked through versioned_static, so a change busts the cache."""
        html = self._html(self.mapped)
        for asset in ("css/network-map.css", "js/network-map.js"):
            self.assertRegex(html, re.escape(asset) + r"\?v=\d+")

    def test_map_tab_is_selected_on_load(self):
        """With geometry stored the server hands over a page whose map tab is already up."""
        html = self._html(self.mapped)
        self.assertIn('id="r-tab-map" data-tab="map" role="tab" aria-controls="r-map-panel" aria-selected="true"', html)
        self.assertIn(
            'id="r-tab-conduits" data-tab="conduits" role="tab" aria-controls="r-table-panel" aria-selected="false"', html
        )
        self.assertIn('<section id="r-table-panel" role="tabpanel" aria-labelledby="r-tab-conduits" hidden>', html)
        self.assertIn('<section id="r-map-panel" role="tabpanel" aria-labelledby="r-tab-map">', html)
        self.assertIn('id="nm-canvas"', html)

    def test_falls_back_to_conduits_when_map_unavailable(self):
        """Without geometry the page looks exactly as it did before the map existed."""
        html = self._html(self.unmapped)
        self.assertIn('id="r-tab-map" data-tab="map" role="tab" aria-controls="r-map-panel" aria-selected="false"', html)
        self.assertIn(
            'id="r-tab-conduits" data-tab="conduits" role="tab" aria-controls="r-table-panel" aria-selected="true"', html
        )
        self.assertIn('<section id="r-table-panel" role="tabpanel" aria-labelledby="r-tab-conduits">', html)
        self.assertIn('<section id="r-map-panel" role="tabpanel" aria-labelledby="r-tab-map" hidden>', html)
        self.assertIn("No network drawing was stored for this session.", html)
        # The tab stays clickable, or the message inside the panel is unreachable.
        tab = html[html.index('id="r-tab-map"') : html.index('id="r-tab-conduits"')]
        self.assertNotIn("disabled", tab)
        self.assertNotIn("nm-canvas", html)

    def test_map_tab_registered(self):
        """results-v2.js registers 'map' first in TABS and keeps it out of ``active``."""
        js_path = Path(__file__).resolve().parent.parent.parent / "static" / "js" / "results-v2.js"
        source = js_path.read_text(encoding="utf-8")

        self.assertIn("const TABS = ['map', 'conduits', 'nodes', 'catchments', 'info'];", source)
        self.assertIn("let currentTab", source)
        self.assertIn("function applyTabState(", source)

        # ``active`` is a table schema key; 'map' has no schema and assigning it there
        # is the mistake this assertion exists to catch.
        self.assertNotRegex(source, r"active\s*=\s*mapAvailable")
        self.assertNotRegex(source, r"active\s*=\s*'map'")
        self.assertIn("let active = 'conduits';", source)

        template_path = Path(__file__).resolve().parent.parent / "templates" / "sa" / "analysis_results.html"
        template = template_path.read_text(encoding="utf-8")
        self.assertIn('id="r-tab-map" data-tab="map"', template)

    # -- stage 4: the schematic layout has to say that it is not model geometry ------
    #
    # Both cases are static checks of the source, and deliberately so: the branch lives
    # in network-map.js, the Django runner executes no JavaScript, and a Python test
    # that re-implemented the branch would prove only that the copy agrees with itself.

    def _map_source(self, name: str) -> str:
        static = Path(__file__).resolve().parent.parent.parent / "static"
        folder = "js" if name.endswith(".js") else "css"
        return (static / folder / name).read_text(encoding="utf-8")

    def _auto_layout_branch(self, source: str) -> str:
        """The body of ``if (data.layout === 'auto') { ... }``, found by brace matching."""
        match = re.search(r"data\.layout\s*===\s*['\"]auto['\"]\s*\)\s*\{", source)
        self.assertIsNotNone(match, "network-map.js has no branch guarded by layout === 'auto'")
        start = match.end() - 1
        depth = 0
        for index in range(start, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    return source[start : index + 1]
        self.fail("unbalanced braces in the layout branch of network-map.js")

    def test_schematic_badge_rendered_for_auto_layout(self):
        """``layout == "auto"`` builds the badge, with that exact sentence and a CSS rule."""
        source = self._map_source("network-map.js")

        self.assertRegex(source, r"data\.layout\s*===\s*['\"]auto['\"]")

        # The sentences are asserted against the branch and not against the whole
        # module: a second, unconditional line carrying the same text would satisfy a
        # module-wide search while breaking the "inp" path, which must show nothing.
        branch = self._auto_layout_branch(source)
        self.assertIn("nm-badge", branch)
        self.assertIn("Schematic layout — not model geometry", branch)
        self.assertIn("nm-legend-note", branch)
        self.assertIn(
            "Positions are computed from network topology; distances and directions are not to scale.",
            branch,
        )

        # Permanent means permanent: nothing inside the branch hides the mark again,
        # and a screen reader gets it as part of the panel description.
        self.assertNotIn("aria-hidden", branch)
        self.assertIn("aria-describedby", branch)

        css = self._map_source("network-map.css")
        self.assertRegex(css, r"#storm-results-v2\s+\.nm-badge\s*\{")
        self.assertRegex(css, r"#storm-results-v2\s+\.nm-legend-note\s*\{")

    def test_no_badge_markup_for_inp_layout(self):
        """In the ``"inp"`` path no branch creates the badge: it is absent, not hidden."""
        source = self._map_source("network-map.js")

        # One single mention in the whole module, and it sits after the test of the
        # layout: no second branch, no template fallback, nothing to hide with CSS.
        branch = self._auto_layout_branch(source)
        self.assertEqual(source.count("nm-badge"), 1, "nm-badge must be created in exactly one place")
        self.assertIn("nm-badge", branch)
        self.assertEqual(source.count("nm-legend-note"), 1, "nm-legend-note must be created in exactly one place")
        self.assertIn("nm-legend-note", branch)

        template_path = Path(__file__).resolve().parent.parent / "templates" / "sa" / "analysis_results.html"
        template = template_path.read_text(encoding="utf-8")
        self.assertNotIn("nm-badge", template)
        self.assertNotIn("nm-legend-note", template)

        # ``make_geometry`` stores a payload with layout "inp", so the served page must
        # not carry the sentence either.
        html = self._html(self.mapped)
        self.assertNotIn("nm-badge", html)
        self.assertNotIn("Schematic layout", html)
        self.assertNotIn("nm-legend-note", html)
        self.assertNotIn("Positions are computed from network topology", html)

    def test_map_panel_shrinks_to_fit_the_workspace(self):
        """The panel is a flex column that may shrink, so the note below the drawing is on screen.

        ``.r-workspace`` clips with ``overflow: hidden`` and the page does not scroll,
        so a map panel keeping the intrinsic height of ``.nm-canvas`` pushes whatever
        follows the canvas out of sight: measured at 390x844 the note sat at y=1038 in
        an 844 px viewport with ``document.scrollHeight == 844``.  The browser gate
        measures the outcome; this test guards the three declarations it rests on.

        Since stage E1 the panel also carries an inset of its own, and the height for it
        is taken from the **canvas** - the flex item that shrinks - never from the note
        under the drawing, which is ``flex: none``.
        """
        workspace = self._map_source("results-v2.css")
        panel_rule = re.search(r"#storm-results-v2 #r-map-panel \{(.+?)\}", workspace, re.DOTALL)
        self.assertIsNotNone(panel_rule, "results-v2.css has no layout rule for #r-map-panel")
        body = panel_rule.group(1)
        for declaration in ("display: flex", "flex-direction: column", "flex: 1", "min-height: 0"):
            self.assertIn(declaration, body)

        css = self._map_source("network-map.css")
        canvas_rule = re.search(r"#storm-results-v2 \.nm-canvas \{(.+?)\}", css, re.DOTALL)
        self.assertIsNotNone(canvas_rule, "network-map.css has no rule for .nm-canvas")
        # Without min-height: 0 a flex item never shrinks below its own height, and the
        # canvas asks for clamp(320px, 100vh - 300px, 900px).
        self.assertIn("min-height: 0", canvas_rule.group(1))
        self.assertIn("flex: 0 1 auto", canvas_rule.group(1))

    # -- stage E1 (plan24, position 1): the inset of the map panel ------------------
    #
    # Nothing below writes a pixel value down.  The point of the stage is *parity* with
    # the furniture of the table view, so parity is what is measured: a literal "20px"
    # here would keep passing after someone changed the toolbar and left the map behind.

    @staticmethod
    def _css_no_comments(source: str) -> str:
        """The stylesheet without its comments.

        Every helper below parses the file with braces, and a comment in this project
        routinely quotes a selector in order to explain it - the rule for the inset of
        the map panel does exactly that.  A quoted selector is not a rule, so it is
        removed before anything is matched.
        """
        return re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)

    @staticmethod
    def _css_without_block(source: str, header: str) -> str:
        """``source`` with one at-rule removed, found by brace matching.

        Line numbers are not usable for this: every stage of the plan moves them.  The
        block is cut out rather than skipped over so that a search for the
        *unconditional* rule cannot land inside a media query by accident.
        """
        start = source.find(header)
        if start < 0:
            return source
        depth = 0
        for index in range(start + len(header) - 1, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    return source[:start] + source[index + 1 :]
        return source

    @staticmethod
    def _css_block(source: str, header: str) -> str:
        """The body of one at-rule, found by brace matching."""
        start = source.find(header)
        if start < 0:
            return ""
        depth = 0
        for index in range(start + len(header) - 1, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    return source[start : index + 1]
        return ""

    def _padding_sides(self, source: str, selector: str) -> tuple[str, str]:
        """The left and right values of the ``padding`` shorthand of one rule.

        Returns them as written (``"20px"``), not as numbers: the comparison the stage
        asks for is between two declarations of the same stylesheet, and normalising to
        a float would quietly equate ``20px`` with ``20%``.
        """
        rule = re.search(r"#storm-results-v2 " + re.escape(selector) + r" \{([^}]*)\}", self._css_no_comments(source))
        self.assertIsNotNone(rule, f"no rule for {selector} in this part of the stylesheet")
        declaration = re.search(r"(?:^|;)\s*padding:\s*([^;}]+)", rule.group(1))
        self.assertIsNotNone(declaration, f"the rule for {selector} declares no padding")
        parts = declaration.group(1).split()
        self.assertIn(len(parts), (1, 2, 3, 4), f"padding of {selector} is not a shorthand: {parts}")
        if len(parts) == 1:
            return parts[0], parts[0]
        if len(parts) == 4:
            return parts[3], parts[1]
        return parts[1], parts[1]

    def test_map_panel_inset_matches_the_table_toolbar(self):
        """The unconditional inset of the map panel equals the toolbar's, side for side.

        The block of ``@media (min-height: 520px)`` is cut out first: it carries the
        *layout* rule for the same selector (display, flex, min-height, overflow) and a
        search for the inset would otherwise stop on it.  What is asserted is equality
        between two rules of this file, never a number typed here.
        """
        source = self._css_no_comments(self._map_source("results-v2.css"))
        unconditional = self._css_without_block(source, "@media (min-height: 520px) {")
        self.assertEqual(
            self._padding_sides(unconditional, "#r-map-panel"),
            self._padding_sides(unconditional, ".r-toolbar"),
        )

    def test_map_panel_inset_has_the_narrow_viewport_override(self):
        """Below 520 px the panel is inset no more than the toolbar is.

        The mode row of the map is the narrowest thing in either panel, so the map may
        take *less* room on the side than the table toolbar does, never more.  The
        inequality, and not an equality, is what leaves decision D2 free to pick either
        measured value without this test having to be rewritten.
        """
        source = self._css_no_comments(self._map_source("results-v2.css"))
        narrow = self._css_block(source, "@media (max-width: 520px) {")
        self.assertTrue(narrow, "results-v2.css has no @media (max-width: 520px) block")
        panel_left, panel_right = self._padding_sides(narrow, "#r-map-panel")
        toolbar_left, toolbar_right = self._padding_sides(narrow, ".r-toolbar")
        self.assertEqual(panel_left, panel_right, "the narrow inset is not symmetrical")
        to_px = lambda value: float(value.removesuffix("px"))  # noqa: E731
        self.assertLessEqual(to_px(panel_left), to_px(toolbar_left))
        self.assertLessEqual(to_px(panel_right), to_px(toolbar_right))

    def test_map_panel_layout_rule_precedes_the_inset_rule(self):
        r"""The layout rule for #r-map-panel stands before the rule that insets it.

        ``test_map_panel_shrinks_to_fit_the_workspace`` reads the **first** match of
        ``#storm-results-v2 #r-map-panel \{(.+?)\}`` and asserts display, flex-direction,
        flex and min-height on it.  An inset rule placed earlier in the file would turn
        that test red with a message about a missing ``display: flex`` and send the next
        reader looking for the wrong defect.  This test is the sentinel of that trap.

        It reads the stylesheet **both ways**, raw and with the comments taken out, and
        the raw pass is the one that matters: the test it guards reads the file raw, so
        a comment carrying ``#storm-results-v2 #r-map-panel {`` above the layout rule
        would turn that test red while a sentinel that only ever saw stripped source
        stayed green - the sentinel would then be pointing at the trap it missed.  The
        stripped pass is kept so that the sentinel also holds if the guarded test is
        ever moved onto ``_css_no_comments``.
        """
        raw = self._map_source("results-v2.css")
        for source in (raw, self._css_no_comments(raw)):
            first = re.search(r"#storm-results-v2 #r-map-panel \{(.+?)\}", source, re.DOTALL)
            self.assertIsNotNone(first, "results-v2.css has no rule for #r-map-panel")
            self.assertIn("display: flex", first.group(1))
            inset = re.search(r"#storm-results-v2 #r-map-panel \{[^}]*padding:", source, re.DOTALL)
            self.assertIsNotNone(inset, "results-v2.css declares no padding for #r-map-panel")
            self.assertLess(first.start(), inset.start())

    def test_map_css_absorbs_the_panel_inset(self):
        """network-map.css gives the inset room: the mode row may wrap, the empty state has no side padding.

        Two assertions, one test, on purpose.  ``.nm-empty`` has no coverage of any kind
        - the string does not occur anywhere else in these tests and the browser gate
        never visits a session without geometry - so without the second assertion the
        D5 declaration would be the only thing this stage changes that nothing checks.

        The first assertion is anchored to ``.nm-modes`` alone.  ``.nm-tools`` stands on
        the line immediately above it and already declares ``flex-wrap: wrap``, so a
        DOTALL search over both would pass on the stylesheet as it was *before* the
        stage.  Wrapping matters because the panel is clipped
        (``overflow: hidden`` on #r-map-panel, results-v2.css section 1): a mode row
        that no longer fits is cut off rather than pushed.
        """
        css = self._css_no_comments(self._map_source("network-map.css"))
        modes = re.search(r"#storm-results-v2 \.nm-modes \{[^}]*\}", css)
        self.assertIsNotNone(modes, "network-map.css has no rule for .nm-modes")
        self.assertIn("flex-wrap: wrap", modes.group(0))

        left, right = self._padding_sides(css, ".nm-empty")
        self.assertEqual((left, right), ("0", "0"), "the side inset of the empty state is the panel's (D5)")


class ResultsV2NetworkMapInteractionTests(TestCase):
    """Stage 5 — zoom, pan, hit testing, tooltip and the selection card (D11).

    The split between this class and the browser gate is deliberate and is the rule
    of the stage: *pure* logic is exported and exercised by ``harness_network_map.js``
    from Node, *behaviour that needs a document* — the controls, the card, Esc, the
    tooltip, the cursor, the hit layer — is measured by
    ``rev/findings/plan23-checks/shots_map.py`` in a real browser.  What is left here
    are contracts of the source itself: which names are exported, which patterns must
    never appear, and which declarations the browser behaviour rests on.  A Python
    test that re-implemented the JavaScript would only prove that the copy agrees
    with itself, which is why none of these execute the module.
    """

    REQUIRED_EXPORTS = (
        "fitViewBox",
        "zoomAt",
        "pointsToPath",
        "outletPath",
        "elementCount",
        "symbolRadiusPx",
        "isDrag",
        "trackDist",
        "wheelAction",
        "kindToTab",
        "tabToKind",
        "dimSet",
        "findResult",
        "pickColor",
        "nextLegendState",
        "readMapPrefs",
        "clampCardPos",
        "writeMapPrefs",
        "placeLabels",
    )
    """Every name the harness calls.  Later stages *add* to this tuple; the test asks
    for a subset and never for an exact count, so stage 7 exporting one more name is
    not a failure of stage 6.  The exact count is asserted in the harness itself,
    where the module is actually loaded."""

    @staticmethod
    def _asset(name: str) -> str:
        static = Path(__file__).resolve().parent.parent.parent / "static"
        folder = "js" if name.endswith(".js") else "css"
        return (static / folder / name).read_text(encoding="utf-8")

    @classmethod
    def _code(cls, name: str) -> str:
        """The asset with its comments removed.

        Every assertion below greps for a pattern that the *explanation* of that same
        pattern is bound to contain — the comment next to the hit layer says why the
        permissive keyword is wrong, the comment over the card says the code never
        assigns to a location.  Grepping the raw file would fail on the prose that
        exists to justify the code.
        """
        source = re.sub(r"/\*.*?\*/", "", cls._asset(name), flags=re.DOTALL)
        return "\n".join(line for line in source.splitlines() if not line.strip().startswith("//"))

    @staticmethod
    def _braced_block(source: str, match: re.Match) -> str:
        """The ``{...}`` that starts at the end of *match*, found by brace matching."""
        start = source.index("{", match.end() - 1)
        depth = 0
        for index in range(start, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    return source[start : index + 1]
        raise AssertionError("unbalanced braces")

    # -- 56 ---------------------------------------------------------------------
    def test_network_map_harness(self):
        """Run ``harness_network_map.js``: the pure layer of the map, from Node.

        Skipped only when Node is missing — never when it fails.
        """
        import shutil

        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("Node.js is not installed; skipping the network map harness")

        harness_path = Path(__file__).resolve().parent / "harness_network_map.js"
        self.assertTrue(harness_path.exists(), f"harness_network_map.js not found at {harness_path}")

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
            f"the network map harness failed with exit code {result.returncode}:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.assertIn("ALL NETWORK MAP PURE-LAYER CASES VERIFIED (25 / 25 PASSED)", result.stdout)

    # -- 57 ---------------------------------------------------------------------
    def test_map_js_has_no_unsafe_patterns(self):
        """Names are data: no markup string, no hand-built URL, no dynamic selector.

        Element names come from the model file, which is user input.  ``innerHTML``
        would execute them, a selector built from one would raise a ``DOMException``
        for a perfectly legal SWMM name such as ``A/B``, and ``encodeURIComponent``
        would corrupt the ``/`` of that same name when pasted into a URL — which is
        why every href comes ready-made from the server's ``reverse()``.  Navigation
        happens through an ``<a href>`` and never through an assignment to a location,
        so that a click on the map can be opened in a new tab like any other link.
        """
        code = self._code("network-map.js")

        self.assertNotIn("innerHTML", code, "the map must never write markup: names are user input")
        self.assertNotIn("encodeURIComponent", code, "URLs come from the server's reverse(), never from string building")
        self.assertNotRegex(code, r"\beval\s*\(", "the map must not evaluate source at runtime")
        self.assertNotRegex(code, r"\bnew\s+Function\s*\(", "the map must not build a function from a string")
        self.assertNotRegex(code, r"location\s*\.\s*href\s*=[^=]", "navigation belongs to <a href>, not to an assignment (D11)")
        self.assertNotRegex(code, r"location\s*\.\s*(assign|replace)\s*\(", "navigation belongs to <a href> (D11)")
        self.assertNotRegex(code, r"\.\s*outerHTML\b", "the map must never write markup")
        self.assertNotRegex(code, r"insertAdjacentHTML", "the map must never write markup")

        # A selector is a literal or it is a bug: an argument holding an interpolation
        # or a concatenation is a name on its way into a selector.
        for call in re.findall(r"querySelector(?:All)?\(([^)]*)\)", code):
            self.assertNotIn("${", call, f"selector built from a value: {call!r}")
            self.assertNotIn("+", call, f"selector built by concatenation: {call!r}")
            self.assertNotIn("`", call, f"selector built from a template literal: {call!r}")

        # The name does reach the DOM — as text and as a data attribute, both of which
        # escape themselves.
        self.assertIn("textContent", code)
        self.assertIn("data-nm-name", code)

    # -- 58 ---------------------------------------------------------------------
    def test_map_js_exports_for_node(self):
        """``module.exports`` is conditional, sits inside the IIFE and carries the pure layer.

        Outside the closing brace of the IIFE the functions are out of scope, so the
        assignment has to be the last statement *within* it; the guard keeps the
        browser from touching a ``module`` that does not exist there.
        """
        code = self._code("network-map.js")

        guard = re.search(r"if\s*\(\s*typeof\s+module\s*!==\s*[\"']undefined[\"']\s*&&\s*module\.exports\s*\)", code)
        self.assertIsNotNone(guard, "module.exports must be guarded for the browser")

        assignment = re.search(r"module\.exports\s*=\s*\{", code)
        self.assertIsNotNone(assignment, "network-map.js must export its pure helpers")
        block = self._braced_block(code, assignment)

        closing = code.rindex("})();")
        self.assertLess(assignment.start(), closing, "module.exports must stand inside the IIFE, not after it")

        keys = {key.strip() for key in re.findall(r"([A-Za-z_$][\w$]*)\s*(?::|,|\})", block)}
        for name in self.REQUIRED_EXPORTS:
            self.assertIn(name, keys, f"{name} must be exported for the harness to call it")
            self.assertRegex(code, r"function\s+" + re.escape(name) + r"\s*\(", f"{name} must be defined in the module")

    # -- 59 ---------------------------------------------------------------------
    def test_view_controls_have_aria_labels(self):
        """Three real buttons with the three contracted labels, and no fake button.

        A ``<div role="button">`` looks the same and behaves differently: it takes no
        Tab stop, fires on no Enter and answers no Space.  The labels are asserted
        literally because a screen reader reads them and the browser gate clicks them.
        """
        code = self._code("network-map.js")

        self.assertIn("createElement('button')", code, "the controls must be real buttons")
        for label in ("Zoom in", "Zoom out", "Fit network to view"):
            self.assertEqual(code.count(f"'{label}'"), 1, f"the label {label!r} must be given exactly once")
        self.assertRegex(code, r"setAttribute\('aria-label',\s*label\)|setAttribute\('aria-label',\s*'Zoom in'\)")
        self.assertIn("'nm-ctrl'", code)

        self.assertNotRegex(code, r"role\s*=\s*[\"']button[\"']", "no element may pretend to be a button")
        self.assertNotRegex(
            code, r"setAttribute\(\s*['\"]role['\"]\s*,\s*['\"]button['\"]", "no element may pretend to be a button"
        )

        # Both the wheel and the buttons go through the same two functions, so there is
        # one definition of what a zoom step is and one of what fitting means.
        self.assertIn("zoomByStep", code)
        self.assertIn("fitToView", code)

    # -- 60 ---------------------------------------------------------------------
    def test_hit_layer_pointer_events_contract(self):
        """Three separate rules, and the permissive keyword nowhere.

        ``all`` means "the interior *or* the outline, whatever the paint" — and the
        interior of a polyline built from ``[VERTICES]`` is the area its chord closes
        off.  A click tens of pixels from the pipe, on its concave side, would hit the
        pipe and steal the click from the subcatchment underneath.
        """
        css = self._code("network-map.css")

        self.assertRegex(css, r"#storm-results-v2\s+path\.nm-hit\s*\{[^}]*pointer-events:\s*stroke")
        self.assertRegex(css, r"#storm-results-v2\s+circle\.nm-hit\s*\{[^}]*pointer-events:\s*fill")
        self.assertRegex(css, r"#storm-results-v2\s+\.nm-poly\s*\{[^}]*pointer-events:\s*fill")
        self.assertNotRegex(css, r"pointer-events:\s*all", "pointer-events: all makes the interior of a bend clickable")

        # The drawn shapes are out of the way, or they would take the pointer from the
        # layer built to catch it: a 2.5 px line is not a target.
        self.assertRegex(css, r"\.nm-link[^{]*,[^{]*\.nm-node\s*\{[^}]*pointer-events:\s*none")

        code = self._code("network-map.js")
        self.assertIn("'nm-hit'", code)
        self.assertIn("non-scaling-stroke", code, "the hit stroke must keep its screen width through the zoom")
        self.assertIn("HIT_RADIUS_PX / pxPerUnit", code, "the node target must be sized in screen pixels")

    # -- 61 ---------------------------------------------------------------------
    def test_selection_card_contract(self):
        """The card is a labelled group with a close button and a *conditional* link.

        ``detail_url`` is legitimately null — a node that ends no conduit has no panel
        — and the contract is that the anchor is then not built at all and the card
        says so in one sentence.  The element still selects and still shows its
        values: under D11 everything with a results row is clickable, and what may be
        missing is only the page behind it.
        """
        code = self._code("network-map.js")

        self.assertIn("'nm-card'", code)
        self.assertRegex(code, r"setAttribute\('role',\s*'group'\)")
        self.assertRegex(code, r"setAttribute\('aria-label',\s*'Selected element'\)")
        self.assertRegex(code, r"setAttribute\('aria-label',\s*'Close selection'\)")
        self.assertIn("'Show in table'", code)

        self.assertEqual(code.count("createElement('a')"), 1, "the card builds exactly one anchor")
        branch = re.search(r"const href = [^;]*detail_url[^;]*;", code)
        self.assertIsNotNone(branch, "the href must be read from the results row, not built")
        guard = re.search(r"if \(href\) \{", code)
        self.assertIsNotNone(guard, "the anchor must stand in a branch on a truthy href")
        self.assertLess(branch.start(), guard.start())
        self.assertIn("createElement('a')", self._braced_block(code, guard))

        fallback_at = code.index("No detail page for this element.")
        self.assertGreater(fallback_at, guard.start(), "the sentence belongs to the else branch of the href test")
        self.assertEqual(code.count("No detail page for this element."), 1)

        # Selecting an element emits nothing; only the button speaks to the table, and
        # it speaks the tab vocabulary.
        self.assertEqual(code.count("nm:select"), 1, "nm:select may be dispatched from one place only")
        dispatch = code.index("nm:select")
        self.assertIn("kindToTab(kind)", code[dispatch : dispatch + 400])

    # -- 62 ---------------------------------------------------------------------
    COLOR_CLASS_RE = re.compile(r"\bnm-(?:c|v|f)-[A-Za-z0-9_]+\b|\bnm-nodata\b")
    """Every colour class of stage 6, wherever it is written out in full."""

    def test_color_classes_defined_in_css(self):
        """Each colour class the module hands out has a rule, and no rule is an orphan.

        ``pickColor`` returns a *class name* and never a colour: the values live in
        network-map.css, and a class the stylesheet does not know is a pipe drawn in
        the default grey with nothing to say why.  The names are matched in both
        directions - a rule nobody asks for is dead paint, and both mistakes are
        invisible until someone looks at the map in that mode.

        Both sides are read with comments stripped, because the comment that explains
        the mapping quotes the very names it explains.
        """
        code = self._code("network-map.js")
        css = self._code("network-map.css")

        used = set(self.COLOR_CLASS_RE.findall(code))
        self.assertEqual(
            len(used),
            9 + 8 + 5 + 1,
            f"expected nine recommendation classes, eight validity, five filling and nm-nodata, found {sorted(used)}",
        )

        # The rule of a class is what the cascade needs; a mention inside a var() name
        # is not one, which is why the tokens are called --nm-rec-*/--nm-val-*/--nm-fill-*.
        defined = {
            name
            for selectors, _body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)
            for name in self.COLOR_CLASS_RE.findall(selectors)
        }
        self.assertEqual(
            used - defined,
            set(),
            f"colour classes used by network-map.js with no rule in network-map.css: {sorted(used - defined)}",
        )
        self.assertEqual(
            defined - used,
            set(),
            f"rules in network-map.css for classes nothing paints: {sorted(defined - used)}",
        )

        for name in sorted(used):
            self.assertRegex(css, r"#storm-results-v2\s+\." + re.escape(name) + r"\s*\{", f"{name} must have a scoped rule")

        # Dimming needs two classes, not one: opacity alone still catches the pointer,
        # and the E5 hit rules (1,1,1) outrank a bare .nm-dim (1,1,0), so the rule that
        # takes the pointer away has to name the element type as well.
        self.assertRegex(
            css,
            r"#storm-results-v2\s+\w*\.nm-hit\.nm-dim[^{]*\{[^}]*pointer-events:\s*none",
            "network-map.css must take the pointer away from a dimmed hit element",
        )
        self.assertRegex(css, r"#storm-results-v2\s+\.nm-dim\s*\{[^}]*opacity", ".nm-dim must fade the drawn element")
        self.assertNotRegex(
            css, r"\.nm-legend-off[^{]*\{[^}]*pointer-events", "the legend entry of a dimmed category must stay clickable"
        )
        self.assertNotIn("!important", css, "the dimming rules win on specificity, not with !important")

        # Both rules are (1,2,0), so the cascade decides on their order: a selected
        # flooded node has to show the selection and not the alarm fill.
        self.assertLess(
            css.index(".nm-node.nm-flooded"),
            css.index(".nm-node.nm-sel"),
            "the flooded rule must stand before the selection rule, or a selected flooded node loses its mark",
        )

    # -- 63 ---------------------------------------------------------------------
    def test_map_prefs_use_own_storage_key(self):
        """The map stores its mode under its own key and never touches the V2 one.

        Two independent sets of preferences read through one key would make a broken
        value of either one break both, and writing into the Results V2 key from here
        would put a second author on a file this stage does not otherwise open.  The
        check is static: ``localStorage`` is JavaScript and the Django runner executes
        none.
        """
        map_code = self._code("network-map.js")
        table_code = self._code("results-v2.js")

        self.assertIn("stormwater-network-map-preferences", map_code, "the map must keep its own preferences key")
        self.assertNotIn("stormwater-results-v2-preferences", map_code, "the map must not read or write the Results V2 key")

        self.assertIn("stormwater-results-v2-preferences", table_code, "results-v2.js must keep its own key")
        self.assertNotIn("stormwater-network-map-preferences", table_code, "results-v2.js must not learn about the map key")

        # Every access is guarded: reading the property itself throws a SecurityError
        # where site data is blocked, and an unknown or broken value has to land on the
        # default mode rather than on an exception in the page.
        self.assertIn("function readMapPrefs(", map_code)
        self.assertIn("const DEFAULT_MODE = 'recommendation';", map_code)
        self.assertGreaterEqual(
            len(re.findall(r"\bcatch\s*\(", map_code)),
            3,
            "the reader, the writer and the access to localStorage itself each need a catch",
        )

    # -- 64 (plan24 E2) ----------------------------------------------------------
    def test_card_is_draggable_and_keeps_its_place(self):
        """The card is dragged by its heading, and where it was dragged to is kept.

        Everything here is a contract of the *source*: the browser gate measures the
        card actually moving.  What a Python test can guard is the handful of decisions
        the behaviour rests on and that a later edit would silently undo - which element
        the four listeners sit on, that the pointer is captured straight away, that the
        position outlives both the selection and the page, and that nothing writes to
        ``localStorage`` from inside a pointermove.
        """
        code = self._code("network-map.js")

        # -- the capture: once, in the pointerdown of the handle, with no threshold ---
        #
        # Counted inside the function and not in the file: ``host.setPointerCapture``
        # has been in the module since plan23 (the pan gesture), so the bare string
        # occurs twice after this stage and "exactly once" over the file would be a
        # misleading red.
        begin = re.search(r"function beginCardDrag\(", code)
        self.assertIsNotNone(begin, "network-map.js must define beginCardDrag")
        begin_body = self._braced_block(code, begin)
        self.assertEqual(
            begin_body.count("setPointerCapture"),
            1,
            "the card gesture captures the pointer exactly once, in the pointerdown of its handle",
        )
        self.assertIn("event.currentTarget.setPointerCapture(event.pointerId)", begin_body)
        self.assertNotIn(
            "isDrag(",
            begin_body,
            "the capture is taken straight away; a threshold would leave a window in which the move escapes",
        )
        # Unconditional, and that is the contract: owning the pointer is the reason the
        # record exists, so a record that is there always belongs to a live gesture.
        self.assertIn("if (cardDrag) return;", begin_body)

        # -- four listeners, all four on the heading ---------------------------------
        build = re.search(r"function buildCard\(", code)
        self.assertIsNotNone(build, "network-map.js must define buildCard")
        build_body = self._braced_block(code, build)
        for event_name, handler in (
            ("pointerdown", "beginCardDrag"),
            ("pointermove", "moveCardDrag"),
            ("pointerup", "endCardDrag"),
            ("pointercancel", "endCardDrag"),
        ):
            self.assertIn(
                f"heading.addEventListener('{event_name}', {handler})",
                build_body,
                f"the {event_name} of the card gesture must sit on the heading",
            )
            self.assertNotIn(f"card.addEventListener('{event_name}'", build_body)
            self.assertNotIn(f"host.addEventListener('{event_name}', {handler}", code)
        # The exclusion selector of the earlier drafts must not come back: the heading
        # is a bare <p> with a name in it and has no button and no values to exclude.
        self.assertNotIn("closest('button, a", code)

        # -- the end of the gesture belongs to the pointer that began it -------------
        end = re.search(r"function endCardDrag\(", code)
        self.assertIsNotNone(end, "network-map.js must define endCardDrag")
        end_body = self._braced_block(code, end)
        first = end_body.strip()[1:].strip().splitlines()[0]
        self.assertIn(
            "cardDrag.pointerId",
            first,
            "the owner test is the first statement of the end of the gesture, before anything is released",
        )
        self.assertLess(
            end_body.index("cardDrag.pointerId"),
            end_body.index("releasePointerCapture"),
            "a second finger's pointerup must be refused before the capture of the first one is released",
        )
        release = re.search(r"try\s*\{[^}]*releasePointerCapture", end_body)
        self.assertIsNotNone(release, "releasePointerCapture must stand inside a try")

        # -- applyCardPos: the guard, the clearing, and bottom -----------------------
        apply_match = re.search(r"function applyCardPos\(", code)
        self.assertIsNotNone(apply_match, "network-map.js must define applyCardPos")
        apply_body = self._braced_block(code, apply_match)
        self.assertTrue(
            apply_body.strip()[1:].strip().startswith("if (!card) return;"),
            "applyCardPos runs from the ResizeObserver before anything is ever selected",
        )
        self.assertIn(
            "'auto'",
            apply_body,
            "the stylesheet pins the card with left + bottom; a top without bottom: auto stretches it",
        )
        self.assertLess(
            build_body.index("host.insertBefore(card"),
            build_body.index("applyCardPos()"),
            "the card is placed after it is inserted, because only then does it have a size",
        )

        # -- the position outlives the selection -------------------------------------
        clear = re.search(r"function clearSelection\(", code)
        self.assertIsNotNone(clear, "network-map.js must define clearSelection")
        clear_body = self._braced_block(code, clear)
        self.assertIn("cardDrag = null;", clear_body, "Esc during a drag removes a card with a captured pointer on it")
        self.assertNotIn("cardPos = null", clear_body, "the dragged position survives every selection; that is the whole point")

        # -- and it is applied on a resize, never on a rescale ------------------------
        observer = re.search(r"new ResizeObserver\(\(\) =>", code)
        self.assertIsNotNone(observer, "network-map.js must observe the size of its container")
        self.assertIn("applyCardPos()", self._braced_block(code, observer))
        rescale = re.search(r"function rescale\(", code)
        self.assertIsNotNone(rescale, "network-map.js must define rescale")
        self.assertNotIn(
            "applyCardPos()",
            self._braced_block(code, rescale),
            "rescale runs on every wheel notch and every pan step; it owns the viewBox, not the HTML layer",
        )

        # -- the preferences: one writer, and it carries both fields ------------------
        writer = re.search(r"function writeMapPrefs\(", code)
        self.assertIsNotNone(writer, "network-map.js must define writeMapPrefs")
        self.assertIn("card:", self._braced_block(code, writer), "the stored preferences carry the position of the card")
        self.assertIn("let cardPos = prefs.card", code, "the position is read where the preferences are read")
        calls = re.findall(r"(?<!function )writeMapPrefs\(", code)
        self.assertEqual(len(calls), 1, "exactly one call site writes the preferences")
        save = re.search(r"function savePrefs\(", code)
        self.assertIsNotNone(save, "network-map.js must define savePrefs")
        self.assertIn("writeMapPrefs(", self._braced_block(code, save), "the one call site is savePrefs")
        self.assertNotIn(
            "writeMapPrefs(mapStorage(), mode)",
            code,
            "writing the mode alone is exactly how the position of the card used to disappear",
        )
        set_mode = re.search(r"function setMode\(", code)
        self.assertIsNotNone(set_mode, "network-map.js must define setMode")
        self.assertIn("savePrefs()", self._braced_block(code, set_mode))
        self.assertIn("savePrefs()", end_body, "the position is stored when the gesture ends, on up and on cancel alike")
        move = re.search(r"function moveCardDrag\(", code)
        self.assertIsNotNone(move, "network-map.js must define moveCardDrag")
        self.assertNotIn(
            "savePrefs()",
            self._braced_block(code, move),
            "localStorage.setItem is synchronous; a write per frame has no business in a pointermove",
        )

    # -- 65 (plan24 E2) ----------------------------------------------------------
    def test_card_fits_the_canvas_and_only_the_heading_drags(self):
        """The card is capped against its canvas, and the three handle declarations are on the heading.

        ``.nm-canvas`` clips with ``overflow: hidden`` and ``clampCardPos`` can move a
        card but never shrink one, so the cap is what keeps a long card - and its close
        button - reachable.  The handle declarations are the other half: on ``.nm-card``
        ``touch-action: none`` would stop the finger from scrolling the very card the cap
        tells it to scroll, and ``user-select: none`` would take away the copying of the
        values.

        The scope of both rules is guarded by ``test_css_scoped_under_root`` in this
        file, which reads *this* stylesheet; ``test_all_css_rules_scoped_under_root_id``
        reads results-v2.css only and never sees the card at all.
        """
        css = self._code("network-map.css")

        card = re.search(r"#storm-results-v2 \.nm-card \{([^}]*)\}", css)
        self.assertIsNotNone(card, "network-map.css has no rule for .nm-card")
        card_body = card.group(1)
        for declaration in ("max-height", "max-width: min(", "overflow-y: auto"):
            self.assertIn(declaration, card_body, f"the card must be capped against its canvas ({declaration})")
        for forbidden in ("touch-action", "user-select"):
            self.assertNotIn(
                forbidden,
                card_body,
                f"{forbidden} on the card itself takes away the scrolling or the copying the cap depends on",
            )

        heading = re.search(r"#storm-results-v2 \.nm-card-name \{([^}]*)\}", css)
        self.assertIsNotNone(heading, "network-map.css has no rule for .nm-card-name")
        heading_body = heading.group(1)
        for declaration in ("cursor: grab", "touch-action: none", "user-select: none"):
            self.assertIn(declaration, heading_body, f"the handle must carry {declaration}")

        # A leftover of a decision that no longer holds: with the handle narrowed to the
        # heading, the values keep the browser default and need no rule of their own.
        self.assertNotIn("user-select: text", css)

    # -- 76 ---------------------------------------------------------------------
    def test_labels_group_created_once(self):
        """One group of names, built in ``buildSvg`` and standing over the nodes.

        The order of the three groups is the order they are painted in, so the names
        have to be appended after ``nm-nodes`` or the symbols would be drawn over the
        text that names them.  The literal appears exactly once: a second occurrence
        would mean a selector built from the class, and this module reaches its
        elements through the two indexes and never through the DOM.
        """
        code = self._code("network-map.js")

        self.assertEqual(code.count("'nm-labels'"), 1, "the group class must be written exactly once")
        self.assertLess(
            code.index("svg.appendChild(nodes);"),
            code.index("svg.appendChild(labels);"),
            "the names must be appended after the nodes, or the symbols cover them",
        )
        self.assertIn("labelEls: labelEls", code, "buildSvg must hand the three label lists back")
        self.assertIn("labelGroup: labels", code, "buildSvg must hand the group itself back")

    # -- 77 ---------------------------------------------------------------------
    def test_labels_group_is_not_a_pointer_target(self):
        """The names lie over the drawing and must not take the pointer from it.

        A name that caught a click would stop the pipe underneath it from being
        selected and would swallow the tooltip of the element it belongs to.  The rule
        is on the group, so it covers every ``<text>`` in it without a second selector.
        """
        css = re.sub(r"/\*.*?\*/", "", self._asset("network-map.css"), flags=re.DOTALL)

        self.assertRegex(
            css,
            r"#storm-results-v2 \.nm-labels\s*\{[^}]*pointer-events:\s*none",
            "the label group must not be a pointer target",
        )
        block = re.search(r"#storm-results-v2 \.nm-labels\s*\{[^}]*\}", css)
        self.assertIsNotNone(block)
        self.assertNotIn("pointer-events: all", block.group(0))

    # -- 78 ---------------------------------------------------------------------
    def test_labels_group_is_aria_hidden(self):
        """The names are out of the accessibility tree.

        Every drawn element already carries its name in a ``<title>``, which is what a
        screen reader announces; a group of ``<text>`` read as well would repeat the
        whole network a second time, in an order nobody chose.
        """
        code = self._code("network-map.js")

        self.assertRegex(
            code,
            r"svgNode\('g',\s*\{\s*'aria-hidden':\s*'true'\s*\},\s*'nm-labels'\)",
            "the label group must be built aria-hidden",
        )

    # -- 79 ---------------------------------------------------------------------
    def test_label_font_size_is_declared_in_css_not_as_attribute(self):
        """The type of a name lives in the stylesheet, and ``y`` stays a baseline.

        A ``font-size`` attribute written from JavaScript would be a second place to
        change the type in, and the measurement of the text would then read a font the
        stylesheet no longer sets.  ``dominant-baseline`` must not come back either
        (N01): the vertical position is the *measured* ascent of the very string being
        drawn, not a mode the browser derives from the metrics of the face.
        """
        code = self._code("network-map.js")
        css = re.sub(r"/\*.*?\*/", "", self._asset("network-map.css"), flags=re.DOTALL)

        self.assertNotIn("setAttribute('font-size'", code, "the type of a name belongs to the stylesheet")
        rule = re.search(r"#storm-results-v2 \.nm-label\s*\{[^}]*\}", css)
        self.assertIsNotNone(rule, "network-map.css must declare a .nm-label rule")
        self.assertRegex(rule.group(0), r"font-size:\s*11px", "the name is 11 px, as E4a measured with")
        self.assertNotIn("dominant-baseline", rule.group(0), "the baseline must stay the default alphabetic one")
        self.assertNotIn("dominant-baseline", css, "no rule of this stylesheet declares a baseline")

    # -- 80 ---------------------------------------------------------------------
    def test_labels_without_a_placement_are_hidden(self):
        """A name the solver found no room for is taken down, not left behind.

        ``placeLabels`` deliberately omits an element it could not place ("an element
        with no free position is simply absent from the answer"), and the culling drops
        everything outside the visible rectangle - but a ``<text>`` is built once,
        lazily, and does not remove itself.  Every name therefore goes down *before*
        the loop over the answer puts the placed ones back up, all three collections of
        them; otherwise the names of the previous frame stay where they were, on top of
        symbols and along pipes.

        The insertion point is asserted here as well: the solver answers with the
        corner of the reserved rectangle, a ``<text>`` writes a baseline, and the
        rectangle contains the halo and the overhang of the glyphs on every side.  So
        both attributes carry ``LABEL_HALO_PX`` and the measured ink offsets - without
        them the ink leaves the rectangle that was reserved for it, in both directions
        for a name holding a 'j' and for every italic (N01).
        """
        code = self._code("network-map.js")
        block = self._braced_block(code, re.search(r"function placeLabelsNow\(\)\s*\{", code))

        for kind in ("node", "link", "poly"):
            self.assertIn(f"hideLabels(drawing.labelEls.{kind})", block, f"the {kind} names must go down first")
        self.assertLess(
            block.index("hideLabels(drawing.labelEls.poly)"),
            block.index("setAttribute('x'"),
            "everything is hidden before anything is placed",
        )
        self.assertIn("setAttribute('x', String(entry.dx + LABEL_HALO_PX + target.box.left))", block)
        self.assertIn("setAttribute('y', String(entry.dy + LABEL_HALO_PX + target.box.ascent))", block)

        measured = self._braced_block(code, re.search(r"function labelBox\(variant, text\)\s*\{", code))
        self.assertIn("m.actualBoundingBoxLeft", measured, "the left offset is the measured ink, never an estimate")
        self.assertIn("m.actualBoundingBoxAscent", measured, "the ascent is the measured ink, never an estimate")
        self.assertIn("m.actualBoundingBoxRight", measured, "the width is the ink box and not the advance")
        self.assertIn("w: left + right + 2 * LABEL_HALO_PX", measured, "the halo counts on both sides")
        self.assertIn("h: ascent + descent + 2 * LABEL_HALO_PX", measured, "and above and below")

    # -- 81 ---------------------------------------------------------------------
    def test_labels_toggle_exposes_pressed_state(self):
        """The names button says that it is on, in the ink and to a screen reader.

        Asserted on the body of the one function that owns the state and never on a
        count of ``aria-pressed``: the file writes that attribute in four places
        already (the mode buttons, the legend entries), so a count would pass while the
        new button stayed silent.
        """
        code = self._code("network-map.js")
        block = self._braced_block(code, re.search(r"function updateLabelButton\(\)\s*\{", code))

        self.assertIn("setAttribute('aria-pressed'", block, "the pressed state must reach the accessibility tree")
        self.assertIn("'nm-ctrl-on'", block, "the pressed state must be visible as well")
        self.assertIn("'Show element names'", code, "the button needs the contracted label")

    # -- 82 ---------------------------------------------------------------------
    def test_labels_placement_does_not_measure_the_dom(self):
        """No measurement of the document on the path of a frame.

        ``getBBox()`` forces a synchronous layout and does not exist in Node, so it is
        banned outright.  ``getComputedStyle`` is allowed in exactly one place - the
        body of ``ensureLabels``, which reads the font of each variant **once**, when
        the names are built.  The ban is on the *measurement of the document*, not on
        the geometry of the container: both ``rescale`` and ``placeLabelsNow`` read
        ``host.getBoundingClientRect()``, once each, because the rectangle the drawing
        lives in is the one fact neither can derive (test 88 pins that budget).
        """
        code = self._code("network-map.js")
        block = self._braced_block(code, re.search(r"function ensureLabels\(\)\s*\{", code))

        self.assertNotIn("getBBox", code, "getBBox forces a layout and has no place here")
        self.assertIn("getComputedStyle", block, "the font of each variant is read where the names are built")
        self.assertEqual(
            code.replace(block, "").count("getComputedStyle"),
            0,
            "nothing outside ensureLabels may measure the document",
        )
        self.assertIn("labelFonts[variant]", block, "the font is read per variant, upright and italic apart")
        self.assertIn("'\\u0000'", code, "the metrics cache is keyed by the font as well as by the text")

    # -- 83 ---------------------------------------------------------------------
    def test_prefs_carry_all_three_fields(self):
        """One object, one writer, every field written every time (N02).

        ``writeMapPrefs`` takes the whole preference object and ``savePrefs`` is the
        only place that calls it, so no gesture can write one field and drop another -
        which is exactly how "turn the names on, switch the colouring, reload, names
        gone" used to happen.  The third field of the object, ``card``, arrives with
        E2 and is merged into ``savePrefs`` there; in this tree the object carries the
        mode and the names.
        """
        code = self._code("network-map.js")

        self.assertIn("function writeMapPrefs(storageLike, prefs)", code, "the writer takes the whole object")
        self.assertRegex(code, r"JSON\.stringify\(\{[^}]*labels:", "the stored object carries the names field")
        self.assertNotIn("writeMapPrefs(mapStorage(), mode)", code, "no gesture may write the mode on its own")
        self.assertIn("let labelsOn = prefs.labels", code, "the one read of the preferences must feed the names")

        calls = re.findall(r"(?<!function )writeMapPrefs\(", code)
        self.assertEqual(len(calls), 1, "writeMapPrefs must be called from exactly one place")
        block = self._braced_block(code, re.search(r"function savePrefs\(\)\s*\{", code))
        self.assertIn("writeMapPrefs(", block, "that one place is savePrefs")
        self.assertIn("mode: mode", block)
        self.assertIn("labels: labelsOn", block)
        self.assertIn("savePrefs();", self._braced_block(code, re.search(r"function setMode\(next\)\s*\{", code)))

    # -- 84 ---------------------------------------------------------------------
    def test_labels_off_by_default(self):
        """D20 - a map opens without its names on it.

        Six hundred names written over a drawing nobody has touched yet is not a map,
        and the gesture that brings them back is a single button.
        """
        code = self._code("network-map.js")

        self.assertIn("const DEFAULT_LABELS = false;", code)
        self.assertIn("labels: DEFAULT_LABELS", code, "the reader must fall back on that constant")

    # -- 85 ---------------------------------------------------------------------
    def test_lazy_labels_get_the_dimming(self):
        """Names born after ``ensureDrawn`` catch up with both reasons to be faded.

        ``markUnanalysed()`` and ``applyDim()`` run once, while the drawing is built, so
        a ``<text>`` created later would carry neither ``.nm-na`` nor ``.nm-dim``.
        ``ensureLabels`` therefore ends with a pass of its own - over the label lists
        alone.  It must *not* call ``applyDim()``: that walks the whole double loop of
        the collections and ends in ``updateLegendState()``, i.e. enters the legend
        machinery from the label path.

        The lazy build is guarded by ``labelsBuilt`` and never by the length of a list:
        on a payload with no nodes the length never stops being zero, and ``rescale()``
        - which runs on every step of a pan - would rebuild the names on every frame.
        """
        code = self._code("network-map.js")
        block = self._braced_block(code, re.search(r"function ensureLabels\(\)\s*\{", code))

        self.assertIn("markUnanalysedLabels()", block, "the unanalysed names must be greyed here")
        self.assertNotIn("applyDim()", block, "the label path must not enter the legend machinery")
        self.assertIn("labelsBuilt = true;", block, "the guard is a flag, set at the end")
        self.assertIn("if (labelsOn && !labelsBuilt) ensureLabels();", code, "rescale asks the flag, not a length")
        self.assertNotIn("labelEls.node.length === 0", code, "a length is not the guard")

    # -- 86 ---------------------------------------------------------------------
    def test_subcatchment_labels_use_the_centroid(self):
        """D22 - the name of a subcatchment stands on the centroid E3a put in the payload.

        Never on a centroid computed in the browser: that would be a second definition
        of the same point, and the payload already carries one that the tests of the
        extractor pin.  An outline whose payload has no ``c`` - a session written
        before E3a - gets no ``<text>`` at all, which is the one visible consequence of
        D17 for the names.

        The name is told apart by its style and not by its colour: ``--r-muted`` is the
        very token an unanalysed name uses, so the italic is what carries the
        difference, and no ``--nm-*`` token is introduced.
        """
        code = self._code("network-map.js")
        css = re.sub(r"/\*.*?\*/", "", self._asset("network-map.css"), flags=re.DOTALL)
        block = self._braced_block(code, re.search(r"function ensureLabels\(\)\s*\{", code))

        self.assertIn("anchor = finitePoint(element.c)", block, "the anchor of an outline is the payload centroid")
        self.assertNotIn("reduce(", block, "no centroid is computed here")
        self.assertNotIn("centroid", block, "no centroid is computed here")
        self.assertLess(
            block.index("if (!anchor) return;"),
            block.index("drawing.labelGroup.appendChild(text)"),
            "an element without an anchor gets no <text> at all",
        )
        self.assertIn("'nm-label nm-label-poly'", block, "the class is given when the element is created")
        self.assertIn("drawing.labelEls[kind][element.i] = text;", block, "identity is (kind, index), sparsely indexed")
        self.assertIn("const labelEls = {link: [], node: [], poly: []};", code, "three lists, not two")

        rule = re.search(r"#storm-results-v2 \.nm-label\.nm-label-poly\s*\{[^}]*\}", css)
        self.assertIsNotNone(rule, "the stylesheet must tell a subcatchment name apart")
        self.assertIn("font-style: italic", rule.group(0))
        self.assertNotIn("--nm-", rule.group(0), "no new token: the contrast test owns that namespace")

    # -- 87 ---------------------------------------------------------------------
    def test_selection_reschedules_the_labels(self):
        """R07 - selecting an element lays the names out again, and only a frame does it.

        Neither ``selectElement`` nor ``clearSelection`` calls ``rescale()``, so without
        these two calls the name of a clicked element that had lost a collision would
        not get the priority D27 promises it until the next pan, zoom or resize.  And
        ``.nm-sel`` changes the geometry of the obstacles - a conduit from 2.5 to 4, a
        node from 1 to 3 - so the layout computed for the previous selection is stale.

        One place coalesces the work: ``requestAnimationFrame`` appears exactly once in
        the file, in ``scheduleLabels``, and ``placeLabelsNow`` is called from there and
        from nowhere else.
        """
        code = self._code("network-map.js")

        for name in (r"selectElement\(hit\)", r"clearSelection\(\)"):
            block = self._braced_block(code, re.search(rf"function {name}\s*\{{", code))
            self.assertIn("scheduleLabels();", block, f"{name} must ask for a new placement")

        self.assertEqual(code.count("requestAnimationFrame"), 1, "one place coalesces the placement")
        frame_block = self._braced_block(code, re.search(r"function scheduleLabels\(\)\s*\{", code))
        self.assertIn("requestAnimationFrame", frame_block, "and that place is scheduleLabels")
        self.assertEqual(
            len(re.findall(r"(?<!function )placeLabelsNow\(", code)),
            1,
            "placeLabelsNow runs from the scheduler and from nowhere else",
        )
        self.assertIn("scheduleLabels();", self._braced_block(code, re.search(r"function toggleLabels\(\)\s*\{", code)))

    # -- 88 ---------------------------------------------------------------------
    def test_label_placement_measures_the_container_itself(self):
        """F1 - the culling frame is read where it is used, not where the scale was.

        ``rescale`` caches what belongs to the *frame* - the scale, the symbol radius
        and the density verdict - and the deferred placement reads those.  The
        rectangle is not one of them: ``left``/``top`` of ``getBoundingClientRect`` are
        the position of the container in the viewport, and the placement maps them
        through the **live** CTM.  A page that scrolls between the two moments - which
        the results page does below 520 px of height, where an unmodified wheel over
        the canvas is not a zoom - would leave those two facts from different frames,
        and the cull would drop the names of a strip as tall as the scroll, then all of
        them.  Nothing refreshes a cached rectangle either: the module observes size,
        never position, and the two ends of a selection reach the placement without
        passing through ``rescale`` at all.
        """
        code = self._code("network-map.js")
        place = self._braced_block(code, re.search(r"function placeLabelsNow\(\)\s*\{", code))
        rescale = self._braced_block(code, re.search(r"function rescale\(\)\s*\{", code))

        self.assertNotIn("lastView.rect", code, "a viewport rectangle must not survive its frame")
        stored = re.search(r"lastView = \{[^}]*\}", rescale)
        self.assertIsNotNone(stored, "rescale must write the frame it measured")
        self.assertNotIn("rect", stored.group(0), "only the scale, the radius and the density are cached")
        self.assertIn("host.getBoundingClientRect()", place, "the placement measures the container itself")
        self.assertEqual(
            place.count("getBoundingClientRect"),
            1,
            "one layout read per frame is the budget; a second one would be a second truth as well",
        )

    # -- 89 ---------------------------------------------------------------------
    def test_names_button_refuses_a_browser_that_cannot_measure(self):
        """F4 - no 2D context means no names, and the button says so instead of latching.

        ``ensureLabels`` walks out when ``getContext('2d')`` hands back null - a canvas
        blocked by a privacy mode, a page that exhausted the canvas memory - and then no
        ``<text>`` exists at any zoom.  Before this fix the button lit up, the
        preference was written and the overlay said "Zoom in to show names", which is
        advice no zoom can follow and which came back on the next visit.  The gesture is
        refused instead: the flag goes back to false, the button is repainted, the hint
        names the real cause and ``savePrefs`` is never reached.

        The predicate is asserted as well, because the obvious one is wrong:
        ``labelsBuilt`` is false for a panel that is merely still hidden, and that one
        does build its names on the next ``rescale``.
        """
        code = self._code("network-map.js")
        block = self._braced_block(code, re.search(r"function toggleLabels\(\)\s*\{", code))

        self.assertIn("labelCanvas && !labelCtx", block, "the question is the context, not the build flag")
        self.assertIn("showHint('Names are not available in this browser')", block)
        self.assertIn("labelsOn = false;", block, "a refused gesture leaves the names off")
        self.assertLess(
            block.index("labelCanvas && !labelCtx"),
            block.index("savePrefs();"),
            "nothing is written for a state the next visit could not restore",
        )
        ensure = self._braced_block(code, re.search(r"function ensureLabels\(\)\s*\{", code))
        self.assertIn("if (!labelCtx) return;", ensure, "the early way out this hint speaks for")

    # -- 90 ---------------------------------------------------------------------
    def test_card_position_is_clamped_for_the_screen_and_for_storage_apart(self):
        """F3 - the viewport may move the card on the screen, never the corner remembered.

        ``applyCardPos`` runs from the ResizeObserver, so writing the clamped corner back
        into ``cardPos`` let a narrow window replace a position chosen in a wide one, and
        the next unrelated ``savePrefs`` - the mode row, the names button - persisted the
        replacement for good; widening the window back restored nothing.  It therefore
        clamps into the style only, and the contract written at the declaration of
        ``cardPos`` is true again.

        The write-back was load-bearing all the same, because ``moveCardDrag`` assigns
        raw pointer arithmetic: the clamp for *storage* moves to ``settleCardPos``, which
        runs where a gesture ends and nowhere else - the one moment at which the clamped
        number is the reader's own intent, because it is where the card came to rest.
        """
        code = self._code("network-map.js")
        apply_body = self._braced_block(code, re.search(r"function applyCardPos\(\)\s*\{", code))
        end_body = self._braced_block(code, re.search(r"function endCardDrag\(event\)\s*\{", code))

        self.assertIsNone(
            re.search(r"\bcardPos\s*=(?!=)", apply_body),
            "applyCardPos clamps for the screen; the corner the reader chose is not its to rewrite",
        )
        self.assertIn("clampedCardCorner()", apply_body, "and it still clamps what it draws")

        settle = self._braced_block(code, re.search(r"function settleCardPos\(\)\s*\{", code))
        self.assertIn("cardPos = placed", settle, "the clamp for storage is the one that writes the corner")
        self.assertIn("settleCardPos();", end_body, "the end of the gesture clamps before it saves")
        self.assertLess(end_body.index("settleCardPos();"), end_body.index("savePrefs();"), "clamp, then save")

        move_body = self._braced_block(code, re.search(r"function moveCardDrag\(event\)\s*\{", code))
        self.assertNotIn("savePrefs", move_body, "pointermove writes nothing; localStorage.setItem is synchronous")

    # -- 91 ---------------------------------------------------------------------
    def test_escape_during_a_card_drag_saves_the_position_it_leaves(self):
        """F2 - Escape ends the gesture as surely as a pointerup does, so it saves like one.

        ``clearSelection`` removes the heading that holds the pointer capture, and the
        browser answers that with ``lostpointercapture``, which nothing listens to; the
        record is nulled before the removal anyway, so ``endCardDrag`` bails on its owner
        test even if an event did arrive.  ``savePrefs`` was therefore never reached,
        while ``cardPos`` kept what the last move wrote: the card opened at the new
        corner for the rest of the visit and at the old one after a reload.  The order
        matters as much as the call - the clamp measures the card, so the save comes
        before the removal - and ``savePrefs`` stays the only writer of the preferences.
        """
        code = self._code("network-map.js")
        clear_body = self._braced_block(code, re.search(r"function clearSelection\(\)\s*\{", code))

        self.assertIn("if (cardDrag) {", clear_body, "only a live gesture is ended here")
        self.assertIn("releasePointerCapture", clear_body, "the capture of the interrupted gesture is released too")
        self.assertIn("settleCardPos();", clear_body, "and the corner it came to rest at is clamped")
        self.assertIn("savePrefs();", clear_body, "and then remembered")
        self.assertLess(clear_body.index("settleCardPos();"), clear_body.index("savePrefs();"), "clamp, then save")
        self.assertLess(
            clear_body.index("savePrefs();"),
            clear_body.index("card.remove();"),
            "the clamp needs the card it is measuring; the removal comes after the save",
        )
        self.assertIn("cardDrag = null;", clear_body, "the record goes with the element that held the capture")
        self.assertNotIn("cardPos = null", clear_body, "the dragged position survives every selection")

    # -- 92 ---------------------------------------------------------------------
    def test_view_controls_fit_a_short_canvas(self):
        """F7 - four buttons in a column are 146 px and the short canvas is 114.4 px.

        Measured on the auto session at 390x844: ``#nm-canvas`` is 304x114.4 px, while
        four 32 px buttons with three 6 px gaps make a 146 px column.  ``.nm-canvas``
        clips with ``overflow: hidden``, so the third button ended 3.5 px past the
        bottom edge and the fourth - the names toggle - 41.5 px past it, and neither
        could be pressed at all.  Below 520 px the same four fold into two columns of
        two, 70x70 px, and keep the top-right corner of the base rule.

        A single row of four also fits the width and was tried first: the browser gate
        answered that it covers the middle of a conduit the reader has to click, so the
        assertion is on the folded shape and not merely on "not a column".
        """
        css = self._code("network-map.css")
        base = re.search(r"#storm-results-v2 \.nm-ctrls \{([^}]*)\}", css)
        self.assertIsNotNone(base, "network-map.css has no rule for .nm-ctrls")
        self.assertIn("flex-direction: column", base.group(1), "the wide layout keeps the column")
        self.assertIn("top: 10px", base.group(1))
        self.assertIn("right: 10px", base.group(1))

        narrow = re.search(r"@media \(max-width: 520px\) \{(.*?)\n\}", css, re.DOTALL)
        self.assertIsNotNone(narrow, "network-map.css has no @media (max-width: 520px) block")
        row = re.search(r"#storm-results-v2 \.nm-ctrls \{([^}]*)\}", narrow.group(1))
        self.assertIsNotNone(row, "the narrow block must lay the view controls out again")
        self.assertIn("display: grid", row.group(1), "a 146 px column does not fit a 114 px canvas")
        self.assertIn("grid-template-columns: repeat(2, 32px)", row.group(1), "two columns of two, 70x70 px")


class ResultsV2MapToTableTests(TestCase):
    """Stage 7 — the map hands an element over to the table (D10, variant "tab").

    One direction only: under the tabbed layout the map and the table are never on
    screen together, so nothing sends the table's selection back to a map nobody can
    see.  Two events cross the gap, both on ``#storm-results-v2`` and both speaking
    the tab vocabulary: ``nm:select`` from the ``Show in table`` button of the
    selection card, and ``rv2:filterchange`` from the end of every render.

    The behaviour itself is measured where it can be measured — in
    ``harness_results_v2.js``, which drives the engine against a document, and in the
    browser gate ``rev/findings/plan23-checks/shots_map.py``.  What is asserted here
    is what a source file can be held to: that the highlight is derived on every
    render rather than applied once, and that no element name is ever pasted into a
    selector.
    """

    @staticmethod
    def _asset(name: str) -> str:
        static = Path(__file__).resolve().parent.parent.parent / "static"
        folder = "js" if name.endswith(".js") else "css"
        return (static / folder / name).read_text(encoding="utf-8")

    @classmethod
    def _code(cls, name: str) -> str:
        """The asset with its comments removed; the prose quotes the patterns."""
        source = re.sub(r"/\*.*?\*/", "", cls._asset(name), flags=re.DOTALL)
        return "\n".join(line for line in source.splitlines() if not line.strip().startswith("//"))

    # -- 64 ---------------------------------------------------------------------
    def test_highlight_reapplied_after_render(self):
        """The mark on the row survives every re-render, and the state is stored first.

        The empirical half runs in ``harness_results_v2.js``: it hands an element over
        while the map is up, sorts the table twice, filters the row away and brings it
        back, and the class has to be on the row each time.  That can only work if the
        highlight is state that ``renderTable`` reads — ``changeTab`` renders
        synchronously, so a class applied after it would be written onto a ``<tbody>``
        the next sort throws away.  This test runs that harness and then holds the
        source to the same order.
        """
        import shutil

        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("Node.js is not installed; skipping the Results V2 harness")

        harness_path = Path(__file__).resolve().parent / "harness_results_v2.js"
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
            f"the Results V2 harness failed with exit code {result.returncode}:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.assertIn("✓ Stage 7 passed:", result.stdout)

        code = self._code("results-v2.js")

        # applyHighlight is called from the end of renderTable and from nowhere else.
        self.assertEqual(code.count("function applyHighlight("), 1, "applyHighlight must be defined once")
        calls = [m.start() for m in re.finditer(r"(?<!function )\bapplyHighlight\(\)", code)]
        self.assertEqual(len(calls), 1, "applyHighlight must be called from exactly one place")

        render = re.search(r"function renderTable\(", code)
        self.assertIsNotNone(render)
        body_end = code.index("\n    }", render.end())
        self.assertLess(render.start(), calls[0], "applyHighlight must be called inside renderTable")
        self.assertLess(calls[0], body_end, "applyHighlight must be called inside renderTable")
        self.assertIn("r-row-hl", code[code.index("function applyHighlight(") : calls[0] + 200])

        # The listener stores the state before it switches the tab; the reverse order
        # is the bug this ordering exists to prevent.
        listener = re.search(r"addEventListener\('nm:select'", code)
        self.assertIsNotNone(listener, "the table must listen for nm:select")
        tail = code[listener.start() :]
        store = tail.index("highlight = {")
        switch = tail.index("changeTab(")
        self.assertLess(store, switch, "the highlight must be stored before changeTab renders the table")

        # Only the button of the selection card speaks, and only about a real schema.
        self.assertIn("TYPES.includes(detail.kind)", code, "'map' and 'info' are not collections of rows")
        self.assertNotIn("rv2:rowfocus", code, "the table -> map direction belongs to the split-layout variant")

        # And the table says what it is showing, once, at the end of the same render.
        emits = [m.start() for m in re.finditer(r"rv2:filterchange", code)]
        self.assertEqual(len(emits), 1, "rv2:filterchange must be emitted from exactly one place")
        self.assertLess(calls[0], emits[0], "the highlight is applied before the table announces itself")
        self.assertLess(emits[0], body_end, "rv2:filterchange must be emitted from the end of renderTable")

    # -- 65 ---------------------------------------------------------------------
    def test_selector_uses_css_escape(self):
        """Every row selector is built through ``CSS.escape`` — names are user input.

        ``A/B``, ``P 1``, ``s#1`` and ``"x"`` are legal SWMM names, and a selector
        holding any of them raw throws a ``DOMException`` that takes the whole click
        with it.  The rule is checked on the pattern ``[data-row="``, which only a
        *selector* contains: the attribute is written out as ``data-row="`` with no
        bracket, and that one goes through the HTML escaper instead.
        """
        code = self._code("results-v2.js")

        selectors = [m.start() for m in re.finditer(r"\[data-row=\"", code)]
        self.assertGreaterEqual(len(selectors), 2, "the highlight and the scroll each look a row up")
        for start in selectors:
            fragment = code[start : code.index("]", start) + 1]
            self.assertIn("CSS.escape(", fragment, f"a row selector without CSS.escape: {fragment!r}")
            self.assertNotRegex(
                fragment,
                r"\$\{(?!\s*CSS\.escape\()",
                f"only an escaped name may be interpolated into a row selector: {fragment!r}",
            )

        # The writer of the attribute is the HTML escaper of the template, not CSS.escape.
        self.assertIn('data-row="${esc(r.id)}"', code, "the row attribute must be written through esc()")

        # No other way of reaching a row by name may creep in beside it.
        self.assertNotRegex(code, r"getElementById\([^)]*name", "a row is found by selector, never by an id built from a name")
        self.assertEqual(code.count("CSS.escape("), len(selectors), "CSS.escape belongs to the row selectors and to nothing else")


class ResultsV2NetworkMapKeyboardTests(TestCase):
    """Stage 8 — the keyboard, the edge cases and the elements nothing was computed for.

    The split is the one of stage 5 and for the same reason: what a *browser* has to
    show — that an arrow really moves the view, that ``Home`` really brings it back and
    that ``Escape`` really closes the card — is measured by
    ``rev/findings/plan23-checks/shots_map.py``, which presses those keys on the live
    page.  What a source file can be held to is asserted here: the aria contract of the
    drawing, the one place the blocking rule reads, and the wording of the counter under
    the map.  The Django runner executes no JavaScript, so a test that re-implemented
    any of this would prove only that the copy agrees with itself.
    """

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="map_keyboard_tester@example.com")
        # One conduit, two junctions and one subcatchment carry results rows; the two
        # extra nodes below are what the blocking rule has to tell apart.
        cls.session = make_session(cls.user, n_conduits=1, n_nodes=2, n_subcatchments=1)
        # A [STORAGE] unit *is* analysed: DataManager.dfn is junctions + outfalls +
        # storage, so the pipeline writes a NodeData row for it.
        make_node(cls.session, node_name="ST1")
        make_geometry(
            cls.session,
            nodes=[
                {"n": "N1", "i": 0, "p": [0.0, 0.0], "k": "junction"},
                {"n": "N2", "i": 1, "p": [10.0, 5.0], "k": "junction"},
                {"n": "ST1", "i": 2, "p": [15.0, 8.0], "k": "storage"},
                # A divider: drawn, never analysed, and it blocks itself by having no row.
                {"n": "DIV1", "i": 3, "p": [20.0, 10.0], "k": "other"},
            ],
            links=[{"n": "C1", "i": 0, "p": [[0.0, 0.0], [10.0, 5.0]], "a": "N1", "b": "N2", "kind": "conduit"}],
            polys=[{"n": "S1", "i": 0, "p": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]}],
        )

    def setUp(self):
        self.client.force_login(self.user)

    @staticmethod
    def _asset(name: str) -> str:
        static = Path(__file__).resolve().parent.parent.parent / "static"
        folder = "js" if name.endswith(".js") else "css"
        return (static / folder / name).read_text(encoding="utf-8")

    @classmethod
    def _code(cls, name: str) -> str:
        """The asset with its comments removed; the prose quotes the patterns."""
        source = re.sub(r"/\*.*?\*/", "", cls._asset(name), flags=re.DOTALL)
        return "\n".join(line for line in source.splitlines() if not line.strip().startswith("//"))

    @classmethod
    def _block(cls, source: str, pattern: str) -> str:
        """The ``{...}`` body that *pattern* opens, found by brace matching."""
        match = re.search(pattern, source)
        if match is None:
            raise AssertionError(f"pattern not found in the source: {pattern}")
        start = source.index("{", match.end() - 1)
        depth = 0
        for index in range(start, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    return source[start : index + 1]
        raise AssertionError("unbalanced braces")

    def _html(self) -> str:
        response = self.client.get(reverse("sa:analysis_results", args=[self.session.id]))
        self.assertEqual(response.status_code, 200)
        return response.content.decode("utf-8")

    # -- 66 ---------------------------------------------------------------------
    def test_skipped_counter_rendered(self):
        """The footer counts what no drawing can show, split into its three reasons.

        ``links_dangling``, ``polys_degenerate`` and ``polys_no_outlet`` sit at the
        *root* of the geometry payload, beside ``counts`` and not inside it, and they
        count exactly three things: a link whose end is not a node of the model, a
        subcatchment whose polygon is not one, and an outlet line whose target the model
        does not know.  The third one names the **line** and never the subcatchment: the
        outline of such a subcatchment is drawn in full, only the line to its outlet is
        not, so "subcatchment without an outlet" would name a different, wider thing.
        The label has to say that and nothing wider — "elements without geometry" would
        name a state that cannot occur, because a model without coordinates puts the
        whole network into the automatic layout.  A component of zero is left out of the
        brackets and a total of zero builds no element at all; a version 1 payload has
        no third key at all and ``skippedLabel`` reads a missing key as zero, so an old
        session gets no false zero either.
        """
        code = self._code("network-map.js")

        # The two fields are read from the root of the payload, with their own wording.
        table = re.search(r"const SKIPPED_FIELDS = \[(.*?)\];", code, re.DOTALL)
        self.assertIsNotNone(table, "network-map.js must name the two counters in one table")
        fields = table.group(1)
        for key in ("links_dangling", "polys_degenerate", "polys_no_outlet"):
            self.assertIn(key, fields)
        for phrase in (
            "link with an unknown endpoint",
            "links with an unknown endpoint",
            "subcatchment without a polygon",
            "subcatchments without a polygon",
            "outlet line with an unknown target",
            "outlet lines with an unknown target",
        ):
            self.assertIn(phrase, fields, f"the label must name what {phrase!r} counts")
        self.assertNotIn("without geometry", code, "the counter names undrawable elements, not missing coordinates")
        self.assertNotIn(
            "subcatchment without an outlet",
            fields,
            "the third counter names the line that is missing, not the subcatchment, which is drawn",
        )
        self.assertNotIn("data.counts.skipped", code, "the counters live beside counts, not inside it")
        self.assertRegex(code, r"skippedLabel\(data\.skipped\)", "the label is built from the root of the payload")

        # Singular and plural, the sentence, and the guard that keeps a healthy model
        # free of a footer reading "0 elements not drawn".
        body = self._block(code, r"function skippedLabel\(skipped\)")
        self.assertIn("not drawn", body)
        self.assertIn("table only", body)
        self.assertIn("count === 1 ? field[1] : field[2]", body, "one element is not 'elements'")
        self.assertIn("total === 1 ? 'element' : 'elements'", body)
        self.assertIn("if (total <= 0) return null;", body, "a total of zero must give no label at all")

        # ...and the caller builds the element only when there is a label to put in it.
        # The id and the class, and nothing else: one element, built in one place.
        self.assertEqual(code.count("'nm-skipped'"), 2, "the footer is created in exactly one place")
        self.assertEqual(code.count("skippedLine.id = 'nm-skipped';"), 1)
        guard = self._block(code, r"if \(skippedText && panel\)")
        self.assertIn("'nm-skipped'", guard, "the footer must stand inside the branch on a non-empty label")

        css = self._code("network-map.css")
        self.assertRegex(css, r"#storm-results-v2\s+\.nm-skipped\s*\{", "the footer needs a scoped rule")

        # Built in the browser and never in the template, like the schematic badge: the
        # stored payload of this session skips nothing, so the page carries no footer.
        template_path = Path(__file__).resolve().parent.parent / "templates" / "sa" / "analysis_results.html"
        self.assertNotIn("nm-skipped", template_path.read_text(encoding="utf-8"))
        html = self._html()
        self.assertNotIn("nm-skipped", html)

        # The counters must reach the browser at the root of the geometry payload and
        # carry exactly the two fields the label names.  A substring test on the served
        # page could not tell this root ``skipped`` from a nested ``counts.skipped``.
        geometry = json.loads(
            re.search(r'<script id="network-map-data" type="application/json">(.*?)</script>', html, re.S).group(1)
        )["data"]
        # This fixture carries a version 1 payload (``make_geometry_payload``), which is
        # exactly what an old session serves: two keys, no ``polys_no_outlet``, and the
        # label reads the missing third key as zero instead of printing a false "0".
        self.assertEqual(
            set(geometry["skipped"]),
            {"links_dangling", "polys_degenerate"},
            "a version 1 payload carries these two fields and no others",
        )
        # ...and this session skips nothing, which is why the page above carries no footer.
        self.assertEqual(sum(geometry["skipped"].values()), 0)

    # -- 67 ---------------------------------------------------------------------
    def test_elements_without_results_row_not_clickable(self):
        """ "Not analysed" is the absence of a row in the Results V2 collection.

        Not ``k`` and not ``kind``: ``DataManager.dfn`` is junctions + outfalls +
        **storage**, and ``_save_node_data`` writes a ``NodeData`` row for every one of
        them, so a [STORAGE] unit is analysed in full and blocking it on its kind would
        be a plain error.  A [DIVIDERS] node gets no row and blocks itself under the
        same rule, with nothing about the pipeline written down in the map.

        The served page is what carries the evidence the browser reads: the storage node
        is in both payloads, the divider only in the geometry one.  The rule itself is
        asserted where it lives.
        """
        html = self._html()
        payload = json.loads(
            re.search(r'<script id="results-v2-data" type="application/json">(.*?)</script>', html, re.S).group(1)
        )
        names = {row["name"] for row in payload["nodes"]}
        self.assertIn("ST1", names, "a [STORAGE] unit carries a NodeData row and stays clickable")
        self.assertNotIn("DIV1", names, "a [DIVIDERS] node gets no row; that is what blocks it")
        self.assertIn('"DIV1"', html, "the divider is still drawn: the geometry payload carries it")

        code = self._code("network-map.js")

        # The rule itself: one function, which asks the collection and nothing else.
        rule = self._block(code, r"function isAnalysed\(kind, name\)")
        self.assertIn("kindToTab(kind)", rule, "the collection is chosen by the drawing kind")
        self.assertIn("rowFor(kind, name)", rule, "the test is the presence of the row")
        for forbidden in ("'other'", '"other"', "'storage'", "'junction'", "'conduit'", ".k", ".kind"):
            self.assertNotIn(forbidden, rule, f"the blocking rule must not look at {forbidden}: it reads the row")
        # A Results V2 payload that carries no such collection is not evidence that
        # nothing was analysed; that branch must leave the whole kind alone.
        self.assertIn("Array.isArray(results[tab])", rule)
        self.assertIn("if (!known) return true;", rule, "an absent collection must not block the drawing")

        # The marking applies that rule and adds nothing of its own.
        block = self._block(code, r"function markUnanalysed\(\)")
        self.assertIn("isAnalysed(kind, String(element.n))", block, "the marking asks the one rule")
        self.assertIn("NOT_ANALYSED_CLASS", block)
        self.assertIn("drawing.els, drawing.hitEls", block, "drawing and hit element are marked together")

        self.assertEqual(code.count("const NOT_ANALYSED_CLASS = 'nm-na';"), 1)
        self.assertRegex(code, r"markUnanalysed\(\);", "the marking must run when the drawing is built")

        # Only the click is refused; the pointer, the tooltip and the cursor stay.
        click = self._block(code, r"host\.addEventListener\('click', event => ")
        self.assertIn("if (hit.classList.contains(NOT_ANALYSED_CLASS)) return;", click)
        self.assertLess(click.index("NOT_ANALYSED_CLASS"), click.index("selectElement(hit)"))
        self.assertEqual(code.count("element not analysed"), 1, "the tooltip says it once")

        css = self._code("network-map.css")
        self.assertRegex(
            css,
            r"#storm-results-v2\s+path\.nm-hit\.nm-na[^{]*\{[^}]*cursor:\s*not-allowed",
            "a blocked element must carry the cursor that says so",
        )
        self.assertNotRegex(
            css,
            r"\.nm-na[^{]*\{[^}]*pointer-events:\s*none",
            "pointer-events: none would take the tooltip away together with the click",
        )
        self.assertRegex(css, r"#storm-results-v2\s+\.nm-link\.nm-na\s*\{[^}]*stroke-dasharray")
        self.assertRegex(css, r"#storm-results-v2\s+\.nm-node\.nm-na\s*\{")
        # The greyed symbol ties with the [data-k] rules on specificity, so it has to
        # stand after them, and before the selection rule it must never reach.
        self.assertLess(css.index('.nm-node[data-k="other"]'), css.index(".nm-node.nm-na"))
        self.assertLess(css.index(".nm-node.nm-na"), css.index(".nm-node.nm-sel"))

    # -- 68 ---------------------------------------------------------------------
    def test_map_has_aria_contract(self):
        """The drawing is a labelled, described group with one Tab stop — not an application.

        ``role="application"`` switches off the browse mode of a screen reader over the
        whole drawing: the reader loses every key they navigate a document with and gets
        back the four this module handles.  ``role="group"`` keeps both.  There is no
        roving tabindex either — the drawn elements take no focus, the arrows move the
        view, and the equivalent accessible view of the same data is the table.
        """
        code = self._code("network-map.js")

        svg = self._block(code, r"const svg = svgNode\('svg', ")
        self.assertIn("role: 'group'", svg)
        self.assertIn("tabindex: '0'", svg)
        self.assertIn("'aria-label': 'Network map'", svg)
        self.assertIn("'aria-describedby': 'nm-help'", svg)

        # The contract is the role, not the word: "application/json" is a legal string in
        # a file like this one, and a test on the substring would one day fail on it.
        self.assertNotRegex(
            code,
            r"role['\"]?\s*[:=]\s*['\"]application",
            'role="application" switches off the browse mode of a screen reader',
        )
        self.assertEqual(code.count("tabindex"), 1, "the map is one Tab stop: no element gets a tabindex of its own")
        # The attribute is one spelling of two: a roving tabindex written as a DOM
        # property would leave the line above untouched and the contract broken.
        self.assertNotIn("tabIndex", code, "a focus stop set as a property is still a focus stop")

        # The described element exists, is built unconditionally and is visible text.
        # Three mentions and no more: the id, the class and the description the <svg>
        # points at.  A fourth would be a second element answering to the same id.
        self.assertEqual(code.count("'nm-help'"), 3)
        self.assertEqual(code.count("help.id = 'nm-help';"), 1)
        self.assertIn("help.textContent", code)
        self.assertNotRegex(code, r"help\.(hidden|setAttribute\('aria-hidden')", "the sentence is for everyone to read")
        css = self._code("network-map.css")
        self.assertRegex(css, r"#storm-results-v2\s+\.nm-help\s*\{")
        self.assertRegex(css, r"#storm-results-v2\s+\.nm-svg:focus-visible\s*\{", "the one Tab stop must show its focus")

        # One keyboard listener, on the container, and every key going through the
        # functions the buttons and the wheel already use.
        self.assertEqual(code.count("addEventListener('keydown'"), 2, "one listener on the <svg>, one for Esc on the document")
        self.assertIn("drawing.svg.addEventListener('keydown', onMapKey);", code)
        keys = self._block(code, r"function mapKeyAction\(event\)")
        for key in ("'ArrowLeft'", "'ArrowRight'", "'ArrowUp'", "'ArrowDown'", "'Home'", "'Enter'", "'+'", "'-'"):
            self.assertIn(key, keys, f"{key} must be answered by the map")
        for call in ("panBy(", "zoomByStep(", "fitToView()"):
            self.assertIn(call, keys, f"the keyboard must go through {call}, not through a second implementation")
        # The direction, not only the call: ``panBy`` moves the *view*, so ArrowRight
        # takes a negative screen delta and the content walks left under it.  A reversed
        # pair still routes through ``panBy`` and would pass every assertion above.
        for key, call in (
            ("ArrowLeft", "panBy(PAN_STEP_PX, 0)"),
            ("ArrowRight", "panBy(-PAN_STEP_PX, 0)"),
            ("ArrowUp", "panBy(0, PAN_STEP_PX)"),
            ("ArrowDown", "panBy(0, -PAN_STEP_PX)"),
        ):
            self.assertIn(call, keys, f"{key} must pan the view that way, not the other")
        self.assertNotIn("viewBox =", keys, "the keys must not compute a view of their own")
        self.assertNotIn("'Escape'", keys, "Escape is handled once, on the document, so it also clears the legend")
        self.assertIn("event.preventDefault()", self._block(code, r"function onMapKey\(event\)"))

        # D11 parity: Enter opens the anchor of the card, the same one a click opens.
        self.assertIn("card.querySelector('.nm-card-open')", keys)
        self.assertEqual(code.count("createElement('a')"), 1, "there is one anchor and one way to navigate")

    # -- 69 ---------------------------------------------------------------------
    def test_filter_dimming_never_reaches_an_unanalysed_element(self):
        """A table nobody filtered may not fade what it never listed.

        ``rv2:filterchange`` carries the names the table is showing and is emitted on
        every render, filtered or not — merely clicking the Conduits tab sends one.
        Everything drawn but analysed nowhere (a pump, a weir, an orifice, an outlet, a
        [DIVIDERS] node, the outline of a storage unit) is missing from that list on
        every single render, so a set built from the whole drawn collection holds them
        always: ``.nm-dim`` takes them to 15 % opacity *and*, through
        ``path.nm-hit.nm-dim``, takes their pointer events away — which is exactly the
        tooltip stage 8 built for them, and Esc does not undo it.

        The fix is a narrower question, not a different answer: only the elements the
        table knows about are offered to ``dimSet``, and "not in the table" stops being
        spelled like "filtered out of the table".
        """
        code = self._code("network-map.js")

        listener = self._block(code, r"root\.addEventListener\('rv2:filterchange', event => ")
        self.assertIn("analysedElements(", listener, "the collections must be narrowed before dimSet sees them")
        self.assertIn("isAnalysed(", listener, "and narrowed by the very rule that greys the others")
        self.assertRegex(
            listener,
            r"collections\[pair\[0\]\] = analysedElements\(data\[pair\[1\]\], name => isAnalysed\(pair\[0\], name\)\)",
            "every collection offered to dimSet must be the analysed subset of the drawn one",
        )
        # One call of dimSet, on those collections: a second one anywhere would be a
        # second answer to the same question, built from the unnarrowed payload.
        self.assertEqual(
            len(re.findall(r"(?<!function )dimSet\(", code)),
            1,
            "dimSet is called once, on the narrowed collections",
        )
        self.assertLess(listener.index("analysedElements("), listener.index("dimSet("), "the narrowing comes first")

        # The two states are exclusive by construction, which is the whole argument: an
        # element is marked "not analysed" exactly when ``isAnalysed`` says no, and the
        # filter is only ever shown the elements it says yes to.
        self.assertEqual(code.count("function isAnalysed(kind, name)"), 1, "one rule, in one place")
        self.assertIn("isAnalysed", self._block(code, r"function markUnanalysed\(\)"))

        # ...and the helper that narrows is the pure, exported one the harness drives.
        self.assertRegex(code, r"function analysedElements\(elements, isKnown\)")
        self.assertRegex(code, r"module\.exports = \{[^}]*analysedElements[^}]*\}")

        # The evidence from the served page: the divider is drawn and has no row, so it
        # is precisely the element the unfiltered event used to put out of reach.
        html = self._html()
        payload = json.loads(
            re.search(r'<script id="results-v2-data" type="application/json">(.*?)</script>', html, re.S).group(1)
        )
        geometry = json.loads(
            re.search(r'<script id="network-map-data" type="application/json">(.*?)</script>', html, re.S).group(1)
        )["data"]
        self.assertNotIn("DIV1", {row["name"] for row in payload["nodes"]})
        self.assertIn("DIV1", {node["n"] for node in geometry["nodes"]})

    # -- 70 ---------------------------------------------------------------------
    def test_click_consumes_the_gesture_it_belongs_to(self):
        """A click reads its own press, or none at all — never the press before it.

        ``gesture`` deliberately outlives ``pointerup`` so that the click can read the
        target of the press that made it.  Kept afterwards, it is read again by the next
        click that has no press of its own, and a keyboard makes exactly those: Enter on
        the card's ``×`` closes the card, the click bubbles to the container, the stale
        target is still the hit element of the last map click, and the card is rebuilt by
        the button that closes it.  The card then cannot be dismissed from the keyboard
        at all, and "Show in table" switches the tab and rebuilds the card behind it.

        A gesture still ``active`` is a second matter: its pointer is *down*.  That
        click came from somewhere else and must neither select nor consume the gesture
        the finger on the glass is still panning with.
        """
        code = self._code("network-map.js")
        click = self._block(code, r"host\.addEventListener\('click', event => ")

        self.assertIn("const gone = gesture;", click, "the gesture is taken into a local")
        self.assertIn("gesture = null;", click, "...and cleared: a click consumes exactly one gesture")
        self.assertIn("if (gone && gone.active) return;", click, "a press still down belongs to no click here")
        self.assertIn("if (gone && isDrag(gone.maxDist)) return;", click, "a drag still selects nothing")
        self.assertNotRegex(click, r"\bgesture\.", "nothing after the hand-over may read the live gesture again")

        # Order matters: the active test guards the clearing, and the clearing stands
        # before the drag return, or a drag would leave the gesture behind exactly as
        # the keyboard path found it.
        self.assertLess(click.index("if (gone && gone.active) return;"), click.index("gesture = null;"))
        self.assertLess(click.index("gesture = null;"), click.index("isDrag(gone.maxDist)"))
        self.assertLess(click.index("gesture = null;"), click.index("selectElement(hit)"))

        # A click that starts on the card or on the view controls is never a selection,
        # whatever the gesture says: those are real controls with listeners of their own.
        self.assertIn("closest('.nm-card, .nm-ctrls')", click)
        self.assertLess(click.index("closest('.nm-card, .nm-ctrls')"), click.index("hitUnder(source)"))

        # The source is still the press's target where there was a press: that is what
        # makes a click under the drag threshold select the element and not the host.
        self.assertIn("const source = gone && gone.target ? gone.target : event.target;", click)

    # -- 71 ---------------------------------------------------------------------
    def test_a_second_pointer_does_not_orphan_the_first(self):
        """The first finger owns the gesture until it ends it.

        The canvas sets ``touch-action: none``, so both contacts of a two-finger touch
        arrive at this handler and the browser sends no ``pointercancel`` to clear the
        first.  Overwriting the gesture with the second one loses ``captured`` — the
        record that the first pointer is captured to the host — and from then on the
        first finger's moves match no gesture (the map freezes mid-pan, and with the
        capture still in place ``event.target`` is the host, so even the tooltip stops),
        while ``endGesture`` returns on the pointerId mismatch and never releases what it
        no longer knows about.
        """
        code = self._code("network-map.js")
        down = self._block(code, r"host\.addEventListener\('pointerdown', event => ")

        guard = "if (gesture && gesture.active && event.pointerId !== gesture.pointerId) return;"
        self.assertIn(guard, down, "a second, simultaneous pointer must be ignored")
        self.assertLess(down.index(guard), down.index("gesture = {"), "the guard must stand before the assignment")

        # ``active`` and not merely ``gesture``: the object survives pointerup on purpose
        # so the click can read it, and the *next* press must replace it as it always did.
        self.assertIn("gesture.active", guard)
        self.assertEqual(code.count("gesture = {"), 1, "there is one place a gesture is started")

        # The capture is still taken in pointermove and released by pointerId in
        # endGesture; the guard is what keeps those two talking about the same pointer.
        self.assertIn("host.setPointerCapture(event.pointerId);", code)
        end = self._block(code, r"function endGesture\(event\)")
        self.assertIn("if (!gesture || event.pointerId !== gesture.pointerId) return;", end)
        self.assertIn("host.releasePointerCapture(event.pointerId);", end)

    # -- 72 ---------------------------------------------------------------------
    def test_outlet_group_drawn_between_polys_and_links(self):
        """The outlet lines are a group of their own, under the pipes and over the outlines.

        Order in the source is order in the drawing: SVG paints in document order and
        knows no z-index.  The line is a hint about where the runoff of a subcatchment
        goes and is not part of the network, so it must run *under* every pipe it
        crosses — on a model with 62 conduits a line drawn on top would read as one more
        pipe of an unknown kind.
        """
        code = self._code("network-map.js")
        body = self._block(code, r"function buildSvg\(data\)")

        for group in ("'nm-polys'", "'nm-outlets'", "'nm-links'", "'nm-nodes'"):
            self.assertIn(group, body, f"buildSvg must create the group {group}")

        # Created in order...
        created = [body.index(f"'{name}'") for name in ("nm-polys", "nm-outlets", "nm-links", "nm-nodes")]
        self.assertEqual(created, sorted(created), "the four groups must be created back to front")

        # ...and appended in the same order, which is the one the browser paints in.
        # ``labels`` (E4b) closes the list: the names are painted over everything they
        # name, which leaves the outlet group where this test puts it.
        appends = re.findall(r"svg\.appendChild\((\w+)\);", body)
        self.assertEqual(
            appends,
            ["polys", "outlets", "links", "nodes", "labels"],
            "the outlet group must be appended between the outlines and the links",
        )

    # -- 73 ---------------------------------------------------------------------
    def test_outlet_line_is_not_a_pointer_target(self):
        """Nothing about the line catches the pointer: not in the DOM, not in the sheet.

        The line covers a long, thin corridor across the drawing and is the one element
        of the map the reader can never want to click: there is no card for it, no row
        in any table and no name of its own.  Two independent guards say so — the
        element carries no ``data-nm-*`` and no ``nm-hit`` class, so the click handler
        (which reads those attributes and nothing else) could not resolve it even if it
        were hit, and the stylesheet takes pointer events away from it outright.

        The rule stands on its own and is deliberately not folded into the hit-layer
        rule of stage 5: that selector is asserted literally by
        ``test_map_hit_layer_is_not_a_pointer_target``-style regexes elsewhere in this
        file, which require the brace directly after ``.nm-node``.
        """
        code = self._code("network-map.js")
        block = self._block(code, r"if \(outletD\)")

        self.assertIn("'nm-outlet'", block, "the line carries the class the stylesheet names")
        self.assertNotIn("data-nm-", block, "the line must carry no identity the click handler could read")
        self.assertNotIn("nm-hit", block, "the line is not a hit element")
        self.assertNotIn("titled(", block, "the line carries no tooltip: it is not a target")
        self.assertNotIn("hitElement(", block)

        css = self._code("network-map.css")
        self.assertRegex(
            css,
            r"#storm-results-v2\s+\.nm-outlet\s*\{[^}]*pointer-events:\s*none",
            "a rule of its own must take pointer events from the outlet line",
        )
        # ...and that rule is its own: the hit-layer rule of stage 5 keeps its selector.
        self.assertRegex(
            css,
            r"\.nm-link[^{]*,[^{]*\.nm-node\s*\{[^}]*pointer-events:\s*none",
            "the hit-layer rule must keep its literal selector",
        )

    # -- 74 ---------------------------------------------------------------------
    def test_outlet_line_style_is_dashed_and_uses_the_palette(self):
        """Thin, dashed, muted — and faded with ``stroke-opacity``, never with ``opacity``.

        ``#storm-results-v2 .nm-dim { opacity: 0.15 }`` has the same specificity (1,1,0)
        as ``#storm-results-v2 .nm-outlet``, so between those two rules source order
        decides.  An ``opacity: 0.7`` written *after* the dimming block would win that
        tie and the line would simply stop fading with its polygon.  ``stroke-opacity``
        composes with ``opacity`` instead of competing with it, so the two cannot break
        each other — and the rule is asserted to stand before the dimming block anyway,
        which is the second, independent guard.

        The colour comes from an existing token: the link palette is a closed set
        measured against its contrast floor, and a new ``--nm-*`` token would break
        ``test_link_colour_tokens_reach_3_to_1``.
        """
        css = self._code("network-map.css")
        match = re.search(r"#storm-results-v2\s+\.nm-outlet\s*\{([^}]*)\}", css)
        self.assertIsNotNone(match, "network-map.css must carry a rule for the outlet line")
        rule = match.group(1)

        self.assertRegex(rule, r"stroke-dasharray:\s*\d", "the line is dashed, so it never reads as a pipe")
        self.assertIn("vector-effect: non-scaling-stroke", rule, "the dash must not thicken with the zoom")
        self.assertRegex(rule, r"stroke-opacity:\s*0?\.\d", "the line is faded with stroke-opacity")
        self.assertNotRegex(rule, r"(?<!-)\bopacity:", "opacity would tie with .nm-dim and source order would decide")
        self.assertRegex(rule, r"stroke:\s*var\(--r-[a-z]+\)", "the stroke comes from an existing token")
        self.assertNotIn("--nm-", rule, "no new --nm-* token: the link palette is a closed, measured set")

        # Source order, asserted as an index: the rule stands before the dimming block.
        dim = re.search(r"#storm-results-v2\s+\.nm-dim\s*\{", css)
        self.assertIsNotNone(dim, "the dimming rule must still exist")
        self.assertLess(
            match.start(),
            dim.start(),
            "the .nm-outlet rule must stand before .nm-dim, or a tie on specificity would go the wrong way",
        )

    # -- 75 ---------------------------------------------------------------------
    def test_outlet_follows_the_polygon_dimming(self):
        """The line fades with its polygon, through a third index that is *sparse*.

        ``buildSvg`` returns the index (``outletEls``) and ``setDim`` iterates it beside
        the two it already had, so a legend gesture and a table filter reach the line
        without knowing it exists: it has no ``.nm-sel`` and no state of its own.

        ``outletEls.poly[poly.i] = line`` and never ``push``: ``(kind, index)`` is the
        identity the whole module looks an element up by, and a dense array would shift
        every line by one the moment a single subcatchment had no outlet — the filter
        would then fade the line of a *different* subcatchment.

        ``outletEls`` is deliberately absent from ``markUnanalysed`` (D30).  That
        function only adds the class ``nm-na``, and every ``.nm-na`` rule of the sheet
        names a kind (``.nm-node.nm-na``, ``.nm-link.nm-na``, ``.nm-poly.nm-na``), so the
        class on an ``.nm-outlet`` would paint nothing: dead code frozen by a test.
        """
        code = self._code("network-map.js")

        build = self._block(code, r"function buildSvg\(data\)")
        self.assertIn("const outletEls = {poly: []};", build, "the third index carries the one kind that has lines")
        self.assertIn("outletEls.poly[poly.i] = line;", build, "the index is sparse: by payload index, never push")
        self.assertNotIn("outletEls.poly.push", build, "a dense array would shift every line by one")
        self.assertRegex(
            build,
            r"return \{ svg: svg, els: els, hitEls: hitEls, outletEls: outletEls[,}]",
            "buildSvg must return the index, or drawing.outletEls never exists",
        )

        dim = self._block(code, r"function setDim\(kind, index, on\)")
        self.assertIn("drawing.outletEls", dim, "the line must fade and come back with its polygon")
        self.assertRegex(dim, r"lists = \[drawing\.els, drawing\.hitEls, drawing\.outletEls, drawing\.labelEls\]")

        unanalysed = self._block(code, r"function markUnanalysed\(\)")
        self.assertNotIn("outletEls", unanalysed, "D30: a .nm-na class on the line would paint nothing at all")


class ResultsV2NetworkMapContrastTests(TestCase):
    """Every colour the map paints a link with, measured against the canvas.

    The map draws a 2.5 px non-scaling stroke on ``--r-white`` and colour is the only
    encoding it carries: no width, no dash and no label says what a pipe is, so a token
    below 3:1 does not make the information harder to read, it removes it.  WCAG 1.4.11
    asks that 3:1 of a meaningful graphical object, and the Filling mode is where it
    used to bite hardest — most conduits of a healthy run sit in the lowest band, which
    was painted at 1.63:1.

    The ratios are computed here from the stylesheet itself rather than written down, so
    that a token edited to a prettier value fails this test and not a screenshot.  There
    is one palette: the workspace declares no ``prefers-color-scheme`` and no theme
    attribute, so ``--r-white`` is the only background a link is ever drawn on.
    """

    CANVAS = "#ffffff"

    @staticmethod
    def _css(name: str) -> str:
        static = Path(__file__).resolve().parent.parent.parent / "static"
        return (static / "css" / name).read_text(encoding="utf-8")

    @classmethod
    def _tokens(cls) -> dict[str, str]:
        """Every ``--name: value`` of the two stylesheets, ``var()`` resolved."""
        source = cls._css("network-map.css") + cls._css("results-v2.css")
        declared = dict(re.findall(r"(--[\w-]+):\s*([^;]+);", source))

        def resolve(value: str, depth: int = 0) -> str:
            value = value.strip()
            reference = re.fullmatch(r"var\((--[\w-]+)\)", value)
            if reference is None or depth > 8:
                return value
            return resolve(declared[reference.group(1)], depth + 1)

        return {name: resolve(value) for name, value in declared.items()}

    @staticmethod
    def _contrast(colour: str, other: str) -> float:
        """WCAG 2.x contrast ratio of two ``#rrggbb`` colours."""

        def luminance(value: str) -> float:
            digits = value.lstrip("#")
            channels = [int(digits[index : index + 2], 16) / 255 for index in (0, 2, 4)]
            linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
            return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

        first, second = luminance(colour), luminance(other)
        lighter, darker = max(first, second), min(first, second)
        return (lighter + 0.05) / (darker + 0.05)

    def test_the_canvas_is_white_and_the_palette_is_the_only_one(self):
        """The assumption the measurement rests on, asserted rather than believed."""
        tokens = self._tokens()
        self.assertEqual(tokens["--r-white"].lower(), self.CANVAS)
        self.assertRegex(self._css("network-map.css"), r"\.nm-canvas\s*\{[^}]*background:\s*var\(--r-white\)")
        for sheet in ("network-map.css", "results-v2.css"):
            self.assertNotIn("prefers-color-scheme", self._css(sheet), "a second palette would need measuring too")
            self.assertNotIn("data-theme", self._css(sheet))

    def test_link_colour_tokens_reach_3_to_1(self):
        """Every ``--nm-*`` token a link can be painted with clears 3:1 on the canvas."""
        tokens = self._tokens()
        paints = {name: value for name, value in tokens.items() if name.startswith("--nm-") and name != "--nm-color"}

        # The ramps are complete: a mode painting a class whose token nobody measured
        # would slip through a loop over whatever happens to be declared.
        expected = [
            "--nm-rec-valid",
            "--nm-rec-dia-up",
            "--nm-rec-dia-down",
            "--nm-rec-slope-up",
            "--nm-rec-slope-down",
            "--nm-rec-depth-up",
            "--nm-rec-pump",
            "--nm-rec-tank",
            "--nm-rec-seepage",
            "--nm-none",
            *[f"--nm-val-{step}" for step in range(8)],
            *[f"--nm-fill-{step}" for step in range(5)],
        ]
        for name in expected:
            self.assertIn(name, paints, f"{name} must be declared")

        for name, value in sorted(paints.items()):
            self.assertRegex(value, r"^#[0-9a-fA-F]{6}$", f"{name} must resolve to a plain hex colour, got {value!r}")
            ratio = self._contrast(value, self.CANVAS)
            self.assertGreaterEqual(
                round(ratio, 2),
                3.0,
                f"{name} ({value}) paints a link at {ratio:.2f}:1 on the white canvas; WCAG 1.4.11 asks 3:1",
            )

    def test_the_two_paired_actions_stay_apart(self):
        """The contrast floor may not collapse a pair into one colour.

        "Increase diameter" and "reduce diameter" are one hue told apart by lightness,
        and so are the steps of the filling ramp.  Darkening the lighter one to reach
        3:1 is only correct while it stays visibly the lighter one.
        """
        tokens = self._tokens()
        for lighter, darker in (
            ("--nm-rec-dia-down", "--nm-rec-dia-up"),
            ("--nm-rec-slope-down", "--nm-rec-slope-up"),
            ("--nm-fill-0", "--nm-fill-1"),
            ("--nm-fill-1", "--nm-fill-2"),
            ("--nm-fill-2", "--nm-fill-3"),
        ):
            self.assertLess(
                self._contrast(tokens[lighter], self.CANVAS),
                self._contrast(tokens[darker], self.CANVAS),
                f"{lighter} must stay lighter than {darker}",
            )
            self.assertGreaterEqual(
                self._contrast(tokens[lighter], tokens[darker]),
                1.25,
                f"{lighter} and {darker} must stay tellable apart",
            )

    def test_a_faded_legend_entry_keeps_its_label_readable(self):
        """The off state lives on the swatch; the text never fades.

        The entry of a faded category is the control that brings it back, and it is a
        live ``<button>`` with ``aria-pressed``, not a disabled one, so WCAG's exemption
        for inactive components does not apply: 4.5:1 is due.  ``opacity: 0.55`` over
        the white toolbar composited ``--r-muted`` down to 2.3:1 — on the one label the
        reader has to read to undo what they just did.
        """
        css = self._css("network-map.css")
        rule = re.search(r"#storm-results-v2 \.nm-legend-off \{([^}]*)\}", css)
        self.assertIsNotNone(rule, "the off state must still have its rule")
        self.assertNotIn("opacity", rule.group(1), "fading the label takes it under 4.5:1")
        self.assertIn("color: var(--r-muted)", rule.group(1))

        self.assertGreaterEqual(
            self._contrast(self._tokens()["--r-muted"], self.CANVAS),
            4.5,
            "the label colour itself must clear 4.5:1 on the toolbar",
        )

        # ...and the state is still shown, on the chip that carries the colour.
        self.assertRegex(
            css,
            r"#storm-results-v2 \.nm-legend-off \.nm-swatch \{[^}]*background:\s*transparent",
            "the swatch is what says the category is off",
        )
        self.assertRegex(css, r"#storm-results-v2 \.nm-legend-off \.nm-swatch \{[^}]*box-shadow:\s*inset")
