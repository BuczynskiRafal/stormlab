"""The documentation's "What the model reads" figure and the graph code it describes.

The figure's data file is generated from the production graph functions. These
tests keep the two in step: the committed file must be what a fresh build gives,
and the weights must behave like the upstream aggregation the served model uses -
rows that are weighted means, reach only upstream, and a hand-checked row (K17).
Upstream reach is checked with a walk over the network written here, independent
of the graph builder.
"""

from __future__ import annotations

import json
import tempfile
from collections import deque
from io import StringIO
from pathlib import Path

from django.core.management import CommandError, call_command
from django.test import TestCase
from django.urls import reverse

from sa.management.commands.export_method_graph import (
    DEFAULT_INP,
    DEFAULT_OUTPUT,
    MODEL_SPEC,
    build_method_graph,
    render_method_graph,
)

TOLERANCE = 1e-3


def _upstream_within(conduits: dict[str, tuple[str, str]], start: str, max_links: int) -> dict[str, int]:
    """Conduits that drain into ``start`` within ``max_links`` links, with their distance (``start`` is 0)."""
    by_outlet: dict[str, list[str]] = {}
    for name, (_inlet, outlet) in conduits.items():
        by_outlet.setdefault(outlet, []).append(name)
    distance = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if distance[current] == max_links:
            continue
        for feeder in by_outlet.get(conduits[current][0], []):
            if feeder not in distance:
                distance[feeder] = distance[current] + 1
                queue.append(feeder)
    return distance


class MethodGraphDataTests(TestCase):
    """The committed figure data against a fresh build and against the network itself."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.built = build_method_graph(DEFAULT_INP)
        cls.committed_text = DEFAULT_OUTPUT.read_text(encoding="utf-8")
        cls.committed = json.loads(cls.committed_text)

        import swmmio

        frame = swmmio.Model(str(DEFAULT_INP), include_rpt=False).conduits()
        cls.network = {str(name): (str(row.InletNode), str(row.OutletNode)) for name, row in frame.iterrows()}

    @staticmethod
    def _support(row: list) -> set[str]:
        return {name for name, _weight in row}

    def test_committed_file_is_a_fresh_build(self):
        """Any change to the graph code or the network must be followed by regenerating the figure data."""
        self.assertEqual(self.committed_text, render_method_graph(self.built))
        self.assertTrue(self.committed_text.endswith("}\n"))

    def test_figure_describes_the_shipped_model(self):
        """The aggregation drawn is the one the served model is configured with."""
        spec = json.loads(Path(MODEL_SPEC).read_text(encoding="utf-8"))

        self.assertEqual(
            self.committed["model"],
            {"operator": spec["operator"], "hops": spec["hops"], "direction": spec["direction"]},
        )

    def test_figure_covers_every_conduit_of_the_network(self):
        names = {conduit["id"] for conduit in self.committed["conduits"]}

        self.assertEqual(names, set(self.network))
        self.assertEqual(set(self.committed["step1"]), names)
        self.assertEqual(set(self.committed["step2"]), names)

    def test_one_step_rows_are_weighted_means_that_include_the_conduit(self):
        for name, row in self.committed["step1"].items():
            with self.subTest(conduit=name):
                self.assertAlmostEqual(sum(weight for _other, weight in row), 1.0, delta=TOLERANCE)
                self.assertIn(name, self._support(row))
                self.assertTrue(all(weight > 0 for _other, weight in row))

    def test_k17_weights_match_a_hand_checked_row(self):
        """K17 ends a plain chain (K13 -> ... -> K17): weights 1 : 2 : 0.5 for itself, one and two links up.

        One step: (1, 2, 0.5) / 3.5 for K17, K16, K15. Two steps: that row applied twice,
        reaching K13 four links up. Checked by hand, independently of the graph code.
        """
        step1 = dict(self.committed["step1"]["K17"])
        step2 = dict(self.committed["step2"]["K17"])

        expected1 = {"K17": 0.286, "K16": 0.571, "K15": 0.143}
        expected2 = {"K17": 0.082, "K16": 0.327, "K15": 0.408, "K14": 0.163, "K13": 0.020}
        self.assertEqual(set(step1), set(expected1))
        self.assertEqual(set(step2), set(expected2))
        for name, weight in expected1.items():
            self.assertAlmostEqual(step1[name], weight, delta=TOLERANCE)
        for name, weight in expected2.items():
            self.assertAlmostEqual(step2[name], weight, delta=TOLERANCE)

    def test_reach_grows_with_the_second_step_and_stays_upstream(self):
        """One step reads at most two links upstream, two steps at most four; nothing downstream or aside."""
        for name in self.network:
            one = self._support(self.committed["step1"][name])
            two = self._support(self.committed["step2"][name])
            with self.subTest(conduit=name):
                self.assertLessEqual(one, two)
                self.assertLessEqual(one, set(_upstream_within(self.network, name, 2)))
                self.assertLessEqual(two, set(_upstream_within(self.network, name, 4)))

    def test_head_conduits_read_only_themselves(self):
        heads = [name for name in self.network if len(_upstream_within(self.network, name, 1)) == 1]

        self.assertTrue(heads)
        for name in heads:
            with self.subTest(conduit=name):
                self.assertEqual(self.committed["step1"][name], [[name, 1.0]])
                self.assertEqual(self.committed["step2"][name], [[name, 1.0]])

    def test_default_selection_has_the_widest_two_step_reach(self):
        widest = max(len(row) for row in self.committed["step2"].values())

        self.assertEqual(len(self.committed["step2"][self.committed["default"]]), widest)

    def test_geometry_fits_the_view_box(self):
        width, height = self.committed["viewBox"]
        points = [(node["x"], node["y"]) for node in self.committed["nodes"]]
        points += [tuple(point) for conduit in self.committed["conduits"] for point in conduit["path"]]

        self.assertEqual(max(width, height), 1000)
        self.assertTrue(all(0 <= x <= width and 0 <= y <= height for x, y in points))
        self.assertEqual([node["id"] for node in self.committed["nodes"] if node["outfall"]], ["U1"])

    def test_file_stays_small(self):
        self.assertLess(len(self.committed_text.encode("utf-8")), 60_000)


class ExportMethodGraphCommandTests(TestCase):
    """``export_method_graph`` writes the file and ``--check`` guards it."""

    def test_check_passes_on_the_committed_file(self):
        out = StringIO()

        call_command("export_method_graph", "--check", stdout=out)

        self.assertIn("up to date", out.getvalue())

    def test_check_fails_on_a_stale_file_and_writing_fixes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "method-graph.json"
            target.write_text("{}\n", encoding="utf-8")

            with self.assertRaises(CommandError):
                call_command("export_method_graph", "--check", "--output", str(target), stdout=StringIO())

            call_command("export_method_graph", "--output", str(target), stdout=StringIO())
            call_command("export_method_graph", "--check", "--output", str(target), stdout=StringIO())
            self.assertEqual(target.read_bytes(), DEFAULT_OUTPUT.read_bytes())

    def test_missing_network_file_is_reported(self):
        with self.assertRaises(CommandError):
            call_command("export_method_graph", "--inp", "/nonexistent/network.inp", "--check", stdout=StringIO())


class MethodPageFigureTests(TestCase):
    """The figure as the Method page serves it."""

    def test_caption_states_the_number_of_conduits_in_the_data(self):
        """The caption is hand-written; it must not drift from the network the data describes."""
        committed = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

        response = self.client.get(reverse("sa:method"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f"{len(committed['conduits'])} conduits.")

    def test_figure_data_url_is_versioned(self):
        response = self.client.get(reverse("sa:method"))

        self.assertContains(response, 'data-src="/static/data/method-graph.json?v=')
