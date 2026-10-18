"""Tests for ``sa.core.geometry`` — geometry extraction and automatic schematic.

The last class, ``SaveGeometryTests``, covers the persistence side: the geometry a
completed analysis stores and the two ways storing it may fail without costing the
caller its results.

Every case parses INP *text* through swmmio; nothing here runs a simulation, so
no ``.rpt``/``.out`` is ever written next to a tracked file.  Fixtures that a
single test needs are written into a temporary directory instead of the tracked
``data/`` folder, which keeps the tracked ones readable as what they are named
after.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import swmmio
from django.db import connection
from django.test import SimpleTestCase, TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from sa.core.data import DataManager
from sa.core.geometry import COMPONENT_GAP, NETMAP_VERSION, auto_layout, extract_geometry
from sa.models import CalculationSession, ConduitData, NodeData, SessionGeometry, SubcatchmentData
from sa.services import CalculationPersistenceService

from .factories import make_session, make_user

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "core" / "tests"
MODEL_STEM = "test_file"


def geometry_of(path: Path | str) -> dict:
    """Payload for the INP at *path*.

    swmmio warns about every section a file does not have, which for these
    deliberately small models is most of them; the filter keeps the test output
    readable.  It is applied here and never inside ``geometry.py``:
    ``catch_warnings`` manipulates global state and the module is imported by a
    threaded server.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return extract_geometry(swmmio.Model(str(path)).inp)


def brute_force_min_gap(positions: list[list[float]]) -> float:
    """Smallest distance between two distinct points, measured the slow, obvious way."""
    distinct = sorted({(point[0], point[1]) for point in positions})
    return min(math.hypot(a[0] - b[0], a[1] - b[1]) for index, a in enumerate(distinct) for b in distinct[index + 1 :])


class GeometryFixtureMixin:
    """Access to the tracked fixtures and to one-off models written on the fly."""

    def fixture(self, name: str) -> dict:
        return geometry_of(DATA_DIR / name)

    def model_from_text(self, text: str) -> dict:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)  # pyright: ignore[reportAttributeAccessIssue]
        path = Path(directory.name) / "case.inp"
        path.write_text(text, encoding="ascii")
        return geometry_of(path)


class ExtractGeometryTests(GeometryFixtureMixin, SimpleTestCase):
    """``extract_geometry`` over the INP sections, in both layout modes."""

    def test_nodes_links_polys_counted(self):
        geometry = self.fixture("geom_crlf_vertices.inp")
        self.assertEqual(len(geometry["nodes"]), 3)
        self.assertEqual(len(geometry["links"]), 3)
        self.assertEqual(len(geometry["polys"]), 1)
        self.assertEqual([node["i"] for node in geometry["nodes"]], [0, 1, 2])
        self.assertEqual([link["i"] for link in geometry["links"]], [0, 1, 2])

    def test_vertices_make_a_polyline(self):
        """The bends of [VERTICES] sit between the two ends, in the order of the file."""
        geometry = self.fixture("geom_crlf_vertices.inp")
        bent = next(link for link in geometry["links"] if link["n"] == "C/2")
        self.assertEqual(bent["p"], [[50.0, 0.0], [60.0, 10.0], [80.0, 30.0], [100.0, 50.0]])

    def test_y_axis_is_flipped(self):
        geometry = self.fixture("geom_crlf_vertices.inp")
        points = {node["n"]: node["p"] for node in geometry["nodes"]}
        # J1 is the topmost node in the file (y = 100) and O1 the lowest (y = 50).
        self.assertEqual(points["J1"][1], 0.0)
        self.assertEqual(points["O1"][1], 50.0)

    def test_origin_allows_reversing(self):
        geometry = self.fixture("geom_crlf_vertices.inp")
        x0, y_top = geometry["origin"]
        points = {node["n"]: node["p"] for node in geometry["nodes"]}
        for name, raw in {"J1": (0.0, 100.0), "J2": (50.0, 100.0), "O1": (100.0, 50.0)}.items():
            self.assertAlmostEqual(points[name][0] + x0, raw[0], delta=0.01)
            self.assertAlmostEqual(y_top - points[name][1], raw[1], delta=0.01)

    def test_crlf_and_mixed_case_polygons(self):
        """CRLF line endings and a ``[Polygons]`` header in mixed case both parse."""
        raw = (DATA_DIR / "geom_crlf_vertices.inp").read_bytes()
        self.assertIn(b"\r\n", raw)
        self.assertIn(b"[Polygons]", raw)
        self.assertEqual(len(self.fixture("geom_crlf_vertices.inp")["polys"]), 1)

    def test_name_with_slash_survives(self):
        geometry = self.fixture("geom_crlf_vertices.inp")
        self.assertIn("C/2", [link["n"] for link in geometry["links"]])

    def test_outfall_kind(self):
        geometry = self.fixture("geom_crlf_vertices.inp")
        kinds = {node["n"]: node["k"] for node in geometry["nodes"]}
        self.assertEqual(kinds, {"J1": "junction", "J2": "junction", "O1": "outfall"})

    def test_missing_coordinates_section_uses_auto_layout(self):
        """No [COORDINATES] at all is a normal model, not an error: everything is drawn."""
        geometry = self.fixture("geom_no_coords.inp")
        self.assertEqual(geometry["layout"], "auto")
        self.assertEqual(len(geometry["nodes"]), 3)
        self.assertEqual(len(geometry["links"]), 2)
        self.assertEqual(geometry["polys"], [])
        for node in geometry["nodes"]:
            self.assertEqual(len(node["p"]), 2)

    def test_dangling_link_endpoint_counted_in_skipped(self):
        geometry = self.model_from_text(
            "[JUNCTIONS]\nJ1   10.0  2.0  0  0  0\n\n"
            "[OUTFALLS]\nO1   8.0   FREE  NO\n\n"
            "[CONDUITS]\nC1   J1  O1     50  0.01  0  0  0\n"
            "C2   J1  GHOST  50  0.01  0  0  0\n\n"
            "[COORDINATES]\nJ1   0.0   10.0\nO1   10.0  0.0\n"
        )
        self.assertEqual(geometry["skipped"]["links_dangling"], 1)
        self.assertEqual([link["n"] for link in geometry["links"]], ["C1"])

    def test_pump_has_kind_pump(self):
        """A pump carries a full record although the pipeline computes no result for it."""
        geometry = self.fixture("geom_crlf_vertices.inp")
        pump = next(link for link in geometry["links"] if link["n"] == "PU1")
        self.assertEqual(pump["kind"], "pump")
        self.assertNotIn("t", pump)

    def test_counts_elements(self):
        """Still not a test of the outlet lines, and that is a property of the fixture.

        ``geom_crlf_vertices.inp`` carries a [Polygons] outline but no [SUBCATCHMENTS]
        section at all, so no outlet line can be built from it and the element count is
        untouched by version 2 of the payload.  The arithmetic of a line — one element
        and one point per resolved pair — is pinned by
        ``test_counts_grow_with_each_outlet_line`` on a fixture that has both sections.
        """
        geometry = self.fixture("geom_crlf_vertices.inp")
        expected = len(geometry["polys"]) + 2 * len(geometry["links"]) + len(geometry["nodes"])
        self.assertEqual(geometry["counts"]["elements"], expected)
        self.assertEqual(expected, 10)

    def test_counts_points_counts_every_coordinate_pair(self):
        """``elements`` says how much is drawn, ``points`` says what it weighs.

        The two are independent: a subcatchment is one element whatever the length of
        its outline and a link is two whatever its [VERTICES] bends, so the element
        count alone lets a payload of hundreds of thousands of coordinate pairs through
        the serializer's guard.  Counted here on a fixture that carries bends *and* a
        polygon, which is where the two numbers part company.
        """
        geometry = self.fixture("geom_crlf_vertices.inp")
        expected = (
            len(geometry["nodes"])
            + sum(len(link["p"]) for link in geometry["links"])
            + sum(len(poly["p"]) for poly in geometry["polys"])
            # Version 2: every polygon carries a centroid and every drawn line an end.
            + len(geometry["polys"])
            + sum(1 for poly in geometry["polys"] if "o" in poly)
        )
        self.assertEqual(geometry["counts"]["points"], expected)
        self.assertGreater(expected, geometry["counts"]["elements"], "a bend is a point and not an element")

    def test_counts_points_on_a_model_without_coordinates(self):
        """The schematic draws two points per link and one per node, and says so."""
        geometry = self.model_from_text(
            "[JUNCTIONS]\nJ1   10.0  2.0  0  0  0\nJ2   9.0   2.0  0  0  0\n\n"
            "[OUTFALLS]\nO1   8.0   FREE  NO\n\n"
            "[CONDUITS]\nC1   J1  J2  50  0.01  0  0  0\nC2   J2  O1  50  0.01  0  0  0\n"
        )
        self.assertEqual(geometry["layout"], "auto")
        self.assertEqual(geometry["counts"]["points"], 3 + 2 * 2)

    def test_nan_coordinate_treated_as_missing(self):
        geometry = self.model_from_text(
            "[JUNCTIONS]\nJ1   10.0  2.0  0  0  0\nJ2   9.0   2.0  0  0  0\n\n"
            "[CONDUITS]\nC1   J1  J2  50  0.01  0  0  0\n\n"
            "[COORDINATES]\nJ1   0.0   10.0\nJ2   NaN   0.0\n"
        )
        self.assertEqual(geometry["layout"], "auto")
        self.assertEqual(len(geometry["nodes"]), 2)

    def test_link_carries_endpoint_names(self):
        geometry = self.fixture("geom_crlf_vertices.inp")
        first = next(link for link in geometry["links"] if link["n"] == "C1")
        self.assertEqual((first["a"], first["b"]), ("J1", "J2"))

    def test_node_and_link_may_share_a_name(self):
        """SWMM allows it, so the two collections are joined separately downstream."""
        geometry = self.fixture("geom_same_name.inp")
        self.assertIn("X", [node["n"] for node in geometry["nodes"]])
        self.assertIn("X", [link["n"] for link in geometry["links"]])

    def test_min_gap_computed(self):
        geometry = self.fixture("geom_crlf_vertices.inp")
        positions = [node["p"] for node in geometry["nodes"]]
        self.assertAlmostEqual(geometry["min_gap"], brute_force_min_gap(positions), places=4)
        self.assertEqual(geometry["min_gap"], 50.0)

    def test_min_gap_is_none_with_fewer_than_two_distinct_positions(self):
        geometry = self.model_from_text("[JUNCTIONS]\nJ1   10.0  2.0  0  0  0\n\n[COORDINATES]\nJ1   5.0  5.0\n")
        self.assertIsNone(geometry["min_gap"])

    def test_min_gap_positive_with_coincident_nodes(self):
        """Two nodes on one point must not collapse the gap to zero for the whole map."""
        geometry = self.fixture("geom_coincident.inp")
        positions = [node["p"] for node in geometry["nodes"]]
        self.assertGreater(geometry["min_gap"], 0.0)
        self.assertAlmostEqual(geometry["min_gap"], brute_force_min_gap(positions), places=4)
        only_coincident = self.model_from_text(
            "[JUNCTIONS]\nJ1   10.0  2.0  0  0  0\nJ2   9.0   2.0  0  0  0\n\n[COORDINATES]\nJ1   5.0  5.0\nJ2   5.0  5.0\n"
        )
        self.assertIsNone(only_coincident["min_gap"])

    def test_zero_bbox_is_raised_to_one(self):
        geometry = self.model_from_text("[JUNCTIONS]\nJ1   10.0  2.0  0  0  0\n\n[COORDINATES]\nJ1   5.0  5.0\n")
        self.assertEqual(geometry["bbox"], {"w": 1.0, "h": 1.0})

    def test_version_is_stamped(self):
        self.assertEqual(self.fixture("geom_crlf_vertices.inp")["version"], NETMAP_VERSION)
        self.assertEqual(NETMAP_VERSION, 2)

    def test_bbox_includes_link_vertices(self):
        """A conduit bent outside the hull of the nodes widens the box instead of being clipped."""
        geometry = self.model_from_text(
            "[JUNCTIONS]\nJ1   10.0  2.0  0  0  0\nJ2   9.0   2.0  0  0  0\n\n"
            "[CONDUITS]\nC1   J1  J2  50  0.01  0  0  0\n\n"
            "[COORDINATES]\nJ1   0.0   0.0\nJ2   10.0  0.0\n\n"
            "[VERTICES]\nC1   -20.0  30.0\n"
        )
        self.assertEqual(geometry["origin"], [-20.0, 30.0])
        self.assertGreaterEqual(geometry["bbox"]["w"], 30.0)
        self.assertGreaterEqual(geometry["bbox"]["h"], 30.0)
        drawn = [point for link in geometry["links"] for point in link["p"]] + [node["p"] for node in geometry["nodes"]]
        for x, y in drawn:
            self.assertGreaterEqual(x, 0.0)
            self.assertGreaterEqual(y, 0.0)

    def test_empty_network_defaults(self):
        geometry = self.fixture("geom_empty.inp")
        self.assertEqual(geometry["origin"], [0.0, 0.0])
        self.assertEqual(geometry["bbox"], {"w": 1.0, "h": 1.0})
        self.assertIsNone(geometry["min_gap"])
        self.assertEqual(geometry["nodes"], [])
        self.assertEqual(geometry["links"], [])
        self.assertEqual(geometry["polys"], [])
        self.assertEqual(geometry["skipped"], {"links_dangling": 0, "polys_degenerate": 0, "polys_no_outlet": 0})
        self.assertEqual(geometry["counts"]["elements"], 0)

    def test_quoted_name_with_space_survives(self):
        """Without the quotes swmmio would silently shorten ``S 1`` to ``S``."""
        geometry = self.fixture("geom_crlf_vertices.inp")
        self.assertEqual([poly["n"] for poly in geometry["polys"]], ["S 1"])

    def test_layout_flag(self):
        complete = self.fixture("geom_crlf_vertices.inp")
        partial = self.fixture("geom_partial_coords.inp")
        self.assertEqual(complete["layout"], "inp")
        self.assertEqual(partial["layout"], "auto")
        for geometry in (complete, partial):
            self.assertTrue(all(len(node["p"]) == 2 for node in geometry["nodes"]))

    def test_storage_node_kind(self):
        geometry = self.fixture("geom_storage.inp")
        kinds = {node["n"]: node["k"] for node in geometry["nodes"]}
        self.assertEqual(kinds["ST1"], "storage")
        self.assertEqual(kinds["DV1"], "other")
        self.assertEqual(kinds["J1"], "junction")

    def test_polygon_without_points_counted_in_skipped(self):
        geometry = self.model_from_text(
            "[JUNCTIONS]\nJ1   10.0  2.0  0  0  0\nJ2   9.0   2.0  0  0  0\n\n"
            "[COORDINATES]\nJ1   0.0   0.0\nJ2   10.0  10.0\n\n"
            "[Polygons]\nS1   0.0  0.0\nS1   1.0  1.0\n"
        )
        self.assertEqual(geometry["skipped"]["polys_degenerate"], 1)
        self.assertEqual(geometry["polys"], [])


class OutletGeometryTests(GeometryFixtureMixin, SimpleTestCase):
    """Version 2 of the payload: the centroid of every outline and the outlet line.

    The contract under test has two halves that are deliberately not symmetric.  ``c``
    is emitted for **every** polygon of the ``"inp"`` layout, because it anchors the
    subcatchment's label and a subcatchment whose outlet nobody recognised still has a
    name.  ``o`` and ``on`` are emitted **together or not at all**, because a line needs
    both ends and a name to say where it goes.
    """

    def outlet_model(self, subcatchments: str = "", polygons: str = "", coordinates: str = "") -> dict:
        """A two-junction model with the [SUBCATCHMENTS] and [Polygons] blocks given.

        The nodes and the frame around them are the same in every case here, so a test
        reads as the one thing it varies.  ``J1`` sits at ``(0, 0)`` and ``J2`` at
        ``(50, 50)``; *coordinates* adds further rows to [COORDINATES] without making
        them nodes of the model.
        """
        text = (
            "[JUNCTIONS]\nJ1   10.0  2.0  0  0  0\nJ2   9.0   2.0  0  0  0\n\n"
            f"[SUBCATCHMENTS]\n{subcatchments}\n"
            f"[COORDINATES]\nJ1   0.0   0.0\nJ2   50.0  50.0\n{coordinates}\n"
            f"[Polygons]\n{polygons}\n"
        )
        return self.model_from_text(text)

    @staticmethod
    def poly(geometry: dict, name: str) -> dict:
        return next(poly for poly in geometry["polys"] if poly["n"] == name)

    def assertInsideOwnBox(self, poly: dict) -> None:
        """The centroid lies inside the bounding box of the outline's own vertices."""
        xs = [point[0] for point in poly["p"]]
        ys = [point[1] for point in poly["p"]]
        self.assertTrue(math.isfinite(poly["c"][0]) and math.isfinite(poly["c"][1]), poly["c"])
        self.assertGreaterEqual(poly["c"][0], min(xs))
        self.assertLessEqual(poly["c"][0], max(xs))
        self.assertGreaterEqual(poly["c"][1], min(ys))
        self.assertLessEqual(poly["c"][1], max(ys))

    def test_poly_carries_outlet_point(self):
        """A recognised outlet gives the far end of the line and the name of the receiver."""
        geometry = self.fixture("geom_outlets.inp")
        nodes = {node["n"]: node["p"] for node in geometry["nodes"]}
        poly = self.poly(geometry, "S1")
        self.assertEqual(poly["o"], nodes["J2"])
        self.assertEqual(poly["on"], "J2")
        self.assertInsideOwnBox(poly)

    def test_poly_outlet_is_flipped_and_rounded(self):
        """``c`` and ``o`` go through the same normalisation as ``p``, and nowhere else."""
        geometry = self.outlet_model(
            subcatchments="S1   RG1  J2  1.0  50  100  0.5  0",
            polygons="S1   0.0   0.0\nS1   10.0  0.0\nS1   10.003  9.999",
        )
        poly = self.poly(geometry, "S1")
        # y_top is 50 (J2), x0 is 0 (J1), so the node at (50, 50) normalises to (50, 0).
        self.assertEqual(poly["o"], [50.0, 0.0])
        for value in (*poly["c"], *poly["o"]):
            self.assertEqual(value, round(value, 2))

    def test_poly_without_subcatchment_row_has_no_outlet(self):
        """An outline no [SUBCATCHMENTS] row names gets a centroid and no line.

        ``geom_crlf_vertices.inp`` has a [Polygons] block and no [SUBCATCHMENTS] section
        at all, so its single outline cannot have an outlet.  That is the correct result
        and not a regression: the polygon is still drawn and still labelled, which is
        why ``c`` is there and the pair ``o``/``on`` is not.

        The counter stays at zero.  It names outlet lines whose target the model does
        not know (D18), and an outline no row names carries no target and attempted no
        line — there is nothing about it the browser should report as unresolved.
        """
        geometry = self.fixture("geom_crlf_vertices.inp")
        poly = self.poly(geometry, "S 1")
        self.assertIn("c", poly)
        self.assertNotIn("o", poly)
        self.assertNotIn("on", poly)
        self.assertEqual(geometry["skipped"]["polys_no_outlet"], 0)

    def test_counts_grow_with_each_outlet_line(self):
        """One resolved pair costs one element and one point; a centroid costs a point.

        The two fixtures differ by the [SUBCATCHMENTS] section alone, so the difference
        between their counts is exactly what one outlet line weighs.  ``c`` is in both,
        because it does not depend on the outlet — hence one point per pair and not two.
        """
        with_outlets = self.fixture("geom_outlets.inp")
        without = self.fixture("geom_outlets_no_subcatchments.inp")
        lines = sum(1 for poly in with_outlets["polys"] if "o" in poly)
        self.assertEqual(lines, 1)
        self.assertEqual(with_outlets["counts"]["elements"] - without["counts"]["elements"], lines)
        self.assertEqual(with_outlets["counts"]["points"] - without["counts"]["points"], lines)

        # The other half of the arithmetic: the centroids are in ``points`` as well.
        drawn = (
            len(without["nodes"])
            + sum(len(link["p"]) for link in without["links"])
            + sum(len(poly["p"]) for poly in without["polys"])
        )
        self.assertEqual(without["counts"]["points"] - drawn, len(without["polys"]))

    def test_degenerate_outlet_emits_no_triple(self):
        """A line of zero length is no line: the pair is dropped, the centroid stays.

        "No triple" is the old name of this case; since ``c`` became unconditional it
        means "no ``o``/``on`` pair".  The polygon keeps its label anchor.  The counter
        of missing lines is **not** raised either: the target is perfectly well known
        and there is simply nothing to draw, which is a different thing from a target
        nobody recognised.
        """
        geometry = self.outlet_model(
            subcatchments="S1   RG1  J2  1.0  50  100  0.5  0",
            polygons="S1   25.0  25.0\nS1   75.0  25.0\nS1   75.0  75.0\nS1   25.0  75.0",
        )
        poly = self.poly(geometry, "S1")
        nodes = {node["n"]: node["p"] for node in geometry["nodes"]}
        self.assertEqual(poly["c"], nodes["J2"])
        self.assertNotIn("o", poly)
        self.assertNotIn("on", poly)
        self.assertEqual(geometry["skipped"]["polys_no_outlet"], 0)
        self.assertEqual(geometry["counts"]["elements"], len(geometry["polys"]) + len(geometry["nodes"]))

    def test_outlet_pointing_at_unknown_node_is_skipped(self):
        """A name the model does not carry is not a receiver; the centroid survives it."""
        geometry = self.fixture("geom_outlets.inp")
        poly = self.poly(geometry, "S2")
        self.assertIn("c", poly)
        self.assertNotIn("o", poly)
        self.assertNotIn("on", poly)
        self.assertEqual(geometry["skipped"]["polys_no_outlet"], 1)

    def test_outlet_that_is_a_subcatchment_uses_its_centroid(self):
        """A subcatchment may drain into another one, and the line ends at its centroid."""
        geometry = self.outlet_model(
            subcatchments=("S1   RG1  S2  1.0  50  100  0.5  0\nS2   RG1  J1  1.0  50  100  0.5  0"),
            polygons=(
                "S1   0.0   0.0\nS1   10.0  0.0\nS1   10.0  10.0\nS1   0.0   10.0\n"
                "S2   30.0  30.0\nS2   40.0  30.0\nS2   40.0  40.0\nS2   30.0  40.0"
            ),
        )
        first, second = self.poly(geometry, "S1"), self.poly(geometry, "S2")
        self.assertEqual(first["o"], second["c"])
        self.assertEqual(first["on"], "S2")
        self.assertEqual(geometry["skipped"]["polys_no_outlet"], 0)

    def test_outlet_chain_is_not_followed(self):
        """One hop and no more: ``S1 -> S2 -> J1`` ends the line of ``S1`` at ``S2``."""
        geometry = self.outlet_model(
            subcatchments=("S1   RG1  S2  1.0  50  100  0.5  0\nS2   RG1  J1  1.0  50  100  0.5  0"),
            polygons=(
                "S1   0.0   0.0\nS1   10.0  0.0\nS1   10.0  10.0\nS1   0.0   10.0\n"
                "S2   30.0  30.0\nS2   40.0  30.0\nS2   40.0  40.0\nS2   30.0  40.0"
            ),
        )
        nodes = {node["n"]: node["p"] for node in geometry["nodes"]}
        first, second = self.poly(geometry, "S1"), self.poly(geometry, "S2")
        self.assertEqual(first["o"], second["c"])
        self.assertNotEqual(first["o"], nodes["J1"])
        self.assertEqual(second["o"], nodes["J1"])

    def test_outlet_name_in_coordinates_but_not_a_node(self):
        """[COORDINATES] names more than the network does, and only the network counts.

        A rain gauge, a label or a leftover row carries a coordinate without being a
        node of the model.  Resolving against the coordinate map instead of the set of
        model nodes would send the line to something that is not in the network at all.
        """
        geometry = self.outlet_model(
            subcatchments="S1   RG1  RG1  1.0  50  100  0.5  0",
            coordinates="RG1  20.0  20.0",
            polygons="S1   0.0   0.0\nS1   10.0  0.0\nS1   10.0  10.0",
        )
        poly = self.poly(geometry, "S1")
        self.assertIn("c", poly)
        self.assertNotIn("o", poly)
        self.assertEqual(geometry["skipped"]["polys_no_outlet"], 1)

    def test_nan_outlet_treated_as_missing(self):
        """A row without an outlet has no outlet; the string ``"nan"`` never reaches ``on``."""
        geometry = self.outlet_model(
            subcatchments="S1   RG1",
            polygons="S1   0.0   0.0\nS1   10.0  0.0\nS1   10.0  10.0",
        )
        poly = self.poly(geometry, "S1")
        self.assertNotIn("on", poly)
        self.assertNotIn("o", poly)
        self.assertNotIn("nan", json.dumps(geometry))

    def test_centroid_of_collinear_polygon_falls_back_to_mean(self):
        """Three collinear points pass the point count and have no area to weigh by."""
        geometry = self.outlet_model(
            subcatchments="S1   RG1  J1  1.0  50  100  0.5  0",
            polygons="S1   0.0   0.0\nS1   10.0  10.0\nS1   20.0  20.0",
        )
        poly = self.poly(geometry, "S1")
        # Mean of the vertices, normalised with x0 = 0 and y_top = 50.
        self.assertEqual(poly["c"], [10.0, 40.0])
        self.assertInsideOwnBox(poly)

    def test_self_intersecting_polygon_centroid_stays_in_its_bbox(self):
        """The counterexample of the review, in both orientations.

        Nothing rejects a self-intersecting outline: an outline is kept on the number of
        usable points alone.  For ``(0, 0), (10, 10), (0, 10), (8, 0)`` the signed area
        is 10 — so the "zero area" fallback does not fire — and the shoelace formula
        answers ``(6, 20)``, outside ``[0, 10] x [0, 10]``.  The vertical mirror
        ``(0, 10), (10, 0), (0, 0), (8, 10)`` answers ``(6, -10)``.  One of the two
        would fall outside the frame and the other would enlarge it, so the fallback on
        the bounding box of the outline's own vertices is what keeps ``bbox`` a promise.
        """
        for polygon in (
            "S1   0.0   0.0\nS1   10.0  10.0\nS1   0.0   10.0\nS1   8.0   0.0",
            "S1   0.0   10.0\nS1   10.0  0.0\nS1   0.0   0.0\nS1   8.0   10.0",
        ):
            with self.subTest(polygon=polygon):
                geometry = self.outlet_model(
                    subcatchments="S1   RG1  J1  1.0  50  100  0.5  0",
                    polygons=polygon,
                )
                self.assertInsideOwnBox(self.poly(geometry, "S1"))

    def test_outlet_pointing_at_orphan_polygon_is_skipped(self):
        """An outline is not a subcatchment: [Polygons] names shapes, not model objects.

        ``ORPHAN`` has an outline and no [SUBCATCHMENTS] row, so it is not a
        subcatchment of the model and cannot receive a line — exactly as an unknown node
        name cannot.  It is drawn and it gets a centroid all the same, and so does the
        polygon that pointed at it.
        """
        geometry = self.outlet_model(
            subcatchments="S1   RG1  ORPHAN  1.0  50  100  0.5  0",
            polygons=(
                "S1       0.0   0.0\nS1       10.0  0.0\nS1       10.0  10.0\n"
                "ORPHAN   30.0  30.0\nORPHAN   40.0  30.0\nORPHAN   40.0  40.0"
            ),
        )
        for name in ("S1", "ORPHAN"):
            poly = self.poly(geometry, name)
            self.assertIn("c", poly)
            self.assertNotIn("o", poly)
            self.assertNotIn("on", poly)
        # Both polygons go without a line, but only ``S1`` tried: its target is not a
        # subcatchment of the model.  ``ORPHAN`` has no row and so no target at all,
        # which is not an unresolved line and does not raise the counter.
        self.assertEqual(geometry["skipped"]["polys_no_outlet"], 1)

    def test_centroid_with_large_coordinates_stays_finite_and_inside_bbox(self):
        """A five-centimetre square in PL-2000 coordinates, where the shoelace sums cancel.

        The author's models live around ``(6 500 000, 5 700 000)``, and a small parcel
        there is the classic loss-of-precision case: the cross products are of the order
        of 10^13 and their sum of the order of 10^-3.  Conditions (2) and (3) of the
        fallback catch numerical degeneration just as they catch a self-intersection.
        """
        geometry = self.outlet_model(
            subcatchments="S1   RG1  J1  1.0  50  100  0.5  0",
            polygons=(
                "S1   6500000.00  5700000.00\nS1   6500000.05  5700000.00\n"
                "S1   6500000.05  5700000.05\nS1   6500000.00  5700000.05"
            ),
        )
        self.assertInsideOwnBox(self.poly(geometry, "S1"))

    def test_centroid_of_an_outline_at_the_float_ceiling_stays_finite(self):
        """F6 - the *fallback* is finite too, which is what the three conditions rely on.

        The previous test measures the loss of precision of the shoelace formula at
        realistic coordinates; this one measures the arithmetic of the fallback itself,
        at the ceiling of a double.  Two vertices at 1e308 make ``sum()`` overflow to
        an infinity, and an infinity divided by four is still an infinity, so the mean
        of the vertices - the value every one of the three conditions returns - was not
        finite for this input and a non-finite ``c`` reached the payload.  The shoelace
        sums overflow here as well (the cross products are of the order of 1e616), so
        condition (2) fires and the fallback is exactly what answers.
        """
        geometry = self.outlet_model(
            subcatchments="S1   RG1  J1  1.0  50  100  0.5  0",
            polygons=("S1   0.0     0.0\nS1   1e308   0.0\nS1   1e308   1e308\nS1   0.0     1e308"),
        )
        self.assertInsideOwnBox(self.poly(geometry, "S1"))

    def test_numeric_node_name_resolves_an_outlet_read_as_a_float(self):
        """F5 - one blank cell in [SUBCATCHMENTS] must not unname a numeric node.

        pandas types a section column by column.  A model whose junctions are called
        ``101``/``102`` has an int64 index, and ``str()`` on it answers ``"102"``; but a
        single row without an outlet promotes the ``Outlet`` column to float64, where
        ``str()`` answers ``"102.0"`` - a name no node of the model carries.  The line
        was then dropped and counted as one whose target the model does not know, for a
        target the model does carry.  :func:`sa.core.geometry._name` is the one place
        that reads a name now, so both sides of the lookup answer ``"102"``.

        The blank row is what makes the promotion happen and is left out of the payload
        by the NaN guard that predates this fix, so the assertion on ``S2`` is that it
        contributes nothing, neither a line nor a count.
        """
        geometry = self.model_from_text(
            "[JUNCTIONS]\n101   10.0  2.0  0  0  0\n102   9.0   2.0  0  0  0\n\n"
            "[SUBCATCHMENTS]\nS1   RG1  102  1.0  50  100  0.5  0\nS2   RG1\n\n"
            "[COORDINATES]\n101   0.0   0.0\n102   50.0  50.0\n\n"
            "[Polygons]\nS1   0.0   0.0\nS1   10.0  0.0\nS1   10.0  10.0\nS1   0.0   10.0\n"
        )
        poly = self.poly(geometry, "S1")
        self.assertEqual(poly["on"], "102")
        self.assertEqual(poly["o"], [50.0, 0.0])
        self.assertEqual(geometry["skipped"]["polys_no_outlet"], 0)
        self.assertEqual([node["n"] for node in geometry["nodes"]], ["101", "102"])

    def test_bbox_unchanged_by_outlet_points(self):
        """The frame is set by what was already drawn; a line adds no new extreme.

        ``o`` is either a node position or another outline's centroid, and a centroid
        lies inside the box of its own vertices, so neither can reach past the drawing.
        Measured on the pair of tracked fixtures and, because that is the only input on
        which the property could have broken, on a self-intersecting outline as well.
        """
        with_outlets = self.fixture("geom_outlets.inp")
        without = self.fixture("geom_outlets_no_subcatchments.inp")
        self.assertEqual(with_outlets["bbox"], without["bbox"])
        self.assertEqual(with_outlets["origin"], without["origin"])

        polygons = "S1   0.0   0.0\nS1   10.0  10.0\nS1   0.0   10.0\nS1   8.0   0.0"
        bent = self.outlet_model(subcatchments="S1   RG1  J2  1.0  50  100  0.5  0", polygons=polygons)
        plain = self.outlet_model(polygons=polygons)
        self.assertIn("o", self.poly(bent, "S1"))
        self.assertEqual(bent["bbox"], plain["bbox"])
        self.assertEqual(bent["origin"], plain["origin"])


class AutoLayoutTests(GeometryFixtureMixin, SimpleTestCase):
    """``auto_layout`` — the layered schematic drawn when the file has no coordinates."""

    def test_auto_is_deterministic(self):
        """Same model, same payload — byte for byte, whatever order the inputs arrive in."""
        first = self.fixture("geom_no_coords_network.inp")
        second = self.fixture("geom_no_coords_network.inp")
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

        names = ["J1", "J2", "J3", "J4", "J5", "O1"]
        links = [("J1", "J3"), ("J2", "J3"), ("J3", "J4"), ("J5", "J4"), ("J4", "O1")]
        self.assertEqual(
            auto_layout(names, links, ["O1"]),
            auto_layout(list(reversed(names)), list(reversed(links)), ["O1"]),
        )

    def test_auto_tree_layers(self):
        """A layer is the number of hops to the outfall; spacing between layers is one unit."""
        positions = auto_layout(
            ["J1", "J2", "J3", "J4", "J5", "O1"],
            [("J1", "J3"), ("J2", "J3"), ("J3", "J4"), ("J5", "J4"), ("J4", "O1")],
            ["O1"],
        )
        deepest = max(y for _, y in positions.values())
        layer_of = {name: deepest - y for name, (_, y) in positions.items()}
        self.assertEqual(layer_of, {"O1": 0.0, "J4": 1.0, "J3": 2.0, "J5": 2.0, "J1": 3.0, "J2": 3.0})

    def test_auto_two_outfalls(self):
        positions = auto_layout(
            ["A1", "A2", "OA", "B1", "B2", "OB"],
            [("A1", "A2"), ("A2", "OA"), ("B1", "B2"), ("B2", "OB")],
            ["OA", "OB"],
        )
        first = {positions[name][0] for name in ("A1", "A2", "OA")}
        second = {positions[name][0] for name in ("B1", "B2", "OB")}
        self.assertFalse(first & second)
        self.assertAlmostEqual(min(second) - max(first), COMPONENT_GAP, places=2)

    def test_auto_cycle_does_not_hang(self):
        """A model that is one cycle has neither an outfall nor a sink to start from."""
        positions = auto_layout(["A", "B", "C"], [("A", "B"), ("B", "C"), ("C", "A")], [])
        self.assertEqual(sorted(positions), ["A", "B", "C"])
        self.assertEqual(len({position for position in positions.values()}), 3)

    def test_auto_disconnected_component(self):
        positions = auto_layout(["A", "B", "O", "X", "Y"], [("A", "B"), ("B", "O"), ("X", "Y")], ["O"])
        connected = {positions[name][0] for name in ("A", "B", "O")}
        separate = {positions[name][0] for name in ("X", "Y")}
        self.assertFalse(connected & separate)
        self.assertGreaterEqual(min(separate) - max(connected), COMPONENT_GAP)
        # The stray component keeps its own orientation: its sink sits at the bottom.
        self.assertEqual(len(separate), 1)
        self.assertGreater(positions["Y"][1], positions["X"][1])

    def test_auto_single_node(self):
        self.assertEqual(auto_layout(["A"], [], []), {"A": (0.0, 0.0)})

    def test_auto_isolated_node(self):
        positions = auto_layout(["A", "B", "O", "ISO"], [("A", "B"), ("B", "O")], ["O"])
        self.assertIn("ISO", positions)
        self.assertNotIn(positions["ISO"][0], {positions[name][0] for name in ("A", "B", "O")})

    def test_auto_has_no_polygons(self):
        """The schematic invents no outlines; subcatchments stay in their table."""
        geometry = self.fixture("geom_no_coords_network.inp")
        self.assertEqual(geometry["layout"], "auto")
        self.assertEqual(geometry["polys"], [])
        self.assertEqual(geometry["skipped"]["polys_degenerate"], 0)
        # No outline means no line to a receiver either, and no polygon to miss one.
        self.assertEqual(geometry["skipped"]["polys_no_outlet"], 0)
        self.assertEqual(geometry["origin"], [0.0, 0.0])

    def test_partial_coordinates_switch_whole_map_to_auto(self):
        """One node short of a full [COORDINATES] sends the whole network to the schematic."""
        geometry = self.fixture("geom_partial_coords.inp")
        self.assertEqual(geometry["layout"], "auto")
        self.assertEqual([node["n"] for node in geometry["nodes"]], ["J1", "J2", "O1"])
        self.assertEqual([link["n"] for link in geometry["links"]], ["C1", "C2"])
        self.assertEqual(
            geometry["skipped"],
            {"links_dangling": 0, "polys_degenerate": 0, "polys_no_outlet": 0},
        )

    def test_auto_orientation_outfall_at_bottom(self):
        """The schematic is already in screen space — the Y flip must not be applied twice."""
        geometry = self.fixture("geom_no_coords_network.inp")
        points = {node["n"]: node["p"] for node in geometry["nodes"]}
        self.assertEqual(points["O1"][1], max(point[1] for point in points.values()))

    def test_auto_two_thousand_nodes_within_budget(self):
        """The threshold catches an algorithmic regression, it does not measure the machine."""
        names = [f"N{index}" for index in range(2000)]
        links = [(names[index], names[(index - 1) // 3]) for index in range(1, 2000)]
        started = time.perf_counter()
        positions = auto_layout(names, links, ["N0"])
        elapsed = time.perf_counter() - started
        self.assertEqual(len(positions), 2000)
        self.assertLess(elapsed, 0.25)


class SaveGeometryTests(GeometryFixtureMixin, TestCase):
    """Geometry is captured when results are saved - and never costs them anything."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # A DataManager runs no simulation, but swmmio and the recommendation models make
        # the constructor slow, so the whole class shares one.  It is built on a *copy* of
        # the tracked model: nothing in this file may touch a tracked .inp/.rpt/.out.
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        workdir = Path(directory.name)
        for suffix in (".inp", ".rpt", ".out"):
            shutil.copy(MODEL_DIR / (MODEL_STEM + suffix), workdir / (MODEL_STEM + suffix))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cls.data_manager = DataManager(str(workdir / (MODEL_STEM + ".inp")))

    def setUp(self):
        self.user = make_user("geometry@example.com")
        self.session = make_session(self.user, status="processing")

    def save(self) -> bool:
        """Save the results of the shared model, with swmmio's section warnings silenced.

        The sections a small model lacks are parsed lazily, on the first access inside
        ``extract_geometry`` - so the filter belongs here, not only around the constructor.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return CalculationPersistenceService.save_calculation_results(self.session, self.data_manager)

    def assertResultsSaved(self) -> None:
        """The three result tables and the session status - what a failure must not touch."""
        self.assertTrue(ConduitData.objects.filter(session=self.session).exists())
        self.assertTrue(NodeData.objects.filter(session=self.session).exists())
        self.assertTrue(SubcatchmentData.objects.filter(session=self.session).exists())
        self.session.refresh_from_db()
        self.assertEqual(self.session.status, "completed")

    def test_analysis_writes_geometry_row(self):
        """Saving results stores one payload of the current contract for the session."""
        self.assertTrue(self.save())
        geometry = SessionGeometry.objects.get(session=self.session)
        self.assertEqual(geometry.data["version"], NETMAP_VERSION)
        self.assertGreater(geometry.data["counts"]["elements"], 0)

    def test_extraction_failure_does_not_fail_analysis(self):
        """A broken extraction is logged and forgotten; the analysis still completes."""
        with patch("sa.services.extract_geometry", side_effect=ValueError("boom")):
            with self.assertLogs("sa.services", level="ERROR") as logs:
                self.assertTrue(self.save())
        self.assertEqual(len(logs.records), 1)
        self.assertFalse(SessionGeometry.objects.filter(session=self.session).exists())
        self.assertResultsSaved()

    def test_geometry_write_failure_keeps_results(self):
        """A real IntegrityError from the engine is undone by the savepoint alone.

        The row is pre-created and ``update_or_create`` is rerouted to ``create``, so the
        INSERT genuinely violates the one-to-one constraint inside the database.  A mock
        raising on the Python side would never reach the connection, Django would not set
        ``needs_rollback``, and the test would pass without a savepoint too.
        """
        SessionGeometry.objects.create(session=self.session, data={})

        def insert_again(**kwargs):
            return SessionGeometry.objects.create(session=kwargs["session"], data=kwargs["defaults"]["data"])

        with patch.object(SessionGeometry.objects, "update_or_create", side_effect=insert_again):
            with self.assertLogs("sa.services", level="ERROR") as logs:
                self.assertTrue(self.save())
        self.assertEqual(len(logs.records), 1)
        self.assertResultsSaved()
        self.assertEqual(SessionGeometry.objects.filter(session=self.session).count(), 1)
        self.assertEqual(SessionGeometry.objects.get(session=self.session).data, {})

    def test_getattr_returns_none_for_old_session(self):
        """A session analysed before this stage has no row, and reading it is not an error."""
        self.assertIsNone(getattr(self.session, "geometry", None))
        fetched = CalculationSession.objects.select_related("geometry").get(pk=self.session.pk)
        self.assertIsNone(getattr(fetched, "geometry", None))

    def test_geometry_is_json_serializable(self):
        """The stored payload is plain JSON - no encoder hook stands between it and the page."""
        self.assertTrue(self.save())
        geometry = SessionGeometry.objects.get(session=self.session)
        self.assertIsInstance(json.dumps(geometry.data), str)

    def test_nan_is_sanitized(self):
        """No NaN or Infinity reaches the payload: RFC 8259 has no word for them."""
        text = (
            "[JUNCTIONS]\nJ1 10 2 0 0 0\nJ2 9 2 0 0 0\n\n"
            "[OUTFALLS]\nO1 8 FREE NO\n\n"
            "[CONDUITS]\nC1 J1 J2 100 0.01 0 0 0\nC2 J2 O1 100 0.01 0 0 0\n\n"
            "[COORDINATES]\nJ1 0 0\nJ2 NaN 10\nO1 20 20\n"
        )
        payload = self.model_from_text(text)
        self.assertIsInstance(json.dumps(payload, allow_nan=False), str)

    def test_history_view_does_not_load_geometry(self):
        """The history page lists whole sessions; it must not drag the payloads along."""
        self.assertTrue(self.save())
        self.client.force_login(self.user)
        with CaptureQueriesContext(connection) as queries:
            response = self.client.get(reverse("sa:history"))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(any("sa_sessiongeometry" in query["sql"] for query in queries.captured_queries))
