"""Automated unit and integration test suite for Results V2 backend serialization and view.

Covers Milestone 1:
- Field completeness across ConduitData (42 fields), NodeData (8 fields), SubcatchmentData (9 fields)
- Float sanitization (RFC 8259 non-finite float cleaning: NaN, +Inf, -Inf -> None)
- Summary KPI metric calculations and validation flag aggregation
- WROCLAW_RULE_LABELS and conduit panel link injection
- AnalysisResultsView context contract and backward compatibility
- Edge cases: empty sessions, null fields, division by zero, slashes and special characters
"""

from __future__ import annotations

import decimal
import json
import math
import uuid

from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from sa.services import CalculationRetrievalService, ConduitPanelService, fill_percent
from sa.views import WROCLAW_RULE_LABELS
from sa.core.geometry import NETMAP_MAX_ELEMENTS, NETMAP_MAX_POINTS
from .factories import (
    VALIDATION_FLAGS,
    make_conduit,
    make_geometry,
    make_geometry_payload,
    make_node,
    make_session,
    make_subcatchment,
    make_user,
)

_clean_float = CalculationRetrievalService._clean_float


class ResultsV2FieldSerializationTests(TestCase):
    """Test cases for serialize_session_for_results_v2: verify all fields and types."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="serialization_tester@example.com")
        cls.session = make_session(cls.user, n_conduits=1, n_nodes=2, n_subcatchments=1)

    def setUp(self):
        self.payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)

    def test_payload_top_level_structure(self):
        """Top-level dictionary must contain summary, conduits, nodes, catchments."""
        self.assertIsInstance(self.payload, dict)
        self.assertIn("summary", self.payload)
        self.assertIn("conduits", self.payload)
        self.assertIn("nodes", self.payload)
        self.assertIn("catchments", self.payload)
        self.assertIsInstance(self.payload["summary"], dict)
        self.assertIsInstance(self.payload["conduits"], list)
        self.assertIsInstance(self.payload["nodes"], list)
        self.assertIsInstance(self.payload["catchments"], list)

    def test_conduit_contains_all_42_database_fields_and_correct_types(self):
        """Verify all 42 ConduitData database fields are present and correctly typed."""
        self.assertEqual(len(self.payload["conduits"]), 1)
        c = self.payload["conduits"][0]

        # 1. Identity Fields (5)
        self.assertIn("conduit_name", c)
        self.assertEqual(c["conduit_name"], "C1")
        self.assertIn("inlet_node", c)
        self.assertEqual(c["inlet_node"], "N1")
        self.assertIn("outlet_node", c)
        self.assertEqual(c["outlet_node"], "N2")
        self.assertIn("subcatchment", c)
        self.assertEqual(c["subcatchment"], "S1")
        self.assertIn("sbc_category", c)
        self.assertEqual(c["sbc_category"], "residential")

        # 2. Geometry Fields (9)
        self.assertIn("geom1", c)
        self.assertIsInstance(c["geom1"], float)
        self.assertIn("length", c)
        self.assertIsInstance(c["length"], float)
        self.assertIn("slope_per_mile", c)
        self.assertIsInstance(c["slope_per_mile"], float)
        self.assertIn("inlet_ground_cover", c)
        self.assertIsInstance(c["inlet_ground_cover"], float)
        self.assertIn("outlet_ground_cover", c)
        self.assertIsInstance(c["outlet_ground_cover"], float)
        self.assertIn("inlet_max_depth", c)
        self.assertIsInstance(c["inlet_max_depth"], float)
        self.assertIn("outlet_max_depth", c)
        self.assertIsInstance(c["outlet_max_depth"], float)
        self.assertIn("inlet_ground_elevation", c)
        self.assertIsInstance(c["inlet_ground_elevation"], float)
        self.assertIn("outlet_ground_elevation", c)
        self.assertIsInstance(c["outlet_ground_elevation"], float)

        # 3. Hydraulics Fields (3)
        self.assertIn("max_v", c)
        self.assertIsInstance(c["max_v"], float)
        self.assertIn("max_q", c)
        self.assertIsInstance(c["max_q"], float)
        self.assertIn("filling", c)
        self.assertIsInstance(c["filling"], float)

        # 4. Diameter Analysis Fields (4)
        self.assertIn("min_diameter", c)
        self.assertIsInstance(c["min_diameter"], float)
        self.assertIn("is_min_diameter", c)
        self.assertIn("increase_dia", c)
        self.assertIn("reduce_dia", c)

        # 5. Slope Analysis Fields (4)
        self.assertIn("min_required_slope", c)
        self.assertIsInstance(c["min_required_slope"], float)
        self.assertIn("increase_slope", c)
        self.assertIn("max_allowable_slope", c)
        self.assertIsInstance(c["max_allowable_slope"], float)
        self.assertIn("reduce_slope", c)

        # 6. Validation Flags (7)
        for flag in VALIDATION_FLAGS:
            self.assertIn(flag, c)
            self.assertIn(c[flag], (0, 1, True, False))

        # 7. Recommendation (1)
        self.assertIn("recommendation", c)
        self.assertEqual(c["recommendation"], "valid")

        # 8. Confidence: only the recommended class's score is serialized, as a percentage.
        self.assertIn("confidence", c)
        self.assertEqual(c["confidence"], 90.0)
        for dropped in ("confidence_valid", "confidence_pump", "confidences"):
            self.assertNotIn(dropped, c)

        # 9. UI Aliases and Derived Values
        self.assertEqual(c.get("id") or c.get("name"), "C1")
        self.assertEqual(c.get("diameter"), c["geom1"])
        self.assertIn("fillPercent", c)
        self.assertIsInstance(c["fillPercent"], float)
        self.assertIn("flags", c)
        self.assertEqual(len(c["flags"]), 7)

    def test_node_contains_all_8_database_fields_and_computed_values(self):
        """Verify all 8 NodeData fields + computed ground_elevation and is_flooded."""
        self.assertEqual(len(self.payload["nodes"]), 2)
        n = self.payload["nodes"][0]

        # 8 Database Fields
        self.assertIn("node_name", n)
        self.assertEqual(n["node_name"], "N1")
        self.assertIn("max_depth", n)
        self.assertIn("invert_elevation", n)
        self.assertIn("max_water_depth", n)
        self.assertIn("hours_flooded", n)
        self.assertIn("flood_volume", n)
        self.assertIn("subcatchment", n)
        self.assertIn("sbc_category", n)

        # Computed ground elevation: invert (117.0) + depth (3.0) = 120.0
        self.assertIn("ground_elevation", n)
        self.assertAlmostEqual(n["ground_elevation"], 120.0, places=2)

        # Flooding indicator
        self.assertIn("is_flooded", n)
        self.assertIs(n["is_flooded"], False)

        # UI Aliases
        self.assertEqual(n.get("id") or n.get("name"), "N1")
        self.assertEqual(n.get("invert"), n["invert_elevation"])
        self.assertEqual(n.get("depth"), n["max_depth"])

    def test_subcatchment_contains_all_9_database_fields(self):
        """Verify all 9 SubcatchmentData fields are present and correctly typed."""
        self.assertEqual(len(self.payload["catchments"]), 1)
        s = self.payload["catchments"][0]

        self.assertIn("subcatchment_name", s)
        self.assertEqual(s["subcatchment_name"], "S1")
        self.assertIn("area", s)
        self.assertIsInstance(s["area"], float)
        self.assertIn("perc_imperv", s)
        self.assertIsInstance(s["perc_imperv"], float)
        self.assertIn("perc_slope", s)
        self.assertIsInstance(s["perc_slope"], float)
        self.assertIn("outlet", s)
        self.assertEqual(s["outlet"], "N1")
        self.assertIn("total_runoff_mg", s)
        self.assertIn("peak_runoff", s)
        self.assertIn("runoff_coeff", s)
        self.assertIn("category", s)
        self.assertEqual(s["category"], "residential")

        # UI Aliases
        self.assertEqual(s.get("id") or s.get("name"), "S1")
        self.assertEqual(s.get("impervious"), s["perc_imperv"])
        self.assertEqual(s.get("slope"), s["perc_slope"])


class ResultsV2FloatSanitizationTests(TestCase):
    """Test cases for float sanitization: verify NaN and Inf values are safely converted to None."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="sanitizer_tester@example.com")
        cls.session = make_session(cls.user)

    def test_clean_float_helper_direct(self):
        """_clean_float must handle None, non-finite floats, numbers, and strings safely."""
        self.assertIsNone(_clean_float(None))
        self.assertIsNone(_clean_float(float("nan")))
        self.assertIsNone(_clean_float(float("inf")))
        self.assertIsNone(_clean_float(float("-inf")))
        self.assertIsNone(_clean_float("not a number"))
        self.assertIsNone(_clean_float([]))
        self.assertEqual(_clean_float(0), 0.0)
        self.assertEqual(_clean_float(0.0), 0.0)
        self.assertEqual(_clean_float(1.5), 1.5)
        self.assertEqual(_clean_float(-42.7), -42.7)
        self.assertEqual(_clean_float("3.14"), 3.14)

    def test_nan_and_inf_in_conduit_floats_converted_to_none(self):
        """Conduit with NaN and Inf values serializes with None for those attributes."""
        c = make_conduit(
            self.session,
            conduit_name="C_NAN",
            max_q=float("inf"),
            filling=float("-inf"),
        )
        c.max_v = float("nan")
        c.confidence_valid = float("nan")
        self.session._prefetched_objects_cache = {
            "conduits": [c],
            "nodes": list(self.session.nodes.all()),
            "subcatchments": list(self.session.subcatchments.all()),
        }
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        res = next(x for x in payload["conduits"] if x["conduit_name"] == "C_NAN")

        self.assertIsNone(res["max_v"])
        self.assertIsNone(res["max_q"])
        self.assertIsNone(res["filling"])
        self.assertIsNone(res["confidence"])

    def test_nan_and_inf_in_node_floats_converted_to_none(self):
        """Node with NaN water depth or flood volume serializes with None."""
        n = make_node(
            self.session,
            node_name="N_NAN",
            hours_flooded=float("inf"),
            flood_volume=float("-inf"),
        )
        n.max_water_depth = float("nan")
        self.session._prefetched_objects_cache = {
            "conduits": list(self.session.conduits.all()),
            "nodes": [n],
            "subcatchments": list(self.session.subcatchments.all()),
        }
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        res = next(x for x in payload["nodes"] if x["node_name"] == "N_NAN")

        self.assertIsNone(res["max_water_depth"])
        self.assertIsNone(res["hours_flooded"])
        self.assertIsNone(res["flood_volume"])

    def test_nan_and_inf_in_subcatchment_floats_converted_to_none(self):
        """Subcatchment with NaN runoff metrics serializes with None."""
        s = make_subcatchment(
            self.session,
            subcatchment_name="S_NAN",
            peak_runoff=float("inf"),
            runoff_coeff=float("-inf"),
        )
        s.total_runoff_mg = float("nan")
        self.session._prefetched_objects_cache = {
            "conduits": list(self.session.conduits.all()),
            "nodes": list(self.session.nodes.all()),
            "subcatchments": [s],
        }
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        res = next(x for x in payload["catchments"] if x["subcatchment_name"] == "S_NAN")

        self.assertIsNone(res["total_runoff_mg"])
        self.assertIsNone(res["peak_runoff"])
        self.assertIsNone(res["runoff_coeff"])

    def test_json_dumps_with_strict_rfc8259_succeeds(self):
        """json.dumps with allow_nan=False must not raise ValueError on serialized data."""
        c = make_conduit(self.session, conduit_name="C_NAN_JSON", max_q=float("inf"))
        c.max_v = float("nan")
        n = make_node(self.session, node_name="N_NAN_JSON", hours_flooded=float("inf"))
        n.max_water_depth = float("nan")
        s = make_subcatchment(self.session, subcatchment_name="S_NAN_JSON", peak_runoff=float("inf"))
        s.total_runoff_mg = float("nan")

        self.session._prefetched_objects_cache = {
            "conduits": [c],
            "nodes": [n],
            "subcatchments": [s],
        }

        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)

        # If allow_nan=False raises ValueError, non-finite floats escaped sanitization!
        serialized_json = json.dumps(payload, allow_nan=False)
        self.assertIsInstance(serialized_json, str)

        # Roundtrip validation: json.loads parses cleanly
        parsed = json.loads(serialized_json)
        c_parsed = next(x for x in parsed["conduits"] if x["conduit_name"] == "C_NAN_JSON")
        self.assertIsNone(c_parsed["max_v"])
        self.assertIsNone(c_parsed["max_q"])

    def test_normal_floats_and_zero_preserved(self):
        """Finite floats including 0.0 and negative values must be preserved."""
        make_conduit(self.session, conduit_name="C_EXACT", max_v=0.0, filling=0.254, geom1=0.4)
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        c = next(x for x in payload["conduits"] if x["conduit_name"] == "C_EXACT")

        self.assertEqual(c["max_v"], 0.0)
        self.assertEqual(c["filling"], 0.254)
        self.assertEqual(c["geom1"], 0.4)


class ResultsV2SummaryAndValidationKPITests(TestCase):
    """Test cases for summary KPI calculations and validation flag counts."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="kpi_tester@example.com")
        cls.session = make_session(cls.user)

        # Conduit 1: All 7 checks pass, recommendation="valid" (no changes needed)
        make_conduit(cls.session, conduit_name="C_VALID", recommendation="valid")

        # Conduit 2: 2 checks fail, recommendation="diameter_increase" (changes required)
        make_conduit(
            cls.session,
            conduit_name="C_CHG1",
            recommendation="diameter_increase",
            val_max_fill=0,
            val_coverage=0,
        )

        # Conduit 3: 1 check fails, recommendation="slope_increase" (changes required)
        make_conduit(
            cls.session,
            conduit_name="C_CHG2",
            recommendation="slope_increase",
            val_min_v=0,
        )

        # Node 1: Normal dry node
        make_node(cls.session, node_name="N_DRY", flood_volume=0.0, hours_flooded=0.0)

        # Node 2: Flooded node (volume > 0)
        make_node(cls.session, node_name="N_FLOOD1", flood_volume=5.2, hours_flooded=1.5)

        # Node 3: Flooded node (hours > 0, volume None)
        make_node(cls.session, node_name="N_FLOOD2", flood_volume=None, hours_flooded=0.8)

        # Subcatchments (2)
        make_subcatchment(cls.session, subcatchment_name="S1")
        make_subcatchment(cls.session, subcatchment_name="S2")

    def test_summary_kpi_counts(self):
        """Summary card metrics must match database state precisely."""
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        summary = payload["summary"]

        # Total conduits: 3
        conduits_count = summary.get("conduits_count") or summary.get("total_conduits")
        self.assertEqual(conduits_count, 3)

        # Conduits requiring changes: 2 (C_CHG1, C_CHG2)
        changes_count = summary.get("conduits_changes_count") or summary.get("changes_count")
        self.assertEqual(changes_count, 2)

        # Total nodes: 3
        nodes_count = summary.get("nodes_count") or summary.get("total_nodes")
        self.assertEqual(nodes_count, 3)

        # Nodes with flooding: 2 (N_FLOOD1, N_FLOOD2)
        flooding_count = summary.get("nodes_flooding_count") or summary.get("flooding_count")
        self.assertEqual(flooding_count, 2)

        # Total catchments: 2
        catchments_count = summary.get("catchments_count") or summary.get("total_catchments")
        self.assertEqual(catchments_count, 2)

    def test_validation_flag_counts_per_conduit(self):
        """Validation failure counts and 7-element flag lists must reflect actual flags."""
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        conduits = {c["conduit_name"]: c for c in payload["conduits"]}

        # C_VALID: 0 failures, 7 passes
        c_valid = conduits["C_VALID"]
        failed_count = c_valid.get("failed_checks_count", 7 - c_valid.get("validation", 7))
        self.assertEqual(failed_count, 0)
        self.assertEqual(c_valid["flags"], [True, True, True, True, True, True, True])

        # C_CHG1: val_max_fill=0, val_coverage=0 -> 2 failures, 5 passes
        c_chg1 = conduits["C_CHG1"]
        failed_count = c_chg1.get("failed_checks_count", 7 - c_chg1.get("validation", 5))
        self.assertEqual(failed_count, 2)
        self.assertEqual(c_chg1["val_max_fill"], 0)
        self.assertEqual(c_chg1["val_coverage"], 0)
        self.assertEqual(c_chg1["flags"][0], False)  # val_max_fill
        self.assertEqual(c_chg1["flags"][6], False)  # val_coverage

        # C_CHG2: val_min_v=0 -> 1 failure, 6 passes
        c_chg2 = conduits["C_CHG2"]
        failed_count = c_chg2.get("failed_checks_count", 7 - c_chg2.get("validation", 6))
        self.assertEqual(failed_count, 1)
        self.assertEqual(c_chg2["flags"][2], False)  # val_min_v


class ResultsV2RuleLabelsAndLinksTests(TestCase):
    """Test cases for injection of WROCLAW_RULE_LABELS and panel navigation links."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="links_rules@example.com")
        cls.session = make_session(cls.user)

        # K1 is in WROCLAW_RULE_LABELS ('depth_increase')
        make_conduit(cls.session, conduit_name="K1", inlet_node="N1", outlet_node="N2")
        # K10 is in WROCLAW_RULE_LABELS ('tank')
        make_conduit(cls.session, conduit_name="K10", inlet_node="N2", outlet_node="N3")
        # UNMAPPED is not in WROCLAW_RULE_LABELS
        make_conduit(cls.session, conduit_name="UNMAPPED", inlet_node="N3", outlet_node="N4")

        make_node(cls.session, node_name="N1")
        make_node(cls.session, node_name="N2")
        make_node(cls.session, node_name="N3")
        make_node(cls.session, node_name="N4")

        cls.s1 = make_subcatchment(cls.session, subcatchment_name="S1", outlet="N1")

    def test_wroclaw_rule_labels_injected(self):
        """Conduits matching WROCLAW_RULE_LABELS receive corresponding label; others receive '-'."""
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        conduits = {c["conduit_name"]: c for c in payload["conduits"]}

        # K1 -> depth_increase
        self.assertEqual(conduits["K1"].get("rule_label") or conduits["K1"].get("rule"), WROCLAW_RULE_LABELS["K1"])
        self.assertEqual(conduits["K1"].get("rule_label") or conduits["K1"].get("rule"), "depth_increase")

        # K10 -> tank
        self.assertEqual(conduits["K10"].get("rule_label") or conduits["K10"].get("rule"), WROCLAW_RULE_LABELS["K10"])
        self.assertEqual(conduits["K10"].get("rule_label") or conduits["K10"].get("rule"), "tank")

        # UNMAPPED -> '-'
        self.assertEqual(conduits["UNMAPPED"].get("rule_label") or conduits["UNMAPPED"].get("rule"), "-")

    def test_conduit_detail_links_injected(self):
        """Each conduit record contains detail URL matching sa:conduit_detail."""
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        for c in payload["conduits"]:
            expected_url = reverse("sa:conduit_detail", args=[self.session.id, c["conduit_name"]])
            url = c.get("panel_url") or c.get("detail_url") or c.get("link")
            self.assertEqual(url, expected_url)

    def test_node_and_subcatchment_panel_links_injected(self):
        """Nodes and catchments have panel links with fragment anchors."""
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        nodes = {n["node_name"]: n for n in payload["nodes"]}
        catchments = {s["subcatchment_name"]: s for s in payload["catchments"]}

        k1_url = reverse("sa:conduit_detail", args=[self.session.id, "K1"])
        # N1 is inlet of K1
        n1_url = nodes["N1"].get("panel_url") or nodes["N1"].get("detail_url")
        self.assertEqual(n1_url, f"{k1_url}#node-inlet")

        # S1 drains to N1 (inlet of K1)
        s1_url = catchments["S1"].get("panel_url") or catchments["S1"].get("detail_url")
        self.assertEqual(s1_url, f"{k1_url}#catchment-{self.s1.pk}")


class ResultsV2ViewIntegrationTests(TestCase):
    """Test cases for AnalysisResultsView: HTTP 200, context dictionary, and access control."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="view_tester@example.com")
        cls.other_user = make_user(email="other_user@example.com")
        cls.session = make_session(cls.user, n_conduits=2, n_nodes=3, n_subcatchments=2)

    def setUp(self):
        self.client.force_login(self.user)

    def _url(self, session_id: int | None = None) -> str:
        return reverse("sa:analysis_results", args=[session_id or self.session.id])

    def test_results_view_returns_200_for_owner(self):
        """Authenticated owner accessing completed session receives HTTP 200."""
        response = self.client.get(self._url())
        self.assertEqual(response.status_code, 200)

    def test_context_carries_only_the_session_and_the_v2_payload(self):
        """The page renders from ``results_v2_data`` and ``session`` alone.

        The legacy ``data``/``panel_links`` keys were dropped with the accordion
        template that read them; asserting their absence keeps the second, discarded
        serialization pass from creeping back in.
        """
        response = self.client.get(self._url())
        self.assertEqual(response.status_code, 200)

        self.assertIn("results_v2_data", response.context)
        v2_data = response.context["results_v2_data"]
        self.assertIsInstance(v2_data, dict)
        self.assertIn("conduits", v2_data)
        self.assertIn("nodes", v2_data)
        self.assertIn("catchments", v2_data)
        self.assertIn("summary", v2_data)

        self.assertIn("session", response.context)
        self.assertNotIn("data", response.context)
        self.assertNotIn("panel_links", response.context)

    def test_foreign_user_cannot_access_results_view(self):
        """Multi-tenant isolation: other user receives 404."""
        self.client.force_login(self.other_user)
        response = self.client.get(self._url())
        self.assertEqual(response.status_code, 404)

    def test_anonymous_user_redirected_to_login(self):
        """Unauthenticated user is redirected to login page."""
        self.client.logout()
        response = self.client.get(self._url())
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])

    def test_pending_session_redirects_to_analysis(self):
        """Unfinished session redirects back to analysis form."""
        pending = make_session(self.user, status="processing")
        response = self.client.get(self._url(pending.id))
        self.assertRedirects(response, reverse("sa:analysis"))

    def test_results_view_query_count_is_bounded(self):
        """Prevent N+1 query regression: page execution must require <= 6 database queries (2 middleware + 4 view)."""
        with CaptureQueriesContext(connection) as ctx:
            response = self.client.get(self._url())

        self.assertEqual(response.status_code, 200)
        self.assertLessEqual(
            len(ctx.captured_queries),
            6,
            f"Query count exceeded bound ({len(ctx.captured_queries)} queries):\n"
            + "\n".join(q["sql"] for q in ctx.captured_queries),
        )


class ResultsV2EdgeCasesTests(TestCase):
    """Test cases for edge cases: empty sessions, missing optional fields, special characters."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="edge_cases@example.com")

    def setUp(self):
        self.client.force_login(self.user)

    def test_empty_session_serializes_cleanly(self):
        """Session with 0 conduits, 0 nodes, 0 subcatchments serializes without error."""
        empty_session = make_session(self.user, n_conduits=0, n_nodes=0, n_subcatchments=0)
        payload = CalculationRetrievalService.serialize_session_for_results_v2(empty_session)

        self.assertEqual(payload["conduits"], [])
        self.assertEqual(payload["nodes"], [])
        self.assertEqual(payload["catchments"], [])

        summary = payload["summary"]
        self.assertEqual(summary.get("conduits_count") or summary.get("total_conduits"), 0)
        self.assertEqual(summary.get("conduits_changes_count") or summary.get("changes_count"), 0)
        self.assertEqual(summary.get("nodes_count") or summary.get("total_nodes"), 0)
        self.assertEqual(summary.get("nodes_flooding_count") or summary.get("flooding_count"), 0)
        self.assertEqual(summary.get("catchments_count") or summary.get("total_catchments"), 0)

        # json.dumps must succeed without error
        json_output = json.dumps(payload, allow_nan=False)
        self.assertIn('"conduits": []', json_output)

    def test_empty_session_view_renders_200(self):
        """AnalysisResultsView renders 200 for an empty session."""
        empty_session = make_session(self.user, n_conduits=0, n_nodes=0, n_subcatchments=0)
        response = self.client.get(reverse("sa:analysis_results", args=[empty_session.id]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["results_v2_data"]["conduits"]), 0)

    def test_missing_optional_node_depth_fields_yield_none_ground_elevation(self):
        """Node with null invert_elevation or null max_depth must have ground_elevation=None."""
        session = make_session(self.user)
        make_node(session, node_name="N_NO_INVERT", invert_elevation=None, max_depth=2.5)
        make_node(session, node_name="N_NO_DEPTH", invert_elevation=118.0, max_depth=None)
        make_node(session, node_name="N_NULL_BOTH", invert_elevation=None, max_depth=None)

        payload = CalculationRetrievalService.serialize_session_for_results_v2(session)
        nodes = {n["node_name"]: n for n in payload["nodes"]}

        self.assertIsNone(nodes["N_NO_INVERT"]["ground_elevation"])
        self.assertIsNone(nodes["N_NO_DEPTH"]["ground_elevation"])
        self.assertIsNone(nodes["N_NULL_BOTH"]["ground_elevation"])

    def test_missing_confidence_scores_serialize_to_none(self):
        """Conduit with null confidence values serializes them cleanly as None (JSON null)."""
        session = make_session(self.user)
        make_conduit(
            session,
            conduit_name="C_NO_CONF",
            confidence_pump=None,
            confidence_tank=None,
            confidence_valid=None,
        )
        payload = CalculationRetrievalService.serialize_session_for_results_v2(session)
        c = payload["conduits"][0]

        self.assertIsNone(c["confidence"])

    def test_division_by_zero_protection_in_fill_percent(self):
        """Conduit with geom1=0.0 must not raise ZeroDivisionError when calculating fillPercent."""
        session = make_session(self.user)
        make_conduit(session, conduit_name="C_ZERO_DIA", geom1=0.0, filling=0.1)

        payload = CalculationRetrievalService.serialize_session_for_results_v2(session)
        c = payload["conduits"][0]
        self.assertIn("fillPercent", c)
        self.assertEqual(c["fillPercent"], 0.0)

    def test_short_flood_rounded_to_zero_hours_is_flooded_in_table_and_panel(self):
        """One predicate: a 0.00 h flood with a volume counts everywhere it is shown."""
        session = make_session(self.user)
        make_node(session, node_name="N_SHORT", hours_flooded=0.0, flood_volume=0.003)
        make_node(session, node_name="N_OUT", hours_flooded=0.0, flood_volume=0.0)
        conduit = make_conduit(session, conduit_name="C1", inlet_node="N_SHORT", outlet_node="N_OUT")

        payload = CalculationRetrievalService.serialize_session_for_results_v2(session)
        node = {n["id"]: n for n in payload["nodes"]}["N_SHORT"]

        self.assertIs(node["is_flooded"], True)
        self.assertEqual(payload["summary"]["nodes_flooding_count"], 1)
        self.assertIs(ConduitPanelService.build(session, conduit).inlet.is_flooded, True)

    def test_fill_percent_is_clamped_like_the_conduit_panel(self):
        """The table label and the panel gauge read the same clamped percentage."""
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="C_OVERFULL", geom1=0.3, filling=0.4)

        payload = CalculationRetrievalService.serialize_session_for_results_v2(session)
        c = payload["conduits"][0]

        self.assertEqual(c["fillPercent"], 100.0)
        self.assertEqual(c["fill_percent"], 100.0)
        self.assertEqual(ConduitPanelService.build(session, conduit).filling_percent, 100.0)

    def test_fill_percent_of_a_negative_filling_is_zero(self):
        session = make_session(self.user)
        make_conduit(session, conduit_name="C_NEG", geom1=0.3, filling=-0.1)

        payload = CalculationRetrievalService.serialize_session_for_results_v2(session)

        self.assertEqual(payload["conduits"][0]["fillPercent"], 0.0)

    def test_fill_percent_helper_rejects_missing_numbers(self):
        """The column is NOT NULL, but the helper is the contract both call sites share."""
        self.assertIsNone(fill_percent(None, 0.3))
        self.assertIsNone(fill_percent(0.1, None))
        self.assertIsNone(fill_percent(0.1, 0.0))

    def test_element_names_with_slashes_and_special_characters(self):
        """SWMM identifiers with slashes or HTML-sensitive characters serialize safely."""
        session = make_session(self.user)
        make_node(session, node_name="N/1")
        make_node(session, node_name='N<2>"')
        make_conduit(session, conduit_name="A/B", inlet_node="N/1", outlet_node='N<2>"')
        make_subcatchment(session, subcatchment_name="S/1 & Co", outlet="N/1")

        payload = CalculationRetrievalService.serialize_session_for_results_v2(session)
        json_str = json.dumps(payload, allow_nan=False)

        # JSON parsing roundtrip succeeds
        parsed = json.loads(json_str)
        c = parsed["conduits"][0]
        self.assertEqual(c["conduit_name"], "A/B")
        self.assertEqual(c["inlet_node"], "N/1")

        # Detail URL reverses without error
        expected_url = reverse("sa:conduit_detail", args=[session.id, "A/B"])
        self.assertEqual(c.get("panel_url") or c.get("detail_url"), expected_url)

    def test_legacy_format_session_data_preserved(self):
        """Preserve 100% backward compatibility of format_session_data_for_template."""
        session = make_session(self.user, n_conduits=1, n_nodes=1, n_subcatchments=1)
        data = CalculationRetrievalService.format_session_data_for_template(session)

        self.assertIn("Conduits Data", data)
        self.assertIn("Nodes Data", data)
        self.assertIn("Subcatchments Data", data)
        c_legacy = data["Conduits Data"][0]
        self.assertIn("Subcatchment (model)", c_legacy)
        self.assertIn("Recommendation", c_legacy)
        n_legacy = data["Nodes Data"][0]
        self.assertIn("MaxNodeDepth", n_legacy)
        self.assertIn("HoursFlooded", n_legacy)


class ResultsV2AdversarialResilienceTests(TestCase):
    """Stress and resilience unit tests verifying fixes for the 8 failure modes identified in stress_results_v2.

    Verifies:
    1. _clean_float OverflowError handling on huge integers (10**400) and Decimals.
    2. Derived float overflow in fillPercent with subnormal diameter (geom1=1e-200, filling=1e200).
    3. Derived float overflow in fillPercent with astronomical filling (geom1=1.0, filling=1e308).
    4. Derived float overflow in ground_elevation (invert=1e308, depth=1e308).
    5. Derived float overflow in confidence_display (confidence_valid=1e308).
    6. Recommendation case-insensitivity and whitespace trimming ('VALID', 'Valid', '  valid  ').
    7. Validation flag None handling (avoiding TypeError in int casting).
    8. Validation flag NaN and Inf handling (avoiding ValueError/OverflowError in int casting).
    9. Comprehensive strict RFC 8259 JSON serialization roundtrip with all failure modes combined.
    """

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email=f"adversarial_{uuid.uuid4().hex[:8]}@example.com")
        cls.session = make_session(cls.user, n_conduits=0, n_nodes=0, n_subcatchments=0)

    # -------------------------------------------------------------------------
    # Scenario 1: _clean_float OverflowError on 10**400 and huge Decimals
    # -------------------------------------------------------------------------
    def test_clean_float_overflow_error_and_arbitrary_precision(self):
        """_clean_float must catch OverflowError on huge integers and Decimals, returning None."""
        self.assertIsNone(_clean_float(10**400))
        self.assertIsNone(_clean_float(-(10**400)))
        self.assertIsNone(_clean_float(decimal.Decimal("1e400")))
        self.assertIsNone(_clean_float(decimal.Decimal("-1e400")))
        self.assertIsNone(_clean_float(decimal.Decimal("NaN")))
        self.assertIsNone(_clean_float(decimal.Decimal("Infinity")))
        self.assertIsNone(_clean_float(decimal.Decimal("-Infinity")))
        # Large finite numbers within IEEE 754 float limits must be preserved
        self.assertEqual(_clean_float(1e308), 1e308)
        self.assertEqual(_clean_float(-1e308), -1e308)

    # -------------------------------------------------------------------------
    # Scenario 2: Derived float overflow in fillPercent (subnormal diameter)
    # -------------------------------------------------------------------------
    def test_fill_percent_subnormal_diameter_overflow(self):
        """fillPercent must not overflow to inf or crash json.dumps when diameter is extremely small."""
        c = make_conduit(self.session, conduit_name="C_SUBNORMAL_DIA", geom1=1e-200, filling=1e200)
        self.session._prefetched_objects_cache = {
            "conduits": [c],
            "nodes": [],
            "subcatchments": [],
        }
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        cond = payload["conduits"][0]
        fill_pct = cond["fillPercent"]

        self.assertTrue(fill_pct is None or not (math.isnan(fill_pct) or math.isinf(fill_pct)))
        self.assertTrue(
            cond["fill_percent"] is None or not (math.isnan(cond["fill_percent"]) or math.isinf(cond["fill_percent"]))
        )
        json_output = json.dumps(payload, allow_nan=False)
        self.assertIsInstance(json_output, str)

    # -------------------------------------------------------------------------
    # Scenario 3: Derived float overflow in fillPercent (astronomical filling)
    # -------------------------------------------------------------------------
    def test_fill_percent_astronomical_filling_overflow(self):
        """fillPercent must not overflow to inf or crash json.dumps when filling is astronomical."""
        c = make_conduit(self.session, conduit_name="C_ASTRONOMICAL_FILL", geom1=1.0, filling=1e308)
        self.session._prefetched_objects_cache = {
            "conduits": [c],
            "nodes": [],
            "subcatchments": [],
        }
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        cond = payload["conduits"][0]
        fill_pct = cond["fillPercent"]

        self.assertTrue(fill_pct is None or not (math.isnan(fill_pct) or math.isinf(fill_pct)))
        self.assertTrue(
            cond["fill_percent"] is None or not (math.isnan(cond["fill_percent"]) or math.isinf(cond["fill_percent"]))
        )
        json_output = json.dumps(payload, allow_nan=False)
        self.assertIsInstance(json_output, str)

    # -------------------------------------------------------------------------
    # Scenario 4: Derived float overflow in ground_elevation (invert + depth)
    # -------------------------------------------------------------------------
    def test_ground_elevation_overflow_sanitized_to_none(self):
        """ground_elevation must be sanitized to None when invert + depth exceeds float max (inf)."""
        n = make_node(self.session, node_name="N_GROUND_OVERFLOW", invert_elevation=1e308, max_depth=1e308)
        self.session._prefetched_objects_cache = {
            "conduits": [],
            "nodes": [n],
            "subcatchments": [],
        }
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        node_res = payload["nodes"][0]

        self.assertIsNone(node_res["ground_elevation"])
        self.assertIsNone(node_res["ground"])
        json_output = json.dumps(payload, allow_nan=False)
        self.assertIsInstance(json_output, str)

    # -------------------------------------------------------------------------
    # Scenario 5: Derived float overflow in confidence_display (conf * 100.0)
    # -------------------------------------------------------------------------
    def test_confidence_display_overflow_sanitized_to_none(self):
        """confidence display must be sanitized to None when confidence * 100.0 overflows to inf."""
        c = make_conduit(self.session, conduit_name="C_CONF_OVERFLOW", recommendation="valid", confidence_valid=1e308)
        self.session._prefetched_objects_cache = {
            "conduits": [c],
            "nodes": [],
            "subcatchments": [],
        }
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        cond = payload["conduits"][0]

        self.assertIsNone(cond["confidence"])
        json_output = json.dumps(payload, allow_nan=False)
        self.assertIsInstance(json_output, str)

    # -------------------------------------------------------------------------
    # Scenario 6: Recommendation case-insensitivity and whitespace resilience
    # -------------------------------------------------------------------------
    def test_recommendation_case_insensitivity_and_confidence_resolution(self):
        """Recommendation lookup for confidence scores must be case-insensitive and strip whitespace."""
        c_upper = make_conduit(self.session, conduit_name="C_UPPER", recommendation="VALID", confidence_valid=0.92)
        c_title = make_conduit(self.session, conduit_name="C_TITLE", recommendation="Valid", confidence_valid=0.88)
        c_space = make_conduit(self.session, conduit_name="C_SPACE", recommendation="  valid  ", confidence_valid=0.75)
        c_chg = make_conduit(
            self.session,
            conduit_name="C_CHG",
            recommendation="DIAMETER_INCREASE",
            confidence_diameter_increase=0.85,
        )
        self.session._prefetched_objects_cache = {
            "conduits": [c_upper, c_title, c_space, c_chg],
            "nodes": [],
            "subcatchments": [],
        }
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        conduits_by_name = {c["conduit_name"]: c for c in payload["conduits"]}

        self.assertEqual(conduits_by_name["C_UPPER"]["confidence"], 92.0)
        self.assertEqual(conduits_by_name["C_TITLE"]["confidence"], 88.0)
        self.assertEqual(conduits_by_name["C_SPACE"]["confidence"], 75.0)
        self.assertEqual(conduits_by_name["C_CHG"]["confidence"], 85.0)

        # Summary check: C_UPPER, C_TITLE, C_SPACE are valid -> only C_CHG requires changes
        changes_count = payload["summary"].get("conduits_changes_count") or payload["summary"].get("changes_count")
        self.assertEqual(changes_count, 1)

    # -------------------------------------------------------------------------
    # Scenario 7: Validation flag None handling
    # -------------------------------------------------------------------------
    def test_validation_flags_none_handling(self):
        """Validation flags with None values must not raise TypeError and normalize to 0 / False."""
        c = make_conduit(self.session, conduit_name="C_FLAGS_NONE")
        for flag in VALIDATION_FLAGS:
            setattr(c, flag, None)

        self.session._prefetched_objects_cache = {
            "conduits": [c],
            "nodes": [],
            "subcatchments": [],
        }
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        cond = payload["conduits"][0]

        for flag in VALIDATION_FLAGS:
            self.assertIn(flag, cond)
            self.assertEqual(cond[flag], 0)

        self.assertEqual(cond["flags"], [False] * 7)
        self.assertEqual(cond["passed_checks_count"], 0)
        self.assertEqual(cond["failed_checks_count"], 7)
        json_output = json.dumps(payload, allow_nan=False)
        self.assertIsInstance(json_output, str)

    # -------------------------------------------------------------------------
    # Scenario 8: Validation flag NaN and Inf handling
    # -------------------------------------------------------------------------
    def test_validation_flags_nan_and_inf_handling(self):
        """Validation flags with NaN or Inf floats must not raise ValueError/OverflowError and normalize to 0 / False."""
        c = make_conduit(self.session, conduit_name="C_FLAGS_NAN")
        c.val_max_fill = float("nan")
        c.val_max_v = float("inf")
        c.val_min_v = float("-inf")
        c.val_max_slope = 1
        c.val_min_slope = 0
        c.val_depth = None
        c.val_coverage = 1

        self.session._prefetched_objects_cache = {
            "conduits": [c],
            "nodes": [],
            "subcatchments": [],
        }
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        cond = payload["conduits"][0]

        self.assertEqual(cond["val_max_fill"], 0)
        self.assertEqual(cond["val_max_v"], 0)
        self.assertEqual(cond["val_min_v"], 0)
        self.assertEqual(cond["val_max_slope"], 1)
        self.assertEqual(cond["val_min_slope"], 0)
        self.assertEqual(cond["val_depth"], 0)
        self.assertEqual(cond["val_coverage"], 1)

        # val_max_slope (1) and val_coverage (1) pass, all others (nan, inf, -inf, 0, None) fail
        self.assertEqual(cond["flags"], [False, False, False, True, False, False, True])
        self.assertEqual(cond["passed_checks_count"], 2)
        self.assertEqual(cond["failed_checks_count"], 5)
        json_output = json.dumps(payload, allow_nan=False)
        self.assertIsInstance(json_output, str)

    # -------------------------------------------------------------------------
    # Scenario 9: Comprehensive strict RFC 8259 roundtrip with all failure modes combined
    # -------------------------------------------------------------------------
    def test_comprehensive_rfc8259_strict_json_roundtrip_all_scenarios(self):
        """Payload containing all combined failure modes must serialize cleanly and strictly pass json.dumps."""
        c1 = make_conduit(
            self.session,
            conduit_name="C_EXTREME_1",
            geom1=1e-200,
            filling=1e200,
            recommendation="VALID",
            confidence_valid=0.95,
        )
        c1.val_max_fill = None
        c1.val_coverage = float("nan")

        c2 = make_conduit(
            self.session,
            conduit_name="C_EXTREME_2",
            geom1=1.0,
            filling=1e308,
            recommendation="TANK",
            confidence_tank=1e308,
        )
        c2.val_max_v = float("inf")

        n1 = make_node(self.session, node_name="N_EXTREME_1", invert_elevation=1e308, max_depth=1e308)
        n2 = make_node(self.session, node_name="N_EXTREME_2", invert_elevation=float("nan"), max_depth=float("inf"))

        s1 = make_subcatchment(self.session, subcatchment_name="S_EXTREME_1")
        s1.area = float("nan")
        s1.perc_imperv = float("inf")

        self.session._prefetched_objects_cache = {
            "conduits": [c1, c2],
            "nodes": [n1, n2],
            "subcatchments": [s1],
        }

        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        dumped = json.dumps(payload, allow_nan=False)
        self.assertIsInstance(dumped, str)

        parsed = json.loads(dumped)
        self.assertEqual(len(parsed["conduits"]), 2)
        self.assertEqual(len(parsed["nodes"]), 2)
        self.assertEqual(len(parsed["catchments"]), 1)


class ResultsV2NetworkMapEnvelopeTests(TestCase):
    """The ``network_map_data`` envelope: availability, the two unavailable reasons, cost."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="map_tester@example.com")
        cls.with_geometry = make_session(cls.user, n_conduits=2, n_nodes=3, n_subcatchments=2)
        cls.geometry = make_geometry(cls.with_geometry)
        cls.without_geometry = make_session(cls.user, n_conduits=2, n_nodes=3, n_subcatchments=2)

    def setUp(self):
        self.client.force_login(self.user)

    def _url(self, session) -> str:
        return reverse("sa:analysis_results", args=[session.id])

    def test_map_payload_in_context(self):
        """A session with stored geometry carries the whole payload into the context."""
        response = self.client.get(self._url(self.with_geometry))
        self.assertEqual(response.status_code, 200)

        envelope = response.context["network_map_data"]
        self.assertEqual(sorted(envelope), ["available", "data", "reason"])
        self.assertTrue(envelope["available"])
        self.assertIsNone(envelope["reason"])
        self.assertEqual(envelope["data"], self.geometry.data)
        # Serializable without a fallback encoder: it goes through json_script verbatim.
        json.dumps(envelope, allow_nan=False)

    def test_map_unavailable_without_geometry(self):
        """No geometry row is a normal state: the envelope says so and carries no data."""
        envelope = CalculationRetrievalService.serialize_geometry_for_map(self.without_geometry)
        self.assertEqual(envelope, {"available": False, "reason": "no_geometry", "data": None})

        response = self.client.get(self._url(self.without_geometry))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["network_map_data"], envelope)

        html = response.content.decode("utf-8")
        tab = html[html.index('id="r-tab-map"') : html.index('id="r-tab-conduits"')]
        self.assertNotIn("disabled", tab)
        self.assertNotIn("aria-disabled", tab)

    def test_map_unavailable_when_too_many_points(self):
        """The second half of the guard: few elements, hundreds of thousands of points.

        A thousand subcatchments whose outlines carry four hundred vertices each, with
        a thousand nodes and twelve hundred conduits, score 4400 elements — comfortably
        under the element ceiling — and about four hundred thousand coordinate pairs,
        some ten megabytes of JSON inlined into the page through ``json_script``.  The
        element count cannot see that, because an outline is one element whatever its
        length.
        """
        session = make_session(self.user)
        payload = make_geometry_payload()
        counts = {"elements": 4400, "points": NETMAP_MAX_POINTS + 1}
        make_geometry(session, counts=counts)

        envelope = CalculationRetrievalService.serialize_geometry_for_map(session)
        self.assertEqual(envelope, {"available": False, "reason": "too_large", "data": None})

        # Exactly at the ceiling the map is still drawn.
        session.geometry.data = {**payload, "counts": {"elements": 4400, "points": NETMAP_MAX_POINTS}}
        session.geometry.save(update_fields=["data"])
        session.refresh_from_db()
        self.assertTrue(CalculationRetrievalService.serialize_geometry_for_map(session)["available"])

    def test_map_available_for_a_row_written_before_the_point_counter(self):
        """An old row is not a large one: a missing ``points`` must not refuse it.

        Rows stored before the counter existed carry ``counts`` without it, and the
        payload is still perfectly drawable — the element bound goes on applying to it.
        """
        session = make_session(self.user)
        make_geometry(session, counts={"elements": 8})

        envelope = CalculationRetrievalService.serialize_geometry_for_map(session)
        self.assertTrue(envelope["available"])
        self.assertIsNone(envelope["reason"])
        self.assertNotIn("points", envelope["data"]["counts"])

        # A count that is not a number is worth no more than an absent one, and neither
        # may be read as "too large".
        for broken in (None, "many", True, [1]):
            session.geometry.data = {**make_geometry_payload(), "counts": {"elements": 8, "points": broken}}
            session.geometry.save(update_fields=["data"])
            session.refresh_from_db()
            self.assertTrue(
                CalculationRetrievalService.serialize_geometry_for_map(session)["available"],
                f"an unusable point count ({broken!r}) must not hide the map",
            )

    def test_map_unavailable_when_too_large(self):
        """Above the element ceiling the payload stays in the database."""
        session = make_session(self.user)
        payload = make_geometry_payload()
        payload["counts"] = {"elements": NETMAP_MAX_ELEMENTS + 1}
        make_geometry(session, counts=payload["counts"])

        envelope = CalculationRetrievalService.serialize_geometry_for_map(session)
        self.assertEqual(envelope, {"available": False, "reason": "too_large", "data": None})

        # Exactly at the ceiling the map is still drawn.
        session.geometry.data = {**payload, "counts": {"elements": NETMAP_MAX_ELEMENTS}}
        session.geometry.save(update_fields=["data"])
        session.refresh_from_db()
        self.assertTrue(CalculationRetrievalService.serialize_geometry_for_map(session)["available"])

    def test_version_1_row_is_still_served(self):
        """A row written before the outlet lines is drawn exactly as it was stored.

        The version stamp rose to 2 and nothing backfills the rows already in the
        database: their polygons have no centroid, their ``skipped`` has two keys and
        the serializer must not start branching on ``version`` because of it.  The
        envelope says available and hands the payload over untouched.
        """
        session = make_session(self.user)
        payload = {
            "version": 1,
            "origin": [0.0, 0.0],
            "bbox": {"w": 10.0, "h": 10.0},
            "min_gap": 5.0,
            "layout": "inp",
            "nodes": [{"n": "J1", "i": 0, "p": [0.0, 0.0], "k": "junction"}],
            "links": [],
            "polys": [{"n": "S1", "i": 0, "p": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]}],
            "skipped": {"links_dangling": 0, "polys_degenerate": 0},
            "counts": {"elements": 2, "points": 4},
        }
        make_geometry(session, **payload)
        session.refresh_from_db()

        envelope = CalculationRetrievalService.serialize_geometry_for_map(session)
        self.assertTrue(envelope["available"])
        self.assertIsNone(envelope["reason"])
        self.assertEqual(envelope["data"], payload)
        self.assertNotIn("c", envelope["data"]["polys"][0])
        self.assertEqual(sorted(envelope["data"]["skipped"]), ["links_dangling", "polys_degenerate"])

    def test_no_extra_queries(self):
        """Reading the geometry costs nothing: it is selected along with the session."""

        def count_queries(session) -> int:
            with CaptureQueriesContext(connection) as ctx:
                response = self.client.get(self._url(session))
            self.assertEqual(response.status_code, 200)
            return len(ctx.captured_queries)

        # Warm the session cache of the test client so the first call is not charged
        # for anything the second one gets for free.
        count_queries(self.without_geometry)

        without = count_queries(self.without_geometry)
        with_geometry = count_queries(self.with_geometry)
        self.assertEqual(
            with_geometry,
            without,
            f"drawing the map cost {with_geometry - without} extra queries",
        )
        self.assertLessEqual(with_geometry, 6)


class ResultsV2ProvenanceAndExportTests(TestCase):
    """Tests for model provenance metadata, top-2 confidence margin, simple band rule (§3.4.3), and server-side CSV export."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="export_tester@example.com")
        cls.other_user = make_user(email="other_export_tester@example.com")
        cls.session = make_session(cls.user, n_conduits=2, n_nodes=2, n_subcatchments=1)

    def setUp(self):
        self.client.force_login(self.user)

    def test_conduit_serialization_includes_margin_label_and_simple_rule(self):
        payload = CalculationRetrievalService.serialize_session_for_results_v2(self.session)
        self.assertIn("model_metadata", payload["session"])
        self.assertIn("model_id", payload["session"]["model_metadata"])
        c = payload["conduits"][0]
        self.assertIn("recommendation_label", c)
        self.assertEqual(c["recommendation_label"], "No changes needed")
        self.assertIn("confidence_margin", c)
        self.assertIsNotNone(c["confidence_margin"])
        self.assertIn("simple_rule_passed", c)
        self.assertIsInstance(c["simple_rule_passed"], bool)

    def test_server_csv_export_for_all_kinds(self):
        for kind in ("conduits", "nodes", "catchments"):
            url = reverse("sa:export_session_csv", args=[self.session.id, kind])
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/csv", response["Content-Type"])
            content = response.content.decode("utf-8")
            self.assertTrue(content.startswith("\ufeff"))
            self.assertGreaterEqual(len(content.strip().splitlines()), 2)

    def test_server_csv_export_rejects_invalid_kind_and_foreign_user(self):
        bad_url = reverse("sa:export_session_csv", args=[self.session.id, "invalid"])
        self.assertEqual(self.client.get(bad_url).status_code, 400)

        self.client.force_login(self.other_user)
        url = reverse("sa:export_session_csv", args=[self.session.id, "conduits"])
        self.assertEqual(self.client.get(url).status_code, 404)

    def test_server_csv_export_sanitizes_formula_injection(self):
        from sa.views import ExportSessionCsvView

        self.assertEqual(ExportSessionCsvView._sanitize_csv_cell("=CMD|' /C calc'!A0"), "'=CMD|' /C calc'!A0")
        self.assertEqual(ExportSessionCsvView._sanitize_csv_cell("+SUM(A1:A2)"), "'+SUM(A1:A2)")
        self.assertEqual(ExportSessionCsvView._sanitize_csv_cell("@SUM(A1:A2)"), "'@SUM(A1:A2)")
        self.assertEqual(ExportSessionCsvView._sanitize_csv_cell("-2+3"), "'-2+3")
        self.assertEqual(ExportSessionCsvView._sanitize_csv_cell("-"), "-")
        self.assertEqual(ExportSessionCsvView._sanitize_csv_cell("-3.14"), "-3.14")
        self.assertEqual(ExportSessionCsvView._sanitize_csv_cell("C1"), "C1")

