"""Tests for ``ConduitPanelService.build`` and the supporting dataclasses.

The service replaces inline queries in ``ConduitDetailView.get_related_objects``
with a single ``build()`` call that returns a frozen ``ConduitPanel`` built from
at most four database queries (three without a model catchment).
"""

from __future__ import annotations

import json

from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext

from sa.core.round import max_depth_value
from sa.core.valid_round import validate_filling, validate_max_velocity, validate_min_velocity
from sa.services import VALIDATION_FLAGS, ConduitPanelService, catchment_anchor

from .factories import DRY_CONDUIT, make_conduit, make_node, make_session, make_subcatchment, make_user


class ConduitPanelServiceTests(TestCase):
    """Unit tests for ``ConduitPanelService.build``."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="panel@example.com")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build(self, session, conduit):
        return ConduitPanelService.build(session, conduit)

    # ------------------------------------------------------------------
    # catchments: 0 / 1 / 2
    # ------------------------------------------------------------------

    def test_zero_direct_catchments(self):
        session = make_session(self.user)
        make_node(session, node_name="W1")
        make_node(session, node_name="W2")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        panel = self._build(session, conduit)

        self.assertEqual(panel.direct_catchments, [])
        self.assertEqual(panel.inlet.catchments, [])
        self.assertIsNone(panel.model_catchment_name)
        self.assertIsNone(panel.model_catchment)
        self.assertFalse(panel.model_catchment_differs)

    def test_one_direct_catchment_same_as_model(self):
        session = make_session(self.user)
        make_node(session, node_name="W1")
        make_node(session, node_name="W2")
        sbc = make_subcatchment(session, subcatchment_name="P1", outlet="W1")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="P1")

        panel = self._build(session, conduit)

        self.assertEqual(len(panel.direct_catchments), 1)
        self.assertEqual(panel.direct_catchments[0].subcatchment.pk, sbc.pk)
        self.assertEqual(panel.model_catchment_name, "P1")
        self.assertIsNotNone(panel.model_catchment)
        self.assertEqual(panel.model_catchment.pk, sbc.pk)
        self.assertFalse(panel.model_catchment_differs)

    def test_one_direct_catchment_different_from_model(self):
        session = make_session(self.user)
        make_node(session, node_name="W1")
        make_node(session, node_name="W2")
        make_subcatchment(session, subcatchment_name="P1", outlet="W1")
        sbc_model = make_subcatchment(session, subcatchment_name="P99", outlet="W99")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="P99")

        panel = self._build(session, conduit)

        self.assertEqual(len(panel.direct_catchments), 1)
        self.assertEqual(panel.model_catchment_name, "P99")
        self.assertEqual(panel.model_catchment.pk, sbc_model.pk)
        self.assertTrue(panel.model_catchment_differs)

    def test_two_direct_catchments_one_is_model(self):
        session = make_session(self.user)
        make_node(session, node_name="W1")
        make_node(session, node_name="W2")
        make_subcatchment(session, subcatchment_name="P2", outlet="W1")
        make_subcatchment(session, subcatchment_name="P1", outlet="W1")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="P1")

        panel = self._build(session, conduit)

        # Created P2 before P1: the rows must come back sorted by name, not by insertion.
        self.assertEqual([row.subcatchment.subcatchment_name for row in panel.direct_catchments], ["P1", "P2"])
        # P1 is one of two direct catchments: the pipeline kept only one of them, so
        # the model feature does not describe the whole inflow -> flagged as differing.
        self.assertTrue(panel.model_catchment_differs)

    def test_model_catchment_name_without_record(self):
        """The pipeline may name a catchment the session never persisted: name kept, record None."""
        session = make_session(self.user)
        make_node(session, node_name="W1")
        make_node(session, node_name="W2")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="P9")

        panel = self._build(session, conduit)

        self.assertEqual(panel.model_catchment_name, "P9")
        self.assertIsNone(panel.model_catchment)
        self.assertTrue(panel.model_catchment_differs)

    def test_model_catchment_dash_means_none(self):
        session = make_session(self.user)
        make_node(session, node_name="W1")
        make_node(session, node_name="W2")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        panel = self._build(session, conduit)

        self.assertIsNone(panel.model_catchment_name)
        self.assertIsNone(panel.model_catchment)
        self.assertFalse(panel.model_catchment_differs)

    # ------------------------------------------------------------------
    # node without record → node None, elevations None
    # ------------------------------------------------------------------

    def test_node_without_record_gives_none(self):
        session = make_session(self.user)
        # Only create inlet node, not outlet
        make_node(session, node_name="W1")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        panel = self._build(session, conduit)

        self.assertIsNotNone(panel.inlet.node)
        self.assertIsNone(panel.outlet.node)
        self.assertIsNone(panel.outlet.invert_elevation)
        self.assertIsNone(panel.outlet.max_depth)
        self.assertIsNone(panel.outlet.max_water_depth)
        self.assertIsNone(panel.outlet.hours_flooded)
        self.assertIsNone(panel.outlet.flood_volume)
        self.assertIsNone(panel.outlet.ground_elevation)

    def test_node_card_carries_simulated_water_and_flooding_results(self):
        session = make_session(self.user)
        make_node(
            session,
            node_name="W1",
            max_water_depth=1.23456,
            hours_flooded=0.42,
            flood_volume=0.01345,
        )
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        card = self._build(session, conduit).inlet

        self.assertEqual(card.max_water_depth, 1.23456)
        self.assertEqual(card.hours_flooded, 0.42)
        self.assertEqual(card.flood_volume, 0.01345)

    def test_node_card_is_flooded_for_positive_hours_or_volume(self):
        """SWMM rounds the hours to 0.00 h for a short flood, so the volume decides too."""
        cases = (
            (None, None, False),
            (0.0, 0.0, False),
            (0.42, None, True),
            (0.0, 0.003, True),
            (None, 0.003, True),
        )
        for index, (hours, volume, expected) in enumerate(cases, start=1):
            with self.subTest(hours=hours, volume=volume):
                session = make_session(self.user)
                node_name = f"W{index}"
                make_node(session, node_name=node_name, hours_flooded=hours, flood_volume=volume)
                conduit = make_conduit(
                    session,
                    conduit_name=f"K{index}",
                    inlet_node=node_name,
                    outlet_node="W-out",
                    subcatchment="-",
                )

                self.assertIs(self._build(session, conduit).inlet.is_flooded, expected)

    # ------------------------------------------------------------------
    # ground_elevation edge cases
    # ------------------------------------------------------------------

    def test_invert_null_gives_ground_none(self):
        session = make_session(self.user)
        make_node(session, node_name="W1", invert_elevation=None, max_depth=3.0)
        make_node(session, node_name="W2")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        panel = self._build(session, conduit)

        self.assertIsNone(panel.inlet.invert_elevation)
        self.assertEqual(panel.inlet.max_depth, 3.0)
        self.assertIsNone(panel.inlet.ground_elevation)

    def test_max_depth_null_gives_ground_none(self):
        session = make_session(self.user)
        make_node(session, node_name="W1", invert_elevation=100.0, max_depth=None)
        make_node(session, node_name="W2")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        panel = self._build(session, conduit)

        self.assertEqual(panel.inlet.invert_elevation, 100.0)
        self.assertIsNone(panel.inlet.max_depth)
        self.assertIsNone(panel.inlet.ground_elevation)

    def test_invert_zero_gives_ground_equals_max_depth(self):
        session = make_session(self.user)
        make_node(session, node_name="W1", invert_elevation=0.0, max_depth=5.0)
        make_node(session, node_name="W2")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        panel = self._build(session, conduit)

        self.assertEqual(panel.inlet.invert_elevation, 0.0)
        self.assertEqual(panel.inlet.ground_elevation, 5.0)

    # ------------------------------------------------------------------
    # incoming / outgoing: exclude current conduit, sorted
    # ------------------------------------------------------------------

    def test_incoming_outgoing_exclude_current_and_sorted(self):
        session = make_session(self.user)
        make_node(session, node_name="W1")
        make_node(session, node_name="W2")
        target = make_conduit(session, conduit_name="K5", inlet_node="W1", outlet_node="W2", subcatchment="-")
        # K3 flows into W1 (outlet_node=W1) → incoming for inlet card
        make_conduit(session, conduit_name="K3", inlet_node="W0", outlet_node="W1", subcatchment="-")
        # K1 flows into W1 → incoming for inlet card
        make_conduit(session, conduit_name="K1", inlet_node="W9", outlet_node="W1", subcatchment="-")
        # K7 flows out of W2 (inlet_node=W2) → outgoing for outlet card
        make_conduit(session, conduit_name="K7", inlet_node="W2", outlet_node="W3", subcatchment="-")

        panel = self._build(session, target)

        self.assertEqual(
            [c.conduit_name for c in panel.inlet.incoming],
            ["K1", "K3"],
        )
        self.assertEqual(panel.inlet.outgoing, [])
        self.assertEqual(panel.outlet.incoming, [])
        self.assertEqual(
            [c.conduit_name for c in panel.outlet.outgoing],
            ["K7"],
        )

    # ------------------------------------------------------------------
    # anchor_id
    # ------------------------------------------------------------------

    def test_anchor_ids(self):
        session = make_session(self.user)
        make_node(session, node_name="W1")
        make_node(session, node_name="W2")
        sbc = make_subcatchment(session, subcatchment_name="P1", outlet="W1")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="P1")

        panel = self._build(session, conduit)

        self.assertEqual(panel.inlet.anchor_id, "node-inlet")
        self.assertEqual(panel.outlet.anchor_id, "node-outlet")
        self.assertEqual(panel.direct_catchments[0].anchor_id, f"catchment-{sbc.pk}")
        self.assertEqual(catchment_anchor(sbc), f"catchment-{sbc.pk}")

    # ------------------------------------------------------------------
    # self-loop (inlet_node == outlet_node)
    # ------------------------------------------------------------------

    def test_self_loop_both_cards_same_node_different_anchor(self):
        session = make_session(self.user)
        make_node(session, node_name="W1")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W1", subcatchment="-")

        panel = self._build(session, conduit)

        self.assertIsNotNone(panel.inlet.node)
        self.assertIsNotNone(panel.outlet.node)
        self.assertEqual(panel.inlet.node.pk, panel.outlet.node.pk)
        self.assertEqual(panel.inlet.anchor_id, "node-inlet")
        self.assertEqual(panel.outlet.anchor_id, "node-outlet")
        self.assertNotEqual(panel.inlet.anchor_id, panel.outlet.anchor_id)

    # ------------------------------------------------------------------
    # diagram_json: new keys and null handling
    # ------------------------------------------------------------------

    def test_diagram_json_contains_new_keys(self):
        session = make_session(self.user)
        make_node(session, node_name="W1", invert_elevation=100.5)
        make_node(session, node_name="W2", invert_elevation=99.3)
        conduit = make_conduit(
            session,
            conduit_name="K1",
            inlet_node="W1",
            outlet_node="W2",
            subcatchment="-",
            inlet_max_depth=3.5,
            outlet_max_depth=2.8,
        )

        panel = self._build(session, conduit)
        data = json.loads(panel.diagram_json)

        self.assertAlmostEqual(data["inletInvert"], 100.5)
        self.assertAlmostEqual(data["outletInvert"], 99.3)
        self.assertAlmostEqual(data["inletMaxDepth"], 3.5)
        self.assertAlmostEqual(data["outletMaxDepth"], 2.8)

    def test_diagram_data_contains_node_water_and_flooding_results_without_rounding(self):
        session = make_session(self.user)
        make_node(
            session,
            node_name="W1",
            max_water_depth=1.23456,
            hours_flooded=0.42,
            flood_volume=0.01345,
        )
        make_node(
            session,
            node_name="W2",
            max_water_depth=2.34567,
            hours_flooded=1.25,
            flood_volume=0.98765,
        )
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        data = self._build(session, conduit).diagram_data

        self.assertEqual(
            {
                key: data[key]
                for key in (
                    "inletWaterDepth",
                    "outletWaterDepth",
                    "inletFloodHours",
                    "outletFloodHours",
                    "inletFloodVolume",
                    "outletFloodVolume",
                )
            },
            {
                "inletWaterDepth": 1.23456,
                "outletWaterDepth": 2.34567,
                "inletFloodHours": 0.42,
                "outletFloodHours": 1.25,
                "inletFloodVolume": 0.01345,
                "outletFloodVolume": 0.98765,
            },
        )

    def test_diagram_data_carries_the_server_side_flooding_verdict(self):
        """The drawing must not re-derive flooding from the hours SWMM rounded to 0.00 h."""
        session = make_session(self.user)
        make_node(session, node_name="W1", hours_flooded=0.0, flood_volume=0.003)
        make_node(session, node_name="W2", hours_flooded=0.0, flood_volume=0.0)
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        data = self._build(session, conduit).diagram_data

        self.assertIs(data["inletFlooded"], True)
        self.assertIs(data["outletFlooded"], False)

    def test_diagram_data_flooding_verdict_is_false_for_missing_nodes(self):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        data = self._build(session, conduit).diagram_data

        self.assertIs(data["inletFlooded"], False)
        self.assertIs(data["outletFlooded"], False)

    def test_diagram_data_node_results_are_null_when_nodes_are_missing(self):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        data = self._build(session, conduit).diagram_data

        for key in (
            "inletWaterDepth",
            "outletWaterDepth",
            "inletFloodHours",
            "outletFloodHours",
            "inletFloodVolume",
            "outletFloodVolume",
        ):
            with self.subTest(key=key):
                self.assertIsNone(data[key])

    def test_diagram_data_node_results_are_null_when_stored_values_are_null(self):
        session = make_session(self.user)
        make_node(session, node_name="W1", max_water_depth=None, hours_flooded=None, flood_volume=None)
        make_node(session, node_name="W2", max_water_depth=None, hours_flooded=None, flood_volume=None)
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        data = self._build(session, conduit).diagram_data

        for key in (
            "inletWaterDepth",
            "outletWaterDepth",
            "inletFloodHours",
            "outletFloodHours",
            "inletFloodVolume",
            "outletFloodVolume",
        ):
            with self.subTest(key=key):
                self.assertIsNone(data[key])

    def test_diagram_json_invert_null_when_node_missing(self):
        session = make_session(self.user)
        # No nodes created
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        panel = self._build(session, conduit)
        data = json.loads(panel.diagram_json)

        self.assertIsNone(data["inletInvert"])
        self.assertIsNone(data["outletInvert"])

    def test_diagram_json_invert_null_when_invert_elevation_null(self):
        session = make_session(self.user)
        make_node(session, node_name="W1", invert_elevation=None)
        make_node(session, node_name="W2", invert_elevation=None)
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        panel = self._build(session, conduit)
        data = json.loads(panel.diagram_json)

        self.assertIsNone(data["inletInvert"])
        self.assertIsNone(data["outletInvert"])

    # ------------------------------------------------------------------
    # query budget
    # ------------------------------------------------------------------

    def test_build_uses_at_most_four_queries(self):
        session = make_session(self.user)
        make_node(session, node_name="W1", invert_elevation=100.0)
        make_node(session, node_name="W2", invert_elevation=99.0)
        make_subcatchment(session, subcatchment_name="P1", outlet="W1")
        make_subcatchment(session, subcatchment_name="P2", outlet="W1")
        make_subcatchment(session, subcatchment_name="P3", outlet="W2")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="P1")
        make_conduit(session, conduit_name="K2", inlet_node="W0", outlet_node="W1", subcatchment="-")
        make_conduit(session, conduit_name="K3", inlet_node="W2", outlet_node="W3", subcatchment="-")

        with CaptureQueriesContext(connection) as ctx:
            panel = self._build(session, conduit)

        self.assertLessEqual(
            len(ctx.captured_queries),
            4,
            "queries: " + "\n".join(q["sql"] for q in ctx.captured_queries),
        )
        # Sanity: the panel was actually built
        self.assertEqual(len(panel.direct_catchments), 2)
        self.assertIsNotNone(panel.model_catchment)

    def test_build_without_model_catchment_uses_at_most_three_queries(self):
        session = make_session(self.user)
        make_node(session, node_name="W1")
        make_node(session, node_name="W2")
        conduit = make_conduit(session, conduit_name="K1", inlet_node="W1", outlet_node="W2", subcatchment="-")

        with CaptureQueriesContext(connection) as ctx:
            self._build(session, conduit)

        self.assertLessEqual(
            len(ctx.captured_queries),
            3,
            "queries: " + "\n".join(q["sql"] for q in ctx.captured_queries),
        )

    # ------------------------------------------------------------------
    # criteria section, design rules and prediction card
    # ------------------------------------------------------------------

    def test_filling_percent_is_the_filling_over_the_diameter(self):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K1", filling=0.2, geom1=0.4)

        panel = self._build(session, conduit)

        self.assertAlmostEqual(panel.filling_percent, 50.0)

    def test_filling_percent_is_clamped_to_a_full_pipe(self):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K1", filling=0.4, geom1=0.3)

        panel = self._build(session, conduit)

        self.assertEqual(panel.filling_percent, 100.0)

    def test_filling_percent_is_none_for_a_conduit_without_a_diameter(self):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K1", filling=0.2, geom1=0.0)

        panel = self._build(session, conduit)

        self.assertIsNone(panel.filling_percent)

    def test_checks_passed_counts_the_seven_validation_flags(self):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K1", val_max_fill=0, val_coverage=0)

        panel = self._build(session, conduit)

        self.assertEqual(panel.checks_total, 7)
        self.assertEqual(panel.checks_passed, 5)
        self.assertEqual(panel.checks_failed, 2)

    def test_predictions_cover_all_nine_classes_sorted_by_confidence(self):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K1", recommendation="valid")

        panel = self._build(session, conduit)

        self.assertEqual(panel.prediction_selected.key, "valid")
        self.assertEqual(len(panel.prediction_others), 8)
        values = [row.value for row in panel.prediction_others]
        self.assertEqual(values, sorted(values, reverse=True))
        keys = {row.key for row in panel.prediction_others} | {panel.prediction_selected.key}
        self.assertEqual(len(keys), 9)

    def test_unknown_recommendation_selects_no_class_and_keeps_all_nine(self):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K1", recommendation="")

        panel = self._build(session, conduit)

        self.assertIsNone(panel.prediction_selected)
        self.assertEqual(len(panel.prediction_others), 9)

    def test_a_missing_confidence_sorts_as_zero_and_keeps_its_own_value(self):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K1", recommendation="valid", confidence_pump=None)

        panel = self._build(session, conduit)

        pump = next(row for row in panel.prediction_others if row.key == "pump")
        self.assertIsNone(pump.value)
        self.assertEqual(panel.prediction_others[-1].key, "pump")


class ValidationCheckRowTests(TestCase):
    """The seven ``CheckRow`` items: stored verdict, measured value, threshold.

    The thresholds shown to the reader have to be the ones the analysis compared
    against, so every case below recomputes the flag with the code of ``sa.core`` and
    stores that verdict on the conduit before asking the service for the tiles.
    """

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="checks@example.com")

    def _rows(self, session, conduit):
        return {row.key: row for row in ConduitPanelService.build(session, conduit).checks}

    def test_rows_follow_the_flag_order_and_repeat_the_stored_verdict(self):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K1", val_min_v=0, val_coverage=0)

        checks = ConduitPanelService.build(session, conduit).checks

        self.assertEqual([row.key for row in checks], list(VALIDATION_FLAGS))
        self.assertEqual(
            [row.label for row in checks],
            ["Max Filling", "Max Velocity", "Min Velocity", "Max Slope", "Min Slope", "Depth", "Coverage"],
        )
        self.assertEqual([row.passed for row in checks], [True, True, False, True, True, True, False])

    def test_values_and_thresholds_match_the_verdict_of_the_analysis_code(self):
        session = make_session(self.user, frost_zone=0.8)
        cases = {
            # name: conduit fields, then the expected value/limit texts per check.
            "slow-and-flat": (
                {"geom1": 0.6, "filling": 0.23, "max_v": 0.35, "slope_per_mile": 0.32},
                {
                    "val_max_fill": ("0.23 m (38 %)", "≤ 0.50 m (82.7 %)"),
                    "val_max_v": ("0.35 m/s", "≤ 5.00 m/s"),
                    "val_min_v": ("0.35 m/s", "≥ 0.70 m/s"),
                    "val_max_slope": ("0.32 ‰", "≤ 54.03 ‰"),
                    "val_min_slope": ("0.32 ‰", "≥ 1.21 ‰"),
                },
            ),
            "full-and-fast": (
                {"geom1": 0.3, "filling": 0.3, "max_v": 5.4, "slope_per_mile": 180.0},
                {
                    "val_max_fill": ("0.30 m (100 %)", "≤ 0.25 m (82.7 %)"),
                    "val_max_v": ("5.40 m/s", "≤ 5.00 m/s"),
                    "val_min_v": ("5.40 m/s", "≥ 0.70 m/s"),
                    "val_max_slope": ("180.00 ‰", "≤ 54.03 ‰"),
                    "val_min_slope": ("180.00 ‰", "≥ 1.21 ‰"),
                },
            ),
            "deep-and-bare": (
                {"inlet_max_depth": 3.4, "outlet_max_depth": 8.5, "inlet_ground_cover": 0.75, "outlet_ground_cover": 2.0},
                {
                    "val_depth": ("8.50 m", "≤ 8.00 m"),
                    "val_coverage": ("0.75 m", "≥ 0.80 m (frost zone)"),
                },
            ),
        }
        for name, (fields, expected) in cases.items():
            with self.subTest(case=name):
                conduit = make_conduit(
                    session,
                    conduit_name=f"K-{name}",
                    min_required_slope=1.21,
                    max_allowable_slope=54.03,
                    **fields,
                    **_flags_from_core(session, fields),
                )
                rows = self._rows(session, conduit)
                for key, (actual, limit) in expected.items():
                    self.assertEqual((rows[key].actual, rows[key].limit), (actual, limit), key)
                    self.assertEqual(rows[key].passed, bool(getattr(conduit, key)), key)

    def test_a_borderline_value_gains_decimals_instead_of_contradicting_the_verdict(self):
        session = make_session(self.user)
        # 0.699 m/s rounds to the limit itself at two decimals, and the pipeline failed
        # the check: the tile has to show the decimal that makes the two differ.  The
        # analysis rounds its columns to two places before saving them, so a stored
        # third decimal comes from a fixture like this one rather than from a run - the
        # guard is on the mechanism, which a live threshold (Max Filling) does exercise.
        conduit = make_conduit(session, conduit_name="K-borderline", max_v=0.699, val_min_v=0)

        row = self._rows(session, conduit)["val_min_v"]

        self.assertFalse(validate_min_velocity(0.699))
        self.assertEqual((row.actual, row.limit), ("0.699 m/s", "≥ 0.700 m/s"))

    def test_a_value_that_clears_the_limit_by_a_hair_keeps_two_decimals(self):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K-exact", max_v=0.7, val_min_v=1)

        row = self._rows(session, conduit)["val_min_v"]

        self.assertTrue(validate_min_velocity(0.7))
        self.assertEqual((row.actual, row.limit), ("0.70 m/s", "≥ 0.70 m/s"))

    def test_a_slope_that_rounds_onto_its_limit_gains_decimals_too(self):
        session = make_session(self.user)
        conduit = make_conduit(
            session,
            conduit_name="K-slope-edge",
            slope_per_mile=54.0349,
            max_allowable_slope=54.034,
            val_max_slope=0,
        )

        row = self._rows(session, conduit)["val_max_slope"]

        self.assertEqual((row.actual, row.limit), ("54.035 ‰", "≤ 54.034 ‰"))

    def test_a_diameter_outside_the_standard_range_names_that_constraint(self):
        session = make_session(self.user)
        # validate_filling rejects the diameter itself, so no filling height explains
        # the verdict and the tile has to say what failed instead.
        conduit = make_conduit(session, conduit_name="K-odd", geom1=0.15, filling=0.05, val_max_fill=0)

        row = self._rows(session, conduit)["val_max_fill"]

        self.assertFalse(validate_filling(0.05, 0.15))
        self.assertEqual((row.actual, row.limit), ("0.05 m", "Ø outside 0.20–2.00 m"))

    def test_stored_numbers_that_disagree_with_the_stored_flag_keep_the_flag(self):
        session = make_session(self.user)
        # Legacy sessions carry placeholder slope limits; the panel reports the run, so
        # the tile shows the stored verdict next to the stored numbers, unretouched.
        conduit = make_conduit(session, conduit_name="K-legacy", slope_per_mile=3.49, max_allowable_slope=1.0)

        row = self._rows(session, conduit)["val_max_slope"]

        self.assertTrue(row.passed)
        self.assertEqual((row.actual, row.limit), ("3.49 ‰", "≤ 1.00 ‰"))


class CriterionCardTests(TestCase):
    """The six ``CriterionCard`` items: one per quantity, built from the ``CheckRow`` items."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="criteria@example.com")

    def _cards(self, session, conduit):
        return {card.key: card for card in ConduitPanelService.build(session, conduit).criteria}

    def test_cards_follow_the_spec_order_and_reuse_the_row_texts(self):
        session = make_session(self.user, frost_zone=1.2)
        conduit = make_conduit(session, conduit_name="K1", max_q=0.0404, geom1=0.3, filling=0.3, val_max_fill=0)

        panel = ConduitPanelService.build(session, conduit)
        rows = {row.key: row for row in panel.checks}

        self.assertEqual(
            [(c.key, c.label, c.unit) for c in panel.criteria],
            [
                ("max_q", "Max flow", "m³/s"),
                ("val_max_fill", "Filling", "%"),
                ("velocity", "Velocity", "m/s"),
                ("slope", "Slope", "‰"),
                ("val_depth", "Depth", "m"),
                ("val_coverage", "Cover", "m"),
            ],
        )
        cards = {c.key: c for c in panel.criteria}
        self.assertEqual((cards["max_q"].value, cards["max_q"].criterion, cards["max_q"].passed), ("0.040", "SWMM peak", None))
        self.assertEqual((cards["val_max_fill"].value, cards["val_max_fill"].criterion), ("100", rows["val_max_fill"].limit))
        self.assertEqual((cards["val_max_fill"].passed, cards["val_max_fill"].status), (False, "exceeds limit"))
        self.assertEqual((cards["velocity"].value, cards["velocity"].criterion), ("1.20", "0.70 – 5.00 m/s"))
        self.assertEqual((cards["slope"].value, cards["slope"].criterion), ("5.00", "2.00 – 40.00 ‰"))
        self.assertEqual((cards["val_depth"].value, cards["val_depth"].criterion), ("3.00", "≤ 8.00 m"))
        # The frost zone of the session is the Cover criterion.
        self.assertEqual((cards["val_coverage"].value, cards["val_coverage"].criterion), ("1.50", "≥ 1.20 m (frost zone)"))

    def test_each_stored_flag_fails_exactly_its_card_and_names_the_side(self):
        session = make_session(self.user)
        expected = {
            "val_max_fill": ("val_max_fill", "exceeds limit"),
            "val_max_v": ("velocity", "above maximum"),
            "val_min_v": ("velocity", "below minimum"),
            "val_max_slope": ("slope", "above maximum"),
            "val_min_slope": ("slope", "below minimum"),
            "val_depth": ("val_depth", "exceeds limit"),
            "val_coverage": ("val_coverage", "below minimum"),
        }
        self.assertEqual(set(expected), set(VALIDATION_FLAGS))
        for flag, (key, status) in expected.items():
            with self.subTest(flag=flag):
                conduit = make_conduit(session, conduit_name=f"K-{flag}", **{flag: 0})
                cards = self._cards(session, conduit)
                self.assertEqual([k for k, c in cards.items() if c.passed is False], [key])
                self.assertEqual(cards[key].status, status)
                for other in cards.values():
                    if other.passed:
                        self.assertEqual(other.status, "within limits")

    def test_both_sides_failing_is_one_failed_card_reporting_the_maximum(self):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K1", val_max_slope=0, val_min_slope=0)

        card = self._cards(session, conduit)["slope"]

        self.assertEqual((card.passed, card.status), (False, "above maximum"))

    def test_a_range_takes_the_raised_precision_of_the_failed_side(self):
        session = make_session(self.user)
        # The min side gained a decimal to stay clear of its verdict (see
        # ValidationCheckRowTests); the range joins the two thresholds as they are.
        conduit = make_conduit(session, conduit_name="K1", max_v=0.699, val_min_v=0)

        card = self._cards(session, conduit)["velocity"]

        self.assertEqual((card.value, card.criterion, card.status), ("0.699", "0.700 – 5.00 m/s", "below minimum"))

    def test_the_filling_share_gains_a_decimal_rather_than_contradict_the_limit(self):
        session = make_session(self.user)
        # 82.6 % passed; "83 %" beside "(82.7 %)" would read as a failure.
        passed = make_conduit(session, conduit_name="K-pass", geom1=1.0, filling=0.826, val_max_fill=1)
        # 82.8 % failed: the whole percent already reads as above the limit.
        failed = make_conduit(session, conduit_name="K-fail", geom1=1.0, filling=0.828, val_max_fill=0)

        self.assertEqual(self._cards(session, passed)["val_max_fill"].value, "82.6")
        self.assertEqual(self._cards(session, failed)["val_max_fill"].value, "83")

    def test_filling_card_for_a_diameter_outside_the_range_and_without_a_diameter(self):
        session = make_session(self.user)
        odd = make_conduit(session, conduit_name="K-odd", geom1=0.15, filling=0.05, val_max_fill=0)
        none = make_conduit(session, conduit_name="K-zero", geom1=0.0, filling=0.05, val_max_fill=0)

        card = self._cards(session, odd)["val_max_fill"]
        self.assertEqual(
            (card.value, card.criterion, card.passed, card.status),
            ("33", "Ø outside 0.20–2.00 m", False, "diameter out of range"),
        )
        self.assertEqual(self._cards(session, none)["val_max_fill"].value, "—")

    def test_a_slope_range_with_the_minimum_above_the_maximum_says_no_value_can_pass(self):
        session = make_session(self.user)
        # Like K18: a dry pipe whose stored minimum lies above its maximum.
        dry = make_conduit(session, conduit_name="K-dry", **DRY_CONDUIT)

        cards = self._cards(session, dry)

        self.assertEqual(
            (cards["slope"].value, cards["slope"].criterion, cards["slope"].passed, cards["slope"].status),
            ("1.41", "no feasible slope (min 229.72 > max 134.56\u00a0‰)", False, "no feasible range"),
        )
        # The velocity card of a dry pipe is right as it is: below the minimum.
        self.assertEqual((cards["velocity"].passed, cards["velocity"].status), (False, "below minimum"))
        # The stored flags still count: the verdicts and "N of 7" do not change.
        self.assertEqual(ConduitPanelService.build(session, dry).checks_failed, 2)

    def test_a_passed_slope_card_keeps_its_text_even_with_the_minimum_above_the_maximum(self):
        # Stored flags cannot produce this, but the card must not break if they do.
        session = make_session(self.user)
        odd = make_conduit(session, conduit_name="K-odd", **{**DRY_CONDUIT, "val_min_slope": 1})

        card = self._cards(session, odd)["slope"]

        self.assertEqual((card.criterion, card.passed, card.status), ("229.72 – 134.56 ‰", True, "within limits"))


class DesignRuleTests(TestCase):
    """``ConduitPanel.design_rules``: only the changes the stored action flags ask for."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="rules@example.com")

    def _rules(self, **fields):
        session = make_session(self.user)
        conduit = make_conduit(session, conduit_name="K1", **fields)
        return [(rule.key, rule.label, rule.change) for rule in ConduitPanelService.build(session, conduit).design_rules]

    def test_no_flag_no_rule(self):
        self.assertEqual(self._rules(), [])

    def test_increase_diameter_goes_to_the_minimum_diameter(self):
        self.assertEqual(
            self._rules(increase_dia=1, geom1=0.3, min_diameter=0.4),
            [("increase_dia", "Diameter", "0.30 → 0.40 m")],
        )

    def test_reduce_diameter_has_no_target(self):
        self.assertEqual(self._rules(reduce_dia=1), [("reduce_dia", "Diameter", "could be reduced")])

    def test_increase_slope_goes_to_the_minimum_slope(self):
        self.assertEqual(
            self._rules(increase_slope=1, slope_per_mile=0.28, min_required_slope=2.04),
            [("increase_slope", "Slope", "0.28 → 2.04 ‰")],
        )

    def test_reduce_slope_goes_to_the_maximum_slope(self):
        self.assertEqual(
            self._rules(reduce_slope=1, slope_per_mile=180.0, max_allowable_slope=134.56),
            [("reduce_slope", "Slope", "180.00 → 134.56 ‰")],
        )

    def test_a_dry_conduit_with_an_empty_slope_range_gets_no_slope_target(self):
        self.assertEqual(
            self._rules(**DRY_CONDUIT),
            [("increase_slope", "Slope", "no feasible value — no flow in the run")],
        )
        # Either sign of no flow is enough.
        self.assertEqual(
            self._rules(**{**DRY_CONDUIT, "filling": 0.1}),
            [("increase_slope", "Slope", "no feasible value — no flow in the run")],
        )
        self.assertEqual(
            self._rules(**{**DRY_CONDUIT, "max_q": 0.02}),
            [("increase_slope", "Slope", "no feasible value — no flow in the run")],
        )

    def test_an_empty_slope_range_with_flow_gets_no_target_and_no_flow_note(self):
        self.assertEqual(
            self._rules(**{**DRY_CONDUIT, "filling": 0.1, "max_q": 0.02}),
            [("increase_slope", "Slope", "no feasible value")],
        )
        self.assertEqual(
            self._rules(**{**DRY_CONDUIT, "increase_slope": 0, "reduce_slope": 1, "filling": 0.1, "max_q": 0.02}),
            [("reduce_slope", "Slope", "no feasible value")],
        )

    def test_an_empty_slope_range_prints_one_slope_row_and_keeps_the_diameter_rules(self):
        self.assertEqual(
            self._rules(**{**DRY_CONDUIT, "reduce_slope": 1, "increase_dia": 1, "min_diameter": 0.4}),
            [
                ("increase_dia", "Diameter", "0.30 → 0.40 m"),
                ("increase_slope", "Slope", "no feasible value — no flow in the run"),
            ],
        )
        # Without a slope flag an empty range adds nothing.
        self.assertEqual(
            self._rules(**{**DRY_CONDUIT, "increase_slope": 0, "reduce_dia": 1}),
            [
                ("reduce_dia", "Diameter", "could be reduced"),
            ],
        )

    def test_the_flags_decide_not_a_fresh_comparison(self):
        # The stored numbers say the pipe is big enough, the stored flag says otherwise:
        # the rule follows the flag, as the criteria cards do.
        self.assertEqual(
            self._rules(increase_dia=1, geom1=0.5, min_diameter=0.4),
            [("increase_dia", "Diameter", "0.50 → 0.40 m")],
        )
        self.assertEqual(self._rules(geom1=0.3, min_diameter=0.4), [])


def _flags_from_core(session, fields: dict) -> dict:
    """The seven flags for ``fields``, computed exactly as ``sa.core.conduits`` does."""
    defaults = {
        "geom1": 0.4,
        "filling": 0.2,
        "max_v": 1.2,
        "slope_per_mile": 5.0,
        "min_required_slope": 1.21,
        "max_allowable_slope": 54.03,
        "inlet_max_depth": 3.0,
        "outlet_max_depth": 3.0,
        "inlet_ground_cover": 1.5,
        "outlet_ground_cover": 1.5,
    }
    values = {**defaults, **fields}
    return {
        "val_max_fill": int(validate_filling(values["filling"], values["geom1"])),
        "val_max_v": int(validate_max_velocity(values["max_v"])),
        "val_min_v": int(validate_min_velocity(values["max_v"])),
        "val_max_slope": int(values["slope_per_mile"] <= values["max_allowable_slope"]),
        "val_min_slope": int(values["slope_per_mile"] >= values["min_required_slope"]),
        # ValDepth compares each node invert with its ground minus the maximum depth,
        # which is the node depth itself (sa/core/conduits.py:118-121).
        "val_depth": int(values["inlet_max_depth"] <= max_depth_value and values["outlet_max_depth"] <= max_depth_value),
        "val_coverage": int(
            values["inlet_ground_cover"] >= session.frost_zone and values["outlet_ground_cover"] >= session.frost_zone
        ),
    }
