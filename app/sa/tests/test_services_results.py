"""Keys of the tables on the results page (``CalculationRetrievalService``)."""

from __future__ import annotations

import math

import pandas as pd
from django.test import TestCase

from sa.services import CalculationPersistenceService, CalculationRetrievalService

from .factories import make_session, make_user


class ResultsTableKeysTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="keys@example.com")
        cls.session = make_session(cls.user, n_conduits=1, n_nodes=2, n_subcatchments=1)

    def test_conduit_table_names_the_model_catchment_column(self):
        data = CalculationRetrievalService.format_session_data_for_template(self.session)
        conduit_keys = list(data["Conduits Data"][0].keys())

        # The pipeline picks one catchment per conduit for the model feature (outlet
        # node first, see api_map §43); the header says so instead of implying inflow.
        self.assertIn("Subcatchment (model)", conduit_keys)
        self.assertNotIn("Subcatchment", conduit_keys)

    def test_node_table_keeps_its_direct_catchment_column(self):
        data = CalculationRetrievalService.format_session_data_for_template(self.session)

        self.assertIn("Subcatchment", data["Nodes Data"][0].keys())
        self.assertIn("MaxNodeDepth", data["Nodes Data"][0].keys())
        self.assertIn("HoursFlooded", data["Nodes Data"][0].keys())
        self.assertIn("TotalFloodVol", data["Nodes Data"][0].keys())


class NodeResultPersistenceTests(TestCase):
    def test_saves_report_values_and_converts_missing_values_to_none(self):
        user = make_user(email="node-results@example.com")
        session = make_session(user)
        nodes = pd.DataFrame(
            [
                {
                    "MaxDepth": 3.0,
                    "InvertElev": 117.0,
                    "MaxNodeDepth": 1.25,
                    "HoursFlooded": 0.5,
                    "TotalFloodVol": 0.012,
                },
                {
                    "MaxDepth": 2.0,
                    "InvertElev": 116.0,
                    "MaxNodeDepth": math.nan,
                    "HoursFlooded": math.nan,
                },
            ],
            index=["N1", "N2"],
        )

        CalculationPersistenceService._save_node_data(session, nodes)

        first = session.nodes.get(node_name="N1")
        self.assertEqual(first.max_water_depth, 1.25)
        self.assertEqual(first.hours_flooded, 0.5)
        self.assertEqual(first.flood_volume, 0.012)

        second = session.nodes.get(node_name="N2")
        self.assertIsNone(second.max_water_depth)
        self.assertIsNone(second.hours_flooded)
        self.assertIsNone(second.flood_volume)
