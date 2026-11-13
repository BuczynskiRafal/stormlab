"""Persistence of the per-node results (``CalculationPersistenceService``)."""

from __future__ import annotations

import math

import pandas as pd
from django.test import TestCase

from sa.services import CalculationPersistenceService

from .factories import make_session, make_user


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
