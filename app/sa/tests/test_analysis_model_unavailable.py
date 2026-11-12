"""An analysis must not end as a success when no recommendation model could be used."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest import mock

from django.test import TestCase

from sa.models import ConduitData
from sa.services import AnalysisService

from .factories import make_swmm_model, make_user

DEMO_NETWORK = Path(__file__).resolve().parents[3] / "examples" / "demo_network_coordinates.inp"
UNAVAILABLE_MESSAGE = "The recommendation model could not be used"


class RecommendationModelUnavailableTests(TestCase):
    """Runs the full analysis on a temporary copy of the demo network."""

    def setUp(self) -> None:
        self.user = make_user()
        self.swmm_model = make_swmm_model(self.user)
        workdir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)
        self.inp_path = workdir / DEMO_NETWORK.name
        shutil.copy2(DEMO_NETWORK, self.inp_path)

    def _analyze(self, *, gnn_enabled: bool, gnn_model: object | None):
        with (
            mock.patch("sa.core.data.gnn_enabled", return_value=gnn_enabled),
            mock.patch("sa.core.data.get_gnn_recommendation", return_value=gnn_model),
            mock.patch("sa.core.data.get_recommendation", return_value=None),
        ):
            return AnalysisService.analyze_file(
                user=self.user,
                file_path=str(self.inp_path),
                frost_zone=1.2,
                swmm_model=self.swmm_model,
            )

    def _assert_failed_without_results(self, session, success: bool) -> None:
        session.refresh_from_db()
        self.assertFalse(success)
        self.assertEqual(session.status, "failed")
        self.assertEqual(session.error_message, UNAVAILABLE_MESSAGE)
        self.assertFalse(ConduitData.objects.filter(session=session).exists())

    def test_session_fails_when_the_gnn_is_enabled_but_unavailable(self):
        with self.assertLogs("sa.core.data", level="ERROR") as logs:
            session, success = self._analyze(gnn_enabled=True, gnn_model=None)

        self._assert_failed_without_results(session, success)
        self.assertIn("GNN model unavailable", "\n".join(logs.output))

    def test_session_fails_when_the_graph_cannot_be_built(self):
        with (
            mock.patch("sa.core.gnn.build_adjacency_from_dfc", side_effect=RuntimeError("broken network graph")),
            self.assertLogs("sa.core.data", level="ERROR") as logs,
        ):
            session, success = self._analyze(gnn_enabled=True, gnn_model=mock.MagicMock())

        self._assert_failed_without_results(session, success)
        self.assertIn("graph_construction_error: broken network graph", "\n".join(logs.output))
        self.assertNotIn("broken network graph", session.error_message)

    def test_disabled_gnn_still_completes_without_recommendations(self):
        session, success = self._analyze(gnn_enabled=False, gnn_model=None)

        session.refresh_from_db()
        self.assertTrue(success)
        self.assertEqual(session.status, "completed")
        conduits = ConduitData.objects.filter(session=session)
        self.assertTrue(conduits.exists())
        self.assertEqual(set(conduits.values_list("recommendation", flat=True)), {"-"})
