"""T7 (kill-switch regression): the MLP fallback must be reachable.

On pre-fix code the GNN model was referenced through an always-truthy
``_LazyModelProxy`` (``gnn_recommendation is not None`` was unconditionally
True), so the MLP fallback branch was unreachable even when the GNN model had
failed to load. These tests pin the fixed behaviour: model selection returns
MLP whenever GNN is disabled, unavailable, or its graph cannot be built.

They exercise ``DataManager._select_recommendation_service`` directly on a bare
instance (no swmmio / SWMM run, no TensorFlow) so the selection logic is tested
in isolation.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from sa.core.data import DataManager


def _bare_data_manager():
    """A DataManager instance without running swmmio.Model.__init__."""
    dm = DataManager.__new__(DataManager)
    dm.dfc = pd.DataFrame(
        {
            "Name": ["C0", "C1"],
            "InletNode": ["N0", "N1"],
            "OutletNode": ["N1", "N2"],
        }
    )
    dm.recommendation_service = None
    return dm


def test_gnn_disabled_selects_mlp():
    dm = _bare_data_manager()
    with (
        patch("sa.core.data.gnn_enabled", return_value=False),
        patch("sa.core.data.get_gnn_recommendation", return_value=MagicMock()),
        patch("sa.core.data.get_recommendation", return_value=MagicMock()),
    ):
        dm._select_recommendation_service()
    assert dm.recommendation_service.model_name == "MLP"


def test_gnn_enabled_but_model_unavailable_falls_back_to_mlp():
    """The core regression: GNN enabled but the model failed to load (None).

    Pre-fix this returned the always-truthy proxy and selected GNN; now it must
    fall back to MLP.
    """
    dm = _bare_data_manager()
    with (
        patch("sa.core.data.gnn_enabled", return_value=True),
        patch("sa.core.data.get_gnn_recommendation", return_value=None),
        patch("sa.core.data.get_recommendation", return_value=MagicMock()),
    ):
        dm._select_recommendation_service()
    assert dm.recommendation_service.model_name == "MLP"
    assert dm.recommendation_service.adjacency is None


def test_graph_build_failure_falls_back_to_mlp():
    """D2: a graph-construction error must yield an explicit MLP fallback, not a
    GNN service served without a graph."""
    dm = _bare_data_manager()
    with (
        patch("sa.core.data.gnn_enabled", return_value=True),
        patch("sa.core.data.get_gnn_recommendation", return_value=MagicMock()),
        patch("sa.core.data.get_recommendation", return_value=MagicMock()),
        patch("sa.core.gnn.build_adjacency_from_dfc", side_effect=RuntimeError("boom")),
    ):
        dm._select_recommendation_service()
    assert dm.recommendation_service.model_name == "MLP"


def test_gnn_selected_when_enabled_available_and_graph_builds():
    """Positive control: with GNN enabled, a model present and a graph built, the
    GNN service is selected and carries a non-None adjacency."""
    dm = _bare_data_manager()
    fake_adjacency = object()
    with (
        patch("sa.core.data.gnn_enabled", return_value=True),
        patch("sa.core.data.get_gnn_recommendation", return_value=MagicMock()),
        patch("sa.core.gnn.build_adjacency_from_dfc", return_value="raw_adj"),
        patch("sa.core.gnn.preprocess_adjacency", return_value=fake_adjacency),
    ):
        dm._select_recommendation_service()
    assert dm.recommendation_service.model_name == "GNN"
    assert dm.recommendation_service.adjacency is fake_adjacency
