"""Integration tests for serving the shipped GNN recommendation model in the Django app.

Verifies that the model loads, binds to the v4_physical contract (66 physical
features), carries its imputation plan, and generates recommendations on the
Wrocław test network. The weights ship with the repository, so a missing file
fails these tests instead of skipping them.
"""

import os
import shutil

import numpy as np
import pytest

from sa.core.conduits import RecommendationService
from sa.core.data import DataManager
from sa.core.enums import RecommendationCategory
from sa.core.graph_constructor import load_gnn_model_weights


WROCLAW_INP = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "examples", "wroclaw_rakowiec.inp")


def test_shipped_model_loads_with_physical_contract():
    """Verify load_gnn_model_weights loads the shipped GNN model with the v4_physical contract."""
    model = load_gnn_model_weights()
    assert model is not None
    assert hasattr(model, "spec")
    assert model.spec.hops == 2
    assert model.spec.direction == "upstream"
    assert model.spec.operator == "legacy_weighted_mean"
    assert model.feature_contract.name == "v4_physical"
    assert model.feature_contract.n_columns == 66
    assert hasattr(model, "imputation_plan")
    assert model.imputation_plan is not None
    assert len(model.imputation_plan["fills"]) == 66


@pytest.fixture
def gnn_enabled_env():
    import sa.core.predictor as predictor_module

    original_env = os.environ.get("GNN_ENABLED")
    original_models_loaded = predictor_module._models_loaded
    original_classifier = predictor_module._classifier
    original_recommendation = predictor_module._recommendation
    original_gnn_recommendation = predictor_module._gnn_recommendation

    os.environ["GNN_ENABLED"] = "true"
    predictor_module._models_loaded = False
    predictor_module._gnn_recommendation = None

    yield

    if original_env is None:
        os.environ.pop("GNN_ENABLED", None)
    else:
        os.environ["GNN_ENABLED"] = original_env
    predictor_module._models_loaded = original_models_loaded
    predictor_module._classifier = original_classifier
    predictor_module._recommendation = original_recommendation
    predictor_module._gnn_recommendation = original_gnn_recommendation


def test_data_manager_gnn_recommendations_on_wroclaw(gnn_enabled_env, tmp_path):
    """Verify DataManager produces recommendations with the shipped GNN model on the Wrocław network."""
    tmp_inp = tmp_path / "wroclaw_rakowiec.inp"
    shutil.copy2(WROCLAW_INP, tmp_inp)

    dm = DataManager(str(tmp_inp), zone=0.8)
    dm.calculate()
    dm.feature_engineering()
    dm.recommendations()

    assert dm.recommendation_service.model_name == "GNN"
    assert dm.dfc is not None
    assert len(dm.dfc) == 62
    assert "recommendation" in dm.dfc.columns
    assert dm.dfc["recommendation"].notna().all()

    # All 9 category confidence columns must be present and non-NaN
    for cat in RecommendationCategory:
        conf_col = f"confidence_{cat.value}"
        assert conf_col in dm.dfc.columns
        assert dm.dfc[conf_col].notna().all()
        assert (dm.dfc[conf_col] >= 0.0).all()
        assert (dm.dfc[conf_col] <= 1.0).all()

    # Check that top classes are predicted
    val_counts = dm.dfc["recommendation"].value_counts().to_dict()
    assert "tank" in val_counts
    assert "slope_increase" in val_counts
    assert val_counts["tank"] > 0


def test_get_model_metadata_respects_gnn_enabled_env(monkeypatch):
    """GNN metadata when enabled; MLP metadata when disabled and its file is installed."""
    from unittest import mock

    from sa.core.predictor import get_model_metadata

    monkeypatch.setenv("GNN_ENABLED", "true")
    meta_gnn = get_model_metadata()
    assert meta_gnn["model_kind"] == "gnn"
    assert meta_gnn["model_id"] == "gnn_recommender_v1"
    assert meta_gnn["feature_contract"] == "v4_physical"

    monkeypatch.setenv("GNN_ENABLED", "false")
    with mock.patch("sa.core.predictor._sha256_file_safe", return_value="ab" * 32):
        meta_mlp = get_model_metadata()
    assert meta_mlp["model_kind"] == "mlp"
    assert meta_mlp["model_id"] == "mlp_historical_20250615"
    assert meta_mlp["weights_sha256"] == "ab" * 32
    assert meta_mlp["feature_contract"] == "v1_historical_33"


def test_get_model_metadata_names_no_model_when_none_is_installed(monkeypatch):
    """With the GNN switched off and no MLP file, no model identity is reported."""
    from unittest import mock

    from sa.core.predictor import get_model_metadata

    monkeypatch.setenv("GNN_ENABLED", "false")
    with mock.patch("sa.core.predictor._sha256_file_safe", return_value=None):
        meta = get_model_metadata()
    assert meta == {
        "model_kind": "none",
        "model_id": None,
        "weights_sha256": None,
        "feature_contract": None,
        "target_contract": None,
        "gnn_enabled": False,
    }


@pytest.mark.slow
def test_full_pipeline_uses_graph_and_differs_from_features_only(gnn_enabled_env, tmp_path):
    """The graph-served confidences must differ from a features-only pass of the same model."""
    from sa.core.tests import TEST_FILE

    tmp_inp = tmp_path / "test_file.inp"
    shutil.copy2(TEST_FILE, tmp_inp)
    confidence_columns = [f"confidence_{category.value}" for category in RecommendationCategory]

    with DataManager(str(tmp_inp)) as dm:
        assert dm.recommendation_service.model_name == "GNN"

        assert "recommendation" in dm.dfc.columns
        assert dm.dfc["recommendation"].notna().all()
        for col in confidence_columns:
            assert col in dm.dfc.columns
        assert not dm.dfc[confidence_columns].isna().any().any()

        row_sums = dm.dfc[confidence_columns].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-2)

        graph_confidences = dm.dfc[confidence_columns].copy()
        gnn_model = dm.gnn_model
        base_dfc = dm.dfc.drop(columns=confidence_columns + ["recommendation"]).copy()

    assert gnn_model is not None

    # Reusing the same model under model_name="MLP" drives it through
    # self.model.predict(input_data) with no adjacency at all.
    featonly_service = RecommendationService(base_dfc, gnn_model, model_name="MLP")
    featonly_confidences = featonly_service.recommendations()[confidence_columns]

    assert not np.allclose(graph_confidences.to_numpy(), featonly_confidences.to_numpy(), atol=1e-9)
