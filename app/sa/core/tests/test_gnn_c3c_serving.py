"""Integration tests for serving the shipped GNN recommendation model in the Django app.

Verifies that the model loads, binds to the v4_physical contract (66 physical
features), carries its imputation plan, and generates recommendations on the
Wrocław test network. The weights ship with the repository, so a missing file
fails these tests instead of skipping them.
"""

import os
import shutil
import pytest

from sa.core.data import DataManager
from sa.core.enums import RecommendationCategory
from sa.core.graph_constructor import load_gnn_model_weights


WROCLAW_INP = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "examples", "wroclaw_rakowiec.inp")


def test_c3c_model_loads_with_physical_contract():
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
    predictor_module._gnn_recommendation = original_gnn_recommendation


def test_data_manager_c3c_gnn_recommendations_on_wroclaw(gnn_enabled_env, tmp_path):
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
    """Verify get_model_metadata() returns GNN metadata when enabled and MLP metadata when disabled."""
    from sa.core.predictor import get_model_metadata

    monkeypatch.setenv("GNN_ENABLED", "true")
    meta_gnn = get_model_metadata()
    assert meta_gnn["model_kind"] == "gnn"
    assert meta_gnn["model_id"] == "gnn_recommender_v1"
    assert meta_gnn["feature_contract"] == "v4_physical"

    monkeypatch.setenv("GNN_ENABLED", "false")
    meta_mlp = get_model_metadata()
    assert meta_mlp["model_kind"] == "mlp"
    assert meta_mlp["model_id"] == "mlp_historical_20250615"
    assert meta_mlp["feature_contract"] == "v1_historical_33"
