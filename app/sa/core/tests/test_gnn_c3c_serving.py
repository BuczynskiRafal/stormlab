"""Integration tests for C3c GNN serving in the Django app.

Verifies that the winning C3c model (gnn_up_k2__f2_l2) loads correctly,
binds to the v4_physical contract (66 physical features), uses the imputation plan,
and generates expected recommendations across the Wrocław test network.
"""

import os
import shutil
import pytest

from sa.core.data import DataManager
from sa.core.enums import RecommendationCategory
from sa.core.graph_constructor import load_gnn_model_weights
from sa.core.predictor import gnn_model_path


WROCLAW_INP = os.path.abspath("models/recomendations/gnn/wroclaw/wroclaw_pipes.inp")
WROCLAW_RPT = os.path.abspath("models/recomendations/gnn/wroclaw/wroclaw_pipes.rpt")
C3C_WEIGHTS_AVAILABLE = os.path.isfile(gnn_model_path)
requires_c3c_weights = pytest.mark.skipif(not C3C_WEIGHTS_AVAILABLE, reason="C3c GNN model weights not available")


@requires_c3c_weights
def test_c3c_model_loads_with_physical_contract():
    """Verify load_gnn_model_weights loads the C3c GNN model with v4_physical contract."""
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


@requires_c3c_weights
def test_data_manager_c3c_gnn_recommendations_on_wroclaw(gnn_enabled_env, tmp_path):
    """Verify DataManager produces recommendations with C3c GNN model on Wrocław network."""
    if not os.path.isfile(WROCLAW_INP):
        pytest.skip(f"Wrocław test file not found: {WROCLAW_INP}")

    tmp_inp = tmp_path / "wroclaw_pipes.inp"
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
    """Verify get_model_metadata() returns C3c GNN metadata when enabled and MLP metadata when disabled."""
    from sa.core.predictor import get_model_metadata

    monkeypatch.setenv("GNN_ENABLED", "true")
    meta_gnn = get_model_metadata()
    assert meta_gnn["model_kind"] == "gnn"
    assert meta_gnn["model_id"] == "gnn_c3c_winner_v1"
    assert meta_gnn["feature_contract"] == "v4_physical"

    monkeypatch.setenv("GNN_ENABLED", "false")
    meta_mlp = get_model_metadata()
    assert meta_mlp["model_kind"] == "mlp"
    assert meta_mlp["model_id"] == "mlp_historical_20250615"
    assert meta_mlp["feature_contract"] == "v1_historical_33"
