"""Golden parity (T5) and full end-to-end DataManager serving (T6) tests.

TensorFlow-dependent throughout; guarded with the same HAS_TF skip pattern
as app/sa/core/tests/test_predictor.py.
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

from sa.core.conduits import RecommendationService
from sa.core.constants import FEATURE_COLUMNS
from sa.core.enums import RecommendationCategory
from sa.core.gnn import build_adjacency_from_dfc, preprocess_adjacency
from sa.core.predictor import GNN_CONFIG

HAS_TF = importlib.util.find_spec("tensorflow") is not None
requires_tf = pytest.mark.skipif(not HAS_TF, reason="requires tensorflow")

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "gnn_golden_fixture.npz")
LEGACY_WEIGHTS_NAME = "graphsage_4hop_model.weights.h5"
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recommendations", LEGACY_WEIGHTS_NAME)
# Serving weights are git-ignored and unprovisioned in CI, so weight-loading
# tests must skip (not error) when the file is absent.
WEIGHTS_AVAILABLE = os.path.exists(WEIGHTS_PATH)
requires_weights = pytest.mark.skipif(not WEIGHTS_AVAILABLE, reason="GNN model weights not available")
N_FEATURES = len(FEATURE_COLUMNS)

SIM_INDICES = (0, 100, 250, 500, 755)
DIVERGENT_SIMS = (0, 500)  # per contract: sims 0/500 diverge, 100/250/755 don't
CONFIDENCE_COLUMNS = [f"confidence_{category.value}" for category in RecommendationCategory]


@pytest.fixture(scope="module")
def golden_fixture():
    if not os.path.exists(FIXTURE_PATH):
        pytest.skip("GNN golden fixture not available")
    return np.load(FIXTURE_PATH, allow_pickle=True)


@pytest.fixture(scope="module")
def gnn_model():
    pytest.importorskip("tensorflow")
    if not WEIGHTS_AVAILABLE:
        pytest.skip("GNN model weights not available")

    import tensorflow as tf
    from sa.core.graph_constructor import GraphSAGEModel

    model = GraphSAGEModel(n_features=N_FEATURES, n_classes=len(RecommendationCategory))
    model([tf.zeros((1, N_FEATURES)), tf.zeros((1, 1))], training=False)
    model.load_weights(WEIGHTS_PATH)
    return model


@requires_tf
class TestGoldenParity:
    """T5: RecommendationService(model_name="GNN") must reproduce the golden
    graph-served softmax for every fixture simulation (K1 fix), and must
    differ from the (buggy, pre-fix) features-only softmax for the two
    fixture sims where the argmax actually diverges.
    """

    @pytest.mark.parametrize("sim_idx", SIM_INDICES)
    def test_confidences_match_graph_softmax(self, golden_fixture, gnn_model, sim_idx):
        conduit_order = list(golden_fixture["conduit_order"])
        adj4 = golden_fixture["adj4_dense"]
        features = golden_fixture[f"features_{sim_idx}"]
        expected = golden_fixture[f"graph_softmax_{sim_idx}"]

        dfc = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        dfc["Name"] = conduit_order

        service = RecommendationService(dfc, gnn_model, model_name="GNN", adjacency=adj4)
        result = service.recommendations()
        actual = result[CONFIDENCE_COLUMNS].to_numpy()

        # The service rounds confidences to 3 dp, so use a looser tolerance here;
        # the raw (pre-rounding) values are checked separately below at 1e-6.
        assert np.allclose(actual, expected, atol=2e-3)
        assert np.array_equal(actual.argmax(axis=-1), expected.argmax(axis=-1))

    @pytest.mark.parametrize("sim_idx", SIM_INDICES)
    def test_raw_predictions_match_before_rounding(self, golden_fixture, gnn_model, sim_idx):
        """Recompute the raw (unrounded) model output directly, bypassing the
        service's 3-dp confidence rounding, for a tight 1e-6 comparison."""
        import tensorflow as tf

        adj4 = golden_fixture["adj4_dense"]
        features = golden_fixture[f"features_{sim_idx}"]
        expected = golden_fixture[f"graph_softmax_{sim_idx}"]

        preds = gnn_model(
            [tf.constant(features, dtype=tf.float32), tf.constant(adj4, dtype=tf.float32)],
            training=False,
        ).numpy()

        assert np.allclose(preds, expected, atol=1e-6)

    @pytest.mark.parametrize("sim_idx", DIVERGENT_SIMS)
    def test_graph_output_differs_from_featonly_for_divergent_sims(self, golden_fixture, sim_idx):
        """Sanity check on the fixture itself: sims 0 and 500 are documented
        to have a graph-vs-features-only argmax divergence; 100/250/755 do
        not. This does not exercise RecommendationService - see
        test_pre_fix_equivalent_path_reproduces_featonly_softmax below for
        the code-level RED demo."""
        graph_sm = golden_fixture[f"graph_softmax_{sim_idx}"]
        featonly_sm = golden_fixture[f"featonly_softmax_{sim_idx}"]

        assert not np.array_equal(graph_sm.argmax(axis=-1), featonly_sm.argmax(axis=-1))

    @pytest.mark.parametrize("sim_idx", DIVERGENT_SIMS)
    def test_pre_fix_equivalent_path_reproduces_featonly_softmax(self, golden_fixture, gnn_model, sim_idx):
        """RED demo via the EXISTING public API (no new kwarg required).

        Pre-fix, RecommendationService.recommendations() had no
        `if self.model_name == "GNN":` branch at all - it unconditionally
        called `self.model.predict(input_data, verbose=0)` regardless of
        model_name. Post-fix, that exact same unconditional-predict code
        path is still reachable today via model_name="MLP" (untouched by the
        fix). So constructing RecommendationService(dfc, gnn_model,
        model_name="MLP") - using only the pre-fix constructor's
        dfc/model/model_name parameters - reproduces byte-for-byte what
        pre-fix code did whenever it was (mis)configured with a GNN model:
        silent features-only prediction, no graph, no exception. This test
        would fail (assertion violated) if that legacy code path ever
        stopped matching the golden features-only softmax, and it directly
        demonstrates - by running real, current RecommendationService code,
        not a stale copy - that the pre-fix behavior differs from the
        graph-served ground truth for the divergent sims.
        """
        conduit_order = list(golden_fixture["conduit_order"])
        features = golden_fixture[f"features_{sim_idx}"]
        expected_featonly = golden_fixture[f"featonly_softmax_{sim_idx}"]
        expected_graph = golden_fixture[f"graph_softmax_{sim_idx}"]

        dfc = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        dfc["Name"] = conduit_order

        # model_name="MLP" forces the legacy unconditional .predict() path
        # even though the model object is the real GNN model - exactly the
        # pre-fix code path, reachable with zero new API surface.
        legacy_path_service = RecommendationService(dfc, gnn_model, model_name="MLP")
        legacy_result = legacy_path_service.recommendations()
        actual = legacy_result[CONFIDENCE_COLUMNS].to_numpy()

        assert np.allclose(actual, expected_featonly, atol=2e-3)
        assert not np.array_equal(actual.argmax(axis=-1), expected_graph.argmax(axis=-1))


@requires_tf
class TestPermutationInvariance:
    """D5 permutation subtest: adjacency must be rebuilt from the *permuted*
    dfc via build_adjacency_from_dfc (the real construction path) - reusing a
    pre-aligned matrix would make this vacuous. The fixture has no
    InletNode/OutletNode, so a small synthetic dfc (like T1) is used here.
    """

    @staticmethod
    def _synthetic_dfc(rng):
        names = ["C1", "C2", "C3", "C4", "C5"]
        dfc = pd.DataFrame(rng.normal(size=(len(names), N_FEATURES)).astype(np.float32), columns=FEATURE_COLUMNS)
        dfc["Name"] = names
        dfc["InletNode"] = ["A", "B", "B", "X", "D"]
        dfc["OutletNode"] = ["B", "C", "D", "Y", "E"]
        return dfc

    def test_permuted_rows_yield_identical_per_conduit_results(self, gnn_model):
        rng = np.random.default_rng(42)
        dfc = self._synthetic_dfc(rng)

        def run(dfc_in):
            raw_adjacency = build_adjacency_from_dfc(dfc_in)
            adjacency = preprocess_adjacency(raw_adjacency, max_hops=GNN_CONFIG["max_hops"])
            service = RecommendationService(dfc_in.copy(), gnn_model, model_name="GNN", adjacency=adjacency)
            return service.recommendations().set_index("Name")

        baseline = run(dfc)

        permutation = [3, 0, 4, 1, 2]
        permuted_dfc = dfc.iloc[permutation].reset_index(drop=True)
        permuted_result = run(permuted_dfc)

        names = list(dfc["Name"])
        baseline_aligned = baseline.loc[names]
        permuted_aligned = permuted_result.loc[names]

        assert np.allclose(
            baseline_aligned[CONFIDENCE_COLUMNS].to_numpy(),
            permuted_aligned[CONFIDENCE_COLUMNS].to_numpy(),
            atol=1e-6,
        )
        assert (baseline_aligned["recommendation"] == permuted_aligned["recommendation"]).all()


@requires_tf
@requires_weights
@pytest.mark.slow
class TestEndToEndGnnServing:
    """T6: full DataManager pipeline on a real .inp file with GNN_ENABLED=true.

    predictor._load_models() reads gnn_enabled() only once and latches via
    _models_loaded, so GNN_ENABLED must be set AND predictor state reset
    BEFORE constructing DataManager. Guarded on WEIGHTS_AVAILABLE: with
    GNN_ENABLED=true the predictor loads the git-ignored serving weights, so
    this must skip (not error) on a fresh CI checkout lacking them.
    """

    @pytest.fixture
    def gnn_enabled_env(self):
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

    def test_full_pipeline_uses_graph_and_differs_from_features_only(self, gnn_enabled_env, tmp_path):
        import shutil
        from sa.core.data import DataManager
        from sa.core.tests import TEST_FILE

        tmp_inp = tmp_path / "test_file.inp"
        shutil.copy2(TEST_FILE, tmp_inp)

        with DataManager(str(tmp_inp)) as dm:
            assert dm.recommendation_service.model_name == "GNN"

            assert "recommendation" in dm.dfc.columns
            assert dm.dfc["recommendation"].notna().all()
            for col in CONFIDENCE_COLUMNS:
                assert col in dm.dfc.columns
            assert not dm.dfc[CONFIDENCE_COLUMNS].isna().any().any()

            row_sums = dm.dfc[CONFIDENCE_COLUMNS].sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-2)

            graph_confidences = dm.dfc[CONFIDENCE_COLUMNS].copy()
            gnn_model = dm.gnn_model
            base_dfc = dm.dfc.drop(columns=CONFIDENCE_COLUMNS + ["recommendation"]).copy()

        assert gnn_model is not None

        # Force the features-only (pre-fix, K1) path by reusing the same GNN
        # model under model_name="MLP": this drives it through
        # self.model.predict(input_data) with no adjacency at all, exactly
        # what pre-fix serving did.
        featonly_service = RecommendationService(base_dfc, gnn_model, model_name="MLP")
        featonly_result = featonly_service.recommendations()
        featonly_confidences = featonly_result[CONFIDENCE_COLUMNS]

        assert not np.allclose(
            graph_confidences.to_numpy(),
            featonly_confidences.to_numpy(),
            atol=1e-9,
        )
