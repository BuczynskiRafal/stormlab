"""Tests for GNN serving behavior (T3, T4, T8, T11).

TensorFlow-dependent (weight loading / model calls); guarded with the same
HAS_TF skip pattern as app/sa/core/tests/test_predictor.py.
"""

import importlib.util
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sa.core.conduits import RecommendationService
from sa.core.constants import FEATURE_COLUMNS
from sa.core.gnn import build_adjacency_from_dfc, preprocess_adjacency
from sa.core.predictor import GNN_CONFIG, gnn_enabled

HAS_TF = importlib.util.find_spec("tensorflow") is not None
requires_tf = pytest.mark.skipif(not HAS_TF, reason="requires tensorflow")

RECOMMENDATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recommendations")
WEIGHTS_PATH = os.path.join(RECOMMENDATIONS_DIR, GNN_CONFIG["weights"])
LEGACY_WEIGHTS_PATH = os.path.join(RECOMMENDATIONS_DIR, "graphsage_4hop_model.weights.h5")
LEGACY_WEIGHTS_AVAILABLE = os.path.exists(LEGACY_WEIGHTS_PATH)
# The serving weights are git-ignored (.gitignore) and are not provisioned in
# CI, so any test that actually loads them must SKIP (not error) when the file
# is absent. Weight provisioning is a separate decision, out of scope here.
WEIGHTS_AVAILABLE = os.path.exists(WEIGHTS_PATH)
requires_weights = pytest.mark.skipif(not WEIGHTS_AVAILABLE, reason="GNN model weights not available")
N_FEATURES = len(FEATURE_COLUMNS)
N_CLASSES = 9


def test_gnn_enablement_requires_runnable_parity_coverage():
    """Enable-gate (plan1.md D10 / Etap 3.2): fail loudly, never silently green.

    T3/T4 (K1 regression), T5 (train/serve parity) and T6 all load the
    git-ignored serving weights and therefore SKIP when the weights are absent.
    A CI job that enables GNN without provisioning those weights would report
    green while every gating test skipped -- 'green' would NOT prove the
    graph-served path. So: whenever GNN is actually enabled, require the weights
    to be present (which forces the gating tests to run instead of skip).

    Stays green today because GNN_ENABLED defaults off; only bites when the flag
    is flipped (Etap 3.3) in an environment that never runs the parity tests.
    """
    if not gnn_enabled():
        pytest.skip("GNN_ENABLED is off; enable-gate not applicable")
    assert WEIGHTS_AVAILABLE, (
        f"GNN_ENABLED=true but serving weights are missing at {WEIGHTS_PATH}. "
        "The K1/parity regression tests (T3-T6) skip without weights, so a green "
        "run would not prove the graph-served path. Provision the weights before "
        "enabling GNN (plan1.md D10 / Etap 3.2)."
    )


def _build_and_load_legacy_model():
    """Build a serving GraphSAGEModel and load the real 4-hop weights.

    Skips (rather than errors) when the weights file is absent so the
    dependent tests are portable to a fresh CI checkout without the weights.
    """
    if not LEGACY_WEIGHTS_AVAILABLE:
        pytest.skip("Legacy GNN model weights not available")

    import tensorflow as tf
    from sa.core.graph_constructor import GraphSAGEModel

    model = GraphSAGEModel(n_features=N_FEATURES, n_classes=N_CLASSES)
    model([tf.zeros((1, N_FEATURES)), tf.zeros((1, 1))], training=False)
    model.load_weights(LEGACY_WEIGHTS_PATH)
    return model


@requires_tf
class TestGraphSAGEWeightCompat:
    """T3: serving GraphSAGEModel(33, 9) must load graphsage_4hop_model.weights.h5.

    The weight name/shape list is pinned so future architecture changes to
    GraphSAGEModel are caught as an explicit, deliberate diff rather than a
    silent shape-mismatch at load time.
    """

    def test_loads_4hop_weights_without_error(self):
        _build_and_load_legacy_model()  # must not raise

    def test_weight_name_and_shape_list(self):
        model = _build_and_load_legacy_model()
        actual = [(w.name, tuple(w.shape)) for w in model.weights]

        expected = [
            ("W_self", (N_FEATURES, 64)),
            ("W_neigh", (N_FEATURES, 64)),
            ("bias", (64,)),
            ("gamma", (64,)),
            ("beta", (64,)),
            ("moving_mean", (64,)),
            ("moving_variance", (64,)),
            ("W_self", (64, 64)),
            ("W_neigh", (64, 64)),
            ("bias", (64,)),
            ("gamma", (64,)),
            ("beta", (64,)),
            ("moving_mean", (64,)),
            ("moving_variance", (64,)),
            ("kernel", (64, N_CLASSES)),
            ("bias", (N_CLASSES,)),
        ]

        assert actual == expected


class _FakeTensor:
    """Minimal stand-in for a tf.Tensor exposing only .numpy(), so the K1
    regression tests don't require a real model forward pass."""

    def __init__(self, array):
        self._array = array

    def numpy(self):
        return self._array


class _SpyGnnModel:
    """Records how it was invoked: direct __call__([features, adjacency])
    (correct, D4) vs. .predict(...) (the Keras list-coercion trap, K1)."""

    def __init__(self, n_classes=N_CLASSES):
        self.call_args = None
        self.predict_called = False
        self.n_classes = n_classes

    def __call__(self, inputs, training=False):
        self.call_args = inputs
        assert isinstance(inputs, list) and len(inputs) == 2, "GNN serving must call model([features, adjacency])"
        features, _adjacency = inputs
        n_rows = np.asarray(features).shape[0]
        result = np.zeros((n_rows, self.n_classes), dtype=np.float32)
        result[:, 0] = 1.0
        return _FakeTensor(result)

    def predict(self, *args, **kwargs):
        self.predict_called = True
        raise AssertionError(
            "GNN serving must never call model.predict(...) - it coerces list inputs to a tuple, dropping adjacency"
        )


class TestGnnRequiresAdjacency:
    """T4 (K1 regression guard): GNN serving without adjacency must raise,
    never silently fall back to features-only prediction.

    These two tests deliberately construct RecommendationService using only
    the dfc/model/model_name arguments (no `adjacency=` kwarg at all,
    relying on the post-fix default of None) rather than passing
    ``adjacency=None`` explicitly. The pre-fix ``__init__`` signature was
    ``def __init__(self, dfc, model, model_name="MLP")`` - it has no
    ``adjacency`` parameter whatsoever, so passing ``adjacency=None``
    explicitly would fail on pre-fix code with
    ``TypeError: unexpected keyword argument 'adjacency'`` - a red for the
    *wrong* reason (an ImportError-style signature mismatch), which the test
    contract explicitly rules out. Omitting the kwarg entirely is valid
    Python on both signatures, so running this exact call against pre-fix
    code exercises the real defect: it silently falls through to
    ``self.model.predict(input_data)`` (features-only, no graph, no
    exception) instead of raising - i.e. these tests are genuinely red on
    pre-fix code, for the right reason.
    """

    def test_raises_value_error_when_adjacency_is_none(self):
        dfc = pd.DataFrame(np.zeros((3, N_FEATURES)), columns=FEATURE_COLUMNS)
        service = RecommendationService(dfc, _SpyGnnModel(), model_name="GNN")

        with pytest.raises(ValueError, match="adjacency"):
            service.recommendations()

    def test_does_not_call_model_when_adjacency_is_none(self):
        """The K1 red demo: refusing to serve must happen *before* any model
        call, not compute features-only and then discard the result."""
        dfc = pd.DataFrame(np.zeros((3, N_FEATURES)), columns=FEATURE_COLUMNS)
        spy = _SpyGnnModel()
        service = RecommendationService(dfc, spy, model_name="GNN")

        with pytest.raises(ValueError):
            service.recommendations()

        assert spy.call_args is None
        assert spy.predict_called is False

    def test_uses_direct_call_not_predict_when_adjacency_provided(self):
        names = ["C1", "C2", "C3"]
        dfc = pd.DataFrame(
            np.random.default_rng(1).normal(size=(3, N_FEATURES)).astype(np.float32),
            columns=FEATURE_COLUMNS,
        )
        dfc["Name"] = names
        dfc["InletNode"] = ["A", "B", "C"]
        dfc["OutletNode"] = ["B", "C", "D"]

        adjacency = preprocess_adjacency(build_adjacency_from_dfc(dfc), max_hops=GNN_CONFIG["max_hops"])
        spy = _SpyGnnModel()
        service = RecommendationService(dfc, spy, model_name="GNN", adjacency=adjacency)
        service.recommendations()

        assert spy.call_args is not None
        assert spy.predict_called is False


@requires_tf
class TestNaNRobustness:
    """T8: NaN feature values must not leak into GNN predictions/confidences
    (np.nan_to_num guard in RecommendationService.recommendations)."""

    def test_predictions_have_no_nan_with_nan_features(self):
        model = _build_and_load_legacy_model()

        rng = np.random.default_rng(7)
        names = ["C1", "C2", "C3"]
        dfc = pd.DataFrame(rng.normal(size=(3, N_FEATURES)).astype(np.float32), columns=FEATURE_COLUMNS)
        dfc.iloc[1, 2] = np.nan
        dfc["Name"] = names
        dfc["InletNode"] = ["A", "B", "C"]
        dfc["OutletNode"] = ["B", "C", "D"]

        adjacency = preprocess_adjacency(build_adjacency_from_dfc(dfc), max_hops=GNN_CONFIG["max_hops"])
        service = RecommendationService(dfc, model, model_name="GNN", adjacency=adjacency)
        result = service.recommendations()

        conf_cols = [c for c in result.columns if c.startswith("confidence_")]
        assert not result[conf_cols].isna().any().any()
        assert result["recommendation"].notna().all()


@requires_tf
@requires_weights
class TestLoadGnnModelWeightsConfig:
    """T11 (D3): load_gnn_model_weights must respect GNN_CONFIG strictly; a
    planted graphsage_model.keras must not bypass the configured weights.

    Guarded on WEIGHTS_AVAILABLE: the whole recommendations/ directory is
    git-ignored, so on a fresh CI checkout it does not exist and planting a
    sentinel .keras there (or resolving WEIGHTS_PATH) would error rather than
    skip. The guard keeps this a clean skip when weights are unprovisioned."""

    def test_planted_keras_file_does_not_bypass_config(self):
        import tensorflow as tf
        from sa.core import graph_constructor

        planted_path = os.path.join(RECOMMENDATIONS_DIR, "graphsage_model.keras")
        already_existed = os.path.exists(planted_path)
        if not already_existed:
            with open(planted_path, "wb") as fh:
                fh.write(b"not a real keras archive - sentinel planted by test_gnn_serving.py T11")

        try:
            with patch.object(tf.keras.models, "load_model") as mock_load_model:
                result = graph_constructor.load_gnn_model_weights()

                assert result is not None
                mock_load_model.assert_not_called()
        finally:
            if not already_existed and os.path.exists(planted_path):
                os.remove(planted_path)

    def test_default_weights_path_derives_from_gnn_config(self):
        from sa.core import graph_constructor

        result = graph_constructor.load_gnn_model_weights()

        assert result is not None
        assert hasattr(result, "spec")
        assert result.spec.slug() == "legacy_weighted_mean_k2_up_h64_d0.2_wd0_ln_in-constant_safe_v4_physical_soft_masked_lam0"
        assert result.feature_contract.name == "v4_physical"
        assert result.imputation_plan is not None

    def test_b4_winner_weights_path_loads_b4_spec(self):
        from sa.core import graph_constructor

        b4_path = os.path.join(RECOMMENDATIONS_DIR, "graphsage_b4_winner_v1.weights.h5")
        if not os.path.exists(b4_path):
            pytest.skip("B4 weights not available")

        result = graph_constructor.load_gnn_model_weights(b4_path)
        assert result is not None
        assert hasattr(result, "spec")
        assert result.spec.slug() == "support_pool_k4_up_h64_d0.5_wd0.0001_ln_in-constant_safe"
        assert result.feature_contract.name == "v2"

    def test_legacy_weights_path_loads_graphsage_model(self):
        from sa.core import graph_constructor
        from sa.core.graph_constructor import GraphSAGEModel

        if not LEGACY_WEIGHTS_AVAILABLE:
            pytest.skip("Legacy weights not available")

        result = graph_constructor.load_gnn_model_weights(LEGACY_WEIGHTS_PATH)
        assert isinstance(result, GraphSAGEModel)
