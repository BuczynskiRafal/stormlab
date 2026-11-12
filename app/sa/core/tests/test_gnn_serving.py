"""Tests for GNN serving behavior.

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
WEIGHTS_PATHS = [os.path.join(RECOMMENDATIONS_DIR, name) for name in GNN_CONFIG["weights"]]
# The serving weights ship with the repository; tests that load them fail, not skip, when the file is missing.
WEIGHTS_AVAILABLE = all(os.path.exists(path) for path in WEIGHTS_PATHS)
N_FEATURES = len(FEATURE_COLUMNS)
N_CLASSES = 9


def test_gnn_enablement_requires_runnable_parity_coverage():
    """Whenever GNN serving is enabled, the configured serving weights must exist.

    Skips only when GNN_ENABLED is explicitly switched off in the environment.
    """
    if not gnn_enabled():
        pytest.skip("GNN_ENABLED is off; enable-gate not applicable")
    assert WEIGHTS_AVAILABLE, f"GNN_ENABLED=true but serving weights are missing among {WEIGHTS_PATHS}."


class _FakeTensor:
    """Minimal stand-in for a tf.Tensor exposing only .numpy(), so the serving
    tests don't require a real model forward pass."""

    def __init__(self, array):
        self._array = array

    def numpy(self):
        return self._array


class _SpyGnnModel:
    """Records how it was invoked: direct __call__([features, adjacency])
    (correct) vs. .predict(...) (the Keras list-coercion trap, which drops the
    adjacency)."""

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
    """GNN serving without adjacency must raise, never silently fall back to
    features-only prediction.

    The service is constructed without an `adjacency=` argument, relying on
    its default of None.
    """

    def test_raises_value_error_when_adjacency_is_none(self):
        dfc = pd.DataFrame(np.zeros((3, N_FEATURES)), columns=FEATURE_COLUMNS)
        service = RecommendationService(dfc, _SpyGnnModel(), model_name="GNN")

        with pytest.raises(ValueError, match="adjacency"):
            service.recommendations()

    def test_does_not_call_model_when_adjacency_is_none(self):
        """Refusing to serve must happen *before* any model call, not compute
        features-only and then discard the result."""
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
class TestLoadGnnModelWeightsConfig:
    """load_gnn_model_weights must respect GNN_CONFIG strictly; a stray
    graphsage_model.keras must not bypass the configured weights.

    The configured weights ship with the repository, so these tests run on every
    checkout."""

    def test_stray_keras_file_does_not_bypass_config(self):
        import tensorflow as tf
        from sa.core import graph_constructor

        stray_path = os.path.join(RECOMMENDATIONS_DIR, "graphsage_model.keras")
        already_existed = os.path.exists(stray_path)
        if not already_existed:
            with open(stray_path, "wb") as fh:
                fh.write(b"not a real keras archive - sentinel planted by test_gnn_serving.py")

        try:
            with patch.object(tf.keras.models, "load_model") as mock_load_model:
                result = graph_constructor.load_gnn_model_weights()

                assert result is not None
                mock_load_model.assert_not_called()
        finally:
            if not already_existed and os.path.exists(stray_path):
                os.remove(stray_path)

    def test_default_weights_path_derives_from_gnn_config(self):
        from sa.core import graph_constructor

        result = graph_constructor.load_gnn_model_weights()

        assert result is not None
        assert hasattr(result, "spec")
        assert result.spec.slug() == "legacy_weighted_mean_k2_up_h64_d0.2_wd0_ln_in-constant_safe_v4_physical_soft_masked_lam0"
        assert result.feature_contract.name == "v4_physical"
        assert result.imputation_plan is not None
