"""Tests for predictor module loading behavior."""

from unittest.mock import MagicMock, patch

import pytest

from sa.core import predictor
from sa.core.predictor import _LazyModelProxy

# Check if TensorFlow is available
import importlib.util

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


class TestPredictorModuleExports:
    """Tests for predictor module exports."""

    def test_classifier_is_exported(self):
        from sa.core.predictor import classifier

        assert classifier is not None

    def test_recommendation_is_exported(self):
        from sa.core.predictor import recommendation

        assert recommendation is not None

    def test_gnn_recommendation_is_exported(self):
        from sa.core import predictor

        assert hasattr(predictor, "gnn_recommendation")

    def test_all_exports_defined(self):
        from sa.core.predictor import __all__

        assert "classifier" in __all__
        assert "recommendation" in __all__
        assert "gnn_recommendation" in __all__
        assert "get_classifier" in __all__
        assert "get_recommendation" in __all__
        assert "get_gnn_recommendation" in __all__


@pytest.fixture
def reset_predictor_state():
    """Reset predictor module global state before and after each test."""
    original_classifier = predictor._classifier
    original_recommendation = predictor._recommendation
    original_gnn_recommendation = predictor._gnn_recommendation
    original_models_loaded = predictor._models_loaded

    predictor._classifier = None
    predictor._recommendation = None
    predictor._gnn_recommendation = None
    predictor._models_loaded = False

    yield

    predictor._classifier = original_classifier
    predictor._recommendation = original_recommendation
    predictor._gnn_recommendation = original_gnn_recommendation
    predictor._models_loaded = original_models_loaded


class TestLoadModels:
    """Tests for _load_models function."""

    def test_early_return_when_models_already_loaded(self, reset_predictor_state):
        """Test that _load_models returns early when models are already loaded."""
        predictor._models_loaded = True
        predictor._classifier = "already_loaded"

        predictor._load_models()

        # If early return works, classifier should still be the value we set
        assert predictor._classifier == "already_loaded"

    def test_sets_models_loaded_flag(self, reset_predictor_state):
        """Test that _load_models sets _models_loaded to True."""
        assert predictor._models_loaded is False

        predictor._load_models()

        assert predictor._models_loaded is True

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
    def test_loads_models_with_tensorflow(self, reset_predictor_state):
        """Test model loading when TensorFlow is available."""
        mock_model = MagicMock()

        with patch("tensorflow.keras.models.load_model", return_value=mock_model):
            with patch.object(predictor, "load_gnn_model_weights", return_value=None, create=True):
                predictor._load_models()

        assert predictor._models_loaded is True

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
    def test_handles_missing_model_files_with_tensorflow(self, reset_predictor_state, monkeypatch, tmp_path):
        """With TensorFlow available but model files absent, loading must degrade gracefully.

        Keras 3's load_model raises ValueError ("File not found: filepath=...") rather than
        FileNotFoundError, so the getters must return None without raising.
        """
        missing_classifier = str(tmp_path / "missing_classifier.keras")
        missing_recommendation = str(tmp_path / "missing_recommendation.keras")
        monkeypatch.setattr(predictor, "catchment_classifier_path", missing_classifier)
        monkeypatch.setattr(predictor, "recommendations_classifier_path", missing_recommendation)

        assert predictor.get_classifier() is None
        assert predictor.get_recommendation() is None
        assert predictor._models_loaded is True

    @pytest.mark.skipif(HAS_TENSORFLOW, reason="Test requires TensorFlow to be unavailable")
    def test_handles_missing_tensorflow(self, reset_predictor_state):
        """Test that _load_models handles missing TensorFlow gracefully."""
        predictor._load_models()

        assert predictor._models_loaded is True
        # Models should be None when TF is not available
        assert predictor._classifier is None
        assert predictor._recommendation is None


class TestGetterFunctions:
    """Tests for model getter functions."""

    def test_get_classifier_returns_classifier(self, reset_predictor_state):
        """Test get_classifier returns the classifier model."""
        mock_classifier = MagicMock()
        predictor._models_loaded = True
        predictor._classifier = mock_classifier

        result = predictor.get_classifier()

        assert result is mock_classifier

    def test_get_classifier_triggers_load(self, reset_predictor_state):
        """Test get_classifier triggers _load_models if not loaded."""
        with patch.object(predictor, "_load_models") as mock_load:
            predictor.get_classifier()
            mock_load.assert_called_once()

    def test_get_classifier_returns_none_when_not_available(self, reset_predictor_state):
        """Test get_classifier returns None when model is not available."""
        predictor._models_loaded = True
        predictor._classifier = None

        result = predictor.get_classifier()

        assert result is None

    def test_get_recommendation_returns_recommendation(self, reset_predictor_state):
        """Test get_recommendation returns the recommendation model."""
        mock_recommendation = MagicMock()
        predictor._models_loaded = True
        predictor._recommendation = mock_recommendation

        result = predictor.get_recommendation()

        assert result is mock_recommendation

    def test_get_recommendation_triggers_load(self, reset_predictor_state):
        """Test get_recommendation triggers _load_models if not loaded."""
        with patch.object(predictor, "_load_models") as mock_load:
            predictor.get_recommendation()
            mock_load.assert_called_once()

    def test_get_recommendation_returns_none_when_not_available(self, reset_predictor_state):
        """Test get_recommendation returns None when model is not available."""
        predictor._models_loaded = True
        predictor._recommendation = None

        result = predictor.get_recommendation()

        assert result is None

    def test_get_gnn_recommendation_returns_gnn_model(self, reset_predictor_state):
        """Test get_gnn_recommendation returns the GNN model."""
        mock_gnn = MagicMock()
        predictor._models_loaded = True
        predictor._gnn_recommendation = mock_gnn

        result = predictor.get_gnn_recommendation()

        assert result is mock_gnn

    def test_get_gnn_recommendation_triggers_load(self, reset_predictor_state):
        """Test get_gnn_recommendation triggers _load_models if not loaded."""
        with patch.object(predictor, "_load_models") as mock_load:
            predictor.get_gnn_recommendation()
            mock_load.assert_called_once()


class TestLazyModelProxy:
    """Tests for _LazyModelProxy class."""

    def test_init_stores_getter(self):
        """Test that __init__ stores the getter function."""
        getter = MagicMock()
        proxy = _LazyModelProxy(getter)

        assert proxy._getter is getter

    def test_getattr_delegates_to_model(self):
        """Test that __getattr__ delegates attribute access to the model."""
        mock_model = MagicMock()
        mock_model.some_attribute = "test_value"
        getter = MagicMock(return_value=mock_model)

        proxy = _LazyModelProxy(getter)
        result = proxy.some_attribute

        getter.assert_called_once()
        assert result == "test_value"

    def test_getattr_raises_when_model_is_none(self):
        """Test that __getattr__ raises RuntimeError when model is None."""
        getter = MagicMock(return_value=None)

        proxy = _LazyModelProxy(getter)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = proxy.some_attribute

    def test_call_delegates_to_model(self):
        """Test that __call__ delegates calls to the model."""
        mock_model = MagicMock(return_value="prediction_result")
        getter = MagicMock(return_value=mock_model)

        proxy = _LazyModelProxy(getter)
        result = proxy("input_data", kwarg="value")

        getter.assert_called_once()
        mock_model.assert_called_once_with("input_data", kwarg="value")
        assert result == "prediction_result"

    def test_call_raises_when_model_is_none(self):
        """Test that __call__ raises RuntimeError when model is None."""
        getter = MagicMock(return_value=None)

        proxy = _LazyModelProxy(getter)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            proxy("input_data")

    def test_call_with_no_arguments(self):
        """Test that __call__ works with no arguments."""
        mock_model = MagicMock(return_value="result")
        getter = MagicMock(return_value=mock_model)

        proxy = _LazyModelProxy(getter)
        result = proxy()

        mock_model.assert_called_once_with()
        assert result == "result"

    def test_call_with_multiple_positional_args(self):
        """Test that __call__ works with multiple positional arguments."""
        mock_model = MagicMock(return_value="result")
        getter = MagicMock(return_value=mock_model)

        proxy = _LazyModelProxy(getter)
        result = proxy("arg1", "arg2", "arg3")

        mock_model.assert_called_once_with("arg1", "arg2", "arg3")
        assert result == "result"

    def test_getattr_method_call(self):
        """Test that __getattr__ allows method calls on the model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.3, 0.2]
        getter = MagicMock(return_value=mock_model)

        proxy = _LazyModelProxy(getter)
        result = proxy.predict([1, 2, 3])

        mock_model.predict.assert_called_once_with([1, 2, 3])
        assert result == [0.5, 0.3, 0.2]
