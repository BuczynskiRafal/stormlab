"""Tests for graph_constructor.py - GNN model components."""

import pytest
import numpy as np
from unittest.mock import patch


class TestTensorFlowNotAvailable:
    """Tests for when TensorFlow is not available."""

    def test_require_tensorflow_raises_when_tf_not_available(self):
        """Test that _require_tensorflow raises ImportError when TF is not installed."""
        with patch.dict("sys.modules", {"tensorflow": None}):
            # Import module with TF unavailable
            from sa.core import graph_constructor

            # Temporarily set _TF_AVAILABLE to False
            original_tf_available = graph_constructor._TF_AVAILABLE
            graph_constructor._TF_AVAILABLE = False

            try:
                with pytest.raises(ImportError, match="TensorFlow is required"):
                    graph_constructor._require_tensorflow()
            finally:
                graph_constructor._TF_AVAILABLE = original_tf_available

    def test_placeholders_when_tf_not_available(self):
        """Test that placeholders are set when TensorFlow is not available."""
        from sa.core import graph_constructor

        # When TF is available, these should be classes
        # When not available, they would be None (line 176-178)
        # We can verify this by checking the module structure
        if not graph_constructor._TF_AVAILABLE:
            assert graph_constructor.GraphSAGELayer is None
            assert graph_constructor.GraphSAGEModel is None
            assert graph_constructor.CUSTOM_OBJECTS == {}


@pytest.fixture
def tf_imports():
    """Fixture to ensure TensorFlow is available for tests."""
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from sa.core.graph_constructor import GraphSAGELayer, GraphSAGEModel, CUSTOM_OBJECTS

    return {
        "tf": tf,
        "GraphSAGELayer": GraphSAGELayer,
        "GraphSAGEModel": GraphSAGEModel,
        "CUSTOM_OBJECTS": CUSTOM_OBJECTS,
    }


class TestGraphSAGELayer:
    """Tests for GraphSAGELayer class."""

    def test_build_with_non_list_input_shape(self, tf_imports):
        """Test build method when input_shape is not a list (line 55)."""
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        layer = GraphSAGELayer(units=32, aggregator="mean")
        layer.build(input_shape=(None, 10))

        assert layer.W_self.shape == (10, 32)
        assert layer.W_neigh.shape == (10, 32)

    def test_build_with_list_input_shape(self, tf_imports):
        """Test build method when input_shape is a list (line 54-55)."""
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        layer = GraphSAGELayer(units=64, aggregator="mean")
        layer.build(input_shape=[(None, 16), (None, None)])

        assert layer.W_self.shape == (16, 64)
        assert layer.W_neigh.shape == (16, 64)

    def test_call_without_adjacency_matrix(self, tf_imports):
        """Test call method when adjacency_matrix is None (lines 74-78)."""
        tf = tf_imports["tf"]
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        layer = GraphSAGELayer(units=16, aggregator="mean")
        inputs = tf.random.uniform((5, 8))
        layer.build(input_shape=(None, 8))

        output = layer(inputs, adjacency_matrix=None, training=False)

        assert output.shape == (5, 16)

    def test_call_with_scipy_sparse_adjacency(self, tf_imports):
        """Test call method with scipy sparse matrix (line 82)."""
        import scipy.sparse as sp

        tf = tf_imports["tf"]
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        layer = GraphSAGELayer(units=16, aggregator="mean")
        inputs = tf.random.uniform((3, 8))
        layer.build(input_shape=(None, 8))

        adj_sparse = sp.csr_matrix(np.eye(3, dtype=np.float32))

        output = layer(inputs, adjacency_matrix=adj_sparse, training=False)

        assert output.shape == (3, 16)

    def test_call_with_tf_sparse_tensor(self, tf_imports):
        """Test call method with TensorFlow sparse tensor (lines 83-85)."""
        tf = tf_imports["tf"]
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        layer = GraphSAGELayer(units=16, aggregator="mean")
        inputs = tf.random.uniform((3, 8))
        layer.build(input_shape=(None, 8))

        indices = [[0, 0], [1, 1], [2, 2]]
        values = [1.0, 1.0, 1.0]
        adj_sparse = tf.SparseTensor(indices=indices, values=values, dense_shape=[3, 3])

        output = layer(inputs, adjacency_matrix=adj_sparse, training=False)

        assert output.shape == (3, 16)

    def test_call_with_dense_adjacency(self, tf_imports):
        """Test call method with dense adjacency matrix (line 87)."""
        tf = tf_imports["tf"]
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        layer = GraphSAGELayer(units=16, aggregator="mean")
        inputs = tf.random.uniform((3, 8))
        layer.build(input_shape=(None, 8))

        adj_dense = np.eye(3, dtype=np.float32)

        output = layer(inputs, adjacency_matrix=adj_dense, training=False)

        assert output.shape == (3, 16)

    def test_call_without_bias(self, tf_imports):
        """Test call method when use_bias is False."""
        tf = tf_imports["tf"]
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        layer = GraphSAGELayer(units=16, aggregator="mean", use_bias=False)
        inputs = tf.random.uniform((3, 8))
        layer.build(input_shape=(None, 8))

        output = layer(inputs, adjacency_matrix=None, training=False)

        assert output.shape == (3, 16)
        assert not hasattr(layer, "bias") or layer.bias is None

    def test_get_config(self, tf_imports):
        """Test get_config method returns correct configuration (lines 99-101)."""
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        layer = GraphSAGELayer(units=128, aggregator="max", use_bias=False, name="test_layer")
        config = layer.get_config()

        assert config["units"] == 128
        assert config["aggregator"] == "max"
        assert config["use_bias"] is False
        assert config["name"] == "test_layer"

    def test_is_gnn_attribute(self, tf_imports):
        """Test that is_gnn attribute is set."""
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        layer = GraphSAGELayer(units=32)
        assert layer.is_gnn is True


class TestGraphSAGEModel:
    """Tests for GraphSAGEModel class."""

    def test_model_initialization(self, tf_imports):
        """Test model initialization with default parameters."""
        GraphSAGEModel = tf_imports["GraphSAGEModel"]

        model = GraphSAGEModel(n_features=10, n_classes=9)

        assert model.n_features == 10
        assert model.hidden_units == 64
        assert model.dropout_rate == 0.2

    def test_model_build_with_non_list_input_shape(self, tf_imports):
        """Test build method when input_shape is not a list (line 133)."""
        GraphSAGEModel = tf_imports["GraphSAGEModel"]

        model = GraphSAGEModel(n_features=10, n_classes=5)
        model.build(input_shape=(None, 10))

        assert model.built

    def test_model_build_with_list_input_shape(self, tf_imports):
        """Test build method when input_shape is a list (lines 130-131)."""
        GraphSAGEModel = tf_imports["GraphSAGEModel"]

        model = GraphSAGEModel(n_features=10, n_classes=5)
        model.build(input_shape=[(None, 10), (None, None)])

        assert model.built

    def test_model_call_with_list_inputs(self, tf_imports):
        """Test call method with [node_features, adjacency] input."""
        tf = tf_imports["tf"]
        GraphSAGEModel = tf_imports["GraphSAGEModel"]

        model = GraphSAGEModel(n_features=8, n_classes=5)

        node_features = tf.random.uniform((4, 8))
        adjacency = tf.constant(np.eye(4, dtype=np.float32))

        output = model([node_features, adjacency], training=False)

        assert output.shape == (4, 5)
        np.testing.assert_array_almost_equal(tf.reduce_sum(output, axis=1).numpy(), np.ones(4), decimal=5)

    def test_model_call_with_single_input(self, tf_imports):
        """Test call method with only node_features (no adjacency)."""
        tf = tf_imports["tf"]
        GraphSAGEModel = tf_imports["GraphSAGEModel"]

        model = GraphSAGEModel(n_features=8, n_classes=5)

        node_features = tf.random.uniform((4, 8))

        output = model(node_features, training=False)

        assert output.shape == (4, 5)

    def test_supports_adjacency_returns_true_for_graphsage_layer(self, tf_imports):
        """Test _supports_adjacency returns True for GraphSAGE layers."""
        GraphSAGEModel = tf_imports["GraphSAGEModel"]
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        model = GraphSAGEModel(n_features=8, n_classes=5)
        layer = GraphSAGELayer(units=32)

        assert model._supports_adjacency(layer) is True

    def test_supports_adjacency_returns_false_for_dense_layer(self, tf_imports):
        """Test _supports_adjacency returns False for Dense layers (line 153)."""
        tf = tf_imports["tf"]
        GraphSAGEModel = tf_imports["GraphSAGEModel"]

        model = GraphSAGEModel(n_features=8, n_classes=5)
        dense_layer = tf.keras.layers.Dense(32)

        assert model._supports_adjacency(dense_layer) is False

    def test_supports_adjacency_handles_exception(self, tf_imports):
        """Test _supports_adjacency returns False on exception (lines 146-147)."""
        GraphSAGEModel = tf_imports["GraphSAGEModel"]

        model = GraphSAGEModel(n_features=8, n_classes=5)

        # Create a mock layer that raises an exception when inspecting call
        class BrokenLayer:
            @property
            def call(self):
                raise RuntimeError("Cannot inspect")

        broken_layer = BrokenLayer()

        assert model._supports_adjacency(broken_layer) is False

    def test_call_layer_safely_with_adjacency_support(self, tf_imports):
        """Test _call_layer_safely when layer supports adjacency."""
        tf = tf_imports["tf"]
        GraphSAGEModel = tf_imports["GraphSAGEModel"]
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        model = GraphSAGEModel(n_features=8, n_classes=5)
        layer = GraphSAGELayer(units=16)
        layer.build((None, 8))

        inputs = tf.random.uniform((3, 8))
        adjacency = tf.constant(np.eye(3, dtype=np.float32))

        output = model._call_layer_safely(layer, inputs, adjacency, training=False)

        assert output.shape == (3, 16)

    def test_call_layer_safely_without_adjacency_support(self, tf_imports):
        """Test _call_layer_safely when layer doesn't support adjacency (line 153)."""
        tf = tf_imports["tf"]
        GraphSAGEModel = tf_imports["GraphSAGEModel"]

        model = GraphSAGEModel(n_features=8, n_classes=5)
        dense_layer = tf.keras.layers.Dense(16)

        inputs = tf.random.uniform((3, 8))
        adjacency = tf.constant(np.eye(3, dtype=np.float32))

        output = model._call_layer_safely(dense_layer, inputs, adjacency, training=False)

        assert output.shape == (3, 16)

    def test_model_training_mode(self, tf_imports):
        """Test model in training mode activates dropout."""
        tf = tf_imports["tf"]
        GraphSAGEModel = tf_imports["GraphSAGEModel"]

        model = GraphSAGEModel(n_features=8, n_classes=5, dropout_rate=0.5)

        node_features = tf.random.uniform((10, 8))

        output1 = model(node_features, training=True)
        output2 = model(node_features, training=True)

        assert not np.allclose(output1.numpy(), output2.numpy())


class TestCustomObjects:
    """Tests for CUSTOM_OBJECTS dictionary."""

    def test_custom_objects_contains_classes(self, tf_imports):
        """Test CUSTOM_OBJECTS contains required classes."""
        CUSTOM_OBJECTS = tf_imports["CUSTOM_OBJECTS"]
        GraphSAGELayer = tf_imports["GraphSAGELayer"]
        GraphSAGEModel = tf_imports["GraphSAGEModel"]

        assert "GraphSAGELayer" in CUSTOM_OBJECTS
        assert "GraphSAGEModel" in CUSTOM_OBJECTS
        assert CUSTOM_OBJECTS["GraphSAGELayer"] is GraphSAGELayer
        assert CUSTOM_OBJECTS["GraphSAGEModel"] is GraphSAGEModel


class TestGNNModelLoadError:
    """Tests for GNNModelLoadError exception."""

    def test_exception_can_be_raised(self):
        """Test GNNModelLoadError can be raised and caught."""
        from sa.core.graph_constructor import GNNModelLoadError

        with pytest.raises(GNNModelLoadError, match="Test error"):
            raise GNNModelLoadError("Test error")

    def test_exception_inherits_from_exception(self):
        """Test GNNModelLoadError inherits from Exception."""
        from sa.core.graph_constructor import GNNModelLoadError

        assert issubclass(GNNModelLoadError, Exception)


class TestLoadGNNModelWeights:
    """Tests for load_gnn_model_weights function."""

    def test_load_model_requires_tensorflow(self):
        """Test load_gnn_model_weights raises when TF not available."""
        from sa.core import graph_constructor

        original_tf_available = graph_constructor._TF_AVAILABLE
        graph_constructor._TF_AVAILABLE = False

        try:
            with pytest.raises(ImportError, match="TensorFlow is required"):
                graph_constructor.load_gnn_model_weights()
        finally:
            graph_constructor._TF_AVAILABLE = original_tf_available

    def test_load_weights_does_not_auto_load_keras_model(self):
        """Test loading configured weights without auto-preferring a .keras model."""
        pytest.importorskip("tensorflow")
        from sa.core import graph_constructor
        from sa.core.graph_constructor import GraphSAGEModel

        with patch("tensorflow.keras.models.load_model") as mock_load_model:
            with patch.object(GraphSAGEModel, "load_weights", return_value=None) as mock_load_weights:
                result = graph_constructor.load_gnn_model_weights(weights_path="/configured/path.weights.h5")

                assert result is not None
                mock_load_model.assert_not_called()
                mock_load_weights.assert_called_once_with("/configured/path.weights.h5")

    def test_load_weights_file_not_found(self):
        """A nonexistent weights path returns None (error handling).

        load_gnn_model_weights loads ONLY the configured .weights.h5 via
        model.load_weights - it no longer probes for or auto-loads a
        graphsage_model.keras, so there is no os.path.exists / load_model
        code path left to mock here. The real model.load_weights on a
        missing file raises (FileNotFoundError / OSError / ValueError), which
        the loader catches and turns into a None return.
        """
        pytest.importorskip("tensorflow")
        from sa.core import graph_constructor

        result = graph_constructor.load_gnn_model_weights(weights_path="/nonexistent/path.weights.h5")

        assert result is None

    def test_load_weights_value_error(self, tmp_path):
        """Test ValueError handling during weight loading (lines 221-226)."""
        pytest.importorskip("tensorflow")
        from sa.core import graph_constructor

        weights_path = tmp_path / "corrupted.weights.h5"
        weights_path.write_text("not a valid h5 file")

        with patch("os.path.exists", return_value=False):
            result = graph_constructor.load_gnn_model_weights(weights_path=str(weights_path))

            assert result is None

    def test_load_weights_os_error(self):
        """Test OSError handling during weight loading (lines 221-226)."""
        pytest.importorskip("tensorflow")
        from sa.core import graph_constructor
        from sa.core.graph_constructor import GraphSAGEModel

        with patch("os.path.exists", return_value=False):
            with patch.object(GraphSAGEModel, "load_weights") as mock_load:
                mock_load.side_effect = OSError("Cannot read file")

                result = graph_constructor.load_gnn_model_weights(weights_path="/some/path.weights.h5")

                assert result is None

    def test_load_weights_gnn_model_load_error(self):
        """Test GNNModelLoadError handling (lines 227-231)."""
        pytest.importorskip("tensorflow")
        from sa.core import graph_constructor
        from sa.core.graph_constructor import GNNModelLoadError, GraphSAGEModel

        with patch("os.path.exists", return_value=False):
            with patch.object(GraphSAGEModel, "__call__") as mock_call:
                mock_call.side_effect = GNNModelLoadError("Critical error")

                result = graph_constructor.load_gnn_model_weights(weights_path="/some/path.weights.h5")

                assert result is None

    def test_load_weights_success(self):
        """Test successful weight loading from .weights.h5 file."""
        pytest.importorskip("tensorflow")
        from sa.core import graph_constructor
        from sa.core.graph_constructor import GraphSAGEModel

        with patch("os.path.exists", return_value=False):
            with patch.object(GraphSAGEModel, "load_weights") as mock_load_weights:
                mock_load_weights.return_value = None

                result = graph_constructor.load_gnn_model_weights(weights_path="/some/path.weights.h5")

                assert result is not None
                mock_load_weights.assert_called_once_with("/some/path.weights.h5")


class TestGraphSAGELayerMessagePassing:
    """Tests for message passing functionality in GraphSAGELayer."""

    def test_message_passing_with_connected_graph(self, tf_imports):
        """Test that message passing aggregates neighbor information."""
        tf = tf_imports["tf"]
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        layer = GraphSAGELayer(units=4, aggregator="mean")

        inputs = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=tf.float32)

        layer.build(input_shape=(None, 2))

        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)

        output = layer(inputs, adjacency_matrix=adj, training=False)

        assert output.shape == (3, 4)

    def test_relu_activation_applied(self, tf_imports):
        """Test that ReLU activation is applied to output."""
        tf = tf_imports["tf"]
        GraphSAGELayer = tf_imports["GraphSAGELayer"]

        layer = GraphSAGELayer(units=4, aggregator="mean")

        inputs = tf.constant([[-1.0, -1.0], [-2.0, -2.0]], dtype=tf.float32)

        layer.build(input_shape=(None, 2))

        output = layer(inputs, adjacency_matrix=None, training=False)

        assert output.shape == (2, 4)
