"""
Graph construction utilities for SWMM conduit networks.
Builds graphs where conduits are nodes, not junctions.
"""
# pyright: reportOptionalMemberAccess=false, reportOptionalCall=false

import os
import sys
import logging
from typing import Optional, Any

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

logger = logging.getLogger(__name__)

# Check if TensorFlow is available
_TF_AVAILABLE = False
tf = None
Dense = None
BatchNormalization = None
Dropout = None

try:
    import tensorflow as _tf

    tf = _tf
    from tensorflow.keras.layers import Dense as _Dense, BatchNormalization as _BatchNormalization, Dropout as _Dropout

    Dense = _Dense
    BatchNormalization = _BatchNormalization
    Dropout = _Dropout
    _TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available - GNN models will not be available")


def _require_tensorflow():
    """Raise ImportError if TensorFlow is not available."""
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required for GNN models but is not installed")


if _TF_AVAILABLE:
    from .constants import GNN_HIDDEN_UNITS, GNN_DROPOUT_RATE, GNN_AGGREGATOR

    class GraphSAGELayer(tf.keras.layers.Layer):
        """Proper GraphSAGE layer with message passing and aggregation."""

        def __init__(self, units, aggregator="mean", use_bias=True, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.aggregator = aggregator
            self.use_bias = use_bias
            self.is_gnn = True

        def build(self, input_shape):
            if isinstance(input_shape, list):
                input_dim = input_shape[0][-1]
            else:
                input_dim = input_shape[-1]

            self.W_self = self.add_weight(
                shape=(input_dim, self.units), initializer="glorot_uniform", name="W_self", trainable=True
            )
            self.W_neigh = self.add_weight(
                shape=(input_dim, self.units), initializer="glorot_uniform", name="W_neigh", trainable=True
            )

            if self.use_bias:
                self.bias = self.add_weight(shape=(self.units,), initializer="zeros", name="bias", trainable=True)

            super().build(input_shape)

        def call(self, inputs, adjacency_matrix=None, training=False):
            """Forward pass with proper message passing."""

            if adjacency_matrix is None:
                result = tf.matmul(inputs, self.W_self)
                if self.use_bias:
                    result += self.bias
                return tf.nn.relu(result)

            # Adjacency matrix must be preprocessed (normalized) before passing here
            if hasattr(adjacency_matrix, "toarray"):
                adj_dense = tf.constant(adjacency_matrix.toarray(), dtype=tf.float32)
            elif hasattr(adjacency_matrix, "indices"):
                adj_dense = tf.sparse.to_dense(adjacency_matrix, default_value=0.0)
                adj_dense = tf.cast(adj_dense, tf.float32)
            else:
                adj_dense = tf.cast(adjacency_matrix, tf.float32)

            neigh_feats = tf.matmul(adj_dense, inputs)
            self_transformed = tf.matmul(inputs, self.W_self)
            neigh_transformed = tf.matmul(neigh_feats, self.W_neigh)
            output = self_transformed + neigh_transformed
            if self.use_bias:
                output += self.bias

            return tf.nn.relu(output)

        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units, "aggregator": self.aggregator, "use_bias": self.use_bias})
            return config

    class GraphSAGEModel(tf.keras.Model):
        """Pure GraphSAGE model."""

        def __init__(
            self,
            n_features,
            n_classes=9,
            aggregator=GNN_AGGREGATOR,
            hidden_units=GNN_HIDDEN_UNITS,
            dropout_rate=GNN_DROPOUT_RATE,
        ):
            super().__init__()
            self.n_features = n_features
            self.hidden_units = hidden_units
            self.dropout_rate = dropout_rate

            self.graphsage1 = GraphSAGELayer(hidden_units, aggregator=aggregator)
            self.bn1 = BatchNormalization()
            self.dropout1 = Dropout(dropout_rate)

            self.graphsage2 = GraphSAGELayer(hidden_units, aggregator=aggregator)
            self.bn2 = BatchNormalization()
            self.dropout2 = Dropout(dropout_rate)

            self.output_layer = Dense(n_classes, activation="softmax")

        def build(self, input_shape):
            if isinstance(input_shape, list) and len(input_shape) == 2:
                node_features_shape = input_shape[0]
            else:
                node_features_shape = input_shape

            self.graphsage1.build(node_features_shape)
            self.graphsage2.build((node_features_shape[0], self.hidden_units))
            self.output_layer.build((node_features_shape[0], self.hidden_units))
            super().build(input_shape)

        def _supports_adjacency(self, layer):
            try:
                import inspect

                sig = inspect.signature(layer.call)
                return "adjacency_matrix" in sig.parameters
            except Exception:
                return False

        def _call_layer_safely(self, layer, inputs, adjacency, training):
            if self._supports_adjacency(layer):
                return layer(inputs, adjacency_matrix=adjacency, training=training)
            else:
                return layer(inputs, training=training)

        def call(self, inputs, training=False):
            if isinstance(inputs, list) and len(inputs) == 2:
                node_features, adjacency = inputs
            else:
                node_features = inputs
                adjacency = None

            x = self._call_layer_safely(self.graphsage1, node_features, adjacency, training)
            x = self.bn1(x, training=training)
            x = self.dropout1(x, training=training)

            x = self._call_layer_safely(self.graphsage2, x, adjacency, training)
            x = self.bn2(x, training=training)
            x = self.dropout2(x, training=training)

            return self.output_layer(x)

    CUSTOM_OBJECTS = {"GraphSAGELayer": GraphSAGELayer, "GraphSAGEModel": GraphSAGEModel}

else:
    # Placeholders when TensorFlow is not available
    GraphSAGELayer = None
    GraphSAGEModel = None
    CUSTOM_OBJECTS = {}


class GNNModelLoadError(Exception):
    """Base exception for errors raised during GNN model loading."""

    pass


class GnnEnsemble:
    """Serves the arithmetic mean of the members' softmax outputs.

    Called exactly like a single model (``model([features, adjacency], training=False)``,
    returning a tensor) and carries the members' shared ``spec``, ``feature_contract``
    and ``imputation_plan``.
    """

    def __init__(self, members: list[Any]) -> None:
        if not members:
            raise GNNModelLoadError("A GNN ensemble needs at least one member")
        digests = {member.spec.model_config_sha for member in members}
        contracts = {member.feature_contract.name for member in members}
        if len(digests) != 1 or len(contracts) != 1:
            raise GNNModelLoadError(f"Ensemble members disagree: specs {sorted(digests)}, contracts {sorted(contracts)}")
        self.members = tuple(members)
        self.spec = members[0].spec
        self.feature_contract = members[0].feature_contract
        self.imputation_plan = members[0].imputation_plan

    def __call__(self, inputs: Any, training: bool = False) -> Any:
        outputs = [member(inputs, training=training) for member in self.members]
        return tf.reduce_mean(tf.stack(outputs, axis=0), axis=0)

    def predict(self, inputs: Any, verbose: int = 0) -> Any:
        """Keras-style ``predict`` on features alone: the mean of the members' outputs, as a NumPy array.

        Not for graph inputs: Keras ``predict`` batches ``[features, adjacency]`` along the
        first axis and fails on it, which is why serving calls the ensemble directly.
        """
        import numpy as np

        return np.mean(np.stack([member.predict(inputs, verbose=verbose) for member in self.members]), axis=0)


def _load_spec_models(weights_paths: list[str], spec_path: str, imputation_path: Optional[str]) -> list[Any]:
    """Build the model a spec sidecar describes once per weight file and load that file into it.

    The spec and the imputation plan are read once and shared by every model.
    """
    import json

    from stormlab.models.gnn_spec import read_spec
    from stormlab.models.gnn import build_graph_model_for_weights, load_graphsage_weights
    from stormlab.contracts.features import FEATURE_CONTRACTS

    spec = read_spec(spec_path)
    # read_spec admits only served contracts, and every served contract is registered.
    contract = FEATURE_CONTRACTS[spec.feature_contract]

    imputation_plan = None
    if imputation_path is not None and os.path.isfile(imputation_path):
        with open(imputation_path, "r", encoding="utf-8") as f:
            imputation_plan = json.load(f)

    models = []
    for weights_path in weights_paths:
        model = build_graph_model_for_weights(spec, n_features=contract.n_columns)
        load_graphsage_weights(model, weights_path)
        model.spec = spec
        model.feature_contract = contract
        model.imputation_plan = imputation_plan
        logger.info(f"Loaded GNN model ({spec.slug()}) with contract {contract.name} from: {weights_path}")
        models.append(model)
    return models


def load_gnn_model_weights(weights_path: Optional[str] = None) -> Optional[Any]:
    """Reconstruct the configured GNN model and load its weights.

    Without ``weights_path`` this loads every configured member and returns a
    :class:`GnnEnsemble` (or the model itself for a single member); a missing
    member, spec or imputation plan fails the load, so a partial ensemble is never served.
    With ``weights_path`` it loads that one file, reading the spec beside it.
    """
    _require_tensorflow()

    from .feature_contract import get_default_feature_columns
    from .enums import RecommendationCategory
    from .predictor import GNN_CONFIG

    recommendations_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "recommendations")
    model_weights_path = weights_path or recommendations_dir

    try:
        if weights_path is None:
            member_paths = [os.path.join(recommendations_dir, name) for name in GNN_CONFIG["weights"]]
            spec_path = os.path.join(recommendations_dir, GNN_CONFIG["spec"])
            imputation_path = os.path.join(recommendations_dir, GNN_CONFIG["imputation"])
            missing = [path for path in [*member_paths, spec_path, imputation_path] if not os.path.isfile(path)]
            if missing:
                raise GNNModelLoadError(f"GNN ensemble files missing, refusing to serve a partial model: {missing}")
            logger.info(f"Loading GNN ensemble of {len(member_paths)} member(s) from: {recommendations_dir}")
            members = _load_spec_models(member_paths, spec_path, imputation_path)
            return members[0] if len(members) == 1 else GnnEnsemble(members)

        weights_dir = os.path.dirname(weights_path)
        spec_path = None
        if os.path.basename(weights_path) in GNN_CONFIG["weights"] and GNN_CONFIG.get("spec"):
            candidate = os.path.join(weights_dir, GNN_CONFIG["spec"])
            if os.path.isfile(candidate):
                spec_path = candidate
        if not spec_path:
            base, _ = os.path.splitext(weights_path)
            if base.endswith(".weights"):
                base = base[:-8]
            candidate = f"{base}.spec.json"
            if os.path.isfile(candidate):
                spec_path = candidate

        logger.info(f"Loading GNN weights from: {weights_path}")
        if spec_path:
            imputation_path = os.path.join(weights_dir, GNN_CONFIG["imputation"])
            return _load_spec_models([weights_path], spec_path, imputation_path)[0]
        else:
            n_features = len(get_default_feature_columns())
            n_classes = len(RecommendationCategory)

            model = GraphSAGEModel(n_features=n_features, n_classes=n_classes)
            model([tf.zeros((1, n_features)), tf.zeros((1, 1))], training=False)
            model.load_weights(model_weights_path)

            logger.info(f"Loaded legacy GNN weights from: {model_weights_path}")
            return model

    except FileNotFoundError:
        logger.error(f"Model file not found at: {model_weights_path}")
    except (ValueError, OSError) as e:
        logger.error(
            f"Failed to load GNN model from {model_weights_path}. "
            f"The file may be corrupted or incompatible with model architecture. Error: {e}",
            exc_info=True,
        )
    except GNNModelLoadError as e:
        logger.error(
            f"An unexpected critical error occurred while loading the GNN model: {e}",
            exc_info=True,
        )
    return None
