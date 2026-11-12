"""Graph recommendation model: architecture, weight loading and forward pass.

The shipped ``.weights.h5`` file is a bare weight file, not a serialised model.
Loading it requires rebuilding the exact layer stack it was saved from, with the
same layer classes, weight names and creation order. That stack is::

    [Normalization] -> GraphOperatorLayer -> norm -> Dropout
                    -> GraphOperatorLayer -> norm -> Dropout -> Dense(9, softmax)

with ``relu(X·W_self + (A·X)·W_neigh + b)`` per graph step for the matrix-product
operators. The model was trained on a labelled dataset of storm-sewer conduits;
this module only builds it and loads its weights.

TensorFlow is imported inside the functions, so this module imports without it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from stormlab.config import (
    GNN_INPUT_NORM_CONSTANT_SAFE,
    GNN_NORMALIZATION_BATCH,
    GNN_NORMALIZATION_LAYER,
    GNN_NORMALIZATION_NONE,
    GNN_NORMALIZATIONS,
)
from stormlab.contracts.enums import RecommendationCategory
from stormlab.contracts.features import EXPECTED_N_FEATURES
from stormlab.models.gnn_spec import (
    GRAPH_FREE_OPERATORS,
    MATMUL_OPERATORS,
    OPERATOR_SUPPORT_MAX,
    OPERATOR_SUPPORT_POOL,
    GnnSpec,
    GnnSpecC1,
)

GraphModelSpec = GnnSpec | GnnSpecC1
"""What :func:`build_graph_model_for_weights` accepts."""

N_CLASSES = len(RecommendationCategory)

NORMALIZATION_BATCH = GNN_NORMALIZATION_BATCH
NORMALIZATION_LAYER = GNN_NORMALIZATION_LAYER
NORMALIZATION_NONE = GNN_NORMALIZATION_NONE
NORMALIZATIONS: tuple[str, ...] = GNN_NORMALIZATIONS
"""Normalisers that may sit after each graph step."""

NORMALIZER_LAYER_NAME = "normalizer"
"""Name of the input ``Normalization`` layer; part of the saved weight layout."""


class GraphSAGEModelError(RuntimeError):
    """Raised when the graph model cannot be built, called or loaded."""


def _normalization_layer(keras: Any, kind: str) -> Any:
    """The normaliser named by ``kind``; ``None`` when there is to be none.

    Raises:
        GraphSAGEModelError: on a normalisation outside :data:`NORMALIZATIONS`.
    """
    if kind == NORMALIZATION_BATCH:
        return keras.layers.BatchNormalization()
    if kind == NORMALIZATION_LAYER:
        return keras.layers.LayerNormalization()
    if kind == NORMALIZATION_NONE:
        return None
    raise GraphSAGEModelError(f"Unknown normalization {kind!r}; expected one of {list(NORMALIZATIONS)}")


def _split_model_inputs(inputs: Any) -> tuple[Any, Any, Any]:
    """``[X]`` / ``[X, A]`` / ``[X, A_down, A_up]`` as a fixed triple.

    Raises:
        GraphSAGEModelError: on a list of any other length.
    """
    if not isinstance(inputs, list):
        return inputs, None, None
    if len(inputs) == 1:
        return inputs[0], None, None
    if len(inputs) == 2:
        return inputs[0], inputs[1], None
    if len(inputs) == 3:
        return inputs[0], inputs[1], inputs[2]
    raise GraphSAGEModelError(f"A graph model takes features and one or two adjacencies, got a list of {len(inputs)} input(s)")


@lru_cache(maxsize=1)
def graph_operator_classes() -> tuple[type, type]:
    """The ``(GraphOperatorLayer, GraphOperatorModel)`` pair, defined against Keras.

    Cached so the classes are defined once per process: Keras identifies custom
    objects by class.

    Raises:
        GraphSAGEModelError: if TensorFlow is not installed.
    """
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - environment dependent
        raise GraphSAGEModelError("TensorFlow is required to build a graph recommendation model") from error

    keras = tf.keras

    class GraphOperatorLayer(keras.layers.Layer):
        """One graph step; every operator shares the self term and the outer ``relu``.

        * ``legacy_weighted_mean``, ``khop_sum``, ``khop_sym``:
          ``relu(X·W_self + (A·X)·W_neigh + b)``; they differ only in the ``A``
          they are given.
        * ``support_max``: ``relu(X·W_self + max_{u in N(v)}(X_u·W_neigh) + b)``.
        * ``support_pool``: ``relu(X·W_self + max_{u in N(v)} relu(X_u·W_pool + b_pool) + b)``.
        * ``nograph``: ``relu(X·W_self + b)``; the adjacency is never read.
        * two channels: ``relu(X·W_self + (A_down·X)·W_neigh_down + (A_up·X)·W_neigh_up + b)``.
        """

        def __init__(self, units: int, operator: str, use_bias: bool = True, two_channel: bool = False, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.units = units
            self.operator = operator
            self.use_bias = use_bias
            self.two_channel = two_channel

        def build(self, input_shape: Any) -> None:
            """Create the weights; their names and order are what the weight file was saved with."""
            input_dim = input_shape[0][-1] if isinstance(input_shape, list) else input_shape[-1]
            self.W_self = self.add_weight(
                shape=(input_dim, self.units), initializer="glorot_uniform", name="W_self", trainable=True
            )
            if self.two_channel:
                self.W_neigh_down = self.add_weight(
                    shape=(input_dim, self.units), initializer="glorot_uniform", name="W_neigh_down", trainable=True
                )
                self.W_neigh_up = self.add_weight(
                    shape=(input_dim, self.units), initializer="glorot_uniform", name="W_neigh_up", trainable=True
                )
            elif self.operator == OPERATOR_SUPPORT_POOL:
                self.W_pool = self.add_weight(
                    shape=(input_dim, self.units), initializer="glorot_uniform", name="W_pool", trainable=True
                )
                self.b_pool = self.add_weight(shape=(self.units,), initializer="zeros", name="b_pool", trainable=True)
            elif self.operator not in GRAPH_FREE_OPERATORS:
                self.W_neigh = self.add_weight(
                    shape=(input_dim, self.units), initializer="glorot_uniform", name="W_neigh", trainable=True
                )
            if self.use_bias:
                self.bias = self.add_weight(shape=(self.units,), initializer="zeros", name="bias", trainable=True)
            super().build(input_shape)

        def _masked_max(self, values: Any, adjacency: Any) -> Any:
            """``max`` of ``values`` over each row's support of ``adjacency`` (dense)."""
            support = tf.not_equal(tf.cast(adjacency, tf.float32), 0.0)
            candidates = tf.expand_dims(values, axis=0)
            neighbours = tf.where(
                tf.expand_dims(support, axis=-1),
                candidates,
                tf.constant(float("-inf"), dtype=values.dtype),
            )
            return tf.reduce_max(neighbours, axis=1)

        def call(self, inputs: Any, adjacency_matrix: Any = None, adjacency_up: Any = None, training: bool = False) -> Any:
            """Aggregate over the graph; without an adjacency, only the self term is used."""
            self_transformed = tf.matmul(inputs, self.W_self)
            if self.two_channel:
                if adjacency_matrix is None or adjacency_up is None:
                    raise GraphSAGEModelError(
                        "A two-channel graph step needs both adjacencies; it was called with "
                        f"downstream={'a matrix' if adjacency_matrix is not None else None} and "
                        f"upstream={'a matrix' if adjacency_up is not None else None}. Reading one channel and "
                        "dropping the other would report a two-channel model's numbers for a one-channel model"
                    )
                down = tf.matmul(tf.matmul(tf.cast(adjacency_matrix, tf.float32), inputs), self.W_neigh_down)
                up = tf.matmul(tf.matmul(tf.cast(adjacency_up, tf.float32), inputs), self.W_neigh_up)
                output = self_transformed + down + up
                if self.use_bias:
                    output += self.bias
                return tf.nn.relu(output)
            if adjacency_matrix is None or self.operator in GRAPH_FREE_OPERATORS:
                output = self_transformed
            elif self.operator in MATMUL_OPERATORS:
                adj_dense = tf.cast(adjacency_matrix, tf.float32)
                output = self_transformed + tf.matmul(tf.matmul(adj_dense, inputs), self.W_neigh)
            elif self.operator == OPERATOR_SUPPORT_MAX:
                output = self_transformed + self._masked_max(tf.matmul(inputs, self.W_neigh), adjacency_matrix)
            elif self.operator == OPERATOR_SUPPORT_POOL:
                pooled = tf.nn.relu(tf.matmul(inputs, self.W_pool) + self.b_pool)
                output = self_transformed + self._masked_max(pooled, adjacency_matrix)
            else:  # pragma: no cover - GnnSpec closes the set before a layer exists
                raise GraphSAGEModelError(f"Graph operator {self.operator!r} has no aggregation branch")
            if self.use_bias:
                output += self.bias
            return tf.nn.relu(output)

        def get_config(self) -> dict[str, Any]:
            config = super().get_config()
            config.update(
                {
                    "units": self.units,
                    "operator": self.operator,
                    "use_bias": self.use_bias,
                    "two_channel": self.two_channel,
                }
            )
            return config

    class GraphOperatorModel(keras.Model):
        """Optional input ``Normalization``, two graph steps over a shared adjacency, softmax head."""

        def __init__(self, spec: GraphModelSpec, *, n_features: int, n_classes: int, normalizer: Any = None) -> None:
            super().__init__()
            self.spec = spec
            self.n_features = n_features
            self.hidden_units = spec.hidden
            self.dropout_rate = spec.dropout
            self.normalization = spec.layer_norm
            self.two_channel = bool(spec.two_channel)

            # `None` rather than an identity layer: with `input_norm='none'` the layer is absent.
            self.normalizer = normalizer

            self.graphsage1 = GraphOperatorLayer(spec.hidden, operator=spec.operator, two_channel=self.two_channel)
            self.bn1 = _normalization_layer(keras, spec.layer_norm)
            self.dropout1 = keras.layers.Dropout(spec.dropout)

            self.graphsage2 = GraphOperatorLayer(spec.hidden, operator=spec.operator, two_channel=self.two_channel)
            self.bn2 = _normalization_layer(keras, spec.layer_norm)
            self.dropout2 = keras.layers.Dropout(spec.dropout)

            self.output_layer = keras.layers.Dense(n_classes, activation="softmax")

        def build(self, input_shape: Any) -> None:
            if isinstance(input_shape, list) and len(input_shape) in (2, 3):
                node_features_shape = input_shape[0]
            else:
                node_features_shape = input_shape
            self.graphsage1.build(node_features_shape)
            self.graphsage2.build((node_features_shape[0], self.hidden_units))
            self.output_layer.build((node_features_shape[0], self.hidden_units))
            super().build(input_shape)

        def call(self, inputs: Any, training: bool = False) -> Any:
            """Two graph steps, then the head. Inputs are ``[X, A]`` or ``[X, A_down, A_up]``.

            Raises:
                GraphSAGEModelError: if the number of adjacencies does not match the model.
            """
            node_features, adjacency, adjacency_up = _split_model_inputs(inputs)
            if self.two_channel and adjacency_up is None:
                raise GraphSAGEModelError(
                    "This model has two-channel graph steps and was called with a single adjacency. Pass "
                    "[features, adjacency_down, adjacency_up]"
                )
            if not self.two_channel and adjacency_up is not None:
                raise GraphSAGEModelError(
                    "This model has one-channel graph steps and was called with two adjacencies; the second one "
                    "would be silently ignored"
                )

            if self.normalizer is not None:
                node_features = self.normalizer(node_features)

            hidden = self.graphsage1(node_features, adjacency_matrix=adjacency, adjacency_up=adjacency_up, training=training)
            if self.bn1 is not None:
                hidden = self.bn1(hidden, training=training)
            hidden = self.dropout1(hidden, training=training)

            hidden = self.graphsage2(hidden, adjacency_matrix=adjacency, adjacency_up=adjacency_up, training=training)
            if self.bn2 is not None:
                hidden = self.bn2(hidden, training=training)
            hidden = self.dropout2(hidden, training=training)

            return self.output_layer(hidden)

    return GraphOperatorLayer, GraphOperatorModel


def build_graph_model_for_weights(
    spec: GraphModelSpec,
    *,
    n_features: int = EXPECTED_N_FEATURES,
    n_classes: int = N_CLASSES,
) -> Any:
    """Build the model a spec describes, with an unadapted ``Normalization``, ready for ``load_weights``.

    The saved weights carry the normaliser's mean and variance, so nothing is
    adapted here; :func:`load_graphsage_weights` overwrites the placeholders.

    Raises:
        GraphSAGEModelError: if TensorFlow is missing or the spec names an unknown normalisation.
    """
    normalizer = None
    if spec.input_norm == GNN_INPUT_NORM_CONSTANT_SAFE:
        import keras

        normalizer = keras.layers.Normalization(axis=-1, name=NORMALIZER_LAYER_NAME)
        normalizer.build((None, n_features))
    return _warm_up_graph_model(spec, n_features=n_features, n_classes=n_classes, normalizer=normalizer)


def _warm_up_graph_model(spec: GraphModelSpec, *, n_features: int, n_classes: int, normalizer: Any) -> Any:
    """Instantiate the model and call it once on one all-zero node, so its weights exist.

    The 1x1 warm-up is what fixes the weight names and order the saved file was
    written against. A two-channel spec gets a second 1x1 matrix.
    """
    import tensorflow as tf

    _, model_class = graph_operator_classes()
    model = model_class(spec, n_features=n_features, n_classes=n_classes, normalizer=normalizer)
    inputs = [tf.zeros((1, n_features)), tf.zeros((1, 1))]
    if spec.two_channel:
        inputs.append(tf.zeros((1, 1)))
    model(inputs, training=False)
    return model


def load_graphsage_weights(model: Any, weights: Path | str) -> Any:
    """Load a weight file into a built model, failing loudly if it does not fit.

    Raises:
        GraphSAGEModelError: if the file is missing or Keras rejects its contents.
    """
    path = Path(weights)
    if not path.is_file():
        raise GraphSAGEModelError(f"GraphSAGE weight file not found: {path}")
    try:
        model.load_weights(path)
    except (ValueError, OSError) as error:
        raise GraphSAGEModelError(
            f"Could not load GraphSAGE weights from {path}: {type(error).__name__}: {error}. "
            "The architecture in stormlab.models.gnn and the saved file disagree."
        ) from error
    return model


__all__ = [
    "N_CLASSES",
    "NORMALIZATIONS",
    "NORMALIZER_LAYER_NAME",
    "GraphModelSpec",
    "GraphSAGEModelError",
    "build_graph_model_for_weights",
    "graph_operator_classes",
    "load_graphsage_weights",
]
