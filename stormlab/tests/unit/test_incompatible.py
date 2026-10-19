"""Spatial consistency on the raw graph, with an explicit incompatibility matrix."""

from __future__ import annotations

import pandas as pd
import pytest
import scipy.sparse as sp

from stormlab.contracts.enums import RecommendationCategory
from stormlab.graph.build import build_graph
from stormlab.graph.incompatible import (
    INCOMPATIBLE_ARGMAX_IDENTITY,
    SpatialConsistencyError,
    pooled_spatial_consistency,
    spatial_consistency,
)

VALID = RecommendationCategory.VALID
PUMP = RecommendationCategory.PUMP
TANK = RecommendationCategory.TANK


def _chain(n: int, prefix: str = "K") -> tuple[sp.csr_matrix, list[str]]:
    """Raw adjacency of an ``n``-conduit chain and its order."""
    dfc = pd.DataFrame(
        {
            "Name": [f"{prefix}{i + 1}" for i in range(n)],
            "InletNode": [f"{prefix}N{i}" for i in range(n)],
            "OutletNode": [f"{prefix}N{i + 1}" for i in range(n)],
        }
    )
    result = build_graph(dfc)
    return result.adjacency, list(result.order)


class TestIncompatibleMatrix:
    def test_it_is_a_nine_by_nine_boolean_frame(self):
        assert INCOMPATIBLE_ARGMAX_IDENTITY.shape == (9, 9)
        assert all(pd.api.types.is_bool_dtype(dtype) for dtype in INCOMPATIBLE_ARGMAX_IDENTITY.dtypes)

    def test_it_is_indexed_by_the_enum_in_declaration_order(self):
        categories = list(RecommendationCategory)
        assert list(INCOMPATIBLE_ARGMAX_IDENTITY.index) == categories
        assert list(INCOMPATIBLE_ARGMAX_IDENTITY.columns) == categories

    def test_it_encodes_the_historical_argmax_identity_rule(self):
        """Same class compatible, any two different classes incompatible."""
        for left in RecommendationCategory:
            for right in RecommendationCategory:
                assert bool(INCOMPATIBLE_ARGMAX_IDENTITY.at[left, right]) == (left is not right)


class TestSpatialConsistency:
    def test_hand_counted_value_on_a_four_conduit_chain(self):
        """Three edges, two of which join equal recommendations: 2/3."""
        adjacency, order = _chain(4)
        labels = pd.Series([VALID, VALID, VALID, PUMP], index=order)

        assert spatial_consistency(labels, adjacency, order, INCOMPATIBLE_ARGMAX_IDENTITY) == pytest.approx(2 / 3)

    def test_a_uniform_field_is_perfectly_consistent(self):
        adjacency, order = _chain(4)
        labels = pd.Series([VALID] * 4, index=order)

        assert spatial_consistency(labels, adjacency, order, INCOMPATIBLE_ARGMAX_IDENTITY) == pytest.approx(1.0)

    def test_an_alternating_field_is_fully_inconsistent(self):
        adjacency, order = _chain(4)
        labels = pd.Series([VALID, PUMP, VALID, PUMP], index=order)

        assert spatial_consistency(labels, adjacency, order, INCOMPATIBLE_ARGMAX_IDENTITY) == pytest.approx(0.0)

    def test_string_labels_are_accepted(self):
        adjacency, order = _chain(4)
        labels = pd.Series(["valid", "valid", "valid", "pump"], index=order)

        assert spatial_consistency(labels, adjacency, order, INCOMPATIBLE_ARGMAX_IDENTITY) == pytest.approx(2 / 3)

    def test_a_custom_matrix_can_make_different_classes_compatible(self):
        adjacency, order = _chain(4)
        labels = pd.Series([VALID, PUMP, VALID, PUMP], index=order)

        tolerant = INCOMPATIBLE_ARGMAX_IDENTITY.copy()
        tolerant.loc[VALID, PUMP] = False
        tolerant.loc[PUMP, VALID] = False

        assert spatial_consistency(labels, adjacency, order, tolerant) == pytest.approx(1.0)

    def test_every_directed_edge_counts_regardless_of_row_order(self):
        """The historical metric kept only the ``i < j`` half of the edges."""
        adjacency, order = _chain(4)
        reversed_order = list(reversed(order))
        reversed_adjacency = build_graph(
            pd.DataFrame(
                {
                    "Name": order,
                    "InletNode": [f"KN{i}" for i in range(4)],
                    "OutletNode": [f"KN{i + 1}" for i in range(4)],
                }
            ),
            conduit_order=reversed_order,
        ).adjacency
        labels = pd.Series([VALID, VALID, VALID, PUMP], index=order)

        forward = spatial_consistency(labels, adjacency, order, INCOMPATIBLE_ARGMAX_IDENTITY)
        backward = spatial_consistency(labels, reversed_adjacency, reversed_order, INCOMPATIBLE_ARGMAX_IDENTITY)
        assert forward == pytest.approx(backward)


class TestSpatialConsistencyValidation:
    def test_a_graph_without_edges_raises_instead_of_returning_zero(self):
        empty = sp.csr_matrix((3, 3))
        order = ["K1", "K2", "K3"]
        labels = pd.Series([VALID] * 3, index=order)

        with pytest.raises(SpatialConsistencyError, match="undefined on a graph without edges"):
            spatial_consistency(labels, empty, order, INCOMPATIBLE_ARGMAX_IDENTITY)

    def test_a_missing_prediction_raises(self):
        adjacency, order = _chain(4)
        labels = pd.Series([VALID] * 3, index=order[:3])

        with pytest.raises(SpatialConsistencyError, match="No prediction for conduit"):
            spatial_consistency(labels, adjacency, order, INCOMPATIBLE_ARGMAX_IDENTITY)

    def test_a_shape_mismatch_raises(self):
        adjacency, order = _chain(4)
        labels = pd.Series([VALID] * 4, index=order)

        with pytest.raises(SpatialConsistencyError, match="does not match"):
            spatial_consistency(labels, adjacency, order[:3], INCOMPATIBLE_ARGMAX_IDENTITY)

    def test_an_unknown_label_raises(self):
        adjacency, order = _chain(4)
        labels = pd.Series(["valid", "valid", "valid", "build_a_bypass"], index=order)

        with pytest.raises(SpatialConsistencyError, match="is not a recommendation class"):
            spatial_consistency(labels, adjacency, order, INCOMPATIBLE_ARGMAX_IDENTITY)

    def test_a_non_enum_indexed_matrix_raises(self):
        adjacency, order = _chain(4)
        labels = pd.Series([VALID] * 4, index=order)
        wrong = INCOMPATIBLE_ARGMAX_IDENTITY.copy()
        wrong.index = pd.Index([category.value for category in RecommendationCategory])

        with pytest.raises(SpatialConsistencyError, match="indexed by RecommendationCategory"):
            spatial_consistency(labels, adjacency, order, wrong)


class TestPooledOverSimulations:
    def test_edges_from_all_graphs_form_one_population(self):
        """Plan §4b asks for SL over all test simulations, not a mean of means."""
        small_adjacency, small_order = _chain(2, prefix="S")  # 1 edge, inconsistent
        large_adjacency, large_order = _chain(5, prefix="L")  # 4 edges, all consistent

        graphs = [
            (pd.Series([VALID, PUMP], index=small_order), small_adjacency, small_order),
            (pd.Series([TANK] * 5, index=large_order), large_adjacency, large_order),
        ]

        assert pooled_spatial_consistency(graphs, INCOMPATIBLE_ARGMAX_IDENTITY) == pytest.approx(4 / 5)

    def test_pooling_differs_from_averaging_per_graph_scores(self):
        small_adjacency, small_order = _chain(2, prefix="S")
        large_adjacency, large_order = _chain(5, prefix="L")
        small = (pd.Series([VALID, PUMP], index=small_order), small_adjacency, small_order)
        large = (pd.Series([TANK] * 5, index=large_order), large_adjacency, large_order)

        per_graph_mean = (
            spatial_consistency(*small, INCOMPATIBLE_ARGMAX_IDENTITY) + spatial_consistency(*large, INCOMPATIBLE_ARGMAX_IDENTITY)
        ) / 2
        assert pooled_spatial_consistency([small, large], INCOMPATIBLE_ARGMAX_IDENTITY) != pytest.approx(per_graph_mean)

    def test_an_edgeless_population_raises(self):
        empty = sp.csr_matrix((2, 2))
        order = ["K1", "K2"]
        labels = pd.Series([VALID, VALID], index=order)

        with pytest.raises(SpatialConsistencyError, match="has an edge"):
            pooled_spatial_consistency([(labels, empty, order)], INCOMPATIBLE_ARGMAX_IDENTITY)


class TestTheMatrixIsAnInput:
    """The metric must never invent a compatibility rule of its own."""

    def test_spatial_consistency_requires_the_matrix(self):
        adjacency, order = _chain(4)
        labels = pd.Series([VALID, VALID, VALID, PUMP], index=order)

        with pytest.raises(TypeError):
            spatial_consistency(labels, adjacency, order)  # type: ignore[call-arg]

    def test_pooled_spatial_consistency_requires_the_matrix(self):
        adjacency, order = _chain(4)
        labels = pd.Series([VALID, VALID, VALID, PUMP], index=order)

        with pytest.raises(TypeError):
            pooled_spatial_consistency([(labels, adjacency, order)])  # type: ignore[call-arg]
