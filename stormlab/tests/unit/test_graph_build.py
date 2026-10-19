"""Adjacency construction and the legacy k-hop preprocessing.

Moved from ``app/sa/core/tests/test_gnn_graph.py`` (16 tests) with the imports
pointed at ``stormlab.graph`` and the direct-constructor test adapted to the
named :class:`GraphBuildResult` that replaced the legacy ``(matrix, None)``
tuple. The golden fixture travelled with them so the unit layer stays
self-contained.

T1 covers ``SWMMGraphConstructor`` / ``build_adjacency_from_dfc``: directed edge
construction from ``OutletNode(u) == InletNode(v)``, case/whitespace
insensitivity, zero raw diagonal, isolated conduits, and permutation stability
of ``conduit_order``.

T2 hand-verifies ``preprocess_adjacency`` on a 4-node chain: self-loops, the
hop-weighting scheme, diagonal restoration, and row-normalization.
"""

import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from stormlab.graph.build import (
    GraphBuildError,
    SWMMGraphConstructor,
    build_adjacency_from_dfc,
    build_graph,
    build_graph_channels,
)
from stormlab.graph.khop import preprocess_adjacency

FIXTURE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "gnn_golden_fixture.npz")


def _synthetic_dfc() -> pd.DataFrame:
    """Chain (C1->C2), branch (C1->C3), continuation (C3->C5), isolate (C4).

    Inlet/Outlet strings deliberately mix case and stray whitespace to
    exercise the case/space-insensitive matching rule.
    """
    return pd.DataFrame(
        {
            "Name": ["C1", "C2", "C3", "C4", "C5"],
            "InletNode": ["A", " B ", "b", "X", "D"],
            "OutletNode": ["B", "C", "D", "Y", "E"],
        }
    )


class TestBuildAdjacencyFromDfc:
    """T1: build_adjacency_from_dfc (serving entry point)."""

    def test_directed_edges_match_outlet_to_inlet(self):
        dfc = _synthetic_dfc()
        A = build_adjacency_from_dfc(dfc).toarray()
        idx = {name: i for i, name in enumerate(dfc["Name"])}

        # C1(Outlet="B") -> C2(Inlet=" B "): case/space-insensitive match.
        assert A[idx["C1"], idx["C2"]] == 1
        # C1(Outlet="B") -> C3(Inlet="b"): branch + case-insensitive match.
        assert A[idx["C1"], idx["C3"]] == 1
        # C3(Outlet="D") -> C5(Inlet="D"): chain continuation.
        assert A[idx["C3"], idx["C5"]] == 1

    def test_no_spurious_reverse_edges(self):
        dfc = _synthetic_dfc()
        A = build_adjacency_from_dfc(dfc).toarray()
        idx = {name: i for i, name in enumerate(dfc["Name"])}

        assert A[idx["C2"], idx["C1"]] == 0
        assert A[idx["C3"], idx["C1"]] == 0
        assert A[idx["C5"], idx["C3"]] == 0

    def test_raw_diagonal_is_zero(self):
        dfc = _synthetic_dfc()
        A = build_adjacency_from_dfc(dfc).toarray()
        assert np.allclose(np.diag(A), 0)

    def test_isolated_conduit_has_zero_row_and_column(self):
        dfc = _synthetic_dfc()
        A = build_adjacency_from_dfc(dfc).toarray()
        idx = {name: i for i, name in enumerate(dfc["Name"])}
        isolate = idx["C4"]

        assert np.allclose(A[isolate, :], 0)
        assert np.allclose(A[:, isolate], 0)

    def test_returns_raw_csr_matrix(self):
        dfc = _synthetic_dfc()
        A = build_adjacency_from_dfc(dfc)
        assert sp.issparse(A)
        assert A.format == "csr"
        assert A.dtype == np.float32

    def test_permutation_of_conduit_order_permutes_adjacency(self):
        """Stability w.r.t. conduit_order: a permuted node order must yield
        exactly the permuted adjacency, not a re-derived/different graph."""
        dfc = _synthetic_dfc()
        names = list(dfc["Name"])
        A1 = build_adjacency_from_dfc(dfc).toarray()

        permuted_order = ["C5", "C3", "C1", "C4", "C2"]
        A2 = build_adjacency_from_dfc(dfc, conduit_order=permuted_order).toarray()

        perm = [names.index(name) for name in permuted_order]
        expected = A1[np.ix_(perm, perm)]

        assert np.array_equal(A2, expected)

    def test_missing_name_column_falls_back_to_index(self):
        dfc = _synthetic_dfc().drop(columns=["Name"])
        A = build_adjacency_from_dfc(dfc)
        assert A.shape == (5, 5)
        assert np.allclose(np.diag(A.toarray()), 0)


class TestSWMMGraphConstructorDirect:
    """Direct construction API used internally by build_adjacency_from_dfc."""

    def test_build_conduit_graph_matches_helper(self):
        dfc = _synthetic_dfc()
        constructor = SWMMGraphConstructor(dfc)
        result = constructor.build_conduit_graph()

        assert result.order == tuple(dfc["Name"])
        A_helper = build_adjacency_from_dfc(dfc)
        assert np.array_equal(result.adjacency.toarray(), A_helper.toarray())

    def test_conduit_to_idx_matches_row_order_when_order_not_given(self):
        dfc = _synthetic_dfc()
        constructor = SWMMGraphConstructor(dfc)
        constructor.build_conduit_graph()

        for i, name in enumerate(dfc["Name"]):
            assert constructor.conduit_to_idx[name] == i
            assert constructor.idx_to_conduit[i] == name


class TestRealTopologyReconstructionParity:
    """T1 extension: the serving-side graph reconstruction must reproduce the
    exact TRAINING adjacency for the real network topology.

    The golden fixture carries the real training network's node columns
    (`names`, `inlet_nodes`, `outlet_nodes`) aligned to `conduit_order`, plus
    the training-computed raw adjacency `A_raw_dense`. Rebuilding adjacency
    from those node columns via the production entry point
    (build_adjacency_from_dfc) must match A_raw_dense exactly. No TF needed.
    """

    def test_build_adjacency_reproduces_training_raw_adjacency(self):
        fixture = np.load(FIXTURE_PATH, allow_pickle=True)

        dfc = pd.DataFrame(
            {
                "Name": [str(n) for n in fixture["names"]],
                "InletNode": [str(n) for n in fixture["inlet_nodes"]],
                "OutletNode": [str(n) for n in fixture["outlet_nodes"]],
            }
        )

        reconstructed = build_adjacency_from_dfc(dfc).toarray()
        expected = fixture["A_raw_dense"]

        assert reconstructed.shape == expected.shape
        assert np.array_equal(reconstructed, expected)


class TestPreprocessAdjacency:
    """T2: hand-computed expected values for a 4-node chain 0->1->2->3."""

    @pytest.fixture
    def chain_adjacency(self):
        return sp.csr_matrix(
            np.array(
                [
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                ],
                dtype=np.float32,
            )
        )

    def test_hand_computed_values_max_hops_4(self, chain_adjacency):
        """Derivation: let N be the 4x4 nilpotent shift (superdiagonal ones).
        adj0 = A + I = I + N.  adj0^k = sum_j C(k,j) N^j (I, N commute).
        multi_hop = adj0 + 0.5*adj0^2 + 0.25*adj0^3 + 0.125*adj0^4
                  = 1.875*I + 3.25*N + 2.0*N^2 + 0.75*N^3
        (coefficients: I: 1+.5+.25+.125=1.875; N: 1+1+.75+.5=3.25;
         N^2: .5+.75+.75=2.0; N^3: .25+.5=0.75)
        setdiag(adj0.diagonal()) resets the diagonal back to 1 (adj0's own
        self-loop value), giving, before row-normalization:
            [[1, 3.25, 2.0, 0.75],
             [0, 1,    3.25, 2.0],
             [0, 0,    1,    3.25],
             [0, 0,    0,    1]]
        Row sums: 7.0, 6.25, 4.25, 1.0.
        """
        result = preprocess_adjacency(chain_adjacency, max_hops=4).toarray()

        expected = np.array(
            [
                [1 / 7, 3.25 / 7, 2.0 / 7, 0.75 / 7],
                [0.0, 1 / 6.25, 3.25 / 6.25, 2.0 / 6.25],
                [0.0, 0.0, 1 / 4.25, 3.25 / 4.25],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        assert np.allclose(result, expected, atol=1e-6)

    def test_row_sums_equal_one_on_nonzero_rows(self, chain_adjacency):
        result = preprocess_adjacency(chain_adjacency, max_hops=4).toarray()
        row_sums = result.sum(axis=1)
        nonzero_rows = result.any(axis=1)
        assert nonzero_rows.all()  # self-loops make every row nonzero here
        assert np.allclose(row_sums[nonzero_rows], 1.0)

    def test_add_self_loops_flag_controls_diagonal(self, chain_adjacency):
        with_loops = preprocess_adjacency(chain_adjacency, add_self_loops=True, normalize=False, max_hops=1).toarray()
        without_loops = preprocess_adjacency(chain_adjacency, add_self_loops=False, normalize=False, max_hops=1).toarray()

        assert np.allclose(np.diag(with_loops), 1)
        assert np.allclose(np.diag(without_loops), 0)

    def test_single_hop_matches_row_normalized_self_loop(self, chain_adjacency):
        result = preprocess_adjacency(chain_adjacency, max_hops=1).toarray()
        # max_hops=1: adj = A + I, row-normalized directly (no multi-hop combination).
        adj0 = chain_adjacency.toarray() + np.eye(4)
        expected = adj0 / adj0.sum(axis=1, keepdims=True)
        assert np.allclose(result, expected)

    def test_accepts_dense_input_and_returns_csr(self, chain_adjacency):
        dense_input = chain_adjacency.toarray()
        result = preprocess_adjacency(dense_input, max_hops=4)
        assert sp.issparse(result)
        assert result.format == "csr"
        assert np.allclose(result.toarray(), preprocess_adjacency(chain_adjacency, max_hops=4).toarray())

    def test_accepts_coo_input_and_returns_csr(self, chain_adjacency):
        coo_input = chain_adjacency.tocoo()
        result = preprocess_adjacency(coo_input, max_hops=4)
        assert sp.issparse(result)
        assert result.format == "csr"
        assert np.allclose(result.toarray(), preprocess_adjacency(chain_adjacency, max_hops=4).toarray())


class TestGraphBuildResult:
    """The reported counts and the direction switch."""

    def test_reports_the_number_of_hydraulic_edges(self):
        result = build_graph(_synthetic_dfc())
        assert result.n_edges_raw == 3
        assert result.n_surrogate_edges == 0
        assert result.adjacency.nnz == 3

    def test_order_follows_conduit_order_when_given(self):
        permuted = ["C5", "C3", "C1", "C4", "C2"]
        result = build_graph(_synthetic_dfc(), conduit_order=permuted)
        assert result.order == tuple(permuted)

    def test_upstream_direction_is_the_transpose(self):
        dfc = _synthetic_dfc()
        downstream = build_graph(dfc, direction="downstream")
        upstream = build_graph(dfc, direction="upstream")

        assert upstream.adjacency.format == "csr"
        assert upstream.adjacency.dtype == np.float32  # transposing goes through csc; dtype must survive
        assert np.array_equal(upstream.adjacency.toarray(), downstream.adjacency.toarray().T)
        assert upstream.n_edges_raw == downstream.n_edges_raw

    def test_unknown_direction_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown graph direction"):
            build_graph(_synthetic_dfc(), direction="sideways")  # type: ignore[arg-type]


class TestMissingDataIsRejected:
    """What the legacy builder dropped in a log line now raises."""

    def test_nan_node_column_raises_with_the_conduit_named(self):
        dfc = _synthetic_dfc()
        dfc.loc[2, "OutletNode"] = np.nan

        with pytest.raises(GraphBuildError, match="C3"):
            build_adjacency_from_dfc(dfc)

    def test_blank_node_column_raises(self):
        dfc = _synthetic_dfc()
        dfc.loc[1, "InletNode"] = "   "

        with pytest.raises(GraphBuildError, match="C2"):
            build_adjacency_from_dfc(dfc)

    def test_absent_node_columns_raise(self):
        dfc = _synthetic_dfc().drop(columns=["OutletNode"])

        with pytest.raises(GraphBuildError, match="missing node columns"):
            build_adjacency_from_dfc(dfc)

    def test_duplicate_names_raise(self):
        dfc = _synthetic_dfc()
        dfc.loc[4, "Name"] = "C1"

        with pytest.raises(GraphBuildError, match="Duplicate conduit names"):
            build_adjacency_from_dfc(dfc)

    def test_conduit_order_with_unknown_name_raises(self):
        with pytest.raises(GraphBuildError, match="missing from conduits_data"):
            build_adjacency_from_dfc(_synthetic_dfc(), conduit_order=["C1", "C2", "C3", "C4", "C9"])


def _disconnected_dfc() -> pd.DataFrame:
    """Five conduits sharing no node: five weak components, zero edges.

    ``X1`` is present only because the spatial fallback refuses to run with
    fewer than two of its expected spatial columns, even though it uses
    ``Length`` alone.
    """
    return pd.DataFrame(
        {
            "Name": ["C1", "C2", "C3", "C4", "C5"],
            "InletNode": ["A1", "A2", "A3", "A4", "A5"],
            "OutletNode": ["B1", "B2", "B3", "B4", "B5"],
            "Length": [100.0, 105.0, 110.0, 300.0, 900.0],
            "X1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Diameter": [0.3, 0.3, 0.3, 0.5, 1.2],
            "Roughness": [0.013, 0.013, 0.013, 0.013, 0.013],
        }
    )


class TestSurrogateEdges:
    """Substitute edges are not hydraulics: off unless asked for, counted when on."""

    def test_disconnected_graph_stays_empty_by_default(self):
        result = build_graph(_disconnected_dfc())

        assert result.n_edges_raw == 0
        assert result.n_surrogate_edges == 0
        assert result.adjacency.nnz == 0

    def test_enabling_surrogate_edges_adds_and_reports_them(self):
        result = build_graph(_disconnected_dfc(), surrogate_edges=True)

        assert result.n_edges_raw == 0
        # Frozen observed value: C1-C2 and C2-C3 are within the 20% length
        # tolerance (2 spatial edges), then the feature fallback fires because
        # fewer than 10 were added and contributes 4 more.
        assert result.n_surrogate_edges == 6
        assert result.adjacency.nnz == result.n_surrogate_edges
        assert set(np.unique(result.adjacency.data)) == {np.float32(0.5), np.float32(0.3)}

    def test_a_connected_graph_never_gets_surrogate_edges(self):
        connected = _synthetic_dfc().assign(Length=[100.0, 105.0, 110.0, 115.0, 120.0], X1=[0.0, 1.0, 2.0, 3.0, 4.0])
        result = build_graph(connected, surrogate_edges=True)

        assert result.n_surrogate_edges == 0
        assert result.adjacency.nnz == result.n_edges_raw


class TestBothChannelsFromOneBuild:
    """``direction="both"`` of plan16 §3 is a pair of matrices, not a third one.

    The downstream channel has to be the matrix the pipeline has always built —
    otherwise a two-channel arm and the one-channel arms of E6 would not be
    comparable — and the upstream channel has to be exactly its transpose, which
    is what ``direction="upstream"`` has always returned.
    """

    def test_the_downstream_channel_is_what_build_graph_has_always_returned(self):
        dfc = _synthetic_dfc()

        channels = build_graph_channels(dfc)
        historical = build_graph(dfc, direction="downstream")

        assert np.array_equal(channels.downstream.toarray(), historical.adjacency.toarray())

    def test_the_upstream_channel_is_what_the_transposing_direction_returned(self):
        dfc = _synthetic_dfc()

        channels = build_graph_channels(dfc)
        historical = build_graph(dfc, direction="upstream")

        assert np.array_equal(channels.upstream.toarray(), historical.adjacency.toarray())

    def test_the_edges_are_counted_once_because_transposing_moves_none(self):
        channels = build_graph_channels(_synthetic_dfc())

        assert channels.n_edges_raw == build_graph(_synthetic_dfc()).n_edges_raw
        assert channels.n_surrogate_edges == 0

    def test_the_order_is_shared_by_both_channels(self):
        order = ["C5", "C1", "C3", "C2", "C4"]

        channels = build_graph_channels(_synthetic_dfc(), conduit_order=order)

        assert channels.order == tuple(order)
        assert channels.downstream.shape == channels.upstream.shape

    def test_a_single_direction_can_be_asked_for_by_name(self):
        channels = build_graph_channels(_synthetic_dfc())

        assert channels.channel("downstream") is channels.downstream
        assert channels.channel("upstream") is channels.upstream

    def test_asking_for_both_at_once_is_refused(self):
        """A pair answered with one matrix is the merge the plan forbids."""
        channels = build_graph_channels(_synthetic_dfc())

        with pytest.raises(ValueError, match="answers a single direction"):
            channels.channel("both")

    def test_an_isolated_conduit_has_no_edge_in_either_channel(self):
        dfc = pd.DataFrame({"Name": ["C0", "C1"], "InletNode": ["J0", "K0"], "OutletNode": ["J1", "K1"]})

        channels = build_graph_channels(dfc)

        assert channels.n_edges_raw == 0
        assert channels.downstream.nnz == 0 and channels.upstream.nnz == 0

    def test_a_confluence_gives_the_shared_conduit_two_upstream_neighbours(self):
        dfc = pd.DataFrame({"Name": ["C0", "C1", "C2"], "InletNode": ["J0", "K0", "J1"], "OutletNode": ["J1", "J1", "J2"]})

        dense = build_graph_channels(dfc).downstream.toarray()

        assert dense.sum(axis=0).tolist() == [0.0, 0.0, 2.0]
        assert dense.sum(axis=1).tolist() == [1.0, 1.0, 0.0]

    def test_a_fork_gives_the_stem_two_downstream_neighbours(self):
        dfc = pd.DataFrame({"Name": ["C0", "C1", "C2"], "InletNode": ["J0", "J1", "J1"], "OutletNode": ["J1", "J2", "J3"]})

        dense = build_graph_channels(dfc).downstream.toarray()

        assert dense.sum(axis=1).tolist() == [2.0, 0.0, 0.0]

    def test_two_parallel_conduits_between_the_same_nodes_stay_two_conduits(self):
        dfc = pd.DataFrame({"Name": ["C0", "C1", "C2"], "InletNode": ["J0", "J1", "J1"], "OutletNode": ["J1", "J2", "J2"]})

        channels = build_graph_channels(dfc)

        assert channels.n_edges_raw == 2
        assert len(channels.order) == 3
