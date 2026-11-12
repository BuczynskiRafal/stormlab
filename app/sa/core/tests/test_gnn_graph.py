"""Tests for sa.core.gnn.graph and sa.core.gnn.preprocessing (T1, T2).

No TensorFlow required: these modules are pure numpy/pandas/scipy.

T1 covers ``SWMMGraphConstructor`` / ``build_adjacency_from_dfc``: directed
edge construction from ``OutletNode(u) == InletNode(v)``, case/whitespace
insensitivity, zero raw diagonal, isolated conduits, and permutation
stability of ``conduit_order`` (D5).

T2 hand-verifies ``preprocess_adjacency`` on a 4-node chain: self-loops, the
1 / 0.5 / 0.25 / 0.125 hop-weighting scheme, diagonal restoration, and
row-normalization.
"""

import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sa.core.gnn import SWMMGraphConstructor, build_adjacency_from_dfc, preprocess_adjacency

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "gnn_golden_fixture.npz")


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
    """T1: build_adjacency_from_dfc (serving entry point, D5)."""

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

    def test_permutation_of_conduit_order_permutes_adjacency(self):
        """Stability w.r.t. conduit_order (D5): a permuted node order must
        yield exactly the permuted adjacency, not a re-derived/different
        graph."""
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
        A_direct, meta = constructor.build_conduit_graph()

        assert meta is None
        A_helper = build_adjacency_from_dfc(dfc)
        assert np.array_equal(A_direct.toarray(), A_helper.toarray())

    def test_conduit_to_idx_matches_row_order_when_order_not_given(self):
        dfc = _synthetic_dfc()
        constructor = SWMMGraphConstructor(dfc)
        constructor.build_conduit_graph()

        for i, name in enumerate(dfc["Name"]):
            assert constructor.conduit_to_idx[name] == i
            assert constructor.idx_to_conduit[i] == name


class TestRealTopologyReconstructionParity:
    """T1 extension: the serving-side graph reconstruction must reproduce the
    exact TRAINING adjacency for the real network topology (the core of the
    K1 fix - previously only covered transitively).

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
