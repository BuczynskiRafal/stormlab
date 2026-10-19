"""The ``legacy_khop_v1`` scheme, pinned on the 11-conduit training chain.

The measurements frozen here come from ``docs/project-map/api_map.md`` §3
(gotcha 1) and are what makes the scheme reproducible: the hop profile is not
monotone, so it cannot be re-derived from the "1.0, 0.5, 0.25" comment the
legacy code carried.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from stormlab.graph.build import build_graph
from stormlab.graph.khop import KHOP_SCHEMES, apply_khop, preprocess_adjacency

CHAIN_LENGTH = 11


@pytest.fixture
def chain_adjacency() -> sp.csr_matrix:
    """The raw adjacency of an 11-conduit chain, as the training template has."""
    dfc = pd.DataFrame(
        {
            "Name": [f"K{i + 1}" for i in range(CHAIN_LENGTH)],
            "InletNode": [f"N{i}" for i in range(CHAIN_LENGTH)],
            "OutletNode": [f"N{i + 1}" for i in range(CHAIN_LENGTH)],
        }
    )
    result = build_graph(dfc)
    assert result.n_edges_raw == CHAIN_LENGTH - 1
    return result.adjacency


class TestGoldenHopProfile:
    def test_first_row_before_normalization_matches_the_measured_profile(self, chain_adjacency):
        """api_map §3 gotcha 1: the 1-hop weight is 3.25, above the 2-hop 2.0."""
        unnormalized = preprocess_adjacency(chain_adjacency, normalize=False, max_hops=4).toarray()

        expected_row = np.array([1.0, 3.25, 2.0, 0.75, 0.125] + [0.0] * (CHAIN_LENGTH - 5))
        assert np.allclose(unnormalized[0], expected_row, atol=1e-6)

    def test_the_profile_is_not_monotone_in_the_hop_distance(self, chain_adjacency):
        unnormalized = preprocess_adjacency(chain_adjacency, normalize=False, max_hops=4).toarray()
        assert unnormalized[0, 1] > unnormalized[0, 2] > unnormalized[0, 3]
        assert unnormalized[0, 1] > 1.0  # the self-loop weight, i.e. the diagonal

    def test_the_diagonal_is_overwritten_with_the_self_loop_value(self, chain_adjacency):
        unnormalized = preprocess_adjacency(chain_adjacency, normalize=False, max_hops=4).toarray()
        # Without the setdiag step the diagonal would carry 1.875, not 1.0.
        assert np.allclose(np.diag(unnormalized), 1.0)

    def test_without_self_loops_the_diagonal_is_reset_to_zero(self, chain_adjacency):
        unnormalized = preprocess_adjacency(chain_adjacency, add_self_loops=False, normalize=False, max_hops=4).toarray()
        assert np.allclose(np.diag(unnormalized), 0.0)

    def test_rows_sum_to_one_after_normalization(self, chain_adjacency):
        normalized = preprocess_adjacency(chain_adjacency, max_hops=4).toarray()
        assert np.allclose(normalized.sum(axis=1), 1.0)


class TestEdgeCounts:
    @pytest.mark.parametrize(("max_hops", "expected"), [(1, 21), (4, 45)])
    def test_measured_nnz_of_the_eleven_conduit_chain(self, chain_adjacency, max_hops, expected):
        """10 raw edges become 21 at one hop and 45 at four (api_map §3)."""
        assert apply_khop(chain_adjacency, max_hops=max_hops).n_edges_khop == expected

    def test_the_khop_count_includes_the_diagonal(self, chain_adjacency):
        result = apply_khop(chain_adjacency, max_hops=1)
        assert chain_adjacency.nnz == 10
        assert result.n_edges_khop == 10 + CHAIN_LENGTH

    def test_the_returned_matrix_is_csr(self, chain_adjacency):
        result = apply_khop(chain_adjacency, max_hops=4)
        assert sp.issparse(result.adjacency)
        assert result.adjacency.format == "csr"

    def test_preprocessing_widens_float32_to_float64(self, chain_adjacency):
        """``sp.eye`` is float64, so ``A + I`` promotes; verbatim legacy behaviour."""
        assert chain_adjacency.dtype == np.float32
        assert apply_khop(chain_adjacency, max_hops=4).adjacency.dtype == np.float64


class TestSchemeValidation:
    def test_the_only_known_scheme_is_the_legacy_one(self):
        assert KHOP_SCHEMES == ("legacy_khop_v1",)

    def test_unknown_scheme_raises(self, chain_adjacency):
        with pytest.raises(ValueError, match="Unknown khop_scheme"):
            apply_khop(chain_adjacency, khop_scheme="binomial_v2", max_hops=2)

    def test_non_positive_max_hops_raises(self, chain_adjacency):
        with pytest.raises(ValueError, match="max_hops must be at least 1"):
            apply_khop(chain_adjacency, max_hops=0)

    def test_apply_khop_agrees_with_the_raw_function(self, chain_adjacency):
        expected = preprocess_adjacency(chain_adjacency, max_hops=3).toarray()
        assert np.allclose(apply_khop(chain_adjacency, max_hops=3).adjacency.toarray(), expected)
