"""One canonical order for the adjacency, the features and the result.

The scenario is the one plan11 §8 warns about: conduits named ``K1..K10``, where
the INP order (``K1, K2, ..., K10``) and the sorted order (``K1, K10, K2, ...``)
disagree on eight of ten positions. The decisive test runs a *graph-dependent*
computation — neighbour aggregation ``A @ X`` — under both schemes; a
prediction that ignored the adjacency would pass no matter how badly A and X
were misaligned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stormlab.graph.build import build_graph
from stormlab.graph.order import (
    OrderError,
    canonical_order,
    conduit_names,
    invert_permutation,
    permute_frames,
)

CONDUIT_COUNT = 10


@pytest.fixture
def chain_dfc() -> pd.DataFrame:
    """A ten-conduit chain whose names sort differently from their file order."""
    names = [f"K{i + 1}" for i in range(CONDUIT_COUNT)]
    return pd.DataFrame(
        {
            "Name": names,
            "InletNode": [f"N{i}" for i in range(CONDUIT_COUNT)],
            "OutletNode": [f"N{i + 1}" for i in range(CONDUIT_COUNT)],
            "Feature": [float(i + 1) for i in range(CONDUIT_COUNT)],
        }
    )


def _aggregate_over_neighbours(dfc: pd.DataFrame, scheme) -> pd.Series:
    """A prediction that only works when A and X share one row order."""
    order = canonical_order(dfc, scheme)
    frame = permute_frames(dfc, order)
    adjacency = build_graph(frame, conduit_order=order).adjacency
    features = frame["Feature"].to_numpy(dtype=float)
    aggregated = adjacency @ features + features
    return pd.Series(aggregated, index=pd.Index(order, name="Name"))


class TestCanonicalOrder:
    def test_inp_scheme_keeps_the_file_order(self, chain_dfc):
        assert canonical_order(chain_dfc, "inp") == list(chain_dfc["Name"])

    def test_sorted_scheme_puts_k10_second(self, chain_dfc):
        assert canonical_order(chain_dfc, "sorted")[:3] == ["K1", "K10", "K2"]

    def test_the_two_schemes_really_differ(self, chain_dfc):
        assert canonical_order(chain_dfc, "inp") != canonical_order(chain_dfc, "sorted")

    def test_names_fall_back_to_the_index(self, chain_dfc):
        without_name = chain_dfc.drop(columns=["Name"]).set_index(pd.Index(list(chain_dfc["Name"])))
        assert conduit_names(without_name) == list(chain_dfc["Name"])

    def test_unknown_scheme_raises(self, chain_dfc):
        with pytest.raises(ValueError, match="Unknown conduit order scheme"):
            canonical_order(chain_dfc, "by_length")  # type: ignore[arg-type]

    def test_duplicate_names_raise(self, chain_dfc):
        duplicated = chain_dfc.copy()
        duplicated.loc[9, "Name"] = "K1"
        with pytest.raises(OrderError, match="not unique"):
            canonical_order(duplicated, "inp")


class TestPermutationRoundTrip:
    def test_permuted_frame_follows_the_order(self, chain_dfc):
        order = canonical_order(chain_dfc, "sorted")
        permuted = permute_frames(chain_dfc, order)

        assert list(permuted.index) == order
        assert permuted.loc["K10", "Feature"] == 10.0

    def test_adjacency_and_features_stay_aligned_under_both_schemes(self, chain_dfc):
        """A and X built from the same order describe the same network."""
        inp_result = _aggregate_over_neighbours(chain_dfc, "inp")
        sorted_result = _aggregate_over_neighbours(chain_dfc, "sorted")

        inp_order = canonical_order(chain_dfc, "inp")
        restored = invert_permutation(sorted_result, inp_order)

        assert list(restored.index) == inp_order
        assert np.allclose(restored.to_numpy(), inp_result.to_numpy())

    def test_the_aggregation_actually_uses_the_graph(self, chain_dfc):
        """Guard for the test above: without the graph term it proves nothing."""
        result = _aggregate_over_neighbours(chain_dfc, "inp")
        # A[upstream, current] + row-wise A @ X means each conduit aggregates
        # over its DOWNSTREAM neighbour: K1 picks up K2, K10 has nobody.
        assert result["K1"] == pytest.approx(1.0 + 2.0)
        assert result["K10"] == pytest.approx(10.0)

    def test_misaligned_features_would_change_the_result(self, chain_dfc):
        """The alignment test has teeth: shuffling X alone moves the numbers."""
        order = canonical_order(chain_dfc, "sorted")
        adjacency = build_graph(permute_frames(chain_dfc, order), conduit_order=order).adjacency
        misaligned = chain_dfc["Feature"].to_numpy(dtype=float)  # still in INP order

        aligned = permute_frames(chain_dfc, order)["Feature"].to_numpy(dtype=float)
        assert not np.allclose(adjacency @ misaligned, adjacency @ aligned)

    def test_invert_permutation_rejects_a_mismatched_order(self, chain_dfc):
        result = _aggregate_over_neighbours(chain_dfc, "inp")
        with pytest.raises(OrderError, match="does not match the conduits present"):
            invert_permutation(result, [*canonical_order(chain_dfc, "inp")[:-1], "K99"])

    def test_permute_frames_rejects_a_short_order(self, chain_dfc):
        with pytest.raises(OrderError, match="does not match the conduits present"):
            permute_frames(chain_dfc, canonical_order(chain_dfc, "inp")[:-1])

    def test_invert_permutation_accepts_a_dataframe(self, chain_dfc):
        order = canonical_order(chain_dfc, "sorted")
        permuted = permute_frames(chain_dfc, order)
        restored = invert_permutation(permuted, canonical_order(chain_dfc, "inp"))

        assert isinstance(restored, pd.DataFrame)
        assert list(restored["Feature"]) == list(chain_dfc["Feature"])
