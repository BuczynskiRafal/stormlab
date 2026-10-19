"""Neighbourhood aggregates: what they see, what they refuse to invent.

The aggregates exist so that E6 compares two architectures and not two amounts of
information, which makes three properties load-bearing. Two conduits with the same
local vector in different neighbourhoods must come out different, or the block
carries no context at all. A conduit with no neighbours must come out at zero and
not at a population average, or the block carries something its own network never
said. And the values must come from the measured columns, never from a label.

The fourth is a definition rather than a property: "within k hops" has to be the
same set the graph models see, so one test asserts the module's own expansion
against :mod:`stormlab.graph.khop` on a real chain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stormlab.contracts.features import (
    FEATURE_CONTRACT_V3_PHYSICAL,
    FeatureContractError,
    neighborhood_column_names,
    neighborhood_contract,
)
from stormlab.features import neighborhood
from stormlab.graph.build import build_graph_channels
from stormlab.graph.khop import preprocess_adjacency, support_mask

COLUMNS: tuple[str, ...] = ("max_full_depth_ratio", "slope_signed")
HOPS: tuple[int, ...] = (1, 2)


def _network(edges: dict[str, tuple[str, str]], values: dict[str, list[float]] | None = None) -> pd.DataFrame:
    """A conduit frame: ``name -> (inlet node, outlet node)``, in declaration order."""
    frame = pd.DataFrame(
        {
            "Name": list(edges),
            "conduit": list(edges),
            "InletNode": [inlet for inlet, _ in edges.values()],
            "OutletNode": [outlet for _, outlet in edges.values()],
        }
    )
    for column in COLUMNS:
        frame[column] = (values or {}).get(column, list(range(1, len(edges) + 1)))
    return frame


def _aggregate(
    frame: pd.DataFrame,
    hops: tuple[int, ...] = HOPS,
    graph_order: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregates of ``frame``, with the graph built in ``graph_order``.

    The default deliberately **reverses** the frame's own order. Building the graph
    from ``frame["conduit"]`` would make the row-to-matrix permutation the identity
    in every test, which is exactly the case that cannot detect a frame read in the
    wrong order.
    """
    order = graph_order if graph_order is not None else list(reversed(frame["conduit"].tolist()))
    channels = build_graph_channels(frame, conduit_order=order)
    return neighborhood.neighborhood_aggregates(frame, channels, columns=COLUMNS, hops=hops)


class TestTheGeneratedColumns:
    def test_the_block_is_three_statistics_plus_a_count_and_a_presence_per_direction_and_hop(self) -> None:
        names = neighborhood_column_names(COLUMNS, HOPS)

        assert len(names) == 2 * len(HOPS) * (len(COLUMNS) * 3 + 2)
        assert names[0] == "up1_max_full_depth_ratio_min"
        assert "up1_count" in names and "down2_present" in names

    def test_the_names_are_a_function_of_the_configured_lists_alone(self) -> None:
        assert neighborhood_column_names(COLUMNS, HOPS) == neighborhood_column_names(COLUMNS, HOPS)

    def test_a_repeated_column_or_hop_is_refused(self) -> None:
        with pytest.raises(FeatureContractError, match="distinct"):
            neighborhood_column_names(("a", "a"), HOPS)
        with pytest.raises(FeatureContractError, match="distinct"):
            neighborhood_column_names(COLUMNS, (1, 1))


class TestTheDerivedContract:
    def test_it_is_the_base_contract_followed_by_the_generated_block(self) -> None:
        derived = neighborhood_contract(FEATURE_CONTRACT_V3_PHYSICAL, columns=COLUMNS, hops=HOPS)

        assert derived.name == "v3_physical_nbr"
        assert derived.columns[: FEATURE_CONTRACT_V3_PHYSICAL.n_columns] == FEATURE_CONTRACT_V3_PHYSICAL.columns
        assert derived.columns[FEATURE_CONTRACT_V3_PHYSICAL.n_columns :] == neighborhood_column_names(COLUMNS, HOPS)

    def test_deriving_it_leaves_the_contract_it_derives_from_untouched(self) -> None:
        before = FEATURE_CONTRACT_V3_PHYSICAL.sha

        derived = neighborhood_contract(FEATURE_CONTRACT_V3_PHYSICAL, columns=COLUMNS, hops=HOPS)

        assert FEATURE_CONTRACT_V3_PHYSICAL.sha == before
        assert derived.sha != before

    def test_a_different_configured_list_is_a_different_contract(self) -> None:
        one = neighborhood_contract(FEATURE_CONTRACT_V3_PHYSICAL, columns=COLUMNS, hops=(1,))
        two = neighborhood_contract(FEATURE_CONTRACT_V3_PHYSICAL, columns=COLUMNS, hops=(1, 2))

        assert one.sha != two.sha

    def test_the_presence_flags_are_the_int8_block_and_the_rest_are_floats(self) -> None:
        derived = neighborhood_contract(FEATURE_CONTRACT_V3_PHYSICAL, columns=COLUMNS, hops=(1,))

        assert derived.dtypes_map["up1_present"] == "int8"
        assert derived.dtypes_map["up1_count"] == "float64"
        assert derived.dtypes_map["up1_slope_signed_mean"] == "float64"

    def test_a_block_that_would_shadow_a_base_column_is_refused(self) -> None:
        with pytest.raises(FeatureContractError, match="shadow"):
            neighborhood_contract(
                neighborhood_contract(FEATURE_CONTRACT_V3_PHYSICAL, columns=COLUMNS, hops=HOPS),
                columns=COLUMNS,
                hops=HOPS,
            )


class TestWhatTheNeighbourhoodSees:
    def test_the_same_local_vector_in_two_neighbourhoods_gives_two_different_rows(self) -> None:
        """A chain and a confluence, with the middle conduit measured identically in both."""
        chain = _network({"a": ("n0", "n1"), "b": ("n1", "n2")}, values={column: [5.0, 5.0] for column in COLUMNS})
        confluence = _network(
            {"a": ("n0", "n1"), "x": ("n9", "n1"), "b": ("n1", "n2")},
            values={column: [5.0, 1.0, 5.0] for column in COLUMNS},
        )

        chain_row = _aggregate(chain).loc[chain["conduit"] == "b"].iloc[0]
        confluence_row = _aggregate(confluence).loc[confluence["conduit"] == "b"].iloc[0]

        assert chain_row["up1_max_full_depth_ratio_min"] == 5.0
        assert confluence_row["up1_max_full_depth_ratio_min"] == 1.0
        assert chain_row["up1_count"] == 1.0 and confluence_row["up1_count"] == 2.0

    def test_a_confluence_counts_both_inflows_upstream(self) -> None:
        frame = _network({"a": ("n0", "n1"), "x": ("n9", "n1"), "b": ("n1", "n2")})

        aggregates = _aggregate(frame)

        assert aggregates.loc[frame["conduit"] == "b", "up1_count"].iloc[0] == 2.0
        assert aggregates.loc[frame["conduit"] == "b", "down1_count"].iloc[0] == 0.0

    def test_an_isolated_conduit_gets_zeros_and_not_the_average_of_the_network(self) -> None:
        frame = _network({"a": ("n0", "n1"), "b": ("n1", "n2"), "lonely": ("z0", "z1")})

        row = _aggregate(frame).loc[frame["conduit"] == "lonely"].iloc[0]

        assert row["up1_present"] == 0 and row["down1_present"] == 0
        assert row["up1_count"] == 0.0
        assert row["up1_max_full_depth_ratio_mean"] == neighborhood.EMPTY_AGGREGATE
        assert row["down2_slope_signed_max"] == neighborhood.EMPTY_AGGREGATE

    def test_the_two_hop_neighbourhood_holds_everything_within_two_hops(self) -> None:
        frame = _network(
            {"a": ("n0", "n1"), "b": ("n1", "n2"), "c": ("n2", "n3")},
            values={column: [1.0, 2.0, 3.0] for column in COLUMNS},
        )

        row = _aggregate(frame).loc[frame["conduit"] == "c"].iloc[0]

        assert row["up1_count"] == 1.0
        assert row["up2_count"] == 2.0
        assert row["up2_max_full_depth_ratio_min"] == 1.0
        assert row["up2_max_full_depth_ratio_mean"] == pytest.approx(1.5)

    def test_the_row_order_of_the_frame_does_not_move_the_assignment(self) -> None:
        frame = _network(
            {"a": ("n0", "n1"), "b": ("n1", "n2"), "c": ("n2", "n3")},
            values={column: [1.0, 2.0, 3.0] for column in COLUMNS},
        )
        shuffled = frame.iloc[[2, 0, 1]].reset_index(drop=True)

        straight = _aggregate(frame).set_index(frame["conduit"])
        permuted = _aggregate(shuffled).set_index(shuffled["conduit"])

        pd.testing.assert_frame_equal(straight.sort_index(), permuted.sort_index())

    def test_the_aggregates_are_taken_from_the_measurements_and_not_from_the_labels(self) -> None:
        frame = _network({"a": ("n0", "n1"), "b": ("n1", "n2")})
        frame["label"] = ["valid", "valid"]
        relabelled = frame.assign(label=["pump", "diameter_increase"])

        pd.testing.assert_frame_equal(_aggregate(frame), _aggregate(relabelled))


class TestTheMissingValues:
    def test_one_neighbour_of_nan_does_not_erase_the_column_for_the_others(self) -> None:
        frame = _network(
            {"a": ("n0", "n1"), "x": ("n9", "n1"), "b": ("n1", "n2")},
            values={"max_full_depth_ratio": [np.nan, 4.0, 9.0], "slope_signed": [0.1, 0.2, 0.3]},
        )

        row = _aggregate(frame).loc[frame["conduit"] == "b"].iloc[0]

        assert row["up1_count"] == 2.0
        assert row["up1_max_full_depth_ratio_min"] == 4.0
        assert row["up1_max_full_depth_ratio_mean"] == pytest.approx(4.0)

    def test_a_neighbourhood_whose_values_are_all_missing_reads_as_empty_not_as_nan(self) -> None:
        frame = _network(
            {"a": ("n0", "n1"), "b": ("n1", "n2")},
            values={"max_full_depth_ratio": [np.nan, 9.0], "slope_signed": [0.1, 0.2]},
        )

        row = _aggregate(frame).loc[frame["conduit"] == "b"].iloc[0]

        assert row["up1_count"] == 1.0
        assert row["up1_present"] == 1
        assert row["up1_max_full_depth_ratio_mean"] == neighborhood.EMPTY_AGGREGATE
        assert row["up1_slope_signed_mean"] == pytest.approx(0.1)


class TestTheHopExpansion:
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_it_is_the_same_set_the_graph_models_call_the_k_hop_support(self, k: int) -> None:
        frame = _network({name: (f"n{index}", f"n{index + 1}") for index, name in enumerate("abcdef")})
        channels = build_graph_channels(frame, conduit_order=frame["conduit"].tolist())

        mine = neighborhood.within_k_hops(channels.downstream, k)
        theirs = support_mask(
            preprocess_adjacency(channels.downstream, add_self_loops=True, normalize=False, max_hops=k)
        ).toarray()
        np.fill_diagonal(theirs, 0.0)

        np.testing.assert_array_equal(mine.toarray(), theirs)

    def test_a_zero_hop_neighbourhood_is_refused(self) -> None:
        frame = _network({"a": ("n0", "n1"), "b": ("n1", "n2")})
        channels = build_graph_channels(frame, conduit_order=frame["conduit"].tolist())

        with pytest.raises(neighborhood.NeighborhoodError, match="at least one hop"):
            neighborhood.within_k_hops(channels.downstream, 0)


class TestWhatTheAggregationRefuses:
    def test_a_column_the_frame_does_not_carry_is_named(self) -> None:
        frame = _network({"a": ("n0", "n1"), "b": ("n1", "n2")})
        channels = build_graph_channels(frame, conduit_order=frame["conduit"].tolist())

        with pytest.raises(neighborhood.NeighborhoodError, match="hours_capacity_limited"):
            neighborhood.neighborhood_aggregates(frame, channels, columns=("hours_capacity_limited",), hops=(1,))

    def test_a_frame_that_does_not_cover_the_graph_is_refused(self) -> None:
        frame = _network({"a": ("n0", "n1"), "b": ("n1", "n2")})
        channels = build_graph_channels(frame, conduit_order=frame["conduit"].tolist())

        with pytest.raises(neighborhood.NeighborhoodError, match="the frame does not"):
            neighborhood.neighborhood_aggregates(frame.iloc[:1], channels, columns=COLUMNS, hops=(1,))


class TestTheRowOrderOfTheFrameAgainstTheGraph:
    """The frame's rows and the graph's rows need not be in the same order.

    They are not, in general: the graph is built once per network from the
    topology, the frame arrives from whatever join produced it. Reading the values
    in the frame's order while indexing them with the graph's would take every
    neighbour's value off the wrong conduit -- and the reindex at the end, which
    happens after the reductions, could not undo it.
    """

    NETWORK: dict[str, tuple[str, str]] = {"a": ("n0", "n1"), "b": ("n1", "n2"), "c": ("n2", "n3")}
    VALUES: dict[str, list[float]] = {"max_full_depth_ratio": [1.0, 2.0, 3.0], "slope_signed": [0.1, 0.2, 0.3]}

    def test_the_answer_does_not_depend_on_the_order_the_graph_was_built_in(self) -> None:
        frame = _network(self.NETWORK, values=self.VALUES)

        in_frame_order = _aggregate(frame, graph_order=["a", "b", "c"])
        reversed_graph = _aggregate(frame, graph_order=["c", "b", "a"])
        permuted_graph = _aggregate(frame, graph_order=["b", "c", "a"])

        pd.testing.assert_frame_equal(in_frame_order, reversed_graph)
        pd.testing.assert_frame_equal(in_frame_order, permuted_graph)

    def test_the_values_are_read_off_the_conduit_they_belong_to(self) -> None:
        """``c`` is one hop below ``b``, whatever order either side happens to use."""
        frame = _network(self.NETWORK, values=self.VALUES)

        row = _aggregate(frame, graph_order=["c", "a", "b"]).loc[frame["conduit"] == "c"].iloc[0]

        assert row["up1_max_full_depth_ratio_min"] == 2.0
        assert row["up2_max_full_depth_ratio_min"] == 1.0
        assert row["up1_slope_signed_mean"] == pytest.approx(0.2)

    def test_a_permuted_frame_against_a_fixed_graph_gives_the_same_rows(self) -> None:
        frame = _network(self.NETWORK, values=self.VALUES)
        shuffled = frame.iloc[[1, 2, 0]].reset_index(drop=True)

        straight = _aggregate(frame, graph_order=["a", "b", "c"]).set_index(frame["conduit"])
        permuted = _aggregate(shuffled, graph_order=["a", "b", "c"]).set_index(shuffled["conduit"])

        pd.testing.assert_frame_equal(straight.sort_index(), permuted.sort_index())


class TestTheLastConduitOfTheGraph:
    """``reduceat`` reads segment ``[starts[i], starts[i + 1])``.

    A conduit whose neighbours run to the end of the CSR index array, followed by
    conduits with none, produces a start equal to ``nnz`` -- and clamping that into
    range shortens the *earlier* conduit's segment, dropping its last neighbour
    from the minimum and the maximum with no sign that it happened. The network
    below is the smallest one that produces it: a divergence in graph row 0, and
    nothing downstream of the rows after it.
    """

    #: `s` splits into `p` and `q`; in graph order [s, p, q] the CSR is
    #: indptr = [0, 2, 2, 2], so rows 1 and 2 both start at nnz.
    NETWORK: dict[str, tuple[str, str]] = {"s": ("n0", "n1"), "p": ("n1", "n2"), "q": ("n1", "n3")}
    VALUES: dict[str, list[float]] = {"max_full_depth_ratio": [1.0, 5.0, 99.0], "slope_signed": [0.1, 0.5, 0.9]}

    def test_the_last_neighbour_of_the_first_row_still_reaches_the_extremes(self) -> None:
        frame = _network(self.NETWORK, values=self.VALUES)

        row = _aggregate(frame, graph_order=["s", "p", "q"]).loc[frame["conduit"] == "s"].iloc[0]

        assert row["down1_count"] == 2.0
        assert row["down1_max_full_depth_ratio_min"] == 5.0
        assert row["down1_max_full_depth_ratio_max"] == 99.0
        assert row["down1_max_full_depth_ratio_mean"] == pytest.approx(52.0)

    def test_the_trailing_rows_with_no_neighbours_read_as_empty(self) -> None:
        frame = _network(self.NETWORK, values=self.VALUES)

        aggregates = _aggregate(frame, graph_order=["s", "p", "q"])

        for conduit in ("p", "q"):
            row = aggregates.loc[frame["conduit"] == conduit].iloc[0]
            assert row["down1_count"] == 0.0
            assert row["down1_present"] == 0
            assert row["down1_max_full_depth_ratio_max"] == neighborhood.EMPTY_AGGREGATE


class TestTheConfiguredNeighbourhood:
    """The declaration in ``config.C1.yaml`` and the block it generates are one thing.

    The point of reading the columns from the protocol rather than from a literal
    in this module is that the derived contract is then a function of something
    frozen. These tests are what makes the two halves of that sentence meet.
    """

    @pytest.fixture(scope="class")
    def declared(self):
        from stormlab.config import ConfigError, load_config

        try:
            return load_config(variant="C1").c1
        except ConfigError as exc:
            pytest.skip(f"C1 config overlay not present in standalone checkout: {exc}")

    def test_the_protocol_declares_the_columns_and_the_hops(self, declared) -> None:
        assert declared is not None
        assert declared.neighborhood.hops == (1, 2)
        assert declared.neighborhood.columns == (
            "max_full_depth_ratio",
            "slope_signed",
            "max_flow_cms",
            "hours_capacity_limited",
        )

    def test_every_declared_column_is_a_measurement_of_the_physical_view(self, declared) -> None:
        assert set(declared.neighborhood.columns) <= set(FEATURE_CONTRACT_V3_PHYSICAL.columns)

    def test_no_declared_column_is_a_mask_or_a_margin(self, declared) -> None:
        """An aggregate over a neighbour's flags summarises what was decided about it."""
        from stormlab.contracts.features import PHYSICAL_FLAG_COLUMNS

        assert not set(declared.neighborhood.columns) & set(PHYSICAL_FLAG_COLUMNS)

    def test_the_declaration_generates_the_block_of_the_derived_contract(self, declared) -> None:
        derived = neighborhood_contract(
            FEATURE_CONTRACT_V3_PHYSICAL,
            columns=declared.neighborhood.columns,
            hops=declared.neighborhood.hops,
        )

        assert derived.n_columns == FEATURE_CONTRACT_V3_PHYSICAL.n_columns + 2 * 2 * (4 * 3 + 2)

    def test_the_protocol_carries_the_section_into_the_frozen_payload(self, declared) -> None:
        assert declared.as_dict()["neighborhood"] == {
            "columns": list(declared.neighborhood.columns),
            "hops": [1, 2],
        }


class TestAggregatingASubsetOfTheDeclaredColumns:
    """The ``noflood`` arm reads three of the four columns C2 declares, and nothing else.

    plan17 §Etap 1b builds that arm as a **subset**: no column takes the place of the one
    removed. What makes that free is the property asserted here -- the aggregates are
    computed per column independently, so the three that stay hold exactly the values
    they hold when the fourth is aggregated beside them.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        frame = _network({"C1": ("J1", "J2"), "C2": ("J2", "J3"), "C3": ("J3", "J4")})
        frame["max_flow_cms"] = [0.11, 0.22, 0.33]
        frame["inlet_node_flood_hours"] = [0.0, 0.01, 2.5]
        return frame

    def test_the_kept_columns_hold_the_values_they_hold_in_the_full_arm(self) -> None:
        # Arrange
        frame = self._frame()
        full = ("max_full_depth_ratio", "slope_signed", "max_flow_cms", "inlet_node_flood_hours")
        subset = full[:3]
        channels = build_graph_channels(frame, conduit_order=list(reversed(frame["conduit"].tolist())))

        # Act
        wide = neighborhood.neighborhood_aggregates(frame, channels, columns=list(full), hops=list(HOPS))
        narrow = neighborhood.neighborhood_aggregates(frame, channels, columns=list(subset), hops=list(HOPS))

        # Assert
        assert list(narrow.columns) == list(neighborhood_column_names(subset, HOPS))
        pd.testing.assert_frame_equal(narrow, wide[list(narrow.columns)])

    def test_the_narrow_block_names_no_column_of_the_one_it_dropped(self) -> None:
        """A replacement column would let the arm read off its neighbours what it may not read."""
        frame = self._frame()
        channels = build_graph_channels(frame, conduit_order=list(frame["conduit"]))

        narrow = neighborhood.neighborhood_aggregates(
            frame, channels, columns=["max_full_depth_ratio", "slope_signed", "max_flow_cms"], hops=list(HOPS)
        )

        assert not [name for name in narrow.columns if "inlet_node_flood_hours" in name]
        assert len(narrow.columns) == 44

    def test_the_counts_and_the_presence_flags_are_the_graphs_and_not_the_columns(self) -> None:
        """They describe the neighbourhood itself, so dropping a measurement cannot move them."""
        frame = self._frame()
        channels = build_graph_channels(frame, conduit_order=list(frame["conduit"]))
        counts = [f"{direction}{hop}_{suffix}" for direction in ("up", "down") for hop in HOPS for suffix in ("count", "present")]

        wide = neighborhood.neighborhood_aggregates(
            frame,
            channels,
            columns=["max_full_depth_ratio", "slope_signed", "max_flow_cms", "inlet_node_flood_hours"],
            hops=list(HOPS),
        )
        narrow = neighborhood.neighborhood_aggregates(
            frame, channels, columns=["max_full_depth_ratio", "slope_signed", "max_flow_cms"], hops=list(HOPS)
        )

        pd.testing.assert_frame_equal(narrow[counts], wide[counts])


class TestTheConfiguredC2Neighbourhood:
    """The C2 overlay declares the same section with the other overload premise in it."""

    @pytest.fixture(scope="class")
    def declared(self):
        from stormlab.config import ConfigError, load_config

        try:
            return load_config(variant="C2").c1
        except ConfigError as exc:
            pytest.skip(f"C2 config overlay not present in standalone checkout: {exc}")

    def test_the_protocol_declares_c2s_own_four_columns(self, declared) -> None:
        assert declared is not None
        assert declared.neighborhood.hops == (1, 2)
        assert declared.neighborhood.columns == (
            "max_full_depth_ratio",
            "slope_signed",
            "max_flow_cms",
            "inlet_node_flood_hours",
        )

    def test_every_declared_column_is_a_measurement_of_the_view_the_protocol_names(self, declared) -> None:
        from stormlab.contracts.features import FEATURE_CONTRACT_V4_PHYSICAL, PHYSICAL_FLAG_COLUMNS

        assert set(declared.neighborhood.columns) <= set(FEATURE_CONTRACT_V4_PHYSICAL.columns)
        assert not set(declared.neighborhood.columns) & set(PHYSICAL_FLAG_COLUMNS)

    def test_the_declaration_generates_the_block_of_the_contract_it_names(self, declared) -> None:
        from stormlab.contracts.features import FEATURE_CONTRACT_V4_PHYSICAL, feature_contract

        derived = neighborhood_contract(
            FEATURE_CONTRACT_V4_PHYSICAL,
            columns=declared.neighborhood.columns,
            hops=declared.neighborhood.hops,
            name=declared.contracts.f2_nbr,
        )

        assert derived.columns == feature_contract(declared.contracts.f2_nbr).columns
        assert derived.sha == feature_contract(declared.contracts.f2_nbr).sha
