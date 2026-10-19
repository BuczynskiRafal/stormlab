"""The 33-column conduit contract: order, dtypes and refusal to repair a frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stormlab.contracts import features as contract
from stormlab.contracts.catchment_features import CLASSIFIER_CLASS_ORDER
from stormlab.contracts.features import (
    FEATURE_CONTRACT_V1,
    FEATURE_CONTRACT_V2,
    FEATURE_CONTRACT_V2_NOFLAGS,
    FEATURE_CONTRACT_V3_MARGINS,
    FEATURE_CONTRACT_V3_PHYSICAL,
    FEATURE_CONTRACT_V3_PHYSICAL_NBR,
    FEATURE_CONTRACT_V4_MARGINS,
    FEATURE_CONTRACT_V4_NOFLOOD_MARGINS,
    FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL,
    FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL_NBR,
    FEATURE_CONTRACT_V4_PHYSICAL,
    FEATURE_CONTRACT_V4_PHYSICAL_NBR,
    FEATURE_COLUMNS,
    NODE_FLOOD_COLUMNS,
    PHYSICAL_FREEBOARD_COLUMNS,
    NEIGHBORHOOD_BLOCK_COLUMNS,
    NEIGHBORHOOD_COLUMNS,
    NEIGHBORHOOD_COLUMNS_V4,
    NEIGHBORHOOD_COLUMNS_V4_NOFLOOD,
    NEIGHBORHOOD_CONTRACT_COLUMNS,
    NEIGHBORHOOD_HOPS,
    FEATURE_ONEHOT_ORDER,
    FLAG_COLUMNS,
    FLOAT_COLUMNS,
    FeatureContractError,
    class_order,
    validate_features,
)


def _valid_frame(rows: int = 3) -> pd.DataFrame:
    data = {column: np.zeros(rows, dtype=np.int8) for column in FLAG_COLUMNS}
    data.update({column: np.zeros(rows, dtype=np.float64) for column in FLOAT_COLUMNS})
    return pd.DataFrame(data)[list(FEATURE_COLUMNS)]


def test_contract_declares_exactly_thirty_three_unique_columns() -> None:
    assert len(FEATURE_COLUMNS) == contract.EXPECTED_N_FEATURES
    assert len(set(FEATURE_COLUMNS)) == contract.EXPECTED_N_FEATURES


def test_flag_and_float_blocks_partition_the_contract() -> None:
    assert len(FLAG_COLUMNS) == 12
    assert len(FLOAT_COLUMNS) == 21
    assert set(FLAG_COLUMNS) | set(FLOAT_COLUMNS) == set(FEATURE_COLUMNS)
    assert not set(FLAG_COLUMNS) & set(FLOAT_COLUMNS)


def test_onehot_block_is_the_last_twelve_columns_of_the_contract() -> None:
    assert FEATURE_COLUMNS[-12:] == FEATURE_ONEHOT_ORDER


def test_feature_onehot_order_differs_from_the_classifier_order() -> None:
    assert set(FEATURE_ONEHOT_ORDER) == set(CLASSIFIER_CLASS_ORDER)
    assert FEATURE_ONEHOT_ORDER != CLASSIFIER_CLASS_ORDER


def test_class_order_helper_returns_named_orders_and_rejects_unknown_names() -> None:
    assert class_order("feature_onehot") == FEATURE_ONEHOT_ORDER
    assert class_order("recommendation")[0] == "pump"
    with pytest.raises(FeatureContractError, match="Unknown class order"):
        class_order("alphabetical")


def test_validate_features_accepts_a_frame_matching_the_contract() -> None:
    validate_features(_valid_frame())


def test_validate_features_rejects_a_missing_column() -> None:
    frame = _valid_frame().drop(columns=["NSlope"])
    with pytest.raises(FeatureContractError, match="Missing feature columns"):
        validate_features(frame)


def test_validate_features_rejects_reordered_columns() -> None:
    frame = _valid_frame()
    reordered = frame[list(FEATURE_COLUMNS[1:]) + [FEATURE_COLUMNS[0]]]
    with pytest.raises(FeatureContractError, match="out of contract order"):
        validate_features(reordered)


def test_validate_features_rejects_a_flag_stored_as_float() -> None:
    frame = _valid_frame()
    frame["ValMaxFill"] = frame["ValMaxFill"].astype("float64")
    with pytest.raises(FeatureContractError, match="Wrong dtypes"):
        validate_features(frame)


def test_validate_features_rejects_a_onehot_column_stored_as_int() -> None:
    frame = _valid_frame()
    frame["rural"] = frame["rural"].astype("int8")
    with pytest.raises(FeatureContractError, match="Wrong dtypes"):
        validate_features(frame)


def test_validate_features_rejects_nan_instead_of_filling_it() -> None:
    frame = _valid_frame()
    frame.loc[0, "NFilling"] = np.nan
    with pytest.raises(FeatureContractError, match="Missing values"):
        validate_features(frame)


def test_validate_features_rejects_a_non_dataframe() -> None:
    with pytest.raises(FeatureContractError, match="Expected a DataFrame"):
        validate_features(np.zeros((3, 33)))  # type: ignore[arg-type]


def test_the_feature_contract_digest_is_frozen() -> None:
    """The digest models are pinned by; changing the 33 columns must change it, and only that."""
    from stormlab.contracts.features import feature_contract_sha

    assert feature_contract_sha() == "fc570502f408b4b88f406eac8070a8d2f728e07cb0181a9369f11d51b98f66b1"


def test_the_feature_contract_digest_is_the_hash_of_the_joined_column_names() -> None:
    import hashlib

    from stormlab.contracts.features import FEATURE_COLUMNS, feature_contract_sha

    expected = hashlib.sha256("\n".join(FEATURE_COLUMNS).encode("utf-8")).hexdigest()

    assert feature_contract_sha() == expected


def test_a_reordered_contract_would_produce_a_different_digest() -> None:
    """Order is part of the contract: the same names in another order is another model input."""
    import hashlib

    from stormlab.contracts.features import FEATURE_COLUMNS, feature_contract_sha

    shuffled = "\n".join(reversed(FEATURE_COLUMNS)).encode("utf-8")

    assert hashlib.sha256(shuffled).hexdigest() != feature_contract_sha()


def test_feature_contract_v1_is_the_source_of_compatibility_aliases() -> None:
    from stormlab.contracts.features import DEFAULT_FEATURE_CONTRACT, EXPECTED_N_FEATURES, FEATURE_COLUMNS

    assert DEFAULT_FEATURE_CONTRACT.name == "v1"
    assert FEATURE_COLUMNS is DEFAULT_FEATURE_CONTRACT.columns
    assert EXPECTED_N_FEATURES == DEFAULT_FEATURE_CONTRACT.n_columns == 33
    assert len(DEFAULT_FEATURE_CONTRACT.sha) == 64


def test_feature_contract_v2_removes_only_the_two_redundant_cover_columns() -> None:
    assert FEATURE_CONTRACT_V2.n_columns == 31
    assert "NInletGroundCover" not in FEATURE_CONTRACT_V2.columns
    assert "NOutletGroundCover" not in FEATURE_CONTRACT_V2.columns
    assert tuple(column for column in FEATURE_COLUMNS if "GroundCover" not in column) == FEATURE_CONTRACT_V2.columns


def test_the_v1_and_v2_digests_are_pinned_literals_no_later_contract_may_move() -> None:
    """Adding a contract to the registry must not relabel a model already trained.

    The digest is over one contract's own ordered column names, so a third entry
    cannot reach the first two — but that is a property of the recipe, and the
    recipe is exactly what a later edit could change. Both literals are written
    out here so such an edit fails as a moved hash rather than as a mismatch
    discovered next to a saved weight file.
    """
    assert FEATURE_CONTRACT_V1.sha == "fc570502f408b4b88f406eac8070a8d2f728e07cb0181a9369f11d51b98f66b1"
    assert FEATURE_CONTRACT_V2.sha == "8184be657c6ba4a321f9932f5747c31fa79ebf473eee289265bebef71c19b3e2"


def test_feature_contract_v2_noflags_is_v2_without_the_twelve_flags() -> None:
    """The E1 arm: same columns, same relative order, twelve fewer inputs."""
    assert FEATURE_CONTRACT_V2_NOFLAGS.n_columns == 19
    assert set(FLAG_COLUMNS).isdisjoint(FEATURE_CONTRACT_V2_NOFLAGS.columns)
    assert FEATURE_CONTRACT_V2_NOFLAGS.columns == tuple(
        column for column in FEATURE_CONTRACT_V2.columns if column not in set(FLAG_COLUMNS)
    )
    assert FEATURE_CONTRACT_V2_NOFLAGS.columns[-12:] == FEATURE_ONEHOT_ORDER


def test_feature_contract_v2_noflags_declares_every_input_as_a_float() -> None:
    """The flags were the only ``int8`` block, so what is left is one float matrix."""
    assert set(FEATURE_CONTRACT_V2_NOFLAGS.dtypes_map.values()) == {contract.FLOAT_DTYPE}


def test_the_registry_resolves_every_contract_and_refuses_an_unknown_name() -> None:
    from stormlab.contracts.features import FEATURE_CONTRACTS, feature_contract

    assert sorted(FEATURE_CONTRACTS) == [
        "v1",
        "v2",
        "v2_noflags",
        "v3_margins",
        "v3_physical",
        "v3_physical_nbr",
        "v3_physical_nbr_nocat",
        "v3_physical_nocat",
        "v4_margins",
        "v4_noflood_margins",
        "v4_noflood_physical",
        "v4_noflood_physical_nbr",
        "v4_physical",
        "v4_physical_nbr",
        "v4_physical_nbr_nocat",
        "v4_physical_nocat",
    ]
    assert feature_contract("v2_noflags") is FEATURE_CONTRACT_V2_NOFLAGS
    assert feature_contract("v3_physical") is FEATURE_CONTRACT_V3_PHYSICAL
    with pytest.raises(FeatureContractError, match="Unknown feature contract"):
        feature_contract("v3")


def test_a_v2_frame_satisfies_v2_noflags_without_being_reordered() -> None:
    """A build writes v2; the ablation arm reads a projection of it, not a new build."""
    frame = _valid_frame()[list(FEATURE_CONTRACT_V2.columns)]

    validate_features(frame, contract=FEATURE_CONTRACT_V2_NOFLAGS)


def test_the_physical_contract_carries_the_measurements_the_masks_and_no_decision() -> None:
    """F2: geometry, the simulated state, the context, the requirements and the masks.

    The negative half matters as much as the positive one. Not a single validity
    flag and not a single proposed action is in the contract -- those are what
    plan16 §2 P3 removes from the model input -- and no column of v1/v2 is
    reachable from it, so the two views cannot be silently mixed.
    """
    from stormlab.contracts.features import (
        MARGIN_COLUMNS,
        PHYSICAL_CONTEXT_COLUMNS,
        PHYSICAL_GEOMETRY_COLUMNS,
        PHYSICAL_HYDRAULIC_COLUMNS,
        PHYSICAL_MASK_COLUMNS,
        PHYSICAL_RECEIVER_COLUMNS,
        PHYSICAL_REQUIREMENT_COLUMNS,
    )

    assert FEATURE_CONTRACT_V3_PHYSICAL.columns == (
        PHYSICAL_GEOMETRY_COLUMNS
        + PHYSICAL_HYDRAULIC_COLUMNS
        + PHYSICAL_CONTEXT_COLUMNS
        + PHYSICAL_REQUIREMENT_COLUMNS
        + PHYSICAL_RECEIVER_COLUMNS
        + PHYSICAL_MASK_COLUMNS
    )
    assert set(FEATURE_CONTRACT_V3_PHYSICAL.columns).isdisjoint(FEATURE_CONTRACT_V1.columns)
    assert set(FEATURE_CONTRACT_V3_PHYSICAL.columns).isdisjoint(MARGIN_COLUMNS)


def test_the_margin_contract_is_the_physical_one_plus_the_margins_in_that_order() -> None:
    from stormlab.contracts.features import MARGIN_COLUMNS, MARGIN_MASK_COLUMNS

    assert FEATURE_CONTRACT_V3_MARGINS.columns == FEATURE_CONTRACT_V3_PHYSICAL.columns + MARGIN_COLUMNS + MARGIN_MASK_COLUMNS
    assert FEATURE_CONTRACT_V3_MARGINS.n_columns == FEATURE_CONTRACT_V3_PHYSICAL.n_columns + 13


def test_the_v3_contracts_type_every_indicator_as_int8_and_everything_else_as_float() -> None:
    from stormlab.contracts.features import PHYSICAL_FLAG_COLUMNS

    for candidate in (FEATURE_CONTRACT_V3_PHYSICAL, FEATURE_CONTRACT_V3_MARGINS):
        flags = {column for column in candidate.columns if candidate.dtypes_map[column] == contract.FLAG_DTYPE}
        assert flags == set(PHYSICAL_FLAG_COLUMNS) & set(candidate.columns)
        assert set(candidate.dtypes_map.values()) == {contract.FLAG_DTYPE, contract.FLOAT_DTYPE}


def test_the_v3_contracts_declare_the_build_they_need() -> None:
    """``schema_version`` names a build, and no v2 parquet can answer these columns."""
    assert FEATURE_CONTRACT_V3_PHYSICAL.schema_version == 3
    assert FEATURE_CONTRACT_V3_MARGINS.schema_version == 3


def test_the_v3_digests_are_pinned_literals() -> None:
    """Two contracts frozen the day they were written, by the same v1 recipe."""
    assert FEATURE_CONTRACT_V3_PHYSICAL.sha == "6e6320bbf670b5af9053a65706fe7c3d0784e0c9c3d5f018d3aa5c70047668b5"
    assert FEATURE_CONTRACT_V3_MARGINS.sha == "f9f23105b6b786b0c30527ad2b554aa3528553b5d3e26dc1687d5c4c0a89be9d"


def test_adding_the_v3_contracts_left_every_earlier_digest_where_it_was() -> None:
    assert FEATURE_CONTRACT_V1.sha == "fc570502f408b4b88f406eac8070a8d2f728e07cb0181a9369f11d51b98f66b1"
    assert FEATURE_CONTRACT_V2.sha == "8184be657c6ba4a321f9932f5747c31fa79ebf473eee289265bebef71c19b3e2"
    assert FEATURE_CONTRACT_V2_NOFLAGS.sha == "ddc0b72fc23fae7a1db5872b07c957d8c87f2fbb87d0d465d134c7b3c5fb2c9c"


def test_the_neighbourhood_contract_is_the_physical_view_followed_by_its_block() -> None:
    """F2 + aggregates: the columns of ``v3_physical``, then the 56 generated ones."""
    base = FEATURE_CONTRACT_V3_PHYSICAL
    assert FEATURE_CONTRACT_V3_PHYSICAL_NBR.columns[: base.n_columns] == base.columns
    assert FEATURE_CONTRACT_V3_PHYSICAL_NBR.columns[base.n_columns :] == NEIGHBORHOOD_BLOCK_COLUMNS
    assert len(NEIGHBORHOOD_BLOCK_COLUMNS) == 2 * len(NEIGHBORHOOD_HOPS) * (len(NEIGHBORHOOD_COLUMNS) * 3 + 2) == 56
    assert FEATURE_CONTRACT_V3_PHYSICAL_NBR.n_columns == 120


def test_the_neighbourhood_block_holds_no_flag_no_margin_and_no_decision() -> None:
    """An aggregate over a neighbour's mask or margin summarises what was decided about it."""
    from stormlab.contracts.features import MARGIN_COLUMNS, PHYSICAL_FLAG_COLUMNS

    assert set(NEIGHBORHOOD_COLUMNS) <= set(FEATURE_CONTRACT_V3_PHYSICAL.columns)
    assert not set(NEIGHBORHOOD_COLUMNS) & set(PHYSICAL_FLAG_COLUMNS)
    assert not set(NEIGHBORHOOD_COLUMNS) & set(MARGIN_COLUMNS)
    assert not set(NEIGHBORHOOD_COLUMNS) & set(FLAG_COLUMNS)


def test_the_neighbourhood_digest_is_a_pinned_literal_and_moved_no_earlier_one() -> None:
    assert FEATURE_CONTRACT_V3_PHYSICAL_NBR.sha == "a23b20e526c3c4101b2caf861d2152fa7ed6755798be0dc083ae956c21acbd12"
    assert FEATURE_CONTRACT_V3_PHYSICAL.sha == "6e6320bbf670b5af9053a65706fe7c3d0784e0c9c3d5f018d3aa5c70047668b5"
    assert FEATURE_CONTRACT_V3_MARGINS.sha == "f9f23105b6b786b0c30527ad2b554aa3528553b5d3e26dc1687d5c4c0a89be9d"
    assert FEATURE_CONTRACT_V1.sha == "fc570502f408b4b88f406eac8070a8d2f728e07cb0181a9369f11d51b98f66b1"
    assert FEATURE_CONTRACT_V2.sha == "8184be657c6ba4a321f9932f5747c31fa79ebf473eee289265bebef71c19b3e2"
    assert FEATURE_CONTRACT_V2_NOFLAGS.sha == "ddc0b72fc23fae7a1db5872b07c957d8c87f2fbb87d0d465d134c7b3c5fb2c9c"


def test_the_nocat_contracts_are_their_bases_without_the_land_cover_category() -> None:
    """The category ablation removes two columns and leaves the rest in place, in order."""
    from stormlab.contracts.features import (
        FEATURE_CONTRACT_V3_PHYSICAL_NBR_NOCAT,
        FEATURE_CONTRACT_V3_PHYSICAL_NOCAT,
        LAND_COVER_CATEGORY_COLUMNS,
    )

    removed = set(LAND_COVER_CATEGORY_COLUMNS)
    assert removed == {"sbc_category_code", "sbc_missing"}
    assert FEATURE_CONTRACT_V3_PHYSICAL_NOCAT.columns == tuple(
        column for column in FEATURE_CONTRACT_V3_PHYSICAL.columns if column not in removed
    )
    assert FEATURE_CONTRACT_V3_PHYSICAL_NBR_NOCAT.columns == tuple(
        column for column in FEATURE_CONTRACT_V3_PHYSICAL_NBR.columns if column not in removed
    )
    assert FEATURE_CONTRACT_V3_PHYSICAL_NOCAT.n_columns == 62
    assert FEATURE_CONTRACT_V3_PHYSICAL_NBR_NOCAT.n_columns == 118


def test_the_nocat_contracts_keep_the_four_subcatchment_parameters() -> None:
    """What goes is the land cover *label*, not the catchment SWMM was run with.

    Area, imperviousness, width and slope are inputs of the simulation and stay: the
    question the ablation asks is what the twelve-class category contributed, and dropping
    the parameters with it would answer a different one.
    """
    from stormlab.contracts.features import FEATURE_CONTRACT_V3_PHYSICAL_NOCAT

    kept = set(FEATURE_CONTRACT_V3_PHYSICAL_NOCAT.columns)
    assert {"subcatchment_area_ha", "subcatchment_imperv_pct", "subcatchment_width_m", "subcatchment_slope_pct"} <= kept
    assert "subcatchment_missing" in kept


def test_the_nocat_digests_are_pinned_literals_and_moved_no_earlier_one() -> None:
    from stormlab.contracts.features import FEATURE_CONTRACT_V3_PHYSICAL_NBR_NOCAT, FEATURE_CONTRACT_V3_PHYSICAL_NOCAT

    assert FEATURE_CONTRACT_V3_PHYSICAL_NOCAT.sha == "31416687db6ecfede4c46607720701c696b027891b18ec251193c817b9214f16"
    assert FEATURE_CONTRACT_V3_PHYSICAL_NBR_NOCAT.sha == "ee4429fc9afb46451524aaeb86cee549b5bf2e2ab540886e4061e4b776e07c76"
    assert FEATURE_CONTRACT_V3_PHYSICAL.sha == "6e6320bbf670b5af9053a65706fe7c3d0784e0c9c3d5f018d3aa5c70047668b5"
    assert FEATURE_CONTRACT_V3_PHYSICAL_NBR.sha == "a23b20e526c3c4101b2caf861d2152fa7ed6755798be0dc083ae956c21acbd12"
    assert FEATURE_CONTRACT_V3_MARGINS.sha == "f9f23105b6b786b0c30527ad2b554aa3528553b5d3e26dc1687d5c4c0a89be9d"
    assert FEATURE_CONTRACT_V1.sha == "fc570502f408b4b88f406eac8070a8d2f728e07cb0181a9369f11d51b98f66b1"
    assert FEATURE_CONTRACT_V2.sha == "8184be657c6ba4a321f9932f5747c31fa79ebf473eee289265bebef71c19b3e2"
    assert FEATURE_CONTRACT_V2_NOFLAGS.sha == "ddc0b72fc23fae7a1db5872b07c957d8c87f2fbb87d0d465d134c7b3c5fb2c9c"


def test_the_nocat_neighbourhood_block_is_the_same_fifty_six_columns() -> None:
    """The aggregates are of measurements, so the ablation does not touch the block.

    Which is what lets one computed aggregate frame serve both contracts: if the block
    differed, the ablated arm would need its own pass over 1547 networks.
    """
    from stormlab.contracts.features import FEATURE_CONTRACT_V3_PHYSICAL_NBR_NOCAT, FEATURE_CONTRACT_V3_PHYSICAL_NOCAT

    base = FEATURE_CONTRACT_V3_PHYSICAL_NOCAT
    assert FEATURE_CONTRACT_V3_PHYSICAL_NBR_NOCAT.columns[base.n_columns :] == NEIGHBORHOOD_BLOCK_COLUMNS


def test_the_presence_flags_are_the_only_int8_column_of_the_neighbourhood_block() -> None:
    flags = {name for name in NEIGHBORHOOD_BLOCK_COLUMNS if name.endswith("_present")}

    dtypes = FEATURE_CONTRACT_V3_PHYSICAL_NBR.dtypes_map
    assert {name for name in NEIGHBORHOOD_BLOCK_COLUMNS if dtypes[name] == contract.FLAG_DTYPE} == flags
    assert len(flags) == 4


def test_the_declared_protocol_and_the_registered_contract_name_the_same_neighbourhood() -> None:
    """One source of truth: ``config.C1.yaml`` is validated against these constants."""
    from stormlab.config import ConfigError, load_config

    try:
        declared = load_config(variant="C1").c1
    except ConfigError as exc:
        pytest.skip(f"C1 config overlay not present in standalone checkout: {exc}")
    assert declared is not None
    assert declared.neighborhood.columns == NEIGHBORHOOD_COLUMNS
    assert declared.neighborhood.hops == NEIGHBORHOOD_HOPS


def test_every_contract_of_the_registry_is_audited_for_provenance() -> None:
    """A contract outside ``AUDITED_CONTRACTS`` is a model input nothing checks."""
    from stormlab.contracts.features import FEATURE_CONTRACTS
    from stormlab.contracts.provenance_columns import AUDITED_CONTRACTS

    assert {item.name for item in AUDITED_CONTRACTS} == set(FEATURE_CONTRACTS)


def test_the_v4_contracts_extend_the_v3_ones_by_the_two_freeboard_columns() -> None:
    """Derived, not retyped: what v4 adds is a suffix, and nothing else moved."""
    assert PHYSICAL_FREEBOARD_COLUMNS == ("inlet_node_freeboard_m", "outlet_node_freeboard_m")
    assert FEATURE_CONTRACT_V4_PHYSICAL.columns == FEATURE_CONTRACT_V3_PHYSICAL.columns + PHYSICAL_FREEBOARD_COLUMNS
    assert FEATURE_CONTRACT_V4_MARGINS.columns[: FEATURE_CONTRACT_V4_PHYSICAL.n_columns] == FEATURE_CONTRACT_V4_PHYSICAL.columns
    assert set(FEATURE_CONTRACT_V3_MARGINS.columns) < set(FEATURE_CONTRACT_V4_MARGINS.columns)


def test_the_v3_physical_view_is_a_prefix_of_both_v4_contracts() -> None:
    """What lets one extraction serve C1 and C2: the older layout opens the newer frame."""
    base = FEATURE_CONTRACT_V3_PHYSICAL
    assert FEATURE_CONTRACT_V4_PHYSICAL.columns[: base.n_columns] == base.columns
    assert FEATURE_CONTRACT_V4_MARGINS.columns[: base.n_columns] == base.columns


def test_the_v4_contracts_are_the_sizes_the_protocol_declares() -> None:
    from stormlab.contracts.features import FEATURE_CONTRACT_V4_PHYSICAL_NBR_NOCAT, FEATURE_CONTRACT_V4_PHYSICAL_NOCAT

    assert FEATURE_CONTRACT_V4_PHYSICAL.n_columns == 66
    assert FEATURE_CONTRACT_V4_MARGINS.n_columns == 79
    assert FEATURE_CONTRACT_V4_PHYSICAL_NBR.n_columns == 122
    assert FEATURE_CONTRACT_V4_PHYSICAL_NOCAT.n_columns == 64
    assert FEATURE_CONTRACT_V4_PHYSICAL_NBR_NOCAT.n_columns == 120


def test_the_v4_contracts_declare_the_build_they_need() -> None:
    """No v3 ``features.parquet`` carries a freeboard column, so this is a fourth build."""
    from stormlab.contracts.features import FEATURE_CONTRACT_V4_PHYSICAL_NBR_NOCAT, FEATURE_CONTRACT_V4_PHYSICAL_NOCAT

    for item in (
        FEATURE_CONTRACT_V4_PHYSICAL,
        FEATURE_CONTRACT_V4_MARGINS,
        FEATURE_CONTRACT_V4_PHYSICAL_NBR,
        FEATURE_CONTRACT_V4_PHYSICAL_NOCAT,
        FEATURE_CONTRACT_V4_PHYSICAL_NBR_NOCAT,
    ):
        assert item.schema_version == 4, item.name


def test_the_v4_neighbourhood_arms_aggregate_c2s_own_list_over_the_same_width() -> None:
    """As wide as C1's block and not the same block: the fourth measurement is the spill."""
    from stormlab.contracts.features import (
        FEATURE_CONTRACT_V4_PHYSICAL_NBR_NOCAT,
        FEATURE_CONTRACT_V4_PHYSICAL_NOCAT,
        neighborhood_column_names,
    )

    generated = neighborhood_column_names(NEIGHBORHOOD_COLUMNS_V4)

    assert len(generated) == len(NEIGHBORHOOD_BLOCK_COLUMNS) == 56
    assert generated != NEIGHBORHOOD_BLOCK_COLUMNS
    assert FEATURE_CONTRACT_V4_PHYSICAL_NBR.columns[FEATURE_CONTRACT_V4_PHYSICAL.n_columns :] == generated
    base = FEATURE_CONTRACT_V4_PHYSICAL_NOCAT
    assert FEATURE_CONTRACT_V4_PHYSICAL_NBR_NOCAT.columns[base.n_columns :] == generated
    assert "up1_inlet_node_flood_hours_mean" in generated
    assert "up1_hours_capacity_limited_mean" not in generated


def test_the_declared_neighbourhoods_differ_only_in_the_overload_premise() -> None:
    """C1 aggregates the hours capacity limited the flow; C2 the flooding of the inlet."""
    assert NEIGHBORHOOD_COLUMNS[:3] == NEIGHBORHOOD_COLUMNS_V4[:3]
    assert NEIGHBORHOOD_COLUMNS[3] == "hours_capacity_limited"
    assert NEIGHBORHOOD_COLUMNS_V4[3] == "inlet_node_flood_hours"
    assert set(NEIGHBORHOOD_COLUMNS_V4) <= set(FEATURE_CONTRACT_V4_PHYSICAL.columns)


def test_every_neighbourhood_contract_names_the_list_it_was_generated_from() -> None:
    """The map the configuration loader validates ``c1.neighborhood.columns`` against."""
    from stormlab.contracts.features import (
        FEATURE_CONTRACTS,
        neighborhood_column_names,
        neighborhood_columns_of,
    )

    assert set(NEIGHBORHOOD_CONTRACT_COLUMNS) <= set(FEATURE_CONTRACTS)
    for name, columns in NEIGHBORHOOD_CONTRACT_COLUMNS.items():
        registered = FEATURE_CONTRACTS[name]
        generated = neighborhood_column_names(columns)
        assert registered.columns[-len(generated) :] == generated, name
        assert neighborhood_columns_of(name) == columns

    with pytest.raises(FeatureContractError, match="Unknown neighbourhood contract"):
        neighborhood_columns_of("v3_physical")


def test_the_noflood_contracts_are_the_v4_views_without_the_four_flood_columns() -> None:
    """Filtered, not retyped: the shared columns keep their order and their meaning."""
    assert set(FEATURE_CONTRACT_V4_PHYSICAL.columns) - set(FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL.columns) == set(
        NODE_FLOOD_COLUMNS
    )
    assert set(FEATURE_CONTRACT_V4_MARGINS.columns) - set(FEATURE_CONTRACT_V4_NOFLOOD_MARGINS.columns) == set(NODE_FLOOD_COLUMNS)
    assert FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL.columns == tuple(
        column for column in FEATURE_CONTRACT_V4_PHYSICAL.columns if column not in set(NODE_FLOOD_COLUMNS)
    )
    assert FEATURE_CONTRACT_V4_NOFLOOD_MARGINS.columns == tuple(
        column for column in FEATURE_CONTRACT_V4_MARGINS.columns if column not in set(NODE_FLOOD_COLUMNS)
    )


def test_the_noflood_arm_keeps_the_hydraulic_strain_it_is_asked_about() -> None:
    """The question of the arm is whether strain predicts retention before the spill."""
    kept = set(FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL.columns)

    assert {
        "max_full_depth_ratio",
        "hours_full_both_ends",
        "hours_full_upstream",
        "hours_full_downstream",
        "hours_above_normal_flow",
        "hours_capacity_limited",
        "inlet_node_max_hgl_m",
        "outlet_node_max_hgl_m",
        "inlet_node_freeboard_m",
        "outlet_node_freeboard_m",
        "flow_capacity_ratio",
    } <= kept
    assert not set(NODE_FLOOD_COLUMNS) & kept


def test_the_noflood_neighbourhood_is_a_subset_and_not_a_substitution() -> None:
    """No replacement column takes the place of the one the arm may not read."""
    from stormlab.contracts.features import neighborhood_column_names

    generated = neighborhood_column_names(NEIGHBORHOOD_COLUMNS_V4_NOFLOOD)

    assert NEIGHBORHOOD_COLUMNS_V4_NOFLOOD == NEIGHBORHOOD_COLUMNS_V4[:3]
    assert len(generated) == 2 * len(NEIGHBORHOOD_HOPS) * (len(NEIGHBORHOOD_COLUMNS_V4_NOFLOOD) * 3 + 2) == 44
    assert set(generated) < set(neighborhood_column_names(NEIGHBORHOOD_COLUMNS_V4))
    assert FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL_NBR.columns[FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL.n_columns :] == generated


def test_the_noflood_contracts_are_the_sizes_the_plan_declares() -> None:
    """62 = 66 - 4, 75 = 79 - 4, 106 = 62 + 4 x (3 x 3 + 2)."""
    assert FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL.n_columns == 62
    assert FEATURE_CONTRACT_V4_NOFLOOD_MARGINS.n_columns == 75
    assert FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL_NBR.n_columns == 106
    assert FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL.schema_version == 4
    assert FEATURE_CONTRACT_V4_NOFLOOD_MARGINS.schema_version == 4
    assert FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL_NBR.schema_version == 4


def test_the_v4_digests_are_pinned_literals_and_moved_no_earlier_one() -> None:
    """The v3 digests are the frozen ones; the v4 neighbourhood digests moved with C2's list."""
    from stormlab.contracts.features import FEATURE_CONTRACT_V4_PHYSICAL_NBR_NOCAT, FEATURE_CONTRACT_V4_PHYSICAL_NOCAT

    from stormlab.contracts.features import FEATURE_CONTRACT_V3_PHYSICAL_NBR_NOCAT, FEATURE_CONTRACT_V3_PHYSICAL_NOCAT

    assert FEATURE_CONTRACT_V4_PHYSICAL.sha == "15c606dbf9ae63298edf2761a6c548407d4f24112b2a3433a857bf73927dc4c9"
    assert FEATURE_CONTRACT_V4_MARGINS.sha == "7d4fa45fd8a6aab4adcdf28625a87200fbd88695202abc8bb1d6ec17f2556441"
    assert FEATURE_CONTRACT_V4_PHYSICAL_NBR.sha == "03162222625a0a4ca1061a8ce091763eb28f904d2059e93b8ab12464dbac1b7d"
    assert FEATURE_CONTRACT_V4_PHYSICAL_NOCAT.sha == "550fc7a2a9efdb91ac075c60d8e1a30b5443ae20e4f05f6f382b68b298e4d4a7"
    assert FEATURE_CONTRACT_V4_PHYSICAL_NBR_NOCAT.sha == "ec216f965877bc1fb5a8ea22528b9cd4439e13906283eb9000ec47adea6c9480"
    assert FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL.sha == "eeb0966761034774d6ab5c3b9bcc0bb1466a46aab90e5e343a72be7664a9dafa"
    assert FEATURE_CONTRACT_V4_NOFLOOD_MARGINS.sha == "b60d35b7c2bf272b0de6105e0e6f181f4de08672474d09a104072f306d4a30f2"
    assert FEATURE_CONTRACT_V4_NOFLOOD_PHYSICAL_NBR.sha == "8427f55a32a41279876d9906fcd2fe459eb104d42140cf8b3b80827fbbc9393b"
    assert FEATURE_CONTRACT_V3_PHYSICAL.sha == "6e6320bbf670b5af9053a65706fe7c3d0784e0c9c3d5f018d3aa5c70047668b5"
    assert FEATURE_CONTRACT_V3_MARGINS.sha == "f9f23105b6b786b0c30527ad2b554aa3528553b5d3e26dc1687d5c4c0a89be9d"
    assert FEATURE_CONTRACT_V3_PHYSICAL_NBR.sha == "a23b20e526c3c4101b2caf861d2152fa7ed6755798be0dc083ae956c21acbd12"
    assert FEATURE_CONTRACT_V3_PHYSICAL_NOCAT.sha == "31416687db6ecfede4c46607720701c696b027891b18ec251193c817b9214f16"
    assert FEATURE_CONTRACT_V3_PHYSICAL_NBR_NOCAT.sha == "ee4429fc9afb46451524aaeb86cee549b5bf2e2ab540886e4061e4b776e07c76"
    assert FEATURE_CONTRACT_V1.sha == "fc570502f408b4b88f406eac8070a8d2f728e07cb0181a9369f11d51b98f66b1"
    assert FEATURE_CONTRACT_V2.sha == "8184be657c6ba4a321f9932f5747c31fa79ebf473eee289265bebef71c19b3e2"
    assert FEATURE_CONTRACT_V2_NOFLAGS.sha == "ddc0b72fc23fae7a1db5872b07c957d8c87f2fbb87d0d465d134c7b3c5fb2c9c"


def test_the_freeboard_columns_are_floats_and_no_contract_calls_them_a_flag() -> None:
    """A measurement, not an indicator: the ``int8`` block of the view is the masks."""
    from stormlab.contracts.features import PHYSICAL_FLAG_COLUMNS

    assert not set(PHYSICAL_FREEBOARD_COLUMNS) & set(PHYSICAL_FLAG_COLUMNS)
    dtypes = FEATURE_CONTRACT_V4_MARGINS.dtypes_map
    assert {dtypes[column] for column in PHYSICAL_FREEBOARD_COLUMNS} == {contract.FLOAT_DTYPE}
