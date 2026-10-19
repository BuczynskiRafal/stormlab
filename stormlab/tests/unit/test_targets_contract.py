"""The target contract: column order, row sums and what an unsupervised row looks like."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stormlab.contracts.enums import RECOMMENDATION_CLASS_ORDER
from stormlab.contracts.targets import (
    PRIMARY_ACTION_V3,
    TargetContract,
    TargetContractError,
    target_contract,
)


def _row(**shares: float) -> list[float]:
    return [float(shares.get(name, 0.0)) for name in PRIMARY_ACTION_V3.classes]


class TestTheContractItself:
    def test_it_carries_the_nine_classes_in_the_enum_order(self) -> None:
        assert PRIMARY_ACTION_V3.classes == RECOMMENDATION_CLASS_ORDER
        assert PRIMARY_ACTION_V3.n_classes == 9
        assert PRIMARY_ACTION_V3.version == 3

    def test_the_sha_is_pinned_so_a_moved_column_is_visible(self) -> None:
        # A literal, not a recomputation: the point of the digest is that a column
        # order changing without anyone noticing becomes a failing test.
        assert PRIMARY_ACTION_V3.sha == "2cdedd116c71a0a40102f77495de724fe3f2e7c4cb272167ea8b6a0c2968dabc"

    def test_reordering_the_classes_changes_the_sha(self) -> None:
        reordered = TargetContract(
            name=PRIMARY_ACTION_V3.name,
            classes=tuple(reversed(PRIMARY_ACTION_V3.classes)),
            version=PRIMARY_ACTION_V3.version,
        )

        assert reordered.sha != PRIMARY_ACTION_V3.sha

    def test_a_contract_naming_a_class_twice_is_refused(self) -> None:
        with pytest.raises(TargetContractError, match="twice"):
            TargetContract(name="broken", classes=("valid", "valid"), version=1)

    def test_the_registry_resolves_the_contract_and_refuses_an_unknown_name(self) -> None:
        assert target_contract("primary_action_v3") is PRIMARY_ACTION_V3
        with pytest.raises(TargetContractError, match="Unknown target contract"):
            target_contract("primary_action_v4")


class TestTheColumnOrderIsCheckedByName:
    def test_the_contract_order_passes(self) -> None:
        PRIMARY_ACTION_V3.assert_columns(list(PRIMARY_ACTION_V3.classes))

    def test_a_swapped_pair_of_columns_is_refused_by_position(self) -> None:
        columns = list(PRIMARY_ACTION_V3.classes)
        columns[0], columns[1] = columns[1], columns[0]

        with pytest.raises(TargetContractError, match="expects column 0 to be 'pump'"):
            PRIMARY_ACTION_V3.assert_columns(columns)

    def test_a_matrix_of_the_right_width_in_the_wrong_order_is_still_refused(self) -> None:
        frame = pd.DataFrame([_row(valid=1.0)], columns=list(reversed(PRIMARY_ACTION_V3.classes)))

        with pytest.raises(TargetContractError, match="expects column 0"):
            PRIMARY_ACTION_V3.validate_frame(frame, np.array([1]))

    def test_too_few_columns_names_the_count(self) -> None:
        with pytest.raises(TargetContractError, match="has 9 column"):
            PRIMARY_ACTION_V3.assert_columns(["pump", "tank"])


class TestTheValuesOfASupervisedRow:
    def test_a_distribution_that_sums_to_one_passes(self) -> None:
        matrix = np.array([_row(valid=0.6, tank=0.4)])

        PRIMARY_ACTION_V3.validate(matrix, np.array([1]))

    def test_a_row_that_does_not_sum_to_one_is_refused_with_its_index(self) -> None:
        matrix = np.array([_row(valid=1.0), _row(valid=0.6, tank=0.3)])

        with pytest.raises(TargetContractError, match="Row 1 sums to"):
            PRIMARY_ACTION_V3.validate(matrix, np.array([1, 1]))

    def test_a_negative_share_is_refused(self) -> None:
        matrix = np.array([_row(valid=1.2, tank=-0.2)])

        with pytest.raises(TargetContractError, match="outside \\[0, 1\\]"):
            PRIMARY_ACTION_V3.validate(matrix, np.array([1]))

    def test_a_supervised_row_may_not_hold_a_missing_value(self) -> None:
        row = _row(valid=1.0)
        row[1] = float("nan")
        matrix = np.array([row])

        with pytest.raises(TargetContractError, match="supervised but holds a missing value"):
            PRIMARY_ACTION_V3.validate(matrix, np.array([1]))

    def test_a_share_a_hair_over_one_passes_and_a_wider_one_does_not(self) -> None:
        """A class every source of the only active group supported comes out `1 + eps`."""
        near = np.array([_row(valid=1.0 + 1e-9)])
        far = np.array([_row(valid=1.0 + 1e-3)])

        PRIMARY_ACTION_V3.validate(near, np.array([1]))
        with pytest.raises(TargetContractError, match="outside \\[0, 1\\]"):
            PRIMARY_ACTION_V3.validate(far, np.array([1]))

    def test_the_row_sum_tolerance_admits_float_accumulation_and_nothing_wider(self) -> None:
        near = np.array([_row(valid=1.0 - 5e-7)])
        far = np.array([_row(valid=1.0 - 5e-5)])

        PRIMARY_ACTION_V3.validate(near, np.array([1]))
        with pytest.raises(TargetContractError, match="sums to"):
            PRIMARY_ACTION_V3.validate(far, np.array([1]))


class TestAnUnsupervisedRowIsNotAnything:
    def test_an_all_nan_row_under_mask_zero_passes(self) -> None:
        matrix = np.array([_row(valid=1.0), [np.nan] * 9])

        PRIMARY_ACTION_V3.validate(matrix, np.array([1, 0]))

    def test_a_masked_row_carrying_valid_is_refused(self) -> None:
        matrix = np.array([_row(valid=1.0)])

        with pytest.raises(TargetContractError, match="carries mask 0 but states"):
            PRIMARY_ACTION_V3.validate(matrix, np.array([0]))

    def test_a_masked_uniform_row_is_refused_too(self) -> None:
        matrix = np.array([[1.0 / 9] * 9])

        with pytest.raises(TargetContractError, match="carries mask 0 but states"):
            PRIMARY_ACTION_V3.validate(matrix, np.array([0]))

    def test_a_partly_filled_masked_row_is_refused(self) -> None:
        row = [np.nan] * 9
        row[0] = 0.0
        matrix = np.array([row])

        with pytest.raises(TargetContractError, match="carries mask 0 but states"):
            PRIMARY_ACTION_V3.validate(matrix, np.array([0]))


class TestTheShapesAndTheMask:
    def test_a_matrix_of_the_wrong_width_is_refused(self) -> None:
        with pytest.raises(TargetContractError, match=r"shape \(n, 9\)"):
            PRIMARY_ACTION_V3.validate(np.zeros((2, 8)), np.array([0, 0]))

    def test_a_mask_that_does_not_address_the_rows_is_refused(self) -> None:
        with pytest.raises(TargetContractError, match="does not address"):
            PRIMARY_ACTION_V3.validate(np.array([_row(valid=1.0)]), np.array([1, 1]))

    def test_a_mask_holding_anything_but_zero_and_one_is_refused(self) -> None:
        with pytest.raises(TargetContractError, match="it is 0 or 1"):
            PRIMARY_ACTION_V3.validate(np.array([_row(valid=1.0)]), np.array([2]))
