"""The 12-feature catchment contract, including the divide-by-Area guard."""

from __future__ import annotations

import pandas as pd
import pytest

from stormlab.contracts.catchment_features import (
    CATCHMENT_AREA_NORMALIZED,
    CATCHMENT_FEATURE_COLUMNS,
    CATCHMENT_INPUT_SHAPE,
    FEATURE_SETS,
    FULL_12,
    INP_2,
    INP_7,
    CatchmentContractError,
    CatchmentFeatureSet,
    assert_classifier_input_shape,
    build_catchment_features,
    feature_set_contract_sha,
    input_shape,
    transform_column,
    validate_catchment_features,
)

EXPECTED_ORDER = (
    "PercImperv",
    "PercSlope",
    "N-Imperv",
    "N-Perv",
    "S-Imperv",
    "S-Perv",
    "PctZero",
    "RunoffCoeff",
    "TotalInfil",
    "ImpervRunoff",
    "TotalRunoffMG",
    "PeakRunoff",
)


def _frame(total_runoff: float = 4.0) -> pd.DataFrame:
    values = {column: [1.0, 2.0] for column in CATCHMENT_FEATURE_COLUMNS}
    values["TotalRunoffMG"] = [total_runoff, total_runoff]
    values["Area"] = [2.0, 4.0]
    return pd.DataFrame(values, index=["S1", "S2"])


def test_contract_declares_twelve_columns_in_the_agreed_order() -> None:
    assert CATCHMENT_FEATURE_COLUMNS == EXPECTED_ORDER
    assert CATCHMENT_INPUT_SHAPE == (None, 12)


def test_only_the_last_four_columns_are_area_normalized() -> None:
    assert CATCHMENT_AREA_NORMALIZED == ("TotalInfil", "ImpervRunoff", "TotalRunoffMG", "PeakRunoff")
    assert "RunoffCoeff" not in CATCHMENT_AREA_NORMALIZED


def test_plain_column_passes_through_unchanged() -> None:
    result = transform_column(_frame(), "RunoffCoeff")
    assert list(result) == [1.0, 2.0]


def test_area_normalized_column_is_divided_by_area() -> None:
    result = transform_column(_frame(), "TotalRunoffMG")
    assert list(result) == [2.0, 1.0]


def test_all_zero_column_is_left_untouched_instead_of_producing_nan() -> None:
    frame = _frame(total_runoff=0.0)
    frame.loc["S1", "Area"] = 0.0
    result = transform_column(frame, "TotalRunoffMG")
    assert list(result) == [0.0, 0.0]
    assert not result.isna().any()


def test_transform_rejects_a_column_outside_the_contract() -> None:
    with pytest.raises(CatchmentContractError, match="not a catchment feature"):
        transform_column(_frame(), "Area")


def test_transform_reports_a_missing_area_column() -> None:
    frame = _frame().drop(columns=["Area"])
    with pytest.raises(CatchmentContractError, match="is required to normalise"):
        transform_column(frame, "PeakRunoff")


def test_build_produces_a_validated_twelve_column_matrix() -> None:
    features = build_catchment_features(_frame())
    validate_catchment_features(features)
    assert list(features.columns) == list(CATCHMENT_FEATURE_COLUMNS)
    assert features.shape == (2, 12)


def test_validate_rejects_a_missing_feature() -> None:
    features = build_catchment_features(_frame()).drop(columns=["PeakRunoff"])
    with pytest.raises(CatchmentContractError, match="Missing catchment feature columns"):
        validate_catchment_features(features)


def test_a_zero_area_row_is_rejected_by_name_instead_of_producing_inf() -> None:
    """The ``sum() != 0`` guard is column-level; one zero row still divides by zero."""
    frame = _frame()
    frame.loc["S2", "Area"] = 0.0

    with pytest.raises(CatchmentContractError, match="S2"):
        transform_column(frame, "TotalRunoffMG")


def test_a_negative_area_row_is_rejected_too() -> None:
    frame = _frame()
    frame.loc["S1", "Area"] = -1.0

    with pytest.raises(CatchmentContractError, match="non-positive"):
        build_catchment_features(frame)


def test_validate_rejects_an_infinite_feature_that_isna_cannot_see() -> None:
    features = build_catchment_features(_frame())
    features.loc["S1", "PeakRunoff"] = float("inf")

    with pytest.raises(CatchmentContractError, match="Non-finite"):
        validate_catchment_features(features)


class _FakeModel:
    def __init__(self, input_shape: object) -> None:
        self.input_shape = input_shape


def test_the_classifier_shape_assertion_accepts_the_contract_shape() -> None:
    assert_classifier_input_shape(_FakeModel(CATCHMENT_INPUT_SHAPE))


def test_the_classifier_shape_assertion_rejects_a_different_width() -> None:
    with pytest.raises(CatchmentContractError, match="does not match the contract"):
        assert_classifier_input_shape(_FakeModel((None, 11)))


def test_the_classifier_shape_assertion_rejects_an_object_without_the_attribute() -> None:
    with pytest.raises(CatchmentContractError, match="no 'input_shape'"):
        assert_classifier_input_shape(object())


def test_the_real_catchment_classifier_takes_the_contract_shape() -> None:
    """Not a mock: the saved model is what chapter 3 actually runs."""
    keras = pytest.importorskip("keras")
    from stormlab.config import repo_root

    saved = repo_root() / "models" / "catchment_classification_model" / "model.keras"
    if not saved.is_file():
        pytest.skip(f"tracked catchment classifier not present in standalone checkout: {saved}")

    assert_classifier_input_shape(keras.saving.load_model(saved, compile=False))


class TestFeatureSets:
    """A feature set is a named subset of the contract, and ``full_12`` is the contract itself."""

    def test_the_three_sets_are_registered_under_their_own_names(self) -> None:
        assert set(FEATURE_SETS) == {"full_12", "inp_7", "inp_2"}
        assert all(name == feature_set.name for name, feature_set in FEATURE_SETS.items())

    def test_the_full_set_is_the_twelve_contract_columns_in_contract_order(self) -> None:
        assert FULL_12.columns == CATCHMENT_FEATURE_COLUMNS
        assert input_shape(FULL_12) == CATCHMENT_INPUT_SHAPE

    def test_the_narrow_sets_are_ordered_subsets_of_the_full_one(self) -> None:
        """Subsets, so a frame of twelve columns can be selected down to either of them."""
        assert INP_7.columns == CATCHMENT_FEATURE_COLUMNS[:7]
        assert INP_2.columns == ("PercImperv", "PercSlope")
        for narrow in (INP_7, INP_2):
            assert set(narrow.columns) <= set(FULL_12.columns)
            assert list(narrow.columns) == [column for column in FULL_12.columns if column in set(narrow.columns)]

    def test_the_full_sets_digest_is_the_one_every_earlier_run_recorded(self) -> None:
        """A literal: a model's registry entry pins this string, and it must not move."""
        assert feature_set_contract_sha(FULL_12) == "e41d7e782c726a1aa5b9e53c66c0d24695dd937c3e58f1110c5e1f76f3f1fdd6"

    def test_a_narrower_set_hashes_to_something_else(self) -> None:
        digests = {feature_set_contract_sha(feature_set) for feature_set in (FULL_12, INP_7, INP_2)}
        assert len(digests) == 3

    def test_the_shape_of_a_narrow_set_is_its_own_width(self) -> None:
        assert input_shape(INP_7) == (None, 7)
        assert input_shape(INP_2) == (None, 2)

    def test_a_set_naming_a_column_outside_the_contract_is_refused(self) -> None:
        with pytest.raises(CatchmentContractError, match="outside the catchment contract"):
            CatchmentFeatureSet("bogus", ("PercImperv", "Area"))

    def test_a_set_repeating_a_column_is_refused(self) -> None:
        with pytest.raises(CatchmentContractError, match="repeats a column"):
            CatchmentFeatureSet("twice", ("PercImperv", "PercImperv"))

    def test_an_empty_set_is_refused(self) -> None:
        with pytest.raises(CatchmentContractError, match="at least one feature"):
            CatchmentFeatureSet("nothing", ())


class TestBuildingAndValidatingUnderASet:
    def test_building_under_a_narrow_set_returns_only_its_columns(self) -> None:
        built = build_catchment_features(_frame(), INP_7)

        assert list(built.columns) == list(INP_7.columns)
        assert len(built) == 2

    def test_validation_under_a_narrow_set_accepts_the_narrow_frame(self) -> None:
        validate_catchment_features(build_catchment_features(_frame(), INP_2), INP_2)

    def test_a_twelve_column_frame_does_not_satisfy_a_narrow_set(self) -> None:
        """The order check is what catches it: extra columns are not silently ignored."""
        with pytest.raises(CatchmentContractError, match="contract order"):
            validate_catchment_features(build_catchment_features(_frame()), INP_7)

    def test_the_shape_assertion_follows_the_set_it_is_given(self) -> None:
        assert_classifier_input_shape(_FakeModel((None, 7)), INP_7)
        with pytest.raises(CatchmentContractError, match="does not match the contract"):
            assert_classifier_input_shape(_FakeModel((None, 12)), INP_7)
