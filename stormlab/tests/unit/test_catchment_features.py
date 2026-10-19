"""Tests for the twelve-feature catchment classifier input.

New tests: the original extraction existed only as notebook cell 9, so it was
never covered. The cases below pin the two things that cell got right — the four
area-normalised columns and the per-column ``sum() != 0`` guard — and the one
thing its inline copy got wrong, by checking each guard against its own column.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stormlab.contracts.catchment_features import (
    CATCHMENT_AREA_NORMALIZED,
    CATCHMENT_FEATURE_COLUMNS,
    CATCHMENT_FEATURE_DTYPE,
    CatchmentContractError,
)
from stormlab.features.catchment import extract_catchment_features

PASSTHROUGH_COLUMNS = tuple(column for column in CATCHMENT_FEATURE_COLUMNS if column not in CATCHMENT_AREA_NORMALIZED)


def subcatchment_frame(**overrides: object) -> pd.DataFrame:
    """A two-row subcatchment frame with every contract column present."""
    data: dict[str, list[float]] = {
        "Area": [2.0, 4.0],
        "PercImperv": [50.0, 25.0],
        "PercSlope": [1.0, 2.0],
        "N-Imperv": [0.01, 0.02],
        "N-Perv": [0.1, 0.2],
        "S-Imperv": [1.5, 2.5],
        "S-Perv": [5.0, 6.0],
        "PctZero": [25.0, 10.0],
        "RunoffCoeff": [0.6, 0.3],
        "TotalInfil": [4.0, 8.0],
        "ImpervRunoff": [2.0, 12.0],
        "TotalRunoffMG": [6.0, 4.0],
        "PeakRunoff": [1.0, 2.0],
    }
    data.update({key: value for key, value in overrides.items()})  # type: ignore[arg-type]
    return pd.DataFrame(data, index=["S1", "S2"])


class TestExtractCatchmentFeatures:
    def test_returns_the_contract_columns_in_order(self):
        result = extract_catchment_features(subcatchment_frame())

        assert list(result.columns) == list(CATCHMENT_FEATURE_COLUMNS)
        assert {str(dtype) for dtype in result.dtypes} == {CATCHMENT_FEATURE_DTYPE}

    def test_area_is_not_a_feature(self):
        result = extract_catchment_features(subcatchment_frame())

        assert "Area" not in result.columns

    def test_passes_the_eight_direct_columns_through_unchanged(self):
        frame = subcatchment_frame()

        result = extract_catchment_features(frame)

        for column in PASSTHROUGH_COLUMNS:
            assert list(result[column]) == list(frame[column].astype(float))

    def test_divides_the_four_runoff_columns_by_area(self):
        result = extract_catchment_features(subcatchment_frame())

        assert list(result["TotalInfil"]) == [2.0, 2.0]
        assert list(result["ImpervRunoff"]) == [1.0, 3.0]
        assert list(result["TotalRunoffMG"]) == [3.0, 1.0]
        assert list(result["PeakRunoff"]) == [0.5, 0.5]

    def test_runoff_coefficient_is_not_divided_by_area(self):
        result = extract_catchment_features(subcatchment_frame())

        assert list(result["RunoffCoeff"]) == [0.6, 0.3]

    @pytest.mark.parametrize("column", CATCHMENT_AREA_NORMALIZED)
    def test_an_all_zero_runoff_column_stays_zero_instead_of_nan(self, column: str):
        """The ``sum() != 0`` guard of the training notebook, checked per column."""
        result = extract_catchment_features(subcatchment_frame(**{column: [0.0, 0.0]}))

        assert list(result[column]) == [0.0, 0.0]
        assert not result[column].isna().any()

    @pytest.mark.parametrize("column", CATCHMENT_AREA_NORMALIZED)
    def test_the_guard_of_one_column_does_not_silence_another(self, column: str):
        """Cell 34 of the notebook guards ``TotalRunoffMG`` on ``TotalRunoffIn``; not here."""
        result = extract_catchment_features(subcatchment_frame(**{column: [0.0, 0.0]}))

        others = [other for other in CATCHMENT_AREA_NORMALIZED if other != column]
        for other in others:
            assert (result[other] != 0).any()

    def test_a_zero_sum_column_that_is_not_identically_zero_is_still_guarded(self):
        """The guard tests the sum, not the values: cancelling values keep the column raw."""
        result = extract_catchment_features(subcatchment_frame(TotalInfil=[5.0, -5.0]))

        assert list(result["TotalInfil"]) == [5.0, -5.0]

    def test_parses_report_columns_that_arrive_as_strings(self):
        frame = subcatchment_frame()
        frame["PeakRunoff"] = ["1.0", "2.0"]

        result = extract_catchment_features(frame)

        assert list(result["PeakRunoff"]) == [0.5, 0.5]

    def test_does_not_mutate_its_input(self):
        frame = subcatchment_frame()
        before = frame.copy()

        extract_catchment_features(frame)

        pd.testing.assert_frame_equal(frame, before)

    def test_preserves_the_index(self):
        result = extract_catchment_features(subcatchment_frame())

        assert list(result.index) == ["S1", "S2"]

    def test_rejects_a_frame_without_area(self):
        with pytest.raises(CatchmentContractError, match="Area"):
            extract_catchment_features(subcatchment_frame().drop(columns=["Area"]))

    def test_rejects_a_frame_missing_a_feature_column(self):
        with pytest.raises(CatchmentContractError, match="PctZero"):
            extract_catchment_features(subcatchment_frame().drop(columns=["PctZero"]))

    def test_extracts_from_the_frozen_network(self, frozen_dfs):
        """The synthetic frames above cannot catch a swmmio column-name or dtype surprise."""
        result = extract_catchment_features(frozen_dfs)

        assert result.shape == (len(frozen_dfs), len(CATCHMENT_FEATURE_COLUMNS))
        assert list(result.columns) == list(CATCHMENT_FEATURE_COLUMNS)
        assert not result.isna().any().any()

    def test_rejects_unparsable_report_values(self):
        frame = subcatchment_frame()
        frame["PeakRunoff"] = ["n/a", "n/a"]

        with pytest.raises(CatchmentContractError, match="Missing values"):
            extract_catchment_features(frame)
