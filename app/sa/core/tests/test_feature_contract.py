"""Contract tests for the serving GNN feature list."""

from sa.core.feature_contract import get_default_feature_columns

# Golden copy of the trained-model feature contract, kept independent of the
# source list so an accidental change to it is caught here.
EXPECTED_FEATURE_COLUMNS = [
    "ValMaxFill",
    "ValMaxV",
    "ValMinV",
    "ValMaxSlope",
    "ValMinSlope",
    "ValDepth",
    "ValCoverage",
    "isMinDiameter",
    "IncreaseDia",
    "ReduceDia",
    "IncreaseSlope",
    "ReduceSlope",
    "NRoughness",
    "NMaxV",
    "NInletDepth",
    "NOutletDepth",
    "NFilling",
    "NMaxQ",
    "NInletGroundCover",
    "NOutletGroundCover",
    "NSlope",
    "marshes",
    "suburban_highly_impervious",
    "suburban_weakly_impervious",
    "arable",
    "meadows",
    "forests",
    "rural",
    "urban_weakly_impervious",
    "urban_moderately_impervious",
    "urban_highly_impervious",
    "mountains_rocky",
    "mountains_vegetated",
]


def test_default_feature_columns_match_trained_gnn_contract():
    assert get_default_feature_columns() == EXPECTED_FEATURE_COLUMNS
    assert len(EXPECTED_FEATURE_COLUMNS) == 33


def test_get_default_feature_columns_returns_a_fresh_ordered_list():
    first = get_default_feature_columns()
    second = get_default_feature_columns()

    assert first == second
    assert first is not second
    assert len(first) == 33
    assert first[:3] == ["ValMaxFill", "ValMaxV", "ValMinV"]
    assert first[-3:] == [
        "urban_highly_impervious",
        "mountains_rocky",
        "mountains_vegetated",
    ]
