"""NMaxQ capacity laws and the validation exposed when clipping is disabled."""

import numpy as np
import pandas as pd
import pytest

from stormlab.features.conduits import ConduitFeatureError, ConduitFeatureSteps
from stormlab.hydraulics.round import MANNING_FULL_PIPE_COEFFICIENT


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": ["C0", "C1"],
            "Geom1": [0.4, 0.6],
            "Roughness": [0.010, 0.013],
            "SlopeFtPerFt": [0.010, 0.004],
            "SlopePerMile": [10.0, 4.0],
            "MaxQ": [0.2, 1.0],
        }
    ).set_index("Name")


def test_legacy_law_with_clipping_matches_the_historical_formula() -> None:
    frame = _frame()
    steps = ConduitFeatureSteps(frame.copy(), None, 1.0, flow_law="legacy_k_d25", nmaxq_clip=True)

    steps.normalize_max_q()

    expected = np.minimum(
        frame["MaxQ"] / (0.312 * frame["Geom1"] ** 2.5 * frame["SlopePerMile"] ** 0.5),
        1.0,
    )
    np.testing.assert_array_equal(steps.dfc["NMaxQ"].to_numpy(), expected.to_numpy())


def test_manning_v2_uses_per_conduit_roughness_and_optional_clipping() -> None:
    frame = _frame()
    capacity = MANNING_FULL_PIPE_COEFFICIENT / frame["Roughness"] * frame["Geom1"] ** (8 / 3) * np.sqrt(frame["SlopeFtPerFt"])
    frame["MaxQ"] = [0.5 * capacity.iloc[0], 1.5 * capacity.iloc[1]]

    raw = ConduitFeatureSteps(frame.copy(), None, 1.0, flow_law="manning_v2", nmaxq_clip=False)
    clipped = ConduitFeatureSteps(frame.copy(), None, 1.0, flow_law="manning_v2", nmaxq_clip=True)
    raw.normalize_max_q()
    clipped.normalize_max_q()

    for conduit, expected in zip(("C0", "C1"), (0.5, 1.5), strict=True):
        assert raw.dfc.loc[conduit, "NMaxQ"] == pytest.approx(expected)
    assert clipped.dfc["NMaxQ"].tolist() == pytest.approx([0.5, 1.0])


@pytest.mark.parametrize(
    ("column", "value", "requirement"),
    [
        ("MaxQ", -0.1, ">= 0"),
        ("Roughness", 0.0, "> 0"),
        ("Roughness", -0.01, "> 0"),
        ("Geom1", 0.0, "> 0"),
        ("Geom1", -0.4, "> 0"),
        ("SlopeFtPerFt", -0.001, ">= 0"),
    ],
)
def test_unclipped_nmaxq_rejects_invalid_lower_bounds(column: str, value: float, requirement: str) -> None:
    frame = _frame().iloc[[0]].copy()
    frame.loc["C0", column] = value
    steps = ConduitFeatureSteps(frame, None, 1.0, flow_law="manning_v2", nmaxq_clip=False)

    with pytest.raises(ConduitFeatureError, match=rf"C0.*{column}=.*expected {column} {requirement}"):
        steps.normalize_max_q()


def test_positive_flow_at_zero_slope_is_rejected_without_clipping() -> None:
    frame = _frame().iloc[[0]].copy()
    frame.loc["C0", "SlopeFtPerFt"] = 0.0
    frame.loc["C0", "SlopePerMile"] = 0.0
    steps = ConduitFeatureSteps(frame, None, 1.0, flow_law="manning_v2", nmaxq_clip=False)

    with pytest.raises(ConduitFeatureError, match="C0.*would be infinite"):
        steps.normalize_max_q()


def test_zero_flow_at_zero_slope_normalizes_to_zero_without_clipping() -> None:
    frame = _frame().iloc[[0]].copy()
    frame.loc["C0", ["SlopeFtPerFt", "SlopePerMile", "MaxQ"]] = 0.0
    steps = ConduitFeatureSteps(frame, None, 1.0, flow_law="manning_v2", nmaxq_clip=False)

    steps.normalize_max_q()

    assert steps.dfc.loc["C0", "NMaxQ"] == 0.0


def test_legacy_law_without_clipping_validates_and_preserves_overload() -> None:
    frame = _frame().iloc[[0]].copy()
    frame.loc["C0", "MaxQ"] = 10.0
    steps = ConduitFeatureSteps(frame, None, 1.0, flow_law="legacy_k_d25", nmaxq_clip=False)

    steps.normalize_max_q()

    assert steps.dfc.loc["C0", "NMaxQ"] > 1.0

    frame.loc["C0", "Geom1"] = 0.0
    invalid = ConduitFeatureSteps(frame, None, 1.0, flow_law="legacy_k_d25", nmaxq_clip=False)
    with pytest.raises(ConduitFeatureError, match=r"C0.*Geom1=.*expected Geom1 > 0"):
        invalid.normalize_max_q()


def test_unknown_flow_law_is_rejected() -> None:
    steps = ConduitFeatureSteps(_frame(), None, 1.0, flow_law="unknown")  # type: ignore[arg-type]

    with pytest.raises(ConduitFeatureError, match="Unknown flow_law 'unknown'"):
        steps.normalize_max_q()


def test_manning_v2_with_clipping_rejects_negative_slope_before_sqrt() -> None:
    frame = _frame().iloc[[0]].copy()
    frame.loc["C0", "SlopeFtPerFt"] = -0.001
    steps = ConduitFeatureSteps(frame, None, 1.0, flow_law="manning_v2", nmaxq_clip=True)

    with pytest.raises(ConduitFeatureError, match=r"C0.*SlopeFtPerFt=.*expected SlopeFtPerFt >= 0"):
        steps.normalize_max_q()


def test_manning_to_legacy_capacity_ratio_increases_with_diameter() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["small", "large"],
            "Geom1": [0.3, 1.2],
            "Roughness": [0.013, 0.013],
            "SlopeFtPerFt": [0.01, 0.01],
            "SlopePerMile": [10.0, 10.0],
            "MaxQ": [0.01, 0.01],
        }
    ).set_index("Name")
    legacy = ConduitFeatureSteps(frame.copy(), None, 1.0, flow_law="legacy_k_d25", nmaxq_clip=False)
    manning = ConduitFeatureSteps(frame.copy(), None, 1.0, flow_law="manning_v2", nmaxq_clip=False)

    legacy.normalize_max_q()
    manning.normalize_max_q()
    capacity_ratio = legacy.dfc["NMaxQ"] / manning.dfc["NMaxQ"]

    assert capacity_ratio.loc["large"] > capacity_ratio.loc["small"]
