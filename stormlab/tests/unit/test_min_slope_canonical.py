"""The corrected slope profile delegates to the one physical implementation."""

import numpy as np
import pytest

from stormlab.hydraulics.round import common_diameters, min_slope
from stormlab.hydraulics.validators import (
    min_slope_canonical,
    min_slope_hydromechanic,
    validate_min_slope,
)


@pytest.mark.parametrize("diameter", common_diameters)
@pytest.mark.parametrize("filling_ratio", np.arange(0.05, 1.0, 0.05))
def test_canonical_adapter_matches_round_min_slope_on_the_full_grid(filling_ratio: float, diameter: float) -> None:
    filling = filling_ratio * diameter

    np.testing.assert_allclose(
        min_slope_canonical(filling, diameter),
        min_slope(filling, diameter) / 1000,
        rtol=1e-9,
        atol=0,
    )


@pytest.mark.parametrize(
    ("filling_ratio", "diameter", "expected"),
    [
        (0.05, 0.2, 0.023486855453792286),
        (0.25, 0.4, 0.0026070644677348097),
        (0.5, 1.0, 0.0006116207951070336),
        (0.95, 2.0, 0.0002668954182337894),
    ],
)
def test_canonical_adapter_has_pinned_grid_values(filling_ratio: float, diameter: float, expected: float) -> None:
    assert min_slope_canonical(filling_ratio * diameter, diameter) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_canonical_validation_uses_the_adapter_threshold() -> None:
    filling = 0.1
    diameter = 0.4
    threshold = min_slope_canonical(filling, diameter)

    assert validate_min_slope(threshold, filling, diameter, rule="hydromechanical_canonical")
    assert not validate_min_slope(
        np.nextafter(threshold, 0.0),
        filling,
        diameter,
        rule="hydromechanical_canonical",
    )


def test_canonical_threshold_scales_with_theta_while_legacy_rule_ignores_it() -> None:
    filling = 0.1
    diameter = 0.4
    default_threshold = min_slope_canonical(filling, diameter)
    doubled_threshold = min_slope_canonical(filling, diameter, theta=3.0)

    assert doubled_threshold == 2 * default_threshold
    assert validate_min_slope(doubled_threshold, filling, diameter, rule="hydromechanical_canonical", theta=3.0)
    assert not validate_min_slope(
        np.nextafter(doubled_threshold, 0.0),
        filling,
        diameter,
        rule="hydromechanical_canonical",
        theta=3.0,
    )

    legacy_threshold = min_slope_hydromechanic(filling, diameter)
    assert validate_min_slope(legacy_threshold, filling, diameter, rule="hydromechanical", theta=3.0) == validate_min_slope(
        legacy_threshold, filling, diameter, rule="hydromechanical"
    )


def test_legacy_and_canonical_reference_values_are_bit_for_bit_stable() -> None:
    assert min_slope_hydromechanic(0.1, 0.4) == 0.00038250000000000003
    assert min_slope(0.1, 0.4) == pytest.approx(2.60706446773481, rel=1e-12)


def test_unknown_minimum_slope_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown minimum slope rule"):
        validate_min_slope(0.001, 0.1, 0.4, rule="unknown")  # type: ignore[arg-type]
