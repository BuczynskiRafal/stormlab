"""Boolean admissibility checks on filling, velocity and slope.

Ported verbatim from ``app/sa/core/valid_round.py``. The one interface change is
:func:`validate_min_slope`, which now takes an explicit ``rule`` selector instead
of hard-coding Imhoff; the default reproduces the previous behaviour exactly.

Three rules are exposed. ``imhoff`` and ``hydromechanical`` are frozen legacy
formulas; the latter approximates the wetted perimeter as ``pi * h`` and keeps
its historical rounded coefficient. ``hydromechanical_canonical`` is the unit
adapter over :func:`stormlab.hydraulics.round.min_slope`, which uses the true
hydraulic radius.
"""

from __future__ import annotations

import logging
import math
from typing import Literal, Mapping

from stormlab.hydraulics.round import (
    DEFAULT_G,
    DEFAULT_THETA,
    MAX_FILLING_RATIO,
    check_dimensions,
    common_diameters,
    max_filling,
    max_slopes,
    max_velocity_value,
    min_velocity_value,
    min_slope as round_min_slope,
)

logger = logging.getLogger(__name__)

MinSlopeRule = Literal["imhoff", "hydromechanical", "hydromechanical_canonical"]

MIN_SLOPE_RULES: tuple[str, ...] = ("imhoff", "hydromechanical", "hydromechanical_canonical")

HYDROMECHANIC_COEFFICIENT = 0.612e-3
"""Coefficient of the hydromechanical minimum slope criterion [-].

The rounded literal of ``4 * theta / g / 1000`` at ``theta = 1.5``, ``g = 9.81``
(exactly ``0.00061162``), carried over verbatim from ``app/sa/core/valid_round.py``
because the recommendation models were trained on the numbers it produces. It is
deliberately a literal and not a computation over the configured constants: the
two differ in the fourth significant digit."""


def validate_max_filling(filling: float, diameter: float, *, ratio: float = MAX_FILLING_RATIO) -> bool:
    """Whether the filling stays within the usable filling of the diameter.

    Args:
        filling: filling height [m].
        diameter: pipe diameter [m].
        ratio: relative filling at which the pipe reaches full capacity. It is a
            parameter because it is the single number behind ``ValMaxFill``, the
            minimum diameter search and ``NFilling``; the thesis backlog varies it
            and all three have to move together.
    """
    return filling <= max_filling(diameter, ratio)


def validate_filling(filling: float, diameter: float, *, ratio: float = MAX_FILLING_RATIO) -> bool:
    """Whether a filling/diameter pair is admissible at all.

    Unlike :func:`stormlab.hydraulics.round.check_dimensions` this never raises:
    it is applied row-wise to model output, where a non-numeric or out-of-range
    value is a fact about the network rather than a programming error.

    Args:
        filling: filling height [m].
        diameter: pipe diameter [m].
        ratio: forwarded to :func:`validate_max_filling`; the type, sign and
            standard-diameter checks do not depend on it.
    """
    if not isinstance(diameter, (float, int)) or not isinstance(filling, (float, int)):
        return False
    if 0 > filling or filling >= diameter:
        return False
    if not (common_diameters[0] <= diameter <= common_diameters[-1]):
        return False
    if not validate_max_filling(filling, diameter, ratio=ratio):
        return False
    return True


def validate_max_velocity(velocity: float) -> bool:
    """Whether the flow velocity stays at or below the maximum [m/s]."""
    return velocity <= max_velocity_value


def validate_min_velocity(velocity: float, v_min: float = min_velocity_value) -> bool:
    """Whether the flow velocity reaches the self-cleansing minimum [m/s].

    Source: PN-EN 752:2008 recommends at least 0.7 m/s so that solids do not
    settle at the bottom of the sewer. The threshold is a parameter because the
    literature also defends 0.8 m/s.
    """
    return velocity >= v_min


def check_slope(slope: float) -> bool:
    """Validate a slope value, raising rather than returning ``False``.

    Raises:
        TypeError: if ``slope`` is not an ``int`` or ``float``.
        ValueError: if ``slope`` is not strictly positive.
    """
    if not isinstance(slope, (int, float)):
        raise TypeError(f"slope must be a float or int, not {type(slope)}")
    if slope <= 0:
        raise ValueError(f"slope must be positive, not {slope}")
    return True


def validate_min_slope(
    slope: float,
    filling: float,
    diameter: float,
    *,
    rule: MinSlopeRule = "imhoff",
    theta: float = DEFAULT_THETA,
    g: float = DEFAULT_G,
) -> bool:
    """Whether the slope reaches the minimum required by the selected rule.

    Args:
        slope: slope of the channel [m/m].
        filling: filling height [m]; used by the hydromechanical rule only.
        diameter: diameter of the channel [m].
        rule: ``"imhoff"`` for the 1/D criterion, ``"hydromechanical"`` for
            the frozen approximate shear-stress criterion, or
            ``"hydromechanical_canonical"`` for the true-hydraulic-radius
            criterion.
        theta: shear stress [Pa], used only by
            ``"hydromechanical_canonical"``.
        g: specific weight [kN/m3], used only by
            ``"hydromechanical_canonical"``. The legacy rules ignore ``theta``
            and ``g``: in particular, their historical rounded coefficient is a
            preserved literal and is not recalculated from configured values.

    Raises:
        ValueError: if ``rule`` is not one of :data:`MIN_SLOPE_RULES`.
    """
    if rule == "imhoff":
        return slope >= min_slope_imhoff(diameter)
    if rule == "hydromechanical":
        return slope >= min_slope_hydromechanic(filling, diameter)
    if rule == "hydromechanical_canonical":
        return slope >= min_slope_canonical(filling, diameter, theta=theta, g=g)
    raise ValueError(f"Unknown minimum slope rule {rule!r}; expected one of {list(MIN_SLOPE_RULES)}")


def min_slope_canonical(
    filling: float,
    diameter: float,
    *,
    theta: float = DEFAULT_THETA,
    g: float = DEFAULT_G,
) -> float:
    """Canonical hydromechanical minimum slope [m/m].

    This is only a unit adapter: :func:`stormlab.hydraulics.round.min_slope`
    returns per mille, so the result is divided by 1000. An empty pipe is
    delegated unchanged to that function, which treats zero filling as 1 mm
    because the true empty-pipe hydraulic radius is undefined.
    """
    return round_min_slope(filling, diameter, theta, g) / 1000


def min_slope_imhoff(diameter: float) -> float:
    """Minimum slope after Imhoff: the reciprocal of the diameter in millimetres.

    Args:
        diameter: diameter of the channel [m].

    Returns:
        Minimal slope [m/m].
    """
    return 1.0 / (diameter * 1000)


def min_slope_hydromechanic(filling: float, diameter: float) -> float:
    """Minimum slope from the hydromechanical (shear stress) criterion.

    Takes no ``theta``/``g``: :data:`HYDROMECHANIC_COEFFICIENT` is the legacy
    rounded literal of ``4 * theta / g / 1000`` and is *not* recomputed from them
    (``4 * 1.5 / 9.81 / 1000 = 0.00061162``, the constant says ``0.000612``).
    The two arguments used to be accepted and silently ignored, which is a worse
    answer than not offering them: a caller configuring ``theta`` here would have
    got the default number back with no sign that it had been dropped. The
    configurable pair lives on :func:`stormlab.hydraulics.round.min_slope`, the
    criterion the feature pipeline actually computes with.

    Args:
        filling: filling height [m].
        diameter: diameter of the channel [m].

    Returns:
        Minimal slope [m/m]; ``inf`` for an empty pipe, where the criterion has
        no finite value.
    """
    r_b = diameter / 4.0  # Hydraulic radius for full flow [m]
    area = math.pi * (filling**2) / 4.0  # Cross-sectional area for partial flow [m2]
    perimeter = math.pi * filling  # Wetted perimeter for partial flow [m]
    r_hn = area / perimeter if perimeter != 0 else 0  # Hydraulic radius [m]
    if r_hn == 0:
        return float("inf")  # Avoid division by zero
    return HYDROMECHANIC_COEFFICIENT * (r_hn / r_b) * (1 / diameter)  # [m/m]


def validate_max_slope(slope: float, diameter: float, *, table: Mapping[str, float] | None = None) -> bool:
    """Whether the slope stays at or below the maximum for the diameter.

    Args:
        slope: slope of the channel [per mille].
        diameter: pipe diameter [m].
        table: maximum slope per diameter. Defaults to
            :data:`stormlab.hydraulics.round.max_slopes`, which is computed at the
            default ``theta``/``g``/``manning``; a caller running under configured
            constants passes :func:`stormlab.hydraulics.round.max_slope_table`
            built from those, because the frozen default would otherwise answer
            with values the configuration never asked for.

    Raises:
        TypeError: if ``slope`` or ``diameter`` is not numeric.
        ValueError: if ``slope`` is not positive, appears to use the wrong unit,
            or ``diameter`` is out of range or absent from the supplied table.
    """
    limits = max_slopes if table is None else table
    check_slope(slope)
    if slope > 1000:
        raise ValueError(f"slope {slope!r} looks like a percentage converted to per mille; expected per mille")
    check_dimensions(diameter, diameter)
    key = str(diameter)
    if key not in limits:
        raise ValueError(f"No maximum slope is defined for diameter {diameter!r}; expected a standard diameter")
    return slope <= limits[key]


__all__ = [
    "HYDROMECHANIC_COEFFICIENT",
    "MIN_SLOPE_RULES",
    "MinSlopeRule",
    "check_slope",
    "min_slope_canonical",
    "min_slope_hydromechanic",
    "min_slope_imhoff",
    "validate_filling",
    "validate_max_filling",
    "validate_max_slope",
    "validate_max_velocity",
    "validate_min_slope",
    "validate_min_velocity",
]
