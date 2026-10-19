"""Manning-family hydraulics used by the conduit feature pipeline.

Ported verbatim from ``HydraulicCalculationsService`` in
``app/sa/core/services.py``, turned from static methods into module functions.
The formulas, branch order and — importantly — the exception *messages* are
unchanged: ``min_conduit_diameter`` substring-matches ``"Slope is too small"`` to
decide whether to retry with a steeper slope, so the message is control flow, not
diagnostics.

This family takes slope in **m/m** and computes the central angle in radians,
unlike the geometry family in :mod:`stormlab.hydraulics.round`. Both are needed:
the feature pipeline calls into each of them.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, TypeVar

from stormlab.hydraulics.round import DEFAULT_G, DEFAULT_THETA, MANNING_COEFFICIENT, MAX_FILLING_RATIO, common_diameters
from stormlab.hydraulics.validators import (
    MinSlopeRule,
    validate_filling,
    validate_max_slope,
    validate_min_slope,
)

logger = logging.getLogger(__name__)

FILLING_CALCULATION_STEP = 0.001
"""Step of the iterative filling search [m]."""

FILLING_CALCULATION_MAX_ITER = 10000
"""Iteration cap of the filling search; 10000 steps of 1 mm cover a 10 m pipe."""

SLOPE_TOO_SMALL_MESSAGE = "Slope is too small"
"""Load-bearing: matched as a substring by the minimum diameter search."""

SLOPE_TOO_LARGE_MESSAGE = "Slope exceeds maximum allowed value"

FILLING_EXCEEDS_DIAMETER_MESSAGE = "Filling exceeds diameter without achieving desired flow"
"""Load-bearing: the minimum diameter search reads it as "this diameter is too small"."""

_F = TypeVar("_F", bound=Callable[..., float])


def validate_inputs(func: _F) -> _F:
    """Short-circuit a geometry function to 0.0 for an inadmissible pipe state.

    An invalid filling, a non-positive diameter or an empty pipe all mean "no
    wetted cross-section", which the callers treat as zero rather than as an
    error.

    The gate is what makes ``max_filling_ratio`` a keyword of every decorated
    function: the ratio decides where "full" is, so a run that moves it has to
    move this short-circuit with it. The geometry itself does not use the ratio,
    but it forwards it, so the whole chain answers under one value.
    """

    def wrapper(
        filling: float, diameter: float, *args: object, max_filling_ratio: float = MAX_FILLING_RATIO, **kwargs: object
    ) -> float:
        if not validate_filling(filling, diameter, ratio=max_filling_ratio) or diameter <= 0:
            return 0.0
        if filling == 0:
            return 0.0
        return func(filling, diameter, *args, max_filling_ratio=max_filling_ratio, **kwargs)

    return wrapper  # type: ignore[return-value]


def calc_filling_percentage(filling: float, diameter: float, *, max_filling_ratio: float = MAX_FILLING_RATIO) -> float:
    """Filling as a percentage of the diameter [%]; 0.0 when inadmissible or empty."""
    if not validate_filling(filling, diameter, ratio=max_filling_ratio):
        return 0.0
    if filling == 0:
        return 0.0
    return (filling / diameter) * 100.0


@validate_inputs
def calc_area(filling: float, diameter: float, *, max_filling_ratio: float = MAX_FILLING_RATIO) -> float:
    """Wetted cross-sectional area of a circular pipe [m2]."""
    radius = diameter / 2.0

    chord = 2.0 * math.sqrt(radius**2 - (filling - radius) ** 2)
    alpha = math.acos((2.0 * radius**2 - chord**2) / (2.0 * radius**2))

    if filling > radius:
        area = math.pi * radius**2 - 0.5 * (alpha - math.sin(alpha)) * radius**2
    elif filling == radius:
        area = 0.5 * math.pi * radius**2
    elif filling == diameter:
        area = math.pi * radius**2
    else:
        area = 0.5 * (alpha - math.sin(alpha)) * radius**2

    return area


@validate_inputs
def calc_u(filling: float, diameter: float, *, max_filling_ratio: float = MAX_FILLING_RATIO) -> float:
    """Wetted perimeter of a circular pipe [m], with the central angle in radians."""
    radius = diameter / 2.0

    chord = 2.0 * math.sqrt(radius**2 - (filling - radius) ** 2)
    alpha = math.acos((2.0 * radius**2 - chord**2) / (2.0 * radius**2))

    if filling > radius:
        if filling == diameter:
            perimeter = 2 * math.pi * radius
        else:
            perimeter = 2 * math.pi * radius - alpha * radius
    else:
        perimeter = alpha * radius

    return perimeter


@validate_inputs
def calc_rh(filling: float, diameter: float, *, max_filling_ratio: float = MAX_FILLING_RATIO) -> float:
    """Hydraulic radius ``A / U`` [m]; 0.0 for a zero wetted perimeter."""
    area = calc_area(filling, diameter, max_filling_ratio=max_filling_ratio)
    perimeter = calc_u(filling, diameter, max_filling_ratio=max_filling_ratio)
    return area / perimeter if perimeter else 0.0


@validate_inputs
def calc_velocity(
    filling: float,
    diameter: float,
    slope: float,
    *,
    manning: float = MANNING_COEFFICIENT,
    min_slope_rule: MinSlopeRule = "imhoff",
    theta: float = DEFAULT_THETA,
    g: float = DEFAULT_G,
    max_filling_ratio: float = MAX_FILLING_RATIO,
) -> float:
    """Flow velocity from Manning's equation [m/s].

    Args:
        filling: filling height [m].
        diameter: diameter of the channel [m].
        slope: slope of the channel [m/m].
        manning: Manning roughness coefficient of the pipe material.
        min_slope_rule: which minimum slope criterion gates the computation.
        theta: shear stress forwarded to the canonical minimum-slope rule only.
        g: specific weight forwarded to the canonical minimum-slope rule only.
            The legacy rules ignore both values and retain their literal.

    Raises:
        ValueError: if the slope is above the maximum for the diameter or below
            the minimum required by ``min_slope_rule``.
    """
    if not validate_max_slope(slope, diameter):
        raise ValueError(SLOPE_TOO_LARGE_MESSAGE)
    if not validate_min_slope(slope, filling, diameter, rule=min_slope_rule, theta=theta, g=g):
        raise ValueError(SLOPE_TOO_SMALL_MESSAGE)

    rh = calc_rh(filling, diameter, max_filling_ratio=max_filling_ratio)

    if rh == 0:
        return 0.0
    return (1.0 / manning) * (rh ** (2.0 / 3.0)) * math.sqrt(slope)


@validate_inputs
def calc_flow(
    filling: float,
    diameter: float,
    slope: float,
    *,
    manning: float = MANNING_COEFFICIENT,
    min_slope_rule: MinSlopeRule = "imhoff",
    theta: float = DEFAULT_THETA,
    g: float = DEFAULT_G,
    max_filling_ratio: float = MAX_FILLING_RATIO,
) -> float:
    """Flow rate ``Q = A * v`` [m3/s].

    Args:
        filling: filling height [m].
        diameter: diameter of the channel [m].
        slope: slope of the channel [m/m].
        manning: Manning roughness coefficient of the pipe material.
        min_slope_rule: which minimum slope criterion gates the computation.
        theta: shear stress forwarded to the canonical minimum-slope rule only.
        g: specific weight forwarded to the canonical minimum-slope rule only.
            The legacy rules ignore both values and retain their literal.

    Raises:
        ValueError: propagated from :func:`calc_velocity` when the slope is out
            of range. The warnings logged just above are historical: the same
            conditions raise a moment later inside :func:`calc_velocity`.
    """
    if not validate_max_slope(slope, diameter):
        logger.warning(SLOPE_TOO_LARGE_MESSAGE)
    if not validate_min_slope(slope, filling, diameter, rule=min_slope_rule, theta=theta, g=g):
        logger.warning(SLOPE_TOO_SMALL_MESSAGE)

    area = calc_area(filling, diameter, max_filling_ratio=max_filling_ratio)
    velocity = calc_velocity(
        filling,
        diameter,
        slope,
        manning=manning,
        min_slope_rule=min_slope_rule,
        theta=theta,
        g=g,
        max_filling_ratio=max_filling_ratio,
    )
    return area * velocity


def calc_filling(
    q: float,
    diameter: float,
    slope: float,
    *,
    manning: float = MANNING_COEFFICIENT,
    min_slope_rule: MinSlopeRule = "imhoff",
    theta: float = DEFAULT_THETA,
    g: float = DEFAULT_G,
    max_filling_ratio: float = MAX_FILLING_RATIO,
    step: float = FILLING_CALCULATION_STEP,
    max_iter: int = FILLING_CALCULATION_MAX_ITER,
) -> float:
    """Filling height [m] needed to convey ``q``, found by 1 mm increments.

    Args:
        q: flow rate [m3/s].
        diameter: diameter of the channel [m].
        slope: slope of the channel [m/m].
        manning: Manning roughness coefficient of the pipe material.
        min_slope_rule: which minimum slope criterion gates the flow computation.
        theta: shear stress forwarded to the canonical minimum-slope rule only.
        g: specific weight forwarded to the canonical minimum-slope rule only.
            The legacy rules ignore both values and retain their literal.
        max_filling_ratio: relative filling at which the pipe is considered full;
            the search stops finding flow above it.
        step: increment of the search [m].
        max_iter: iteration cap of the search.

    Raises:
        ValueError: if the diameter is outside the standard range, the flow is
            negative, or the pipe fills up without reaching ``q``. Also
            propagated from :func:`calc_flow` for an out-of-range slope.
    """
    if not (common_diameters[0] <= diameter <= common_diameters[-1]):
        raise ValueError("Diameter out of range for common diameters")
    if q < 0:
        raise ValueError("Invalid flow rate, must be positive")
    if q == 0:
        return 0.0

    filling = 0.0

    for _ in range(max_iter):
        if filling > diameter:
            break
        flow = calc_flow(
            filling,
            diameter,
            slope,
            manning=manning,
            min_slope_rule=min_slope_rule,
            theta=theta,
            g=g,
            max_filling_ratio=max_filling_ratio,
        )
        if flow >= q:
            break
        filling += step

    if filling > diameter:
        raise ValueError(FILLING_EXCEEDS_DIAMETER_MESSAGE)
    return filling


__all__ = [
    "FILLING_CALCULATION_MAX_ITER",
    "FILLING_CALCULATION_STEP",
    "FILLING_EXCEEDS_DIAMETER_MESSAGE",
    "SLOPE_TOO_LARGE_MESSAGE",
    "SLOPE_TOO_SMALL_MESSAGE",
    "calc_area",
    "calc_filling",
    "calc_filling_percentage",
    "calc_flow",
    "calc_rh",
    "calc_u",
    "calc_velocity",
    "validate_inputs",
]
