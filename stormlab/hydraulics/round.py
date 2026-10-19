"""Circular pipe geometry and the slope envelope of a standard sewer diameter.

Ported verbatim from ``app/sa/core/round.py``: every formula, branch and unit is
kept as it was, because the recommendation models were trained on the numbers
these functions produce. The only additions are explicit ``theta``/``g``
parameters on the slope functions (defaulting to the values that were hard-coded
before), :data:`MANNING_FULL_PIPE_COEFFICIENT`, and type annotations.

Units are not uniform and that is intentional: :func:`calc_velocity` and
:func:`min_slope` speak per mille, while the Manning family in
:mod:`stormlab.hydraulics.formulas` speaks m/m.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

from numpy import pi

logger = logging.getLogger(__name__)

common_diameters: list[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
"""Standard sewer diameters [m], ascending. Also the accepted diameter range."""

DEFAULT_THETA = 1.5
"""Shear stress used by the hydromechanical minimum slope criterion [Pa]."""

DEFAULT_G = 9.81
"""Specific weight of water ``gamma = rho * g`` [kN/m3].

With shear stress ``theta`` expressed in Pa (N/m2), ``theta / gamma`` is
numerically a length in millimetres. That unit convention is why
:func:`min_slope` returns per mille without a separate factor of 1000. The name
``g`` and the value are retained for API and numerical compatibility.
"""

MANNING_FULL_PIPE_COEFFICIENT = math.pi / 4 * 0.25 ** (2 / 3)
"""Geometry coefficient in the full-pipe Manning capacity equation.

For a circular pipe, ``A = pi * D**2 / 4`` and ``R = D / 4``. Therefore
``A * R**(2/3) = (pi / 4) * 0.25**(2/3) * D**(8/3)``. Hydraulic literature
often rounds this coefficient to 0.3117.
"""

MAX_FILLING_RATIO = 0.827
"""Relative filling at which a circular pipe reaches full capacity (Colebrook-White)."""

MANNING_COEFFICIENT = 0.013
"""Manning roughness of the pipe material assumed by :func:`calc_velocity`."""

MAX_VELOCITY = 5.0
"""Maximum admissible stormwater flow velocity in the sewer [m/s]."""

MIN_VELOCITY = 0.7
"""Self-cleansing minimum velocity per PN-EN 752:2008 [m/s]."""

MAX_DEPTH = 8.0
"""Maximum admissible installation depth of a sewer pipe [m].

A **default**, not the value in force. Since plan16 §2 P0 point 5 the depth
ceiling is ``config.max_depth``, forwarded to
:func:`stormlab.features.conduits.extract_conduit_features` and to the leakage
audit's tolerances, because it is a design rule that a variant may move rather
than a constant of the physics — and it moves ``ValDepth``, the depth
normalisation and the cover normalisation together. This value is what the
configuration ships and what a caller with no configuration gets.
"""


def check_dimensions(filling: float, diameter: float) -> bool:
    """Validate a filling/diameter pair, raising rather than returning ``False``.

    Args:
        filling: height of the filling in the pipe [m].
        diameter: pipe diameter [m].

    Returns:
        ``True`` when both values are admissible.

    Raises:
        TypeError: if either value is not an ``int`` or ``float``.
        ValueError: if filling exceeds the diameter, or either value is outside
            the admissible range (filling 0–2.0 m, diameter 0.2–2.0 m).
    """
    if not isinstance(filling, (int, float)):
        raise TypeError("Filling must be an int or float")
    if not isinstance(diameter, (int, float)):
        raise TypeError("Diameter must be an int or float")
    if filling > diameter:
        raise ValueError("Filling must be less than or equal to the diameter")
    if not (0 <= filling <= 2.0 and 0.2 <= diameter <= 2.0):
        raise ValueError(
            """Value out of bounds. Filling must be between 0 and 2.0
            meters and diameter must be between 0.2 and 2.0 meters"""
        )
    return True


def max_filling(diameter: float, ratio: float = MAX_FILLING_RATIO) -> float:
    """Maximum usable filling height of a circular rainwater drain [m].

    Total channel capacity is already reached at a relative filling of 0.827
    (Kotowski, Kazmierczak & Dancewicz 2010, after Colebrook-White), so the
    usable filling is a fraction of the diameter rather than the diameter.

    Args:
        diameter: pipe diameter [m].
        ratio: relative filling at full capacity.

    Raises:
        TypeError: if ``diameter`` is not numeric.
        ValueError: if ``diameter`` is outside 0.2–2.0 m.
    """
    if isinstance(diameter, (int, float)):
        if 0.2 <= diameter <= 2.0:
            return ratio * diameter
        raise ValueError("Invalid diameter value.")
    raise TypeError("Diameter must be an int or float")


def max_velocity() -> float:
    """Maximum stormwater flow velocity in the sewer [m/s]."""
    return MAX_VELOCITY


def min_velocity() -> float:
    """Minimum (self-cleansing) stormwater flow velocity in the sewer [m/s]."""
    return MIN_VELOCITY


def max_depth() -> float:
    """Default maximum depth of the sewer pipe [m]; see :data:`MAX_DEPTH`."""
    return MAX_DEPTH


def calc_area(filling: float, diameter: float) -> float:
    """Wetted cross-sectional area of a partially filled circular pipe [m2].

    Source: Biedugnis S., "Metody informatyczne w wodociagach i kanalizacji",
    Oficyna Wydawnicza Politechniki Warszawskiej, Warszawa 1998.

    Args:
        filling: pipe filling height [m].
        diameter: pipe diameter [m].

    Raises:
        TypeError, ValueError: as raised by :func:`check_dimensions`.
    """
    check_dimensions(filling, diameter)
    radius = diameter / 2
    chord = math.sqrt((radius**2 - ((filling - radius) ** 2))) * 2
    alpha = math.acos((radius**2 + radius**2 - chord**2) / (2 * radius**2))
    if filling > radius:
        return pi * radius**2 - (1 / 2 * (alpha - math.sin(alpha)) * radius**2)
    elif filling == radius:
        return pi * radius**2 / 2
    elif filling == diameter:
        return pi * radius**2
    else:
        return 1 / 2 * (alpha - math.sin(alpha)) * radius**2


def calc_u(filling: float, diameter: float) -> float:
    """Wetted perimeter of a partially filled circular pipe [m].

    The central angle is taken in degrees here; the Manning family in
    :mod:`stormlab.hydraulics.formulas` uses radians for the same quantity.

    Args:
        filling: pipe filling height [m].
        diameter: pipe diameter [m].

    Raises:
        TypeError, ValueError: as raised by :func:`check_dimensions`.
    """
    check_dimensions(filling, diameter)
    radius = diameter / 2
    chord = math.sqrt((radius**2 - (filling - radius) ** 2)) * 2
    alpha = math.degrees(math.acos((radius**2 + radius**2 - chord**2) / (2 * radius**2)))
    if filling > radius:
        return 2 * math.pi * radius - (alpha / 360 * 2 * math.pi * radius)
    return alpha / 360 * 2 * math.pi * radius


def calc_rh(filling: float, diameter: float) -> float:
    """Hydraulic radius, the ratio of wetted area to wetted perimeter [m].

    A zero perimeter (empty pipe) yields 0.0 rather than propagating a
    ``ZeroDivisionError``.

    Args:
        filling: pipe filling height [m].
        diameter: pipe diameter [m].

    Raises:
        TypeError, ValueError: as raised by :func:`check_dimensions`.
    """
    check_dimensions(filling, diameter)
    try:
        return calc_area(filling, diameter) / calc_u(filling, diameter)
    except ZeroDivisionError:
        return 0.0


def calc_velocity(filling: float, diameter: float, slope: float, manning: float = MANNING_COEFFICIENT) -> float:
    """Sewage flow velocity from Manning's equation [m/s].

    Args:
        filling: pipe filling height [m].
        diameter: pipe diameter [m].
        slope: slope of the sewer bottom, in **per mille**.
        manning: Manning roughness of the pipe material.

    Raises:
        TypeError, ValueError: as raised by :func:`check_dimensions`.
    """
    check_dimensions(filling, diameter)
    slope = slope / 1000
    return 1 / manning * calc_rh(filling, diameter) ** (2 / 3) * (slope**0.5)


def min_slope(filling: float, diameter: float, theta: float = DEFAULT_THETA, g: float = DEFAULT_G) -> float:
    """Minimum sewer slope from the hydromechanical (shear stress) criterion.

    Sources: Suligowski, "Samooczyszczanie przewodow kanalizacyjnych", Instal
    2010 nr 2, s. 48-53; seidel-przywecki.eu (2021-06-04).

    An empty pipe is treated as filled to 1 mm, because the hydraulic radius of a
    truly empty pipe is zero and the criterion is undefined there.

    Args:
        filling: pipe filling height [m].
        diameter: pipe diameter [m].
        theta: shear stress [Pa].
        g: specific weight of water ``gamma = rho * g`` [kN/m3]. With
            ``theta`` in Pa, their numerical ratio is in millimetres, which
            makes the returned slope per mille without another scale factor.

    Returns:
        The minimum slope of the channel, in **per mille**.

    Raises:
        TypeError, ValueError: as raised by :func:`check_dimensions`.
    """
    check_dimensions(filling, diameter)
    filling = max(0.001, filling)
    return 4 * (theta / g) * ((diameter / 4) / calc_rh(filling, diameter)) * (1 / diameter)


def max_slope(
    diameter: float,
    theta: float = DEFAULT_THETA,
    g: float = DEFAULT_G,
    manning: float = MANNING_COEFFICIENT,
) -> float:
    """Largest slope at which a full pipe still stays at the maximum velocity.

    Starts from the minimum slope of the full pipe and bisects until the computed
    velocity rounds to :func:`max_velocity`.

    Args:
        diameter: pipe diameter [m].
        theta: shear stress [Pa] forwarded to :func:`min_slope`.
        g: specific weight of water [kN/m3] forwarded to :func:`min_slope`.
        manning: Manning roughness forwarded to :func:`calc_velocity`.

    Returns:
        The maximum slope of the channel, in **per mille**.

    Raises:
        TypeError, ValueError: as raised by :func:`check_dimensions`.
    """
    check_dimensions(diameter, diameter)
    start_slope = min_slope(diameter, diameter, theta=theta, g=g)
    slope = start_slope
    v_max = max_velocity()
    v_clc = 0.0
    while round(v_clc, 2) != float(v_max):
        v_clc = calc_velocity(diameter, diameter, slope, manning=manning)
        if v_clc < v_max:
            slope += start_slope
        else:
            start_slope = start_slope / 2
            slope -= start_slope
    return slope


def max_slope_table(
    *,
    theta: float = DEFAULT_THETA,
    g: float = DEFAULT_G,
    manning: float = MANNING_COEFFICIENT,
    diameters: Sequence[float] = tuple(common_diameters),
) -> dict[str, float]:
    """Maximum slope per diameter, keyed by ``str(diameter)``, computed from the parameters.

    The module-level :data:`max_slopes` is this table at the default constants.
    A caller that configures ``theta``, ``g`` or ``manning`` differently has to
    build its own table, because a dictionary frozen at import time answers with
    the defaults whatever it was asked — which is the one way a configured
    constant can fail to reach the number it names.
    """
    return {str(dim): max_slope(dim, theta=theta, g=g, manning=manning) for dim in diameters}


max_slopes: dict[str, float] = max_slope_table()
"""Maximum slope per standard diameter at the default ``theta``/``g``/``manning``.

Kept as the default argument of :func:`stormlab.hydraulics.validators.validate_max_slope`;
anything running under configured constants passes its own
:func:`max_slope_table` instead. A non-standard diameter has no entry at all.
"""

max_velocity_value: float = max_velocity()
min_velocity_value: float = min_velocity()
max_depth_value: float = max_depth()
