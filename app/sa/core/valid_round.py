import logging
from .round import (
    max_filling,
    max_slope,
    max_slopes,
    max_velocity_value,
    min_slope,
    min_velocity_value,
    check_dimensions,
    common_diameters,
)

logger = logging.getLogger(__name__)


def validate_max_filling(filling: float, diameter: float) -> bool:
    """
    Check that the maximum filling is not exceeded.

    Args:
        filling (float): Filling height [m].
        diameter (float): Diameter of the channel [m].

    Returns:
        bool: True if filling is within range, False otherwise.
    """
    return filling <= max_filling(diameter)


def validate_filling(filling: float, diameter: float) -> bool:
    """
    Validates the filling height.

    Args:
        filling (float): Filling height [m].
        diameter (float): Diameter of the channel [m].

    Returns:
        bool: True if filling is valid, else False.
    """
    if not isinstance(diameter, (float, int)) or not isinstance(filling, (float, int)):
        return False
    if 0 > filling or filling >= diameter:
        return False
    if not (common_diameters[0] <= diameter <= common_diameters[-1]):
        return False
    if not validate_max_filling(filling, diameter):
        return False
    return True


def validate_max_velocity(velocity: float) -> bool:
    """
    Check that the maximum velocity is not exceeded.

    Args:
        velocity (int, float): flow velocity in the sewer [m/s].

    Return:
        bool: given velocity value is lower than the maximum velocity.
    """
    return velocity <= max_velocity_value


def validate_min_velocity(velocity: float) -> bool:
    """
    Check that the minimum velocity is not exceeded.
    For hydraulic calculations of sewers and rainwater collectors,
    it is recommended to adopt a self-cleaning velocity
    of not less than 0.7 m - s-1 to prevent solids from settling at the bottom of the sewer.
    source: PN-EN 752:2008 Zewnętrzne systemy kanalizacyjne - in polish

    Args:
        velocity (int, float): flow velocity in the sewer [m/s].

    Return:
        bool: given velocity value is higher than the minimum velocity.
    """
    return velocity >= min_velocity_value  # type: ignore


def check_slope(slope: float) -> bool:
    """
    Check passed value for slope.

    Args:
        slope (float): the slope of a pipe.

    Returns:
        bool: True if the slope is positive, False otherwise.

    Raises:
        TypeError: if slope is not a float or int.
        ValueError: if slope is not positive.
    """
    if not isinstance(slope, (int, float)):
        raise TypeError(f"slope must be a float or int, not {type(slope)}")
    if slope <= 0:
        raise ValueError(f"slope must be positive, not {slope}")
    return True


def validate_min_slope(slope: float, filling: float, diameter: float, theta: float = 1.5, g: float = 9.81) -> bool:
    """
    Check that the minimum slope is not exceeded using the shear-stress criterion (tau = 1.5 Pa).

    Args:
        slope (float): Slope of the channel [m/m].
        filling (float): Filling height [m].
        diameter (float): Diameter of the channel [m].
        theta (float, optional): Critical shear stress [Pa]. Defaults to 1.5.
        g (float, optional): Specific weight factor [kN/m³]. Defaults to 9.81.

    Returns:
        bool: True if slope is above the minimum required, False otherwise.
    """
    if diameter <= 0:
        return False
    effective_filling = min(max(float(filling), 0.3 * float(diameter)), float(diameter))
    return slope >= min_slope_hydromechanic(effective_filling, diameter, theta=theta, g=g)


def min_slope_imhoff(diameter: float) -> float:
    """
    Calculate the minimal slope using Imhoff's formula.

    Imhoff defines the minimum slope as the reciprocal of the pipe diameter in millimeters.

    Args:
        diameter (float): Diameter of the channel [m].

    Returns:
        float: Minimal slope [m/m], calculated as 1 / D[mm].
    """
    return 1.0 / (diameter * 1000)


def min_slope_hydromechanic(filling: float, diameter: float, theta: float = 1.5, g: float = 9.81) -> float:
    """
    Calculate the minimal slope [m/m] using the shear-stress criterion (tau = 1.5 Pa).

    Delegates to round.min_slope (which returns per-mille [‰]) and converts to [m/m].

    Args:
        filling (float): Filling height [m].
        diameter (float): Diameter of the channel [m].
        theta (float, optional): Shear stress [Pa]. Defaults to 1.5.
        g (float, optional): Acceleration due to gravity [m/s²]. Defaults to 9.81.

    Returns:
        float: Minimal slope [m/m].
    """
    if filling <= 0 or diameter <= 0:
        return float("inf")
    clamped_filling = min(float(filling), float(diameter))
    try:
        return float(min_slope(clamped_filling, float(diameter), theta=theta, g=g)) / 1000.0
    except ValueError:
        return float("inf")


def validate_max_slope(slope: float, diameter: float) -> bool:
    """
    Check that the maximum slope is not exceeded.
    """
    if check_slope(slope) and check_dimensions(diameter, diameter):
        max_allowed = max_slopes.get(str(diameter))
        if max_allowed is None:
            max_allowed = float(max_slope(diameter))
        return slope <= max_allowed
    return False

