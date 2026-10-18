import logging
import math
from typing import Union

from numpy import pi

logger = logging.getLogger(__name__)
common_diameters = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]


def check_dimensions(filling: float, diameter: float) -> bool:
    """
    Check if the given filling and diameter values are valid.

    Args:
        filling (float, int): the height of the filling in the pipe, in meters.
        diameter (float, int): the diameter of the pipe, in meters.

    Returns:
        bool: True if the values are valid, False otherwise.

    Raises:
        TypeError: If either the filling or the
        diameter is not an int or float.
        ValueError: If either the filling
        or the diameter is not between 0.2 and 2.0 meters.

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


def max_filling(diameter: float) -> float:
    """
    > Calculates the maximum filling of a rainwater
    drain with a circular cross-section.

    According to a methodology based on the Colebrook-White formula, the total
    capacity of the channel (100%), i.e., with total cross-sectional filling
    (100%), is already achieved at a relative
    filling of 4/D = 0.827 - in circular pipes,
    Kotowski, A., Kaźmierczak, B., & Dancewicz, A. (2010).
    Modelowanie opadów do wymiarowania kanalizacji.
    Polska Akademia Nauk. Instytut Podstawowych Problemów Techniki.

    Args:
        diameter (int, float): the diameter of the rainwater pipe [m].

    Return:
        maximum filling (float): he maximum pipe filling [m].
    """
    if isinstance(diameter, (int, float)):
        if 0.2 <= diameter <= 2.0:
            return 0.827 * diameter
        else:
            raise ValueError("Invalid diameter value.")
    else:
        raise TypeError("Diameter must be an int or float")


def max_velocity() -> float:
    """
    Maximum stormwater flow velocity in the sewer [m/s].

    Return:
         velocity (int): maximum stormwater flow velocity in pipe [m].
    """
    return 5


def min_velocity():
    """
    Minimum stormwater flow velocity in the sewer [m/s].

    Return:
         velocity (int): minimum stormwater flow velocity in pipe [m].
    """
    return 0.7


def max_depth():
    """
    The maximum depth of the sewer pipe [m].

    Return:
         depth (int): maximum depth of the sewer pipe [m].
    """
    return 8


def calc_area(filling: float, diameter: float) -> float:
    """
    Calculate the cross-sectional area of a pipe.
    The cross-sectional area through which the wastewater flows,
    active cross-section f,
    characterized by the filling h and the diameter of the pipe D.
    source: Biedugnis S., "Metody informatyczne w wodociągach i kanalizacji",
    Oficyna Wydawnicza Politechniki Warszawskiej, Warszawa 1998. - in polish.

    Args:
        filling (int, float): pipe filling height [m]
        diameter (int, float): pipe diameter [m]

    Return:
        area (int, float): cross-sectional
        area of the wetted part of the pipe [m2]
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
    """
    Calculate the circumference of a wetted part of pipe.
    source: Biedugnis S., "Metody informatyczne w wodociągach i kanalizacji",
    Oficyna Wydawnicza Politechniki Warszawskiej, Warszawa 1998. - in polish.

    Args:
        filling (int, float): pipe filling height [m]
        diameter (int, float): pipe diameter [m]

    Return:
        circumference (int, float): circumference of a wetted part of pipe
    """
    check_dimensions(filling, diameter)
    radius = diameter / 2
    chord = math.sqrt((radius**2 - (filling - radius) ** 2)) * 2
    alpha = math.degrees(math.acos((radius**2 + radius**2 - chord**2) / (2 * radius**2)))
    if filling > radius:
        return 2 * math.pi * radius - (alpha / 360 * 2 * math.pi * radius)
    return alpha / 360 * 2 * math.pi * radius


def calc_rh(filling: float, diameter: float) -> float:
    """
    Calculate the hydraulic radius Rh, i.e. the ratio of the cross-section f
    to the contact length of the sewage with the sewer wall,
    called the wetted circuit U.
    source: Biedugnis S., "Metody informatyczne w wodociągach i kanalizacji",
    Oficyna Wydawnicza Politechniki Warszawskiej, Warszawa 1998. - in polish.

    Args:
        filling (int, float): pipe filling height [m]
        diameter (int, float): pipe diameter [m]

    Return:
        Rh (int, float): hydraulic radius [m]
    """
    check_dimensions(filling, diameter)
    try:
        return calc_area(filling, diameter) / calc_u(filling, diameter)
    except ZeroDivisionError:
        return 0.0


def calc_velocity(
    filling: Union[float, int],
    diameter: Union[float, int],
    slope: Union[float, int],
) -> Union[float, int]:
    """
    Calculate the speed of the sewage flow in the sewer.

    Args:
        filling (int, float): pipe filling height [m]
        diameter (int, float): pipe diameter [m]
        slope (int, float): slope of the sewer bottom [‰]

    Return:
        v (int, float): sewage flow velocity in the sewer [m/s]
    """
    check_dimensions(filling, diameter)
    slope = slope / 1000
    return 1 / 0.013 * calc_rh(filling, diameter) ** (2 / 3) * (slope**0.5)


def min_slope(filling: float, diameter: float, theta: float = 1.5, g: float = 9.81) -> float:
    """
    Get the minimal slope for sewer pipe.
    If the pipe  filling is greater than 0.3,
    then the minimum slope is 1/d, otherwise it's 0.25/rh

    source: Suligowski : Samooczyszczanie przewodów kanalizacyjnych.
    Instal 2010, nr 2, s. 48-53. - in polish
    source: https://seidel-przywecki.eu/2021/06/04/obliczenia-
    hydraulicznych-kanalow-sciekowych-i-deszczowych/

    Args:
        filling (int, float): pipe filling height [m]
        diameter (int, float): pipe diameter [m]
        theta (float, optional): theta value.
        Defaults to 1.5, shear stress [Pa].
        g (float, optional): specific gravity
        of liquid (water/wastewater) [N/m3].

    Return:
        slope (int, float): The minimum slope of the channel [‰]
    """
    check_dimensions(filling, diameter)
    filling = max(0.001, filling)
    return 4 * (theta / g) * ((diameter / 4) / calc_rh(filling, diameter)) * (1 / diameter)


def max_slope(diameter: Union[float, int]) -> Union[float, int]:
    """
    Calculates the maximum slope for a given pipe diameter.

    The function starts with an initial slope based
    on the diameter, and then iteratively
    adjusts the slope until the calculated velocity
    matches the maximum velocity for the
    pipeline.

    Args:
        diameter (Union[float, int]): pipe diameter [m].

    Returns:
        Union[float, int]: The maximum slope that can be achieved for the pipe.
    """
    check_dimensions(diameter, diameter)
    start_slope = min_slope(diameter, diameter)
    slope = start_slope
    v_max = max_velocity()
    v_clc = 0.0
    while round(v_clc, 2) != float(v_max):
        v_clc = calc_velocity(diameter, diameter, slope)
        if v_clc < v_max:
            slope += start_slope
        else:
            start_slope = start_slope / 2
            slope -= start_slope
    return slope


max_slopes = {str(dim): max_slope(dim) for dim in common_diameters}
max_velocity_value = max_velocity()
min_velocity_value = min_velocity()  # type: ignore
max_depth_value = max_depth()  # type: ignore
