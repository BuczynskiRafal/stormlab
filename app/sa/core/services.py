"""
Utility services for hydraulic calculations, simulation running, and trace analysis.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, List

import pandas as pd

if TYPE_CHECKING:
    from .data import DataManager
from pyswmm import Simulation
from swmmio.utils.functions import trace_from_node

from .constants import (
    FILLING_CALCULATION_MAX_ITER,
    FILLING_CALCULATION_STEP,
    MANNING_COEFFICIENT,
)
from .round import common_diameters
from .valid_round import (
    validate_filling,
    validate_max_slope,
    validate_min_slope,
)

logger = logging.getLogger(__name__)


def validate_inputs(func):
    """Decorator to validate filling and diameter inputs."""

    def wrapper(filling, diameter, *args, **kwargs):
        if not validate_filling(filling, diameter) or diameter <= 0:
            return 0.0
        if filling == 0:
            return 0.0
        return func(filling, diameter, *args, **kwargs)

    return wrapper


class HydraulicCalculationsService:
    @staticmethod
    def calc_filling_percentage(filling: float, diameter: float) -> float:
        """
        Returns the percentage filling of the pipe.

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].

        Returns:
            float: Percentage filling [%]. Returns 0.0 if filling is invalid or 0.
        """
        if not validate_filling(filling, diameter):
            return 0.0
        if filling == 0:
            return 0.0
        return (filling / diameter) * 100.0

    @staticmethod
    @validate_inputs
    def calc_area(filling: float, diameter: float) -> float:
        """
        Computes the wetted cross-sectional area (m^2) for a circular pipe filled to height h.

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].

        Returns:
            float: Wetted cross-sectional area [m^2].
        """
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

    @staticmethod
    @validate_inputs
    def calc_u(filling: float, diameter: float) -> float:
        """
        Computes the wetted perimeter U for a circular pipe filled to 'filling'.

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].

        Returns:
            float: Wetted perimeter [m].
        """
        radius = diameter / 2.0

        # Obliczenie długości cięciwy
        chord = 2.0 * math.sqrt(radius**2 - (filling - radius) ** 2)
        # Obliczenie kąta centralnego w radianach
        alpha = math.acos((2.0 * radius**2 - chord**2) / (2.0 * radius**2))

        if filling > radius:
            if filling == diameter:
                perimeter = 2 * math.pi * radius  # Pełne wypełnienie
            else:
                perimeter = 2 * math.pi * radius - alpha * radius
        else:
            perimeter = alpha * radius

        return perimeter

    @staticmethod
    @validate_inputs
    def calc_rh(filling: float, diameter: float) -> float:
        """
        Computes the hydraulic radius Rh = A / U.

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].

        Returns:
            float: Hydraulic radius [m].
        """

        area = HydraulicCalculationsService.calc_area(filling, diameter)
        perimeter = HydraulicCalculationsService.calc_u(filling, diameter)
        rh = area / perimeter if perimeter else 0.0

        return rh

    @staticmethod
    @validate_inputs
    def calc_velocity(filling: float, diameter: float, slope: float) -> float:
        """
        Calculates the flow velocity using Manning's equation.

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].
            slope (float): Slope of the channel [m/m].

        Returns:
            float: Flow velocity [m/s].

        Raises:
            ValueError: If slope is too small or exceeds maximum allowed value.
        """
        if not validate_max_slope(slope, diameter):
            raise ValueError("Slope exceeds maximum allowed value")
        if not validate_min_slope(slope, filling, diameter):
            raise ValueError("Slope is too small")

        rh = HydraulicCalculationsService.calc_rh(filling, diameter)

        if rh == 0:
            return 0.0
        velocity = (1.0 / MANNING_COEFFICIENT) * (rh ** (2.0 / 3.0)) * math.sqrt(slope)
        return velocity

    @staticmethod
    @validate_inputs
    def calc_flow(filling: float, diameter: float, slope: float) -> float:
        """
        Computes flow rate Q [m³/s].

        Args:
            filling (float): Filling height [m].
            diameter (float): Diameter of the channel [m].
            slope (float): Slope of the channel [m/m].

        Returns:
            float: Flow rate [m³/s].

        Raises:
            ValueError: If slope is too small or exceeds maximum allowed.
        """
        if not validate_max_slope(slope, diameter):
            # raise ValueError("Slope exceeds maximum allowed value")
            logger.warning("Slope exceeds maximum allowed value")
        if not validate_min_slope(slope, filling, diameter):
            # raise ValueError("Slope is too small")
            logger.warning("Slope is too small")

        area = HydraulicCalculationsService.calc_area(filling, diameter)
        velocity = HydraulicCalculationsService.calc_velocity(filling, diameter, slope)
        flow = area * velocity  # Removed the 1000.0 multiplier since we're now using m³/s instead of dm³/s

        return flow

    @staticmethod
    def calc_filling(q: float, diameter: float, slope: float) -> float:
        """
        Iteratively approximates the filling height (m) needed to achieve flow q [m³/s].

        Args:
            q (float): Flow rate [m³/s].
            diameter (float): Diameter of the channel [m].
            slope (float): Slope of the channel [m/m].

        Returns:
            float: Filling height [m].

        Raises:
            ValueError: If filling exceeds diameter without achieving desired flow.
        """
        if not (common_diameters[0] <= diameter <= common_diameters[-1]):
            raise ValueError("Diameter out of range for common diameters")
        if q < 0:
            raise ValueError("Invalid flow rate, must be positive")
        if q == 0:
            return 0.0

        filling = 0.0

        for _ in range(FILLING_CALCULATION_MAX_ITER):
            if filling > diameter:
                break
            flow = HydraulicCalculationsService.calc_flow(filling, diameter, slope)  # już [m³/s]
            if flow >= q:
                break
            filling += FILLING_CALCULATION_STEP

        if filling > diameter:
            raise ValueError("Filling exceeds diameter without achieving desired flow")
        return filling


class SimulationRunnerService:
    """Class responsible for running the simulation using PySWMM."""

    def __init__(self, inp_path: str):
        self.inp_path = inp_path

    def run_simulation(self) -> None:
        """Runs the PySWMM simulation in a loop to update model values."""
        with Simulation(self.inp_path) as sim:
            for _ in sim:
                pass


class TraceAnalysisService:
    """
    Class contains the logic for analyzing flows and overflows in the SWMM network.
    """

    def __init__(self, model: "DataManager"):  # noqa: F821
        self.model = model

    def all_traces(self):
        """
        Returns the route (trace) for each outfall in the model.
        """
        outfalls = self.model.inp.outfalls.index
        return {outfall: trace_from_node(self.model.dfc, outfall) for outfall in outfalls}

    def overflowing_pipes(self) -> pd.DataFrame:
        """
        Returns all conduits that exceeded the allowed filling (ValMaxFill == 0).
        """
        return pd.DataFrame(self.model.dfc[self.model.dfc["ValMaxFill"] == 0])

    def overflowing_traces(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Identifies segments with overflow in the model and returns their traces.
        """
        all_tr = self.all_traces()
        overflow_df = self.overflowing_pipes()

        results = {}
        for outfall_id, trace_data in all_tr.items():
            overlap = [c for c in trace_data["conduits"] if c in overflow_df.index.tolist()]
            if overlap:
                indexes = {c: trace_data["conduits"].index(c) for c in overlap}
                results[outfall_id] = indexes

        return {
            key: trace_from_node(
                conduits=self.model.dfc,
                startnode=overflow_df.loc[list(value)[-1]]["InletNode"],
                mode="down",
                stopnode=overflow_df.loc[list(value)[0]]["OutletNode"],
            )
            for key, value in results.items()
        }

    def place_to_change(self) -> List[str]:
        """
        Determines nodes where intervention is needed based on overflow traces.
        """
        over_traces = self.overflowing_traces()
        locations = []
        for outfall, data in over_traces.items():
            if "nodes" in data:
                locations.append(data["nodes"][0])
        return locations
