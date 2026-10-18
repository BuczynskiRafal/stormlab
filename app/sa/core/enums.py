from enum import Enum


class RecommendationCategory(Enum):
    PUMP = "pump"
    TANK = "tank"
    SEEPAGE_BOXES = "seepage_boxes"
    DIAMETER_INCREASE = "diameter_increase"
    DIAMETER_REDUCTION = "diameter_reduction"
    SLOPE_INCREASE = "slope_increase"
    SLOPE_REDUCTION = "slope_reduction"
    DEPTH_INCREASE = "depth_increase"
    VALID = "valid"
