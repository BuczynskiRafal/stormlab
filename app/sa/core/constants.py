"""
Hydraulic constants for stormwater analysis.

This module centralizes all magic numbers used in hydraulic calculations
to improve maintainability and make the codebase more readable.
"""

# Manning's roughness coefficient for concrete/PVC pipes
MANNING_COEFFICIENT = 0.013

# Pipe roughness range for normalization (Manning's n values)
# Based on typical pipe materials:
# - Smoothest (PVC/Plastic): 0.009
# - Roughest (Stone channels): 0.020
MIN_ROUGHNESS = 0.009
MAX_ROUGHNESS = 0.020

# Maximum filling ratio for circular pipes (Colebrook-White formula)
# A circular pipe reaches maximum flow capacity at ~82.7% of diameter
MAX_FILLING_RATIO = 0.827

# Filling height calculation parameters
FILLING_CALCULATION_STEP = 0.001  # Step size for iterative filling calculation [m]
FILLING_CALCULATION_MAX_ITER = 10000  # Maximum iterations for filling calculation

# Slope adjustment multipliers for diameter optimization
SLOPE_MULTIPLIERS = [1.0, 2.0, 3.0, 5.0]

# Filling ratio thresholds for diameter optimization
LOW_FILLING_RATIO_THRESHOLD = 0.3  # Below this, consider reducing diameter
TARGET_FILLING_RATIO = 0.5  # Target for hydraulic efficiency

# Flow thresholds
VERY_SMALL_FLOW_THRESHOLD = 0.01  # Below 10 l/s considered very small [m³/s]
MIN_PRACTICAL_DIAMETER = 0.3  # Minimum practical diameter [m] (300mm)

# Flow calculation constant for circular pipes
# Q_max = k * D^2.5 * S^0.5 (with Manning's n=0.013)
CIRCULAR_PIPE_FLOW_CONSTANT = 0.312

# Network propagation
SUBCATCHMENT_PROPAGATION_ITERATIONS = 5

# Subcatchment categories in alphabetical order (matches pd.get_dummies class order
# used during catchment_classifier/model.keras training; see stormlab.contracts.catchment_features.CLASSIFIER_CLASS_ORDER)
SUBCATCHMENT_CATEGORIES = [
    "arable",
    "forests",
    "marshes",
    "meadows",
    "mountains_rocky",
    "mountains_vegetated",
    "rural",
    "suburban_highly_impervious",
    "suburban_weakly_impervious",
    "urban_highly_impervious",
    "urban_moderately_impervious",
    "urban_weakly_impervious",
]

# GNN Model Configuration (optimal hyperparameters from tuning)
GNN_HIDDEN_UNITS = 64
GNN_DROPOUT_RATE = 0.2
GNN_AGGREGATOR = "mean"

# Feature columns for ML models
FEATURE_COLUMNS = [
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
