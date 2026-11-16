"""The 33 feature columns of the legacy model path.

Read only when a GNN weight file is loaded without a specification (``GraphSAGEModel``
in ``graph_constructor.py``). The shipped model has a specification and reads the 66
columns of ``v4_physical`` (``stormlab/contracts/features.py``) instead.
"""


def get_default_feature_columns():
    """Get default feature columns for SWMM conduits."""
    return [
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
