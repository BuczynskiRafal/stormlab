"""Mapping subcatchments onto the nodes they drain into.

Ported from ``NodeFeatureEngineeringService`` in ``app/sa/core/nodes.py``, which
was already pure pandas. The only change is that the function returns a new node
frame instead of mutating the caller's.
"""

from __future__ import annotations

import pandas as pd

NO_SUBCATCHMENT = "-"
"""Sentinel written into ``Subcatchment``/``SbcCategory`` for nodes with no catchment.

The downstream propagation step compares against this exact string, so it is part
of the contract rather than a display value.
"""


def assign_subcatchment_info(dfn: pd.DataFrame, dfs: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``dfn`` carrying the subcatchment that outfalls into each node.

    Args:
        dfn: node frame indexed by node name.
        dfs: subcatchment frame indexed by subcatchment name; its ``Outlet``
            column names the receiving node, and its optional ``category``
            column carries the land cover class.

    Returns:
        ``dfn`` plus ``Subcatchment`` and ``SbcCategory``, both defaulting to
        :data:`NO_SUBCATCHMENT`. A node that receives a subcatchment always takes
        that subcatchment's ``category`` verbatim, ``NaN`` included: the one-hot
        encoder emits an all-zero block for ``NaN`` exactly as it does for the
        sentinel, so the two are equivalent downstream, and writing the sentinel
        instead would diverge from ``app/sa/core/nodes.py``. When several
        subcatchments share an outlet the last one in frame order wins, as it did
        historically.
    """
    nodes = dfn.copy()
    nodes["Subcatchment"] = NO_SUBCATCHMENT
    nodes["SbcCategory"] = NO_SUBCATCHMENT

    if "Outlet" not in dfs.columns:
        return nodes

    name_mapping = dfs.reset_index().set_index("Outlet")["Name"].to_dict()

    category_mapping: dict[object, object] = {}
    if "category" in dfs.columns:
        category_mapping = dfs.reset_index().set_index("Outlet")["category"].to_dict()

    for node_id in nodes.index:
        if node_id in name_mapping:
            nodes.at[node_id, "Subcatchment"] = name_mapping[node_id]
            if node_id in category_mapping:
                nodes.at[node_id, "SbcCategory"] = category_mapping[node_id]

    return nodes


__all__ = ["NO_SUBCATCHMENT", "assign_subcatchment_info"]
