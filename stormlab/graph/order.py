"""The one canonical conduit order shared by the adjacency and the features.

Every graph step runs as ``dfc.loc[order] -> A -> X -> prediction -> inverse
permutation``. Deriving the adjacency from one order and the feature matrix
from another permutes the graph silently, with no error and no obviously wrong
number: with names ``K1..K10`` the INP order is ``K1, K2, ..., K10`` while
``sorted`` gives ``K1, K10, K2, ...``. Both schemes are legitimate as long as a
single ``order`` object drives the whole chain, which is what the functions here
exist to enforce — and every output stays indexed by conduit name, never by
position.
"""

from __future__ import annotations

from typing import Literal, Sequence

import pandas as pd

from stormlab.config import CONDUIT_ORDERS as ORDER_SCHEMES

OrderScheme = Literal["inp", "sorted"]


class OrderError(ValueError):
    """Raised when a frame and an order do not describe the same conduits."""


def conduit_names(dfc: pd.DataFrame) -> list[str]:
    """Conduit names in the frame's own row order.

    ``Name`` is used when present, otherwise the index — the same fallback the
    adjacency builder applies.
    """
    source = dfc["Name"] if "Name" in dfc.columns else dfc.index.to_series()
    names = [str(name) for name in source]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise OrderError(f"Conduit names are not unique: {duplicates}")
    return names


def canonical_order(dfc: pd.DataFrame, scheme: OrderScheme = "inp") -> list[str]:
    """Conduit names in the order the graph and the features will both use.

    Args:
        dfc: conduit frame.
        scheme: ``"inp"`` keeps the file's row order, ``"sorted"`` sorts the
            names lexicographically (the order the historical training code
            used, via ``dfc.sort_values("Name")``).

    Raises:
        OrderError: on duplicate conduit names.
        ValueError: on an unknown scheme.
    """
    if scheme not in ORDER_SCHEMES:
        raise ValueError(f"Unknown conduit order scheme {scheme!r}; expected one of {list(ORDER_SCHEMES)}")
    names = conduit_names(dfc)
    return names if scheme == "inp" else sorted(names)


def permute_frames(dfc: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Reindex a conduit frame onto ``order``, keeping the name as the index.

    Applies to any name-addressable conduit frame — the raw ``dfc``, the feature
    matrix ``X``, a topology table — so that all of them enter the graph step in
    exactly the same row order.

    Raises:
        OrderError: if ``order`` and the frame do not cover the same conduits.
    """
    names = conduit_names(dfc)
    indexed = dfc.copy()
    indexed.index = pd.Index(names, name="Name")
    _check_same_conduits(set(names), order)
    return indexed.loc[list(order)]


def invert_permutation(result: pd.DataFrame | pd.Series, order: Sequence[str]) -> pd.DataFrame | pd.Series:
    """Put a name-indexed result back into ``order``.

    ``order`` is the order the caller wants to read the result in — normally the
    original INP order — which makes the round trip through any graph ordering
    scheme invisible to downstream code.

    Raises:
        OrderError: if ``result`` and ``order`` do not cover the same conduits.
    """
    labels = [str(label) for label in result.index]
    _check_same_conduits(set(labels), order)
    restored = result.copy()
    restored.index = pd.Index(labels, name=result.index.name or "Name")
    return restored.loc[list(order)]


def _check_same_conduits(available: set[str], order: Sequence[str]) -> None:
    requested = [str(name) for name in order]
    if len(set(requested)) != len(requested):
        duplicates = sorted({name for name in requested if requested.count(name) > 1})
        raise OrderError(f"Order contains duplicate conduit names: {duplicates}")
    missing = sorted(set(requested) - available)
    extra = sorted(available - set(requested))
    if missing or extra:
        raise OrderError(f"Order does not match the conduits present: missing={missing}, unexpected={extra}")
