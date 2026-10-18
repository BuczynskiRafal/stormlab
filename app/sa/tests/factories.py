"""Minimal object builders for the ``sa`` view tests.

Plain functions instead of ``factory_boy``: the models have many required numeric
fields but no interesting invariants, so a dictionary of defaults that the caller
can override keyword by keyword is both shorter and easier to read than a factory
class hierarchy.
"""

from __future__ import annotations

from typing import Any

from accounts.models import User
from sa.models import CalculationSession, ConduitData, NodeData, SessionGeometry, SubcatchmentData, SWMMModel

# The seven booleans rendered by the "Validation Results" card of sa/conduit_panel.html.
VALIDATION_FLAGS = (
    "val_max_fill",
    "val_max_v",
    "val_min_v",
    "val_max_slope",
    "val_min_slope",
    "val_depth",
    "val_coverage",
)


def make_user(email: str = "smoke@example.com", password: str = "smoke-pass-123", **extra: Any) -> User:
    """Create an active user; ``User`` has no ``username`` field, e-mail is the login."""
    return User.objects.create_user(email=email, password=password, **extra)


def make_swmm_model(user: User, **extra: Any) -> SWMMModel:
    """Create a SWMM model row. ``file`` is only a name here - no file is written."""
    fields: dict[str, Any] = {"file": "user_models/smoke.inp", "zone": 1}
    fields.update(extra)
    return SWMMModel.objects.create(user=user, **fields)


def make_session(
    user: User,
    *,
    status: str = "completed",
    n_conduits: int = 0,
    n_nodes: int = 0,
    n_subcatchments: int = 0,
    swmm_model: SWMMModel | None = None,
    **extra: Any,
) -> CalculationSession:
    """Create a session and, optionally, its conduits/nodes/subcatchments.

    Conduits are wired to the nodes and subcatchments created alongside them
    (``C<i>`` runs from ``N<i>`` to ``N<i+1>`` and drains ``S<i>``), so the detail
    views find the related objects they render.
    """
    fields: dict[str, Any] = {"frost_zone": 0.8, "status": status}
    fields.update(extra)
    session = CalculationSession.objects.create(
        user=user,
        swmm_model=swmm_model or make_swmm_model(user),
        **fields,
    )

    for index in range(1, n_nodes + 1):
        make_node(session, node_name=f"N{index}")
    for index in range(1, n_subcatchments + 1):
        make_subcatchment(session, subcatchment_name=f"S{index}")
    for index in range(1, n_conduits + 1):
        make_conduit(
            session,
            conduit_name=f"C{index}",
            inlet_node=f"N{index}",
            outlet_node=f"N{index + 1}",
            subcatchment=f"S{index}" if index <= n_subcatchments else "-",
        )
    return session


# A dry conduit as the pipeline stores it (K18/K30 in every session): no flow, so the
# minimum slope derived from the filling lies above the maximum.
DRY_CONDUIT: dict[str, Any] = {
    "geom1": 0.3,
    "filling": 0.0,
    "max_q": 0.0,
    "max_v": 0.0,
    "slope_per_mile": 1.41,
    "min_required_slope": 229.72,
    "max_allowable_slope": 134.56,
    "val_min_v": 0,
    "val_min_slope": 0,
    "val_max_slope": 1,
    "increase_slope": 1,
}


def make_conduit(session: CalculationSession, **overrides: Any) -> ConduitData:
    """Create one conduit; pass any of ``VALIDATION_FLAGS`` as 0/1 to drive the card."""
    fields: dict[str, Any] = {
        "conduit_name": "C1",
        "geom1": 0.4,
        "max_v": 1.2,
        "max_q": 0.15,
        "filling": 0.2,
        "slope_per_mile": 5.0,
        "length": 50.0,
        "min_diameter": 0.3,
        "is_min_diameter": 0,
        "increase_dia": 0,
        "reduce_dia": 0,
        "min_required_slope": 2.0,
        "increase_slope": 0,
        "max_allowable_slope": 40.0,
        "reduce_slope": 0,
        "inlet_node": "N1",
        "outlet_node": "N2",
        "inlet_max_depth": 3.0,
        "outlet_max_depth": 3.0,
        "inlet_ground_elevation": 120.0,
        "outlet_ground_elevation": 119.5,
        "inlet_ground_cover": 1.5,
        "outlet_ground_cover": 1.5,
        "subcatchment": "S1",
        "sbc_category": "residential",
        "recommendation": "valid",
        "confidence_valid": 0.9,
        "confidence_pump": 0.01,
        "confidence_tank": 0.01,
        "confidence_seepage_boxes": 0.01,
        "confidence_diameter_increase": 0.02,
        "confidence_diameter_reduction": 0.02,
        "confidence_slope_increase": 0.02,
        "confidence_slope_reduction": 0.01,
        "confidence_depth_increase": 0.0,
    }
    fields.update({flag: 1 for flag in VALIDATION_FLAGS})
    fields.update(overrides)
    return ConduitData.objects.create(session=session, **fields)


def make_node(session: CalculationSession, **overrides: Any) -> NodeData:
    fields: dict[str, Any] = {
        "node_name": "N1",
        "max_depth": 3.0,
        "invert_elevation": 117.0,
        "subcatchment": "S1",
        "sbc_category": "residential",
    }
    fields.update(overrides)
    return NodeData.objects.create(session=session, **fields)


def make_subcatchment(session: CalculationSession, **overrides: Any) -> SubcatchmentData:
    fields: dict[str, Any] = {
        "subcatchment_name": "S1",
        "area": 1.5,
        "perc_imperv": 45.0,
        "perc_slope": 1.2,
        "outlet": "N1",
        "total_runoff_mg": 0.4,
        "peak_runoff": 0.2,
        "runoff_coeff": 0.55,
        "category": "residential",
    }
    fields.update(overrides)
    return SubcatchmentData.objects.create(session=session, **fields)


def make_geometry_payload(**overrides: Any) -> dict[str, Any]:
    """A minimal but complete network-map payload, shaped like ``extract_geometry``.

    Two junctions, one outfall and the two conduits between them, plus one catchment
    outline — enough for every consumer to find each of the three drawing groups.
    """
    payload: dict[str, Any] = {
        "version": 1,
        "origin": [0.0, 0.0],
        "bbox": {"w": 20.0, "h": 10.0},
        "min_gap": 10.0,
        "layout": "inp",
        "nodes": [
            {"n": "NM1", "i": 0, "p": [0.0, 0.0], "k": "junction"},
            {"n": "NM2", "i": 1, "p": [10.0, 5.0], "k": "junction"},
            {"n": "NM3", "i": 2, "p": [20.0, 10.0], "k": "outfall"},
        ],
        "links": [
            {"n": "LM1", "i": 0, "p": [[0.0, 0.0], [10.0, 5.0]], "a": "NM1", "b": "NM2", "kind": "conduit"},
            {"n": "LM2", "i": 1, "p": [[10.0, 5.0], [20.0, 10.0]], "a": "NM2", "b": "NM3", "kind": "conduit"},
        ],
        "polys": [{"n": "SM1", "i": 0, "p": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]}],
        "skipped": {"links_dangling": 0, "polys_degenerate": 0},
        "counts": {"elements": 8},
    }
    payload.update(overrides)
    return payload


def make_geometry(session: CalculationSession, **overrides: Any) -> SessionGeometry:
    """Attach a stored drawing geometry to *session*."""
    return SessionGeometry.objects.create(session=session, data=make_geometry_payload(**overrides))
