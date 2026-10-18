"""
Services for data persistence and calculation management.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import pandas as pd
from django.db import transaction
from django.db.models import Q
from django.urls import NoReverseMatch, reverse
from django.utils import timezone

from accounts.models import User
from .models import CalculationSession, ConduitData, NodeData, SessionGeometry, SubcatchmentData, SWMMModel
from .core.data import DataManager
from .core.geometry import NETMAP_MAX_ELEMENTS, NETMAP_MAX_POINTS, extract_geometry
from .core.predictor import get_model_metadata
from .core.round import max_depth_value, max_filling, max_velocity_value, min_velocity_value

logger = logging.getLogger(__name__)


def _positive(value: Any) -> bool:
    """Whether a stored SWMM number is a real number greater than zero."""
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def node_is_flooded(hours_flooded: Any, flood_volume: Any) -> bool:
    """The one verdict on node flooding, shared by every view of it.

    SWMM writes Hours Flooded with two decimals, so a flood shorter than 0.6 min is
    reported as 0.00 h while its volume stays positive: either signal means the node
    flooded.
    """
    return _positive(hours_flooded) or _positive(flood_volume)


def fill_percent(filling: Any, diameter: Any) -> float | None:
    """Filling as a percentage of the diameter, clamped to a full pipe.

    ``None`` when either number is missing or the diameter is not positive.  A filling
    reported above the diameter is a full pipe, not 133 % of one, so the value never
    leaves [0, 100] — the table label and the drawn water level have to agree.
    """
    if filling is None or diameter is None:
        return None
    try:
        filling_val = float(filling)
        diameter_val = float(diameter)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(diameter_val) or diameter_val <= 0.0 or not math.isfinite(filling_val):
        return None
    return min(100.0, max(0.0, 100.0 * filling_val / diameter_val))


def simple_band_rule_passed(filling: Any, diameter: Any, velocity: Any) -> bool:
    """Simple operational band rule (§3.4.3): 0.2 <= fill/D <= 0.7 and 0.8 <= v <= 3.0 m/s."""
    try:
        fill_f = float(filling)
        dia_f = float(diameter)
        vel_f = float(velocity)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(fill_f) and math.isfinite(dia_f) and math.isfinite(vel_f)) or dia_f <= 0.0:
        return False
    ratio = fill_f / dia_f
    return (0.2 <= ratio <= 0.7) and (0.8 <= vel_f <= 3.0)



class AnalysisService:
    """Service responsible for running SWMM analysis."""

    @staticmethod
    def analyze_file(
        user: User,
        file_path: str,
        frost_zone: float,
        swmm_model: SWMMModel,
    ) -> tuple[CalculationSession, bool]:
        """
        Run SWMM analysis on a file and save results.

        Args:
            user: User who owns the analysis
            file_path: Path to the SWMM .inp file
            frost_zone: Frost zone value (0.8-1.6)
            swmm_model: SWMMModel instance

        Returns:
            Tuple of (session, success) where success indicates if analysis completed
        """
        session = CalculationPersistenceService.create_session(user=user, swmm_model=swmm_model, frost_zone=frost_zone)

        try:
            with DataManager(file_path, zone=frost_zone) as model:
                success = CalculationPersistenceService.save_calculation_results(session, model)
                return session, success
        except Exception as e:
            logger.error(f"Analysis error for session {session.id}: {e}")
            session.status = "failed"
            session.error_message = str(e)
            session.save()
            return session, False


class CalculationPersistenceService:
    """Service for saving calculation results to the database."""

    @staticmethod
    def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
        """
        Safely convert value to a finite float, returning *default* for None, NaN, Inf, or invalid values.
        """
        if value is None or isinstance(value, bool):
            return default
        try:
            result = float(value)
            return result if math.isfinite(result) else default
        except (ValueError, TypeError, OverflowError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """
        Safely convert value to int, handling None, NaN, booleans, and invalid values.
        """
        if isinstance(value, bool):
            return int(value)
        fval = CalculationPersistenceService._safe_float(value, None)
        if fval is None:
            return default
        try:
            return int(fval)
        except (ValueError, TypeError, OverflowError):
            return default

    @staticmethod
    def _safe_str(value: Any, default: str = "-") -> str:
        """Safely convert a scalar cell value to string, mapping None/NaN/empty to *default*."""
        if value is None:
            return default
        if isinstance(value, float) and not math.isfinite(value):
            return default
        try:
            if bool(pd.isna(value)):
                return default
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        return text if text and text.lower() != "nan" else default

    @staticmethod
    def create_session(user, swmm_model: SWMMModel, frost_zone: float) -> CalculationSession:
        """Create a new calculation session."""
        return CalculationSession.objects.create(user=user, swmm_model=swmm_model, frost_zone=frost_zone, status="processing")

    @staticmethod
    def save_calculation_results(session: CalculationSession, data_manager: DataManager) -> bool:
        """
        Save calculation results from DataManager to database.

        Args:
            session: CalculationSession instance
            data_manager: DataManager instance with calculated data

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with transaction.atomic():
                # Save conduit data
                CalculationPersistenceService._save_conduit_data(session, data_manager.dfc)

                # Save node data
                CalculationPersistenceService._save_node_data(session, data_manager.dfn)

                # Save subcatchment data
                CalculationPersistenceService._save_subcatchment_data(session, data_manager.dfs)

                # Capture the drawable geometry; its absence is a normal state, not a failure
                CalculationPersistenceService._save_geometry(session, data_manager.inp)

                # Update session status
                session.status = "completed"
                session.updated_at = timezone.now()
                session.save()

                logger.info(f"Successfully saved calculation results for session {session.id}")
                return True

        except Exception as e:
            logger.error(f"Error saving calculation results for session {session.id}: {str(e)}")
            session.status = "failed"
            session.error_message = str(e)
            session.save()
            return False

    @staticmethod
    def _save_conduit_data(session: CalculationSession, dfc: pd.DataFrame) -> None:
        """Save conduit data to database."""
        conduit_objects = []
        sf = CalculationPersistenceService._safe_float
        si = CalculationPersistenceService._safe_int
        ss = CalculationPersistenceService._safe_str

        for idx, row in dfc.iterrows():
            try:
                conduit_obj = ConduitData(
                    session=session,
                    conduit_name=str(idx),
                    geom1=sf(row.get("Geom1"), 0.0),
                    max_v=sf(row.get("MaxV"), 0.0),
                    max_q=sf(row.get("MaxQ"), 0.0),
                    filling=sf(row.get("Filling"), 0.0),
                    slope_per_mile=sf(row.get("SlopePerMile"), 0.0),
                    length=sf(row.get("Length"), 0.0),
                    val_max_fill=si(row.get("ValMaxFill"), 0),
                    val_max_v=si(row.get("ValMaxV"), 0),
                    val_min_v=si(row.get("ValMinV"), 0),
                    val_max_slope=si(row.get("ValMaxSlope"), 0),
                    val_min_slope=si(row.get("ValMinSlope"), 0),
                    val_depth=si(row.get("ValDepth"), 0),
                    val_coverage=si(row.get("ValCoverage"), 0),
                    min_diameter=sf(row.get("MinDiameter"), 0.0),
                    is_min_diameter=si(row.get("isMinDiameter"), 0),
                    increase_dia=si(row.get("IncreaseDia"), 0),
                    reduce_dia=si(row.get("ReduceDia"), 0),
                    min_required_slope=sf(row.get("MinRequiredSlope"), 0.0),
                    increase_slope=si(row.get("IncreaseSlope"), 0),
                    max_allowable_slope=sf(row.get("MaxAllowableSlope"), 0.0),
                    reduce_slope=si(row.get("ReduceSlope"), 0),
                    inlet_node=ss(row.get("InletNode"), ""),
                    outlet_node=ss(row.get("OutletNode"), ""),
                    inlet_max_depth=sf(row.get("InletMaxDepth"), 0.0),
                    outlet_max_depth=sf(row.get("OutletMaxDepth"), 0.0),
                    inlet_ground_elevation=sf(row.get("InletGroundElevation"), 0.0),
                    outlet_ground_elevation=sf(row.get("OutletGroundElevation"), 0.0),
                    inlet_ground_cover=sf(row.get("InletGroundCover"), 0.0),
                    outlet_ground_cover=sf(row.get("OutletGroundCover"), 0.0),
                    subcatchment=ss(row.get("Subcatchment"), "-"),
                    sbc_category=ss(row.get("SbcCategory"), "-"),
                    recommendation=ss(row.get("recommendation"), ""),
                    # Confidence scores - handle NaN and None values
                    confidence_pump=sf(row.get("confidence_pump"), 0.0),
                    confidence_tank=sf(row.get("confidence_tank"), 0.0),
                    confidence_seepage_boxes=sf(row.get("confidence_seepage_boxes"), 0.0),
                    confidence_diameter_increase=sf(row.get("confidence_diameter_increase"), 0.0),
                    confidence_diameter_reduction=sf(row.get("confidence_diameter_reduction"), 0.0),
                    confidence_slope_increase=sf(row.get("confidence_slope_increase"), 0.0),
                    confidence_slope_reduction=sf(row.get("confidence_slope_reduction"), 0.0),
                    confidence_depth_increase=sf(row.get("confidence_depth_increase"), 0.0),
                    confidence_valid=sf(row.get("confidence_valid"), 0.0),
                )
                conduit_objects.append(conduit_obj)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing conduit {idx}: {str(e)}")
                continue

        if conduit_objects:
            ConduitData.objects.bulk_create(conduit_objects, batch_size=1000)
            logger.info(f"Saved {len(conduit_objects)} conduit records")

    @staticmethod
    def _save_node_data(session: CalculationSession, dfn: pd.DataFrame) -> None:
        """Save node data to database."""
        node_objects = []
        sf = CalculationPersistenceService._safe_float
        ss = CalculationPersistenceService._safe_str

        for idx, row in dfn.iterrows():
            try:
                node_obj = NodeData(
                    session=session,
                    node_name=str(idx),
                    max_depth=sf(row.get("MaxDepth"), 0.0),
                    invert_elevation=sf(row.get("InvertElev"), None),
                    max_water_depth=sf(row.get("MaxNodeDepth"), None),
                    hours_flooded=sf(row.get("HoursFlooded"), None),
                    flood_volume=sf(row.get("TotalFloodVol"), None),
                    subcatchment=ss(row.get("Subcatchment"), "-"),
                    sbc_category=ss(row.get("SbcCategory"), "-"),
                )
                node_objects.append(node_obj)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing node {idx}: {str(e)}")
                continue

        if node_objects:
            NodeData.objects.bulk_create(node_objects, batch_size=1000)
            logger.info(f"Saved {len(node_objects)} node records")

    @staticmethod
    def _save_subcatchment_data(session: CalculationSession, dfs: pd.DataFrame) -> None:
        """Save subcatchment data to database."""
        subcatchment_objects = []
        sf = CalculationPersistenceService._safe_float
        ss = CalculationPersistenceService._safe_str

        for idx, row in dfs.iterrows():
            try:
                subcatchment_obj = SubcatchmentData(
                    session=session,
                    subcatchment_name=str(idx),
                    area=sf(row.get("Area"), 0.0),
                    perc_imperv=sf(row.get("PercImperv"), 0.0),
                    perc_slope=sf(row.get("PercSlope"), 0.0),
                    outlet=ss(row.get("Outlet"), ""),
                    total_runoff_mg=sf(row.get("TotalRunoffMG"), None),
                    peak_runoff=sf(row.get("PeakRunoff"), None),
                    runoff_coeff=sf(row.get("RunoffCoeff"), None),
                    category=ss(row.get("category"), "-"),
                )
                subcatchment_objects.append(subcatchment_obj)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing subcatchment {idx}: {str(e)}")
                continue

        if subcatchment_objects:
            SubcatchmentData.objects.bulk_create(subcatchment_objects, batch_size=1000)
            logger.info(f"Saved {len(subcatchment_objects)} subcatchment records")

    @staticmethod
    def _save_geometry(session: CalculationSession, inp: Any) -> None:
        """Store the network drawing geometry of *inp* alongside the results.

        Geometry is a convenience for the map, never a part of the analysis, so no
        failure here may cost the caller its results: the payload is computed before
        the database is touched, and the write lives in its own savepoint.  Without
        that nested ``atomic``, a database error caught here would leave the outer
        transaction marked for rollback and the following ``session.save()`` would
        raise ``TransactionManagementError``.
        """
        try:
            payload = extract_geometry(inp)
        except Exception as e:
            logger.error(f"Error extracting geometry for session {session.id}: {str(e)}")
            return

        try:
            with transaction.atomic():
                SessionGeometry.objects.update_or_create(session=session, defaults={"data": payload})
        except Exception as e:
            logger.error(f"Error saving geometry for session {session.id}: {str(e)}")
            return

        logger.info(f"Saved network geometry for session {session.id}")


class CalculationRetrievalService:
    """Service for retrieving calculation results from the database."""

    @staticmethod
    def get_session_data(session: CalculationSession) -> Dict[str, Any]:
        """
        Retrieve all data for a calculation session.

        Args:
            session: CalculationSession instance

        Returns:
            Dict containing conduits, nodes, and subcatchments data
        """
        return {
            "session": session,
            "conduits": list(session.conduits.all().values()),
            "nodes": list(session.nodes.all().values()),
            "subcatchments": list(session.subcatchments.all().values()),
        }

    @staticmethod
    def get_user_sessions(user, limit: Optional[int] = None, status_filter: Optional[str] = None) -> list:
        """
        Get calculation sessions for a user.

        Args:
            user: User instance
            limit: Maximum number of sessions to return
            status_filter: Filter by session status

        Returns:
            List of CalculationSession instances
        """
        queryset = (
            CalculationSession.objects.filter(user=user)
            .select_related("swmm_model")
            .prefetch_related("conduits", "nodes", "subcatchments")
            .order_by("-created_at")
        )

        if status_filter:
            queryset = queryset.filter(status=status_filter)

        if limit:
            queryset = queryset[:limit]

        return list(queryset)

    @staticmethod
    def format_session_data_for_template(session: CalculationSession) -> Dict[str, list]:
        """
        Format session data for template display.

        Args:
            session: CalculationSession instance

        Returns:
            Dict formatted for template rendering
        """
        conduits_data = []
        for conduit in session.conduits.all():
            conduits_data.append(
                {
                    "Name": conduit.conduit_name,
                    "Diameter": conduit.geom1,
                    "MaxV": conduit.max_v,
                    "MaxQ": conduit.max_q,
                    "Filling": conduit.filling,
                    "Recommendation": conduit.recommendation,
                    "InletNode": conduit.inlet_node,
                    "OutletNode": conduit.outlet_node,
                    "Subcatchment (model)": conduit.subcatchment,
                    "Category": conduit.sbc_category,
                }
            )

        nodes_data = []
        for node in session.nodes.all():
            nodes_data.append(
                {
                    "Name": node.node_name,
                    "MaxDepth": node.max_depth,
                    "InvertElev": node.invert_elevation,
                    "MaxNodeDepth": node.max_water_depth,
                    "HoursFlooded": node.hours_flooded,
                    "TotalFloodVol": node.flood_volume,
                    "Subcatchment": node.subcatchment,
                    "Category": node.sbc_category,
                }
            )

        subcatchments_data = []
        for subcatchment in session.subcatchments.all():
            subcatchments_data.append(
                {
                    "Name": subcatchment.subcatchment_name,
                    "Area": subcatchment.area,
                    "PercImperv": subcatchment.perc_imperv,
                    "PercSlope": subcatchment.perc_slope,
                    "Outlet": subcatchment.outlet,
                    "Category": subcatchment.category,
                }
            )

        return {
            "Conduits Data": conduits_data,
            "Nodes Data": nodes_data,
            "Subcatchments Data": subcatchments_data,
        }

    @staticmethod
    def panel_links(
        session: CalculationSession,
        conduits_qs: list[ConduitData] | None = None,
        subcatchments_qs: list[SubcatchmentData] | None = None,
    ) -> dict[str, dict[str, str]]:
        """Map node and subcatchment names to conduit-panel hrefs.

        Returns two nested dicts keyed by the dataset name used in
        ``format_session_data_for_template`` (``"Nodes Data"`` / ``"Subcatchments
        Data"``).  Each inner dict maps an element name to a ready-made ``href``
        pointing to the conduit panel that shows that element, with the matching
        fragment anchor.

        Two queries at most: one for conduits, one for subcatchments (or zero if passed).
        """
        # Sorted in Python, not with ``order_by``: the results view hands us a prefetched
        # queryset, and re-ordering it in SQL would discard that cache and cost a query.
        raw_conduits = conduits_qs if conduits_qs is not None else session.conduits.all()
        conduits = [(c.conduit_name, c.inlet_node, c.outlet_node) for c in sorted(raw_conduits, key=lambda c: c.pk or 0)]

        # Build inlet-first and outlet-fallback indexes (first by id wins).
        inlet_index: dict[str, str] = {}  # node_name -> conduit_name
        outlet_index: dict[str, str] = {}  # node_name -> conduit_name
        for cname, inlet, outlet in conduits:
            inlet_index.setdefault(inlet, cname)
            outlet_index.setdefault(outlet, cname)

        # --- Nodes Data ---
        all_node_names: set[str] = set()
        for _, inlet, outlet in conduits:
            all_node_names.add(inlet)
            all_node_names.add(outlet)

        nodes_links: dict[str, str] = {}

        def panel_href(conduit_name: str, fragment: str) -> str | None:
            # A legal SWMM name may contain "/", which no URL pattern of ours accepts.
            try:
                return reverse("sa:conduit_detail", args=[session.id, conduit_name]) + fragment
            except NoReverseMatch:
                return None

        for node_name in all_node_names:
            href = None
            if node_name in inlet_index:
                href = panel_href(inlet_index[node_name], "#node-inlet")
            elif node_name in outlet_index:
                href = panel_href(outlet_index[node_name], "#node-outlet")
            if href:
                nodes_links[node_name] = href

        # --- Subcatchments Data ---
        raw_subcatchments = subcatchments_qs if subcatchments_qs is not None else session.subcatchments.all()
        subcatchments = [(s.subcatchment_name, s.outlet, s.pk) for s in sorted(raw_subcatchments, key=lambda s: s.pk or 0)]

        sbc_links: dict[str, str] = {}
        for sbc_name, sbc_outlet, sbc_pk in subcatchments:
            href = None
            if sbc_outlet in inlet_index:
                href = panel_href(inlet_index[sbc_outlet], f"#catchment-{sbc_pk}")
            elif sbc_outlet in outlet_index:
                href = panel_href(outlet_index[sbc_outlet], "#node-outlet")
            if href:
                sbc_links[sbc_name] = href

        return {
            "Nodes Data": nodes_links,
            "Subcatchments Data": sbc_links,
        }

    @staticmethod
    def _clean_float(val: Any) -> float | None:
        """Sanitize a float value, converting NaN, Inf, and invalid numbers to None for RFC 8259 JSON compliance."""
        return CalculationPersistenceService._safe_float(val, None)

    @staticmethod
    def _clean_flag(val: Any) -> int:
        """Defensively sanitize a validation flag to an integer (0 or 1), handling None, NaN, Inf, and invalid types."""
        if val is None:
            return 0
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return 0
            return 1 if int(f) == 1 else 0
        except (ValueError, TypeError, OverflowError):
            return 0

    @staticmethod
    def serialize_geometry_for_map(session: CalculationSession) -> dict[str, Any]:
        """Wrap the stored drawing geometry of *session* in the envelope the page embeds.

        The shape is always the same three keys, so the template and the browser never
        branch on a missing field::

            {"available": bool, "reason": None | "no_geometry" | "too_large",
             "data": <payload of core.geometry.extract_geometry> | None}

        A session analysed before the geometry was captured — or one whose capture
        failed — simply has no related row: that is a normal state, not an error, and
        it yields ``"no_geometry"``.  A network above :data:`NETMAP_MAX_ELEMENTS`
        drawable elements *or* above :data:`NETMAP_MAX_POINTS` coordinate pairs yields
        ``"too_large"`` and the payload **stays in the database**: sending megabytes
        the browser would only choke on helps nobody.  Both bounds are needed and
        neither implies the other: a thousand subcatchments with four-hundred-point
        outlines are a thousand elements and four hundred thousand points.

        ``counts.points`` is read with a default of zero, because a row written before
        the counter existed does not carry it, and an old row is not a large one: the
        element bound still applies to it and the payload is still served.

        Nothing here branches on the payload's ``version`` and nothing should start to:
        a row of version 1 is drawn by the same browser code as one of version 2, which
        merely says more about each polygon.  What version 2 does change is the weight
        on these two thresholds, and deliberately so: an outlet line is one more element
        and one more coordinate pair, and the centroid every polygon now carries is one
        more pair again, so a model of a thousand subcatchments sits some two thousand
        points closer to :data:`NETMAP_MAX_POINTS` than the same model did before.

        The relation is read through ``getattr`` and costs no query of its own, as the
        results view selects ``geometry`` along with the session.
        """
        geometry = getattr(session, "geometry", None)
        if geometry is None:
            return {"available": False, "reason": "no_geometry", "data": None}

        payload = geometry.data
        if not isinstance(payload, dict):
            # A row written by an older or a broken producer is worth no more than
            # no row at all, and the page must not fail over it.
            return {"available": False, "reason": "no_geometry", "data": None}

        stored = payload.get("counts")
        counts: dict[str, Any] = stored if isinstance(stored, dict) else {}

        def _count(key: str) -> float:
            value = counts.get(key, 0)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return 0
            return value

        if _count("elements") > NETMAP_MAX_ELEMENTS or _count("points") > NETMAP_MAX_POINTS:
            return {"available": False, "reason": "too_large", "data": None}

        return {"available": True, "reason": None, "data": payload}

    @classmethod
    def serialize_session_for_results_v2(
        cls,
        session: CalculationSession,
        rule_labels: dict[str, str] | None = None,
        panel_links: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Serialize a session to the JSON-ready payload the Results V2 page embeds.

        Carries the conduit, node and subcatchment fields the page reads, with NaN/Inf
        sanitized to ``None`` (RFC 8259 has no literal for either), plus summary counts,
        validation counts, rule labels and panel navigation URLs.

        The nine per-class ``confidence_*`` scores are deliberately *not* serialized: the
        table shows only the recommended class's score, as ``confidence`` in percent, and
        the full distribution belongs to the conduit panel, which reads it from the model.
        """
        clean = cls._clean_float
        clean_flag = cls._clean_flag

        # Resolve rule labels avoiding circular imports
        if rule_labels is None:
            try:
                from .views import WROCLAW_RULE_LABELS

                rule_labels = WROCLAW_RULE_LABELS
            except ImportError:
                rule_labels = {}

        conduits_qs = list(session.conduits.all())
        nodes_qs = list(session.nodes.all())
        subcatchments_qs = list(session.subcatchments.all())

        # Resolve panel links reusing already-fetched querysets
        if panel_links is None:
            panel_links = cls.panel_links(
                session,
                conduits_qs=conduits_qs,
                subcatchments_qs=subcatchments_qs,
            )

        nodes_links = panel_links.get("Nodes Data", {})
        sbc_links = panel_links.get("Subcatchments Data", {})

        # Build lookup maps for node/catchment -> conduit associations
        inlet_conduit_map: dict[str, str] = {}
        outlet_conduit_map: dict[str, str] = {}
        sbc_conduit_map: dict[str, str] = {}

        for c in conduits_qs:
            inlet_conduit_map.setdefault(c.inlet_node, c.conduit_name)
            outlet_conduit_map.setdefault(c.outlet_node, c.conduit_name)
            if c.subcatchment and c.subcatchment != "-":
                sbc_conduit_map.setdefault(c.subcatchment, c.conduit_name)

        for s in subcatchments_qs:
            if s.subcatchment_name not in sbc_conduit_map:
                if s.outlet in inlet_conduit_map:
                    sbc_conduit_map[s.subcatchment_name] = inlet_conduit_map[s.outlet]
                elif s.outlet in outlet_conduit_map:
                    sbc_conduit_map[s.subcatchment_name] = outlet_conduit_map[s.outlet]

        # The recommendation code -> confidence field mapping the conduit panel already owns.
        rec_to_conf_field = {code: field for code, _label, field in PREDICTION_CLASSES}
        rec_to_label = {code: label for code, label, _field in PREDICTION_CLASSES}

        # -------------------------------------------------------------------------
        # 1. Conduits Serialization
        # -------------------------------------------------------------------------
        serialized_conduits = []
        conduits_changes_count = 0

        for c in conduits_qs:
            rec = c.recommendation or ""
            rec_norm = rec.strip().lower()
            if rec_norm != "valid":
                conduits_changes_count += 1

            flag_max_fill = clean_flag(c.val_max_fill)
            flag_max_v = clean_flag(c.val_max_v)
            flag_min_v = clean_flag(c.val_min_v)
            flag_max_slope = clean_flag(c.val_max_slope)
            flag_min_slope = clean_flag(c.val_min_slope)
            flag_depth = clean_flag(c.val_depth)
            flag_coverage = clean_flag(c.val_coverage)

            val_flags = [
                bool(flag_max_fill),
                bool(flag_max_v),
                bool(flag_min_v),
                bool(flag_max_slope),
                bool(flag_min_slope),
                bool(flag_depth),
                bool(flag_coverage),
            ]
            passed_checks = sum(1 for f in val_flags if f)
            failed_checks = 7 - passed_checks

            geom1_val = clean(c.geom1)
            filling_val = clean(c.filling)
            pct = fill_percent(filling_val, geom1_val)
            fill_pct = 0.0 if pct is None else (clean(round(pct, 2)) or 0.0)

            inlet_depth = clean(c.inlet_max_depth) or 0.0
            outlet_depth = clean(c.outlet_max_depth) or 0.0
            max_depth = max(inlet_depth, outlet_depth)

            rule_lbl = rule_labels.get(c.conduit_name, "-")

            try:
                conduit_url = reverse("sa:conduit_detail", args=[session.id, c.conduit_name])
            except NoReverseMatch:
                conduit_url = None

            conf_field = rec_to_conf_field.get(rec_norm)
            selected_conf = clean(getattr(c, conf_field, None)) if conf_field else None
            confidence_display = clean(round(selected_conf * 100.0, 1)) if selected_conf is not None else None
            confidence_margin = confidence_margin_pct(c)
            rec_label = rec_to_label.get(rec_norm, rec or "—")
            simple_rule = simple_band_rule_passed(filling_val, geom1_val, clean(c.max_v))

            is_min_dia = bool(c.is_min_diameter)
            inc_dia = bool(c.increase_dia)
            red_dia = bool(c.reduce_dia)
            inc_slope = bool(c.increase_slope)
            red_slope = bool(c.reduce_slope)

            serialized_conduits.append(
                {
                    # Identity
                    "id": c.conduit_name,
                    "name": c.conduit_name,
                    "conduit_name": c.conduit_name,
                    "inlet": c.inlet_node,
                    "inlet_node": c.inlet_node,
                    "outlet": c.outlet_node,
                    "outlet_node": c.outlet_node,
                    "catchment": c.subcatchment,
                    "subcatchment": c.subcatchment,
                    "category": c.sbc_category,
                    "sbc_category": c.sbc_category,
                    # Geometry (all 9 fields + aliases)
                    "diameter": geom1_val,
                    "geom1": geom1_val,
                    "length": clean(c.length),
                    "slope": clean(c.slope_per_mile),
                    "slope_per_mile": clean(c.slope_per_mile),
                    "coverIn": clean(c.inlet_ground_cover),
                    "inlet_ground_cover": clean(c.inlet_ground_cover),
                    "coverOut": clean(c.outlet_ground_cover),
                    "outlet_ground_cover": clean(c.outlet_ground_cover),
                    "inlet_max_depth": clean(c.inlet_max_depth),
                    "outlet_max_depth": clean(c.outlet_max_depth),
                    "depth": max_depth,
                    "inlet_ground_elevation": clean(c.inlet_ground_elevation),
                    "outlet_ground_elevation": clean(c.outlet_ground_elevation),
                    # Hydraulics (all 3 fields + aliases)
                    "flow": clean(c.max_q),
                    "max_q": clean(c.max_q),
                    "velocity": clean(c.max_v),
                    "max_v": clean(c.max_v),
                    "filling": filling_val,
                    "fillPercent": fill_pct,
                    "fill_percent": fill_pct,
                    # Diameter Analysis (all 4 fields + aliases)
                    "minDiameter": clean(c.min_diameter),
                    "min_diameter": clean(c.min_diameter),
                    "is_min_diameter": is_min_dia,
                    "isMinDiameter": is_min_dia,
                    "increase_dia": inc_dia,
                    "increaseDia": inc_dia,
                    "reduce_dia": red_dia,
                    "reduceDia": red_dia,
                    # Slope Analysis (all 4 fields + aliases)
                    "minSlope": clean(c.min_required_slope),
                    "min_required_slope": clean(c.min_required_slope),
                    "maxSlope": clean(c.max_allowable_slope),
                    "maxAllowableSlope": clean(c.max_allowable_slope),
                    "max_allowable_slope": clean(c.max_allowable_slope),
                    "increase_slope": inc_slope,
                    "increaseSlope": inc_slope,
                    "reduce_slope": red_slope,
                    "reduceSlope": red_slope,
                    # Validation Flags (all 7 fields + derived)
                    "val_max_fill": flag_max_fill,
                    "val_max_v": flag_max_v,
                    "val_min_v": flag_min_v,
                    "val_max_slope": flag_max_slope,
                    "val_min_slope": flag_min_slope,
                    "val_depth": flag_depth,
                    "val_coverage": flag_coverage,
                    "flags": val_flags,
                    "validation": passed_checks,
                    "passed_checks_count": passed_checks,
                    "failed_checks_count": failed_checks,
                    "simple_rule_passed": simple_rule,
                    # Recommendation & Rules
                    "action": rec,
                    "recommendation": rec,
                    "recommendation_label": rec_label,
                    "rule": rule_lbl,
                    "rule_label": rule_lbl,
                    # URLs
                    "detail_url": conduit_url,
                    "panel_url": conduit_url,
                    "link": conduit_url,
                    # Confidences
                    "confidence": confidence_display,
                    "confidence_margin": confidence_margin,
                }
            )

        # -------------------------------------------------------------------------
        # 2. Nodes Serialization
        # -------------------------------------------------------------------------
        serialized_nodes = []
        nodes_flooding_count = 0

        for n in nodes_qs:
            inv = clean(n.invert_elevation)
            dep = clean(n.max_depth)
            ground_elev = clean(round(inv + dep, 4)) if inv is not None and dep is not None else None

            hours = clean(n.hours_flooded)
            vol = clean(n.flood_volume)
            is_flooded = node_is_flooded(hours, vol)
            if is_flooded:
                nodes_flooding_count += 1

            node_conduit = inlet_conduit_map.get(n.node_name) or outlet_conduit_map.get(n.node_name) or ""
            node_url = nodes_links.get(n.node_name)

            serialized_nodes.append(
                {
                    # Identity
                    "id": n.node_name,
                    "name": n.node_name,
                    "node_name": n.node_name,
                    "conduit": node_conduit,
                    "subcatchment": n.subcatchment,
                    "category": n.sbc_category,
                    "sbc_category": n.sbc_category,
                    # Geometry
                    "invert": inv,
                    "invert_elevation": inv,
                    "depth": dep,
                    "max_depth": dep,
                    "ground": ground_elev,
                    "ground_elevation": ground_elev,
                    # Simulation
                    "waterDepth": clean(n.max_water_depth),
                    "max_water_depth": clean(n.max_water_depth),
                    # Flooding
                    "floodHours": hours,
                    "hours_flooded": hours,
                    "floodVolume": vol,
                    "flood_volume": vol,
                    "is_flooded": is_flooded,
                    # URLs
                    "detail_url": node_url,
                    "panel_url": node_url,
                    "link": node_url,
                }
            )

        # -------------------------------------------------------------------------
        # 3. Subcatchments Serialization
        # -------------------------------------------------------------------------
        serialized_catchments = []

        for s in subcatchments_qs:
            sbc_conduit = sbc_conduit_map.get(s.subcatchment_name, "")
            sbc_url = sbc_links.get(s.subcatchment_name)

            serialized_catchments.append(
                {
                    # Identity
                    "id": s.subcatchment_name,
                    "name": s.subcatchment_name,
                    "subcatchment_name": s.subcatchment_name,
                    "outlet": s.outlet,
                    "conduit": sbc_conduit,
                    "category": s.category,
                    # Characteristics
                    "area": clean(s.area),
                    "impervious": clean(s.perc_imperv),
                    "perc_imperv": clean(s.perc_imperv),
                    "slope": clean(s.perc_slope),
                    "perc_slope": clean(s.perc_slope),
                    # Runoff
                    "total_runoff_mg": clean(s.total_runoff_mg),
                    "runoff": clean(s.total_runoff_mg),
                    "peak_runoff": clean(s.peak_runoff),
                    "peak": clean(s.peak_runoff),
                    "runoff_coeff": clean(s.runoff_coeff),
                    "coefficient": clean(s.runoff_coeff),
                    # URLs
                    "detail_url": sbc_url,
                    "panel_url": sbc_url,
                    "link": sbc_url,
                }
            )

        # -------------------------------------------------------------------------
        # 4. Summary KPIs
        # -------------------------------------------------------------------------
        summary = {
            "conduits_count": len(serialized_conduits),
            "total_conduits": len(serialized_conduits),
            "conduits_changes_count": conduits_changes_count,
            "changes_count": conduits_changes_count,
            "nodes_count": len(serialized_nodes),
            "total_nodes": len(serialized_nodes),
            "nodes_flooding_count": nodes_flooding_count,
            "flooding_count": nodes_flooding_count,
            "catchments_count": len(serialized_catchments),
            "total_catchments": len(serialized_catchments),
        }

        # -------------------------------------------------------------------------
        # 5. Session Metadata
        # -------------------------------------------------------------------------
        # The header renders the file name, date and zone straight from the ``session``
        # object, so the payload carries only what the script itself reads: the id for
        # the CSV file name, the frost zone for the coverage threshold, and active model metadata.
        session_meta = {
            "id": session.id,
            "frost_zone": clean(session.frost_zone),
            "model_metadata": get_model_metadata(),
        }

        return {
            "session": session_meta,
            "summary": summary,
            "conduits": serialized_conduits,
            "nodes": serialized_nodes,
            "catchments": serialized_catchments,
        }


class NavigationService:
    """Service for handling prev/next navigation between conduit panels."""

    @staticmethod
    def get_navigation_context(session: CalculationSession, conduit_name: str) -> Dict[str, Any]:
        """Get previous/next conduit URLs for panel navigation."""
        conduit_names = list(session.conduits.order_by("id").values_list("conduit_name", flat=True))

        try:
            current_index = conduit_names.index(conduit_name)
        except ValueError:
            return {"previous": None, "next": None}

        previous_name = conduit_names[current_index - 1] if current_index > 0 else None
        next_name = conduit_names[current_index + 1] if current_index < len(conduit_names) - 1 else None

        def panel_url(name: str | None) -> str | None:
            if not name:
                return None
            try:
                return reverse("sa:conduit_detail", args=[session.id, name])
            except NoReverseMatch:  # a legal SWMM name with "/" has no page
                return None

        return {
            "previous": {"name": previous_name, "url": panel_url(previous_name)},
            "next": {"name": next_name, "url": panel_url(next_name)},
        }


# =============================================================================
# Conduit Panel
# =============================================================================


def catchment_anchor(subcatchment: SubcatchmentData) -> str:
    """Stable anchor id for a subcatchment row in the conduit panel."""
    return f"catchment-{subcatchment.pk}"


# The seven validation booleans of a conduit, in the order the panel renders them.
VALIDATION_FLAGS: tuple[str, ...] = (
    "val_max_fill",
    "val_max_v",
    "val_min_v",
    "val_max_slope",
    "val_min_slope",
    "val_depth",
    "val_coverage",
)

# The nine recommendation classes of the model: the value of ``ConduitData.recommendation``,
# the label shown to the user and the field holding the confidence of that class.
PREDICTION_CLASSES: tuple[tuple[str, str, str], ...] = (
    ("valid", "No changes needed", "confidence_valid"),
    ("diameter_increase", "Increase diameter", "confidence_diameter_increase"),
    ("diameter_reduction", "Reduce diameter", "confidence_diameter_reduction"),
    ("slope_increase", "Increase slope", "confidence_slope_increase"),
    ("slope_reduction", "Reduce slope", "confidence_slope_reduction"),
    ("depth_increase", "Increase depth", "confidence_depth_increase"),
    ("pump", "Pump station", "confidence_pump"),
    ("tank", "Retention tank", "confidence_tank"),
    ("seepage_boxes", "Seepage boxes", "confidence_seepage_boxes"),
)


def confidence_margin_pct(conduit: ConduitData) -> float | None:
    """Return top-1 minus top-2 recommendation probability margin in percentage points."""
    clean = CalculationRetrievalService._clean_float
    scores = [
        val
        for _code, _label, field in PREDICTION_CLASSES
        if (val := clean(getattr(conduit, field, None))) is not None and 0.0 <= val <= 1.0
    ]
    if len(scores) < 2:
        return None
    scores.sort(reverse=True)
    return clean(round((scores[0] - scores[1]) * 100.0, 1))


# The two comparison signs a check can carry, spelled as the operator of the pipeline.
AT_MOST = "≤"
AT_LEAST = "≥"

# Label of each check, in the order ``VALIDATION_FLAGS`` lists them.
VALIDATION_LABELS: tuple[str, ...] = (
    "Max Filling",
    "Max Velocity",
    "Min Velocity",
    "Max Slope",
    "Min Slope",
    "Depth",
    "Coverage",
)


@dataclass(frozen=True)
class CheckRow:
    """One validation check: the stored verdict plus the numbers behind it.

    ``passed`` is the flag the analysis pipeline wrote, never a fresh verdict, so the
    criteria cards always report what the run concluded.  ``actual`` and ``limit`` are the
    quantity and the threshold that flag compared, formatted for display.
    """

    key: str
    label: str
    passed: bool
    actual: str
    limit: str


@dataclass(frozen=True)
class CriterionCard:
    """One card of the "Results and design criteria" section.

    A card groups the checks that read the same quantity (the two velocity checks, the
    two slope checks), so a quantity is shown once, with its criterion beside it.
    ``passed`` is the AND of the stored flags the card covers and ``None`` for a card
    without a criterion (Max flow); ``status`` names the side that failed (or, for a
    slope range whose minimum lies above its maximum, that there is none).  Every
    number is taken from the ``CheckRow`` strings, so the card inherits their precision
    guarantee (a printed pair never contradicts the stored verdict).
    """

    key: str
    label: str
    value: str
    unit: str
    criterion: str
    passed: bool | None
    status: str


@dataclass(frozen=True)
class DesignRule:
    """One change the rule-based design check asks for (``increase_dia`` & co.)."""

    key: str
    label: str
    change: str


@dataclass(frozen=True)
class PredictionRow:
    """One recommendation class with the confidence the model gave it."""

    key: str
    label: str
    value: float | None
    is_selected: bool


@dataclass(frozen=True)
class CatchmentRow:
    subcatchment: SubcatchmentData
    anchor_id: str


@dataclass(frozen=True)
class NodeCard:
    role: Literal["inlet", "outlet"]
    anchor_id: str
    name: str
    node: NodeData | None
    invert_elevation: float | None
    max_depth: float | None
    max_water_depth: float | None
    hours_flooded: float | None
    flood_volume: float | None
    ground_elevation: float | None
    incoming: list[ConduitData]
    outgoing: list[ConduitData]
    catchments: list[CatchmentRow]

    @property
    def is_flooded(self) -> bool:
        return node_is_flooded(self.hours_flooded, self.flood_volume)


@dataclass(frozen=True)
class ConduitPanel:
    conduit: ConduitData
    inlet: NodeCard
    outlet: NodeCard
    direct_catchments: list[CatchmentRow]
    model_catchment_name: str | None
    model_catchment: SubcatchmentData | None
    model_catchment_differs: bool
    model_catchment_is_direct: bool
    diagram_data: dict[str, Any]
    diagram_json: str
    filling_percent: float | None
    checks: list[CheckRow]
    criteria: list[CriterionCard]
    design_rules: list[DesignRule]
    checks_passed: int
    checks_total: int
    prediction_selected: PredictionRow | None
    prediction_others: list[PredictionRow]
    confidence_margin: float | None = None
    simple_rule_passed: bool = False
    model_metadata: dict[str, Any] | None = None

    @property
    def checks_failed(self) -> int:
        return self.checks_total - self.checks_passed


class ConduitPanelService:
    """Build the data for the conduit detail panel in ≤ 4 queries."""

    @staticmethod
    def build(session: CalculationSession, conduit: ConduitData) -> ConduitPanel:
        inlet_name = conduit.inlet_node
        outlet_name = conduit.outlet_node
        node_names = [inlet_name, outlet_name]

        # --- query 1: nodes ---
        nodes_by_name: dict[str, NodeData] = {}
        for node in session.nodes.filter(node_name__in=node_names):
            nodes_by_name[node.node_name] = node

        # --- query 2: neighbouring conduits ---
        all_conduits = list(
            session.conduits.filter(
                Q(inlet_node__in=node_names) | Q(outlet_node__in=node_names),
            )
        )

        # --- query 3: subcatchments draining to either node ---
        sbc_by_outlet: dict[str, list[SubcatchmentData]] = {}
        for sbc in session.subcatchments.filter(outlet__in=node_names):
            sbc_by_outlet.setdefault(sbc.outlet, []).append(sbc)

        # --- query 4 (conditional): model catchment ---
        raw_sbc_name = conduit.subcatchment
        model_catchment_name: str | None = None if (raw_sbc_name in ("-", "")) else raw_sbc_name
        model_catchment: SubcatchmentData | None = None
        if model_catchment_name is not None:
            model_catchment = session.subcatchments.filter(subcatchment_name=model_catchment_name).first()

        # --- build NodeCards ---
        inlet_card = ConduitPanelService._build_node_card(
            "inlet",
            inlet_name,
            nodes_by_name.get(inlet_name),
            all_conduits,
            conduit,
            sbc_by_outlet.get(inlet_name, []),
        )
        outlet_card = ConduitPanelService._build_node_card(
            "outlet",
            outlet_name,
            nodes_by_name.get(outlet_name),
            all_conduits,
            conduit,
            sbc_by_outlet.get(outlet_name, []),
        )

        direct_catchments = inlet_card.catchments

        # The model feature agrees with the panel only when the inlet node has exactly
        # one direct catchment and that is the one the pipeline picked; with two or more
        # direct catchments the pipeline kept just one of them ("last wins"), so the
        # note has to say so.  False when there is no model catchment at all.
        model_catchment_differs: bool
        model_catchment_is_direct: bool
        if model_catchment_name is None:
            model_catchment_differs = False
            model_catchment_is_direct = False
        else:
            direct_names = [row.subcatchment.subcatchment_name for row in direct_catchments]
            model_catchment_is_direct = model_catchment_name in direct_names
            model_catchment_differs = direct_names != [model_catchment_name]

        # --- diagram_json ---
        inlet_node = nodes_by_name.get(inlet_name)
        outlet_node = nodes_by_name.get(outlet_name)
        inlet_flooded = node_is_flooded(inlet_node.hours_flooded, inlet_node.flood_volume) if inlet_node else False
        outlet_flooded = node_is_flooded(outlet_node.hours_flooded, outlet_node.flood_volume) if outlet_node else False
        diagram_data: dict[str, Any] = {
            "name": conduit.conduit_name,
            "length": float(conduit.length),
            "diameter": float(conduit.geom1),
            "filling": float(conduit.filling),
            "inletNode": conduit.inlet_node,
            "outletNode": conduit.outlet_node,
            "inletElevation": float(conduit.inlet_ground_elevation),
            "outletElevation": float(conduit.outlet_ground_elevation),
            "inletCover": float(conduit.inlet_ground_cover),
            "outletCover": float(conduit.outlet_ground_cover),
            "slope": float(conduit.slope_per_mile),
            "inletInvert": float(inlet_node.invert_elevation) if inlet_node and inlet_node.invert_elevation is not None else None,
            "outletInvert": float(outlet_node.invert_elevation)
            if outlet_node and outlet_node.invert_elevation is not None
            else None,
            "inletMaxDepth": float(conduit.inlet_max_depth),
            "outletMaxDepth": float(conduit.outlet_max_depth),
            "inletWaterDepth": inlet_node.max_water_depth if inlet_node else None,
            "outletWaterDepth": outlet_node.max_water_depth if outlet_node else None,
            "inletFloodHours": inlet_node.hours_flooded if inlet_node else None,
            "outletFloodHours": outlet_node.hours_flooded if outlet_node else None,
            "inletFloodVolume": inlet_node.flood_volume if inlet_node else None,
            "outletFloodVolume": outlet_node.flood_volume if outlet_node else None,
            # The drawing must not re-derive the verdict from the rounded hours.
            "inletFlooded": inlet_flooded,
            "outletFlooded": outlet_flooded,
        }
        diagram_json = json.dumps(diagram_data)

        # --- derived numbers for the criteria section and the prediction card ---
        filling_percent = fill_percent(conduit.filling, conduit.geom1)
        checks = ConduitPanelService._checks(session, conduit)
        checks_passed = sum(1 for row in checks if row.passed)
        criteria = ConduitPanelService._criteria(conduit, checks, filling_percent)
        design_rules = ConduitPanelService._design_rules(conduit)
        prediction_selected, prediction_others = ConduitPanelService._predictions(conduit)

        confidence_margin = confidence_margin_pct(conduit)
        simple_rule = simple_band_rule_passed(conduit.filling, conduit.geom1, conduit.max_v)

        return ConduitPanel(
            conduit=conduit,
            inlet=inlet_card,
            outlet=outlet_card,
            direct_catchments=direct_catchments,
            model_catchment_name=model_catchment_name,
            model_catchment=model_catchment,
            model_catchment_differs=model_catchment_differs,
            model_catchment_is_direct=model_catchment_is_direct,
            diagram_data=diagram_data,
            diagram_json=diagram_json,
            filling_percent=filling_percent,
            checks=checks,
            criteria=criteria,
            design_rules=design_rules,
            checks_passed=checks_passed,
            checks_total=len(checks),
            prediction_selected=prediction_selected,
            prediction_others=prediction_others,
            confidence_margin=confidence_margin,
            simple_rule_passed=simple_rule,
            model_metadata=get_model_metadata(),
        )

    @staticmethod
    def _decimals(actual: float, limit: float, *, at_most: bool, passed: bool, base: int = 2) -> int:
        """Smallest precision at which the printed pair does not contradict the verdict.

        Two decimals turn a velocity of 0.699 m/s into "0.70 m/s" beside a failed
        "at least 0.70 m/s"; one more decimal makes the card readable again.  When the
        raw numbers already disagree with the stored flag no precision can help - the
        base precision then stands and the card keeps showing what the run concluded.
        """

        def holds(left: float, right: float) -> bool:
            return left <= right if at_most else left >= right

        if holds(actual, limit) != passed:
            return base
        decimals = base
        while decimals < 6 and holds(round(actual, decimals), round(limit, decimals)) != passed:
            decimals += 1
        return decimals

    @staticmethod
    def _measure(
        actual: float,
        limit: float,
        *,
        at_most: bool,
        passed: bool,
        unit: str,
        suffix: str = "",
    ) -> tuple[str, str]:
        """Format one measured value and its threshold with a matching precision."""
        decimals = ConduitPanelService._decimals(actual, limit, at_most=at_most, passed=passed)
        sign = AT_MOST if at_most else AT_LEAST
        return f"{actual:.{decimals}f} {unit}", f"{sign} {limit:.{decimals}f} {unit}{suffix}"

    @staticmethod
    def _checks(session: CalculationSession, conduit: ConduitData) -> list[CheckRow]:
        """The seven checks behind the criteria cards: stored verdict, value and threshold.

        Every threshold comes from the code that computed the flag - ``max_filling`` and
        the two velocity limits from ``sa.core.round``, the per-conduit slope limits the
        pipeline stored in ``min_required_slope``/``max_allowable_slope``,
        ``max_depth_value`` for the depth and the session's frost zone for the cover -
        so a change in the analysis moves the cards with it.
        """
        diameter = float(conduit.geom1)
        filling = float(conduit.filling)
        velocity = float(conduit.max_v)
        slope = float(conduit.slope_per_mile)
        measured: list[tuple[bool, str, str]] = []

        # Max Filling: the filling height against 82.7 % of the diameter.  The same
        # check also rejects a diameter outside the standard range, which no filling
        # height can explain, so the card then names that constraint instead.
        passed = bool(conduit.val_max_fill)
        try:
            filling_limit = max_filling(diameter)
        except (TypeError, ValueError):
            measured.append((passed, f"{filling:.2f} m", "Ø outside 0.20–2.00 m"))
        else:
            actual, limit = ConduitPanelService._measure(filling, filling_limit, at_most=True, passed=passed, unit="m")
            measured.append(
                (
                    passed,
                    f"{actual} ({100.0 * filling / diameter:.0f} %)",
                    f"{limit} ({100.0 * filling_limit / diameter:.1f} %)",
                )
            )

        # Both velocity checks read the same quantity: the maximum velocity of the run.
        passed = bool(conduit.val_max_v)
        measured.append(
            (passed, *ConduitPanelService._measure(velocity, max_velocity_value, at_most=True, passed=passed, unit="m/s"))
        )
        passed = bool(conduit.val_min_v)
        measured.append(
            (passed, *ConduitPanelService._measure(velocity, min_velocity_value, at_most=False, passed=passed, unit="m/s"))
        )

        passed = bool(conduit.val_max_slope)
        measured.append(
            (
                passed,
                *ConduitPanelService._measure(slope, float(conduit.max_allowable_slope), at_most=True, passed=passed, unit="‰"),
            )
        )
        passed = bool(conduit.val_min_slope)
        measured.append(
            (
                passed,
                *ConduitPanelService._measure(slope, float(conduit.min_required_slope), at_most=False, passed=passed, unit="‰"),
            )
        )

        # Depth compares each node's invert with the ground minus the maximum depth,
        # which is the node depth itself; the deeper of the two nodes decides.
        passed = bool(conduit.val_depth)
        depth = max(float(conduit.inlet_max_depth), float(conduit.outlet_max_depth))
        measured.append((passed, *ConduitPanelService._measure(depth, max_depth_value, at_most=True, passed=passed, unit="m")))

        # Coverage: the shallower of the two covers has to reach the frost zone.
        passed = bool(conduit.val_coverage)
        cover = min(float(conduit.inlet_ground_cover), float(conduit.outlet_ground_cover))
        measured.append(
            (
                passed,
                *ConduitPanelService._measure(
                    cover, float(session.frost_zone), at_most=False, passed=passed, unit="m", suffix=" (frost zone)"
                ),
            )
        )

        return [
            CheckRow(key=key, label=label, passed=passed, actual=actual, limit=limit)
            for key, label, (passed, actual, limit) in zip(VALIDATION_FLAGS, VALIDATION_LABELS, measured, strict=True)
        ]

    @staticmethod
    def _percent(percent: float, limit: float | None, *, passed: bool) -> str:
        """The filling share, whole percent unless that contradicts the printed limit.

        The limit is printed with one decimal ("82.7 %"), so a filling of 82.6 % that
        passed would read "83 %" beside it; the value then gains a decimal, as ``_measure``
        does for the other quantities.  Without a limit (diameter out of range) or when
        the raw numbers already disagree with the flag the whole percent stands.
        """
        decimals = 0
        if limit is not None and (percent <= limit) == passed:
            while decimals < 3 and (round(percent, decimals) <= round(limit, 1)) != passed:
                decimals += 1
        return f"{percent:.{decimals}f}"

    @staticmethod
    def _criteria(conduit: ConduitData, checks: list[CheckRow], filling_percent: float | None) -> list[CriterionCard]:
        """The six cards of the criteria section, built from the seven ``CheckRow`` items.

        Nothing is compared or formatted anew: values and thresholds are the row strings
        (the precision ``_measure`` chose for them) and verdicts the stored flags.  A range
        joins the two thresholds as they are, so when one of them gained a decimal to stay
        clear of its verdict the two ends may differ in precision ("0.700 – 5.00 m/s").
        """
        rows = {row.key: row for row in checks}

        def number(text: str, unit: str) -> str:
            # "0.27 m/s" -> "0.27": the card prints the unit small, beside the number.
            return text.removesuffix(f" {unit}")

        def bound(text: str, unit: str = "") -> str:
            # "≥ 0.70 m/s" -> "0.70" (min side, unit dropped) / "≤ 5.00 m/s" -> "5.00 m/s".
            for sign in (AT_LEAST, AT_MOST):
                text = text.removeprefix(f"{sign} ")
            return number(text, unit) if unit else text

        def single(key: str, label: str, unit: str, failed: str, value: str | None = None) -> CriterionCard:
            row = rows[key]
            return CriterionCard(
                key=key,
                label=label,
                value=number(row.actual, unit) if value is None else value,
                unit=unit,
                criterion=row.limit,
                passed=row.passed,
                status="within limits" if row.passed else failed,
            )

        def span(key: str, label: str, unit: str, low_key: str, high_key: str, *, empty: str = "") -> CriterionCard:
            # Both checks read the same quantity; the value comes from the failed row when
            # there is one, since that is the one whose precision was raised for its verdict.
            low, high = rows[low_key], rows[high_key]
            passed = low.passed and high.passed
            criterion = f"{bound(low.limit, unit)} – {bound(high.limit)}"
            if not high.passed:
                status = "above maximum"
            elif not low.passed:
                status = "below minimum"
            else:
                status = "within limits"
            # ``empty`` names a range whose minimum lies above its maximum.  A failed card
            # then says that no value can pass instead of printing the inverted range as if
            # it were one; a passed card (impossible with the stored flags) keeps its text.
            if empty and not passed:
                # A no-break space keeps the unit with its number when the long line wraps
                # in a narrow card (390 px left "‰)" alone on the last line).
                high_text = bound(high.limit).replace(" ", "\u00a0")
                criterion = f"{empty} (min {bound(low.limit, unit)} > max {high_text})"
                status = "no feasible range"
            shown = low if (high.passed and not low.passed) else high
            return CriterionCard(
                key=key,
                label=label,
                value=number(shown.actual, unit),
                unit=unit,
                criterion=criterion,
                passed=passed,
                status=status,
            )

        # Filling reads as a share of the diameter; the criterion stays the check-row text
        # (also "Ø outside 0.20–2.00 m", which no filling height can explain).
        fill = rows["val_max_fill"]
        out_of_range = fill.limit.startswith("Ø")
        fill_failed = "diameter out of range" if out_of_range else "exceeds limit"
        if filling_percent is None:
            fill_value = "—"
        else:
            limit_percent = None if out_of_range else 100.0 * max_filling(float(conduit.geom1)) / float(conduit.geom1)
            fill_value = ConduitPanelService._percent(filling_percent, limit_percent, passed=fill.passed)
        return [
            CriterionCard(
                key="max_q",
                label="Max flow",
                value=f"{float(conduit.max_q):.3f}",
                unit="m³/s",
                criterion="SWMM peak",
                passed=None,
                status="",
            ),
            single("val_max_fill", "Filling", "%", fill_failed, value=fill_value),
            span("velocity", "Velocity", "m/s", "val_min_v", "val_max_v"),
            span(
                "slope",
                "Slope",
                "‰",
                "val_min_slope",
                "val_max_slope",
                empty="no feasible slope" if ConduitPanelService._slope_range_is_empty(conduit) else "",
            ),
            single("val_depth", "Depth", "m", "exceeds limit"),
            single("val_coverage", "Cover", "m", "below minimum"),
        ]

    @staticmethod
    def _slope_range_is_empty(conduit: ConduitData) -> bool:
        """True when the stored minimum slope lies above the stored maximum.

        The pipeline derives the minimum from the filling of the run (``min_slope`` in
        ``sa.core.conduits``), so a dry conduit can get a minimum above its maximum, e.g.
        229.72 ‰ against 134.56 ‰ for a 0.30 m pipe.  The panel reports that as it is and
        leaves the stored data and flags alone.
        """
        return float(conduit.min_required_slope) > float(conduit.max_allowable_slope)

    @staticmethod
    def _design_rules(conduit: ConduitData) -> list[DesignRule]:
        """Only the changes the design rules ask for, current value → target.

        The four stored action flags decide, never a fresh comparison: the analysis rounds
        its columns to two places when it saves them, so a comparison made here could
        contradict the criteria cards.  A diameter reduction has no target value in the
        data, so it says only that one is possible; neither does a slope change when the
        stored slope range is empty (see ``_slope_range_is_empty``).  Empty when nothing
        has to change.
        """
        rules: list[DesignRule] = []
        if conduit.increase_dia:
            rules.append(
                DesignRule("increase_dia", "Diameter", f"{float(conduit.geom1):.2f} → {float(conduit.min_diameter):.2f} m")
            )
        if conduit.reduce_dia:
            rules.append(DesignRule("reduce_dia", "Diameter", "could be reduced"))
        slope = float(conduit.slope_per_mile)
        if ConduitPanelService._slope_range_is_empty(conduit):
            # Any target would break the other limit, so a slope rule proposes none; one
            # row covers both flags, which would otherwise print the same text twice.
            key = next((flag for flag in ("increase_slope", "reduce_slope") if getattr(conduit, flag)), None)
            if key is not None:
                change = "no feasible value"
                if float(conduit.max_q) == 0 or float(conduit.filling) == 0:
                    change += " — no flow in the run"
                rules.append(DesignRule(key, "Slope", change))
            return rules
        if conduit.increase_slope:
            rules.append(DesignRule("increase_slope", "Slope", f"{slope:.2f} → {float(conduit.min_required_slope):.2f} ‰"))
        if conduit.reduce_slope:
            rules.append(DesignRule("reduce_slope", "Slope", f"{slope:.2f} → {float(conduit.max_allowable_slope):.2f} ‰"))
        return rules

    @staticmethod
    def _predictions(conduit: ConduitData) -> tuple[PredictionRow | None, list[PredictionRow]]:
        """The recommended class and the remaining ones, most confident first.

        A confidence may be ``None`` (the model did not score that class); it sorts as
        zero but keeps rendering as its own value. An unknown ``recommendation`` leaves
        no row selected and all nine classes in the list.
        """
        rows = [
            PredictionRow(
                key=key,
                label=label,
                value=getattr(conduit, field),
                is_selected=conduit.recommendation == key,
            )
            for key, label, field in PREDICTION_CLASSES
        ]
        selected = next((row for row in rows if row.is_selected), None)
        others = sorted(
            (row for row in rows if not row.is_selected),
            key=lambda row: row.value if row.value is not None else 0.0,
            reverse=True,
        )
        return selected, others

    @staticmethod
    def _build_node_card(
        role: Literal["inlet", "outlet"],
        name: str,
        node: NodeData | None,
        all_conduits: list[ConduitData],
        current_conduit: ConduitData,
        catchment_models: list[SubcatchmentData],
    ) -> NodeCard:
        invert = node.invert_elevation if node else None
        depth = node.max_depth if node else None
        ground: float | None = None
        if invert is not None and depth is not None:
            ground = float(invert) + float(depth)

        incoming = sorted(
            [c for c in all_conduits if c.outlet_node == name and c.pk != current_conduit.pk],
            key=lambda c: c.conduit_name,
        )
        outgoing = sorted(
            [c for c in all_conduits if c.inlet_node == name and c.pk != current_conduit.pk],
            key=lambda c: c.conduit_name,
        )
        catchments = [
            CatchmentRow(subcatchment=sbc, anchor_id=catchment_anchor(sbc))
            for sbc in sorted(catchment_models, key=lambda s: s.subcatchment_name)
        ]

        return NodeCard(
            role=role,
            anchor_id=f"node-{role}",
            name=name,
            node=node,
            invert_elevation=invert,
            max_depth=depth,
            max_water_depth=node.max_water_depth if node else None,
            hours_flooded=node.hours_flooded if node else None,
            flood_volume=node.flood_volume if node else None,
            ground_elevation=ground,
            incoming=incoming,
            outgoing=outgoing,
            catchments=catchments,
        )
