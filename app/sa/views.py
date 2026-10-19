"""Views for stormwater analysis application."""

import csv
import json
import logging
import os
import shutil
import tempfile
from functools import cached_property
from pathlib import Path

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count, IntegerField, OuterRef, Prefetch, Subquery, Value
from django.db.models.functions import Coalesce
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views import View
from django.views.generic import ListView, TemplateView

from .forms import SWMMModelForm
from .models import CalculationSession, ConduitData, NodeData, SubcatchmentData, SWMMModel
from .services import AnalysisService, CalculationRetrievalService, ConduitPanelService, NavigationService

REPO_ROOT = Path(__file__).resolve().parents[2]
_PRIMARY_TEST_FILE = REPO_ROOT / "models" / "recomendations" / "gnn" / "wroclaw" / "wroclaw_pipes.inp"
_FALLBACK_TEST_FILE = REPO_ROOT / "models" / "recomendations" / "wroclaw_pipes.inp"
_PUBLIC_DEMO_FILE = REPO_ROOT / "examples" / "demo_network_coordinates.inp"
TEST_FILE = str(next((p for p in (_PRIMARY_TEST_FILE, _FALLBACK_TEST_FILE) if p.is_file()), _PUBLIC_DEMO_FILE))

logger = logging.getLogger(__name__)

# Prawidłowe (wzorcowe / regułowe) etykiety dla sieci Wrocławia (62 przewody)
# Wyznaczone deterministycznie przez kaskadę reguł eksperckich (classify_conduits)
WROCLAW_RULE_LABELS = {
    "K1": "depth_increase",
    "K2": "depth_increase",
    "K3": "depth_increase",
    "K4": "depth_increase",
    "K5": "depth_increase",
    "K6": "depth_increase",
    "K7": "depth_increase",
    "K8": "depth_increase",
    "K9": "depth_increase",
    "K10": "tank",
    "K11": "tank",
    "K12": "tank",
    "K13": "tank",
    "K14": "tank",
    "K15": "tank",
    "K16": "tank",
    "K17": "tank",
    "K18": "slope_increase",
    "K19": "tank",
    "K20": "tank",
    "K21": "tank",
    "K22": "tank",
    "K23": "tank",
    "K24": "diameter_increase",
    "K25": "diameter_increase",
    "K26": "diameter_increase",
    "K27": "diameter_increase",
    "K28": "diameter_increase",
    "K29": "tank",
    "K30": "tank",
    "K31": "tank",
    "K32": "tank",
    "K33": "tank",
    "K34": "tank",
    "K35": "tank",
    "K36": "valid",
    "K37": "tank",
    "K38": "tank",
    "K39": "tank",
    "K40": "tank",
    "K41": "diameter_increase",
    "K42": "diameter_increase",
    "K43": "slope_increase",
    "K44": "diameter_increase",
    "K45": "diameter_increase",
    "K46": "diameter_reduction",
    "K47": "seepage_boxes",
    "K48": "valid",
    "K49": "diameter_increase",
    "K50": "tank",
    "K51": "tank",
    "K52": "tank",
    "K53": "tank",
    "K54": "tank",
    "K55": "valid",
    "K56": "valid",
    "K57": "valid",
    "K58": "tank",
    "K60": "slope_increase",
    "K62": "slope_increase",
    "K63": "diameter_reduction",
    "K64": "slope_increase",
}


def _inject_rule_labels(data):
    """Wstrzykuje kolumnę 'Rule Label' do tabeli Conduits Data obok Recommendation."""
    if "Conduits Data" in data:
        updated_conduits = []
        for item in data["Conduits Data"]:
            name = item.get("Name")
            new_item = {}
            for k, v in item.items():
                new_item[k] = v
                if k == "Recommendation":
                    new_item["Rule Label"] = WROCLAW_RULE_LABELS.get(name, "-")
            updated_conduits.append(new_item)
        data["Conduits Data"] = updated_conduits
    return data


# =============================================================================
# Mixins
# =============================================================================


class SessionOwnerMixin:
    """Mixin ensuring access only to user's own completed sessions."""

    def get_session(self):
        """Get session owned by current user."""
        return get_object_or_404(
            CalculationSession,
            id=self.kwargs["session_id"],
            user=self.request.user,
        )

    def get_session_with_prefetched_elements(self):
        """Get session owned by current user with conduits, nodes, and subcatchments prefetched in id order."""
        return get_object_or_404(
            CalculationSession.objects.select_related("swmm_model", "geometry").prefetch_related(
                Prefetch("conduits", queryset=ConduitData.objects.order_by("id")),
                Prefetch("nodes", queryset=NodeData.objects.order_by("id")),
                Prefetch("subcatchments", queryset=SubcatchmentData.objects.order_by("id")),
            ),
            id=self.kwargs["session_id"],
            user=self.request.user,
        )

    def validate_completed_or_redirect(self, session):
        """Check if session is completed. Returns redirect response or None."""
        if session.status != "completed":
            messages.warning(self.request, f"Analysis session is {session.status}.")
            return redirect("sa:analysis")
        return None


# =============================================================================
# Simple Views
# =============================================================================


class IndexView(TemplateView):
    """Home page view."""

    template_name = "sa/index.html"


class AboutView(TemplateView):
    """About page view."""

    template_name = "sa/about.html"


# =============================================================================
# Analysis Views
# =============================================================================


class AnalysisView(LoginRequiredMixin, View):
    """View for SWMM file analysis."""

    login_url = "/accounts/login/"
    template_name = "sa/analysis.html"

    def get(self, request):
        """Display analysis form."""
        return render(request, self.template_name, {"swmm_form": SWMMModelForm()})

    def post(self, request):
        """Handle file upload and analysis."""
        if request.POST.get("use_test_file") == "true":
            return self._handle_test_file(request)
        return self._handle_uploaded_file(request)

    @staticmethod
    def _parse_zone(raw_zone) -> int:
        """Parse and validate frost zone against SWMMModel.ZONE_CHOICES (1..4)."""
        valid_zones = {choice[0] for choice in SWMMModel.ZONE_CHOICES}
        try:
            zone_val = int(raw_zone) if raw_zone is not None else 1
            return zone_val if zone_val in valid_zones else 1
        except (TypeError, ValueError):
            return 1

    def _run_analysis_and_respond(
        self,
        request,
        temp_path: str,
        swmm_model: SWMMModel,
        success_msg: str,
        error_prefix: str,
    ):
        """Run SWMM analysis on *temp_path* for *swmm_model* and render the appropriate response."""
        frost_zone = swmm_model.get_frost_zone_value()
        session, success = AnalysisService.analyze_file(
            user=request.user,
            file_path=temp_path,
            frost_zone=frost_zone,
            swmm_model=swmm_model,
        )
        if success:
            messages.success(request, success_msg)
            return self._render_success(request, session)

        detail = f": {session.error_message}" if getattr(session, "error_message", None) else ""
        messages.error(request, f"{error_prefix}{detail}.")
        return self._render_form(request)

    def _handle_test_file(self, request):
        """Process test file analysis on an isolated temporary copy so tracked files are never modified."""
        temp_path = None
        try:
            zone_val = self._parse_zone(request.POST.get("zone"))

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".inp")
            temp_file.close()
            temp_path = temp_file.name
            shutil.copy2(TEST_FILE, temp_path)

            swmm_model = SWMMModel.objects.create(
                user=request.user,
                file=os.path.basename(TEST_FILE),
                zone=zone_val,
            )
            return self._run_analysis_and_respond(
                request,
                temp_path=temp_path,
                swmm_model=swmm_model,
                success_msg="Test file analysis completed successfully!",
                error_prefix="Error occurred while processing test file",
            )
        except Exception as e:
            logger.error(f"Test file analysis error: {e}")
            messages.error(request, f"Error occurred while analyzing test file: {e}")
            return self._render_form(request)
        finally:
            if temp_path:
                self._cleanup_temp_file(temp_path)

    def _handle_uploaded_file(self, request):
        """Process uploaded file analysis."""
        form = SWMMModelForm(request.POST, request.FILES)

        if not form.is_valid():
            return self._render_form(request, form)

        instance = form.save(commit=False)
        instance.user = request.user
        uploaded_file = request.FILES["file"]

        temp_path = None
        try:
            temp_path = self._create_temp_file(uploaded_file)
            instance.file = uploaded_file.name
            instance.save()

            return self._run_analysis_and_respond(
                request,
                temp_path=temp_path,
                swmm_model=instance,
                success_msg="Analysis completed successfully!",
                error_prefix="Error occurred while saving calculation results",
            )
        finally:
            if temp_path:
                self._cleanup_temp_file(temp_path)

    def _render_form(self, request, form=None):
        """Render analysis form."""
        return render(request, self.template_name, {"swmm_form": form or SWMMModelForm()})

    def _render_success(self, request, session):
        """Render successful analysis results."""
        data = CalculationRetrievalService.format_session_data_for_template(session)
        data = _inject_rule_labels(data)
        panel_links = CalculationRetrievalService.panel_links(session)
        return render(
            request,
            self.template_name,
            {
                "swmm_form": SWMMModelForm(),
                "data": data,
                "session": session,
                "show_results": True,
                "panel_links": panel_links,
            },
        )

    @staticmethod
    def _create_temp_file(uploaded_file) -> str:
        """Create temporary file from uploaded file, cleaning up on write error."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".inp")
        try:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_file.close()
            return temp_file.name
        except Exception:
            temp_file.close()
            try:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
            except OSError:
                pass
            raise

    @staticmethod
    def _cleanup_temp_file(path: str):
        """Remove temporary .inp file along with any SWMM-generated .rpt, .out, and .bak sibling files."""
        base, _ = os.path.splitext(path)
        for candidate in (path, f"{base}.rpt", f"{base}.out", f"{base}.bak"):
            try:
                if os.path.exists(candidate):
                    os.unlink(candidate)
            except OSError:
                pass


class AnalysisResultsView(LoginRequiredMixin, SessionOwnerMixin, TemplateView):
    """View for displaying analysis results."""

    login_url = "/accounts/login/"
    template_name = "sa/analysis_results.html"

    def get_session(self):
        """Get session owned by current user, prefetching related objects ordered by id."""
        return self.get_session_with_prefetched_elements()

    def get(self, request, *args, **kwargs):
        session = self.get_session()

        redirect_response = self.validate_completed_or_redirect(session)
        if redirect_response:
            return redirect_response

        results_v2_data = CalculationRetrievalService.serialize_session_for_results_v2(
            session=session,
            rule_labels=WROCLAW_RULE_LABELS,
        )

        # The map envelope is always present; ``available`` tells the template and the
        # browser whether there is anything to draw.  ``geometry`` is already selected
        # with the session, so this costs no extra query.
        network_map_data = CalculationRetrievalService.serialize_geometry_for_map(session)

        return render(
            request,
            self.template_name,
            {
                "session": session,
                "results_v2_data": results_v2_data,
                "network_map_data": network_map_data,
            },
        )


# =============================================================================
# Detail Views
# =============================================================================


class ConduitDetailView(LoginRequiredMixin, SessionOwnerMixin, TemplateView):
    """The conduit panel: the conduit, its two nodes and the catchments of the inlet node."""

    login_url = "/accounts/login/"
    template_name = "sa/conduit_panel.html"

    def get(self, request, *args, **kwargs):
        session = self.get_session()

        redirect_response = self.validate_completed_or_redirect(session)
        if redirect_response:
            return redirect_response

        conduit = session.conduits.filter(conduit_name=kwargs["conduit_name"]).first()
        if not conduit:
            messages.error(request, f"Conduit '{kwargs['conduit_name']}' not found in this session.")
            return redirect("sa:analysis_results", session_id=session.id)

        panel = ConduitPanelService.build(session, conduit)
        navigation = NavigationService.get_navigation_context(session, kwargs["conduit_name"])

        context = {
            "session": session,
            "conduit": conduit,
            "data_type": "conduit",
            "navigation": navigation,
            "inlet_node": panel.inlet.node,
            "outlet_node": panel.outlet.node,
            "subcatchment": panel.model_catchment,
            "conduit_json": panel.diagram_json,
            "rule_label": WROCLAW_RULE_LABELS.get(conduit.conduit_name, "-"),
            "panel": panel,
        }

        return self.render_to_response(context)


# =============================================================================
# History View
# =============================================================================


class HistoryView(LoginRequiredMixin, ListView):
    """User's calculation history: search, status filter, sorting and pagination.

    Every control is a GET parameter, so a filtered page is a shareable URL and the
    pagination links only have to carry the query string forward.
    """

    login_url = "/accounts/login/"
    template_name = "sa/history.html"
    context_object_name = "sessions"
    paginate_by = 10
    page_size_options = [10, 20, 50, 100]
    status_options = ("completed", "processing", "failed")
    sort_options = {"newest": "-created_at", "oldest": "created_at", "name": "swmm_model__file"}
    default_sort = "newest"

    def get_paginate_by(self, queryset):
        per_page = self.request.GET.get("per_page")
        if per_page is None:
            return self.paginate_by
        try:
            per_page = int(per_page)
            if per_page in self.page_size_options:
                return per_page
            logger.warning(f"Invalid per_page value: {per_page}. Allowed: {self.page_size_options}")
        except (ValueError, TypeError):
            logger.warning(f"Invalid per_page parameter: {per_page!r}")
        return self.paginate_by

    @cached_property
    def filters(self) -> dict[str, str]:
        """The validated ``q``/``status``/``sort`` of this request.

        Cached because both ``get_queryset`` and ``get_context_data`` need them and an
        invalid value should be logged once per request, not once per reader.
        """
        return {
            "q": self.request.GET.get("q", "").strip(),
            "status": self._validated_choice("status", self.status_options, ""),
            "sort": self._validated_choice("sort", self.sort_options, self.default_sort),
        }

    def _validated_choice(self, parameter: str, allowed, default: str) -> str:
        """Read one GET parameter; an absent or empty value means "not set"."""
        value = self.request.GET.get(parameter, "").strip()
        if not value:
            return default
        if value in allowed:
            return value
        logger.warning(f"Invalid {parameter} value: {value!r}. Allowed: {sorted(allowed)}")
        return default

    @staticmethod
    def _related_count(model):
        """Rows of one related model per session, as a scalar subquery.

        Three ``Count()`` annotations over three relations would join them together and
        multiply the rows; each relation therefore gets its own subquery. ``Coalesce``
        turns the ``NULL`` of an empty relation into a ``0`` the template can print.
        """
        return Coalesce(
            Subquery(
                model.objects.filter(session=OuterRef("pk")).order_by().values("session").annotate(c=Count("pk")).values("c")[:1],
                output_field=IntegerField(),
            ),
            Value(0),
        )

    def get_queryset(self):
        queryset = (
            CalculationSession.objects.filter(user=self.request.user)
            .select_related("swmm_model")
            .annotate(
                n_conduits=self._related_count(ConduitData),
                n_nodes=self._related_count(NodeData),
                n_subcatchments=self._related_count(SubcatchmentData),
            )
        )
        filters = self.filters
        if filters["q"]:
            queryset = queryset.filter(swmm_model__file__icontains=filters["q"])
        if filters["status"]:
            queryset = queryset.filter(status=filters["status"])
        # The primary key breaks ties: "name" sorts by a column that repeats freely
        # (the same file uploaded twice), and LIMIT/OFFSET over a non-unique order
        # lets a row show up on two pages while another never shows up at all.
        return queryset.order_by(self.sort_options[filters["sort"]], "-pk")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        filters = self.filters
        context.update(
            {
                "page_size_options": self.page_size_options,
                "current_per_page": self.get_paginate_by(self.object_list),
                "current_q": filters["q"],
                "current_status": filters["status"],
                "current_sort": filters["sort"],
                # Tells the empty state whether the list is empty because nothing matched
                # or because the account has no sessions - without a second query.
                "filters_active": bool(filters["q"] or filters["status"]),
            }
        )
        return context


# =============================================================================
# AJAX Views
# =============================================================================


class DeleteSessionView(LoginRequiredMixin, SessionOwnerMixin, View):
    """AJAX endpoint for deleting a calculation session. Returns JSON response."""

    login_url = "/accounts/login/"

    def post(self, request, *args, **kwargs):
        session = self.get_session()
        session_id = session.id

        try:
            session.delete()
            logger.info(f"Successfully deleted session {session_id}")
            return JsonResponse({"success": True, "message": f"Session {session_id} deleted successfully."})
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return JsonResponse(
                {"success": False, "message": "An error occurred while deleting the session."},
                status=500,
            )


class BulkDeleteSessionsView(LoginRequiredMixin, View):
    """AJAX endpoint for bulk deleting calculation sessions. Returns JSON response."""

    login_url = "/accounts/login/"

    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            session_ids = data.get("session_ids", [])

            if not session_ids:
                return JsonResponse({"success": False, "message": "No sessions selected."}, status=400)

            sessions = CalculationSession.objects.filter(id__in=session_ids, user=request.user)
            deleted_count = sessions.count()

            if deleted_count == 0:
                return JsonResponse({"success": False, "message": "No valid sessions found."}, status=404)

            sessions.delete()
            logger.info(f"Successfully bulk deleted {deleted_count} sessions for user {request.user.id}")

            return JsonResponse(
                {"success": True, "message": f"Successfully deleted {deleted_count} session(s).", "deleted_count": deleted_count}
            )
        except json.JSONDecodeError:
            return JsonResponse({"success": False, "message": "Invalid request data."}, status=400)
        except Exception as e:
            logger.error(f"Error bulk deleting sessions: {e}")
            return JsonResponse(
                {"success": False, "message": "An error occurred while deleting sessions."},
                status=500,
            )


class ExportSessionCsvView(LoginRequiredMixin, SessionOwnerMixin, View):
    """Server-side full CSV export for conduits, nodes, or catchments (un-truncated by pagination)."""

    login_url = "/accounts/login/"
    ALLOWED_KINDS = ("conduits", "nodes", "catchments")

    def get_session(self):
        return self.get_session_with_prefetched_elements()

    @staticmethod
    def _sanitize_csv_cell(val):
        """Neutralize spreadsheet formula injection (CWE-1236) on string cells."""
        if not isinstance(val, str) or not val or val == "-":
            return val
        if val[0] in ("=", "+", "@", "\t", "\r"):
            return f"'{val}"
        if val[0] == "-":
            try:
                float(val)
                return val
            except ValueError:
                return f"'{val}"
        return val

    def get(self, request, *args, **kwargs):
        kind = kwargs.get("kind", "")
        if kind not in self.ALLOWED_KINDS:
            return HttpResponseBadRequest("Invalid export kind.")

        session = self.get_session()
        redirect_response = self.validate_completed_or_redirect(session)
        if redirect_response:
            return redirect_response

        payload = CalculationRetrievalService.serialize_session_for_results_v2(
            session=session,
            rule_labels=WROCLAW_RULE_LABELS,
        )
        rows = payload.get(kind, [])

        response = HttpResponse(content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = f'attachment; filename="session_{session.id}_{kind}.csv"'
        response.write("\ufeff")

        if rows:
            fieldnames = [k for k in rows[0].keys() if not isinstance(rows[0][k], (list, dict))]
            writer = csv.DictWriter(response, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: self._sanitize_csv_cell(row.get(k)) for k in fieldnames})
        return response
