from django.urls import path

from .views import (
    AboutView,
    AnalysisResultsView,
    AnalysisView,
    BulkDeleteSessionsView,
    ConduitDetailView,
    DeleteSessionView,
    ExportSessionCsvView,
    HistoryView,
    IndexView,
)

app_name = "sa"

urlpatterns = [
    path("", IndexView.as_view(), name="index"),
    path("about/", AboutView.as_view(), name="about"),
    path("analysis/", AnalysisView.as_view(), name="analysis"),
    path("session/<int:session_id>/delete/", DeleteSessionView.as_view(), name="delete_session"),
    path("sessions/bulk-delete/", BulkDeleteSessionsView.as_view(), name="bulk_delete_sessions"),
    path("analysis/results/<int:session_id>/", AnalysisResultsView.as_view(), name="analysis_results"),
    path("analysis/results/<int:session_id>/export/<str:kind>/", ExportSessionCsvView.as_view(), name="export_session_csv"),
    # ``path`` rather than ``str``: a legal SWMM name may contain "/" and still has to route.
    path("analysis/results/<int:session_id>/conduit/<path:conduit_name>/", ConduitDetailView.as_view(), name="conduit_detail"),
    path("history/", HistoryView.as_view(), name="history"),
]

