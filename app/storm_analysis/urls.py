from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import include, path

from storm_analysis.health import healthz

urlpatterns = (
    [
        path("healthz/", healthz, name="healthz"),
        path("admin/", admin.site.urls),
        path("", include("sa.urls")),
        path("accounts/", include("accounts.urls")),
    ]
    + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
)

urlpatterns += staticfiles_urlpatterns()
