from django.contrib import admin

from sa.models import SWMMModel


@admin.register(SWMMModel)
class SWMMModelAdmin(admin.ModelAdmin):
    list_display = ["user", "file", "zone"]
    filter = [
        "user",
    ]
