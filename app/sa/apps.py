from django.apps import AppConfig


class SaConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "sa"

    def ready(self) -> None:
        from . import checks  # noqa: F401  (registers the system checks)
