# ruff: noqa: F401
from django.apps import AppConfig


class AccountsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "accounts"


class UsersConfig(AppConfig):
    name = "accounts"

    def ready(self):
        import accounts.signals
