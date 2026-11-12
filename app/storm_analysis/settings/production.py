"""
Django production settings.

Usage:
    export DJANGO_ENV=production
    export SECRET_KEY=<50+ random characters>
    export ALLOWED_HOSTS=your-domain.com,www.your-domain.com

See app/.env.example for the optional variables.
"""

from .base import *  # noqa: F401, F403
from .envvars import env_bool, env_int, env_list, required_list, secret_key

SECRET_KEY = secret_key()

DEBUG = False

ALLOWED_HOSTS = required_list("ALLOWED_HOSTS")
CSRF_TRUSTED_ORIGINS = env_list("CSRF_TRUSTED_ORIGINS")

# HTTPS is terminated by the reverse proxy, which must overwrite X-Forwarded-Proto.
SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# HSTS is hard to undo once browsers have cached it, so start short and opt in to the rest.
SECURE_HSTS_SECONDS = env_int("SECURE_HSTS_SECONDS", 3600, minimum=1)
SECURE_HSTS_INCLUDE_SUBDOMAINS = env_bool("SECURE_HSTS_INCLUDE_SUBDOMAINS", False)
SECURE_HSTS_PRELOAD = env_bool("SECURE_HSTS_PRELOAD", False)
# These warnings only say that an opt-in above is off, which is a deliberate choice here.
SILENCED_SYSTEM_CHECKS = [
    *([] if SECURE_HSTS_INCLUDE_SUBDOMAINS else ["security.W005"]),
    *([] if SECURE_HSTS_PRELOAD else ["security.W021"]),
]

# Several worker processes share one SQLite file: wait for locks instead of failing,
# take the write lock at the start of a transaction, and let readers run during writes.
DATABASES["default"]["OPTIONS"] = {  # noqa: F405
    "timeout": 20,
    "transaction_mode": "IMMEDIATE",
    "init_command": "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;",
}

# WhiteNoise serves the collected static files; media files are served by the proxy.
MIDDLEWARE = list(MIDDLEWARE)  # noqa: F405
MIDDLEWARE.insert(
    MIDDLEWARE.index("django.middleware.security.SecurityMiddleware") + 1, "whitenoise.middleware.WhiteNoiseMiddleware"
)
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}

# Production logging - less verbose
LOGGING["root"]["level"] = "WARNING"  # noqa: F405
LOGGING["loggers"]["sa"]["level"] = "WARNING"  # noqa: F405
