"""
Django settings module with environment-based configuration.

Usage:
    Set DJANGO_ENV environment variable to select configuration:
    - development (default)
    - production
    - testing

Example:
    export DJANGO_ENV=production
    python manage.py runserver
"""

import os
from django.core.exceptions import ImproperlyConfigured

env = os.getenv("DJANGO_ENV", "development").strip()

if env == "production":
    from .production import *  # noqa: F401, F403
elif env == "testing":
    from .testing import *  # noqa: F401, F403
elif env == "development":
    from .development import *  # noqa: F401, F403
else:
    raise ImproperlyConfigured(
        f"Unknown DJANGO_ENV={env!r}; expected one of: 'development', 'production', 'testing'."
    )

