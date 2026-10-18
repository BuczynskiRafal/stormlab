"""
Django development settings.

Usage:
    export DJANGO_ENV=development  (or leave unset - this is the default)
"""

import os

from .base import *  # noqa: F401, F403

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv("SECRET_KEY", "django-insecure-dev-key-do-not-use-in-production")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

# Development-specific logging
LOGGING["loggers"]["sa"]["level"] = "DEBUG"  # noqa: F405
