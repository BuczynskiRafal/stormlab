"""
Django testing settings optimized for pytest.

Usage:
    export DJANGO_ENV=testing
    pytest
"""

import os

from .base import *  # noqa: F401, F403

SECRET_KEY = os.getenv("SECRET_KEY", "django-insecure-test-key-for-testing-only")

DEBUG = False

ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]

# Use faster password hasher for tests
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.MD5PasswordHasher",
]

# Use in-memory SQLite for faster tests
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

# Disable logging during tests to reduce noise
LOGGING = {
    "version": 1,
    "disable_existing_loggers": True,
    "handlers": {
        "null": {
            "class": "logging.NullHandler",
        },
    },
    "root": {
        "handlers": ["null"],
        "level": "CRITICAL",
    },
}
