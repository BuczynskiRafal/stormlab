"""
Django production settings.

Usage:
    export DJANGO_ENV=production
    export SECRET_KEY=your-secure-secret-key
    export ALLOWED_HOSTS=your-domain.com,www.your-domain.com
"""

import os

from .base import *  # noqa: F401, F403

# SECURITY: Secret key must be set via environment variable
SECRET_KEY = os.environ["SECRET_KEY"]

DEBUG = False

# SECURITY: Set allowed hosts from environment
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",")

# Security settings for HTTPS
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Production logging - less verbose
LOGGING["root"]["level"] = "WARNING"  # noqa: F405
LOGGING["loggers"]["sa"]["level"] = "WARNING"  # noqa: F405
