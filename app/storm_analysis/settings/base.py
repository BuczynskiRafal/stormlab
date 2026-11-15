"""
Base Django settings shared across all environments.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .envvars import env_bool, env_int, env_path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
# BASE_DIR points to the 'app' directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(BASE_DIR / ".env")
load_dotenv()

# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "crispy_forms",
    "crispy_bootstrap4",
    "accounts.apps.AccountsConfig",
    "sa.apps.SaConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "storm_analysis.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "storm_analysis" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "storm_analysis.context_processors.registration",
            ],
        },
    },
]

WSGI_APPLICATION = "storm_analysis.wsgi.application"

# Database - SQLite for all environments
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": env_path("SQLITE_PATH", BASE_DIR / "db.sqlite3"),
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Custom user model
AUTH_USER_MODEL = "accounts.User"
LOGIN_URL = "accounts:login"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/"

# Self-registration; when false the sign-up page answers 404 and its links are hidden.
REGISTRATION_OPEN = env_bool("REGISTRATION_OPEN", True)

# Crispy Forms
CRISPY_TEMPLATE_PACK = "bootstrap4"

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Media files
MEDIA_URL = "/media/"
MEDIA_ROOT = str(env_path("MEDIA_ROOT", os.path.join(BASE_DIR, "media")))

# Largest accepted .inp upload, in bytes.
MAX_INP_BYTES = env_int("MAX_INP_BYTES", 50 * 1024 * 1024, minimum=1)

# What one analysis may cost, checked by the upload form from the file itself. A run lives
# inside a request (gunicorn timeout 180 s) and writes its binary output to /tmp (a 512 MB
# tmpfs in production), whose size grows with elements x report steps.
# Simulated time, START to END: design storms run for hours; 72 h keeps a 3-day event with
# its drain-down and is 6x the 12-hour demo networks.
MAX_SIMULATION_HOURS = env_int("MAX_SIMULATION_HOURS", 72, minimum=1)
# Report steps over the reporting period (period / REPORT_STEP); the Rakowiec model has 240.
MAX_REPORT_STEPS = env_int("MAX_REPORT_STEPS", 1000, minimum=1)
# Conduits + nodes + subcatchments; an estate-scale network has a few hundred (Rakowiec 162).
MAX_MODEL_ELEMENTS = env_int("MAX_MODEL_ELEMENTS", 5000, minimum=1)

# Upload limits (50 MB max .inp model + request overhead; stream files > 5 MB via temporary file)
DATA_UPLOAD_MAX_MEMORY_SIZE = 52 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024

# Static files (CSS, JavaScript, Images)
STATICFILES_DIRS = [os.path.join(BASE_DIR, "static")]
STATIC_ROOT = str(env_path("STATIC_ROOT", os.path.join(BASE_DIR, "staticfiles")))
STATIC_URL = "/static/"

# The health check must answer over plain HTTP; this only matters when SECURE_SSL_REDIRECT is on.
SECURE_REDIRECT_EXEMPT = [r"^healthz/$"]

# Logging configuration
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(asctime)s - %(levelname)-8s [%(filename)s:%(lineno)d] - %(message)s",
        },
        "simple": {
            "format": "%(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "sa": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "django": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}
