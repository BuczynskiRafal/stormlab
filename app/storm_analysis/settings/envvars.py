"""Read and validate settings from environment variables.

Every function takes the environment as an argument so it can be tested without
re-importing the settings. A variable that is unset or blank counts as missing.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from django.core.checks.security.base import (
    SECRET_KEY_INSECURE_PREFIX,
    SECRET_KEY_MIN_LENGTH,
    SECRET_KEY_MIN_UNIQUE_CHARACTERS,
)
from django.core.exceptions import ImproperlyConfigured

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})
_PLACEHOLDER_KEYS = ("your-secret-key", "change-me", "changeme")


def _raw(name: str, environ: Mapping[str, str]) -> str | None:
    value = environ.get(name, "").strip()
    return value or None


def env_path(name: str, default: Path | str, environ: Mapping[str, str] = os.environ) -> Path:
    return Path(_raw(name, environ) or default)


def env_bool(name: str, default: bool, environ: Mapping[str, str] = os.environ) -> bool:
    value = _raw(name, environ)
    if value is None:
        return default
    if value.lower() in _TRUE:
        return True
    if value.lower() in _FALSE:
        return False
    raise ImproperlyConfigured(f"{name}={value!r} is not a boolean; use true or false.")


def env_int(name: str, default: int, environ: Mapping[str, str] = os.environ, *, minimum: int = 0) -> int:
    value = _raw(name, environ)
    if value is None:
        return default
    try:
        number = int(value)
    except ValueError:
        raise ImproperlyConfigured(f"{name}={value!r} is not an integer.") from None
    if number < minimum:
        raise ImproperlyConfigured(f"{name} must be at least {minimum} (got {number}).")
    return number


def env_list(name: str, environ: Mapping[str, str] = os.environ) -> list[str]:
    """Split a comma-separated variable into stripped, non-empty items."""
    return [item.strip() for item in environ.get(name, "").split(",") if item.strip()]


def required_list(name: str, environ: Mapping[str, str] = os.environ) -> list[str]:
    items = env_list(name, environ)
    if not items:
        raise ImproperlyConfigured(f"{name} must list at least one value (comma-separated).")
    return items


def secret_key(environ: Mapping[str, str] = os.environ) -> str:
    """Return ``SECRET_KEY``, refusing values that Django's deploy check would flag."""
    value = _raw("SECRET_KEY", environ)
    if value is None:
        raise ImproperlyConfigured("SECRET_KEY must be set in production.")
    if value.startswith(SECRET_KEY_INSECURE_PREFIX) or value.lower().startswith(_PLACEHOLDER_KEYS):
        raise ImproperlyConfigured("SECRET_KEY is a placeholder or development key; generate a new one.")
    if len(value) < SECRET_KEY_MIN_LENGTH or len(set(value)) < SECRET_KEY_MIN_UNIQUE_CHARACTERS:
        raise ImproperlyConfigured(
            f"SECRET_KEY must be at least {SECRET_KEY_MIN_LENGTH} characters long "
            f"with at least {SECRET_KEY_MIN_UNIQUE_CHARACTERS} distinct characters."
        )
    return value
