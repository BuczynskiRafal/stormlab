"""Filters that turn stored codes and raw values into what the reader sees.

``action_label`` and ``category_label`` go through ``sa.labels``, the one place the
display labels live; ``utc_datetime`` writes a timestamp the way every page writes
one ("15 Nov 2026, 04:43 UTC"); ``file_label`` is the bare name of an uploaded file.
"""

from __future__ import annotations

import datetime
import posixpath

from django import template

from sa.labels import action_label as _action_label
from sa.labels import category_label as _category_label

register = template.Library()


@register.filter
def action_label(value):
    return _action_label(value)


@register.filter
def category_label(value):
    return _category_label(value)


@register.filter
def utc_datetime(value):
    """``15 Nov 2026, 04:43 UTC``; an empty string for anything that is not a datetime."""
    if not isinstance(value, datetime.datetime):
        return ""
    if value.tzinfo is not None:
        value = value.astimezone(datetime.timezone.utc)
    return f"{value.day} {value:%b %Y, %H:%M} UTC"


@register.filter
def file_label(value):
    """The file name without any directory part; ``model.inp`` when there is none."""
    name = str(getattr(value, "name", value) or "").replace("\\", "/")
    return posixpath.basename(name) or "model.inp"
