"""Filters that turn stored codes and raw values into what the reader sees.

``action_label`` and ``category_label`` go through ``sa.labels``, the one place the
display labels live; ``utc_datetime`` writes a timestamp the way every page writes
one ("15 Nov 2026, 04:43 UTC"); ``zone_label`` names the frost zone of a model the way
every page names it ("Zone I (0.8 m)"); ``file_label`` is the bare name of an uploaded
file.  Templates use these filters and nothing else for dates and zones, so that the
site writes each of them in one way.
"""

from __future__ import annotations

import datetime
import posixpath

from django import template

from sa.labels import action_label as _action_label
from sa.labels import category_label as _category_label
from sa.models import zone_label as _zone_label

register = template.Library()


@register.filter
def action_label(value):
    return _action_label(value)


@register.filter
def category_label(value):
    return _category_label(value)


@register.filter
def utc_datetime(value, part="full"):
    """``15 Nov 2026, 04:43 UTC``; an empty string for anything that is not a datetime.

    The day has no leading zero ("5 Mar 2026, 09:07 UTC").  ``part`` picks a piece of
    the same text for a layout that sets them apart: ``"date"`` gives "15 Nov 2026" and
    ``"time"`` gives "04:43 UTC".
    """
    if not isinstance(value, datetime.datetime):
        return ""
    if value.tzinfo is not None:
        value = value.astimezone(datetime.timezone.utc)
    date, time = f"{value.day} {value:%b %Y}", f"{value:%H:%M} UTC"
    if part == "date":
        return date
    if part == "time":
        return time
    return f"{date}, {time}"


@register.filter
def zone_label(swmm_model):
    """``Zone III (1.2 m)`` for an uploaded model; an empty string without one."""
    zone = getattr(swmm_model, "zone", None)
    if zone is None:
        return ""
    return _zone_label(zone, swmm_model.get_frost_zone_value())


@register.filter
def file_label(value):
    """The file name without any directory part; ``model.inp`` when there is none."""
    name = str(getattr(value, "name", value) or "").replace("\\", "/")
    return posixpath.basename(name) or "model.inp"
