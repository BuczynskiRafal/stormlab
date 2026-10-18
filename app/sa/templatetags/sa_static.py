"""``{% versioned_static %}``: a static URL that changes whenever the file does.

The app serves its stylesheet and scripts under fixed names and without a
``Cache-Control`` header, so a browser may keep using a stale copy for hours after
the file changed (Safari does, heuristically, from ``Last-Modified`` alone). Appending
the modification time of the source file makes every change a new URL.
"""

from __future__ import annotations

import os

from django import template
from django.contrib.staticfiles import finders
from django.templatetags.static import static

register = template.Library()


@register.simple_tag
def versioned_static(path: str) -> str:
    """Return ``static(path)`` with ``?v=<mtime>``; the bare URL if the file is not found."""
    url = static(path)
    source = finders.find(path)
    if not source:
        return url
    return f"{url}?v={int(os.path.getmtime(source))}"
