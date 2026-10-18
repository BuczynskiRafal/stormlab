"""Query-string helpers for links that have to preserve the current filters."""

from django import template
from django.http import QueryDict
from django.utils.http import urlencode

register = template.Library()


@register.simple_tag(takes_context=True)
def querystring(context, **overrides) -> str:
    """Return the current query string with ``overrides`` applied.

    ``{% querystring page=3 %}`` on ``/history/?q=alpha&page=2`` renders
    ``?q=alpha&amp;page=3`` - the autoescaping of ``simple_tag`` is what makes it a
    valid ``href`` attribute. Keys whose value is ``None`` or empty are dropped, so a
    link never carries ``?status=&sort=``.

    Django gained a built-in ``{% querystring %}`` in 5.1; this project runs 5.0.
    """
    request = context.get("request")
    params = request.GET.copy() if request is not None else QueryDict(mutable=True)

    for key, value in overrides.items():
        params[key] = value

    # Single-valued on purpose: every control on the history page is a single-valued
    # GET parameter, and repeated keys would only make the links ambiguous.
    query = {key: value for key, value in params.items() if value not in (None, "")}
    encoded = urlencode(query)
    return f"?{encoded}" if encoded else ""


@register.filter
def dict_get(mapping, key):
    """Safe dictionary lookup for chained access in templates."""
    return mapping.get(key) if isinstance(mapping, dict) else None
