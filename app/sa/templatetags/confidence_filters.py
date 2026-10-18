from django import template

register = template.Library()


@register.filter
def to_percent(value):
    """Convert 0-1 value to percentage with 1 decimal place"""
    try:
        return f"{float(value) * 100:.1f}"
    except (ValueError, TypeError):
        return "0.0"


@register.filter
def to_percent_2(value):
    """Convert 0-1 value to percentage with 2 decimal places"""
    try:
        return f"{float(value) * 100:.2f}"
    except (ValueError, TypeError):
        return "0.00"
