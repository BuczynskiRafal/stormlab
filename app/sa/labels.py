"""What the user reads for the internal codes of a recommendation and a land cover.

The database, the model and the CSV machine columns keep the codes (``slope_increase``,
``urban_weakly_impervious``); every place that shows one to a person goes through the
two lookups below.  The results page hands both maps to its scripts in the
``results-v2-data`` payload, so the browser holds no second copy.
"""

from __future__ import annotations

from typing import Any

# One label per class of ``RecommendationCategory``, in the order the filters list them.
ACTION_LABELS: dict[str, str] = {
    "valid": "No changes",
    "diameter_increase": "Increase diameter",
    "diameter_reduction": "Reduce diameter",
    "slope_increase": "Increase slope",
    "slope_reduction": "Reduce slope",
    "depth_increase": "Increase depth",
    "pump": "Pump station",
    "tank": "Retention tank",
    "seepage_boxes": "Seepage boxes",
}

# One label per entry of ``SUBCATCHMENT_CATEGORIES`` (the twelve land-cover classes).
CATEGORY_LABELS: dict[str, str] = {
    "arable": "Arable land",
    "forests": "Forests",
    "marshes": "Marshes",
    "meadows": "Meadows",
    "mountains_rocky": "Mountains, rocky",
    "mountains_vegetated": "Mountains, vegetated",
    "rural": "Rural",
    "suburban_highly_impervious": "Suburban, highly impervious",
    "suburban_weakly_impervious": "Suburban, weakly impervious",
    "urban_highly_impervious": "Urban, highly impervious",
    "urban_moderately_impervious": "Urban, moderately impervious",
    "urban_weakly_impervious": "Urban, weakly impervious",
}

# What an absent value reads as.  The pipeline stores "-" for "no catchment".
MISSING = "—"


def _lookup(table: dict[str, str], code: Any) -> str:
    if code is None:
        return MISSING
    text = str(code).strip()
    if text in ("", "-", MISSING):
        return MISSING
    key = text.lower()
    if key in table:
        return table[key]
    # A code nobody labelled (a future class, an odd [TAGS] entry) still reads as words.
    words = key.replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def action_label(code: Any) -> str:
    """The label of a recommendation code; unknown codes are spelled out as words."""
    return _lookup(ACTION_LABELS, code)


def category_label(code: Any) -> str:
    """The label of a land-cover code; unknown codes are spelled out as words."""
    return _lookup(CATEGORY_LABELS, code)
