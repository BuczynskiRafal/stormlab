"""Subcatchment land cover: reading it from ``[TAGS]`` and one-hot encoding it.

Ported from ``SubcatchmentFeatureEngineeringService`` in
``app/sa/core/subcatchments.py``, minus its coupling to a live ``swmmio.Model``:
the tags are read once by :func:`stormlab.io.inp.read_tags` and arrive here as a
plain Series. The commented-out ML branch of the original is not carried over —
it divided four columns by ``Area`` without the ``sum() != 0`` guard the training
notebook used, and the corrected version lives in
:mod:`stormlab.features.catchment`.
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

CATEGORY_COLUMN = "category"

NO_CATEGORY_SENTINEL = "-"
"""Placeholder the conduit pipeline writes for "no subcatchment reaches this element".

It is a second spelling of "no land cover information" next to ``NaN`` and encodes
to the same all-zero one-hot row.
"""

SUBCATCHMENT_TAG_PREFIX = "Subcatch"
"""Element type prefix swmmio writes for subcatchment rows of the ``[TAGS]`` section."""


class SubcatchmentFeatureError(ValueError):
    """Raised when subcatchment categories cannot be encoded."""


def assign_categories(dfs: pd.DataFrame, tags: pd.Series, *, category_column: str = CATEGORY_COLUMN) -> pd.DataFrame:
    """Return a copy of ``dfs`` with the land cover class of each subcatchment.

    Args:
        dfs: subcatchment frame indexed by subcatchment name.
        tags: the ``[TAGS]`` section as produced by
            :func:`stormlab.io.inp.read_tags` — a Series indexed by
            ``(ElementType, Name)``, or by name alone if the element type has
            already been filtered out.
        category_column: name of the column to write.

    Returns:
        ``dfs`` plus ``category``. Subcatchments without a tag get ``NaN``, and a
        model with no subcatchment tags at all (the Wroclaw network) gets an
        all-``NaN`` column and a warning rather than a missing column.
    """
    frame = dfs.copy()
    if len(frame) == 0:
        frame[category_column] = pd.Series(dtype="object")
        return frame

    category_map = _subcatchment_tag_map(tags)
    if not category_map:
        logger.warning(
            "No subcatchment tags found: %d subcatchments keep category=NaN; "
            "downstream one-hot encoding will be all zeros for them",
            len(frame),
        )
        frame[category_column] = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
        return frame

    frame[category_column] = frame.index.map(category_map)
    categorized_count = int(frame[category_column].notna().sum())
    logger.info("Assigned categories to %d subcatchments from the TAGS section", categorized_count)
    if categorized_count < len(frame):
        logger.warning("%d of %d subcatchments have no tag and keep category=NaN", len(frame) - categorized_count, len(frame))
    return frame


def one_hot_categories(
    dfs: pd.DataFrame,
    order: Sequence[str],
    *,
    category_column: str = CATEGORY_COLUMN,
) -> pd.DataFrame:
    """Return a copy of ``dfs`` with one indicator column per class in ``order``.

    Every class of ``order`` gets a column even when absent from the data, so the
    encoding has the same width for every network. ``order`` must be passed
    explicitly: the feature matrix and the catchment classifier use two different
    canonical orders and deriving one from the other mis-decodes the classes.

    Args:
        dfs: subcatchment frame carrying ``category_column``.
        order: the canonical class order to emit.
        category_column: name of the column to encode.

    Raises:
        SubcatchmentFeatureError: if the category column is missing, or the data
            contains a class that ``order`` does not cover — that class cannot be
            encoded, and emitting zeros for it would hide the mismatch.
    """
    if dfs is None or len(dfs) == 0:
        return dfs

    if category_column not in dfs.columns:
        raise SubcatchmentFeatureError(f"Column '{category_column}' not found in dataframe with columns {list(dfs.columns)}")

    encoded = encode_categories(dfs[category_column], order, what="subcatchment")
    return pd.concat([dfs.copy(), encoded], axis=1)


def encode_categories(categories: pd.Series, order: Sequence[str], *, what: str = "subcatchment") -> pd.DataFrame:
    """One-hot encode a category Series onto exactly the classes of ``order``.

    This is the single encoder behind both the subcatchment frame and the conduit
    feature block, so a class that one of them rejects cannot be accepted by the
    other.

    Args:
        categories: the land cover class of each element, indexed by element name.
        order: the canonical class order; the returned frame has exactly these
            columns, in this order.
        what: the kind of element being encoded, used in the error message.

    Returns:
        An ``int`` indicator frame with one column per class of ``order`` and the
        index of ``categories``. A missing class — ``NaN``/``None`` or
        :data:`NO_CATEGORY_SENTINEL` — is an all-zero row, which is what the
        historical ``pd.get_dummies`` produced for it.

    Raises:
        SubcatchmentFeatureError: if any value is neither a missing-class marker
            nor a member of ``order``. Zeroing an unrecognised class would make a
            typo indistinguishable from a documented absence of land cover.
    """
    known = set(order)
    unknown = [(str(name), value) for name, value in categories.items() if not _is_missing_category(value) and value not in known]
    if unknown:
        listed = ", ".join(f"{name}={value!r}" for name, value in unknown[:10])
        raise SubcatchmentFeatureError(
            f"{len(unknown)} {what}(s) carry a category that is not part of the requested order {list(order)}: {listed}"
        )

    return pd.DataFrame(
        {category: (categories == category).to_numpy(dtype=int) for category in order},
        index=categories.index,
    )


def _is_missing_category(value: object) -> bool:
    """Whether a category value means "no land cover information".

    The null test comes first on purpose: ``assign_categories`` writes ``pd.NA``
    for an untagged subcatchment, and ``pd.NA == "-"`` evaluates to ``pd.NA``,
    whose truth value raises rather than being ``False``.
    """
    if value is None or pd.isna(value):
        return True
    return value == NO_CATEGORY_SENTINEL


def _subcatchment_tag_map(tags: pd.Series) -> dict[str, str]:
    """Name-to-tag mapping for the subcatchment rows of a ``[TAGS]`` Series."""
    if tags is None or len(tags) == 0:
        return {}

    if isinstance(tags.index, pd.MultiIndex):
        element_types = tags.index.get_level_values(0).astype(str)
        subcatchment_tags = tags[element_types.str.startswith(SUBCATCHMENT_TAG_PREFIX)]
        names = subcatchment_tags.index.get_level_values(-1).astype(str)
    else:
        subcatchment_tags = tags
        names = subcatchment_tags.index.astype(str)

    return {str(name): str(tag) for name, tag in zip(names, subcatchment_tags.to_numpy())}


__all__ = [
    "CATEGORY_COLUMN",
    "NO_CATEGORY_SENTINEL",
    "SUBCATCHMENT_TAG_PREFIX",
    "SubcatchmentFeatureError",
    "assign_categories",
    "encode_categories",
    "one_hot_categories",
]
