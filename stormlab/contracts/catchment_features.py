"""Catchment classifier contract: 12 features and the class order they decode to.

Eight columns are taken verbatim from the INP/RPT frames; the last four are
runoff volumes that must be divided by ``Area`` (``Area`` itself is not a
feature). The division carries a ``sum() != 0`` guard: a column that is
identically zero across the frame stays zero instead of producing NaN/inf.

The guard is a column-level guard only: a zero ``Area`` on an individual row is
rejected outright rather than allowed to produce ``inf``.

``CLASSIFIER_CLASS_ORDER`` is alphabetical, the order of the classifier's output
units. It is written as an explicit literal — deriving it from the enum
declaration order would decode 11 of 12 classes wrongly.

The twelve columns are the contract; :class:`CatchmentFeatureSet` names a subset
of them a classifier can be built on. ``full_12`` is the default everywhere.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

CATCHMENT_FEATURE_COLUMNS: tuple[str, ...] = (
    "PercImperv",
    "PercSlope",
    "N-Imperv",
    "N-Perv",
    "S-Imperv",
    "S-Perv",
    "PctZero",
    "RunoffCoeff",
    "TotalInfil",
    "ImpervRunoff",
    "TotalRunoffMG",
    "PeakRunoff",
)

CATCHMENT_AREA_NORMALIZED: tuple[str, ...] = CATCHMENT_FEATURE_COLUMNS[-4:]
"""The four columns divided by ``Area``. ``RunoffCoeff`` is deliberately not one of them."""

AREA_COLUMN = "Area"

CLASSIFIER_CLASS_ORDER: tuple[str, ...] = (
    "arable",
    "forests",
    "marshes",
    "meadows",
    "mountains_rocky",
    "mountains_vegetated",
    "rural",
    "suburban_highly_impervious",
    "suburban_weakly_impervious",
    "urban_highly_impervious",
    "urban_moderately_impervious",
    "urban_weakly_impervious",
)

CATCHMENT_INPUT_SHAPE: tuple[int | None, int] = (None, len(CATCHMENT_FEATURE_COLUMNS))

CATCHMENT_FEATURE_DTYPE = "float64"


class CatchmentContractError(ValueError):
    """Raised when a frame does not satisfy the 12-feature catchment contract."""


@dataclass(frozen=True)
class CatchmentFeatureSet:
    """A named subset of the contract columns one classifier is built and scored on.

    The twelve columns stay the contract; a feature set is a *reading* of them. The
    name travels with the weights, because a model that takes seven inputs and a
    model that takes twelve are two different models under one architecture.
    """

    name: str
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        outside = [column for column in self.columns if column not in CATCHMENT_FEATURE_COLUMNS]
        if outside:
            raise CatchmentContractError(
                f"Feature set {self.name!r} names column(s) {outside} outside the catchment contract "
                f"{list(CATCHMENT_FEATURE_COLUMNS)}"
            )
        if not self.columns:
            raise CatchmentContractError(f"Feature set {self.name!r} is empty; a classifier needs at least one feature")
        if len(set(self.columns)) != len(self.columns):
            raise CatchmentContractError(f"Feature set {self.name!r} repeats a column: {list(self.columns)}")

    @property
    def n_features(self) -> int:
        """How many inputs a classifier built on this set takes."""
        return len(self.columns)


FULL_12 = CatchmentFeatureSet("full_12", CATCHMENT_FEATURE_COLUMNS)
"""Every contract column: the default set."""

INP_7 = CatchmentFeatureSet("inp_7", CATCHMENT_FEATURE_COLUMNS[:7])
"""The seven INP columns alone, without ``RunoffCoeff`` and the four report quantities."""

INP_2 = CatchmentFeatureSet("inp_2", ("PercImperv", "PercSlope"))
"""Imperviousness and slope alone."""

FEATURE_SETS: Mapping[str, CatchmentFeatureSet] = {feature_set.name: feature_set for feature_set in (FULL_12, INP_7, INP_2)}
"""Every feature set a configuration may name, by name."""


def feature_set_contract_sha(feature_set: CatchmentFeatureSet = FULL_12) -> str:
    """sha256 of the set's column names joined by newlines, in contract order.

    One recipe for every set, so the digest of :data:`FULL_12` is the plain digest
    of the twelve contract column names.
    """
    return hashlib.sha256("\n".join(feature_set.columns).encode("utf-8")).hexdigest()


def input_shape(feature_set: CatchmentFeatureSet = FULL_12) -> tuple[int | None, int]:
    """The Keras input shape a classifier on ``feature_set`` has."""
    return (None, feature_set.n_features)


def transform_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return the contract value of one catchment feature.

    The four area-normalised columns are divided elementwise by ``Area``; if the
    column sums to exactly zero over ``frame`` the original (all-zero) column is
    returned untouched. Every other column is passed through unchanged.

    Raises:
        CatchmentContractError: if ``column`` is not a contract column, or the
            required source column is absent from ``frame``.
    """
    if column not in CATCHMENT_FEATURE_COLUMNS:
        raise CatchmentContractError(f"{column!r} is not a catchment feature; expected one of {list(CATCHMENT_FEATURE_COLUMNS)}")
    if column not in frame.columns:
        raise CatchmentContractError(f"Source column {column!r} missing from frame with columns {list(frame.columns)}")

    values = _numeric_series(frame, column)
    if column not in CATCHMENT_AREA_NORMALIZED:
        return values

    if AREA_COLUMN not in frame.columns:
        raise CatchmentContractError(f"Column {AREA_COLUMN!r} is required to normalise {column!r} but is missing from the frame")
    if values.sum() == 0:
        return values

    area = _numeric_series(frame, AREA_COLUMN)
    _assert_positive_area(area)
    return values / area


def _assert_positive_area(area: pd.Series) -> None:
    """Reject a non-positive area before it turns a runoff volume into ``inf``.

    The ``sum() != 0`` guard is a statement about the *column*, not about a row:
    a single subcatchment of zero area inside an otherwise non-zero column
    divides by zero and produces ``inf``, which then travels into the classifier
    as a very confident nonsense feature.

    Raises:
        CatchmentContractError: naming the offending subcatchments.
    """
    invalid = area[~(area > 0)]
    if not invalid.empty:
        listed = ", ".join(f"{name}={value!r}" for name, value in list(invalid.items())[:10])
        raise CatchmentContractError(
            f"{len(invalid)} subcatchment(s) have a non-positive {AREA_COLUMN!r} and cannot be area-normalised: {listed}"
        )


def assert_classifier_input_shape(model: Any, feature_set: CatchmentFeatureSet = FULL_12) -> None:
    """Check that a loaded catchment classifier takes ``feature_set``.

    Takes the model object rather than importing Keras: this module sits in the
    contracts layer and must stay importable without TensorFlow. Only the
    ``input_shape`` attribute is touched.

    Raises:
        CatchmentContractError: if the attribute is missing or does not equal the
            shape of ``feature_set``.
    """
    expected = input_shape(feature_set)
    shape = getattr(model, "input_shape", None)
    if shape is None:
        raise CatchmentContractError(
            f"Model {type(model).__name__} exposes no 'input_shape'; the {feature_set.n_features}-feature "
            f"catchment contract {feature_set.name!r} cannot be verified against it"
        )
    if tuple(shape) != expected:
        raise CatchmentContractError(
            f"Catchment classifier input shape {tuple(shape)} does not match the contract {expected}; "
            f"the {feature_set.n_features} features of {feature_set.name!r} are {list(feature_set.columns)}"
        )


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """One column of ``frame`` as a float Series, failing on duplicated column names."""
    values = frame[column]
    if isinstance(values, pd.DataFrame):
        raise CatchmentContractError(f"Column {column!r} appears {values.shape[1]} times in the frame; expected exactly one")
    return values.astype(float)


def build_catchment_features(frame: pd.DataFrame, feature_set: CatchmentFeatureSet = FULL_12) -> pd.DataFrame:
    """Assemble the classifier input of ``feature_set`` from a subcatchment frame."""
    data = {column: transform_column(frame, column) for column in feature_set.columns}
    return pd.DataFrame(data, index=frame.index).astype(CATCHMENT_FEATURE_DTYPE)


def validate_catchment_features(df: pd.DataFrame, feature_set: CatchmentFeatureSet = FULL_12) -> None:
    """Check column presence, order, dtype and absence of missing values.

    Raises:
        CatchmentContractError: with the offending column names in the message.
    """
    if not isinstance(df, pd.DataFrame):
        raise CatchmentContractError(f"Expected a DataFrame, got {type(df).__name__}")

    required = list(feature_set.columns)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise CatchmentContractError(f"Missing catchment feature columns ({len(missing)}): {missing}")
    if list(df.columns) != required:
        raise CatchmentContractError(f"Catchment features out of contract order; expected {required}, got {list(df.columns)}")

    wrong_dtype = {column: str(df[column].dtype) for column in required if str(df[column].dtype) != CATCHMENT_FEATURE_DTYPE}
    if wrong_dtype:
        raise CatchmentContractError(f"Wrong dtypes: {wrong_dtype} (expected {CATCHMENT_FEATURE_DTYPE})")

    na_counts = {column: int(df[column].isna().sum()) for column in required if df[column].isna().any()}
    if na_counts:
        raise CatchmentContractError(f"Missing values in catchment features: {na_counts}")

    # An infinity is what a division by a zero area leaves behind, and it is not a
    # missing value: `isna` does not see it, and the classifier happily consumes it.
    non_finite = {column: int((~np.isfinite(df[column].to_numpy())).sum()) for column in required}
    offending = {column: count for column, count in non_finite.items() if count}
    if offending:
        raise CatchmentContractError(f"Non-finite values in catchment features: {offending}")
