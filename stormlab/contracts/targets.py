"""Versioned target contracts: what a row of a soft label matrix is allowed to be.

The historical seam does not check anything. ``labels/rules.py::prepare_swmm_labels``
lets a ``DataFrame`` through with ``labels.values.astype(np.float32)`` and no
statement about its columns, their order or their sum — which is exactly what a
soft target needs a statement about, because a distribution written in the wrong
column order is still a valid float matrix and still trains (plan16 §2 P4,
"obecny szew w ``rules.py`` przepuszcza dowolną macierz bez walidacji").

A target contract is the other half of :mod:`stormlab.contracts.features`: the
feature contract says what goes in, this says what comes out. It carries the class
names in the order the columns stand in, a version, and a sha256 over both, so a
run that trained on ``primary_action_v3`` can be told apart from one that trained
on a nine-column matrix in some other order.

Supervision is part of the contract rather than a convention around it. plan16 §2
P4 keeps cases "których nie da się uczciwie sprowadzić do jednej rodziny
działania" without a target at all, and forbids giving them ``valid`` or a uniform
row; the honest wire form of "no target" is a row of ``NaN`` under a zero mask, and
:meth:`TargetContract.validate` is where that is enforced instead of hoped for.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from stormlab.contracts.enums import RECOMMENDATION_CLASS_ORDER

ROW_SUM_TOLERANCE = 1e-6
"""How far a supervised row may stand from summing to one.

Also how far a single share may stand outside ``[0, 1]``: the weights inside a group
are shares of one, so a class every source of the only active group supported is
``1 + eps`` rather than ``1``. Loose enough for the float64 accumulation of nine
averaged shares, tight enough that a row which lost a class to an exclusion after
normalisation is caught."""


class TargetContractError(ValueError):
    """Raised when a target matrix does not match the contract it is validated against."""


@dataclass(frozen=True)
class TargetContract:
    """One named, versioned layout of a target matrix.

    Attributes:
        name: the contract's name, as a run records it.
        classes: the class names in column order.
        version: the schema version of the layout; bumped when the meaning of a
            column changes, not when a run changes.
    """

    name: str
    classes: tuple[str, ...]
    version: int

    def __post_init__(self) -> None:
        if not self.classes:
            raise TargetContractError(f"Target contract {self.name!r} declares no classes")
        duplicated = sorted({name for name in self.classes if list(self.classes).count(name) > 1})
        if duplicated:
            raise TargetContractError(f"Target contract {self.name!r} names class(es) {duplicated} twice")

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def sha(self) -> str:
        """sha256 over the class names and the version, and over nothing else.

        Deliberately the same shape of promise the feature contracts make: the
        digest changes when a column moves, is renamed or is added, and does not
        change when a docstring or a helper around it does.
        """
        canonical = json.dumps(
            {"name": self.name, "version": self.version, "classes": list(self.classes)},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def assert_columns(self, columns: Sequence[str]) -> None:
        """Fail unless ``columns`` are exactly this contract's classes, in order.

        Order is checked by name and not by count, because a nine-column matrix in
        the wrong order is the failure this contract exists to catch.

        Raises:
            TargetContractError: naming the first position that disagrees.
        """
        actual = list(columns)
        if actual == list(self.classes):
            return
        if len(actual) != self.n_classes:
            raise TargetContractError(
                f"Target contract {self.name!r} has {self.n_classes} column(s) {list(self.classes)}, got {len(actual)}: {actual}"
            )
        position = next(index for index, (left, right) in enumerate(zip(actual, self.classes)) if left != right)
        raise TargetContractError(
            f"Target contract {self.name!r} expects column {position} to be {self.classes[position]!r}, "
            f"got {actual[position]!r}; the full order is {list(self.classes)}"
        )

    def validate(self, matrix: np.ndarray, mask: np.ndarray) -> None:
        """Fail unless the matrix is a valid target under this contract.

        Args:
            matrix: ``(n, n_classes)`` of float; supervised rows hold a
                distribution, unsupervised rows hold ``NaN``.
            mask: ``(n,)`` of 0/1; 1 marks a supervised row.

        Raises:
            TargetContractError: on a wrong shape, a supervised row that is not a
                distribution (out of ``[0, 1]``, holding ``NaN`` or not summing to
                one — both within :data:`ROW_SUM_TOLERANCE`, because a single class
                that collected the whole of a weighted average sums to one only up to
                float accumulation), or an unsupervised row that
                is anything other than all ``NaN``. Every message names the first
                offending row index, because a report that says only "a row is
                wrong" over 33 000 rows says nothing.
        """
        values = np.asarray(matrix, dtype="float64")
        flags = np.asarray(mask)
        if values.ndim != 2 or values.shape[1] != self.n_classes:
            raise TargetContractError(
                f"Target contract {self.name!r} expects a matrix of shape (n, {self.n_classes}), got {values.shape}"
            )
        if flags.ndim != 1 or flags.shape[0] != values.shape[0]:
            raise TargetContractError(
                f"Supervision mask of shape {flags.shape} does not address the {values.shape[0]} row(s) of the target matrix"
            )
        unexpected = np.setdiff1d(np.unique(flags), np.array([0, 1]))
        if unexpected.size:
            raise TargetContractError(f"Supervision mask holds value(s) {unexpected.tolist()}; it is 0 or 1")

        supervised = flags.astype(bool)
        self._validate_supervised(values, supervised)
        self._validate_unsupervised(values, supervised)

    def _validate_supervised(self, values: np.ndarray, supervised: np.ndarray) -> None:
        rows = values[supervised]
        if not rows.size:
            return
        positions = np.flatnonzero(supervised)

        missing = np.isnan(rows).any(axis=1)
        if missing.any():
            index = int(positions[np.flatnonzero(missing)[0]])
            raise TargetContractError(
                f"Row {index} is supervised but holds a missing value: {values[index].tolist()}. "
                "A supervised row is a distribution; absence of a target is written as mask 0 and an all-NaN row"
            )
        out_of_range = ((rows < -ROW_SUM_TOLERANCE) | (rows > 1.0 + ROW_SUM_TOLERANCE)).any(axis=1)
        if out_of_range.any():
            index = int(positions[np.flatnonzero(out_of_range)[0]])
            raise TargetContractError(f"Row {index} holds share(s) outside [0, 1]: {values[index].tolist()}")
        sums = rows.sum(axis=1)
        off = np.abs(sums - 1.0) > ROW_SUM_TOLERANCE
        if off.any():
            offset = int(np.flatnonzero(off)[0])
            index = int(positions[offset])
            raise TargetContractError(
                f"Row {index} sums to {sums[offset]!r}, not 1 within {ROW_SUM_TOLERANCE}: {values[index].tolist()}"
            )

    def _validate_unsupervised(self, values: np.ndarray, supervised: np.ndarray) -> None:
        rows = values[~supervised]
        if not rows.size:
            return
        positions = np.flatnonzero(~supervised)
        stated = ~np.isnan(rows).all(axis=1)
        if stated.any():
            index = int(positions[np.flatnonzero(stated)[0]])
            raise TargetContractError(
                f"Row {index} carries mask 0 but states {values[index].tolist()}. An unsupervised case has no target: "
                "not `valid`, not a uniform distribution, not a partly filled row (plan16 §2 P4)"
            )

    def validate_frame(self, frame: pd.DataFrame, mask: np.ndarray) -> None:
        """Check the column names of a frame and then its values.

        The two halves are separate methods because they fail for different
        reasons — a wrong order is a wiring defect, a wrong sum is an aggregation
        defect — and a caller holding a bare matrix can only run the second.
        """
        self.assert_columns(list(frame.columns))
        self.validate(frame.to_numpy(dtype="float64"), mask)


PRIMARY_ACTION_V3 = TargetContract(name="primary_action_v3", classes=RECOMMENDATION_CLASS_ORDER, version=3)
"""The soft target of series C1: nine intervention classes in the enum's own order.

The order is :data:`stormlab.contracts.enums.RECOMMENDATION_CLASS_ORDER` and not a
new literal, because the nine names are kept (plan16 §2 P4, "zachować dziewięć
nazw jako wersjonowany kontrakt") and a second copy of an order is a second thing
to keep in step. Version 3 names the third *meaning* the nine columns have had:
v1 the historical one-hot of the cascade, v2 the same under the corrected
hydraulics of B3/B4, v3 a distribution over the actions several sources support.

``valid`` here means "no source supports an intervention of its own on this
conduit" and not "this network is correct" (plan16 §2 P4)."""

TARGET_CONTRACTS: dict[str, TargetContract] = {PRIMARY_ACTION_V3.name: PRIMARY_ACTION_V3}
"""Every target contract by name, for a run that records which one it used."""


def target_contract(name: str) -> TargetContract:
    """Resolve a target contract by name.

    Raises:
        TargetContractError: on an unknown name, listing the known ones.
    """
    try:
        return TARGET_CONTRACTS[name]
    except KeyError as error:
        raise TargetContractError(f"Unknown target contract {name!r}; known: {sorted(TARGET_CONTRACTS)}") from error


__all__ = [
    "PRIMARY_ACTION_V3",
    "ROW_SUM_TOLERANCE",
    "TARGET_CONTRACTS",
    "TargetContract",
    "TargetContractError",
    "target_contract",
]
