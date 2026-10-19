"""Schema of the ``numbers.json`` registry — every figure that reaches the thesis.

One record per number. NaN is always a bug and is rejected; an infinite value
is a legitimate result (cross-entropy of a class the model can never predict),
so it is stored as the string ``"inf"``/``"-inf"`` together with an ``is_inf``
flag instead of being silently turned into null. Keys are unique within a
registry: recomputing a number under a different convention needs a different
key, never an overwrite.

One kind carries text rather than a figure: ``"note"``. It exists for the
statements the thesis has to make where no number can be computed — the
historical 7128-row dataset cannot be reproduced, and writing that as ``0`` or
``NaN`` would be a lie in the shape of a measurement. Text is accepted *only*
under that kind; a string under ``kind="metric"`` is still an error.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

KEY_SEPARATOR = "."
KEY_MIN_PARTS = 2

VALID_KINDS: frozenset[str] = frozenset({"metric", "count", "ratio", "flag", "duration", "parameter", "note"})

SCHEMA_VERSION = 2
"""What :meth:`NumberRecord.to_dict` writes today.

Version 2 added ``variant``: the frozen result set of path B a figure belongs to.
Version 1 records — every registry written before variants existed, the frozen
golden among them — carry neither field, and are read as ``variant=None`` rather
than rejected. A reference file that stops being readable the moment the schema
moves is a reference that cannot be compared against, which is the one thing it
exists for."""

SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1, 2})
SCHEMA_VERSION_FIELD = "schema_version"

INFINITY_TEXT = "inf"
NEGATIVE_INFINITY_TEXT = "-inf"
"""How an infinite value is written down wherever this package emits JSON.

``json.dumps`` writes a bare ``Infinity`` by default, which is not JSON and which
strict readers reject; ``allow_nan=False`` turns that into an error instead. Both
are worse than saying the word, so every layer that serialises a possibly
infinite figure uses these two tokens."""

NOTE_KIND = "note"
"""The one kind whose value is text; every other kind carries a real number."""
VALID_PROB_ROUNDING: frozenset[str] = frozenset({"raw", "3dp", "2dp"})


class NumberValidationError(ValueError):
    """Raised when a number record violates the registry schema."""


class DuplicateNumberKeyError(NumberValidationError):
    """Raised when a key is added to a registry that already holds it."""


@dataclass(frozen=True)
class NumberRecord:
    """A single reported figure with the provenance needed to reproduce it.

    Attributes:
        key: ``<study>.<model_id>.<population>.<metric>`` — unique per registry.
        value: the figure itself; may be infinite, never NaN. Under
            ``kind="note"`` it is a non-empty string instead.
        unit: ``"1"`` for dimensionless ratios, otherwise the physical unit.
        kind: one of :data:`VALID_KINDS`.
        producer: dotted name of the module that computed the value.
        run_id: identifier of the pipeline run.
        git_commit: commit the run was executed from.
        tolerance: absolute tolerance for golden comparison, if any.
        population: human-readable description of the sample the value covers.
        model_id: registry id of the model that produced the value, if any.
        class_order: name of the class order the value was computed under.
        thesis_ref: table or figure number in the dissertation, if any.
        input_sha256: hash of the primary input artifact, if any.
        prob_rounding: rounding convention of the probabilities behind the value.
        variant: the frozen result set of path B the figure belongs to, if the run
            named one. Deliberately **not** part of the golden comparison or of
            :data:`stormlab.reports.numbers.IDENTITY_FIELDS`: B1 and B2 disagreeing
            on one key is the content of the variant table, not a merge conflict,
            and a golden frozen before variants existed has to keep comparing.
    """

    key: str
    value: float | str
    unit: str
    kind: str
    producer: str
    run_id: str
    git_commit: str
    tolerance: float | None = None
    population: str | None = None
    model_id: str | None = None
    class_order: str | None = None
    thesis_ref: str | None = None
    input_sha256: str | None = None
    prob_rounding: str | None = None
    variant: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.key or len(self.key.split(KEY_SEPARATOR)) < KEY_MIN_PARTS:
            raise NumberValidationError(f"Key {self.key!r} must contain at least {KEY_MIN_PARTS} dot-separated parts")
        for name in ("unit", "producer", "run_id", "git_commit"):
            if not getattr(self, name):
                raise NumberValidationError(f"Field {name!r} must be a non-empty string (key {self.key!r})")
        if self.kind not in VALID_KINDS:
            raise NumberValidationError(f"Unknown kind {self.kind!r} for key {self.key!r}; allowed: {sorted(VALID_KINDS)}")
        if self.prob_rounding is not None and self.prob_rounding not in VALID_PROB_ROUNDING:
            raise NumberValidationError(
                f"Unknown prob_rounding {self.prob_rounding!r} for key {self.key!r}; allowed: {sorted(VALID_PROB_ROUNDING)}"
            )
        if self.variant is not None and (not isinstance(self.variant, str) or not self.variant.strip()):
            raise NumberValidationError(f"Variant of {self.key!r} must be a non-empty string or absent, got {self.variant!r}")
        self._validate_value()
        self._validate_tolerance()

    def _validate_value(self) -> None:
        """Notes and named parameters may hold text; measurements hold numbers."""
        if self.kind == NOTE_KIND or (self.kind == "parameter" and isinstance(self.value, str)):
            if not isinstance(self.value, str) or not self.value.strip():
                raise NumberValidationError(f"Value of {self.key!r} must be a non-empty string, got {self.value!r}")
            return
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise NumberValidationError(
                f"Value of {self.key!r} must be a real number, got {type(self.value).__name__}; "
                f"text is only reportable under kind={NOTE_KIND!r}"
            )
        if math.isnan(self.value):
            raise NumberValidationError(f"Value of {self.key!r} is NaN; NaN is never a reportable result")

    def _validate_tolerance(self) -> None:
        """A tolerance is the width of an interval, so it is a finite, non-negative number.

        ``True`` would pass an ``isinstance(..., int)`` check and silently mean a
        tolerance of 1.0, so booleans are rejected explicitly; a negative tolerance
        would make every golden comparison fail with an unreadable message.
        """
        if self.tolerance is None:
            return
        if isinstance(self.tolerance, bool) or not isinstance(self.tolerance, (int, float)):
            raise NumberValidationError(f"Tolerance of {self.key!r} must be a real number, got {type(self.tolerance).__name__}")
        if math.isnan(self.tolerance) or math.isinf(self.tolerance):
            raise NumberValidationError(f"Tolerance of {self.key!r} must be finite, got {self.tolerance}")
        if self.tolerance < 0:
            raise NumberValidationError(f"Tolerance of {self.key!r} must be non-negative, got {self.tolerance}")

    @property
    def is_inf(self) -> bool:
        """Whether the value is positive or negative infinity; a note never is."""
        return isinstance(self.value, (int, float)) and not isinstance(self.value, bool) and math.isinf(self.value)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe mapping; infinities become ``"inf"``/``"-inf"``.

        The payload carries :data:`SCHEMA_VERSION` so a reader never has to infer
        the shape from which optional fields happen to be present.
        """
        payload: dict[str, Any] = asdict(self)
        payload[SCHEMA_VERSION_FIELD] = SCHEMA_VERSION
        payload["is_inf"] = self.is_inf
        if self.is_inf:
            payload["value"] = INFINITY_TEXT if float(self.value) > 0 else NEGATIVE_INFINITY_TEXT
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NumberRecord":
        """Rebuild a record from :meth:`to_dict` output.

        Version 1 payloads — written before variants existed — name neither
        ``schema_version`` nor ``variant``; they are read as version 1 with
        ``variant=None``.

        Raises:
            NumberValidationError: if ``is_inf`` disagrees with the stored value,
                the value cannot be read as a number, or the payload declares a
                schema version this package does not know how to read.
        """
        data = dict(payload)
        version = data.pop(SCHEMA_VERSION_FIELD, 1)
        # Checked as a type before it is checked as a value: ``True`` and ``2.0``
        # both compare equal to a supported version, so a payload carrying either
        # would be read as valid while saying something the schema never allows.
        if isinstance(version, bool) or not isinstance(version, int):
            raise NumberValidationError(
                f"Record {data.get('key')!r} declares {SCHEMA_VERSION_FIELD}={version!r} "
                f"({type(version).__name__}); it must be an integer"
            )
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise NumberValidationError(
                f"Record {data.get('key')!r} declares {SCHEMA_VERSION_FIELD}={version!r}; this package reads "
                f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        declared_inf = bool(data.pop("is_inf", False))
        raw_value = data.get("value")
        # Notes and named parameters carry text by design, so the
        # "inf"/"-inf" branch below must never see them.
        if data.get("kind") == NOTE_KIND or (data.get("kind") == "parameter" and isinstance(raw_value, str) and not declared_inf):
            pass
        elif declared_inf:
            # The serialised form stores inf as the string "inf" / "-inf"
            # alongside is_inf=true; restore the float.
            if isinstance(raw_value, str) and raw_value in {INFINITY_TEXT, NEGATIVE_INFINITY_TEXT}:
                data["value"] = math.inf if raw_value == INFINITY_TEXT else -math.inf
        elif isinstance(raw_value, str):
            raise NumberValidationError(
                f"Value {raw_value!r} of key {data.get('key')!r} is a string but kind is "
                f"{data.get('kind')!r} and is_inf was not set"
            )
        record = cls(**data)
        if record.is_inf != declared_inf:
            raise NumberValidationError(f"is_inf={declared_inf} disagrees with value {record.value} for key {record.key!r}")
        return record


@dataclass(frozen=True)
class RecordFactory:
    """Builds :class:`NumberRecord` objects that already carry a run's provenance.

    Four fields repeat on every number a run produces — which run, which commit,
    which module computed it, and the digest of the bytes it was computed from —
    and every experiment used to spell them out at each call site. The one that
    kept being forgotten was ``input_sha256``: without it a figure names no
    input, so it can never be recomputed and never be shown to describe the data
    it claims to. Making it a constructor argument means a study cannot register
    a number before it knows what it read.

    :meth:`prediction` exists separately because a figure derived from model
    probabilities needs three more facts to be interpretable — which model, in
    which class order, at which rounding. ``n_correct`` under a 2-decimal
    rounding and ``n_correct`` under raw probabilities are different numbers, and
    a record that omits the convention cannot say which one it holds.
    """

    run_id: str
    git_commit: str
    input_sha256: str
    producer: str
    variant: str | None = None
    """The frozen result set this run belongs to, stamped onto every record it makes."""

    def __post_init__(self) -> None:
        for name in ("run_id", "git_commit", "input_sha256", "producer"):
            if not getattr(self, name):
                raise NumberValidationError(f"RecordFactory needs a non-empty {name!r}")

    def make(self, key: str, value: float | str, *, kind: str, unit: str = "1", **fields: Any) -> NumberRecord:
        """One record, with this run's provenance already filled in.

        Raises:
            NumberValidationError: if a caller tries to override the provenance
                fields the factory exists to guarantee.
        """
        clash = sorted(set(fields) & {"run_id", "git_commit", "input_sha256", "producer", "variant"})
        if clash:
            raise NumberValidationError(f"Field(s) {clash} are fixed by the factory and cannot be overridden for key {key!r}")
        return NumberRecord(
            key=key,
            value=value,
            unit=unit,
            kind=kind,
            producer=self.producer,
            run_id=self.run_id,
            git_commit=self.git_commit,
            input_sha256=self.input_sha256,
            variant=self.variant,
            **fields,
        )

    def prediction(
        self,
        key: str,
        value: float | str,
        *,
        kind: str,
        model_id: str,
        class_order: str,
        prob_rounding: str,
        unit: str = "1",
        **fields: Any,
    ) -> NumberRecord:
        """One record derived from model predictions, with its convention attached."""
        return self.make(
            key,
            value,
            kind=kind,
            unit=unit,
            model_id=model_id,
            class_order=class_order,
            prob_rounding=prob_rounding,
            **fields,
        )


@dataclass
class NumbersRegistry:
    """An append-only, key-unique collection of :class:`NumberRecord`."""

    records: dict[str, NumberRecord] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, key: object) -> bool:
        return key in self.records

    def __iter__(self) -> Iterator[NumberRecord]:
        return iter(self.records.values())

    def add(self, record: NumberRecord) -> None:
        """Register one number.

        Raises:
            DuplicateNumberKeyError: if the key is already present.
        """
        if record.key in self.records:
            raise DuplicateNumberKeyError(
                f"Key {record.key!r} already registered with value {self.records[record.key].value!r}; "
                "reporting a number under a new convention requires a new key"
            )
        self.records[record.key] = record

    def extend(self, records: Iterable[NumberRecord]) -> None:
        """Register several numbers, failing on the first duplicate key."""
        for record in records:
            self.add(record)

    def get(self, key: str) -> NumberRecord:
        """Return one record.

        Raises:
            KeyError: if the key was never registered.
        """
        if key not in self.records:
            raise KeyError(f"Key {key!r} not in registry; known keys: {sorted(self.records)}")
        return self.records[key]

    def to_json(self, path: Path | str | None = None, *, indent: int = 2) -> str:
        """Serialise the registry, sorted by key, optionally writing it to ``path``."""
        payload = [record.to_dict() for record in sorted(self.records.values(), key=lambda item: item.key)]
        text = json.dumps(payload, indent=indent, ensure_ascii=False, allow_nan=False) + "\n"
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, path: Path | str) -> "NumbersRegistry":
        """Load a registry previously written by :meth:`to_json`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise NumberValidationError(f"Expected a JSON list of records, got {type(payload).__name__}")
        registry = cls()
        registry.extend(NumberRecord.from_dict(item) for item in payload)
        return registry
