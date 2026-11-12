"""Specification of a graph recommendation model, read from its ``.spec.json`` sidecar.

A weight file stores numbers, not an architecture. The sidecar next to it names the
architecture those numbers belong to, and :func:`read_spec` turns it into a frozen,
validated object that :func:`stormlab.models.gnn.build_graph_model_for_weights`
builds the model from.

Two sidecar formats exist. :class:`GnnContractSpec` is the format of the shipped model:
the architecture plus the feature and target contracts it was trained under.
:class:`GnnSpec` is the older architecture-only format; it is recognised so that
such a sidecar is refused with a clear message instead of being misread.

Training-only fields (schedule, target kind, supervision mask, context penalty)
stay in both classes: they are part of what the sidecar records, unknown or missing
fields are refused, and :attr:`GnnContractSpec.model_config_sha` is computed over all of
them. They have no effect on inference.

Nothing here imports TensorFlow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from stormlab.config import (
    SPEC_GRAPH_DIRECTION_BOTH,
    SPEC_GRAPH_DIRECTION_UPSTREAM,
    SPEC_GRAPH_DIRECTIONS,
    GNN_INPUT_NORM_CONSTANT_SAFE,
    GNN_INPUT_NORM_NONE,
    GNN_INPUT_NORMS,
    GNN_MONITOR_VAL_LOSS,
    GNN_MONITORS,
    GNN_NORMALIZATION_BATCH,
    GNN_NORMALIZATION_LAYER,
    GNN_NORMALIZATION_NONE,
    GNN_NORMALIZATIONS,
    GNN_OPERATOR_KHOP_SUM,
    GNN_OPERATOR_KHOP_SYM,
    GNN_OPERATOR_LEGACY_WEIGHTED_MEAN,
    GNN_OPERATOR_NOGRAPH,
    GNN_OPERATOR_SUPPORT_MAX,
    GNN_OPERATOR_SUPPORT_POOL,
    GNN_OPERATORS,
    GRAPH_DIRECTIONS,
    MAX_GNN_HOPS,
)

MAX_HOPS = MAX_GNN_HOPS
"""Largest hop count a spec may name."""

OPERATOR_LEGACY_WEIGHTED_MEAN = GNN_OPERATOR_LEGACY_WEIGHTED_MEAN
OPERATOR_KHOP_SUM = GNN_OPERATOR_KHOP_SUM
OPERATOR_KHOP_SYM = GNN_OPERATOR_KHOP_SYM
OPERATOR_SUPPORT_MAX = GNN_OPERATOR_SUPPORT_MAX
OPERATOR_SUPPORT_POOL = GNN_OPERATOR_SUPPORT_POOL
OPERATOR_NOGRAPH = GNN_OPERATOR_NOGRAPH

MATMUL_OPERATORS: frozenset[str] = frozenset({OPERATOR_LEGACY_WEIGHTED_MEAN, OPERATOR_KHOP_SUM, OPERATOR_KHOP_SYM})
"""Operators whose neighbour term is ``(A·X)·W_neigh``; the others pool over a mask."""

SUPPORT_OPERATORS: frozenset[str] = frozenset({OPERATOR_SUPPORT_MAX, OPERATOR_SUPPORT_POOL})
"""Operators that read only the support of the adjacency, never its values."""

GRAPH_FREE_OPERATORS: frozenset[str] = frozenset({OPERATOR_NOGRAPH})
"""Operators with no neighbour term; the layer never reads the adjacency."""

SERVED_FEATURE_CONTRACTS: tuple[str, ...] = ("v4_physical",)
"""Feature contracts this package serves; :func:`read_spec` refuses any other."""

_DIRECTION_TOKENS: dict[str, str] = {"downstream": "down", "upstream": "up", SPEC_GRAPH_DIRECTION_BOTH: "both"}
_LAYER_NORM_TOKENS: dict[str, str] = {
    GNN_NORMALIZATION_BATCH: "bn",
    GNN_NORMALIZATION_LAYER: "ln",
    GNN_NORMALIZATION_NONE: "nonorm",
}


class GnnSpecError(ValueError):
    """Raised when a sidecar does not describe a graph model this package can build."""


def _assert_choice(name: str, value: Any, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise GnnSpecError(f"{name} must be one of {list(allowed)}, got {value!r}")


def _assert_int(name: str, value: Any, *, low: int, high: int | None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GnnSpecError(f"{name} must be an integer, got {value!r}")
    if value < low or (high is not None and value > high):
        bound = f"[{low}, {high}]" if high is not None else f">= {low}"
        raise GnnSpecError(f"{name} must be {bound}, got {value}")


def _assert_rate(name: str, value: Any) -> None:
    """A dropout rate: a number in ``[0, 1)``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value < 1.0:
        raise GnnSpecError(f"{name} must be a number in [0, 1), got {value!r}")


def _assert_non_negative(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0.0:
        raise GnnSpecError(f"{name} must be a non-negative number, got {value!r}")


def _assert_positive(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0.0:
        raise GnnSpecError(f"{name} must be a positive number, got {value!r}")


def _number(value: float) -> str:
    """A float with no trailing zeros: ``0.2``, ``0``, ``0.0001``."""
    return f"{value:g}"


@dataclass(frozen=True)
class GnnSpec:
    """Architecture-only sidecar format.

    Attributes:
        operator: one of :data:`stormlab.config.GNN_OPERATORS`; fixes the adjacency
            preprocessing and the layer's aggregation.
        hops: hop count of the adjacency expansion, ``1..``:data:`MAX_HOPS`.
        direction: ``downstream``, or ``upstream`` (the stored adjacency transposed).
        hidden: units of both graph steps.
        dropout: rate of both dropout layers (inactive at inference).
        weight_decay: training regularisation; no effect at inference.
        layer_norm: normaliser after each graph step.
        input_norm: whether a ``Normalization`` layer precedes the first step.
        epochs, lr, patience, monitor: training schedule; no effect at inference.
    """

    operator: str = OPERATOR_LEGACY_WEIGHTED_MEAN
    hops: int = 4
    direction: str = GRAPH_DIRECTIONS[0]
    hidden: int = 64
    dropout: float = 0.2
    weight_decay: float = 0.0
    layer_norm: str = GNN_NORMALIZATION_BATCH
    input_norm: str = GNN_INPUT_NORM_NONE
    epochs: int = 150
    lr: float = 1e-3
    patience: int = 10
    monitor: str = GNN_MONITOR_VAL_LOSS

    def __post_init__(self) -> None:
        _assert_choice("operator", self.operator, GNN_OPERATORS)
        _assert_choice("direction", self.direction, GRAPH_DIRECTIONS)
        _assert_choice("layer_norm", self.layer_norm, GNN_NORMALIZATIONS)
        _assert_choice("input_norm", self.input_norm, GNN_INPUT_NORMS)
        _assert_choice("monitor", self.monitor, GNN_MONITORS)

        _assert_int("hops", self.hops, low=1, high=MAX_HOPS)
        _assert_int("hidden", self.hidden, low=1, high=None)
        _assert_int("epochs", self.epochs, low=1, high=None)
        _assert_int("patience", self.patience, low=1, high=None)

        _assert_rate("dropout", self.dropout)
        _assert_non_negative("weight_decay", self.weight_decay)
        _assert_positive("lr", self.lr)

        # Stored as floats, so `dropout=0` and `dropout=0.0` are one spec with one digest.
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "weight_decay", float(self.weight_decay))
        object.__setattr__(self, "lr", float(self.lr))

    @property
    def two_channel(self) -> bool:
        """Always ``False``: this format has no two-channel direction."""
        return False

    @property
    def transpose(self) -> bool:
        """Whether the stored adjacency is transposed before preprocessing."""
        return self.direction != GRAPH_DIRECTIONS[0]

    def as_dict(self) -> dict[str, Any]:
        """The fields as a plain mapping."""
        return asdict(self)

    def canonical_json(self) -> str:
        """The exact text :attr:`model_config_sha` is the digest of."""
        return json.dumps(self.as_dict(), sort_keys=True)

    @property
    def model_config_sha(self) -> str:
        """sha256 of the canonical JSON of every field."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def slug(self) -> str:
        """A short readable name for logs; not an identity (see :attr:`model_config_sha`)."""
        return "_".join(
            (
                self.operator,
                f"k{self.hops}",
                _DIRECTION_TOKENS[self.direction],
                f"h{self.hidden}",
                f"d{_number(self.dropout)}",
                f"wd{_number(self.weight_decay)}",
                _LAYER_NORM_TOKENS[self.layer_norm],
                f"in-{self.input_norm}",
            )
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GnnSpec":
        """Rebuild a spec from its mapping, refusing unknown or missing fields.

        Raises:
            GnnSpecError: on unknown or missing fields, or an invalid value.
        """
        if not isinstance(payload, Mapping):
            raise GnnSpecError(f"A spec must be a mapping, got {type(payload).__name__}")
        known = set(FIELD_NAMES)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise GnnSpecError(f"Unknown spec field(s) {unknown}; this package knows {sorted(known)}")
        missing = sorted(known - set(payload))
        if missing:
            raise GnnSpecError(f"Spec is missing field(s) {missing}")
        return cls(**{name: payload[name] for name in FIELD_NAMES})


FIELD_NAMES: tuple[str, ...] = tuple(GnnSpec().as_dict())
"""Field names of :class:`GnnSpec`, in declaration order."""


TARGET_KIND_SOFT = "soft"
TARGET_KIND_HARD = "hard"
TARGET_KINDS: tuple[str, ...] = (TARGET_KIND_SOFT, TARGET_KIND_HARD)
"""Whether the model was trained on class distributions or on one-hot rows."""


@dataclass(frozen=True)
class GnnContractSpec:
    """Sidecar format of the shipped model: architecture plus the contracts it was trained under.

    Architecture fields are those of :class:`GnnSpec`, except that ``direction`` may
    also be ``both`` (two adjacency channels with separate weights).

    Attributes:
        feature_contract: name of the feature contract the model reads.
        target_contract: name of the target contract the classes come from.
        target_kind: one of :data:`TARGET_KINDS`; no effect at inference.
        supervision_mask: training detail; no effect at inference.
        context_lambda: coefficient of a training-time context penalty. This
            package serves only models trained without it (``0.0``).
        context_motifs: identifiers the penalty was built from. This package
            serves only models with none, because it carries no catalogue to
            validate identifiers against.
    """

    operator: str = OPERATOR_LEGACY_WEIGHTED_MEAN
    hops: int = 1
    direction: str = SPEC_GRAPH_DIRECTION_BOTH
    hidden: int = 64
    dropout: float = 0.2
    weight_decay: float = 0.0
    layer_norm: str = GNN_NORMALIZATION_LAYER
    input_norm: str = GNN_INPUT_NORM_CONSTANT_SAFE
    epochs: int = 150
    lr: float = 1e-3
    patience: int = 15
    monitor: str = GNN_MONITOR_VAL_LOSS
    feature_contract: str = SERVED_FEATURE_CONTRACTS[0]
    target_contract: str = "primary_action_v3"
    target_kind: str = TARGET_KIND_SOFT
    supervision_mask: bool = True
    context_lambda: float = 0.0
    context_motifs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _assert_choice("operator", self.operator, GNN_OPERATORS)
        _assert_choice("direction", self.direction, SPEC_GRAPH_DIRECTIONS)
        _assert_choice("layer_norm", self.layer_norm, GNN_NORMALIZATIONS)
        _assert_choice("input_norm", self.input_norm, GNN_INPUT_NORMS)
        _assert_choice("monitor", self.monitor, GNN_MONITORS)
        _assert_choice("target_kind", self.target_kind, TARGET_KINDS)

        _assert_int("hops", self.hops, low=1, high=MAX_HOPS)
        _assert_int("hidden", self.hidden, low=1, high=None)
        _assert_int("epochs", self.epochs, low=1, high=None)
        _assert_int("patience", self.patience, low=1, high=None)

        _assert_rate("dropout", self.dropout)
        _assert_non_negative("weight_decay", self.weight_decay)
        _assert_non_negative("context_lambda", self.context_lambda)
        _assert_positive("lr", self.lr)

        for name in ("feature_contract", "target_contract"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise GnnSpecError(f"{name} must be a non-empty string, got {value!r}")
        if not isinstance(self.supervision_mask, bool):
            raise GnnSpecError(f"supervision_mask must be a bool, got {self.supervision_mask!r}")

        # Two channels mean two matrix products; pooling and graph-free operators have no such term.
        if self.direction == SPEC_GRAPH_DIRECTION_BOTH and self.operator not in MATMUL_OPERATORS:
            raise GnnSpecError(
                f"direction={SPEC_GRAPH_DIRECTION_BOTH!r} needs an operator whose neighbour term is a matrix product, "
                f"got {self.operator!r}; expected one of {sorted(MATMUL_OPERATORS)}"
            )

        motifs = tuple(self.context_motifs)
        if motifs:
            raise GnnSpecError(
                f"context_motifs names {list(motifs)}; this package serves only models trained without a context "
                "penalty and has no catalogue to validate these identifiers against"
            )
        if self.context_lambda != 0.0:
            raise GnnSpecError(
                f"context_lambda is {self.context_lambda}; this package serves only models trained without a "
                "context penalty (context_lambda = 0)"
            )

        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "weight_decay", float(self.weight_decay))
        object.__setattr__(self, "lr", float(self.lr))
        object.__setattr__(self, "context_lambda", float(self.context_lambda))
        object.__setattr__(self, "context_motifs", motifs)

    @property
    def two_channel(self) -> bool:
        """Whether the graph steps aggregate two neighbourhoods with two weight sets."""
        return self.direction == SPEC_GRAPH_DIRECTION_BOTH

    @property
    def transpose(self) -> bool:
        """Whether the stored adjacency is transposed before preprocessing.

        Raises:
            GnnSpecError: for ``both``, which reads the matrix and its transpose as a pair.
        """
        if self.direction == SPEC_GRAPH_DIRECTION_BOTH:
            raise GnnSpecError(
                "transpose is undefined for direction='both': the two channels are the stored matrix and its "
                "transpose, so a caller must read them as a pair"
            )
        return self.direction == SPEC_GRAPH_DIRECTION_UPSTREAM

    def as_dict(self) -> dict[str, Any]:
        """The fields as a plain mapping, with ``context_motifs`` as a list."""
        payload = asdict(self)
        payload["context_motifs"] = list(self.context_motifs)
        return payload

    def canonical_json(self) -> str:
        """The exact text :attr:`model_config_sha` is the digest of."""
        return json.dumps(self.as_dict(), sort_keys=True)

    @property
    def model_config_sha(self) -> str:
        """sha256 of the canonical JSON of every field."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def slug(self) -> str:
        """A short readable name for logs; not an identity (see :attr:`model_config_sha`)."""
        return "_".join(
            (
                self.operator,
                f"k{self.hops}",
                _DIRECTION_TOKENS[self.direction],
                f"h{self.hidden}",
                f"d{_number(self.dropout)}",
                f"wd{_number(self.weight_decay)}",
                _LAYER_NORM_TOKENS[self.layer_norm],
                f"in-{self.input_norm}",
                self.feature_contract,
                self.target_kind,
                "masked" if self.supervision_mask else "unmasked",
                f"lam{_number(self.context_lambda)}",
            )
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GnnContractSpec":
        """Rebuild a spec from its mapping, refusing unknown or missing fields.

        Raises:
            GnnSpecError: on unknown or missing fields, or an invalid value.
        """
        if not isinstance(payload, Mapping):
            raise GnnSpecError(f"A spec must be a mapping, got {type(payload).__name__}")
        known = set(CONTRACT_SPEC_FIELD_NAMES)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise GnnSpecError(f"Unknown spec field(s) {unknown}; this package knows {sorted(known)}")
        missing = sorted(known - set(payload))
        if missing:
            raise GnnSpecError(f"Spec is missing field(s) {missing}")
        values = {name: payload[name] for name in CONTRACT_SPEC_FIELD_NAMES}
        motifs = values["context_motifs"]
        if not isinstance(motifs, (list, tuple)):
            raise GnnSpecError(f"context_motifs must be a list, got {motifs!r}")
        values["context_motifs"] = tuple(motifs)
        return cls(**values)


CONTRACT_SPEC_FIELD_NAMES: tuple[str, ...] = tuple(GnnContractSpec().as_dict())
"""Field names of :class:`GnnContractSpec`, in declaration order."""

CONTRACT_SPEC_ONLY_FIELDS: frozenset[str] = frozenset(CONTRACT_SPEC_FIELD_NAMES) - frozenset(FIELD_NAMES)
"""Fields only a :class:`GnnContractSpec` sidecar has; their presence selects that class."""


def spec_class_for(payload: Mapping[str, Any]) -> type[GnnSpec] | type[GnnContractSpec]:
    """Which spec class a sidecar payload describes.

    Any field of :data:`CONTRACT_SPEC_ONLY_FIELDS` selects :class:`GnnContractSpec`, whose
    ``from_dict`` then names whatever else is missing.

    Raises:
        GnnSpecError: if the payload is not a mapping.
    """
    if not isinstance(payload, Mapping):
        raise GnnSpecError(f"A spec must be a mapping, got {type(payload).__name__}")
    return GnnContractSpec if CONTRACT_SPEC_ONLY_FIELDS & set(payload) else GnnSpec


def read_spec(path: Path | str) -> GnnContractSpec:
    """Read a sidecar and return the spec, if this package can serve it.

    Raises:
        GnnSpecError: if the file is missing or not JSON, if it does not describe a
            valid spec, or if the model reads a feature contract outside
            :data:`SERVED_FEATURE_CONTRACTS` (an architecture-only :class:`GnnSpec`
            sidecar names no feature contract and is refused for that reason).
    """
    source = Path(path)
    if not source.is_file():
        raise GnnSpecError(f"Model spec sidecar not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GnnSpecError(f"Could not read the model spec {source}: {type(error).__name__}: {error}") from error
    try:
        spec = spec_class_for(payload).from_dict(payload)
    except GnnSpecError as error:
        raise GnnSpecError(f"{source} does not describe a graph model this package can build: {error}") from error
    if not isinstance(spec, GnnContractSpec) or spec.feature_contract not in SERVED_FEATURE_CONTRACTS:
        contract = spec.feature_contract if isinstance(spec, GnnContractSpec) else None
        raise GnnSpecError(
            f"{source} describes a model on feature contract {contract!r}; this package serves only "
            f"{list(SERVED_FEATURE_CONTRACTS)}"
        )
    return spec


__all__ = [
    "CONTRACT_SPEC_FIELD_NAMES",
    "CONTRACT_SPEC_ONLY_FIELDS",
    "FIELD_NAMES",
    "GRAPH_FREE_OPERATORS",
    "MATMUL_OPERATORS",
    "MAX_HOPS",
    "OPERATOR_KHOP_SUM",
    "OPERATOR_KHOP_SYM",
    "OPERATOR_LEGACY_WEIGHTED_MEAN",
    "OPERATOR_NOGRAPH",
    "OPERATOR_SUPPORT_MAX",
    "OPERATOR_SUPPORT_POOL",
    "SERVED_FEATURE_CONTRACTS",
    "SUPPORT_OPERATORS",
    "TARGET_KINDS",
    "TARGET_KIND_HARD",
    "TARGET_KIND_SOFT",
    "GnnSpec",
    "GnnContractSpec",
    "GnnSpecError",
    "read_spec",
    "spec_class_for",
]
