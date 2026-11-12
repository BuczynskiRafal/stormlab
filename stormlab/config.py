"""Configuration: graph and model vocabulary, and the typed loader for ``config.yaml``.

``config.yaml`` holds the design rules and hydraulic constants the physical feature
extraction reads (:func:`stormlab.features.physical.extract_physical_features`).
It reaches the code only through :class:`Config`, validated at load time, so a
typo or an out-of-range value fails immediately.

The module also names the closed vocabularies the graph and model layers share:
conduit orders, adjacency directions and expansion schemes, graph operators,
normalisations. They live here because ``config`` may be imported from every layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_FILENAME = "config.yaml"

VARIANT_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
"""What a configuration may call itself: an identifier."""

CONDUIT_ORDERS: tuple[str, ...] = ("inp", "sorted")
"""How conduits are ordered into graph rows: as in the INP file, or sorted by name."""

GRAPH_DIRECTIONS: tuple[str, ...] = ("downstream", "upstream")
"""Orientation of a single adjacency: as stored (downstream) or transposed (upstream)."""

KHOP_SCHEMES: tuple[str, ...] = ("legacy_khop_v1",)
"""How the stored adjacency is expanded to ``k`` hops (:mod:`stormlab.graph.khop`)."""

MAX_GNN_HOPS = 5
"""Largest hop count a graph model may name."""

GNN_OPERATOR_LEGACY_WEIGHTED_MEAN = "legacy_weighted_mean"
GNN_OPERATOR_KHOP_SUM = "khop_sum"
GNN_OPERATOR_KHOP_SYM = "khop_sym"
GNN_OPERATOR_SUPPORT_MAX = "support_max"
GNN_OPERATOR_SUPPORT_POOL = "support_pool"
GNN_OPERATOR_NOGRAPH = "nograph"
GNN_OPERATORS: tuple[str, ...] = (
    GNN_OPERATOR_LEGACY_WEIGHTED_MEAN,
    GNN_OPERATOR_KHOP_SUM,
    GNN_OPERATOR_KHOP_SYM,
    GNN_OPERATOR_SUPPORT_MAX,
    GNN_OPERATOR_SUPPORT_POOL,
    GNN_OPERATOR_NOGRAPH,
)
"""Graph operators. Each fixes both the adjacency preprocessing and the layer's aggregation."""

GNN_NORMALIZATION_BATCH = "batch"
GNN_NORMALIZATION_LAYER = "layer"
GNN_NORMALIZATION_NONE = "none"
GNN_NORMALIZATIONS: tuple[str, ...] = (GNN_NORMALIZATION_BATCH, GNN_NORMALIZATION_LAYER, GNN_NORMALIZATION_NONE)
"""Normaliser after each graph step."""

GNN_INPUT_NORM_NONE = "none"
GNN_INPUT_NORM_CONSTANT_SAFE = "constant_safe"
GNN_INPUT_NORMS: tuple[str, ...] = (GNN_INPUT_NORM_NONE, GNN_INPUT_NORM_CONSTANT_SAFE)
"""Whether a ``Normalization`` layer precedes the first graph step."""

GNN_MONITOR_VAL_LOSS = "val_loss"
GNN_MONITORS: tuple[str, ...] = (GNN_MONITOR_VAL_LOSS,)
"""Quantity a model spec records as its early-stopping monitor."""

SPEC_GRAPH_DIRECTION_UPSTREAM = "upstream"
SPEC_GRAPH_DIRECTION_DOWNSTREAM = "downstream"
SPEC_GRAPH_DIRECTION_BOTH = "both"
SPEC_GRAPH_DIRECTIONS: tuple[str, ...] = (
    SPEC_GRAPH_DIRECTION_UPSTREAM,
    SPEC_GRAPH_DIRECTION_DOWNSTREAM,
    SPEC_GRAPH_DIRECTION_BOTH,
)
"""Directions a model spec may name; ``both`` is two adjacency channels with separate weights."""

FROST_ZONE_RANGE = (0.8, 1.6)
"""Admissible frost zone depth [m]."""

_TOP_LEVEL_KEYS = frozenset({"variant", "frost_zone_depth", "required_cover", "max_depth", "max_filling_ratio", "hydraulics"})
_HYDRAULICS_KEYS: tuple[str, ...] = ("theta", "g", "manning_coefficient", "v_min")


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or out of range."""


def repo_root() -> Path:
    """Absolute path of the repository root (the parent of the ``stormlab`` package)."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Hydraulics:
    """Physical constants of the slope and velocity criteria."""

    theta: float
    """Shear stress of the minimum slope criterion [Pa]."""
    g: float
    """Specific weight of water [kN/m3]."""
    manning_coefficient: float
    """Manning roughness assumed for velocity checks."""
    v_min: float
    """Minimum self-cleansing velocity [m/s]."""

    def as_dict(self) -> dict[str, float]:
        """The constants as a plain mapping."""
        return {"theta": self.theta, "g": self.g, "manning_coefficient": self.manning_coefficient, "v_min": self.v_min}


@dataclass(frozen=True)
class Config:
    """Validated configuration."""

    source: Path
    root: Path
    variant: str
    """Name the configuration declares; :func:`load_config` refuses a request for another."""
    frost_zone_depth: float
    """Frost zone depth the physical feature extraction uses [m]."""
    required_cover: float | None
    """Soil cover a conduit must carry above its crown [m]; ``None`` means the frost zone."""
    max_depth: float
    """Deepest admissible pipe invert below ground [m]."""
    max_filling_ratio: float
    """Largest admissible filling as a share of the diameter."""
    hydraulics: Hydraulics

    @property
    def v_min(self) -> float:
        """Self-cleansing minimum velocity [m/s]."""
        return self.hydraulics.v_min


def default_config_path() -> Path:
    """Path of the config file shipped with the package."""
    return Path(__file__).resolve().parent / DEFAULT_CONFIG_FILENAME


def load_config(
    path: Path | str | None = None,
    *,
    root: Path | None = None,
    check_inputs: bool = True,
    variant: str | None = None,
) -> Config:
    """Read and validate the configuration.

    Args:
        path: config file; defaults to the one shipped with the package.
        root: repository root recorded on the result; defaults to the parent of the package.
        check_inputs: accepted for compatibility. The configuration names no input
            files, so there is nothing to check.
        variant: the configuration name the caller expects. ``None`` accepts the
            file's own; any other value must equal it.

    Raises:
        ConfigError: if the file is missing or not a mapping, if keys are unknown
            or missing, if a value is out of range, or if ``variant`` does not match.
    """
    del check_inputs  # no configured inputs to verify
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Config {config_path} must contain a mapping, got {type(raw).__name__}")
    config = _build(raw, source=config_path, base=(root or repo_root()).resolve())
    if variant is not None and variant != config.variant:
        raise ConfigError(f"Config {config_path} declares variant {config.variant!r}, but {variant!r} was requested")
    return config


def _build(raw: Mapping[str, Any], *, source: Path, base: Path) -> Config:
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"Unknown config keys in {source}: {unknown}; allowed: {sorted(_TOP_LEVEL_KEYS)}")
    missing = sorted(_TOP_LEVEL_KEYS - set(raw))
    if missing:
        raise ConfigError(f"Missing config keys in {source}: {missing}")

    variant = raw["variant"]
    if not isinstance(variant, str) or not VARIANT_NAME_PATTERN.match(variant):
        raise ConfigError(f"variant must match {VARIANT_NAME_PATTERN.pattern}, got {variant!r} (config {source})")

    frost_zone = _as_float(raw, "frost_zone_depth", source)
    low, high = FROST_ZONE_RANGE
    if not low <= frost_zone <= high:
        raise ConfigError(f"frost_zone_depth must be within [{low}, {high}] m, got {frost_zone} (config {source})")

    max_depth = _as_float(raw, "max_depth", source)
    if max_depth <= 0.0:
        raise ConfigError(f"max_depth must be positive, got {max_depth} m (config {source})")

    required_cover: float | None = None
    if raw["required_cover"] is not None:
        required_cover = _as_float(raw, "required_cover", source)
        if required_cover <= 0.0:
            raise ConfigError(f"required_cover must be positive when given, got {required_cover} m (config {source})")
        if required_cover >= max_depth:
            raise ConfigError(f"required_cover {required_cover} m is not below max_depth {max_depth} m (config {source})")

    max_filling_ratio = _as_float(raw, "max_filling_ratio", source)
    if not 0.0 < max_filling_ratio <= 1.0:
        raise ConfigError(f"max_filling_ratio must be in (0, 1], got {max_filling_ratio} (config {source})")

    return Config(
        source=source,
        root=base,
        variant=variant,
        frost_zone_depth=frost_zone,
        required_cover=required_cover,
        max_depth=max_depth,
        max_filling_ratio=max_filling_ratio,
        hydraulics=_build_hydraulics(raw["hydraulics"], source=source),
    )


def _build_hydraulics(raw: Any, *, source: Path) -> Hydraulics:
    """The hydraulic constants: exactly :data:`_HYDRAULICS_KEYS`, each a positive number."""
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'hydraulics' must be a mapping, got {type(raw).__name__} (config {source})")
    if set(raw) != set(_HYDRAULICS_KEYS):
        raise ConfigError(f"'hydraulics' must name exactly {sorted(_HYDRAULICS_KEYS)} in {source}, got {sorted(raw)}")
    values = {key: _as_float(raw, key, source, prefix="hydraulics.") for key in _HYDRAULICS_KEYS}
    for key, value in values.items():
        if value <= 0.0:
            raise ConfigError(f"hydraulics.{key} must be positive, got {value} (config {source})")
    return Hydraulics(**values)


def _as_float(raw: Mapping[str, Any], key: str, source: Path, *, prefix: str = "") -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{prefix}{key} must be a number, got {value!r} (config {source})")
    return float(value)
