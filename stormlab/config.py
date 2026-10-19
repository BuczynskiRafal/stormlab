"""Typed loader for ``config.yaml``.

Everything that can vary between runs — input paths, frost zones, seeds,
hydraulic constants and behavioural switches — lives in the YAML file and
reaches the code only through :class:`Config`. Relative paths are resolved
against the repository root and input paths are checked for existence at load
time, so a typo fails immediately with the full list of missing files instead of
surfacing as an empty dataset much later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Collection, Mapping

import yaml

from stormlab.contracts.enums import RECOMMENDATION_CLASS_ORDER
from stormlab.contracts.features import (
    FEATURE_CONTRACTS,
    NEIGHBORHOOD_CONTRACT_COLUMNS,
    NEIGHBORHOOD_CONTRACT_NAME,
    NEIGHBORHOOD_HOPS,
    FeatureContract,
    feature_contract,
    neighborhood_columns_of,
)

DEFAULT_CONFIG_FILENAME = "config.yaml"

_STRATUM_FORCEABLE: frozenset[str] = frozenset({"categories", "outlet_cover", "node_depth"})
"""Dimensions a stratum may pin.

Deliberately narrow: these three are cyclic per-file choices, so pinning them
changes which combination a file realises without disturbing the grid axes
(diameter, slope, storm) whose pool-wide coverage ``min_occurrences`` gates.
"""

VARIANT_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
"""What a variant may be called: an identifier, so it can be a directory name.

The one definition: ``reports.variants`` imports it from here rather than keeping
a second copy, because a manifest that accepted a name this loader refuses would
describe variants no run can be started under.

Deliberately *not* validated against ``results/variants.yaml``. The manifest is an
artifact of the results tree and this loader is the thing every experiment starts
from; making the config refuse to load until the manifest agrees with it would
mean a run cannot start because a results file is out of date, which is the wrong
failure at the wrong moment."""

CONFIG_OVERLAY_TEMPLATE = "{stem}.{variant}{suffix}"
"""How a variant's overlay is named beside the base config: ``config.B2.yaml``."""

CONDUIT_ORDERS: tuple[str, ...] = ("inp", "sorted")
GRAPH_DIRECTIONS: tuple[str, ...] = ("downstream", "upstream")
KHOP_SCHEMES: tuple[str, ...] = ("legacy_khop_v1",)
FLOW_LAWS: tuple[str, ...] = ("legacy_k_d25", "manning_v2")
SPLIT_PROTOCOLS: tuple[str, ...] = ("file_v1", "group_v2")
LINEAGE_SOURCES: tuple[str, ...] = ("filename_family_v1", "manifest")
MIN_SLOPE_RULES: frozenset[str] = frozenset({"hydromechanical", "hydromechanical_canonical", "imhoff"})
MLP_NORMALIZATION_CONSTANT_SAFE = "constant_safe"
MLP_NORMALIZATION_LEGACY = "legacy"
MLP_NORMALIZATIONS: tuple[str, ...] = (MLP_NORMALIZATION_CONSTANT_SAFE, MLP_NORMALIZATION_LEGACY)
GNN_NORMALIZATION_BATCH = "batch"
GNN_NORMALIZATION_LAYER = "layer"
GNN_NORMALIZATION_NONE = "none"
GNN_NORMALIZATIONS: tuple[str, ...] = (GNN_NORMALIZATION_BATCH, GNN_NORMALIZATION_LAYER, GNN_NORMALIZATION_NONE)
WROCLAW_TRAIN_SUPPORT_CONSTANT_COLUMNS = "constant_columns"
WROCLAW_TRAIN_SUPPORT_HYPERCUBE = "hypercube"
WROCLAW_TRAIN_SUPPORTS: tuple[str, ...] = (
    WROCLAW_TRAIN_SUPPORT_CONSTANT_COLUMNS,
    WROCLAW_TRAIN_SUPPORT_HYPERCUBE,
)
"""The vocabularies the graph layer and the configuration share.

They live here, in the cross-cutting module every layer may import, because the
alternative — a copy in ``config`` and a copy in ``graph`` kept in step by a test
asserting the two are equal — is a duplication with a test attached, not a
contract. Ordered tuples rather than sets so the "expected one of ..." messages
read the same on every run.
"""

MAX_GNN_HOPS = 5
"""Largest hop count any part of studies (b)/(b2) may ask for.

``gnn.hops`` is ``1..5`` and every grid stops there. Stated once, here, because
:data:`stormlab.models.gnn_spec.MAX_HOPS` is this value and a second literal
would be a bound that can drift from the one the configuration enforces."""

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
"""The named graph operators of study (b2), in declaration order.

An operator is a **pair**: how the stored adjacency is preprocessed, and how the
layer aggregates over it. Splitting the pair into two free dimensions was tried
and dropped — the legacy layer computes a plain ``A·X`` and the "mean" comes
entirely from the row-wise ``D⁻¹`` of :mod:`stormlab.graph.khop`, so
``mean × none`` and ``sum × rownorm`` name the same arithmetic twice, and the two
``support_*`` operators ignore the matrix values altogether.

The first five are the **search space** of the ablation (``gnn.ablation.s1``).
``nograph`` is not in :data:`SEARCHABLE_GNN_OPERATORS` and never competes: it is
the ablation *arm* of the final step — the same architecture and the same schedule as the winner, trained
from scratch with the neighbour term removed — so that the effect of message
passing is a difference between two models trained under controlled conditions
rather than the effect of blanking the adjacency of a model that learned with
one (``e_gnn.zero_adjacencies``, which stays and measures the other thing)."""

SEARCHABLE_GNN_OPERATORS: tuple[str, ...] = tuple(operator for operator in GNN_OPERATORS if operator != GNN_OPERATOR_NOGRAPH)
"""The operators the ablation is allowed to search over: everything but ``nograph``.

``gnn.ablation.s1.operators`` is validated against this and not against
:data:`GNN_OPERATORS`. Putting ``nograph`` in the S1 grid would enter the control
arm of V3 as a competitor in the search that chooses what V3 compares against —
and the answer would still look like a valid ablation. Derived by exclusion so a
sixth operator cannot be added to one list and forgotten in the other."""

GNN_INPUT_NORM_NONE = "none"
GNN_INPUT_NORM_CONSTANT_SAFE = MLP_NORMALIZATION_CONSTANT_SAFE
GNN_INPUT_NORMS: tuple[str, ...] = (GNN_INPUT_NORM_NONE, GNN_INPUT_NORM_CONSTANT_SAFE)
"""Whether a ``Normalization`` layer stands in front of the first graph step.

``none`` is the architecture B1 was trained under and means the layer is
*physically absent*, not present with neutral parameters: a layer with mean 0 and
variance 1 still creates weights, still appears in a weight file and still has to
be loaded, so the two are not the same model. ``constant_safe`` is the guard
:data:`stormlab.models.mlp.DEFAULT_NORMALIZATION` describes, adapted on the
training rows of the fold and on nothing else."""

GNN_MONITOR_VAL_LOSS = "val_loss"
GNN_MONITORS: tuple[str, ...] = (GNN_MONITOR_VAL_LOSS,)
"""What early stopping restores the best weights by.

One value, and a closed set rather than a bare string, because study (b2) had to
decide it: macro-F1 on a small validation fold is a step function and noisier
than the loss, so using it to *restore* would add to the very instability the
study measures. The selection criterion is a different quantity from the restore
criterion, and keeping them apart is the point (``api_map`` section 11.8)."""

SLOPE_PROFILES: dict[str, dict[str, str]] = {
    "legacy_mixed_v1": {
        "feature_min_slope_rule": "hydromechanical",
        "hydraulic_validation_min_slope_rule": "imhoff",
    },
    "corrected_hydromechanical_v2": {
        "feature_min_slope_rule": "hydromechanical",
        "hydraulic_validation_min_slope_rule": "hydromechanical_canonical",
    },
}

FROST_ZONE_RANGE = (0.8, 1.6)

SWMM_FLOW_ROUTINGS: tuple[str, ...] = ("STEADY", "KINWAVE", "DYNWAVE")
"""The three ``FLOW_ROUTING`` values SWMM 5 accepts, spelled as the INP spells them."""

SWMM_LINK_OFFSETS: tuple[str, ...] = ("DEPTH", "ELEVATION")
"""The two ``LINK_OFFSETS`` conventions SWMM 5 accepts.

``DEPTH`` measures a conduit end above the node invert, ``ELEVATION`` states it
absolutely. Which one a file uses decides how a pipe invert is computed from the
same numbers, so it is a property of the population rather than a preference."""

SWMM_PROTOCOL_OPTIONS: tuple[str, ...] = ("FLOW_ROUTING", "LINK_OFFSETS", "ROUTING_STEP")
"""The three ``[OPTIONS]`` a C1-style protocol declares about a simulation.

Named here rather than beside the stage that reads them, because
``c1.simulation.rewrite_options`` is validated against this list at load time and
``stormlab.datasets.c1_cases`` sits above this module: a configuration that
accepted an option the stage cannot check would declare a rewrite nothing
verifies."""

C1_MOTIF_SET_TRAIN = "train"
C1_MOTIF_SET_HELD = "held"
C1_MOTIF_SET_DIAGNOSTIC = "diagnostic"
C1_MOTIF_SETS: tuple[str, ...] = (C1_MOTIF_SET_TRAIN, C1_MOTIF_SET_HELD, C1_MOTIF_SET_DIAGNOSTIC)
"""Where a conflict motif of series C1 goes: into the context loss, or out of it.

``held`` motifs never enter ``C_ij``; they are what the effect of the penalty is
measured on, so the measurement is not a reading of the training objective.

``diagnostic`` is the third answer, and it is not a shade of the first two: the
motif is counted and reported like any other, and it enters neither the penalty nor
the held-out measurement. It exists because a pattern can stop being a statement
about what the network may not do while remaining a number worth watching — M5
after the author's decision of 2026-09-07 that a regrade may move both ends of a
conduit. Dropping such a motif from the catalogue would stop the count; leaving it
in ``held`` would make E7 answer for a conflict nobody claims is one."""

C1_MOTIF_SETS_REQUIRED: tuple[str, ...] = (C1_MOTIF_SET_TRAIN, C1_MOTIF_SET_HELD)
"""The sets a protocol has to fill for E7 to be reportable at all.

A penalty built from no motif is zero everywhere, and an effect measured on no motif
is not measured; ``diagnostic`` carries neither role, so requiring it non-empty would
force every protocol to keep a motif out of use forever."""

C1_GRAPH_DIRECTION_UPSTREAM = "upstream"
C1_GRAPH_DIRECTION_DOWNSTREAM = "downstream"
C1_GRAPH_DIRECTION_BOTH = "both"
"""The two separate channels with separate weights of plan16 §3 (E6).

Named rather than spelled out at each use, because the string is the switch that
decides how many adjacency matrices a sample carries and how many projections a
graph step creates; a literal in five places is five chances to write ``"Both"``
in one of them."""

C1_GRAPH_DIRECTIONS: tuple[str, ...] = (
    C1_GRAPH_DIRECTION_UPSTREAM,
    C1_GRAPH_DIRECTION_DOWNSTREAM,
    C1_GRAPH_DIRECTION_BOTH,
)
"""Message-passing directions series C1 declares as its search space.

Deliberately **not** :data:`GRAPH_DIRECTIONS`, which is the vocabulary of the
adjacency the historical ablation grids are iterated over: ``e_gnn.py`` builds the
B2--B4 search space out of that tuple directly, so a third value there would grow
every one of those grids by half and move results the goldens pin. The two tuples
also do **not** share an order — ``GRAPH_DIRECTIONS[0]`` is ``downstream`` and
``C1_GRAPH_DIRECTIONS[0]`` is ``upstream`` — so nothing may read "the default
direction" off an index; the direction is always compared against a name."""

C1_GROUP_WEIGHTS_EQUAL = "equal"
C1_GROUP_WEIGHT_MODES: tuple[str, ...] = (C1_GROUP_WEIGHTS_EQUAL,)
"""How the aggregator weighs the active source groups against each other.

One value, and that is the point: plan16 §2 P4 records equal group weights as an
a priori assumption precisely because the sources are designed together and their
agreement is not independent evidence. Anything else would be a weighting
estimated from the sources, which the plan rules out — so the protocol may say it
and the loader will refuse it."""

C1_OVERLOAD_PREMISE_FILLING_HOURS = "filling_hours"
C1_OVERLOAD_PREMISE_INLET_FLOODING = "inlet_flooding"
C1_OVERLOAD_PREMISES: tuple[str, ...] = (C1_OVERLOAD_PREMISE_FILLING_HOURS, C1_OVERLOAD_PREMISE_INLET_FLOODING)
"""What a label source of the series treats as evidence that a conduit is overloaded.

``filling_hours`` is the premise of C1 and the default of the schema: the conduit is
overloaded when its filling breaks the requirement or its capacity limited the flow.
``inlet_flooding`` is the premise of C2: under a dynamic wave a pipe running full is
ordinary, and what marks an overload is that the upper node spilled onto the surface
(plan17 §Etap 5). The premise is a parameter of the protocol rather than a branch on
the variant name, so that C1 keeps its meaning by declaring nothing."""

C1_CATEGORY_ROLE_INFILTRATION_ONLY = "infiltration_only"
C1_CATEGORY_ROLES: tuple[str, ...] = (C1_CATEGORY_ROLE_INFILTRATION_ONLY,)
"""The only role the land cover category may play in a label source.

plan16 §0 change 6 admits it as an infiltration premise and forbids it as a cost
or "less work" preference. The value is enforced rather than described: the source
registry checks that every source declaring a category role declares this one."""

_TIME_STEP_PATTERN = re.compile(r"^\d+:[0-5]\d:[0-5]\d$")
"""SWMM ``H:MM:SS`` step, the form ``ROUTING_STEP`` is written in."""

_PATH_KEYS = ("dataset_dir", "catchment_dir", "catchment_ood", "wroclaw_inp", "results_dir", "work_dir")
_INPUT_DIR_KEYS = ("dataset_dir", "catchment_dir")
_INPUT_FILE_KEYS = ("catchment_ood", "wroclaw_inp")
_SEED_KEYS = ("python", "numpy", "tf", "split", "bootstrap")

_OPTIONAL_TOP_LEVEL_KEYS = frozenset(
    {"c1", "catchment", "catchment_dataset", "dataset_run_id", "eval", "generate", "feature_contract", "tabular", "wroclaw"}
)
"""Keys a config may omit.

``dataset_run_id`` names the ``build-dataset`` run the experiments read their
artifacts from. It is optional because a config that predates the dataset step is
still a valid config, and because the CLI can override it per invocation; every
other key is mandatory, so a typo is a failure rather than a silent default."""

_TOP_LEVEL_KEYS = frozenset(
    {
        "variant",
        "paths",
        "wroclaw_raingage",
        "frost_zone_dataset",
        "frost_zone_serving",
        "required_cover",
        "max_depth",
        "seeds",
        "flow_constant",
        "flow_law",
        "nmaxq_clip",
        "slope_profile",
        "khop_scheme",
        "surrogate_edges",
        "conduit_order",
        "graph_direction",
        "max_filling_ratio",
        "hydraulics",
        "gnn",
        "mlp_normalization",
        "expected",
        "split",
    }
)

_ALL_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS | _OPTIONAL_TOP_LEVEL_KEYS


class ConfigError(ValueError):
    """Raised when the configuration file is malformed or points at missing inputs."""


class MissingInputsError(ConfigError):
    """Raised when configured input paths do not exist on disk."""


def repo_root() -> Path:
    """Absolute path of the repository root (the parent of the ``stormlab`` package)."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Seeds:
    """Random seeds; every stochastic step draws from exactly one of these."""

    python: int
    numpy: int
    tf: int
    split: int
    bootstrap: int
    """Seed of every resampling procedure — the paired bootstrap of study (b2).

    Its own key rather than a reuse of ``numpy``: a confidence interval is a
    statement about a resampling that has to be reproducible on its own, and
    sharing a seed with the general numpy stream would make the interval depend on
    how much randomness the code before it happened to draw."""

    def as_dict(self) -> dict[str, int]:
        """Seeds as a plain mapping, for provenance records."""
        return {
            "python": self.python,
            "numpy": self.numpy,
            "tf": self.tf,
            "split": self.split,
            "bootstrap": self.bootstrap,
        }


@dataclass(frozen=True)
class SlopeProfile:
    """Which minimum-slope rule applies to feature engineering and to validation.

    The historical code used two different rules simultaneously; naming the pair
    keeps that visible instead of collapsing it into a single switch.
    """

    name: str
    feature_min_slope_rule: str
    hydraulic_validation_min_slope_rule: str


@dataclass(frozen=True)
class SplitConfig:
    """Leakage-firewall protocols and the relations used to form their units."""

    protocols: tuple[str, ...]
    lineage: str
    topology_equivalence: bool
    merge_reduction_warn: float
    merge_reduction_gate: float | None = None
    """Above this share, over-merging is a build failure rather than a warning.

    ``None`` keeps the base behaviour, in which an over-merged pool is only
    reported. A generated variant sets it, because there the merge is a fact
    about a pool that was *designed* and can be redesigned, and a pool whose
    duplicate closure has eaten its split units cannot be split at all
    (plan14 §3.3/§4.3(c) amendment 2026-08-30).
    """

    def as_dict(self) -> dict[str, object]:
        return {
            "protocols": list(self.protocols),
            "lineage": self.lineage,
            "topology_equivalence": self.topology_equivalence,
            "merge_reduction_warn": self.merge_reduction_warn,
            "merge_reduction_gate": self.merge_reduction_gate,
        }


OBSERVED_CLASS_SOURCES: tuple[str, ...] = ("train_test", "train_val")
"""Which parts the ``observed8`` class set may be read from (plan14 §4.3 g i).

``train_test`` is what B1--B3 were frozen under and stays the default: the name
``observed8`` is a published convention every cross-entropy in Tab. 3.16 is
averaged over, so moving it is a change of figure and belongs to the variant that
introduces the new protocol. ``train_val`` is the honest definition and is what
B4 uses."""

EXPERIMENT_MODES: tuple[str, ...] = ("tune", "evaluate")
"""The two halves of the sealed-test process (plan14 §4.3 g).

``tune`` never reads the test part -- it is not merely discouraged from doing so,
:class:`stormlab.experiments.e_tabular.SealedTestError` is raised if it tries --
and freezes a configuration into a ``tune_manifest.json``. ``evaluate`` reads the
test part once, for a configuration that is already frozen."""

DEFAULT_EXPERIMENT_MODE = "evaluate"
"""Today's behaviour, so a command line that names no mode is bit-identical."""


@dataclass(frozen=True)
class TabularConfig:
    """How study (a) treats the sealed test part and the ``observed8`` class set.

    Both fields default to what variants B1--B3 were frozen under, so a variant
    that says nothing about them behaves exactly as it did before this section
    existed (plan14 §4.3 g).
    """

    require_tune: bool = False
    """Whether ``evaluate`` refuses to run without a ``tune_manifest.json``.

    ``False`` for B1--B3, whose numbers were produced by a single pass that both
    chose and scored; the manifest cannot be conjured for a run that has already
    happened. ``True`` for B4, where the test part is unsealed only for a
    configuration frozen beforehand."""

    observed_classes_from: str = "train_test"
    """Which parts name the ``observed8`` set; see :data:`OBSERVED_CLASS_SOURCES`."""

    report_per_class_metrics: bool = False
    """Publish per-class F1 and balanced accuracy for this variant.

    The metrics were introduced for B4.  Keeping the default false preserves the
    frozen key sets of B1--B3 when their studies are recomputed.
    """

    def as_dict(self) -> dict[str, object]:
        return {
            "require_tune": self.require_tune,
            "observed_classes_from": self.observed_classes_from,
            "report_per_class_metrics": self.report_per_class_metrics,
        }


@dataclass(frozen=True)
class WroclawConfig:
    """Wroclaw evaluation switches and diagnostic modes (plan14 rozstrzygnięcie 14)."""

    train_support: str = WROCLAW_TRAIN_SUPPORT_CONSTANT_COLUMNS

    def as_dict(self) -> dict[str, object]:
        return {
            "train_support": self.train_support,
        }


@dataclass(frozen=True)
class ExpectedDataset:
    """The size a full recommendation-dataset build must come out at."""

    files: int | None = None
    rows: int | None = None
    conduits_per_file: int | None = None
    manifest: Path | None = None

    @property
    def generated(self) -> bool:
        return self.manifest is not None


@dataclass(frozen=True)
class ExpectedCatchment:
    """The size a full catchment study must find in its inputs."""

    files: int
    rows: int
    subcatchments: int
    ood_subcatchments: int


@dataclass(frozen=True)
class ExpectedWroclaw:
    """The size of the real network chapter 4 is about."""

    conduits: int
    subcatchments: int


@dataclass(frozen=True)
class Expected:
    """Population gates per study.

    Declared here rather than derived from whatever is on disk at run time. A
    gate that measures its own expectation from the current input directory
    cannot fail, and the failure it cannot report — a dataset that lost files
    since the numbers were produced — is exactly the one that would invalidate
    the thesis.
    """

    dataset: ExpectedDataset
    catchment: ExpectedCatchment
    wroclaw: ExpectedWroclaw
    min_relative_feature_iqr: float = 0.0


@dataclass(frozen=True)
class CatchmentRegime:
    """The hydrological regime of Wrocław, written as a band over two INP columns.

    Two numbers decide whether a subcatchment is "the kind Wrocław is made of":
    it is flat and it is sealed. They are a property of the *target* network, not
    of the training set, so they are declared rather than measured from whatever
    the sampler happened to draw (plan18 §5 [M]).
    """

    slope_pct_max: float
    imperv_pct: tuple[float, float]


@dataclass(frozen=True)
class CatchmentDatasetProfile:
    """How the classifier's training subcatchments are drawn (plan18 §9b).

    Two layers, two seeds: the base layer reproduces the mechanism of the 992
    historical subcatchments, and the flat layer is the declared provenance
    deviation that brings Wrocław's regime inside the set. Splitting the seed is
    what lets the flat layer be resized without moving a single row of the base
    layer, so the two decisions stay separable in the dissertation.
    """

    seed: int
    n_base: int
    n_flat: int
    template: Path
    rain_gages: tuple[str, ...]
    out_dir: Path
    area_range: tuple[float, float]
    regime: CatchmentRegime


@dataclass(frozen=True)
class CatchmentClassTableProfile:
    """How the classifier's subcatchments are drawn from the frozen class table (plan19 §1).

    The other profile draws around fuzzy prototypes and lets the class histogram
    follow the prototype table; this one draws ``n_per_class`` rows inside the seven
    bands each class is *defined* by, so the label of a row is the definition it came
    from rather than a consequence of a generator nobody wrote down.

    ``class_table_sha`` is declared in the YAML and checked against
    :func:`stormlab.contracts.catchment_classes.class_table_sha` at load time. It is
    the only field here that is not a sampling parameter, and it is the point of the
    section: the table was frozen before any model was trained, and a configuration
    that could name a different one would let a run publish accuracies against a
    definition that had quietly moved. There is no ``regime`` band, because nothing
    in this draw depends on one -- the Wrocław share of the drawn population is a
    read taken afterwards, not a target the profile is tuned towards.
    """

    seed: int
    n_per_class: int
    template: Path
    rain_gages: tuple[str, ...]
    out_dir: Path
    area_range: tuple[float, float]
    class_table_sha: str


CATCHMENT_GENERATOR_PROTOTYPES = "prototypes"
"""The fuzzy-prototype sampler of plan18 §9b; the value assumed when the key is absent."""

CATCHMENT_GENERATOR_CLASS_TABLE = "class_table_v2"
"""The class-table sampler of plan19 §1."""

CATCHMENT_GENERATORS: tuple[str, ...] = (CATCHMENT_GENERATOR_PROTOTYPES, CATCHMENT_GENERATOR_CLASS_TABLE)
"""Which sampler a ``catchment_dataset:`` section describes.

The key is optional and defaults to :data:`CATCHMENT_GENERATOR_PROTOTYPES`, which is
what keeps ``config.C3.yaml`` loading into the profile it always loaded into and its
``catchment_dataset_config_sha`` byte for byte what the frozen manifest records.
"""


DEFAULT_CATCHMENT_FEATURE_SET = "full_12"
"""The feature set every run used before plan19: all twelve contract columns.

The literal rather than an import of ``stormlab.contracts.catchment_features``:
this module is read by everything and imports the contracts lazily, inside the
builders. A unit test pins the two together."""

DEFAULT_CATCHMENT_HIDDEN_UNITS = 32
"""The width of ``stormlab.models.catchment.HIDDEN_UNITS``, restated for the same reason."""


@dataclass(frozen=True)
class ExtraHoldout:
    """A directory of INP files the classifier is read on beside its own holdout.

    The seven INP columns of a subcatchment are in the file; the four report
    columns are not, so a directory named here is scored **without running SWMM**
    and only by models whose feature set fits inside those seven. That is the
    whole reason it is a sample rather than a population: ``limit`` files drawn
    under ``seed`` cost nothing to read, and the figure they yield is an
    observation about a pool built for another study, never a number this study
    is judged by (plan19 Etap 5, D6).

    ``dir`` stays relative to the repository root for the reason every configured
    path does, and ``seed`` is declared rather than derived so that two arms of an
    ablation read the *same* files -- a cross-arm difference computed on two
    different samples would not be a difference between the arms.
    """

    id: str
    dir: Path
    limit: int
    seed: int


@dataclass(frozen=True)
class CatchmentConfig:
    """Which classifier a catchment run trains, and what it is compared against.

    Absent for every variant that does not retrain one, and that absence is the
    normal state: it is what keeps the keys, the weights file name and the
    registry entry of a run under the base configuration byte for byte what they
    were before this section existed.
    """

    model_id: str
    reference_run_id: str | None = None
    """A finished run whose classifier joins the out-of-distribution comparison.

    Named rather than hard-coded because it is a run id, and a run id in the
    experiment module would be a second place to keep the pipeline's history."""
    reference_run_ids: tuple[str, ...] = ()
    """Further finished runs whose classifiers join the same comparison, in file order.

    Beside ``reference_run_id`` rather than replacing it: ``config.C3.yaml`` names
    one reference and its keys must not move, while plan19 compares the new
    classifier against three earlier ones at once. Both are read, in this order,
    and a run named twice is compared once."""
    feature_set: str = DEFAULT_CATCHMENT_FEATURE_SET
    """Which named subset of the twelve contract columns the classifier is built on.

    A key of ``stormlab.contracts.catchment_features.FEATURE_SETS``. The ablations
    of plan19 §Etap 4 differ from ``K1`` in this value and nothing else, which is
    what makes them ablations rather than second pipelines."""
    hidden_units: int = DEFAULT_CATCHMENT_HIDDEN_UNITS
    """Width of both dense layers of the classifier.

    A variant parameter (plan19 [Z]3): the alternative to the notebook's 32 is
    declared here and chosen on the validation split, never by editing the module
    constant every earlier run was trained under."""
    extra_holdouts: tuple[ExtraHoldout, ...] = ()
    """Further directories of INP files the run reads its models on, in file order.

    Empty for every variant that declares none, which is what keeps the base
    configuration, ``C3`` and ``K1`` registering exactly the keys they registered
    before this field existed."""


@dataclass(frozen=True)
class Hydraulics:
    """The physical constants the feature pipeline computes with.

    They were module constants of ``hydraulics/`` until a change to any of them
    meant editing code. A change to ``theta`` moves ``MinRequiredSlope``,
    ``MaxAllowableSlope``, ``NSlope`` and ``ValMinSlope`` at once, so it is a
    configuration change followed by a rerun, not a patch — which is only true
    while every reader takes the value from here.
    """

    theta: float
    g: float
    manning_coefficient: float
    min_roughness: float
    max_roughness: float
    v_min: float

    def as_dict(self) -> dict[str, float]:
        """The constants as a plain mapping, for provenance records."""
        return {
            "theta": self.theta,
            "g": self.g,
            "manning_coefficient": self.manning_coefficient,
            "min_roughness": self.min_roughness,
            "max_roughness": self.max_roughness,
            "v_min": self.v_min,
        }


@dataclass(frozen=True)
class GenerateLength:
    """Log-normal length distribution parameters."""

    dist: str
    median: float
    min: float
    max: float


@dataclass(frozen=True)
class GenerateArea:
    """Log-uniform area distribution parameters."""

    dist: str
    min: float
    max: float


@dataclass(frozen=True)
class GenerateFlowLength:
    """Uniform flow length distribution parameters."""

    dist: str
    min: float
    max: float


@dataclass(frozen=True)
class Generate:
    """Configuration of the synthetic SWMM network generator.

    Every field is validated at load time: lists are sorted, unique, and
    non-empty; ``min < median < max`` for the length distribution; category
    names are a subset of ``SubcatchmentCategory``; topology families are a
    subset of the known families.
    """

    seed: int
    template: str
    output_dir: str
    n_realisations: int
    combinations_per_skeleton: int | None
    """How many (diameter, slope, storm) cells each topology realisation gets.

    ``None`` is the full Cartesian product, which ties pool size to topology
    count: ~1500 files then means ~6 skeletons, and six split units do not
    survive a handful of @2dp collisions (plan14 §3.3/§4.3(c) amendment
    2026-08-30). A number gives each skeleton an evenly spaced slice instead,
    so the same pool size carries many more independent topologies while the
    pool as a whole still covers every value of every axis.
    """

    topology_families: tuple[str, ...]
    diameters_m: tuple[float, ...]
    slopes_per_mille: tuple[float, ...]
    roughness: tuple[float, ...]
    rain_gages: tuple[str, ...]
    categories: tuple[str, ...]
    subcatchments_per_conduit: tuple[int, ...]
    outlet_cover: tuple[str, ...]
    node_depth: tuple[str, ...]
    length: GenerateLength
    area: GenerateArea
    flow_length: GenerateFlowLength
    sizing: str
    elevation_precision_m: float
    slope_tolerance_per_mille: float
    outfall_invert_m: float
    min_occurrences: int
    frost_zone_dataset: float
    two_subcatchment_share: float
    strata: tuple[Mapping[str, Any], ...] = ()
    """Declared over-samplings of the base pool (``generate.strata``).

    Part of the pool design, declared before the split is drawn, for label
    branches the balanced grid reaches too rarely to place 50 rows in each
    holdout (plan14 §3.3/§4.3(c) amendment 2026-08-30). Held as plain mappings:
    ``generate`` sits above ``config`` in the layer order, so the typed form
    lives there and is built from these.
    """

    augmentation_targets: tuple[str, ...] = ()
    """Opt-in hydraulic profiles used only by ``augment-train``.

    The shipped pool configuration leaves this empty.  Keeping the profiles on
    the typed generation specification, rather than as hidden mutations in the
    augmentation loop, makes their effect part of the deterministic config
    digest and lets the generator manifest state which profile produced each
    file.
    """

    category_params_override: Mapping[str, Mapping[str, tuple[float, float]]] = field(default_factory=dict)
    """Explicit ``perc_imperv``/``perc_slope`` bands replacing a category's built-in ones.

    Held as plain validated mappings for the reason ``strata`` is: ``generate``
    sits above ``config`` in the layer order, so the typed form
    (:class:`stormlab.generate.grid.CategoryOverride`) lives there and is built
    from these. The bands are stated in the configuration rather than added to
    ``CATEGORY_PARAMS``, because a pool that samples a category over a narrower
    band is a property of that pool and not a correction of what the category
    means everywhere (plan18 §Etap 7).
    """

    cover_m: tuple[float, float] | None = None
    """Soil cover band [m] the ``flat_cover`` node depth mode draws over each pipe.

    Required exactly when ``node_depth`` names ``flat_cover`` and meaningless
    otherwise: the other three modes draw their depth from the frost zone and
    from fixed literals, and a band declared beside them would describe nothing.
    """

    parity_reference: bool = True
    """Whether the rows of this pool exist in the parity gate's reference profile.

    ``True`` for every pool that *is* the reference or a rebuild of it (B4, C1,
    C2). A pool the reference profile never measured declares ``False``, and the
    case stage then reports its files as ``out_of_reference`` instead of failing
    them as cases the rebuild lost (:func:`stormlab.datasets.c1_cases.parity_scope`).
    """

    family: str | None = None
    """The name this pool goes under as a *population*, e.g. ``urban_flat``.

    Not a topology family: those are per file and live in
    ``topology_families``. This one names the whole directory, so a reader of the
    generator manifest can tell a pool drawn for one hydraulic regime from the
    balanced pool it is added to.
    """

    skeleton_id_offset: int = 0
    """Where this pool's ``skeleton_id``/``lineage_id`` numbering starts.

    A pool drawn to be **added** to another one may not number from zero. The
    lineage is the identity of a design, and ``family_v3`` closes over it
    (:func:`stormlab.datasets.splits.family_units`), so a new network numbered 0
    would land in one split family with the unrelated network 0 of the pool it
    joins -- and, more visibly, in one file name, since the generator names files
    ``g_<lineage>_<gage>_<index>.inp``. The augmentation batches of B4 already
    solved this outside the configuration, with offsets of 25 000 000 and
    26 000 000 (``datasets/augment.py``); this key is the same device, declared.
    """


@dataclass(frozen=True)
class EvalDiscreteBand:
    """A sourced finite distribution used only to design an evaluation set."""

    values: tuple[float, ...]
    weights: tuple[float, ...]
    source: str


@dataclass(frozen=True)
class EvalCategoricalBand:
    """A sourced categorical distribution used only to design an evaluation set."""

    values: tuple[str, ...]
    weights: tuple[float, ...]
    source: str


@dataclass(frozen=True)
class EvalContinuousBand:
    """A sourced bounded continuous distribution used only for evaluation."""

    dist: str
    min: float
    max: float
    source: str
    median: float | None = None


@dataclass(frozen=True)
class BenchmarkHoldoutConfig:
    """Predeclared design and quality gate of the realistic benchmark holdout."""

    seed: int
    n_networks: int
    max_rejection_share: float
    families: tuple[str, ...]
    diameters_m: EvalDiscreteBand
    slopes_per_mille: EvalContinuousBand
    lengths_m: EvalContinuousBand
    roughness: EvalDiscreteBand
    categories: tuple[str, ...]
    outlet_cover: EvalCategoricalBand
    node_depth: EvalCategoricalBand
    rain_gages: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkHoldoutStratifiedConfig(BenchmarkHoldoutConfig):
    """Realistic holdout sampled until every rule-label class is represented."""

    min_rows_per_class: int


@dataclass(frozen=True)
class RuleConformanceConfig:
    """Predeclared target of the synthetic rule-conformance suite."""

    seed: int
    target_arcs: str
    min_rows_per_class: int
    n_realisations: int
    """Topology realisations of the balanced stage of the suite's own pool.

    The suite is generated, not selected: its files must not be the files the
    models were trained on, or ``conformance.<m>.accuracy`` would measure
    memorisation of the training set rather than conformance to the rule
    cascade (plan14 §5.2).
    """


@dataclass(frozen=True)
class EvalConfig:
    """The P7 evaluation artifacts; none participates in training or HPO."""

    benchmark_holdout: BenchmarkHoldoutConfig
    benchmark_holdout_stratified: BenchmarkHoldoutStratifiedConfig
    rule_conformance_suite: RuleConformanceConfig


@dataclass(frozen=True)
class GnnDiagEarlyStopping:
    """H3 of the instability diagnosis: hop counts, seeds and the epoch budget.

    ``epochs`` is its own key because the hypothesis is *about* the schedule:
    the run has to go past the point the production patience would have stopped
    it at, and a diagnosis that inherited ``gnn.epochs`` could not say so.
    """

    hops: tuple[int, ...]
    seeds: int
    epochs: int

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for the artifact's effective config."""
        return {"hops": list(self.hops), "seeds": self.seeds, "epochs": self.epochs}


@dataclass(frozen=True)
class GnnDiagDirection:
    """H4: how many paired downstream/upstream trainings, at which hop counts."""

    hops: tuple[int, ...]
    seeds: int

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for the artifact's effective config."""
        return {"hops": list(self.hops), "seeds": self.seeds}


@dataclass(frozen=True)
class GnnDiagLayerNormalization:
    """H5: the normalisers compared, at which hop counts and over how many seeds."""

    hops: tuple[int, ...]
    seeds: int
    norms: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for the artifact's effective config."""
        return {"hops": list(self.hops), "seeds": self.seeds, "norms": list(self.norms)}


@dataclass(frozen=True)
class GnnDiagnostics:
    """The instrumented rerun that diagnoses the seed-to-seed spread of study (b).

    ``seeds`` and ``hops`` size the shared grid H1 and H2 are read off; H3, H4 and
    H5 each name their own because each answers a different question and none of
    them needs the whole grid. The seeds themselves are *not* configured: they
    are ``seeds.tf + i``, the same set for every hop count, direction and
    normaliser, which is what makes every comparison in the diagnosis paired.
    """

    seeds: int
    hops: tuple[int, ...]
    h3: GnnDiagEarlyStopping
    h4: GnnDiagDirection
    h5: GnnDiagLayerNormalization

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for the artifact's effective config."""
        return {
            "seeds": self.seeds,
            "hops": list(self.hops),
            "h3": self.h3.as_dict(),
            "h4": self.h4.as_dict(),
            "h5": self.h5.as_dict(),
        }


@dataclass(frozen=True)
class GnnAblationScreening:
    """How wide and how deep the cheap stages S0-S2 of study (b2) run.

    One rotation of the fold roles and five seeds per configuration. That is
    **screening**, not evaluation, and the artifact says so in as many words: a
    single rotation scores one held-out fold, so its numbers rank configurations
    and are not the measurement any verdict rests on. S3 pays for the full
    rotation.
    """

    hops: int
    seeds: int
    rotations: int

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for the artifact's effective config."""
        return {"hops": self.hops, "seeds": self.seeds, "rotations": self.rotations}


@dataclass(frozen=True)
class GnnAblationArchitecture:
    """S0: the one-factor-at-a-time screen around ``legacy_b1``.

    Each list holds the values of one axis to try *instead of* the B1 value, never
    a product with the others: a full grid over four axes is 24 more trainings for
    an answer to a question nobody asked, and the deviations are only interpretable
    one at a time.

    ``tolerance`` and ``sigma_factor`` are the acceptance rule, declared before
    any of it runs: a value is taken when it costs at most ``tolerance`` of mean
    macro-F1 **and** either beats the base or cuts the spread to ``sigma_factor``
    of the base's.
    """

    hidden: tuple[int, ...]
    dropout: tuple[float, ...]
    weight_decay: tuple[float, ...]
    layer_norm: tuple[str, ...]
    tolerance: float
    sigma_factor: float

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for the artifact's effective config."""
        return {
            "hidden": list(self.hidden),
            "dropout": list(self.dropout),
            "weight_decay": list(self.weight_decay),
            "layer_norm": list(self.layer_norm),
            "tolerance": self.tolerance,
            "sigma_factor": self.sigma_factor,
        }


@dataclass(frozen=True)
class GnnAblationOperators:
    """S1: operators x directions x input normalisations, at the screening hop count.

    How many survive is exact, not a maximum, and it depends on **one** thing
    decided before S1 runs: whether S0 accepted a deviation. With one, S2 runs each
    survivor twice — with and without it — so three survivors make six candidates;
    without one there is a single arm, and four survivors are needed to leave the
    shortlist its declared length. Both counts are declared here, so the number is
    known in advance either way and only the branch is decided by S0.

    ``tolerance`` decides only which *rejection code* a configuration is filed
    under — below the threshold or merely outside the cut — so the number of
    configurations reaching S2 can never depend on how the scores happened to fall.
    """

    operators: tuple[str, ...]
    directions: tuple[str, ...]
    input_norms: tuple[str, ...]
    tolerance: float
    keep_with_deviations: int
    keep_without_deviations: int

    def keep(self, *, deviations: bool) -> int:
        """How many configurations reach S2, given whether S0 accepted anything."""
        return self.keep_with_deviations if deviations else self.keep_without_deviations

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for the artifact's effective config."""
        return {
            "operators": list(self.operators),
            "directions": list(self.directions),
            "input_norms": list(self.input_norms),
            "tolerance": self.tolerance,
            "keep_with_deviations": self.keep_with_deviations,
            "keep_without_deviations": self.keep_without_deviations,
        }


@dataclass(frozen=True)
class GnnAblationHops:
    """S2: the shortlist swept over hop counts, and how many positions reach S3."""

    hops: tuple[int, ...]
    shortlist: int

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for the artifact's effective config."""
        return {"hops": list(self.hops), "shortlist": self.shortlist}


@dataclass(frozen=True)
class GnnAblationEvaluation:
    """S3: the full rotation, the seeds it is repeated under, and the bootstrap size.

    The paired difference against the control is resampled over **seeds**, which
    is the independent unit here: the five fold scores of one seed share an
    initialisation, so treating all 100 cells as independent would report an
    interval several times too narrow.
    """

    seeds: int
    bootstrap: int

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for the artifact's effective config."""
        return {"seeds": self.seeds, "bootstrap": self.bootstrap}


@dataclass(frozen=True)
class GnnAblation:
    """The whole declared search space of study (b2), and every rule that closes it.

    The space is fixed **before** the first training and this object is what fixes
    it: it is written into ``config.json`` of the ablation artifact before a model
    exists, so "we kept looking until something worked" is a claim the artifact
    can refute rather than one the text has to deny.
    """

    folds: int
    screening: GnnAblationScreening
    s0: GnnAblationArchitecture
    s1: GnnAblationOperators
    s2: GnnAblationHops
    s3: GnnAblationEvaluation

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for the artifact's effective config."""
        return {
            "folds": self.folds,
            "screening": self.screening.as_dict(),
            "s0": self.s0.as_dict(),
            "s1": self.s1.as_dict(),
            "s2": self.s2.as_dict(),
            "s3": self.s3.as_dict(),
        }


@dataclass(frozen=True)
class GnnFinalStability:
    """V1: what "the training is repeatable" is allowed to mean.

    Both bounds, not one. A standard deviation alone is a summary that a single
    catastrophic seed can hide behind at ``n = 20``, and a range alone is the
    worst pair of twenty draws; the verdict asks for the spread *and* the extreme
    to be small, and both numbers are written into the artifact before the first
    of the twenty models exists.
    """

    accuracy_std_max: float
    accuracy_range_max: float

    def as_dict(self) -> dict[str, Any]:
        """The thresholds as a plain mapping, for the artifact and provenance."""
        return {"accuracy_std_max": self.accuracy_std_max, "accuracy_range_max": self.accuracy_range_max}


@dataclass(frozen=True)
class GnnFinalNonInferiority:
    """V2: the non-inferiority margin against the tabular model, as a count of errors.

    Declared in **errors on the test split** rather than as a rounded percentage:
    ``mlp_retrained_v1`` misses 7 of 1254 rows, so twice its error budget is a
    margin somebody can argue about on its own terms. The fraction it becomes is
    ``margin_errors / n_rows`` and is computed from the split the run actually
    reads, so a margin quoted here can never silently describe a different test
    set from the one measured.
    """

    margin_errors: int

    def as_dict(self) -> dict[str, Any]:
        """The threshold as a plain mapping, for the artifact and provenance."""
        return {"margin_errors": self.margin_errors}


@dataclass(frozen=True)
class GnnFinalMessagePassing:
    """V3: the smallest accuracy gain over the graph-free arm that counts as an effect."""

    minimum_effect: float

    def as_dict(self) -> dict[str, Any]:
        """The threshold as a plain mapping, for the artifact and provenance."""
        return {"minimum_effect": self.minimum_effect}


@dataclass(frozen=True)
class GnnFinalVerdicts:
    """The three verdicts' thresholds and the resampling they are decided under.

    They live in the configuration and not in the code of the final step for one
    reason: the artifact writes them out **before** the held-out split is read, so
    "the threshold was chosen after seeing the number" is a claim the run can
    refute rather than one the text has to deny.
    """

    stability: GnnFinalStability
    vs_mlp: GnnFinalNonInferiority
    message_passing: GnnFinalMessagePassing
    bootstrap: int
    level: float

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for the artifact and provenance."""
        return {
            "stability": self.stability.as_dict(),
            "vs_mlp": self.vs_mlp.as_dict(),
            "message_passing": self.message_passing.as_dict(),
            "bootstrap": self.bootstrap,
            "level": self.level,
        }


@dataclass(frozen=True)
class GnnFinal:
    """The final step of study (b2): retrain the ablation's winner and read the test once.

    Present only in the overlay of a variant that has an ablation to read, and
    absent from the base configuration. That absence is the switch: ``e_gnn.run``
    trains study (b)'s hop sweep when there is no ``gnn.final`` and hands the run
    to :mod:`stormlab.experiments.e_gnn_final` when there is, so variant B1's
    chain is not merely unchanged but takes literally the same path it always did.

    Attributes:
        winner_from: the ablation artifact's ``winner.json``, relative to the
            repository root. The spec is read from the artifact rather than
            copied into this file, because a spec typed twice is a spec that can
            disagree with the search that chose it.
        seeds: how many seeds each arm is retrained under.
        main_seed_index: which of them is the *reported* model — the one whose
            weights, probabilities and per-key figures are registered. An index
            rather than a seed value, so it cannot name a seed outside the set.
        nograph: whether the graph-free arm is trained. Only ``true`` is accepted
            — V3 is a paired difference against that arm, so a final step without
            it has no message-passing verdict — and the field exists so the
            artifact states what was trained rather than leaving it implied.
        baseline_legacy_k: hop count of the ``legacy_b1`` anchor trained beside
            the two arms. Its figures must come out bit for bit equal to variant
            B1's at the same k, which is what proves the new module trains the
            same way the old one did.
        serving_model_id: which registered model study (d2) serves under this
            variant. Named here rather than derived, because "which model is the
            chapter written about" is a decision and belongs where decisions are.
        verdicts: the three thresholds, declared before the measurement.
        select_hops_by_cv: whether ``k`` of the winner's spec is replaced by the
            hop count a ``gnn --mode tune`` pass chose on cross-validated folds of
            train and validation. ``false`` — B2's behaviour — takes the ``k`` the
            ablation froze into ``winner.json``. B4 sets it because the operator is
            carried over from another variant's ablation while the pool it is
            trained on is a different one: the architecture is inherited, the hop
            count is re-selected, and neither decision reads the test split.
        artifact_dir: where the write-once verdict declaration and the ``s4``
            tables are written, relative to the repository root. ``None`` — B2's
            behaviour — puts them beside the ablation the winner was read from,
            which is right only while the variant that runs the final step is the
            variant that ran the ablation. A variant reading somebody else's
            frozen winner must name a directory of its own, or it would write its
            declaration into a frozen artifact.
    """

    winner_from: str
    seeds: int
    main_seed_index: int
    nograph: bool
    baseline_legacy_k: int
    serving_model_id: str
    verdicts: GnnFinalVerdicts
    select_hops_by_cv: bool = False
    artifact_dir: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for provenance records."""
        return {
            "winner_from": self.winner_from,
            "seeds": self.seeds,
            "main_seed_index": self.main_seed_index,
            "nograph": self.nograph,
            "baseline_legacy_k": self.baseline_legacy_k,
            "serving_model_id": self.serving_model_id,
            "verdicts": self.verdicts.as_dict(),
            "select_hops_by_cv": self.select_hops_by_cv,
            "artifact_dir": self.artifact_dir,
        }


@dataclass(frozen=True)
class GnnCrossValidation:
    """The grouped K-fold that selects a hop count without reading the test split.

    ``gnn.k_selected.by_mean_accuracy`` is chosen by the seed sweep's **test**
    accuracy, so the 114-file holdout has already informed a decision. This
    section sizes the selection that does not: folds cut out of train and
    validation together, grouped by source file, scored by macro-F1 over the seven
    classes the dataset actually contains.

    ``folds: 0`` turns it off, the way ``seed_sweep: 0`` turns the sweep off.
    """

    folds: int
    seeds: int

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for provenance records."""
        return {"folds": self.folds, "seeds": self.seeds}


@dataclass(frozen=True)
class GnnExperiment:
    """Every hop count, seed count and training hyper-parameter of study (b).

    ``k_serving`` and ``report_hops`` are separate from ``hops`` on purpose: what
    the sweep measures and what the serving path answers under are two decisions,
    and the gap between them is one of the findings rather than an oversight.
    """

    hops: tuple[int, ...]
    k_serving: int
    seed_sweep: int
    epochs: int
    lr: float
    patience: int
    upstream_variant_hops: int
    report_hops: int
    cv: GnnCrossValidation
    diag: GnnDiagnostics
    ablation: GnnAblation
    final: GnnFinal | None = None
    """The final step, when this variant has one; ``None`` is study (b) as B1 ran it."""

    def as_dict(self) -> dict[str, Any]:
        """The section as a plain mapping, for provenance records."""
        return {
            "hops": list(self.hops),
            "k_serving": self.k_serving,
            "seed_sweep": self.seed_sweep,
            "epochs": self.epochs,
            "lr": self.lr,
            "patience": self.patience,
            "upstream_variant_hops": self.upstream_variant_hops,
            "report_hops": self.report_hops,
            "cv": self.cv.as_dict(),
            "diag": self.diag.as_dict(),
            "ablation": self.ablation.as_dict(),
            "final": self.final.as_dict() if self.final is not None else None,
        }


@dataclass(frozen=True)
class C1Simulation:
    """The base simulation series C1 reruns the B4 pool under.

    Every field is a SWMM option or a gate on its output, written down here so
    that "the same options as B4" is a checkable claim rather than a sentence in a
    plan (plan16 §2 P2).

    ``continuity_max_abs_percent`` is always **measured**; the two booleans decide
    what is done with the measurement. Under the author's decision of 2026-09-07 the
    mass balance no longer removes a case from the population
    (``reject_on_continuity = False``), because a pool split by a gate nobody applied
    to B4 is not the B4 pool; the balance survives as a figure of the survivability
    report. With nothing to be rescued from, the second attempt at the shorter step
    loses its purpose too (``retry_on_continuity = False``) — and it is worth
    switching off in its own right, since a case computed at another step is another
    simulation and its historical columns cannot be bit-equal to B4's.

    ``retry_routing_step`` stays declared: it is the step the second attempt uses
    where the retry is switched on, and a protocol that named the switch without the
    step would be describing half a behaviour.

    ``rewrite_options`` names the subset of :data:`SWMM_PROTOCOL_OPTIONS` a protocol
    asks the case stage to *write into its working copies* instead of requiring the
    pool files to carry already. Empty is the historical behaviour and the default:
    the pool is the one the reference measured, and an INP edited into agreement
    would move the parity gate silently. A protocol that reruns the same pool under
    another routing scheme cannot state that requirement -- the files say ``KINWAVE``
    -- so it declares the difference here, and every change lands in the run's
    ``options_manifest.json``. The key may be absent, which is what keeps a
    configuration written before the field existed loadable unchanged.
    """

    flow_routing: str
    link_offsets: str
    routing_step: str
    retry_routing_step: str
    continuity_max_abs_percent: float
    reject_on_continuity: bool
    retry_on_continuity: bool
    keep_rpt_out: bool
    pilot_families: int
    rewrite_options: tuple[str, ...] = ()
    routing_dependent_columns: tuple[str, ...] = ()
    """Reference feature columns the routing is allowed to move (plan17 §Etap 3).

    The parity gate of ``c1-build-cases`` compares the rebuilt historical view with the
    frozen reference column by column and exactly. A protocol that simulates the same
    pool under another routing cannot hold every hydraulic column to that standard, and
    would not want to: the columns that *do* stay equal are the evidence that nothing
    but the routing changed. So the protocol names the columns the routing is expected
    to move; the gate holds the rest exactly, and reports these in a section of their
    own, measured not gated. Empty for C1, where the gate covers every column."""

    def as_dict(self) -> dict[str, object]:
        return {
            "flow_routing": self.flow_routing,
            "link_offsets": self.link_offsets,
            "routing_step": self.routing_step,
            "retry_routing_step": self.retry_routing_step,
            "continuity_max_abs_percent": self.continuity_max_abs_percent,
            "reject_on_continuity": self.reject_on_continuity,
            "retry_on_continuity": self.retry_on_continuity,
            "keep_rpt_out": self.keep_rpt_out,
            "pilot_families": self.pilot_families,
            "rewrite_options": list(self.rewrite_options),
            "routing_dependent_columns": list(self.routing_dependent_columns),
        }


@dataclass(frozen=True)
class C1Contracts:
    """The four feature views of the experiment matrix, named before they exist.

    ``F0``/``F1`` are the historical contract and the historical contract without
    the twelve flags; ``F2``/``F3`` are the physical view and its requirement
    margins. Held as plain strings and **not** resolved against
    :data:`stormlab.contracts.features.FEATURE_CONTRACTS`: the last two are
    introduced by a later stage of plan16 §5.2, and a configuration that refused
    to load until they existed would make the protocol undeclarable before the
    thing it declares was built.

    ``F2_nbr`` is the exception, and it is optional. It names the fifth view --
    ``F2`` followed by the generated neighbourhood block -- and it **is** resolved,
    because ``c1.neighborhood`` is checked against the measurements that contract
    was generated from and there is no longer one such list in the codebase: C1
    aggregates the hours a conduit's capacity limited the flow, C2 the flooding of
    its inlet node. A protocol that says nothing keeps C1's
    :data:`~stormlab.contracts.features.NEIGHBORHOOD_CONTRACT_NAME`.
    """

    f0: str
    f1: str
    f2: str
    f3: str
    f2_nbr: str = NEIGHBORHOOD_CONTRACT_NAME

    def as_dict(self) -> dict[str, str]:
        return {"F0": self.f0, "F1": self.f1, "F2": self.f2, "F3": self.f3, "F2_nbr": self.f2_nbr}


@dataclass(frozen=True)
class C1SplitFractions:
    """How the families are divided; the three add up to one."""

    train: float
    val: float
    test: float

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass(frozen=True)
class C1Split:
    """The ``family_v3`` partition and the gate its leakage audit has to pass.

    ``leakage_seen_fraction_warn`` is not a pass/fail threshold on the split: it is
    the share of test conduits whose F2 vector was already seen in training, above
    which the report marks the result on that part as a possible memorisation
    effect (plan16 §2 P5a).
    """

    protocol: str
    fractions: C1SplitFractions
    seed: int
    leakage_seen_fraction_warn: float

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "fractions": self.fractions.as_dict(),
            "seed": self.seed,
            "leakage_seen_fraction_warn": self.leakage_seen_fraction_warn,
        }


@dataclass(frozen=True)
class C1ReferenceMargins:
    """Reference scales of the requirement margins, one per requirement family.

    A margin is an exceedance divided by the requirement it is measured against,
    so every one of these is a strictly positive and **dimensionless**
    denominator. The margins of the F3 view are relative throughout
    (``(C - C_req)/C_req``, ``(d_max - d)/d_max``, ``(S - S_min)/S_min``), so a
    reference given in metres or per mille would not divide into them; what a
    value means physically depends on the requirement of the row it is applied to.
    """

    filling: float
    velocity: float
    slope: float
    cover: float
    depth: float

    def as_dict(self) -> dict[str, float]:
        return {
            "filling": self.filling,
            "velocity": self.velocity,
            "slope": self.slope,
            "cover": self.cover,
            "depth": self.depth,
        }


@dataclass(frozen=True)
class C1SupportSplits:
    """How each source spreads its support when one violation supports several actions.

    plan16 §2 P4 allows a source to assign one violation to more than one action
    "z jawnie zapisanym podziałem poparcia", and calls every such split an
    engineering premise approved in D7. The split therefore lives in the frozen
    protocol rather than in the source, so that changing it is visibly a change of
    the protocol and not a change of code.

    Attributes:
        splits: source id to action-to-share mapping. Every inner mapping names
            classes of ``RECOMMENDATION_CLASS_ORDER`` and sums to 1.
    """

    splits: Mapping[str, Mapping[str, float]]

    def __getitem__(self, source_id: str) -> Mapping[str, float]:
        try:
            return self.splits[source_id]
        except KeyError as error:
            raise ConfigError(
                f"No support split declared for source {source_id!r}; the protocol declares {sorted(self.splits)}"
            ) from error

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {source_id: dict(shares) for source_id, shares in self.splits.items()}


@dataclass(frozen=True)
class C1Sources:
    """How the label sources of plan16 §2 P4 are weighed against each other.

    ``category_role`` is a constraint rather than a setting: the land-cover
    category may enter a source as an infiltration premise and may never enter one
    as a cost preference (plan16 §0 change 6), and writing the permitted role into
    the protocol is what makes the prohibition checkable.

    The four ``*_position*`` and ``*_ratio*`` values are the historical cascade's
    own thresholds, carried over unchanged so that a C1 source expresses the
    premise the cascade already held rather than a newly chosen one. Each source
    recomputes the position it compares against from F2 columns; none of them
    reads a v1/v2 normalisation.

    ``infiltration_imperv_max_pct`` and ``retention_imperv_min_pct`` are the two
    thresholds that are *not* the cascade's: they replace the land cover class in the
    two catchment premises with the measured imperviousness of the catchment
    (author's decision 2026-09-07), so both rest on how much of the ground can take
    water rather than on a name drawn independently of it. They are separated by a
    band in which neither premise holds, which is why the loader refuses an overlap:
    a catchment that is at once permeable enough to infiltrate into and sealed enough
    to need retention would hand one row two opposite answers from one measurement.

    The last six fields carry defaults, and the defaults are C1's behaviour written
    down rather than a convenience: ``overload_premise`` defaults to the filling
    premise the series was frozen on, and the five numbers of the flooding premise
    default to zero, i.e. no de-minimis band, no support floor and no support for a
    pressurised conduit. A protocol that declares none of them is C1 to the byte,
    which is what lets ``stormlab/config.C1.yaml`` stay untouched while variant C2
    declares the other premise (plan17 §Poza zakresem).
    """

    m_ref: C1ReferenceMargins
    group_weights: str
    category_role: str
    infiltration_imperv_max_pct: float
    retention_imperv_min_pct: float
    hours_ref: float
    oversized_min_steps: int
    oversized_steps_ref: int
    shallow_position_max: float
    deepen_position_max: float
    low_slope_position_max: float
    high_flow_ratio_min: float
    support_splits: C1SupportSplits
    overload_premise: str = C1_OVERLOAD_PREMISE_FILLING_HOURS
    flood_hours_ref: float = 1.0
    flood_hours_deminimis: float = 0.0
    flood_volume_deminimis_m3: float = 0.0
    flood_deminimis_support_min: float = 0.0
    pressurised_flow_support_max: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "m_ref": self.m_ref.as_dict(),
            "group_weights": self.group_weights,
            "category_role": self.category_role,
            "infiltration_imperv_max_pct": self.infiltration_imperv_max_pct,
            "retention_imperv_min_pct": self.retention_imperv_min_pct,
            "hours_ref": self.hours_ref,
            "oversized_min_steps": self.oversized_min_steps,
            "oversized_steps_ref": self.oversized_steps_ref,
            "shallow_position_max": self.shallow_position_max,
            "deepen_position_max": self.deepen_position_max,
            "low_slope_position_max": self.low_slope_position_max,
            "high_flow_ratio_min": self.high_flow_ratio_min,
            "support_splits": self.support_splits.as_dict(),
            "overload_premise": self.overload_premise,
            "flood_hours_ref": self.flood_hours_ref,
            "flood_hours_deminimis": self.flood_hours_deminimis,
            "flood_volume_deminimis_m3": self.flood_volume_deminimis_m3,
            "flood_deminimis_support_min": self.flood_deminimis_support_min,
            "pressurised_flow_support_max": self.pressurised_flow_support_max,
        }


@dataclass(frozen=True)
class C1Targets:
    """The training targets: their contract, the admissibility threshold and the class gate.

    ``admissible_tau`` is fixed before the test is opened because the top-two
    metric is defined against it (plan16 §4): a class counts as admissible when its
    L2 support reaches the threshold, and a threshold chosen after seeing the
    distribution would be a metric chosen after seeing the result.
    """

    contract: str
    admissible_tau: float
    min_class_train_l1: int

    def as_dict(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "admissible_tau": self.admissible_tau,
            "min_class_train_l1": self.min_class_train_l1,
        }


@dataclass(frozen=True)
class C1TrainingGrid:
    """The eight configurations both models are tuned over: 2 widths x 2 dropouts x 2 rates."""

    width: tuple[int, ...]
    dropout: tuple[float, ...]
    learning_rate: tuple[float, ...]

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "width": list(self.width),
            "dropout": list(self.dropout),
            "learning_rate": list(self.learning_rate),
        }


@dataclass(frozen=True)
class C1Training:
    """One training regime for the MLP and the GNN alike (plan16 §3).

    ``class_weights`` and ``sample_weighting`` are separate on purpose: the series
    forbids class weights derived from the argmax of a soft target and instead
    weighs *examples*, so that a family counts once however many cases and
    conduits it contributes.
    """

    grid: C1TrainingGrid
    hidden_layers: int
    max_epochs: int
    patience: int
    cv_folds: int
    seeds: tuple[int, ...]
    class_weights: str
    sample_weighting: str

    def as_dict(self) -> dict[str, object]:
        return {
            "grid": self.grid.as_dict(),
            "hidden_layers": self.hidden_layers,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "cv_folds": self.cv_folds,
            "seeds": list(self.seeds),
            "class_weights": self.class_weights,
            "sample_weighting": self.sample_weighting,
        }


@dataclass(frozen=True)
class C1Neighborhood:
    """Which raw measurements the tabular models see summarised over the graph.

    The aggregates are the tabular arm's answer to the graph models: a conduit is
    given the minimum, maximum and mean of these columns over its upstream and
    downstream neighbourhood, so the comparison of plan16 §3 (E6) is between a
    model that passes messages and a model that was handed the same neighbourhood
    as columns -- rather than between a model with context and one without any.

    They are read from the *measured* columns of the physical view and never from
    a label or a decision flag: an aggregate over the neighbours' labels would
    hand a model the answer its neighbours were given.
    """

    columns: tuple[str, ...]
    hops: tuple[int, ...]

    def as_dict(self) -> dict[str, list[object]]:
        return {"columns": list(self.columns), "hops": list(self.hops)}


@dataclass(frozen=True)
class C1Gnn:
    """The receptive field and the direction arms of E6, declared before the search."""

    hops: tuple[int, ...]
    directions: tuple[str, ...]

    def as_dict(self) -> dict[str, list[object]]:
        return {"hops": list(self.hops), "directions": list(self.directions)}


@dataclass(frozen=True)
class C1Motif:
    """One conflict motif and the set it belongs to.

    The assignment is fixed before training and stored here rather than derived,
    because the whole point of :data:`C1_MOTIF_SETS` is that ``held`` motifs are
    outside the loss: a split computed at training time could be recomputed at
    reporting time and nobody would see the difference.
    """

    identifier: str
    set_name: str


@dataclass(frozen=True)
class C1ContextLoss:
    """The additional relational supervision of E7 and the motifs it is measured on.

    ``lambdas`` is spelled in the plural because the YAML key is ``lambda``, which
    is a Python keyword; it holds the arms of the comparison (``0`` and the
    penalised value), not a single coefficient.
    """

    lambdas: tuple[float, ...]
    motifs: tuple[C1Motif, ...]

    def motifs_in(self, set_name: str) -> tuple[str, ...]:
        """Identifiers of the motifs assigned to ``set_name``, in declaration order."""
        return tuple(motif.identifier for motif in self.motifs if motif.set_name == set_name)

    def as_dict(self) -> dict[str, object]:
        return {
            "lambda": list(self.lambdas),
            "motifs": {motif.identifier: motif.set_name for motif in self.motifs},
        }


@dataclass(frozen=True)
class C1EffectMargins:
    """Effect sizes pre-declared before the test is opened (plan16 §4).

    Signs are part of the declaration and are not normalised away: a Brier delta
    that has to *fall* is a negative maximum, a top-1 gap that has to be *reached*
    is a positive minimum, and a conflict frequency that has to fall by a fifth is
    a negative relative change.
    """

    e4_brier_delta_max: float
    e6_context_top1_pp_min: float
    e6_architecture_top1_pp_min: float
    e7_held_conflict_rel_change_max: float
    e7_brier_delta_max: float

    def as_dict(self) -> dict[str, float]:
        return {
            "E4_brier_delta_max": self.e4_brier_delta_max,
            "E6_context_top1_pp_min": self.e6_context_top1_pp_min,
            "E6_architecture_top1_pp_min": self.e6_architecture_top1_pp_min,
            "E7_held_conflict_rel_change_max": self.e7_held_conflict_rel_change_max,
            "E7_brier_delta_max": self.e7_brier_delta_max,
        }


@dataclass(frozen=True)
class C1Bootstrap:
    """The resampling behind every confidence interval of the series.

    ``unit`` names what is resampled. Conduits of one network are not independent
    observations, so the unit is the lineage the network belongs to; stating it
    here keeps the interval and the population it describes together.
    """

    unit: str
    n_resamples: int
    seed: int
    ci: float

    def as_dict(self) -> dict[str, object]:
        return {"unit": self.unit, "n_resamples": self.n_resamples, "seed": self.seed, "ci": self.ci}


@dataclass(frozen=True)
class C1AuthorReview:
    """Sizes of the three author-review rounds and of the repeatability retest.

    A qualitative study with declared sizes, not a statistical test of one model
    against another (plan16 §0 change 8): the counts are here so that the report
    can say what was reviewed before it says what the review found.
    """

    dev_chains: int
    dev_min_families: int
    val_chains: int
    val_min_families: int
    test_chains: int
    test_min_families: int
    retest_chains: int
    retest_min_days: int

    def as_dict(self) -> dict[str, int]:
        return {
            "dev_chains": self.dev_chains,
            "dev_min_families": self.dev_min_families,
            "val_chains": self.val_chains,
            "val_min_families": self.val_min_families,
            "test_chains": self.test_chains,
            "test_min_families": self.test_min_families,
            "retest_chains": self.retest_chains,
            "retest_min_days": self.retest_min_days,
        }


@dataclass(frozen=True)
class C1Protocol:
    """The frozen research protocol of series C1 (plan16 §2 P0).

    Present only in the overlay of the variant that declares it; ``None`` for
    B1--B4, whose protocols were never written down in this form and whose frozen
    results this section must not be able to move. It reaches no feature value and
    no hydraulic constant: it is a declaration of how the series will be run,
    which the ``c1-freeze-protocol`` command hashes and writes once.

    Paths are resolved against the repository root at load time, exactly as
    ``paths`` and ``expected.dataset.manifest`` are.
    """

    dataset_run_reference: str
    pool_manifest: Path
    pool_dirs: tuple[Path, ...]
    dev_material: tuple[str, ...]
    """Networks already used while designing the method, so they are development
    material and never evidence (plan16 §2 P0 point 6)."""
    simulation: C1Simulation
    contracts: C1Contracts
    split: C1Split
    sources: C1Sources
    targets: C1Targets
    training: C1Training
    neighborhood: C1Neighborhood
    gnn: C1Gnn
    context_loss: C1ContextLoss
    effect_margins: C1EffectMargins
    bootstrap: C1Bootstrap
    author_review: C1AuthorReview

    def as_dict(self, *, root: Path | None = None) -> dict[str, object]:
        """The protocol as a JSON-safe mapping.

        Args:
            root: when given, the two path fields are written relative to it, so
                the same protocol hashes the same on two machines.
        """

        def render(path: Path) -> str:
            if root is None:
                return str(path)
            try:
                return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
            except ValueError:
                return str(path)

        return {
            "dataset_run_reference": self.dataset_run_reference,
            "pool_manifest": render(self.pool_manifest),
            "pool_dirs": [render(item) for item in self.pool_dirs],
            "dev_material": list(self.dev_material),
            "simulation": self.simulation.as_dict(),
            "contracts": self.contracts.as_dict(),
            "split": self.split.as_dict(),
            "sources": self.sources.as_dict(),
            "targets": self.targets.as_dict(),
            "training": self.training.as_dict(),
            "neighborhood": self.neighborhood.as_dict(),
            "gnn": self.gnn.as_dict(),
            "context_loss": self.context_loss.as_dict(),
            "effect_margins": self.effect_margins.as_dict(),
            "bootstrap": self.bootstrap.as_dict(),
            "author_review": self.author_review.as_dict(),
        }


@dataclass(frozen=True)
class Paths:
    """Absolute, resolved filesystem locations."""

    dataset_dir: Path
    catchment_dir: Path
    catchment_ood: Path
    wroclaw_inp: Path
    results_dir: Path
    work_dir: Path


@dataclass(frozen=True)
class Config:
    """Fully resolved and validated run configuration."""

    source: Path
    root: Path
    variant: str
    """Which frozen result set of path B this run belongs to (``B1``, ``B2``, …).

    It reaches every number the run registers and every provenance record it
    writes, so a registry can say which variant produced it without the reader
    having to know which run ids were current when it was written."""
    paths: Paths
    wroclaw_raingage: str
    frost_zone_dataset: float
    frost_zone_serving: float
    required_cover: float | None
    """Soil cover a conduit must carry above its crown [m]; ``None`` means "the frost zone".

    Three requirements the historical code held as two: the frost zone is a
    property of the climate, the required cover is a design rule that happened to
    equal it, and :attr:`max_depth` was a module literal (plan16 §2 P0 point 5).
    Separating them is what lets series C1 move one without moving the others.

    ``None`` rather than a repetition of the frost zone, because the frost zone
    reaches the extraction as **two** values -- ``frost_zone_dataset`` when the
    pool is built, ``frost_zone_serving`` when Wroclaw is scored -- and a single
    number here could only mirror one of them. ``None`` mirrors whichever is in
    force; :func:`stormlab.features.conduits.resolve_required_cover` is the one
    place that resolves it, and the extraction manifest records the resolved
    value rather than the null."""
    max_depth: float
    """Deepest admissible pipe invert below ground [m].

    The upper end of the depth and cover normalisations and the threshold behind
    ``ValDepth``. It was ``hydraulics.round.MAX_DEPTH``, a module literal, and a
    second copy of it lived in the leakage audit; that made a change to the
    accepted burial depth an edit in two files rather than a configuration
    change and a rerun."""
    seeds: Seeds
    flow_constant: float
    flow_law: str
    nmaxq_clip: bool
    slope_profile: SlopeProfile
    khop_scheme: str
    surrogate_edges: bool
    conduit_order: str
    graph_direction: str
    max_filling_ratio: float
    hydraulics: Hydraulics
    gnn: GnnExperiment
    mlp_normalization: str
    expected: Expected
    split: SplitConfig
    feature_contract: FeatureContract
    tabular: TabularConfig = field(default_factory=TabularConfig)
    wroclaw: WroclawConfig = field(default_factory=WroclawConfig)
    dataset_run_id: str | None = None
    """The ``build-dataset`` run the experiments read; ``None`` until one is named."""
    generate: Generate | None = None
    """Generator configuration; ``None`` when the YAML has no ``generate:`` section."""
    eval: EvalConfig | None = None
    """Evaluation-set designs; absent before P7 and mandatory for its two CLI commands."""
    c1: C1Protocol | None = None
    """The frozen research protocol of series C1; ``None`` for every other variant.

    Optional in the strong sense: B1--B4 declare no such section and must keep
    loading byte-identically, so the absence is the normal state rather than a
    gap to be filled in."""
    catchment: CatchmentConfig | None = None
    """What the catchment study trains and compares; ``None`` restores its historical names."""
    catchment_dataset: CatchmentDatasetProfile | CatchmentClassTableProfile | None = None
    """The subcatchment-sampling profile; ``None`` means the variant has no such dataset to build.

    Which of the two it is follows from ``catchment_dataset.generator``, and the type
    is what the generator dispatches on -- a variant cannot half-declare one sampler
    and half the other, because the two arms name disjoint key sets."""
    overlay: Path | None = None
    """The variant overlay this configuration was merged with, if any.

    Recorded rather than inferred: a run under ``config.B2.yaml`` and a run under
    the base file can differ in every hyper-parameter, and the provenance record
    has to be able to say which one produced the numbers."""

    @property
    def v_min(self) -> float:
        """Self-cleansing minimum velocity [m/s]; stored once, under ``hydraulics``."""
        return self.hydraulics.v_min

    @property
    def provenance_root(self) -> Path:
        """Directory holding ``<run_id>/<step>.json`` provenance records."""
        return self.paths.results_dir / "runs" / "provenance"

    def run_dir(self, run_id: str) -> Path:
        """Result directory of a single run."""
        return self.paths.results_dir / "runs" / run_id

    def work_run_dir(self, run_id: str) -> Path:
        """Scratch directory holding the INP copies of a single run."""
        return self.paths.work_dir / run_id


def default_config_path() -> Path:
    """Path of the config file shipped with the package."""
    return Path(__file__).resolve().parent / DEFAULT_CONFIG_FILENAME


def overlay_path(config_path: Path | str, variant: str) -> Path:
    """Where a variant's overlay sits beside the base config: ``config.B2.yaml``."""
    base = Path(config_path)
    return base.with_name(CONFIG_OVERLAY_TEMPLATE.format(stem=base.stem, variant=variant, suffix=base.suffix))


def load_config(
    path: Path | str | None = None,
    *,
    root: Path | None = None,
    check_inputs: bool = True,
    variant: str | None = None,
) -> Config:
    """Read, validate and resolve the configuration, optionally under a variant overlay.

    Args:
        path: config file; defaults to the one shipped with the package.
        root: repository root used to resolve relative paths; defaults to the
            parent of the package.
        check_inputs: verify that the configured input paths exist. Turn off
            only for tests that do not touch the datasets.
        variant: which frozen result set to configure for. ``None`` — and the
            variant the base file already names — load the base file alone.
            Anything else loads ``config.<variant>.yaml`` on top of it: a file
            holding **only the differences**, merged mapping by mapping, with a
            list replacing its counterpart whole rather than being appended to.
            A list of hop counts is a decision, not an accumulation.

    Raises:
        ConfigError: on unknown/missing keys, values outside their domain, a
            variant name that is not an identifier, a missing overlay, or an
            overlay that names a different variant than the one requested. A
            missing overlay is an error rather than a silent fall-back to the base
            file: a run asked for B2 and getting B1's configuration would publish
            B1's numbers under B2's name.
        MissingInputsError: when ``check_inputs`` is on and inputs are absent.
    """
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Config {config_path} must contain a mapping, got {type(raw).__name__}")

    # `--config stormlab/config.B4.yaml --variant B4`: the file named on the
    # command line is the *overlay*, not the base, so the base is merged **under**
    # it rather than the overlay being merged over the base. Documented here
    # because it was never listed among the F1-F11/G1-G5 corrections and a reader
    # of `_apply_overlay` would not find it (P8-FIX3b J4, api_map §22.11).
    #
    # Two properties worth knowing before relying on it:
    #
    # * It fires only on an **incomplete** overlay. A variant file that happened
    #   to name every key in `_TOP_LEVEL_KEYS` would take the `_apply_overlay`
    #   branch instead and could resolve differently — the trigger is the shape of
    #   the file, not the intent of the caller.
    # * It does **not** check `variant` against `VARIANT_NAME_PATTERN`, which
    #   `_apply_overlay` does. Here the requested variant has to equal the string
    #   the file itself declares, which is the binding that matters for publishing
    #   one variant's numbers under another's name; the pattern is a separate,
    #   weaker check that this path skips.
    #
    # Left as it behaves. This comment is the declaration, not a change.
    missing_base_keys = _TOP_LEVEL_KEYS - set(raw)
    direct_overlay = (
        bool(missing_base_keys)
        and variant is not None
        and raw.get("variant") == variant
        and config_path.name != DEFAULT_CONFIG_FILENAME
    )
    if direct_overlay:
        base_path = config_path.with_name(DEFAULT_CONFIG_FILENAME)
        if not base_path.is_file():
            raise ConfigError(
                f"Config {config_path} is an overlay (missing base keys {sorted(missing_base_keys)}) but its "
                f"base {base_path} does not exist"
            )
        base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if not isinstance(base_raw, Mapping):
            raise ConfigError(f"Base config {base_path} must contain a mapping, got {type(base_raw).__name__}")
        merged = _deep_merge(base_raw, raw)
        merged["variant"] = variant
        overlay: Path | None = config_path
    else:
        merged, overlay = _apply_overlay(raw, config_path=config_path, variant=variant)
    base = (root or repo_root()).resolve()
    config = _build(merged, source=config_path, base=base, overlay=overlay)
    if check_inputs:
        verify_inputs(config)
    return config


def _apply_overlay(raw: Mapping[str, Any], *, config_path: Path, variant: str | None) -> tuple[Mapping[str, Any], Path | None]:
    """The base configuration, with a variant overlay merged over it when one applies."""
    if variant is None:
        return raw, None
    requested = str(variant).strip()
    if not VARIANT_NAME_PATTERN.match(requested):
        raise ConfigError(f"Variant {variant!r} must match {VARIANT_NAME_PATTERN.pattern}")
    if requested == raw.get("variant"):
        return raw, None

    candidate = overlay_path(config_path, requested)
    if not candidate.is_file():
        raise ConfigError(
            f"Variant {requested!r} differs from the {raw.get('variant')!r} of {config_path}, so it needs an overlay "
            f"at {candidate}, which does not exist. An overlay holds only the keys the variant changes"
        )
    overlay = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(overlay, Mapping):
        raise ConfigError(f"Config overlay {candidate} must contain a mapping, got {type(overlay).__name__}")
    declared = overlay.get("variant")
    if declared is not None and declared != requested:
        raise ConfigError(f"Config overlay {candidate} names variant {declared!r} but was loaded for {requested!r}")
    merged = _deep_merge(raw, overlay)
    merged["variant"] = requested
    return merged, candidate


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """``overlay`` over ``base``: mappings merge key by key, everything else replaces.

    A list replaces rather than extends because every list in this configuration
    is a closed decision — ``gnn.hops`` is the set of hop counts the study trains,
    not a set it adds to — and an overlay that could only ever grow such a list
    could not express "train two of them".
    """
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if key == "dataset" and isinstance(value, Mapping) and set(value) == {"manifest"}:
            # ``expected.dataset`` is a tagged union. An overlay selecting its
            # manifest arm replaces the legacy count arm rather than merging
            # the two mutually exclusive shapes.
            merged[key] = dict(value)
        else:
            merged[key] = _deep_merge(current, value) if isinstance(current, Mapping) and isinstance(value, Mapping) else value
    return merged


def verify_inputs(config: Config, *, produced: Collection[str] = ()) -> None:
    """Check that every configured input exists.

    Args:
        produced: path keys this command *writes* rather than reads. ``generate``
            populates ``dataset_dir`` for a generated variant, so demanding that
            it already exist would mean such a variant could never be produced a
            first time.

    Raises:
        MissingInputsError: listing every missing path at once.
        ConfigError: if ``produced`` names a key that is not a configured input.
    """
    known = set(_INPUT_DIR_KEYS) | set(_INPUT_FILE_KEYS)
    unknown = sorted(set(produced) - known)
    if unknown:
        raise ConfigError(f"verify_inputs was told {unknown} are produced, but they are not configured inputs")
    missing: list[str] = []
    for key in _INPUT_DIR_KEYS:
        if key in produced:
            continue
        candidate = getattr(config.paths, key)
        if not candidate.is_dir():
            missing.append(f"{key}: {candidate} (expected a directory)")
    for key in _INPUT_FILE_KEYS:
        if key in produced:
            continue
        candidate = getattr(config.paths, key)
        if not candidate.is_file():
            missing.append(f"{key}: {candidate} (expected a file)")
    if missing:
        raise MissingInputsError(
            f"Configured inputs are missing (config {config.source}, root {config.root}):\n  " + "\n  ".join(missing)
        )


def _build(raw: Mapping[str, Any], *, source: Path, base: Path, overlay: Path | None = None) -> Config:
    unknown = set(raw) - _ALL_TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"Unknown config keys in {source}: {sorted(unknown)}; allowed: {sorted(_ALL_TOP_LEVEL_KEYS)}")
    missing = _TOP_LEVEL_KEYS - set(raw)
    if missing:
        raise ConfigError(f"Missing config keys in {source}: {sorted(missing)}")

    paths = _build_paths(raw["paths"], source=source, base=base)
    seeds = _build_seeds(raw["seeds"], source=source)
    slope_profile = _build_slope_profile(raw["slope_profile"], source=source)

    frost_dataset = _as_float(raw, "frost_zone_dataset", source)
    frost_serving = _as_float(raw, "frost_zone_serving", source)
    for name, value in (("frost_zone_dataset", frost_dataset), ("frost_zone_serving", frost_serving)):
        low, high = FROST_ZONE_RANGE
        if not low <= value <= high:
            raise ConfigError(f"{name} must be within [{low}, {high}] m, got {value} (config {source})")

    max_depth = _as_float(raw, "max_depth", source)
    if max_depth <= 0.0:
        raise ConfigError(f"max_depth must be positive, got {max_depth} m (config {source})")

    required_cover = raw["required_cover"]
    if required_cover is not None:
        required_cover = _as_float(raw, "required_cover", source)
        # Not FROST_ZONE_RANGE: a cover requirement is a design rule, not a
        # climate measurement, and bounding it by the frost zone's range would
        # forbid exactly the separation this key exists to make possible.
        if required_cover <= 0.0:
            raise ConfigError(f"required_cover must be positive when given, got {required_cover} m (config {source})")
        if required_cover >= max_depth:
            raise ConfigError(
                f"required_cover {required_cover} m is not below max_depth {max_depth} m (config {source}); the "
                "admissible burial band would be empty"
            )

    max_filling_ratio = _as_float(raw, "max_filling_ratio", source)
    if not 0.0 < max_filling_ratio <= 1.0:
        raise ConfigError(f"max_filling_ratio must be in (0, 1], got {max_filling_ratio} (config {source})")

    flow_constant = _as_float(raw, "flow_constant", source)
    if flow_constant <= 0.0:
        raise ConfigError(f"flow_constant must be positive, got {flow_constant} (config {source})")

    raingage = raw["wroclaw_raingage"]
    if not isinstance(raingage, str) or not raingage.strip():
        raise ConfigError(f"wroclaw_raingage must be a non-empty string, got {raingage!r} (config {source})")

    surrogate_edges = raw["surrogate_edges"]
    if not isinstance(surrogate_edges, bool):
        raise ConfigError(f"surrogate_edges must be a boolean, got {surrogate_edges!r} (config {source})")

    nmaxq_clip = raw["nmaxq_clip"]
    if not isinstance(nmaxq_clip, bool):
        raise ConfigError(f"nmaxq_clip must be a boolean, got {nmaxq_clip!r} (config {source})")

    return Config(
        source=source,
        root=base,
        variant=_as_variant(raw, "variant", source),
        overlay=overlay,
        paths=paths,
        wroclaw_raingage=raingage,
        frost_zone_dataset=frost_dataset,
        frost_zone_serving=frost_serving,
        required_cover=required_cover,
        max_depth=max_depth,
        seeds=seeds,
        flow_constant=flow_constant,
        flow_law=_as_choice(raw, "flow_law", FLOW_LAWS, source),
        nmaxq_clip=nmaxq_clip,
        slope_profile=slope_profile,
        khop_scheme=_as_choice(raw, "khop_scheme", KHOP_SCHEMES, source),
        surrogate_edges=surrogate_edges,
        conduit_order=_as_choice(raw, "conduit_order", CONDUIT_ORDERS, source),
        graph_direction=_as_choice(raw, "graph_direction", GRAPH_DIRECTIONS, source),
        max_filling_ratio=max_filling_ratio,
        hydraulics=_build_hydraulics(raw["hydraulics"], source=source),
        gnn=_build_gnn(raw["gnn"], source=source),
        mlp_normalization=_as_choice(raw, "mlp_normalization", MLP_NORMALIZATIONS, source),
        expected=_build_expected(raw["expected"], source=source, base=base),
        split=_build_split(raw["split"], source=source),
        tabular=_build_tabular(raw.get("tabular"), source=source),
        wroclaw=_build_wroclaw(raw.get("wroclaw"), source=source),
        feature_contract=_build_feature_contract(raw.get("feature_contract", "v1"), source=source),
        dataset_run_id=_as_optional_run_id(raw, "dataset_run_id", source),
        generate=_build_generate(raw["generate"], source=source) if "generate" in raw else None,
        eval=_build_eval(raw["eval"], source=source) if "eval" in raw else None,
        c1=_build_c1(raw["c1"], source=source, base=base) if raw.get("c1") is not None else None,
        catchment=_build_catchment(raw["catchment"], source=source) if raw.get("catchment") is not None else None,
        catchment_dataset=(
            _build_catchment_dataset(raw["catchment_dataset"], source=source)
            if raw.get("catchment_dataset") is not None
            else None
        ),
    )


_C1_KEYS: tuple[str, ...] = (
    "dataset_run_reference",
    "pool_manifest",
    "pool_dirs",
    "dev_material",
    "simulation",
    "contracts",
    "split",
    "sources",
    "targets",
    "training",
    "neighborhood",
    "gnn",
    "context_loss",
    "effect_margins",
    "bootstrap",
    "author_review",
)

_C1_FRACTION_SUM_TOLERANCE = 1e-9
"""How far the three split fractions may sum from one; binary rounding, not slack."""


def _build_c1(raw: Any, *, source: Path, base: Path) -> C1Protocol:
    """The optional ``c1`` section: the frozen research protocol of series C1.

    Validated as a declaration, not as a set of switches. Nothing here selects a
    code path today -- the stages of plan16 §5.2 that consume it do not exist yet
    -- so the checks are the ones that make the declaration internally consistent
    and would be silent defects if left to the stage that finally reads it: the
    split fractions add up, every reference margin is a usable denominator, every
    conflict motif lands in one of the two sets, no penalty coefficient is
    negative and no seed is repeated.

    Raises:
        ConfigError: on a malformed section, an unknown or missing key, a value
            outside its domain, or a declaration that contradicts itself.
    """
    section = _c1_mapping(raw, name="c1", keys=_C1_KEYS, source=source)

    dataset_run_reference = _c1_text(section["dataset_run_reference"], name="c1.dataset_run_reference", source=source)
    if "/" in dataset_run_reference or "\\" in dataset_run_reference:
        raise ConfigError(f"c1.dataset_run_reference must be a path-free run id, got {dataset_run_reference!r} (config {source})")

    pool_manifest = _c1_path(section["pool_manifest"], name="c1.pool_manifest", source=source, base=base)
    pool_dirs_raw = section["pool_dirs"]
    if not isinstance(pool_dirs_raw, list) or not pool_dirs_raw:
        raise ConfigError(f"c1.pool_dirs must be a non-empty list of directories, got {pool_dirs_raw!r} (config {source})")
    pool_dirs = tuple(
        _c1_path(item, name=f"c1.pool_dirs[{index}]", source=source, base=base) for index, item in enumerate(pool_dirs_raw)
    )
    if len({str(item) for item in pool_dirs}) != len(pool_dirs):
        raise ConfigError(f"c1.pool_dirs names the same directory twice (config {source})")

    dev_material_raw = section["dev_material"]
    if not isinstance(dev_material_raw, list) or not dev_material_raw:
        raise ConfigError(f"c1.dev_material must be a non-empty list, got {dev_material_raw!r} (config {source})")
    dev_material = tuple(
        _c1_text(item, name=f"c1.dev_material[{index}]", source=source) for index, item in enumerate(dev_material_raw)
    )
    if len(set(dev_material)) != len(dev_material):
        raise ConfigError(f"c1.dev_material names the same network twice (config {source})")

    contracts = _build_c1_contracts(section["contracts"], source=source)
    return C1Protocol(
        dataset_run_reference=dataset_run_reference,
        pool_manifest=pool_manifest,
        pool_dirs=pool_dirs,
        dev_material=dev_material,
        simulation=_build_c1_simulation(section["simulation"], source=source),
        contracts=contracts,
        split=_build_c1_split(section["split"], source=source),
        sources=_build_c1_sources(section["sources"], source=source),
        targets=_build_c1_targets(section["targets"], source=source),
        training=_build_c1_training(section["training"], source=source),
        neighborhood=_build_c1_neighborhood(section["neighborhood"], source=source, contract=contracts.f2_nbr),
        gnn=_build_c1_gnn(section["gnn"], source=source),
        context_loss=_build_c1_context_loss(section["context_loss"], source=source),
        effect_margins=_build_c1_effect_margins(section["effect_margins"], source=source),
        bootstrap=_build_c1_bootstrap(section["bootstrap"], source=source),
        author_review=_build_c1_author_review(section["author_review"], source=source),
    )


_C1_SIMULATION_KEYS: tuple[str, ...] = (
    "flow_routing",
    "link_offsets",
    "routing_step",
    "retry_routing_step",
    "continuity_max_abs_percent",
    "reject_on_continuity",
    "retry_on_continuity",
    "keep_rpt_out",
    "pilot_families",
)

_C1_SIMULATION_OPTIONAL_KEYS: tuple[str, ...] = ("rewrite_options", "routing_dependent_columns")
"""Absent means the empty tuple, i.e. the pool files are taken as they are.

Optional rather than required so that ``config.C1.yaml`` -- the file the frozen C1
runs were produced from -- keeps loading unchanged; a variant that rewrites options
states so, and one that does not says nothing."""


def _build_c1_simulation(raw: Any, *, source: Path) -> C1Simulation:
    section = _c1_mapping(
        raw,
        name="c1.simulation",
        keys=_C1_SIMULATION_KEYS,
        source=source,
        optional=_C1_SIMULATION_OPTIONAL_KEYS,
    )
    flow_routing = _c1_text(section["flow_routing"], name="c1.simulation.flow_routing", source=source)
    if flow_routing not in SWMM_FLOW_ROUTINGS:
        raise ConfigError(
            f"c1.simulation.flow_routing must be one of {list(SWMM_FLOW_ROUTINGS)}, got {flow_routing!r} (config {source})"
        )
    link_offsets = _c1_text(section["link_offsets"], name="c1.simulation.link_offsets", source=source)
    if link_offsets not in SWMM_LINK_OFFSETS:
        raise ConfigError(
            f"c1.simulation.link_offsets must be one of {list(SWMM_LINK_OFFSETS)}, got {link_offsets!r} (config {source})"
        )
    routing_step = _c1_time_step(section["routing_step"], name="c1.simulation.routing_step", source=source)
    retry_routing_step = _c1_time_step(section["retry_routing_step"], name="c1.simulation.retry_routing_step", source=source)
    continuity = _c1_positive_float(
        section["continuity_max_abs_percent"], name="c1.simulation.continuity_max_abs_percent", source=source
    )
    keep = section["keep_rpt_out"]
    if not isinstance(keep, bool):
        raise ConfigError(f"c1.simulation.keep_rpt_out must be a boolean, got {keep!r} (config {source})")
    reject = _c1_flag(section["reject_on_continuity"], name="c1.simulation.reject_on_continuity", source=source)
    retry = _c1_flag(section["retry_on_continuity"], name="c1.simulation.retry_on_continuity", source=source)
    if retry and not reject:
        raise ConfigError(
            f"c1.simulation declares retry_on_continuity=True with reject_on_continuity=False (config {source}); the "
            "second attempt exists to rescue a case that would otherwise be rejected, so retrying without rejecting "
            "would recompute cases at another step and keep them either way"
        )
    pilot = _c1_positive_int(section["pilot_families"], name="c1.simulation.pilot_families", source=source)
    rewrite_options = _c1_rewrite_options(section.get("rewrite_options"), source=source)
    routing_dependent = _c1_routing_dependent_columns(section.get("routing_dependent_columns"), source=source)
    return C1Simulation(
        flow_routing=flow_routing,
        link_offsets=link_offsets,
        routing_step=routing_step,
        retry_routing_step=retry_routing_step,
        continuity_max_abs_percent=continuity,
        reject_on_continuity=reject,
        retry_on_continuity=retry,
        keep_rpt_out=keep,
        pilot_families=pilot,
        rewrite_options=rewrite_options,
        routing_dependent_columns=routing_dependent,
    )


def _c1_routing_dependent_columns(raw: Any, *, source: Path) -> tuple[str, ...]:
    """The declared columns the routing may move, empty where unstated.

    Names only: whether each one exists in the reference frame is checked by
    ``c1-build-cases`` against the file it actually reads, since the config cannot
    see that frame. Duplicates are refused -- a column listed twice would be a typo
    the gate could not tell from an intention.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() for item in raw):
        raise ConfigError(
            f"c1.simulation.routing_dependent_columns must be a list of column names, got {raw!r} (config {source})"
        )
    columns = tuple(item.strip() for item in raw)
    duplicates = sorted({item for item in columns if columns.count(item) > 1})
    if duplicates:
        raise ConfigError(f"c1.simulation.routing_dependent_columns repeats {duplicates} (config {source})")
    return columns


def _c1_rewrite_options(raw: Any, *, source: Path) -> tuple[str, ...]:
    """The declared subset of :data:`SWMM_PROTOCOL_OPTIONS`, empty where unstated.

    Refused rather than normalised when it names an option outside that list or the
    same one twice: the stage writes these values into a working copy, so an option
    it cannot afterwards check would be an edit nobody verifies.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError(f"c1.simulation.rewrite_options must be a list, got {raw!r} (config {source})")
    options = tuple(
        _c1_text(item, name=f"c1.simulation.rewrite_options[{index}]", source=source) for index, item in enumerate(raw)
    )
    unknown = [option for option in options if option not in SWMM_PROTOCOL_OPTIONS]
    if unknown:
        raise ConfigError(
            f"c1.simulation.rewrite_options names {unknown}, which {list(SWMM_PROTOCOL_OPTIONS)} does not contain "
            f"(config {source}); only an option the protocol declares can be rewritten"
        )
    if len(set(options)) != len(options):
        raise ConfigError(f"c1.simulation.rewrite_options names the same option twice: {list(options)} (config {source})")
    return options


def _c1_flag(raw: Any, *, name: str, source: Path) -> bool:
    """One boolean switch of the ``c1`` section, refused when it is not one."""
    if not isinstance(raw, bool):
        raise ConfigError(f"{name} must be a boolean, got {raw!r} (config {source})")
    return raw


_C1_CONTRACT_KEYS: tuple[str, ...] = ("F0", "F1", "F2", "F3")

_C1_CONTRACT_OPTIONAL_KEYS: tuple[str, ...] = ("F2_nbr",)
"""Absent means C1's neighbourhood contract, i.e. the block over C1's own measurements.

Optional so that ``config.C1.yaml`` -- the file the frozen C1 runs were produced from --
keeps loading unchanged; a variant that aggregates another list over the graph names the
contract that carries it."""


def _build_c1_contracts(raw: Any, *, source: Path) -> C1Contracts:
    section = _c1_mapping(
        raw,
        name="c1.contracts",
        keys=_C1_CONTRACT_KEYS,
        source=source,
        optional=_C1_CONTRACT_OPTIONAL_KEYS,
    )
    values = {key: _c1_text(section[key], name=f"c1.contracts.{key}", source=source) for key in _C1_CONTRACT_KEYS}
    declared_nbr = section.get("F2_nbr")
    neighbourhood = (
        NEIGHBORHOOD_CONTRACT_NAME if declared_nbr is None else _c1_text(declared_nbr, name="c1.contracts.F2_nbr", source=source)
    )
    if neighbourhood not in NEIGHBORHOOD_CONTRACT_COLUMNS:
        raise ConfigError(
            f"c1.contracts.F2_nbr is {neighbourhood!r}, which is not a registered neighbourhood contract "
            f"(config {source}); expected one of {sorted(NEIGHBORHOOD_CONTRACT_COLUMNS)}"
        )
    values["F2_nbr"] = neighbourhood
    if len(set(values.values())) != len(values):
        raise ConfigError(f"c1.contracts names the same contract for two views: {values} (config {source})")
    return C1Contracts(f0=values["F0"], f1=values["F1"], f2=values["F2"], f3=values["F3"], f2_nbr=neighbourhood)


_C1_SPLIT_KEYS: tuple[str, ...] = ("protocol", "fractions", "seed", "leakage_seen_fraction_warn")
_C1_FRACTION_KEYS: tuple[str, ...] = ("train", "val", "test")


def _build_c1_split(raw: Any, *, source: Path) -> C1Split:
    section = _c1_mapping(raw, name="c1.split", keys=_C1_SPLIT_KEYS, source=source)
    fractions_raw = _c1_mapping(section["fractions"], name="c1.split.fractions", keys=_C1_FRACTION_KEYS, source=source)
    fractions = {
        key: _c1_positive_float(fractions_raw[key], name=f"c1.split.fractions.{key}", source=source) for key in _C1_FRACTION_KEYS
    }
    total = sum(fractions.values())
    if abs(total - 1.0) > _C1_FRACTION_SUM_TOLERANCE:
        raise ConfigError(f"c1.split.fractions must sum to 1, got {total} (config {source})")
    for key, value in fractions.items():
        if value >= 1.0:
            raise ConfigError(f"c1.split.fractions.{key} must be below 1, got {value} (config {source})")
    warn = _c1_positive_float(section["leakage_seen_fraction_warn"], name="c1.split.leakage_seen_fraction_warn", source=source)
    if warn > 1.0:
        raise ConfigError(f"c1.split.leakage_seen_fraction_warn is a share and must be at most 1, got {warn} (config {source})")
    return C1Split(
        protocol=_c1_text(section["protocol"], name="c1.split.protocol", source=source),
        fractions=C1SplitFractions(**fractions),
        seed=_c1_seed(section["seed"], name="c1.split.seed", source=source),
        leakage_seen_fraction_warn=warn,
    )


_C1_SOURCES_KEYS: tuple[str, ...] = (
    "m_ref",
    "group_weights",
    "category_role",
    "infiltration_imperv_max_pct",
    "retention_imperv_min_pct",
    "hours_ref",
    "oversized_min_steps",
    "oversized_steps_ref",
    "shallow_position_max",
    "deepen_position_max",
    "low_slope_position_max",
    "high_flow_ratio_min",
    "support_splits",
)
_C1_SOURCES_OPTIONAL_KEYS: tuple[str, ...] = (
    "overload_premise",
    "flood_hours_ref",
    "flood_hours_deminimis",
    "flood_volume_deminimis_m3",
    "flood_deminimis_support_min",
    "pressurised_flow_support_max",
)
"""The flooding premise of variant C2, absent from every C1 protocol.

Optional, and each absence resolves to the C1 default on :class:`C1Sources`: the
filling premise, no de-minimis band, no support floor and no support for a conduit
that ran pressurised without spilling. ``config.C1.yaml`` therefore keeps loading
byte for byte, which plan17 makes a hard requirement of the whole variant."""

_C1_M_REF_KEYS: tuple[str, ...] = ("filling", "velocity", "slope", "cover", "depth")
_C1_SUPPORT_SPLIT_KEYS: tuple[str, ...] = (
    "velocity_high",
    "depth_exceeded",
    "upstream_overload",
)
"""The sources whose one violation supports more than one action (plan16 §2 P4).

Fixed here rather than discovered from the file, for the same reason every other
``c1`` subsection has an exact key set: a split declared for a source that does not
exist would be a premise nobody applies, and a source whose split is missing would
fall back on a share written in code."""

_C1_SPLIT_SUM_TOLERANCE = 1e-9


def _build_c1_sources(raw: Any, *, source: Path) -> C1Sources:
    section = _c1_mapping(
        raw,
        name="c1.sources",
        keys=_C1_SOURCES_KEYS,
        source=source,
        optional=_C1_SOURCES_OPTIONAL_KEYS,
    )
    m_ref_raw = _c1_mapping(section["m_ref"], name="c1.sources.m_ref", keys=_C1_M_REF_KEYS, source=source)
    m_ref = {key: _c1_positive_float(m_ref_raw[key], name=f"c1.sources.m_ref.{key}", source=source) for key in _C1_M_REF_KEYS}

    group_weights = _c1_text(section["group_weights"], name="c1.sources.group_weights", source=source)
    if group_weights not in C1_GROUP_WEIGHT_MODES:
        raise ConfigError(
            f"c1.sources.group_weights must be one of {list(C1_GROUP_WEIGHT_MODES)}, got {group_weights!r} "
            f"(config {source}); plan16 §2 P4 records equal group weights as an a priori assumption and forbids "
            "estimating them from how far the sources agree"
        )
    category_role = _c1_text(section["category_role"], name="c1.sources.category_role", source=source)
    if category_role not in C1_CATEGORY_ROLES:
        raise ConfigError(
            f"c1.sources.category_role must be one of {list(C1_CATEGORY_ROLES)}, got {category_role!r} (config {source})"
        )

    infiltration_max = _c1_percentage(
        section["infiltration_imperv_max_pct"], name="c1.sources.infiltration_imperv_max_pct", source=source
    )
    retention_min = _c1_percentage(section["retention_imperv_min_pct"], name="c1.sources.retention_imperv_min_pct", source=source)
    if retention_min <= infiltration_max:
        raise ConfigError(
            f"c1.sources.retention_imperv_min_pct ({retention_min}) is not above infiltration_imperv_max_pct "
            f"({infiltration_max}) (config {source}); the two catchment premises are opposite answers read off one "
            "measurement, so the band between them has to be a band and not an overlap"
        )

    min_steps = _c1_positive_int(section["oversized_min_steps"], name="c1.sources.oversized_min_steps", source=source)
    steps_ref = _c1_positive_int(section["oversized_steps_ref"], name="c1.sources.oversized_steps_ref", source=source)
    if steps_ref < min_steps:
        raise ConfigError(
            f"c1.sources.oversized_steps_ref ({steps_ref}) is below oversized_min_steps ({min_steps}) (config {source}); "
            "a source cannot reach full support before it fires"
        )

    declared_premise = section.get("overload_premise")
    premise = (
        C1_OVERLOAD_PREMISE_FILLING_HOURS
        if declared_premise is None
        else _c1_text(declared_premise, name="c1.sources.overload_premise", source=source)
    )
    if premise not in C1_OVERLOAD_PREMISES:
        raise ConfigError(
            f"c1.sources.overload_premise must be one of {list(C1_OVERLOAD_PREMISES)}, got {premise!r} (config {source})"
        )
    flood_hours_ref = _c1_optional(section, "flood_hours_ref", default=1.0, parse=_c1_positive_float, source=source)
    flood_hours_deminimis = _c1_optional(
        section, "flood_hours_deminimis", default=0.0, parse=_c1_non_negative_float, source=source
    )
    flood_volume_deminimis = _c1_optional(
        section, "flood_volume_deminimis_m3", default=0.0, parse=_c1_non_negative_float, source=source
    )
    deminimis_support_min = _c1_optional(
        section, "flood_deminimis_support_min", default=0.0, parse=_c1_unit_interval, source=source
    )
    pressurised_support_max = _c1_optional(
        section, "pressurised_flow_support_max", default=0.0, parse=_c1_unit_interval, source=source
    )
    if flood_hours_deminimis > flood_hours_ref:
        raise ConfigError(
            f"c1.sources.flood_hours_deminimis ({flood_hours_deminimis}) is above flood_hours_ref ({flood_hours_ref}) "
            f"(config {source}); the band of spills too small to act on cannot reach past the spill that counts as full "
            "support, or every flood would be de minimis"
        )

    return C1Sources(
        m_ref=C1ReferenceMargins(**m_ref),
        group_weights=group_weights,
        category_role=category_role,
        infiltration_imperv_max_pct=infiltration_max,
        retention_imperv_min_pct=retention_min,
        hours_ref=_c1_positive_float(section["hours_ref"], name="c1.sources.hours_ref", source=source),
        oversized_min_steps=min_steps,
        oversized_steps_ref=steps_ref,
        shallow_position_max=_c1_unit_interval(
            section["shallow_position_max"], name="c1.sources.shallow_position_max", source=source
        ),
        deepen_position_max=_c1_unit_interval(
            section["deepen_position_max"], name="c1.sources.deepen_position_max", source=source
        ),
        low_slope_position_max=_c1_unit_interval(
            section["low_slope_position_max"], name="c1.sources.low_slope_position_max", source=source
        ),
        high_flow_ratio_min=_c1_positive_float(
            section["high_flow_ratio_min"], name="c1.sources.high_flow_ratio_min", source=source
        ),
        support_splits=_build_c1_support_splits(section["support_splits"], source=source),
        overload_premise=premise,
        flood_hours_ref=flood_hours_ref,
        flood_hours_deminimis=flood_hours_deminimis,
        flood_volume_deminimis_m3=flood_volume_deminimis,
        flood_deminimis_support_min=deminimis_support_min,
        pressurised_flow_support_max=pressurised_support_max,
    )


def _build_c1_support_splits(raw: Any, *, source: Path) -> C1SupportSplits:
    """One action-to-share mapping per source, each naming real classes and summing to 1."""
    section = _c1_mapping(raw, name="c1.sources.support_splits", keys=_C1_SUPPORT_SPLIT_KEYS, source=source)
    splits: dict[str, Mapping[str, float]] = {}
    for source_id in _C1_SUPPORT_SPLIT_KEYS:
        name = f"c1.sources.support_splits.{source_id}"
        shares_raw = section[source_id]
        if not isinstance(shares_raw, Mapping) or not shares_raw:
            raise ConfigError(f"{name} must be a non-empty mapping of action to share, got {shares_raw!r} (config {source})")
        unknown = sorted(set(shares_raw) - set(RECOMMENDATION_CLASS_ORDER))
        if unknown:
            raise ConfigError(f"{name} names action(s) {unknown}, not among {list(RECOMMENDATION_CLASS_ORDER)} (config {source})")
        if len(shares_raw) < 2:
            raise ConfigError(
                f"{name} declares a split over {list(shares_raw)}; a split describes how one violation is shared "
                f"between at least two actions (config {source})"
            )
        shares = {
            action: _c1_positive_float(value, name=f"{name}.{action}", source=source) for action, value in shares_raw.items()
        }
        total = sum(shares.values())
        if abs(total - 1.0) > _C1_SPLIT_SUM_TOLERANCE:
            raise ConfigError(f"{name} sums to {total}, not 1 (config {source})")
        splits[source_id] = MappingProxyType(dict(shares))
    return C1SupportSplits(splits=MappingProxyType(splits))


_C1_TARGETS_KEYS: tuple[str, ...] = ("contract", "admissible_tau", "min_class_train_l1")


def _build_c1_targets(raw: Any, *, source: Path) -> C1Targets:
    section = _c1_mapping(raw, name="c1.targets", keys=_C1_TARGETS_KEYS, source=source)
    tau = _c1_positive_float(section["admissible_tau"], name="c1.targets.admissible_tau", source=source)
    if tau > 1.0:
        raise ConfigError(f"c1.targets.admissible_tau is a support share and must be at most 1, got {tau} (config {source})")
    return C1Targets(
        contract=_c1_text(section["contract"], name="c1.targets.contract", source=source),
        admissible_tau=tau,
        min_class_train_l1=_c1_positive_int(section["min_class_train_l1"], name="c1.targets.min_class_train_l1", source=source),
    )


_C1_TRAINING_KEYS: tuple[str, ...] = (
    "grid",
    "hidden_layers",
    "max_epochs",
    "patience",
    "cv_folds",
    "seeds",
    "class_weights",
    "sample_weighting",
)
_C1_GRID_KEYS: tuple[str, ...] = ("width", "dropout", "learning_rate")


def _build_c1_training(raw: Any, *, source: Path) -> C1Training:
    section = _c1_mapping(raw, name="c1.training", keys=_C1_TRAINING_KEYS, source=source)
    grid_raw = _c1_mapping(section["grid"], name="c1.training.grid", keys=_C1_GRID_KEYS, source=source)
    width = tuple(
        _c1_positive_int(item, name="c1.training.grid.width", source=source)
        for item in _c1_list(grid_raw["width"], name="c1.training.grid.width", source=source)
    )
    dropout = tuple(
        _c1_unit_interval(item, name="c1.training.grid.dropout", source=source)
        for item in _c1_list(grid_raw["dropout"], name="c1.training.grid.dropout", source=source)
    )
    learning_rate = tuple(
        _c1_positive_float(item, name="c1.training.grid.learning_rate", source=source)
        for item in _c1_list(grid_raw["learning_rate"], name="c1.training.grid.learning_rate", source=source)
    )
    for name, values in (
        ("c1.training.grid.width", width),
        ("c1.training.grid.dropout", dropout),
        ("c1.training.grid.learning_rate", learning_rate),
    ):
        if len(set(values)) != len(values):
            raise ConfigError(f"{name} repeats a value: {list(values)} (config {source})")
    seeds = tuple(
        _c1_seed(item, name="c1.training.seeds", source=source)
        for item in _c1_list(section["seeds"], name="c1.training.seeds", source=source)
    )
    if len(set(seeds)) != len(seeds):
        raise ConfigError(f"c1.training.seeds repeats a seed: {list(seeds)} (config {source}); paired runs need distinct seeds")
    return C1Training(
        grid=C1TrainingGrid(width=width, dropout=dropout, learning_rate=learning_rate),
        hidden_layers=_c1_positive_int(section["hidden_layers"], name="c1.training.hidden_layers", source=source),
        max_epochs=_c1_positive_int(section["max_epochs"], name="c1.training.max_epochs", source=source),
        patience=_c1_positive_int(section["patience"], name="c1.training.patience", source=source),
        cv_folds=_c1_positive_int(section["cv_folds"], name="c1.training.cv_folds", source=source),
        seeds=seeds,
        class_weights=_c1_text(section["class_weights"], name="c1.training.class_weights", source=source),
        sample_weighting=_c1_text(section["sample_weighting"], name="c1.training.sample_weighting", source=source),
    )


_C1_NEIGHBORHOOD_KEYS: tuple[str, ...] = ("columns", "hops")


def _build_c1_neighborhood(raw: Any, *, source: Path, contract: str = NEIGHBORHOOD_CONTRACT_NAME) -> C1Neighborhood:
    """The neighbourhood aggregate declaration: which columns, over how many hops.

    Unlike ``c1.contracts``, this section **is** resolved against the contract
    layer, and has to be: the columns it names are the columns the neighbourhood
    contract of ``c1.contracts.F2_nbr`` was generated from. A protocol declaring one
    list while the registered contract carries another would be two declarations of
    the same thing, and the digest of the contract would describe a view the
    protocol never asked for. So the contract is the source and this section is
    checked against it.

    Which contract that is depends on the protocol, because the list does: C1
    aggregates the hours a conduit's capacity limited the flow and C2 the flooding
    of its inlet node. Checking every protocol against C1's list would refuse C2 for
    declaring exactly what its contract carries.

    Args:
        raw: the ``c1.neighborhood`` mapping.
        source: the configuration file, for the error message.
        contract: the neighbourhood contract the protocol names; C1's by default.

    Raises:
        ConfigError: on a malformed section, or on a list that is not the one the
            named contract was built from.
    """
    section = _c1_mapping(raw, name="c1.neighborhood", keys=_C1_NEIGHBORHOOD_KEYS, source=source)
    columns = tuple(
        _c1_text(item, name="c1.neighborhood.columns", source=source)
        for item in _c1_list(section["columns"], name="c1.neighborhood.columns", source=source)
    )
    hops = tuple(
        _c1_positive_int(item, name="c1.neighborhood.hops", source=source)
        for item in _c1_list(section["hops"], name="c1.neighborhood.hops", source=source)
    )
    declared = neighborhood_columns_of(contract)
    if columns != declared:
        raise ConfigError(
            f"c1.neighborhood.columns is {list(columns)}, but contract {contract!r} was built from "
            f"{list(declared)} (config {source}); the two have to name the same neighbourhood, so move "
            "the list in contracts/features.py and let the contract's digest change with it"
        )
    if hops != NEIGHBORHOOD_HOPS:
        raise ConfigError(
            f"c1.neighborhood.hops is {list(hops)}, but contract {contract!r} was built from "
            f"{list(NEIGHBORHOOD_HOPS)} (config {source})"
        )
    return C1Neighborhood(columns=columns, hops=hops)


_C1_GNN_KEYS: tuple[str, ...] = ("hops", "directions")


def _build_c1_gnn(raw: Any, *, source: Path) -> C1Gnn:
    section = _c1_mapping(raw, name="c1.gnn", keys=_C1_GNN_KEYS, source=source)
    hops = tuple(
        _c1_positive_int(item, name="c1.gnn.hops", source=source)
        for item in _c1_list(section["hops"], name="c1.gnn.hops", source=source)
    )
    too_far = sorted({item for item in hops if item > MAX_GNN_HOPS})
    if too_far:
        raise ConfigError(f"c1.gnn.hops names {too_far}, above the {MAX_GNN_HOPS}-hop ceiling (config {source})")
    if len(set(hops)) != len(hops):
        raise ConfigError(f"c1.gnn.hops repeats a value: {list(hops)} (config {source})")
    directions = tuple(
        _c1_text(item, name="c1.gnn.directions", source=source)
        for item in _c1_list(section["directions"], name="c1.gnn.directions", source=source)
    )
    unknown = sorted(set(directions) - set(C1_GRAPH_DIRECTIONS))
    if unknown:
        raise ConfigError(f"c1.gnn.directions names {unknown}, not among {list(C1_GRAPH_DIRECTIONS)} (config {source})")
    if len(set(directions)) != len(directions):
        raise ConfigError(f"c1.gnn.directions repeats a value: {list(directions)} (config {source})")
    return C1Gnn(hops=hops, directions=directions)


_C1_CONTEXT_LOSS_KEYS: tuple[str, ...] = ("lambda", "motifs")


def _build_c1_context_loss(raw: Any, *, source: Path) -> C1ContextLoss:
    section = _c1_mapping(raw, name="c1.context_loss", keys=_C1_CONTEXT_LOSS_KEYS, source=source)
    lambdas: list[float] = []
    for item in _c1_list(section["lambda"], name="c1.context_loss.lambda", source=source):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"c1.context_loss.lambda must hold numbers, got {item!r} (config {source})")
        if float(item) < 0.0:
            raise ConfigError(f"c1.context_loss.lambda must be non-negative, got {item} (config {source})")
        lambdas.append(float(item))
    if len(set(lambdas)) != len(lambdas):
        raise ConfigError(f"c1.context_loss.lambda repeats a value: {lambdas} (config {source})")

    motifs_raw = section["motifs"]
    if not isinstance(motifs_raw, Mapping) or not motifs_raw:
        raise ConfigError(f"c1.context_loss.motifs must be a non-empty mapping, got {motifs_raw!r} (config {source})")
    motifs: list[C1Motif] = []
    for identifier, set_name in motifs_raw.items():
        name = _c1_text(identifier, name="c1.context_loss.motifs key", source=source)
        assigned = _c1_text(set_name, name=f"c1.context_loss.motifs.{name}", source=source)
        if assigned not in C1_MOTIF_SETS:
            raise ConfigError(
                f"c1.context_loss.motifs.{name} must be one of {list(C1_MOTIF_SETS)}, got {assigned!r} (config {source})"
            )
        motifs.append(C1Motif(identifier=name, set_name=assigned))
    assigned_sets = {motif.set_name for motif in motifs}
    missing = sorted(set(C1_MOTIF_SETS_REQUIRED) - assigned_sets)
    if missing:
        raise ConfigError(
            f"c1.context_loss.motifs leaves {missing} empty (config {source}); the effect of the penalty is measured on "
            f"the {C1_MOTIF_SET_HELD!r} motifs and built from the {C1_MOTIF_SET_TRAIN!r} ones, so a protocol without "
            "either cannot report E7"
        )
    return C1ContextLoss(lambdas=tuple(lambdas), motifs=tuple(motifs))


_C1_EFFECT_MARGIN_KEYS: tuple[str, ...] = (
    "E4_brier_delta_max",
    "E6_context_top1_pp_min",
    "E6_architecture_top1_pp_min",
    "E7_held_conflict_rel_change_max",
    "E7_brier_delta_max",
)


def _build_c1_effect_margins(raw: Any, *, source: Path) -> C1EffectMargins:
    section = _c1_mapping(raw, name="c1.effect_margins", keys=_C1_EFFECT_MARGIN_KEYS, source=source)
    values: dict[str, float] = {}
    for key in _C1_EFFECT_MARGIN_KEYS:
        value = section[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"c1.effect_margins.{key} must be a number, got {value!r} (config {source})")
        values[key] = float(value)
    for key in ("E4_brier_delta_max", "E7_held_conflict_rel_change_max"):
        if values[key] >= 0.0:
            raise ConfigError(
                f"c1.effect_margins.{key} declares an improvement and must be negative, got {values[key]} (config {source})"
            )
    for key in ("E6_context_top1_pp_min", "E6_architecture_top1_pp_min", "E7_brier_delta_max"):
        if values[key] <= 0.0:
            raise ConfigError(
                f"c1.effect_margins.{key} declares a gain or an admissible cost and must be positive, "
                f"got {values[key]} (config {source})"
            )
    return C1EffectMargins(
        e4_brier_delta_max=values["E4_brier_delta_max"],
        e6_context_top1_pp_min=values["E6_context_top1_pp_min"],
        e6_architecture_top1_pp_min=values["E6_architecture_top1_pp_min"],
        e7_held_conflict_rel_change_max=values["E7_held_conflict_rel_change_max"],
        e7_brier_delta_max=values["E7_brier_delta_max"],
    )


_C1_BOOTSTRAP_KEYS: tuple[str, ...] = ("unit", "n_resamples", "seed", "ci")


def _build_c1_bootstrap(raw: Any, *, source: Path) -> C1Bootstrap:
    section = _c1_mapping(raw, name="c1.bootstrap", keys=_C1_BOOTSTRAP_KEYS, source=source)
    level = _c1_positive_float(section["ci"], name="c1.bootstrap.ci", source=source)
    if level >= 1.0:
        raise ConfigError(f"c1.bootstrap.ci is a confidence level in (0, 1), got {level} (config {source})")
    return C1Bootstrap(
        unit=_c1_text(section["unit"], name="c1.bootstrap.unit", source=source),
        n_resamples=_c1_positive_int(section["n_resamples"], name="c1.bootstrap.n_resamples", source=source),
        seed=_c1_seed(section["seed"], name="c1.bootstrap.seed", source=source),
        ci=level,
    )


_C1_AUTHOR_REVIEW_KEYS: tuple[str, ...] = (
    "dev_chains",
    "dev_min_families",
    "val_chains",
    "val_min_families",
    "test_chains",
    "test_min_families",
    "retest_chains",
    "retest_min_days",
)


def _build_c1_author_review(raw: Any, *, source: Path) -> C1AuthorReview:
    section = _c1_mapping(raw, name="c1.author_review", keys=_C1_AUTHOR_REVIEW_KEYS, source=source)
    values = {
        key: _c1_positive_int(section[key], name=f"c1.author_review.{key}", source=source) for key in _C1_AUTHOR_REVIEW_KEYS
    }
    for chains, families in (
        ("dev_chains", "dev_min_families"),
        ("val_chains", "val_min_families"),
        ("test_chains", "test_min_families"),
    ):
        if values[families] > values[chains]:
            raise ConfigError(
                f"c1.author_review.{families} ({values[families]}) exceeds {chains} ({values[chains]}) (config {source}); "
                "a round cannot draw from more families than it has chains"
            )
    if values["retest_chains"] > values["test_chains"]:
        raise ConfigError(
            f"c1.author_review.retest_chains ({values['retest_chains']}) exceeds test_chains ({values['test_chains']}) "
            f"(config {source}); the retest re-presents chains the test round already covered"
        )
    return C1AuthorReview(**values)


def _c1_mapping(
    raw: Any,
    *,
    name: str,
    keys: Collection[str],
    source: Path,
    optional: Collection[str] = (),
) -> Mapping[str, Any]:
    """A ``c1`` subsection with exactly ``keys``; anything else is a defect in the file.

    ``optional`` names keys the section may carry and may leave out. It exists for a
    field added to a section whose already-frozen files must keep loading byte for
    byte, and it is deliberately narrow: an optional key still has to be one this
    loader knows, so a typo remains an error rather than a silent default.
    """
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'{name}' must be a mapping, got {type(raw).__name__} (config {source})")
    required = set(keys)
    expected = required | set(optional)
    unknown = sorted(set(raw) - expected)
    missing = sorted(required - set(raw))
    if unknown or missing:
        raise ConfigError(f"'{name}' unknown={unknown}, missing={missing} (config {source}); allowed: {sorted(expected)}")
    return raw


def _c1_list(raw: Any, *, name: str, source: Path) -> list[Any]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{name} must be a non-empty list, got {raw!r} (config {source})")
    return raw


def _c1_text(value: Any, *, name: str, source: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string, got {value!r} (config {source})")
    return value.strip()


def _c1_path(value: Any, *, name: str, source: Path, base: Path) -> Path:
    """A repository-relative path, resolved like every other configured path."""
    text = _c1_text(value, name=name, source=source)
    candidate = Path(text)
    return candidate if candidate.is_absolute() else base / candidate


def _c1_time_step(value: Any, *, name: str, source: Path) -> str:
    text = _c1_text(value, name=name, source=source)
    if not _TIME_STEP_PATTERN.match(text):
        raise ConfigError(f"{name} must be a SWMM H:MM:SS step, got {text!r} (config {source})")
    return text


def _c1_positive_float(value: Any, *, name: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r} (config {source})")
    if float(value) <= 0.0:
        raise ConfigError(f"{name} must be positive, got {value} (config {source})")
    return float(value)


def _c1_non_negative_float(value: Any, *, name: str, source: Path) -> float:
    """A number that may be zero, refused below it.

    Separate from :func:`_c1_positive_float` because zero is a *meaning* for the two
    de-minimis thresholds and not a mistake: it says the band is empty, which is what
    every C1 protocol declares by declaring nothing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r} (config {source})")
    if float(value) < 0.0:
        raise ConfigError(f"{name} must be non-negative, got {value} (config {source})")
    return float(value)


def _c1_optional(
    section: Mapping[str, Any],
    key: str,
    *,
    default: float,
    parse: Callable[..., float],
    source: Path,
) -> float:
    """``section[key]`` parsed by ``parse``, or ``default`` where the protocol is silent.

    The default is the value that reproduces C1, so an absent key and the C1 file say
    the same thing -- which is the point: C1 declares its behaviour by declaring nothing.
    """
    raw = section.get(key)
    if raw is None:
        return default
    return parse(raw, name=f"c1.sources.{key}", source=source)


def _c1_unit_interval(value: Any, *, name: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r} (config {source})")
    if not 0.0 <= float(value) < 1.0:
        raise ConfigError(f"{name} must be in [0, 1), got {value} (config {source})")
    return float(value)


def _c1_percentage(value: Any, *, name: str, source: Path) -> float:
    """A share written in percent, refused outside ``[0, 100]``.

    Separate from :func:`_c1_unit_interval` because the column it is compared against
    (``subcatchment_imperv_pct``) is in percent: a threshold silently read as a
    fraction would let every catchment through.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r} (config {source})")
    if not 0.0 <= float(value) <= 100.0:
        raise ConfigError(f"{name} is a percentage and must be in [0, 100], got {value} (config {source})")
    return float(value)


def _c1_positive_int(value: Any, *, name: str, source: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer, got {value!r} (config {source})")
    return int(value)


def _c1_seed(value: Any, *, name: str, source: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer seed, got {value!r} (config {source})")
    return int(value)


def _build_wroclaw(raw: Any, *, source: Path) -> WroclawConfig:
    """The optional ``wroclaw`` section, defaulting to constant_columns."""
    if raw is None:
        return WroclawConfig()
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'wroclaw' must be a mapping, got {type(raw).__name__} (config {source})")
    allowed = {"train_support"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"'wroclaw' names unknown key(s) {unknown} in {source}; allowed: {sorted(allowed)}")
    train_support = raw.get("train_support", WROCLAW_TRAIN_SUPPORT_CONSTANT_COLUMNS)
    if train_support not in WROCLAW_TRAIN_SUPPORTS:
        raise ConfigError(
            f"wroclaw.train_support must be one of {list(WROCLAW_TRAIN_SUPPORTS)}, got {train_support!r} (config {source})"
        )
    return WroclawConfig(train_support=str(train_support))


def _build_feature_contract(raw: Any, *, source: Path) -> FeatureContract:
    """Resolve the configured versioned model-input schema."""
    if not isinstance(raw, str) or raw not in FEATURE_CONTRACTS:
        raise ConfigError(f"feature_contract must be one of {sorted(FEATURE_CONTRACTS)}, got {raw!r} (config {source})")
    return feature_contract(raw)


_HYDRAULICS_KEYS: tuple[str, ...] = ("theta", "g", "manning_coefficient", "min_roughness", "max_roughness", "v_min")


def _build_tabular(raw: Any, *, source: Path) -> TabularConfig:
    """The optional ``tabular`` section, defaulting to the B1--B3 behaviour."""
    if raw is None:
        return TabularConfig()
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'tabular' must be a mapping, got {type(raw).__name__} (config {source})")
    allowed = {"require_tune", "observed_classes_from", "report_per_class_metrics"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"'tabular' names unknown key(s) {unknown} in {source}; allowed: {sorted(allowed)}")
    require_tune = raw.get("require_tune", False)
    if not isinstance(require_tune, bool):
        raise ConfigError(f"tabular.require_tune must be a boolean, got {require_tune!r} (config {source})")
    observed = raw.get("observed_classes_from", "train_test")
    if observed not in OBSERVED_CLASS_SOURCES:
        raise ConfigError(
            f"tabular.observed_classes_from must be one of {list(OBSERVED_CLASS_SOURCES)}, got {observed!r} (config {source})"
        )
    report_per_class = raw.get("report_per_class_metrics", False)
    if not isinstance(report_per_class, bool):
        raise ConfigError(f"tabular.report_per_class_metrics must be a boolean, got {report_per_class!r} (config {source})")
    return TabularConfig(
        require_tune=require_tune,
        observed_classes_from=str(observed),
        report_per_class_metrics=report_per_class,
    )


def _build_split(raw: Any, *, source: Path) -> SplitConfig:
    """Validate the closed leakage-firewall configuration."""
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'split' must be a mapping, got {type(raw).__name__} (config {source})")
    required = ("protocols", "lineage", "topology_equivalence", "merge_reduction_warn")
    unknown = sorted(set(raw) - set(required) - {"merge_reduction_gate"})
    missing = sorted(set(required) - set(raw))
    if unknown or missing:
        raise ConfigError(
            f"'split' must name {sorted(required)} plus the optional 'merge_reduction_gate' in {source}; "
            f"unknown={unknown}, missing={missing}"
        )
    section = raw
    protocols_raw = section["protocols"]
    if not isinstance(protocols_raw, list) or not protocols_raw:
        raise ConfigError(f"split.protocols must be a non-empty list, got {protocols_raw!r} (config {source})")
    protocols = tuple(str(value) for value in protocols_raw)
    unknown = [value for value in protocols if value not in SPLIT_PROTOCOLS]
    if unknown:
        raise ConfigError(f"split.protocols contains unknown values {unknown}; expected {list(SPLIT_PROTOCOLS)}")
    if len(set(protocols)) != len(protocols):
        raise ConfigError(f"split.protocols must not contain duplicates, got {list(protocols)}")
    if "file_v1" not in protocols:
        raise ConfigError("split.protocols must include mandatory file_v1")
    if protocols[0] != "file_v1":
        raise ConfigError("split.protocols must put mandatory file_v1 first")
    lineage = section["lineage"]
    if lineage not in LINEAGE_SOURCES:
        raise ConfigError(f"split.lineage must be one of {list(LINEAGE_SOURCES)}, got {lineage!r}")
    topology = section["topology_equivalence"]
    if not isinstance(topology, bool):
        raise ConfigError(f"split.topology_equivalence must be a boolean, got {topology!r}")
    warning = section["merge_reduction_warn"]
    if isinstance(warning, bool) or not isinstance(warning, (int, float)) or not 0.0 <= float(warning) <= 1.0:
        raise ConfigError(f"split.merge_reduction_warn must be within [0, 1], got {warning!r}")
    gate_raw = section.get("merge_reduction_gate")
    if gate_raw is None:
        gate = None
    elif isinstance(gate_raw, bool) or not isinstance(gate_raw, (int, float)) or not 0.0 <= float(gate_raw) <= 1.0:
        raise ConfigError(f"split.merge_reduction_gate must be within [0, 1] or absent, got {gate_raw!r}")
    else:
        gate = float(gate_raw)
    return SplitConfig(protocols, str(lineage), topology, float(warning), gate)


def _build_hydraulics(raw: Any, *, source: Path) -> Hydraulics:
    """The hydraulic constants, every one a positive number."""
    values = _section_of_numbers(raw, keys=_HYDRAULICS_KEYS, name="hydraulics", source=source)
    for key, value in values.items():
        if value <= 0.0:
            raise ConfigError(f"hydraulics.{key} must be positive, got {value} (config {source})")
    if values["min_roughness"] >= values["max_roughness"]:
        raise ConfigError(
            f"hydraulics.min_roughness must be below max_roughness, got "
            f"{values['min_roughness']} >= {values['max_roughness']} (config {source}); "
            "the normalisation of Roughness divides by their difference"
        )
    return Hydraulics(**values)


GENERATE_SIZING_MODES: tuple[str, ...] = ("random", "rational")
"""Valid values for ``generate.sizing``."""

GENERATE_KNOWN_FAMILIES: tuple[str, ...] = (
    "chain_11",
    "chain_13",
    "chain_15",
    "chain_17",
    "chain_19",
    "chain_21",
    "tree_3",
    "tree_4",
    "irregular",
)
"""Topology families the generator can build."""

GENERATE_AUGMENTATION_TARGETS: tuple[str, ...] = (
    "deep_shallow_adjacency",
    "catalog_max_overflow",
    "increase_dia_steep",
    "min_diameter_near_zero_flow",
)
"""Opt-in profiles accepted in a train-only generator specification."""

GENERATE_FLAT_COVER_MODE = "flat_cover"
"""The ``node_depth`` mode that buries every fed junction at a configured cover.

Named here as well as in :mod:`stormlab.generate.grid` because the configuration
has to know when ``cover_m`` is required, and ``config`` sits below ``generate``
in the layer order.
"""

GENERATE_OVERRIDABLE_CATEGORY_BANDS: tuple[str, ...] = ("perc_imperv", "perc_slope")
"""The two per-category bands ``generate.category_params_override`` may restate.

Deliberately narrow: these two are drawn per subcatchment from a uniform band, so
restating them changes the population a pool samples. The subarea constants
(``n_imperv``, ``s_perv``, ``pct_zero`` ...) are properties of the surface type
itself and a pool that changed them would be sampling a different category under
the same name.
"""


def _build_category_overrides(
    raw: Any,
    *,
    categories: Collection[str],
    source: Path,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Validate ``generate.category_params_override`` against the profile's categories.

    Every band is checked here rather than at sampling time, because a band that
    named a category the profile never draws, or that ran high-to-low, would
    produce an ordinary pool with no error at all — the failure mode the strata
    vocabulary check already exists to prevent.

    Raises:
        ConfigError: on a category outside ``categories``, an unknown band name, a
            band that is not ``[low, high]`` inside ``[0, 100]``, or ``low >= high``.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping) or not raw:
        raise ConfigError(f"generate.category_params_override must be a non-empty mapping (config {source})")
    known = set(categories)
    overrides: dict[str, dict[str, tuple[float, float]]] = {}
    for category, bands in raw.items():
        name = str(category)
        if name not in known:
            raise ConfigError(
                f"generate.category_params_override names {name!r}, which is not in generate.categories "
                f"{sorted(known)}; a band nothing draws from would leave the pool unchanged (config {source})"
            )
        if not isinstance(bands, Mapping) or not bands:
            raise ConfigError(f"generate.category_params_override[{name}] must be a non-empty mapping (config {source})")
        unknown = sorted(set(map(str, bands)) - set(GENERATE_OVERRIDABLE_CATEGORY_BANDS))
        if unknown:
            raise ConfigError(
                f"generate.category_params_override[{name}] has unknown band(s) {unknown}; "
                f"overridable: {list(GENERATE_OVERRIDABLE_CATEGORY_BANDS)} (config {source})"
            )
        resolved: dict[str, tuple[float, float]] = {}
        for band, value in bands.items():
            if not isinstance(value, list) or len(value) != 2:
                raise ConfigError(
                    f"generate.category_params_override[{name}].{band} must be a two-element [low, high] list (config {source})"
                )
            low, high = (float(item) for item in value)
            if not 0.0 <= low < high <= 100.0:
                raise ConfigError(
                    f"generate.category_params_override[{name}].{band} must satisfy 0 <= low < high <= 100, "
                    f"got [{low}, {high}] (config {source})"
                )
            resolved[str(band)] = (low, high)
        overrides[name] = resolved
    return overrides


def _build_generate(raw: Any, *, source: Path) -> Generate:
    """The synthetic generator configuration, with thorough validation.

    ``categories: all`` is expanded to the twelve subcatchment category names.
    """
    from stormlab.contracts.enums import SubcatchmentCategory

    if not isinstance(raw, Mapping):
        raise ConfigError(f"'generate' must be a mapping, got {type(raw).__name__} (config {source})")

    def _require(key: str) -> Any:
        if key not in raw:
            raise ConfigError(f"generate.{key} is required (config {source})")
        return raw[key]

    seed = _require("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"generate.seed must be an integer, got {seed!r} (config {source})")

    template = str(_require("template"))
    output_dir = str(_require("output_dir"))

    n_realisations = _require("n_realisations")
    if isinstance(n_realisations, bool) or not isinstance(n_realisations, int) or n_realisations < 1:
        raise ConfigError(f"generate.n_realisations must be a positive integer, got {n_realisations!r} (config {source})")

    # Topology families
    families_raw = _require("topology_families")
    if not isinstance(families_raw, list) or not families_raw:
        raise ConfigError(f"generate.topology_families must be a non-empty list (config {source})")
    for f in families_raw:
        if f not in GENERATE_KNOWN_FAMILIES:
            raise ConfigError(
                f"generate.topology_families: unknown family {f!r}; known: {GENERATE_KNOWN_FAMILIES} (config {source})"
            )
    topology_families = tuple(families_raw)

    # Numeric lists
    def _positive_sorted_unique(key: str, *, allow_zero: bool = False) -> tuple[float, ...]:
        values = _require(key)
        if not isinstance(values, list) or not values:
            raise ConfigError(f"generate.{key} must be a non-empty list (config {source})")
        nums: list[float] = []
        for v in values:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ConfigError(f"generate.{key}: {v!r} is not a number (config {source})")
            fv = float(v)
            if not allow_zero and fv <= 0:
                raise ConfigError(f"generate.{key}: {fv} must be positive (config {source})")
            if allow_zero and fv < 0:
                raise ConfigError(f"generate.{key}: {fv} must be non-negative (config {source})")
            nums.append(fv)
        if sorted(nums) != nums:
            raise ConfigError(f"generate.{key} must be sorted ascending (config {source})")
        if len(set(nums)) != len(nums):
            raise ConfigError(f"generate.{key} must have unique values (config {source})")
        return tuple(nums)

    diameters_m = _positive_sorted_unique("diameters_m")
    slopes_per_mille = _positive_sorted_unique("slopes_per_mille")
    roughness_values = _positive_sorted_unique("roughness")

    # Rain gages
    rain_raw = _require("rain_gages")
    if not isinstance(rain_raw, list) or not rain_raw:
        raise ConfigError(f"generate.rain_gages must be a non-empty list (config {source})")
    rain_gages = tuple(str(r) for r in rain_raw)

    # Categories
    cat_raw = _require("categories")
    all_categories = tuple(c.value for c in SubcatchmentCategory)
    if cat_raw == "all":
        categories = all_categories
    elif isinstance(cat_raw, list):
        for c in cat_raw:
            if c not in all_categories:
                raise ConfigError(f"generate.categories: {c!r} is not a valid SubcatchmentCategory (config {source})")
        categories = tuple(cat_raw)
    else:
        raise ConfigError(f"generate.categories must be 'all' or a list (config {source})")

    category_params_override = _build_category_overrides(
        raw.get("category_params_override"), categories=categories, source=source
    )

    # Subcatchments per conduit
    sbc_raw = _require("subcatchments_per_conduit")
    if not isinstance(sbc_raw, list) or not sbc_raw:
        raise ConfigError(f"generate.subcatchments_per_conduit must be a non-empty list (config {source})")
    subcatchments_per_conduit = tuple(int(s) for s in sbc_raw)

    # Outlet cover and node depth
    cover_raw = _require("outlet_cover")
    if not isinstance(cover_raw, list) or not cover_raw:
        raise ConfigError(f"generate.outlet_cover must be a non-empty list (config {source})")
    outlet_cover = tuple(str(c) for c in cover_raw)

    depth_raw = _require("node_depth")
    if not isinstance(depth_raw, list) or not depth_raw:
        raise ConfigError(f"generate.node_depth must be a non-empty list (config {source})")
    node_depth = tuple(str(d) for d in depth_raw)

    # The band is required exactly where it is read and refused where it is not,
    # so a profile can neither ask for `flat_cover` without saying how deep the
    # cover is, nor declare a band that nothing draws from.
    cover_raw = raw.get("cover_m")
    if GENERATE_FLAT_COVER_MODE in node_depth and cover_raw is None:
        raise ConfigError(
            f"generate.cover_m is required when generate.node_depth names {GENERATE_FLAT_COVER_MODE!r}: that mode "
            f"draws the soil cover over each pipe from it (config {source})"
        )
    if cover_raw is not None and GENERATE_FLAT_COVER_MODE not in node_depth:
        raise ConfigError(
            f"generate.cover_m is set but generate.node_depth does not name {GENERATE_FLAT_COVER_MODE!r}, so nothing "
            f"would draw from it (config {source})"
        )
    cover_m: tuple[float, float] | None = None
    if cover_raw is not None:
        if not isinstance(cover_raw, list) or len(cover_raw) != 2:
            raise ConfigError(f"generate.cover_m must be a two-element [low, high] list (config {source})")
        cover_low, cover_high = (float(value) for value in cover_raw)
        if not 0.0 < cover_low < cover_high:
            raise ConfigError(f"generate.cover_m must satisfy 0 < low < high, got [{cover_low}, {cover_high}] (config {source})")
        cover_m = (cover_low, cover_high)

    # Length distribution
    length_raw = _require("length_m")
    if not isinstance(length_raw, Mapping):
        raise ConfigError(f"generate.length_m must be a mapping (config {source})")
    length = GenerateLength(
        dist=str(length_raw.get("dist", "lognormal")),
        median=float(length_raw["median"]),
        min=float(length_raw["min"]),
        max=float(length_raw["max"]),
    )
    if not length.min < length.median < length.max:
        raise ConfigError(
            f"generate.length_m: need min < median < max, got {length.min} < {length.median} < {length.max} (config {source})"
        )

    # Area distribution
    area_raw = _require("area_ha")
    if not isinstance(area_raw, Mapping):
        raise ConfigError(f"generate.area_ha must be a mapping (config {source})")
    area = GenerateArea(
        dist=str(area_raw.get("dist", "loguniform")),
        min=float(area_raw["min"]),
        max=float(area_raw["max"]),
    )
    if not area.min < area.max:
        raise ConfigError(f"generate.area_ha: need min < max (config {source})")

    # Flow length distribution
    flow_raw = _require("flow_length_m")
    if not isinstance(flow_raw, Mapping):
        raise ConfigError(f"generate.flow_length_m must be a mapping (config {source})")
    flow_length = GenerateFlowLength(
        dist=str(flow_raw.get("dist", "uniform")),
        min=float(flow_raw["min"]),
        max=float(flow_raw["max"]),
    )
    if not flow_length.min < flow_length.max:
        raise ConfigError(f"generate.flow_length_m: need min < max (config {source})")

    # Sizing
    sizing_raw = str(_require("sizing"))
    if sizing_raw not in GENERATE_SIZING_MODES:
        raise ConfigError(f"generate.sizing must be one of {GENERATE_SIZING_MODES}, got {sizing_raw!r} (config {source})")

    # Precision and tolerance
    precision = float(_require("elevation_precision_m"))
    if precision <= 0:
        raise ConfigError(f"generate.elevation_precision_m must be positive (config {source})")

    tolerance = float(_require("slope_tolerance_per_mille"))
    if tolerance <= 0:
        raise ConfigError(f"generate.slope_tolerance_per_mille must be positive (config {source})")

    outfall_invert = float(_require("outfall_invert_m"))

    # min_occurrences: optional coverage gate (default 60)
    min_occ_raw = raw.get("min_occurrences", 60)
    min_occurrences = int(min_occ_raw)
    if min_occurrences < 0:
        raise ConfigError(f"generate.min_occurrences must be non-negative (config {source})")

    # frost_zone_dataset for shallow cover calculation
    frost_zone_gen = float(raw.get("frost_zone_dataset", 1.6))
    if frost_zone_gen <= 0:
        raise ConfigError(f"generate.frost_zone_dataset must be positive (config {source})")

    # two_subcatchment_share: fraction of nodes that get 2 subcatchments
    two_sbc_share = float(raw.get("two_subcatchment_share", 0.1))
    if not 0.0 <= two_sbc_share <= 1.0:
        raise ConfigError(f"generate.two_subcatchment_share must be in [0, 1] (config {source})")

    strata_raw = raw.get("strata", [])
    if not isinstance(strata_raw, list):
        raise ConfigError(f"generate.strata must be a list, got {type(strata_raw).__name__} (config {source})")
    strata: list[Any] = []
    for index, item in enumerate(strata_raw):
        if not isinstance(item, Mapping) or not {"name", "share", "force"} <= set(item):
            raise ConfigError(f"generate.strata[{index}] must name at least name/share/force (config {source})")
        unknown_stratum = sorted(set(item) - {"name", "share", "force", "reason"})
        if unknown_stratum:
            raise ConfigError(f"generate.strata[{index}] has unknown keys {unknown_stratum} (config {source})")
        share = item["share"]
        if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0.0 < float(share) <= 1.0:
            raise ConfigError(f"generate.strata[{index}].share must be within (0, 1], got {share!r} (config {source})")
        force = item["force"]
        if not isinstance(force, Mapping) or not force:
            raise ConfigError(f"generate.strata[{index}].force must be a non-empty mapping (config {source})")
        unknown_force = sorted(set(force) - _STRATUM_FORCEABLE)
        if unknown_force:
            raise ConfigError(
                f"generate.strata[{index}].force may only set {sorted(_STRATUM_FORCEABLE)}, got {unknown_force} (config {source})"
            )
        # The values, not only the keys. A stratum naming a category or a cover
        # mode that does not exist would silently produce an ordinary pool and
        # the class it was declared for would stay rare with no error.
        vocabularies = {
            "categories": set(categories),
            "outlet_cover": set(outlet_cover) | {"normal", "shallow"},
            "node_depth": set(node_depth) | {"normal", "deep", "minimal_cover", GENERATE_FLAT_COVER_MODE},
        }
        for key, value in force.items():
            offered = set(value) if isinstance(value, list) else {value}
            unknown_value = sorted(str(v) for v in offered - vocabularies[key])
            if unknown_value:
                raise ConfigError(
                    f"generate.strata[{index}].force.{key} names {unknown_value}, which is not in "
                    f"{sorted(str(v) for v in vocabularies[key])} (config {source})"
                )
        strata.append(
            {
                "name": str(item["name"]),
                "share": float(share),
                "force": {str(key): value for key, value in force.items()},
                "reason": str(item.get("reason", "")),
            }
        )

    combinations_raw = raw.get("combinations_per_skeleton")
    if combinations_raw is None:
        combinations_per_skeleton = None
    elif isinstance(combinations_raw, bool) or not isinstance(combinations_raw, int) or combinations_raw < 1:
        raise ConfigError(
            f"generate.combinations_per_skeleton must be a positive integer or absent, got {combinations_raw!r} (config {source})"
        )
    else:
        combinations_per_skeleton = combinations_raw

    targets_raw = raw.get("augmentation_targets", [])
    if not isinstance(targets_raw, list):
        raise ConfigError(f"generate.augmentation_targets must be a list (config {source})")
    unknown_targets = sorted(set(str(item) for item in targets_raw) - set(GENERATE_AUGMENTATION_TARGETS))
    if unknown_targets:
        raise ConfigError(
            f"generate.augmentation_targets names unknown profiles {unknown_targets}; "
            f"known: {list(GENERATE_AUGMENTATION_TARGETS)} (config {source})"
        )
    augmentation_targets = tuple(str(item) for item in targets_raw)
    if len(set(augmentation_targets)) != len(augmentation_targets):
        raise ConfigError(f"generate.augmentation_targets must be unique (config {source})")

    parity_reference_raw = raw.get("parity_reference", True)
    if not isinstance(parity_reference_raw, bool):
        raise ConfigError(f"generate.parity_reference must be a boolean, got {parity_reference_raw!r} (config {source})")

    family_raw = raw.get("family")
    family = str(family_raw) if family_raw is not None else None
    if family is not None and not family.strip():
        raise ConfigError(f"generate.family must not be blank (config {source})")

    offset_raw = raw.get("skeleton_id_offset", 0)
    if isinstance(offset_raw, bool) or not isinstance(offset_raw, int) or offset_raw < 0:
        raise ConfigError(f"generate.skeleton_id_offset must be a non-negative integer, got {offset_raw!r} (config {source})")

    return Generate(
        seed=seed,
        template=template,
        output_dir=output_dir,
        n_realisations=n_realisations,
        combinations_per_skeleton=combinations_per_skeleton,
        strata=tuple(strata),
        topology_families=topology_families,
        diameters_m=diameters_m,
        slopes_per_mille=slopes_per_mille,
        roughness=roughness_values,
        rain_gages=rain_gages,
        categories=categories,
        subcatchments_per_conduit=subcatchments_per_conduit,
        outlet_cover=outlet_cover,
        node_depth=node_depth,
        length=length,
        area=area,
        flow_length=flow_length,
        sizing=sizing_raw,
        elevation_precision_m=precision,
        slope_tolerance_per_mille=tolerance,
        outfall_invert_m=outfall_invert,
        min_occurrences=min_occurrences,
        frost_zone_dataset=frost_zone_gen,
        two_subcatchment_share=two_sbc_share,
        augmentation_targets=augmentation_targets,
        category_params_override=category_params_override,
        cover_m=cover_m,
        parity_reference=parity_reference_raw,
        family=family,
        skeleton_id_offset=offset_raw,
    )


_EVAL_CONTINUOUS_DISTS = frozenset({"uniform", "loguniform", "lognormal"})


def _eval_source(raw: Mapping[str, Any], *, name: str, source: Path) -> str:
    value = raw.get("source")
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"{name}.source must be a non-empty citation or the literal note 'bez źródła, wartość arbitralna' (config {source})"
        )
    return value.strip()


def _eval_discrete_band(raw: Any, *, name: str, source: Path) -> EvalDiscreteBand:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{name} must be a mapping (config {source})")
    allowed = {"values", "weights", "source"}
    if set(raw) != allowed:
        raise ConfigError(f"{name} must name exactly {sorted(allowed)}, got {sorted(raw)} (config {source})")
    values_raw, weights_raw = raw["values"], raw["weights"]
    if not isinstance(values_raw, list) or not values_raw:
        raise ConfigError(f"{name}.values must be a non-empty list (config {source})")
    if not isinstance(weights_raw, list) or len(weights_raw) != len(values_raw):
        raise ConfigError(f"{name}.weights must have one entry per value (config {source})")
    values = tuple(float(value) for value in values_raw)
    weights = tuple(float(value) for value in weights_raw)
    if any(value <= 0 for value in values) or any(weight <= 0 for weight in weights):
        raise ConfigError(f"{name} values and weights must be positive (config {source})")
    if len(set(values)) != len(values):
        raise ConfigError(f"{name}.values must be unique (config {source})")
    total = sum(weights)
    return EvalDiscreteBand(
        values=values, weights=tuple(weight / total for weight in weights), source=_eval_source(raw, name=name, source=source)
    )


def _eval_continuous_band(raw: Any, *, name: str, source: Path) -> EvalContinuousBand:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{name} must be a mapping (config {source})")
    allowed = {"dist", "min", "max", "median", "source"}
    unknown = sorted(set(raw) - allowed)
    required = {"dist", "min", "max", "source"}
    missing = sorted(required - set(raw))
    if unknown or missing:
        raise ConfigError(f"{name}: unknown={unknown}, missing={missing} (config {source})")
    dist = str(raw["dist"])
    if dist not in _EVAL_CONTINUOUS_DISTS:
        raise ConfigError(f"{name}.dist must be one of {sorted(_EVAL_CONTINUOUS_DISTS)}, got {dist!r}")
    low, high = float(raw["min"]), float(raw["max"])
    if not 0 < low < high:
        raise ConfigError(f"{name} needs 0 < min < max, got {low}, {high} (config {source})")
    median = float(raw["median"]) if "median" in raw else None
    if dist == "lognormal" and (median is None or not low < median < high):
        raise ConfigError(f"{name} lognormal distribution needs min < median < max (config {source})")
    return EvalContinuousBand(dist=dist, min=low, max=high, median=median, source=_eval_source(raw, name=name, source=source))


def _eval_categorical_band(
    raw: Any,
    *,
    name: str,
    vocabulary: Collection[str],
    source: Path,
) -> EvalCategoricalBand:
    if not isinstance(raw, Mapping) or set(raw) != {"values", "weights", "source"}:
        raise ConfigError(f"{name} must name exactly values, weights and source (config {source})")
    values_raw, weights_raw = raw["values"], raw["weights"]
    if not isinstance(values_raw, list) or not values_raw or not isinstance(weights_raw, list):
        raise ConfigError(f"{name}.values and .weights must be non-empty lists (config {source})")
    values = tuple(str(value) for value in values_raw)
    if len(weights_raw) != len(values) or len(set(values)) != len(values):
        raise ConfigError(f"{name} needs one weight per unique value (config {source})")
    unknown = sorted(set(values) - set(vocabulary))
    if unknown:
        raise ConfigError(f"{name}.values contains unknown values {unknown} (config {source})")
    weights = tuple(float(value) for value in weights_raw)
    if any(value <= 0 for value in weights):
        raise ConfigError(f"{name}.weights must be positive (config {source})")
    total = sum(weights)
    return EvalCategoricalBand(
        values=values,
        weights=tuple(weight / total for weight in weights),
        source=_eval_source(raw, name=name, source=source),
    )


def _build_benchmark_holdout(
    raw: Any,
    *,
    name: str,
    source: Path,
    stratified: bool,
) -> BenchmarkHoldoutConfig | BenchmarkHoldoutStratifiedConfig:
    """Validate one realistic holdout design, including every sourced band."""
    from stormlab.contracts.enums import SubcatchmentCategory

    holdout = raw
    if not isinstance(holdout, Mapping):
        raise ConfigError(f"{name} must be a mapping (config {source})")
    required = {
        "seed",
        "n_networks",
        "max_rejection_share",
        "families",
        "diameters_m",
        "slopes_per_mille",
        "lengths_m",
        "roughness",
        "categories",
        "outlet_cover",
        "node_depth",
        "rain_gages",
    }
    if stratified:
        required.add("min_rows_per_class")
    if set(holdout) != required:
        raise ConfigError(f"{name} must name exactly {sorted(required)}, got {sorted(holdout)} (config {source})")
    seed, n_networks = holdout["seed"], holdout["n_networks"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"{name}.seed must be an integer (config {source})")
    if isinstance(n_networks, bool) or not isinstance(n_networks, int) or n_networks < 1:
        raise ConfigError(f"{name}.n_networks must be positive (config {source})")
    maximum = holdout["max_rejection_share"]
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not 0 <= float(maximum) <= 0.2:
        raise ConfigError(f"{name}.max_rejection_share must be within [0, 0.2] (config {source})")
    families_raw = holdout["families"]
    if not isinstance(families_raw, list) or not families_raw:
        raise ConfigError(f"{name}.families must be a weighted non-empty list (config {source})")
    families = tuple(str(value) for value in families_raw)
    unknown_families = sorted(set(families) - set(GENERATE_KNOWN_FAMILIES))
    if unknown_families:
        raise ConfigError(f"{name}.families contains unknown values {unknown_families}")
    if families.count("irregular") / len(families) > 0.2:
        raise ConfigError(f"{name}.families must weight irregular at no more than 20%")
    categories_raw = holdout["categories"]
    categories = tuple(item.value for item in SubcatchmentCategory) if categories_raw == "all" else tuple(categories_raw)
    if not categories or set(categories) - {item.value for item in SubcatchmentCategory}:
        raise ConfigError(f"{name}.categories contains an unknown category (config {source})")
    rain = holdout["rain_gages"]
    if not isinstance(rain, list) or len(rain) != 3 or len(set(rain)) != 3:
        raise ConfigError(f"{name}.rain_gages must name exactly three unique gages (config {source})")
    fields = dict(
        seed=seed,
        n_networks=n_networks,
        max_rejection_share=float(maximum),
        families=families,
        diameters_m=_eval_discrete_band(holdout["diameters_m"], name=f"{name}.diameters_m", source=source),
        slopes_per_mille=_eval_continuous_band(holdout["slopes_per_mille"], name=f"{name}.slopes_per_mille", source=source),
        lengths_m=_eval_continuous_band(holdout["lengths_m"], name=f"{name}.lengths_m", source=source),
        roughness=_eval_discrete_band(holdout["roughness"], name=f"{name}.roughness", source=source),
        categories=categories,
        outlet_cover=_eval_categorical_band(
            holdout["outlet_cover"],
            name=f"{name}.outlet_cover",
            vocabulary=("normal", "shallow"),
            source=source,
        ),
        node_depth=_eval_categorical_band(
            holdout["node_depth"],
            name=f"{name}.node_depth",
            vocabulary=("normal", "minimal_cover", "deep"),
            source=source,
        ),
        rain_gages=tuple(str(value) for value in rain),
    )
    if not stratified:
        return BenchmarkHoldoutConfig(**fields)
    minimum = holdout["min_rows_per_class"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ConfigError(f"{name}.min_rows_per_class must be positive (config {source})")
    return BenchmarkHoldoutStratifiedConfig(**fields, min_rows_per_class=minimum)


def _build_eval(raw: Any, *, source: Path) -> EvalConfig:
    """Validate P7's evaluation-set designs and every sourced physical band."""
    expected = {"benchmark_holdout", "benchmark_holdout_stratified", "rule_conformance_suite"}
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ConfigError(f"eval must name exactly {sorted(expected)} (config {source})")
    benchmark = _build_benchmark_holdout(raw["benchmark_holdout"], name="eval.benchmark_holdout", source=source, stratified=False)
    stratified = _build_benchmark_holdout(
        raw["benchmark_holdout_stratified"],
        name="eval.benchmark_holdout_stratified",
        source=source,
        stratified=True,
    )
    assert isinstance(benchmark, BenchmarkHoldoutConfig)
    assert isinstance(stratified, BenchmarkHoldoutStratifiedConfig)
    conformance = raw["rule_conformance_suite"]
    conformance_keys = {"seed", "target_arcs", "min_rows_per_class", "n_realisations"}
    if not isinstance(conformance, Mapping) or set(conformance) != conformance_keys:
        raise ConfigError(f"eval.rule_conformance_suite must name exactly {sorted(conformance_keys)} (config {source})")
    conformance_seed = conformance["seed"]
    minimum = conformance["min_rows_per_class"]
    realisations = conformance["n_realisations"]
    if isinstance(conformance_seed, bool) or not isinstance(conformance_seed, int):
        raise ConfigError(f"eval.rule_conformance_suite.seed must be an integer (config {source})")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ConfigError(f"eval.rule_conformance_suite.min_rows_per_class must be positive (config {source})")
    if isinstance(realisations, bool) or not isinstance(realisations, int) or realisations < 1:
        raise ConfigError(f"eval.rule_conformance_suite.n_realisations must be positive (config {source})")
    return EvalConfig(
        benchmark_holdout=benchmark,
        benchmark_holdout_stratified=stratified,
        rule_conformance_suite=RuleConformanceConfig(
            seed=conformance_seed,
            target_arcs=str(conformance["target_arcs"]),
            min_rows_per_class=minimum,
            n_realisations=realisations,
        ),
    )


def _section_of_numbers(raw: Any, *, keys: tuple[str, ...], name: str, source: Path) -> dict[str, float]:
    """A mapping section that must name exactly ``keys``, each a number."""
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'{name}' must be a mapping, got {type(raw).__name__} (config {source})")
    if set(raw) != set(keys):
        raise ConfigError(f"'{name}' must name exactly {sorted(keys)} in {source}, got {sorted(raw)}")
    values: dict[str, float] = {}
    for key in keys:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{name}.{key} must be a number, got {value!r} (config {source})")
        values[key] = float(value)
    return values


_GNN_KEYS: tuple[str, ...] = (
    "ablation",
    "hops",
    "k_serving",
    "seed_sweep",
    "epochs",
    "lr",
    "patience",
    "upstream_variant_hops",
    "report_hops",
    "cv",
    "diag",
)

_GNN_OPTIONAL_KEYS: tuple[str, ...] = ("final",)
"""Keys the ``gnn`` section may omit.

``final`` is the whole switch between study (b) as variant B1 ran it and the
final step of study (b2): a configuration without it is B1's, and one with it
names an ablation artifact to read a winner from. It is optional rather than
nullable so that the base configuration says nothing about a step it does not
have, and an overlay that adds it is the only place the new path is declared."""

_GNN_POSITIVE_INT_KEYS: tuple[str, ...] = ("k_serving", "epochs", "patience", "upstream_variant_hops", "report_hops")


def _build_gnn(raw: Any, *, source: Path) -> GnnExperiment:
    """The GNN experiment section: hop counts, sweep size and training schedule.

    ``seed_sweep`` may be zero — that is the documented way to turn the sweep off
    — but every hop count has to be a positive integer, and the three hop counts
    a downstream step reads (``k_serving``, ``upstream_variant_hops``,
    ``report_hops``) have to be values the sweep actually trains, or the run would
    name a model it never produced.
    """
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'gnn' must be a mapping, got {type(raw).__name__} (config {source})")
    missing = set(_GNN_KEYS) - set(raw)
    unknown = set(raw) - set(_GNN_KEYS) - set(_GNN_OPTIONAL_KEYS)
    if missing or unknown:
        raise ConfigError(
            f"'gnn' must name exactly {sorted(_GNN_KEYS)} (optionally {sorted(_GNN_OPTIONAL_KEYS)}) in {source}, "
            f"got {sorted(raw)}"
        )

    hops_raw = raw["hops"]
    if not isinstance(hops_raw, list) or not hops_raw:
        raise ConfigError(f"gnn.hops must be a non-empty list, got {hops_raw!r} (config {source})")
    hops: list[int] = []
    for value in hops_raw:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"gnn.hops must hold positive integers, got {value!r} (config {source})")
        hops.append(value)
    if len(set(hops)) != len(hops):
        raise ConfigError(f"gnn.hops must not repeat a hop count, got {hops} (config {source})")

    integers: dict[str, int] = {}
    for key in _GNN_POSITIVE_INT_KEYS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"gnn.{key} must be a positive integer, got {value!r} (config {source})")
        integers[key] = value

    seed_sweep = raw["seed_sweep"]
    if isinstance(seed_sweep, bool) or not isinstance(seed_sweep, int) or seed_sweep < 0:
        raise ConfigError(f"gnn.seed_sweep must be a non-negative integer, got {seed_sweep!r} (config {source})")

    lr = raw["lr"]
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or lr <= 0.0:
        raise ConfigError(f"gnn.lr must be a positive number, got {lr!r} (config {source})")

    for key in ("k_serving", "upstream_variant_hops", "report_hops"):
        if integers[key] not in hops:
            raise ConfigError(
                f"gnn.{key} = {integers[key]} is not one of gnn.hops {sorted(hops)} (config {source}); "
                "a hop count that is never trained cannot be served, transposed or reported"
            )

    diag = _build_gnn_diag(raw["diag"], source=source)
    cv = _build_gnn_cv(raw["cv"], source=source)
    ablation = _build_gnn_ablation(raw["ablation"], source=source)
    final = _build_gnn_final(raw["final"], source=source) if "final" in raw else None

    return GnnExperiment(
        hops=tuple(hops),
        seed_sweep=seed_sweep,
        lr=float(lr),
        cv=cv,
        diag=diag,
        ablation=ablation,
        final=final,
        **integers,
    )


_GNN_FINAL_KEYS: tuple[str, ...] = (
    "winner_from",
    "seeds",
    "main_seed_index",
    "nograph",
    "baseline_legacy_k",
    "serving_model_id",
    "verdicts",
)
_GNN_FINAL_OPTIONAL_KEYS: tuple[str, ...] = ("select_hops_by_cv", "artifact_dir")
"""Fields a final step may declare and B2's frozen overlay does not.

Optional rather than required because ``_mapping_named`` demands an exact key
set: a required field would make the tracked ``config.B2.yaml`` unloadable, and
re-freezing B2's overlay to add a field its run never used would move a frozen
variant to describe a later one."""

_GNN_FINAL_VERDICT_KEYS: tuple[str, ...] = ("stability", "vs_mlp", "message_passing", "bootstrap", "level")
_GNN_FINAL_STABILITY_KEYS: tuple[str, ...] = ("accuracy_std_max", "accuracy_range_max")
_GNN_FINAL_VS_MLP_KEYS: tuple[str, ...] = ("margin_errors",)
_GNN_FINAL_MESSAGE_PASSING_KEYS: tuple[str, ...] = ("minimum_effect",)


def _build_gnn_final(raw: Any, *, source: Path) -> GnnFinal:
    """The final step of study (b2), including the three verdict thresholds.

    Every threshold is validated here rather than where it is applied, so a
    malformed one costs a second instead of forty-one trainings — and so the
    object that is written into the artifact before the held-out split is read is
    the same object the verdicts are decided with.

    Raises:
        ConfigError: on a malformed section, a seed index outside the seed set, a
            non-positive count, a confidence level outside ``(0, 1)`` or a
            threshold that is not a non-negative number.
    """
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'gnn.final' must be a mapping, got {type(raw).__name__} (config {source})")
    unknown = sorted(set(raw) - set(_GNN_FINAL_KEYS) - set(_GNN_FINAL_OPTIONAL_KEYS))
    if unknown:
        raise ConfigError(f"'gnn.final' names unknown key(s) {unknown} in {source}")
    section = _mapping_named(
        {key: value for key, value in raw.items() if key in _GNN_FINAL_KEYS},
        keys=_GNN_FINAL_KEYS,
        name="gnn.final",
        source=source,
    )
    select_hops_by_cv = raw.get("select_hops_by_cv", False)
    if not isinstance(select_hops_by_cv, bool):
        raise ConfigError(f"gnn.final.select_hops_by_cv must be a boolean, got {select_hops_by_cv!r} (config {source})")
    artifact_dir = raw.get("artifact_dir")
    if artifact_dir is not None and (not isinstance(artifact_dir, str) or not artifact_dir.strip()):
        raise ConfigError(f"gnn.final.artifact_dir must be a non-empty path or absent, got {artifact_dir!r} (config {source})")

    winner_from = section["winner_from"]
    if not isinstance(winner_from, str) or not winner_from.strip():
        raise ConfigError(f"gnn.final.winner_from must be a non-empty path, got {winner_from!r} (config {source})")
    serving_model_id = section["serving_model_id"]
    if not isinstance(serving_model_id, str) or not serving_model_id.strip():
        raise ConfigError(f"gnn.final.serving_model_id must be a non-empty model id, got {serving_model_id!r} (config {source})")
    nograph = section["nograph"]
    if nograph is not True:
        # Not "must be a boolean": `false` parses and then costs 21 trainings
        # before V3 discovers it has nothing to be a paired difference against.
        # The field stays, because it is what the artifact says the arm was
        # trained, but the only value this pipeline can carry out is `true`.
        raise ConfigError(
            f"gnn.final.nograph must be true, got {nograph!r} (config {source}); V3 is a paired difference "
            "against the graph-free arm, so a final step without it has no message-passing verdict to report"
        )

    seeds = _positive_int(section["seeds"], name="gnn.final.seeds", source=source)
    main_seed_index = section["main_seed_index"]
    if isinstance(main_seed_index, bool) or not isinstance(main_seed_index, int) or not 0 <= main_seed_index < seeds:
        raise ConfigError(
            f"gnn.final.main_seed_index must be an integer in [0, {seeds}), got {main_seed_index!r} (config "
            f"{source}); it selects the reported model out of the seeds this step trains"
        )
    baseline_legacy_k = _positive_int(section["baseline_legacy_k"], name="gnn.final.baseline_legacy_k", source=source)
    if baseline_legacy_k > MAX_GNN_HOPS:
        raise ConfigError(
            f"gnn.final.baseline_legacy_k must be at most {MAX_GNN_HOPS}, got {baseline_legacy_k} (config {source})"
        )

    verdicts_raw = _mapping_named(section["verdicts"], keys=_GNN_FINAL_VERDICT_KEYS, name="gnn.final.verdicts", source=source)
    stability_raw = _mapping_named(
        verdicts_raw["stability"], keys=_GNN_FINAL_STABILITY_KEYS, name="gnn.final.verdicts.stability", source=source
    )
    vs_mlp_raw = _mapping_named(
        verdicts_raw["vs_mlp"], keys=_GNN_FINAL_VS_MLP_KEYS, name="gnn.final.verdicts.vs_mlp", source=source
    )
    message_passing_raw = _mapping_named(
        verdicts_raw["message_passing"],
        keys=_GNN_FINAL_MESSAGE_PASSING_KEYS,
        name="gnn.final.verdicts.message_passing",
        source=source,
    )
    level = verdicts_raw["level"]
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not 0.0 < level < 1.0:
        raise ConfigError(f"gnn.final.verdicts.level must be a number in (0, 1), got {level!r} (config {source})")

    verdicts = GnnFinalVerdicts(
        stability=GnnFinalStability(
            accuracy_std_max=_non_negative_number(
                stability_raw["accuracy_std_max"], name="gnn.final.verdicts.stability.accuracy_std_max", source=source
            ),
            accuracy_range_max=_non_negative_number(
                stability_raw["accuracy_range_max"],
                name="gnn.final.verdicts.stability.accuracy_range_max",
                source=source,
            ),
        ),
        vs_mlp=GnnFinalNonInferiority(
            margin_errors=_positive_int(
                vs_mlp_raw["margin_errors"], name="gnn.final.verdicts.vs_mlp.margin_errors", source=source
            )
        ),
        message_passing=GnnFinalMessagePassing(
            minimum_effect=_non_negative_number(
                message_passing_raw["minimum_effect"],
                name="gnn.final.verdicts.message_passing.minimum_effect",
                source=source,
            )
        ),
        bootstrap=_positive_int(verdicts_raw["bootstrap"], name="gnn.final.verdicts.bootstrap", source=source),
        level=float(level),
    )
    return GnnFinal(
        winner_from=winner_from,
        seeds=seeds,
        main_seed_index=main_seed_index,
        nograph=nograph,
        baseline_legacy_k=baseline_legacy_k,
        serving_model_id=serving_model_id,
        verdicts=verdicts,
        select_hops_by_cv=select_hops_by_cv,
        artifact_dir=artifact_dir,
    )


_GNN_ABLATION_KEYS: tuple[str, ...] = ("folds", "screening", "s0", "s1", "s2", "s3")
_GNN_ABLATION_SCREENING_KEYS: tuple[str, ...] = ("hops", "seeds", "rotations")
_GNN_ABLATION_S0_KEYS: tuple[str, ...] = (
    "hidden",
    "dropout",
    "weight_decay",
    "layer_norm",
    "tolerance",
    "sigma_factor",
)
_GNN_ABLATION_S1_KEYS: tuple[str, ...] = (
    "operators",
    "directions",
    "input_norms",
    "tolerance",
    "keep_with_deviations",
    "keep_without_deviations",
)
_GNN_ABLATION_S2_KEYS: tuple[str, ...] = ("hops", "shortlist")
_GNN_ABLATION_S3_KEYS: tuple[str, ...] = ("seeds", "bootstrap")


def _build_gnn_ablation(raw: Any, *, source: Path) -> GnnAblation:
    """The declared search space of study (b2), validated before anything trains.

    Every value is checked against the same closed vocabularies the model builder
    uses, so a typo in an operator name costs a second rather than a stage: the
    alternative is discovering it after S0 has spent thirty-five trainings.

    Raises:
        ConfigError: on a malformed section, an unknown operator, direction,
            normaliser or input normalisation, a non-positive count, a threshold
            outside its range, or a shortlist that cannot be filled from the
            configurations S1 keeps.
    """
    section = _mapping_named(raw, keys=_GNN_ABLATION_KEYS, name="gnn.ablation", source=source)
    folds = _positive_int(section["folds"], name="gnn.ablation.folds", source=source)
    if folds < MINIMUM_CV_FOLDS:
        raise ConfigError(
            f"gnn.ablation.folds must be at least {MINIMUM_CV_FOLDS}, got {folds} (config {source}); one fold is "
            "scored and one holds out the early stopping, so fewer leaves nothing to fit on"
        )

    screening_raw = _mapping_named(
        section["screening"], keys=_GNN_ABLATION_SCREENING_KEYS, name="gnn.ablation.screening", source=source
    )
    screening = GnnAblationScreening(
        hops=_positive_int(screening_raw["hops"], name="gnn.ablation.screening.hops", source=source),
        seeds=_positive_int(screening_raw["seeds"], name="gnn.ablation.screening.seeds", source=source),
        rotations=_positive_int(screening_raw["rotations"], name="gnn.ablation.screening.rotations", source=source),
    )
    if screening.rotations > folds:
        raise ConfigError(
            f"gnn.ablation.screening.rotations = {screening.rotations} exceeds gnn.ablation.folds = {folds} "
            f"(config {source}); there is no {screening.rotations + 1}-th fold to score"
        )

    s0_raw = _mapping_named(section["s0"], keys=_GNN_ABLATION_S0_KEYS, name="gnn.ablation.s0", source=source)
    s0 = GnnAblationArchitecture(
        hidden=tuple(
            _positive_int(item, name="gnn.ablation.s0.hidden[]", source=source)
            for item in _list_named(s0_raw["hidden"], name="gnn.ablation.s0.hidden", source=source)
        ),
        dropout=tuple(
            _unit_interval(item, name="gnn.ablation.s0.dropout[]", source=source)
            for item in _list_named(s0_raw["dropout"], name="gnn.ablation.s0.dropout", source=source)
        ),
        weight_decay=tuple(
            _non_negative_number(item, name="gnn.ablation.s0.weight_decay[]", source=source)
            for item in _list_named(s0_raw["weight_decay"], name="gnn.ablation.s0.weight_decay", source=source)
        ),
        layer_norm=_choice_list(
            s0_raw["layer_norm"], allowed=GNN_NORMALIZATIONS, name="gnn.ablation.s0.layer_norm", source=source
        ),
        tolerance=_non_negative_number(s0_raw["tolerance"], name="gnn.ablation.s0.tolerance", source=source),
        sigma_factor=_non_negative_number(s0_raw["sigma_factor"], name="gnn.ablation.s0.sigma_factor", source=source),
    )

    s1_raw = _mapping_named(section["s1"], keys=_GNN_ABLATION_S1_KEYS, name="gnn.ablation.s1", source=source)
    s1 = GnnAblationOperators(
        operators=_choice_list(
            s1_raw["operators"], allowed=SEARCHABLE_GNN_OPERATORS, name="gnn.ablation.s1.operators", source=source
        ),
        directions=_choice_list(s1_raw["directions"], allowed=GRAPH_DIRECTIONS, name="gnn.ablation.s1.directions", source=source),
        input_norms=_choice_list(
            s1_raw["input_norms"], allowed=GNN_INPUT_NORMS, name="gnn.ablation.s1.input_norms", source=source
        ),
        tolerance=_non_negative_number(s1_raw["tolerance"], name="gnn.ablation.s1.tolerance", source=source),
        keep_with_deviations=_positive_int(
            s1_raw["keep_with_deviations"], name="gnn.ablation.s1.keep_with_deviations", source=source
        ),
        keep_without_deviations=_positive_int(
            s1_raw["keep_without_deviations"], name="gnn.ablation.s1.keep_without_deviations", source=source
        ),
    )
    combinations = len(s1.operators) * len(s1.directions) * len(s1.input_norms)
    for name in ("keep_with_deviations", "keep_without_deviations"):
        if getattr(s1, name) > combinations:
            raise ConfigError(
                f"gnn.ablation.s1.{name} = {getattr(s1, name)} exceeds the {combinations} configuration(s) S1 "
                f"trains (config {source}); the stage keeps exactly that many, so it cannot keep more than it has"
            )

    s2_raw = _mapping_named(section["s2"], keys=_GNN_ABLATION_S2_KEYS, name="gnn.ablation.s2", source=source)
    s2 = GnnAblationHops(
        hops=_hop_list(s2_raw["hops"], name="gnn.ablation.s2.hops", source=source),
        shortlist=_positive_int(s2_raw["shortlist"], name="gnn.ablation.s2.shortlist", source=source),
    )
    # The shortlist has to be fillable on *both* S0 branches, or its declared
    # length is a promise the stage cannot keep on one of them — which is the
    # silent truncation this whole artifact exists to prevent. With a deviation
    # each survivor yields two candidates, without one it yields a single one; the
    # control adds a position only when it is not already among them, so the
    # guaranteed supply is the candidate count itself plus one.
    for name, reachable in (
        ("keep_with_deviations", 2 * s1.keep_with_deviations + 1),
        ("keep_without_deviations", s1.keep_without_deviations + 1),
    ):
        if s2.shortlist > reachable:
            raise ConfigError(
                f"gnn.ablation.s2.shortlist = {s2.shortlist} cannot be filled when S1 keeps "
                f"{getattr(s1, name)} configuration(s) (gnn.ablation.s1.{name}): {reachable} position(s) at most "
                f"(config {source})"
            )
    if screening.hops not in s2.hops:
        raise ConfigError(
            f"gnn.ablation.screening.hops = {screening.hops} is not among gnn.ablation.s2.hops "
            f"{list(s2.hops)} (config {source}); S0 and S1 screen at a hop count S2 never sweeps"
        )

    s3_raw = _mapping_named(section["s3"], keys=_GNN_ABLATION_S3_KEYS, name="gnn.ablation.s3", source=source)
    s3 = GnnAblationEvaluation(
        seeds=_positive_int(s3_raw["seeds"], name="gnn.ablation.s3.seeds", source=source),
        bootstrap=_positive_int(s3_raw["bootstrap"], name="gnn.ablation.s3.bootstrap", source=source),
    )

    return GnnAblation(folds=folds, screening=screening, s0=s0, s1=s1, s2=s2, s3=s3)


def _list_named(value: Any, *, name: str, source: Path) -> list[Any]:
    """A non-empty list; the items are the caller's problem."""
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name} must be a non-empty list, got {value!r} (config {source})")
    return list(value)


def _choice_list(value: Any, *, allowed: tuple[str, ...], name: str, source: Path) -> tuple[str, ...]:
    """A non-empty list of distinct values drawn from a closed vocabulary."""
    items = _list_named(value, name=name, source=source)
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ConfigError(f"{name} names {unknown}, which are not among {list(allowed)} (config {source})")
    if len(set(items)) != len(items):
        raise ConfigError(f"{name} must not repeat a value, got {items} (config {source})")
    return tuple(str(item) for item in items)


def _non_negative_number(value: Any, *, name: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0.0:
        raise ConfigError(f"{name} must be a non-negative number, got {value!r} (config {source})")
    return float(value)


def _unit_interval(value: Any, *, name: str, source: Path) -> float:
    """A rate in ``[0, 1)``: dropping every unit is an absent layer, not a strong one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value < 1.0:
        raise ConfigError(f"{name} must be a number in [0, 1), got {value!r} (config {source})")
    return float(value)


_GNN_DIAG_KEYS: tuple[str, ...] = ("seeds", "hops", "h3", "h4", "h5")
_GNN_DIAG_H3_KEYS: tuple[str, ...] = ("hops", "seeds", "epochs")
_GNN_DIAG_H4_KEYS: tuple[str, ...] = ("hops", "seeds")
_GNN_DIAG_H5_KEYS: tuple[str, ...] = ("hops", "seeds", "norms")


_GNN_CV_KEYS: tuple[str, ...] = ("folds", "seeds")


MINIMUM_CV_FOLDS = 3
"""Fewest folds the selection protocol can run on.

Each round spends one fold on the score and one on the early stopping that
produced the model being scored, so two folds leave nothing to fit on. Three is
the smallest partition where the three roles are three different folds."""


def _build_gnn_cv(raw: Any, *, source: Path) -> GnnCrossValidation:
    """The cross-validated hop selection: how many folds and how many seeds each.

    ``folds`` may be ``0`` — the documented way to turn the selection off — but
    anything between that and :data:`MINIMUM_CV_FOLDS` is refused rather than
    quietly producing a degenerate score under a name that promises five.
    """
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'gnn.cv' must be a mapping, got {type(raw).__name__} (config {source})")
    if set(raw) != set(_GNN_CV_KEYS):
        raise ConfigError(f"'gnn.cv' must name exactly {sorted(_GNN_CV_KEYS)} in {source}, got {sorted(raw)}")
    folds = raw["folds"]
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 0 or 0 < folds < MINIMUM_CV_FOLDS:
        raise ConfigError(
            f"gnn.cv.folds must be 0 (off) or at least {MINIMUM_CV_FOLDS}, got {folds!r} (config {source}); "
            "one fold is scored and one holds out the early stopping, so fewer leaves nothing to fit on"
        )
    seeds = raw["seeds"]
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise ConfigError(f"gnn.cv.seeds must be a positive integer, got {seeds!r} (config {source})")
    return GnnCrossValidation(folds=folds, seeds=seeds)


def _build_gnn_diag(raw: Any, *, source: Path) -> GnnDiagnostics:
    """The instability diagnosis: the shared grid and the three hypothesis grids.

    Every hop count and every seed count is a positive integer, and every
    normaliser has to be one the model can actually be built with — a diagnosis
    that discovered its own typo after eighty trainings would have measured
    nothing but the cost of the typo.
    """
    section = _mapping_named(raw, keys=_GNN_DIAG_KEYS, name="gnn.diag", source=source)
    diagnostics = GnnDiagnostics(
        seeds=_positive_int(section["seeds"], name="gnn.diag.seeds", source=source),
        hops=_hop_list(section["hops"], name="gnn.diag.hops", source=source),
        h3=_build_gnn_diag_h3(section["h3"], source=source),
        h4=_build_gnn_diag_h4(section["h4"], source=source),
        h5=_build_gnn_diag_h5(section["h5"], source=source),
    )

    # H4 and H5 are paired against the shared grid rather than retraining their
    # own reference arm, so their hop counts and seed counts have to be inside
    # it. Checked here because the alternative is discovering it after eighty
    # trainings, with a KeyError instead of a sentence.
    for name, hypothesis in (("h4", diagnostics.h4), ("h5", diagnostics.h5)):
        outside = [value for value in hypothesis.hops if value not in diagnostics.hops]
        if outside:
            raise ConfigError(
                f"gnn.diag.{name}.hops names {outside}, which gnn.diag.hops {list(diagnostics.hops)} does not "
                f"train (config {source}); the paired arm of that comparison comes from the shared grid"
            )
        if hypothesis.seeds > diagnostics.seeds:
            raise ConfigError(
                f"gnn.diag.{name}.seeds = {hypothesis.seeds} exceeds gnn.diag.seeds = {diagnostics.seeds} "
                f"(config {source}); the paired arm of that comparison comes from the shared grid"
            )
    if GNN_NORMALIZATION_BATCH not in diagnostics.h5.norms:
        raise ConfigError(
            f"gnn.diag.h5.norms must include {GNN_NORMALIZATION_BATCH!r} (config {source}); it is the arm the "
            "others are compared against, and it is read from the shared grid rather than retrained"
        )
    return diagnostics


def _build_gnn_diag_h3(raw: Any, *, source: Path) -> GnnDiagEarlyStopping:
    section = _mapping_named(raw, keys=_GNN_DIAG_H3_KEYS, name="gnn.diag.h3", source=source)
    return GnnDiagEarlyStopping(
        hops=_hop_list(section["hops"], name="gnn.diag.h3.hops", source=source),
        seeds=_positive_int(section["seeds"], name="gnn.diag.h3.seeds", source=source),
        epochs=_positive_int(section["epochs"], name="gnn.diag.h3.epochs", source=source),
    )


def _build_gnn_diag_h4(raw: Any, *, source: Path) -> GnnDiagDirection:
    section = _mapping_named(raw, keys=_GNN_DIAG_H4_KEYS, name="gnn.diag.h4", source=source)
    return GnnDiagDirection(
        hops=_hop_list(section["hops"], name="gnn.diag.h4.hops", source=source),
        seeds=_positive_int(section["seeds"], name="gnn.diag.h4.seeds", source=source),
    )


def _build_gnn_diag_h5(raw: Any, *, source: Path) -> GnnDiagLayerNormalization:
    section = _mapping_named(raw, keys=_GNN_DIAG_H5_KEYS, name="gnn.diag.h5", source=source)
    norms_raw = section["norms"]
    if not isinstance(norms_raw, list) or not norms_raw:
        raise ConfigError(f"gnn.diag.h5.norms must be a non-empty list, got {norms_raw!r} (config {source})")
    norms = [str(value) for value in norms_raw]
    unknown = [value for value in norms if value not in GNN_NORMALIZATIONS]
    if unknown:
        raise ConfigError(f"gnn.diag.h5.norms names {unknown} which are not among {list(GNN_NORMALIZATIONS)} (config {source})")
    if len(set(norms)) != len(norms):
        raise ConfigError(f"gnn.diag.h5.norms must not repeat a normaliser, got {norms} (config {source})")
    return GnnDiagLayerNormalization(
        hops=_hop_list(section["hops"], name="gnn.diag.h5.hops", source=source),
        seeds=_positive_int(section["seeds"], name="gnn.diag.h5.seeds", source=source),
        norms=tuple(norms),
    )


def _mapping_named(raw: Any, *, keys: tuple[str, ...], name: str, source: Path) -> Mapping[str, Any]:
    """A mapping section that must name exactly ``keys``; the values are unchecked."""
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'{name}' must be a mapping, got {type(raw).__name__} (config {source})")
    if set(raw) != set(keys):
        raise ConfigError(f"'{name}' must name exactly {sorted(keys)} in {source}, got {sorted(raw)}")
    return raw


def _positive_int(value: Any, *, name: str, source: Path) -> int:
    """An integer of at least one; booleans are not integers here."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer, got {value!r} (config {source})")
    return value


def _hop_list(value: Any, *, name: str, source: Path) -> tuple[int, ...]:
    """A non-empty list of distinct positive hop counts."""
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name} must be a non-empty list, got {value!r} (config {source})")
    hops = [_positive_int(item, name=f"{name}[]", source=source) for item in value]
    if len(set(hops)) != len(hops):
        raise ConfigError(f"{name} must not repeat a hop count, got {hops} (config {source})")
    return tuple(hops)


_EXPECTED_SECTIONS: dict[str, type] = {
    "dataset": ExpectedDataset,
    "catchment": ExpectedCatchment,
    "wroclaw": ExpectedWroclaw,
}


def _build_expected(raw: Any, *, source: Path, base: Path) -> Expected:
    """The population gates, every count a positive integer."""
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'expected' must be a mapping, got {type(raw).__name__} (config {source})")
    unknown = set(raw) - set(_EXPECTED_SECTIONS) - {"min_relative_feature_iqr"}
    if unknown:
        raise ConfigError(f"Unknown keys under 'expected' in {source}: {sorted(unknown)}")
    missing = set(_EXPECTED_SECTIONS) - set(raw)
    if missing:
        raise ConfigError(f"Missing keys under 'expected' in {source}: {sorted(missing)}")

    sections: dict[str, Any] = {}
    dataset = raw["dataset"]
    if not isinstance(dataset, Mapping):
        raise ConfigError(f"expected.dataset must be a mapping, got {type(dataset).__name__} (config {source})")
    legacy_keys = {"files", "rows", "conduits_per_file"}
    if set(dataset) == {"manifest"}:
        value = dataset["manifest"]
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"expected.dataset.manifest must be a non-empty path, got {value!r} (config {source})")
        candidate = Path(value)
        sections["dataset"] = ExpectedDataset(manifest=candidate if candidate.is_absolute() else base / candidate)
    elif set(dataset) == legacy_keys:
        values = _positive_count_mapping(dataset, name="expected.dataset", source=source)
        sections["dataset"] = ExpectedDataset(**values)
    else:
        raise ConfigError(
            f"expected.dataset must name exactly either {sorted(legacy_keys)} or ['manifest'] in {source}, got {sorted(dataset)}"
        )

    for name, factory in _EXPECTED_SECTIONS.items():
        if name == "dataset":
            continue
        section = raw[name]
        if not isinstance(section, Mapping):
            raise ConfigError(f"expected.{name} must be a mapping, got {type(section).__name__} (config {source})")
        wanted = {field for field in factory.__dataclass_fields__}  # type: ignore[attr-defined]
        if set(section) != wanted:
            raise ConfigError(f"expected.{name} must name exactly {sorted(wanted)} in {source}, got {sorted(section)}")
        values: dict[str, int] = {}
        for key, value in section.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"expected.{name}.{key} must be a positive integer, got {value!r} (config {source})")
            values[key] = value
        sections[name] = factory(**values)
    threshold = raw.get("min_relative_feature_iqr", 0.0)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) < 1.0:
        raise ConfigError(f"expected.min_relative_feature_iqr must be within [0, 1), got {threshold!r} (config {source})")
    return Expected(**sections, min_relative_feature_iqr=float(threshold))


_CATCHMENT_KEYS: tuple[str, ...] = ("model_id",)
_CATCHMENT_OPTIONAL_KEYS: tuple[str, ...] = (
    "reference_run_id",
    "reference_run_ids",
    "feature_set",
    "hidden_units",
    "extra_holdouts",
)

_EXTRA_HOLDOUT_KEYS: tuple[str, ...] = ("id", "dir", "limit", "seed")

_CATCHMENT_DATASET_KEYS: tuple[str, ...] = (
    "seed",
    "n_base",
    "n_flat",
    "template",
    "rain_gages",
    "out_dir",
    "area_range",
    "regime",
)

_CATCHMENT_CLASS_TABLE_KEYS: tuple[str, ...] = (
    "generator",
    "seed",
    "n_per_class",
    "template",
    "rain_gages",
    "out_dir",
    "area_range",
    "class_table_sha",
)

_REGIME_KEYS: tuple[str, ...] = ("slope_pct_max", "imperv_pct")


def _build_catchment(raw: Any, *, source: Path) -> CatchmentConfig:
    """The ``catchment`` section: the identity a retrained classifier is published under.

    ``model_id`` is required whenever the section exists, because the whole point
    of the section is to name the model; leaving it optional would let a variant
    declare the section and still write another variant's file name. The other
    four keys are optional and default to what every run before plan19 did, so a
    file that names none of them loads into the section it always loaded into.

    Raises:
        ConfigError: on a malformed section, an unknown or missing key, an
            identifier that could not be used as a file name, an unknown feature
            set, or a width that is not a positive integer.
    """
    from stormlab.contracts.catchment_features import FEATURE_SETS

    section = _c1_mapping(raw, name="catchment", keys=_CATCHMENT_KEYS, optional=_CATCHMENT_OPTIONAL_KEYS, source=source)
    model_id = _c1_text(section["model_id"], name="catchment.model_id", source=source)
    if "/" in model_id or "\\" in model_id or model_id.startswith("."):
        raise ConfigError(
            f"catchment.model_id must be a plain identifier, got {model_id!r} (config {source}); it becomes the name "
            "of the weights file and of every numbers.json key that carries a model"
        )
    reference = section.get("reference_run_id")
    reference_run_id = None if reference is None else _c1_text(reference, name="catchment.reference_run_id", source=source)
    if reference_run_id is not None and ("/" in reference_run_id or "\\" in reference_run_id):
        raise ConfigError(f"catchment.reference_run_id must be a path-free run id, got {reference_run_id!r} (config {source})")

    references_raw = section.get("reference_run_ids")
    reference_run_ids: tuple[str, ...] = ()
    if references_raw is not None:
        items = _c1_list(references_raw, name="catchment.reference_run_ids", source=source)
        reference_run_ids = tuple(
            _c1_text(item, name=f"catchment.reference_run_ids[{index}]", source=source) for index, item in enumerate(items)
        )
        offending = [run_id for run_id in reference_run_ids if "/" in run_id or "\\" in run_id]
        if offending:
            raise ConfigError(f"catchment.reference_run_ids must hold path-free run ids, got {offending} (config {source})")

    feature_set = _c1_text(section.get("feature_set", DEFAULT_CATCHMENT_FEATURE_SET), name="catchment.feature_set", source=source)
    if feature_set not in FEATURE_SETS:
        raise ConfigError(f"catchment.feature_set must be one of {sorted(FEATURE_SETS)}, got {feature_set!r} (config {source})")
    hidden_units = _positive_int(
        section.get("hidden_units", DEFAULT_CATCHMENT_HIDDEN_UNITS), name="catchment.hidden_units", source=source
    )
    return CatchmentConfig(
        model_id=model_id,
        reference_run_id=reference_run_id,
        reference_run_ids=reference_run_ids,
        feature_set=feature_set,
        hidden_units=hidden_units,
        extra_holdouts=_build_extra_holdouts(section.get("extra_holdouts"), source=source),
    )


def _build_extra_holdouts(raw: Any, *, source: Path) -> tuple[ExtraHoldout, ...]:
    """The optional ``catchment.extra_holdouts`` list, validated entry by entry.

    Absent means no extra holdout at all rather than an empty declaration: a
    variant that names none has to register the keys it always registered, and an
    ``id`` reaches a ``numbers.json`` key, a workbook name and a parquet file name,
    so two entries sharing one would overwrite each other's figures rather than
    report two populations.

    Raises:
        ConfigError: on a malformed list or entry, an unknown or missing key, an
            ``id`` that could not be used as a file name, an absolute directory, a
            non-positive ``limit``, a negative ``seed``, or a repeated ``id``.
    """
    if raw is None:
        return ()
    items = _c1_list(raw, name="catchment.extra_holdouts", source=source)
    holdouts: list[ExtraHoldout] = []
    for index, item in enumerate(items):
        name = f"catchment.extra_holdouts[{index}]"
        entry = _c1_mapping(item, name=name, keys=_EXTRA_HOLDOUT_KEYS, source=source)
        identifier = _c1_text(entry["id"], name=f"{name}.id", source=source)
        if "/" in identifier or "\\" in identifier or identifier.startswith("."):
            raise ConfigError(
                f"{name}.id must be a plain identifier, got {identifier!r} (config {source}); it becomes part of every "
                "numbers.json key, workbook and probability file this holdout produces"
            )
        seed = entry["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ConfigError(f"{name}.seed must be a non-negative integer, got {seed!r} (config {source})")
        holdouts.append(
            ExtraHoldout(
                id=identifier,
                dir=_repo_relative_path(entry["dir"], name=f"{name}.dir", source=source),
                limit=_positive_int(entry["limit"], name=f"{name}.limit", source=source),
                seed=seed,
            )
        )
    repeated = sorted({item.id for item in holdouts if [other.id for other in holdouts].count(item.id) > 1})
    if repeated:
        raise ConfigError(
            f"catchment.extra_holdouts names {repeated} more than once (config {source}); one id is one population, "
            "and two entries under it would write over each other's keys and files"
        )
    return tuple(holdouts)


def _build_catchment_dataset(raw: Any, *, source: Path) -> CatchmentDatasetProfile | CatchmentClassTableProfile:
    """The ``catchment_dataset`` section, dispatched on which sampler it describes.

    ``generator`` is optional and absent means :data:`CATCHMENT_GENERATOR_PROTOTYPES`,
    so every file written before plan19 keeps loading into the profile it always did.
    The two arms name disjoint key sets and are validated separately, because a
    section that mixed them -- ``n_flat`` beside ``n_per_class`` -- describes no
    sampler at all and would otherwise be accepted by whichever arm ran first.

    Raises:
        ConfigError: if ``generator`` is not one of :data:`CATCHMENT_GENERATORS`, or
            the section is malformed for the sampler it names.
    """
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'catchment_dataset' must be a mapping, got {type(raw).__name__} (config {source})")
    generator = raw.get("generator", CATCHMENT_GENERATOR_PROTOTYPES)
    if generator not in CATCHMENT_GENERATORS:
        raise ConfigError(
            f"catchment_dataset.generator must be one of {list(CATCHMENT_GENERATORS)}, got {generator!r} (config {source})"
        )
    if generator == CATCHMENT_GENERATOR_CLASS_TABLE:
        return _build_catchment_class_table(raw, source=source)
    return _build_catchment_prototypes(raw, source=source)


def _build_catchment_class_table(raw: Mapping[str, Any], *, source: Path) -> CatchmentClassTableProfile:
    """The ``class_table_v2`` arm: a balanced draw from the frozen class table (plan19 §1).

    The declared ``class_table_sha`` is compared with the digest the contract computes
    right here, at load time, rather than when the dataset is written: the table was
    frozen before any model was trained, and a run whose configuration named a
    definition that had since moved would report accuracies against one table and cite
    another. Both digests go into the message, because "they differ" without the two
    values leaves the reader to recompute them.

    Raises:
        ConfigError: on a malformed section, an unknown or missing key, a non-positive
            count, an inverted range, a repeated rain gage, or a ``class_table_sha``
            that is not the frozen table's.
    """
    from stormlab.contracts.catchment_classes import class_table_sha

    section = _c1_mapping(raw, name="catchment_dataset", keys=_CATCHMENT_CLASS_TABLE_KEYS, source=source)

    seed = section["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"catchment_dataset.seed must be a non-negative integer, got {seed!r} (config {source})")
    n_per_class = _positive_int(section["n_per_class"], name="catchment_dataset.n_per_class", source=source)

    gages_raw = _c1_list(section["rain_gages"], name="catchment_dataset.rain_gages", source=source)
    rain_gages = tuple(
        _c1_text(item, name=f"catchment_dataset.rain_gages[{index}]", source=source) for index, item in enumerate(gages_raw)
    )
    if len(set(rain_gages)) != len(rain_gages):
        raise ConfigError(f"catchment_dataset.rain_gages names the same gage twice: {list(rain_gages)} (config {source})")

    area_range = _float_pair(section["area_range"], name="catchment_dataset.area_range", source=source)
    if not 0.0 < area_range[0] < area_range[1]:
        raise ConfigError(
            f"catchment_dataset.area_range must be an increasing pair of positive areas, got {list(area_range)} (config {source})"
        )

    declared = _c1_text(section["class_table_sha"], name="catchment_dataset.class_table_sha", source=source)
    frozen = class_table_sha()
    if declared != frozen:
        raise ConfigError(
            f"catchment_dataset.class_table_sha is {declared!r} but the frozen class table hashes to {frozen!r} "
            f"(config {source}); the dataset and the definition it is labelled by must be the same table"
        )

    return CatchmentClassTableProfile(
        seed=seed,
        n_per_class=n_per_class,
        template=_repo_relative_path(section["template"], name="catchment_dataset.template", source=source),
        rain_gages=rain_gages,
        out_dir=_repo_relative_path(section["out_dir"], name="catchment_dataset.out_dir", source=source),
        area_range=area_range,
        class_table_sha=declared,
    )


def _build_catchment_prototypes(raw: Mapping[str, Any], *, source: Path) -> CatchmentDatasetProfile:
    """The ``prototypes`` arm: how the classifier's subcatchments are drawn (plan18 §9b).

    Everything here is checked at load time rather than at sampling time, for the
    reason the generator profile is: a flat layer of zero rows, or a regime band
    that no subcatchment can satisfy, would otherwise produce a dataset with no
    error at all and a Wrocław share of zero that reads as a measurement.

    The one thing this loader deliberately does **not** check is whether the rain
    gages are declared in the template. That check needs the template parsed, and
    a configuration module that reaches into :mod:`stormlab.generate` to read an
    INP would close an import cycle -- the sampling module reads its profile from
    here. It is made where the template is opened anyway
    (:func:`stormlab.generate.catchment_dataset.write_catchment_dataset`).

    Raises:
        ConfigError: on a malformed section, an unknown or missing key, a
            non-positive count, an inverted range, or a repeated rain gage.
    """
    section = _c1_mapping(raw, name="catchment_dataset", keys=_CATCHMENT_DATASET_KEYS, optional=("generator",), source=source)

    seed = section["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"catchment_dataset.seed must be a non-negative integer, got {seed!r} (config {source})")
    n_base = _positive_int(section["n_base"], name="catchment_dataset.n_base", source=source)
    n_flat = _positive_int(section["n_flat"], name="catchment_dataset.n_flat", source=source)

    gages_raw = _c1_list(section["rain_gages"], name="catchment_dataset.rain_gages", source=source)
    rain_gages = tuple(
        _c1_text(item, name=f"catchment_dataset.rain_gages[{index}]", source=source) for index, item in enumerate(gages_raw)
    )
    if len(set(rain_gages)) != len(rain_gages):
        raise ConfigError(f"catchment_dataset.rain_gages names the same gage twice: {list(rain_gages)} (config {source})")

    area_range = _float_pair(section["area_range"], name="catchment_dataset.area_range", source=source)
    if not 0.0 < area_range[0] < area_range[1]:
        raise ConfigError(
            f"catchment_dataset.area_range must be an increasing pair of positive areas, got {list(area_range)} (config {source})"
        )

    regime_section = _c1_mapping(section["regime"], name="catchment_dataset.regime", keys=_REGIME_KEYS, source=source)
    slope_pct_max = regime_section["slope_pct_max"]
    if isinstance(slope_pct_max, bool) or not isinstance(slope_pct_max, (int, float)) or slope_pct_max <= 0.0:
        raise ConfigError(
            f"catchment_dataset.regime.slope_pct_max must be a positive slope [%], got {slope_pct_max!r} (config {source})"
        )
    imperv_pct = _float_pair(section["regime"]["imperv_pct"], name="catchment_dataset.regime.imperv_pct", source=source)
    if not 0.0 <= imperv_pct[0] < imperv_pct[1] <= 100.0:
        raise ConfigError(
            f"catchment_dataset.regime.imperv_pct must be an increasing band inside [0, 100], got {list(imperv_pct)} "
            f"(config {source}); an inverted band matches nothing and would report a share of zero as a measurement"
        )

    return CatchmentDatasetProfile(
        seed=seed,
        n_base=n_base,
        n_flat=n_flat,
        template=_repo_relative_path(section["template"], name="catchment_dataset.template", source=source),
        rain_gages=rain_gages,
        out_dir=_repo_relative_path(section["out_dir"], name="catchment_dataset.out_dir", source=source),
        area_range=area_range,
        regime=CatchmentRegime(slope_pct_max=float(slope_pct_max), imperv_pct=imperv_pct),
    )


def _repo_relative_path(value: Any, *, name: str, source: Path) -> Path:
    """A path kept relative to the repository root, the way ``generate.template`` is.

    Resolved paths would travel into :func:`catchment_dataset_config_sha` and make
    the digest that names a dataset a property of the machine it was built on.
    """
    text = _c1_text(value, name=name, source=source)
    candidate = Path(text)
    if candidate.is_absolute():
        raise ConfigError(f"{name} must be relative to the repository root, got {text!r} (config {source})")
    return candidate


def _float_pair(value: Any, *, name: str, source: Path) -> tuple[float, float]:
    """A two-element list of real numbers, as YAML writes a closed interval."""
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{name} must be a list of two numbers, got {value!r} (config {source})")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"{name} must hold numbers, got {item!r} (config {source})")
        numbers.append(float(item))
    return (numbers[0], numbers[1])


def _positive_count_mapping(raw: Mapping[str, Any], *, name: str, source: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"{name}.{key} must be a positive integer, got {value!r} (config {source})")
        values[str(key)] = value
    return values


def _build_paths(raw: Any, *, source: Path, base: Path) -> Paths:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'paths' must be a mapping, got {type(raw).__name__} (config {source})")
    unknown = set(raw) - set(_PATH_KEYS)
    if unknown:
        raise ConfigError(f"Unknown keys under 'paths' in {source}: {sorted(unknown)}")
    missing = set(_PATH_KEYS) - set(raw)
    if missing:
        raise ConfigError(f"Missing keys under 'paths' in {source}: {sorted(missing)}")

    resolved: dict[str, Path] = {}
    for key in _PATH_KEYS:
        value = raw[key]
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"paths.{key} must be a non-empty string, got {value!r} (config {source})")
        candidate = Path(value)
        resolved[key] = candidate if candidate.is_absolute() else (base / candidate)
    return Paths(**resolved)


def _build_seeds(raw: Any, *, source: Path) -> Seeds:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'seeds' must be a mapping, got {type(raw).__name__} (config {source})")
    unknown = set(raw) - set(_SEED_KEYS)
    if unknown:
        raise ConfigError(f"Unknown keys under 'seeds' in {source}: {sorted(unknown)}")
    missing = set(_SEED_KEYS) - set(raw)
    if missing:
        raise ConfigError(f"Missing keys under 'seeds' in {source}: {sorted(missing)}")
    values: dict[str, int] = {}
    for key in _SEED_KEYS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"seeds.{key} must be an integer, got {value!r} (config {source})")
        values[key] = value
    return Seeds(**values)


def _build_slope_profile(raw: Any, *, source: Path) -> SlopeProfile:
    if not isinstance(raw, str):
        raise ConfigError(f"slope_profile must be a string, got {type(raw).__name__} (config {source})")
    if raw not in SLOPE_PROFILES:
        raise ConfigError(f"Unknown slope_profile {raw!r} (config {source}); allowed: {sorted(SLOPE_PROFILES)}")
    rules = SLOPE_PROFILES[raw]
    for rule in rules.values():
        if rule not in MIN_SLOPE_RULES:
            raise ConfigError(f"Slope profile {raw!r} references unknown rule {rule!r}; allowed: {sorted(MIN_SLOPE_RULES)}")
    return SlopeProfile(name=raw, **rules)


def _as_float(raw: Mapping[str, Any], key: str, source: Path) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number, got {value!r} (config {source})")
    return float(value)


def _as_optional_run_id(raw: Mapping[str, Any], key: str, source: Path) -> str | None:
    """An optional run identifier: absent, or a non-empty path-free string."""
    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    if not isinstance(value, str) or not value.strip() or "/" in value or "\\" in value:
        raise ConfigError(f"{key} must be a non-empty path-free string, got {value!r} (config {source})")
    return value


def _as_variant(raw: Mapping[str, Any], key: str, source: Path) -> str:
    """The variant name, checked as an identifier rather than against the manifest."""
    value = raw[key]
    if not isinstance(value, str) or not VARIANT_NAME_PATTERN.match(value):
        raise ConfigError(f"{key} must match {VARIANT_NAME_PATTERN.pattern}, got {value!r} (config {source})")
    return value


def _as_choice(raw: Mapping[str, Any], key: str, allowed: Collection[str], source: Path) -> str:
    value = raw[key]
    if value not in allowed:
        raise ConfigError(f"{key} must be one of {sorted(allowed)}, got {value!r} (config {source})")
    return str(value)
