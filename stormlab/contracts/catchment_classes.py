"""The catchment class definition as a table of parameter bands, with the provenance of every band end.

Three different things have been called "the class of a subcatchment" in this
project (plan19 "Stan zastany" §1): the fuzzy successor the author's ``rcg``
engine writes into ``[TAGS]``, the band a network generator draws
``PercImperv``/``PercSlope`` from, and the land-use judgement behind the tags of
the one real network. A classifier trained on one of them and measured against
another cannot succeed, which is what the negative result of plan18 stage 9
measured. This module is the answer to that: **one** written definition, twelve
classes, seven INP columns each, and a recorded source for every single end of
every band -- so that "what does ``urban_weakly_impervious`` mean" has a place
to be answered from rather than being an emergent property of a sampler.

The table is a literal on purpose. Six of the seven columns could be derived
mechanically (plan19 §0.1) from ``stormlab.generate.grid.CATEGORY_PARAMS`` and
``stormlab.generate.catchment_prototypes.SUBAREA_BASES``, but this module sits
in the contracts layer and must not import ``generate``; more importantly, a
table a hydrologist can read line by line is the artifact the author approves,
and a derivation is not. That the literal *does* satisfy the mechanical rules is
a statement the contract tests make by importing both generators and comparing.

Two reference rules are defined here next to the table, both closed and both
reported (plan19 §0.3). :func:`table_rule_7` reads all seven columns and always
answers; :func:`band_rule_2` reads the two columns the series-C generator draws
per class and abstains when no class contains the point. Neither is a model:
they are the yardstick a trained classifier is read against, and
:func:`bayes_ceiling` turns the first one into the accuracy ceiling the table's
own overlaps impose. Every threshold and every band end here comes from a
generator constant, the SWMM manual, or a stated assumption -- none was chosen
to improve a score on any network.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from stormlab.contracts.catchment_features import CLASSIFIER_CLASS_ORDER

CLASS_COLUMNS: tuple[str, ...] = (
    "PercImperv",
    "PercSlope",
    "N-Imperv",
    "N-Perv",
    "S-Imperv",
    "S-Perv",
    "PctZero",
)
"""The seven INP columns a class band is written over, in contract order.

They are the first seven of :data:`CATCHMENT_FEATURE_COLUMNS` -- the classifier's
remaining five features are RPT quantities, which under plan19 [Z]5 carry no
class information beyond these seven, so a ceiling computed here is a ceiling for
the twelve-feature model as well. Units: ``%``, ``%``, --, --, mm, mm, ``%``.
"""

BAND_COLUMNS_2: tuple[str, ...] = CLASS_COLUMNS[:2]
"""The two columns :func:`band_rule_2` reads: the only ones the series-C generator draws per class."""

AMBIGUOUS = "ambiguous"
"""Prediction written when several classes are equally good; the candidate set is the real answer."""

OUTSIDE = "outside"
"""Prediction written by :func:`band_rule_2` when no class contains the point at all."""

IMPERV_DELTA_PP = 2.5
"""Half-width added to each end of the ``CATEGORY_PARAMS`` imperviousness band (plan19 D3)."""

PCT_ZERO_DELTA_PP = 10.0
"""Half-width added to each end of the ``PctZero`` range spanned by the two generators (plan19 D3)."""

FLAT_FAMILY_SLOPE_PCT: tuple[float, float] = (0.1, 1.0)
"""The slope band ``config.C3.yaml:120-124`` overrides ``UW``/``UM``/``UH``/``SH`` with.

It is not a physical statement about those classes: it is the closure of the
slope gap of the Wroclaw network measured in plan18 §7d, declared as a pool
property. The table inherits its lower end for the four urban/suburban classes
and says so in ``source_low``, because a band whose origin is one particular real
network must not look like a band read off a manual.
"""

IMPERV_ORACLE_THRESHOLDS: tuple[float, float, float] = (42.1, 53.1, 66.6)
"""Thresholds of the in-sample optimum inside the family of three ordered cuts on ``PercImperv``.

Fitted on the 37 subcatchments of the Wroclaw (Rakowiec) network on 2026-09-10
(``rev/findings/plan19-checks/rak_ceiling.py``). This is an upper reference **for
that family of rules**; it is not a ceiling for other rule families and not a
ceiling for any model. It is in-sample by construction -- the same 37 rows chose
the cuts and are scored by them -- so it belongs next to a result as a yardstick,
never as a baseline anyone is asked to beat honestly.
"""

IMPERV_ORACLE_CLASSES: tuple[str, ...] = (
    "suburban_weakly_impervious",
    "urban_weakly_impervious",
    "urban_moderately_impervious",
    "urban_highly_impervious",
)
"""The four classes the oracle's ordered cuts separate, from least to most sealed."""

SH_UM_MERGE: Mapping[str, str] = {
    "suburban_highly_impervious": "SH_UM",
    "urban_moderately_impervious": "SH_UM",
}
"""Merged reading of the two classes that are indistinguishable **inside the overlap** of their bands (plan19 D2).

``SH`` and ``UM`` differ by a 5 pp shift of the imperviousness band, by the lower
end of ``N-Perv`` and by the ``PctZero`` band, so they are not globally identical;
but their two-column boxes have exactly equal area, which makes every point of the
overlap a tie for :func:`band_rule_2`. Reporting the merged figure next to the full
twelve-class one states that cost instead of hiding it behind a tie-break.
"""


class CatchmentClassError(ValueError):
    """Raised when a class table, a prior or a frame of rows does not satisfy this contract."""


@dataclass(frozen=True)
class Band:
    """One closed interval of one column, with the origin of each end recorded separately.

    Both ends carry their own source string because they do not always share one:
    the lower end of ``PercSlope`` for the flat family comes from a pool override
    justified by one real network, while the upper end comes from the generator
    constant (plan19 R1). Sources are part of the frozen digest -- changing where a
    number came from is a change to the table, even when the number stays.
    """

    low: float
    high: float
    source_low: str
    source_high: str

    def __post_init__(self) -> None:
        if not (self.low < self.high):
            raise CatchmentClassError(
                f"Band low must be strictly below high, got low={self.low!r} high={self.high!r} (source_low={self.source_low!r})"
            )
        if not str(self.source_low).strip() or not str(self.source_high).strip():
            raise CatchmentClassError(
                f"Band [{self.low!r}, {self.high!r}] has an empty source; every band end must record where it came from "
                f"(source_low={self.source_low!r}, source_high={self.source_high!r})"
            )

    @property
    def width(self) -> float:
        """``high - low``; the factor this column contributes to the class volume."""
        return self.high - self.low


@dataclass(frozen=True)
class ClassBands:
    """The seven bands of one class, one per column of :data:`CLASS_COLUMNS`."""

    imperv: Band
    slope: Band
    n_imperv: Band
    n_perv: Band
    s_imperv_mm: Band
    s_perv_mm: Band
    pct_zero: Band

    def __post_init__(self) -> None:
        for column, band in zip(CLASS_COLUMNS, self.bands()):
            if not isinstance(band, Band):
                raise CatchmentClassError(f"Column {column!r} of a class must hold a Band, got {type(band).__name__}")

    def bands(self) -> tuple[Band, ...]:
        """The seven bands in :data:`CLASS_COLUMNS` order."""
        return (self.imperv, self.slope, self.n_imperv, self.n_perv, self.s_imperv_mm, self.s_perv_mm, self.pct_zero)


_TABLE_A5 = "SWMM 5 User's Manual App. A Table A.5 Depression storage"
_TABLE_A6 = "SWMM 5 User's Manual App. A Table A.6 Manning's n overland flow"
_TO_CONFIRM = " (table number to be confirmed by the author)"
_ASSIGNED = "; [A] surface type assigned to class by plan19 §0.1"
_ASSUMED_CATEGORY = "; [A] assumed in grid.py, unconfirmed"

N_IMPERV_BAND = Band(
    0.011,
    0.024,
    f"{_TABLE_A6}: smooth asphalt 0.011{_TO_CONFIRM}",
    f"{_TABLE_A6}: corrugated metal 0.024{_TO_CONFIRM}",
)
"""One impervious-roughness band for all twelve classes: the surface is paving whatever the land cover is."""

S_IMPERV_MM_BAND = Band(
    1.27,
    2.54,
    f"{_TABLE_A5}: impervious 0.05 in = 1.27 mm{_TO_CONFIRM}",
    f"{_TABLE_A5}: impervious 0.10 in = 2.54 mm{_TO_CONFIRM}",
)
"""One impervious depression-storage band for all twelve classes, in millimetres."""


def _imperv_band(low: float, high: float, *, category: str, assumed: bool) -> Band:
    """``CATEGORY_PARAMS`` imperviousness band widened by :data:`IMPERV_DELTA_PP`, clipped to ``[0, 100]``."""
    note = _ASSUMED_CATEGORY if assumed else ""
    low_source = f"grid.py CATEGORY_PARAMS {category} perc_imperv_low - {IMPERV_DELTA_PP} pp"
    if low == 0.0:
        low_source += ", clipped to 0"
    high_source = f"grid.py CATEGORY_PARAMS {category} perc_imperv_high + {IMPERV_DELTA_PP} pp"
    if high == 100.0:
        high_source += ", clipped to 100"
    return Band(low, high, low_source + note, high_source + note)


def _slope_band(low: float, high: float, *, category: str, flat_family: bool) -> Band:
    """``CATEGORY_PARAMS`` slope band, with the flat-family override substituted for the lower end where it applies."""
    if flat_family:
        low_source = (
            f"config.C3.yaml category_params_override perc_slope {FLAT_FAMILY_SLOPE_PCT[0]} "
            "(flat family, Wroclaw regime plan18 §7d)"
        )
    else:
        low_source = f"grid.py CATEGORY_PARAMS {category} perc_slope_low"
    return Band(low, high, low_source, f"grid.py CATEGORY_PARAMS {category} perc_slope_high")


def _n_perv_band(low: float, high: float, low_surface: str, high_surface: str) -> Band:
    """Pervious roughness from table A.6; which surface belongs to which class is a stated assumption."""
    return Band(
        low,
        high,
        f"{_TABLE_A6}: {low_surface}{_TO_CONFIRM}{_ASSIGNED}",
        f"{_TABLE_A6}: {high_surface}{_TO_CONFIRM}{_ASSIGNED}",
    )


def _s_perv_band(low: float, high: float, low_surface: str, high_surface: str) -> Band:
    """Pervious depression storage from table A.5 in millimetres; the surface assignment is a stated assumption."""
    return Band(
        low,
        high,
        f"{_TABLE_A5}: {low_surface}{_TO_CONFIRM}{_ASSIGNED}",
        f"{_TABLE_A5}: {high_surface}{_TO_CONFIRM}{_ASSIGNED}",
    )


def _pct_zero_band(low: float, high: float, *, rcg: float, grid: float) -> Band:
    """Range spanned by the two generators' ``PctZero`` constants, widened by :data:`PCT_ZERO_DELTA_PP`.

    The SWMM manual has no ``PctZero`` table, so unlike the roughness and storage
    columns this one has no literature end to quote: the two generator constants
    are the whole evidence, and equal constants (arable 10/10) still produce a band
    because a single point is not a band.
    """
    low_source = f"min(SUBAREA_BASES {rcg:g}, CATEGORY_PARAMS {grid:g}) - {PCT_ZERO_DELTA_PP} pp"
    if low == 0.0:
        low_source += ", clipped to 0"
    high_source = f"max(SUBAREA_BASES {rcg:g}, CATEGORY_PARAMS {grid:g}) + {PCT_ZERO_DELTA_PP} pp"
    if high == 100.0:
        high_source += ", clipped to 100"
    return Band(low, high, low_source, high_source)


CATCHMENT_CLASS_TABLE_V2: Mapping[str, ClassBands] = {
    "marshes": ClassBands(
        imperv=_imperv_band(0.0, 5.5, category="marshes", assumed=True),
        slope=_slope_band(0.5, 3.0, category="marshes", flat_family=False),
        n_imperv=N_IMPERV_BAND,
        n_perv=_n_perv_band(0.13, 0.80, "range (natural) 0.13", "woods with dense underbrush 0.80"),
        s_imperv_mm=S_IMPERV_MM_BAND,
        s_perv_mm=_s_perv_band(5.08, 7.62, "pasture 0.20 in = 5.08 mm", "forest litter 0.30 in = 7.62 mm"),
        pct_zero=_pct_zero_band(0.0, 20.0, rcg=10.0, grid=5.0),
    ),
    "arable": ClassBands(
        imperv=_imperv_band(0.5, 12.5, category="arable", assumed=True),
        slope=_slope_band(1.0, 8.0, category="arable", flat_family=False),
        n_imperv=N_IMPERV_BAND,
        n_perv=_n_perv_band(
            0.06, 0.17, "cultivated soils, residue cover < 20% 0.06", "cultivated soils, residue cover > 20% 0.17"
        ),
        s_imperv_mm=S_IMPERV_MM_BAND,
        s_perv_mm=_s_perv_band(2.54, 5.08, "lawns 0.10 in = 2.54 mm", "pasture 0.20 in = 5.08 mm"),
        pct_zero=_pct_zero_band(0.0, 20.0, rcg=10.0, grid=10.0),
    ),
    "meadows": ClassBands(
        imperv=_imperv_band(0.0, 10.5, category="meadows", assumed=True),
        slope=_slope_band(1.0, 10.0, category="meadows", flat_family=False),
        n_imperv=N_IMPERV_BAND,
        n_perv=_n_perv_band(0.13, 0.41, "range (natural) 0.13", "bermuda grass 0.41"),
        s_imperv_mm=S_IMPERV_MM_BAND,
        s_perv_mm=_s_perv_band(2.54, 5.08, "lawns 0.10 in = 2.54 mm", "pasture 0.20 in = 5.08 mm"),
        pct_zero=_pct_zero_band(0.0, 20.0, rcg=10.0, grid=5.0),
    ),
    "forests": ClassBands(
        imperv=_imperv_band(0.0, 7.5, category="forests", assumed=True),
        slope=_slope_band(1.0, 15.0, category="forests", flat_family=False),
        n_imperv=N_IMPERV_BAND,
        n_perv=_n_perv_band(0.40, 0.80, "woods with light underbrush 0.40", "woods with dense underbrush 0.80"),
        s_imperv_mm=S_IMPERV_MM_BAND,
        s_perv_mm=_s_perv_band(5.08, 7.62, "pasture 0.20 in = 5.08 mm", "forest litter 0.30 in = 7.62 mm"),
        pct_zero=_pct_zero_band(0.0, 15.0, rcg=5.0, grid=5.0),
    ),
    "rural": ClassBands(
        imperv=_imperv_band(7.5, 32.5, category="rural", assumed=False),
        slope=_slope_band(1.0, 10.0, category="rural", flat_family=False),
        n_imperv=N_IMPERV_BAND,
        n_perv=_n_perv_band(0.15, 0.41, "short grass prairie 0.15", "bermuda grass 0.41"),
        s_imperv_mm=S_IMPERV_MM_BAND,
        s_perv_mm=_s_perv_band(2.54, 5.08, "lawns 0.10 in = 2.54 mm", "pasture 0.20 in = 5.08 mm"),
        pct_zero=_pct_zero_band(0.0, 45.0, rcg=35.0, grid=10.0),
    ),
    "suburban_weakly_impervious": ClassBands(
        imperv=_imperv_band(17.5, 42.5, category="suburban_weakly_impervious", assumed=False),
        slope=_slope_band(1.0, 8.0, category="suburban_weakly_impervious", flat_family=False),
        n_imperv=N_IMPERV_BAND,
        n_perv=_n_perv_band(0.15, 0.41, "short grass prairie 0.15", "bermuda grass 0.41"),
        s_imperv_mm=S_IMPERV_MM_BAND,
        s_perv_mm=_s_perv_band(2.54, 5.08, "lawns 0.10 in = 2.54 mm", "pasture 0.20 in = 5.08 mm"),
        pct_zero=_pct_zero_band(0.0, 50.0, rcg=40.0, grid=10.0),
    ),
    "suburban_highly_impervious": ClassBands(
        imperv=_imperv_band(47.5, 72.5, category="suburban_highly_impervious", assumed=False),
        slope=_slope_band(0.1, 5.0, category="suburban_highly_impervious", flat_family=True),
        n_imperv=N_IMPERV_BAND,
        n_perv=_n_perv_band(0.15, 0.41, "short grass prairie 0.15", "bermuda grass 0.41"),
        s_imperv_mm=S_IMPERV_MM_BAND,
        s_perv_mm=_s_perv_band(2.54, 5.08, "lawns 0.10 in = 2.54 mm", "pasture 0.20 in = 5.08 mm"),
        pct_zero=_pct_zero_band(30.0, 90.0, rcg=40.0, grid=80.0),
    ),
    "urban_weakly_impervious": ClassBands(
        imperv=_imperv_band(37.5, 62.5, category="urban_weakly_impervious", assumed=False),
        slope=_slope_band(0.1, 10.0, category="urban_weakly_impervious", flat_family=True),
        n_imperv=N_IMPERV_BAND,
        n_perv=_n_perv_band(0.15, 0.41, "short grass prairie 0.15", "bermuda grass 0.41"),
        s_imperv_mm=S_IMPERV_MM_BAND,
        s_perv_mm=_s_perv_band(2.54, 5.08, "lawns 0.10 in = 2.54 mm", "pasture 0.20 in = 5.08 mm"),
        pct_zero=_pct_zero_band(0.0, 60.0, rcg=50.0, grid=10.0),
    ),
    "urban_moderately_impervious": ClassBands(
        imperv=_imperv_band(52.5, 77.5, category="urban_moderately_impervious", assumed=False),
        slope=_slope_band(0.1, 5.0, category="urban_moderately_impervious", flat_family=True),
        n_imperv=N_IMPERV_BAND,
        n_perv=_n_perv_band(0.13, 0.41, "range (natural) 0.13", "bermuda grass 0.41"),
        s_imperv_mm=S_IMPERV_MM_BAND,
        s_perv_mm=_s_perv_band(2.54, 5.08, "lawns 0.10 in = 2.54 mm", "pasture 0.20 in = 5.08 mm"),
        pct_zero=_pct_zero_band(0.0, 60.0, rcg=50.0, grid=5.0),
    ),
    "urban_highly_impervious": ClassBands(
        imperv=_imperv_band(67.5, 97.5, category="urban_highly_impervious", assumed=False),
        slope=_slope_band(0.1, 5.0, category="urban_highly_impervious", flat_family=True),
        n_imperv=N_IMPERV_BAND,
        n_perv=_n_perv_band(0.13, 0.24, "range (natural) 0.13", "dense grass 0.24"),
        s_imperv_mm=S_IMPERV_MM_BAND,
        s_perv_mm=_s_perv_band(2.54, 5.08, "lawns 0.10 in = 2.54 mm", "pasture 0.20 in = 5.08 mm"),
        pct_zero=_pct_zero_band(40.0, 90.0, rcg=50.0, grid=80.0),
    ),
    "mountains_rocky": ClassBands(
        imperv=_imperv_band(37.5, 82.5, category="mountains_rocky", assumed=True),
        slope=_slope_band(5.0, 40.0, category="mountains_rocky", flat_family=False),
        n_imperv=N_IMPERV_BAND,
        n_perv=_n_perv_band(0.05, 0.13, "fallow soils, no residue 0.05", "range (natural) 0.13"),
        s_imperv_mm=S_IMPERV_MM_BAND,
        s_perv_mm=_s_perv_band(1.27, 2.54, "impervious 0.05 in = 1.27 mm", "impervious 0.10 in = 2.54 mm"),
        pct_zero=_pct_zero_band(0.0, 70.0, rcg=10.0, grid=60.0),
    ),
    "mountains_vegetated": ClassBands(
        imperv=_imperv_band(2.5, 22.5, category="mountains_vegetated", assumed=True),
        slope=_slope_band(5.0, 30.0, category="mountains_vegetated", flat_family=False),
        n_imperv=N_IMPERV_BAND,
        n_perv=_n_perv_band(0.24, 0.80, "dense grass 0.24", "woods with dense underbrush 0.80"),
        s_imperv_mm=S_IMPERV_MM_BAND,
        s_perv_mm=_s_perv_band(5.08, 7.62, "pasture 0.20 in = 5.08 mm", "forest litter 0.30 in = 7.62 mm"),
        pct_zero=_pct_zero_band(0.0, 30.0, rcg=10.0, grid=20.0),
    ),
}
"""Twelve classes as seven closed parameter bands each -- the definition plan19 freezes before any training.

Six of the twelve rows carry ``[A] assumed in grid.py, unconfirmed`` on their
imperviousness band: those categories never existed in the legacy generator and
their bands are the plan author's physical guess. The ``[A]`` marker in
``grid.py`` covers the whole ``CategoryParams`` row, so the slope and ``PctZero``
ends of those six are equally assumed even though the marker is written only
where plan19 §0.1 asks for it.
"""


def _class_names(table: Mapping[str, ClassBands]) -> tuple[str, ...]:
    """Classes of ``table`` in :data:`CLASSIFIER_CLASS_ORDER`, rejecting names outside the vocabulary.

    A subset is allowed on purpose: the rules are exercised on two-class tables
    built to reason about ties. The order is taken from the classifier contract and
    never from the mapping's insertion order, because it is also the order candidate
    sets are reported in and a set whose order depends on how a dict was typed is not
    reproducible.

    Raises:
        CatchmentClassError: if the table is empty or names a class that is not one of the twelve.
    """
    unknown = sorted(set(table) - set(CLASSIFIER_CLASS_ORDER))
    if unknown:
        raise CatchmentClassError(
            f"Unknown class name(s) in the class table: {unknown}; expected a subset of {list(CLASSIFIER_CLASS_ORDER)}"
        )
    names = tuple(name for name in CLASSIFIER_CLASS_ORDER if name in table)
    if not names:
        raise CatchmentClassError("The class table is empty; at least one class is required")
    return names


def class_table_sha(table: Mapping[str, ClassBands] = CATCHMENT_CLASS_TABLE_V2) -> str:
    """sha256 of the canonical JSON of the table, sources included.

    The recipe is spelled out so that anyone can recompute it: a JSON array with
    one object per class in ``CLASSIFIER_CLASS_ORDER``, each holding its columns in
    :data:`CLASS_COLUMNS` order as ``{column, low, high, source_low, source_high}``
    with the two bounds formatted to six decimal places, dumped with
    ``separators=(",", ":")``, ``sort_keys=True`` and ``ensure_ascii=False``, encoded
    as UTF-8.

    Bounds are formatted rather than passed as floats because ``json`` writes the
    shortest round-tripping repr, which would make the digest depend on the platform's
    float formatting; sources are inside the digest because a band end whose origin
    changed is a different band end even when the number is untouched.
    """
    payload = [
        {
            "class": name,
            "columns": [
                {
                    "column": column,
                    "low": f"{band.low:.6f}",
                    "high": f"{band.high:.6f}",
                    "source_low": band.source_low,
                    "source_high": band.source_high,
                }
                for column, band in zip(CLASS_COLUMNS, table[name].bands())
            ],
        }
        for name in _class_names(table)
    ]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def class_table_frame(table: Mapping[str, ClassBands] = CATCHMENT_CLASS_TABLE_V2) -> pd.DataFrame:
    """The table as one long frame -- one row per class and column -- for reports and appendices."""
    records = [
        {
            "class": name,
            "column": column,
            "low": band.low,
            "high": band.high,
            "source_low": band.source_low,
            "source_high": band.source_high,
        }
        for name in _class_names(table)
        for column, band in zip(CLASS_COLUMNS, table[name].bands())
    ]
    return pd.DataFrame.from_records(records, columns=["class", "column", "low", "high", "source_low", "source_high"])


def _bounds(table: Mapping[str, ClassBands], names: Sequence[str], n_columns: int) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds of the first ``n_columns`` columns as ``(len(names), n_columns)`` arrays."""
    lows = np.array([[band.low for band in table[name].bands()[:n_columns]] for name in names], dtype=float)
    highs = np.array([[band.high for band in table[name].bands()[:n_columns]] for name in names], dtype=float)
    return lows, highs


def _feature_matrix(rows: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """The named columns of ``rows`` as a float matrix, refusing anything a rule cannot decide on.

    Raises:
        CatchmentClassError: if ``rows`` is not a frame, a column is missing or duplicated,
            a column is not numeric, or any value is NaN or infinite. A rule that silently
            treats NaN as "outside every band" would report a confident nearest class for a
            row it never read.
    """
    if not isinstance(rows, pd.DataFrame):
        raise CatchmentClassError(f"Expected a DataFrame of rows to classify, got {type(rows).__name__}")

    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise CatchmentClassError(
            f"Missing column(s) {missing}; the rule reads {list(columns)} and the frame has {list(rows.columns)}"
        )

    selected = rows.loc[:, list(columns)]
    if selected.shape[1] != len(columns):
        raise CatchmentClassError(
            f"Column(s) {list(columns)} appear more than once in the frame with columns {list(rows.columns)}"
        )

    try:
        matrix = selected.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        dtypes = {column: str(selected[column].dtype) for column in columns}
        raise CatchmentClassError(f"Columns {list(columns)} must be numeric; got dtypes {dtypes}") from error

    non_finite = {column: int((~np.isfinite(matrix[:, index])).sum()) for index, column in enumerate(columns)}
    offending = {column: count for column, count in non_finite.items() if count}
    if offending:
        raise CatchmentClassError(f"Non-finite values in {offending}; a rule cannot place NaN or inf inside or outside a band")
    return matrix


def _prior_vector(priors: Mapping[str, float] | None, names: Sequence[str]) -> np.ndarray:
    """Class priors in ``names`` order, defaulting to the uniform prior over the table's classes.

    Raises:
        CatchmentClassError: if the keys are not exactly the table's classes, a value is
            not positive, or the values do not sum to one.
    """
    if priors is None:
        return np.full(len(names), 1.0 / len(names), dtype=float)

    missing = sorted(set(names) - set(priors))
    extra = sorted(set(priors) - set(names))
    if missing or extra:
        raise CatchmentClassError(
            f"Priors must name exactly the {len(names)} classes of the table; missing {missing}, unexpected {extra}"
        )

    values = np.array([float(priors[name]) for name in names], dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
        raise CatchmentClassError(f"Priors must be finite and positive, got {dict(zip(names, values.tolist()))}")
    total = float(values.sum())
    if abs(total - 1.0) > 1e-9:
        raise CatchmentClassError(f"Priors must sum to 1, they sum to {total!r}")
    return values


def _candidate_sets(
    scores: np.ndarray, names: Sequence[str], *, supported: np.ndarray, tolerance: float
) -> list[tuple[str, ...]]:
    """Per row, every class whose score is within ``tolerance`` of the best one, in table order.

    Ties are kept as sets rather than broken. Breaking them by class order would
    manufacture a decision the model behind the rule does not make: equal posteriors
    are a real ambiguity of the table, and the count of them is one of the things
    plan19 §0.3 asks to report.
    """
    best = np.where(supported, scores.max(axis=1), 0.0)
    within = np.abs(scores - best[:, None]) <= tolerance * np.abs(best[:, None])
    mask = supported[:, None] & within
    counts = mask.sum(axis=1)
    first = np.argmax(mask, axis=1)
    name_array = np.array(names, dtype=object)
    return [
        (str(names[int(first[row])]),) if counts[row] == 1 else tuple(str(name) for name in name_array[mask[row]])
        for row in range(scores.shape[0])
    ]


def table_rule_7(
    rows: pd.DataFrame,
    *,
    table: Mapping[str, ClassBands] = CATCHMENT_CLASS_TABLE_V2,
    priors: Mapping[str, float] | None = None,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Reference rule A: the Bayes decision of the model "class ~ priors, parameters uniform in the class box".

    Three steps, exactly as plan19 §0.3 writes them. ``S(x)`` is the set of classes
    whose seven closed bands all contain ``x``; when it is non-empty the winner is
    ``argmax pi_c / V_c``, where ``V_c`` is the product of the seven widths -- the
    narrower support wins, which is the posterior of the stated model and not a
    preference added on top. When ``S(x)`` is empty the answer is the nearest class
    under ``d_c = sum_j max(0, l_cj - x_j, x_j - h_cj) / (h_cj - l_cj)``, a
    width-normalised out-of-box distance, and the row is flagged: the rule still
    answers, but the answer is an extrapolation and must be read as one.

    The rule is a **continuous** approximation of a generator that rounds its output
    (plan19 R6): measured on the frozen table, the continuous rule and the rule on
    exact post-rounding masses disagree on 0.046 % of rows, so the approximation is
    stated rather than hidden.

    Args:
        rows: frame carrying :data:`CLASS_COLUMNS`; extra columns are ignored.
        table: class definition; defaults to the frozen one.
        priors: class prior; defaults to uniform over the table's classes.
        tolerance: relative width of the tie window around the best score.

    Returns:
        Frame on ``rows``' index with ``candidates`` (tuple of equally good classes in
        table order), ``prediction`` (the class when there is exactly one candidate,
        else :data:`AMBIGUOUS`), ``n_containing`` (``|S(x)|``), ``outside_support``,
        ``tie`` and ``score``. ``score`` is the winner's ``pi_c / V_c`` for a row inside
        at least one box and ``-d_c`` of the nearest class for a row outside all of
        them; the two are on different scales and comparable only within a group.

    Raises:
        CatchmentClassError: on a malformed frame, table or prior.
    """
    names = _class_names(table)
    matrix = _feature_matrix(rows, CLASS_COLUMNS)
    lows, highs = _bounds(table, names, len(CLASS_COLUMNS))
    prior = _prior_vector(priors, names)

    contained = np.all((matrix[:, None, :] >= lows[None, :, :]) & (matrix[:, None, :] <= highs[None, :, :]), axis=2)
    n_containing = contained.sum(axis=1)
    supported = n_containing > 0

    density = prior / np.prod(highs - lows, axis=1)
    scores = np.where(contained, density[None, :], -np.inf)

    if not supported.all():
        widths = highs - lows
        below = np.maximum(0.0, lows[None, :, :] - matrix[:, None, :])
        above = np.maximum(0.0, matrix[:, None, :] - highs[None, :, :])
        distance = ((below + above) / widths[None, :, :]).sum(axis=2)
        scores = np.where(supported[:, None], scores, -distance)

    candidates = _candidate_sets(scores, names, supported=np.ones(len(matrix), dtype=bool), tolerance=tolerance)
    return _decision_frame(rows.index, candidates, n_containing, scores=scores.max(axis=1), outside_prediction=None)


def band_rule_2(
    rows: pd.DataFrame,
    *,
    table: Mapping[str, ClassBands] = CATCHMENT_CLASS_TABLE_V2,
    priors: Mapping[str, float] | None = None,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Reference rule B: the same decision on the two columns the series-C generator actually draws, with abstention.

    ``PercImperv`` and ``PercSlope`` are the only quantities ``CATEGORY_PARAMS``
    samples per class; the five ``[SUBAREAS]`` columns are constants attached to a
    class there, and flat-rate constants in real files, so they are not part of the
    definition this rule reads. That is a property of the generator, decided before
    any measurement, not a column dropped because it scored badly.

    Where rule A always answers, this one abstains: a point inside no class box gets
    :data:`OUTSIDE` and an empty candidate set, because "no class of this table
    contains this subcatchment" is a more honest answer than the nearest box.

    Returns:
        Frame on ``rows``' index with ``candidates``, ``prediction`` (a class,
        :data:`AMBIGUOUS`, or :data:`OUTSIDE`), ``n_containing``, ``outside_support``
        and ``tie``.

    Raises:
        CatchmentClassError: on a malformed frame, table or prior.
    """
    names = _class_names(table)
    matrix = _feature_matrix(rows, BAND_COLUMNS_2)
    lows, highs = _bounds(table, names, len(BAND_COLUMNS_2))
    prior = _prior_vector(priors, names)

    contained = np.all((matrix[:, None, :] >= lows[None, :, :]) & (matrix[:, None, :] <= highs[None, :, :]), axis=2)
    n_containing = contained.sum(axis=1)
    supported = n_containing > 0

    density = prior / np.prod(highs - lows, axis=1)
    scores = np.where(contained, density[None, :], -np.inf)

    candidates = _candidate_sets(scores, names, supported=supported, tolerance=tolerance)
    return _decision_frame(rows.index, candidates, n_containing, scores=None, outside_prediction=OUTSIDE)


def _decision_frame(
    index: pd.Index,
    candidates: Sequence[tuple[str, ...]],
    n_containing: np.ndarray,
    *,
    scores: np.ndarray | None,
    outside_prediction: str | None,
) -> pd.DataFrame:
    """Assemble the frame both rules return, so their columns cannot drift apart."""
    predictions = [
        outside_prediction
        if not candidate and outside_prediction is not None
        else candidate[0]
        if len(candidate) == 1
        else AMBIGUOUS
        for candidate in candidates
    ]
    data = {
        "candidates": pd.Series(list(candidates), index=index, dtype=object),
        "prediction": pd.Series(predictions, index=index, dtype=object),
        "n_containing": pd.Series(n_containing.astype(int), index=index),
        "outside_support": pd.Series(n_containing == 0, index=index),
        "tie": pd.Series([len(candidate) > 1 for candidate in candidates], index=index),
    }
    if scores is not None:
        data["score"] = pd.Series(scores.astype(float), index=index)
    return pd.DataFrame(data, index=index)


@dataclass(frozen=True)
class CeilingEstimate:
    """Accuracy of rule A on draws from the table itself, with the sampling error of the estimate."""

    accuracy: float
    standard_error: float
    n: int
    seed: int
    tie_rate: float
    per_class: Mapping[str, float]


def bayes_ceiling(
    *,
    table: Mapping[str, ClassBands] = CATCHMENT_CLASS_TABLE_V2,
    priors: Mapping[str, float] | None = None,
    n_per_class: int,
    seed: int,
) -> CeilingEstimate:
    """Monte-Carlo estimate of how often rule A recovers the class a row was drawn from.

    This is the ceiling any classifier of this definition faces on synthetic data:
    where two class boxes overlap, the row carries no information that would settle
    which of them it came from, and the shortfall from 1.0 is that overlap, not noise
    and not a training failure. It is an estimate, so it comes with
    ``standard_error = sqrt(p(1-p)/n)`` and the seed that produced it; reporting the
    figure without the error would give a Monte-Carlo wobble the authority of a bound.

    Rows are drawn without rounding, i.e. against the continuous reading of the rule.
    The generator that will build the training set rounds its output, and plan19 §0.3
    measured the gap that opens between the two readings -- 0.8669 against 0.8666, with
    0.046 % of decisions differing -- which is why the continuous number is used and the
    approximation named. Under plan19 [Z]5 the remaining five classifier features carry
    no class information beyond these seven columns, so a ceiling in R^7 is a ceiling
    for the twelve-feature model too.

    Args:
        table: class definition to draw from and decide with.
        priors: prior used by the rule; the draw itself is balanced across classes.
        n_per_class: rows drawn per class; the estimate's ``n`` is this times the class count.
        seed: seed of ``numpy.random.default_rng``.

    Raises:
        CatchmentClassError: if ``n_per_class`` is not positive, or the table or prior is malformed.
    """
    if n_per_class < 1:
        raise CatchmentClassError(f"n_per_class must be at least 1, got {n_per_class!r}")

    names = _class_names(table)
    lows, highs = _bounds(table, names, len(CLASS_COLUMNS))
    rng = np.random.default_rng(seed)
    # One `uniform` call per class, in table order: the stream is part of the result,
    # and a different call pattern would give a different (equally valid) sample that
    # no longer reproduces the figure recorded in the manifest.
    draws = np.vstack(
        [rng.uniform(lows[position], highs[position], size=(n_per_class, len(CLASS_COLUMNS))) for position in range(len(names))]
    )

    frame = pd.DataFrame(draws, columns=list(CLASS_COLUMNS))
    decided = table_rule_7(frame, table=table, priors=priors)

    truth = np.repeat(np.array(names, dtype=object), n_per_class)
    hit = decided["prediction"].to_numpy() == truth
    accuracy = float(hit.mean())
    total = int(len(names) * n_per_class)
    per_class = {
        name: float(hit[position * n_per_class : (position + 1) * n_per_class].mean()) for position, name in enumerate(names)
    }
    return CeilingEstimate(
        accuracy=accuracy,
        standard_error=float(np.sqrt(accuracy * (1.0 - accuracy) / total)),
        n=total,
        seed=seed,
        tie_rate=float(decided["tie"].mean()),
        per_class=per_class,
    )


@dataclass(frozen=True)
class BandRuleMetrics:
    """What rule B scored against a set of true labels, ties and abstentions counted separately."""

    strict_accuracy: float
    set_accuracy: float
    tie_rate: float
    outside_rate: float
    n: int


def band_rule_metrics(
    result: pd.DataFrame,
    truth: pd.Series,
    *,
    merge: Mapping[str, str] | None = None,
) -> BandRuleMetrics:
    """Score a rule's candidate sets against true labels, optionally under a class merge.

    Two accuracies are reported because a rule that answers with a set has two honest
    readings: ``strict_accuracy`` counts only rows where a single class came back and it
    was right, ``set_accuracy`` counts rows whose true class is among the candidates. A
    tie and an abstention are both failures of the strict reading and are also reported
    on their own, so the three numbers say *how* the rule failed and not only how often.

    ``merge`` (e.g. :data:`SH_UM_MERGE`) is applied to the candidate sets **and** to the
    truth before anything is counted, so a ``{SH, UM}`` tie collapses into the single
    label ``SH_UM`` and counts as a strict hit whenever the truth is either of the two.
    ``tie_rate`` is therefore the tie rate *after* merging; ``outside_rate`` is unaffected,
    since merging cannot fill an empty candidate set.

    Raises:
        CatchmentClassError: if the frame lacks ``candidates``, the index of ``truth``
            differs from the result's, or a class name is outside the vocabulary.
    """
    if "candidates" not in result.columns:
        raise CatchmentClassError(f"Rule result must carry a 'candidates' column; got {list(result.columns)}")
    if not result.index.equals(truth.index):
        raise CatchmentClassError("The truth series must be indexed exactly like the rule result; align them before scoring")
    if len(result) == 0:
        raise CatchmentClassError("Cannot score an empty rule result; a rate over zero rows is not a number")
    if not result.index.is_unique:
        duplicated = sorted({str(key) for key in result.index[result.index.duplicated()]})[:10]
        raise CatchmentClassError(
            f"Rule result index must be unique to pair every row with its truth; duplicated keys: {duplicated}"
        )

    mapping = dict(merge or {})
    unknown_keys = sorted(set(mapping) - set(CLASSIFIER_CLASS_ORDER))
    if unknown_keys:
        raise CatchmentClassError(
            f"Merge maps unknown class name(s) {unknown_keys}; expected a subset of {list(CLASSIFIER_CLASS_ORDER)}"
        )
    unknown_truth = sorted(set(truth.astype(str)) - set(CLASSIFIER_CLASS_ORDER))
    if unknown_truth:
        raise CatchmentClassError(
            f"Truth carries unknown class name(s) {unknown_truth}; expected a subset of {list(CLASSIFIER_CLASS_ORDER)}"
        )

    strict = 0
    in_set = 0
    ties = 0
    outside = 0
    for candidates, true_label in zip(result["candidates"], truth.astype(str)):
        merged = tuple(dict.fromkeys(mapping.get(name, name) for name in candidates))
        actual = mapping.get(true_label, true_label)
        if not merged:
            outside += 1
            continue
        if len(merged) > 1:
            ties += 1
        elif merged[0] == actual:
            strict += 1
        if actual in merged:
            in_set += 1

    total = int(len(result))
    return BandRuleMetrics(
        strict_accuracy=strict / total,
        set_accuracy=in_set / total,
        tie_rate=ties / total,
        outside_rate=outside / total,
        n=total,
    )


def imperv_oracle(perc_imperv: pd.Series) -> pd.Series:
    """Label subcatchments with the three ordered cuts of :data:`IMPERV_ORACLE_THRESHOLDS`.

    The cuts are the in-sample optimum of that family on the 37 subcatchments of the
    Wroclaw (Rakowiec) network, fitted 2026-09-10 by
    ``rev/findings/plan19-checks/rak_ceiling.py``. They are an upper reference **for
    this family of rules** and nothing else: not a ceiling for other rule families,
    not a ceiling for a model, and not a baseline in any out-of-sample sense, since
    the rows that chose the cuts are the rows it is scored on.

    Raises:
        CatchmentClassError: if the series is not numeric or holds a non-finite value.
    """
    try:
        values = perc_imperv.astype(float)
    except (TypeError, ValueError) as error:
        raise CatchmentClassError(f"The oracle needs a numeric 'PercImperv' series; got dtype {perc_imperv.dtype}") from error
    if not np.isfinite(values.to_numpy()).all():
        raise CatchmentClassError("The oracle cannot place a non-finite 'PercImperv' value against a threshold")

    first, second, third = IMPERV_ORACLE_THRESHOLDS
    labels = np.select(
        [values < first, values < second, values < third],
        list(IMPERV_ORACLE_CLASSES[:3]),
        default=IMPERV_ORACLE_CLASSES[3],
    )
    return pd.Series(labels, index=values.index, dtype=object, name="prediction")


__all__ = [
    "AMBIGUOUS",
    "BAND_COLUMNS_2",
    "Band",
    "BandRuleMetrics",
    "CATCHMENT_CLASS_TABLE_V2",
    "CLASS_COLUMNS",
    "CatchmentClassError",
    "CeilingEstimate",
    "ClassBands",
    "FLAT_FAMILY_SLOPE_PCT",
    "IMPERV_DELTA_PP",
    "IMPERV_ORACLE_CLASSES",
    "IMPERV_ORACLE_THRESHOLDS",
    "N_IMPERV_BAND",
    "OUTSIDE",
    "PCT_ZERO_DELTA_PP",
    "SH_UM_MERGE",
    "S_IMPERV_MM_BAND",
    "band_rule_2",
    "band_rule_metrics",
    "bayes_ceiling",
    "class_table_frame",
    "class_table_sha",
    "imperv_oracle",
    "table_rule_7",
]
