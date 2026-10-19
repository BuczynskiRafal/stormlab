"""Reading SWMM models with swmmio.

A thin, stateless wrapper: the frames produced here are the same ones the legacy
``DataManager`` built in its constructor (subcatchments, nodes, conduits, with
the node id columns cast to ``str`` so downstream merges keep matching), but
without the class, the implicit simulation run and the hidden rounding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import swmmio as sw

NODE_ID_COLUMNS: tuple[str, ...] = ("InletNode", "OutletNode")
TAG_ELEMENT_INDEX_NAME = "ElementType"
RAINGAGE_COLUMN = "Raingage"
"""Column of ``[SUBCATCHMENTS]`` naming the gage that drives each subcatchment."""

LINK_OFFSETS_OPTION = "LINK_OFFSETS"
LINK_OFFSETS_DEPTH = "DEPTH"
LINK_OFFSETS_ELEVATION = "ELEVATION"
LINK_OFFSETS_MODES: tuple[str, ...] = (LINK_OFFSETS_DEPTH, LINK_OFFSETS_ELEVATION)
"""How ``[CONDUITS]`` records ``InOffset``/``OutOffset``.

``DEPTH`` is a height above the invert of the node the end sits in; ``ELEVATION``
is the invert elevation of the pipe end itself. SWMM's own default is ``DEPTH``,
which is what a file that omits the option means.
"""

BARRELS_COLUMN = "Barrels"
DEFAULT_BARRELS = 1.0
"""Parallel pipes per conduit when ``[XSECTIONS]`` does not say.

``Barrels`` is a column of ``[XSECTIONS]``, not of ``[CONDUITS]``: the generator
writes it as the seventh field of the cross-section line
(``generate/template.py::_emit_xsections``), and every conduit of the generated
pool carries ``1``. SWMM treats an omitted value as one barrel, so that is the
default here.
"""

CONDUIT_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "InletNode",
    "OutletNode",
    "Length",
    "Roughness",
    "InOffset",
    "OutOffset",
    "Shape",
    "Geom1",
    BARRELS_COLUMN,
)
"""What :func:`read_conduit_geometry` returns, in order."""


class InpReadError(RuntimeError):
    """Raised when a SWMM model cannot be read or lacks the requested data."""


def load_model(inp: Path | str, *, crs: str | None = None, include_rpt: bool = True) -> sw.Model:
    """Open an INP file with swmmio.

    Args:
        inp: path to the INP file.
        crs: coordinate reference system passed through to swmmio.
        include_rpt: attach the sibling ``.rpt`` so simulated results
            (``MaxQ``, ``MaxV``, runoff volumes) appear in the frames.

    Raises:
        InpReadError: if the file is missing or swmmio cannot parse it.
    """
    path = Path(inp)
    if not path.is_file():
        raise InpReadError(f"INP file {path} does not exist")
    try:
        return sw.Model(str(path), crs=crs, include_rpt=include_rpt)
    except Exception as error:
        raise InpReadError(f"swmmio could not open {path}: {type(error).__name__}: {error}") from error


def read_frames(model: sw.Model, include_rpt: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(dfc, dfn, dfs)`` — conduits, nodes and subcatchments.

    Args:
        model: a model opened by :func:`load_model`.
        include_rpt: assert that simulated results are available. Passing
            ``True`` for a model opened without a valid report is an error
            rather than a silent fall back to INP-only columns.

    Raises:
        InpReadError: if report data was requested but the model has none, or a
            frame could not be built.
    """
    if include_rpt and not _has_valid_rpt(model):
        raise InpReadError(
            f"Model {model.inp.path} has no valid report; open it with include_rpt=True after running SWMM "
            "on the working copy, otherwise the result columns are absent"
        )

    dfc = _frame(model.conduits(), "conduits", model)
    for column in NODE_ID_COLUMNS:
        if column in dfc.columns:
            dfc[column] = dfc[column].astype(str)
    dfn = _frame(model.nodes.dataframe, "nodes", model)
    dfs = _frame(model.subcatchments.dataframe, "subcatchments", model)
    return dfc, dfn, dfs


def read_subcatchments(model: sw.Model, include_rpt: bool = True) -> pd.DataFrame:
    """Return the subcatchment frame alone, without touching the conduit frame.

    The catchment classifier networks are runoff-only: their INP files carry no
    ``[CONDUITS]`` and no ``[XSECTIONS]`` section at all, and ``model.conduits()``
    raises while merging node elevations onto an empty link frame. Asking for the
    three frames together would therefore fail on a model that has everything
    this study needs, so the subcatchment read is available on its own.

    Args:
        model: a model opened by :func:`load_model`.
        include_rpt: assert that simulated results are available, exactly as in
            :func:`read_frames` — the four runoff features come from the report.

    Raises:
        InpReadError: if report data was requested but the model has none, or the
            frame could not be built.
    """
    if include_rpt and not _has_valid_rpt(model):
        raise InpReadError(
            f"Model {model.inp.path} has no valid report; run SWMM on the working copy first, "
            "otherwise the runoff result columns are absent"
        )
    return _frame(model.subcatchments.dataframe, "subcatchments", model)


def read_tags(model: sw.Model) -> pd.Series:
    """Return the ``[TAGS]`` section as a Series indexed by ``(element_type, name)``.

    Element type is kept in the index because subcatchment, node and link tags
    share a namespace in the INP file and names may repeat across types. An INP
    without tags yields an empty Series rather than an error — the Wroclaw
    network legitimately has no conduit tags.
    """
    tags = getattr(model.inp, "tags", None)
    empty_index = pd.MultiIndex.from_arrays([[], []], names=[TAG_ELEMENT_INDEX_NAME, "Name"])
    if tags is None or len(tags) == 0:
        return pd.Series(dtype="object", index=empty_index, name="Tag")

    missing = [column for column in ("Name", "Tag") if column not in tags.columns]
    if missing:
        raise InpReadError(f"[TAGS] section of {model.inp.path} lacks columns {missing}; got {list(tags.columns)}")

    frame = tags.reset_index()
    element_column = frame.columns[0]
    index = pd.MultiIndex.from_arrays(
        [frame[element_column].astype(str), frame["Name"].astype(str)],
        names=[TAG_ELEMENT_INDEX_NAME, "Name"],
    )
    return pd.Series(frame["Tag"].astype(str).to_numpy(), index=index, name="Tag")


def raingage_names(model: sw.Model) -> list[str]:
    """Names declared in the ``[RAINGAGES]`` section, in file order.

    Declaring a gage is not using it. All four Wroclaw INP files declare the same
    six Euler rainfalls and differ only in which one ``[SUBCATCHMENTS]`` points
    at, so an assertion about the rainfall scenario belongs on
    :func:`subcatchment_raingages`, not here.
    """
    raingages = getattr(model.inp, "raingages", None)
    if raingages is None or len(raingages) == 0:
        return []
    return [str(name) for name in raingages.index]


def subcatchment_raingages(model: sw.Model) -> pd.Series:
    """The raingage each subcatchment is driven by, indexed by subcatchment name.

    This is what fixes the rainfall scenario of a simulation: the scenario is a
    property of the file, never an API parameter.

    Raises:
        InpReadError: if the model declares no subcatchments or the section has
            no raingage column.
    """
    subcatchments = getattr(model.inp, "subcatchments", None)
    if subcatchments is None or len(subcatchments) == 0:
        raise InpReadError(f"{model.inp.path} declares no subcatchments, so it has no rainfall scenario")
    frame = _frame(subcatchments, "subcatchments", model)
    if RAINGAGE_COLUMN not in frame.columns:
        raise InpReadError(f"[SUBCATCHMENTS] of {model.inp.path} has no {RAINGAGE_COLUMN!r} column; got {list(frame.columns)}")
    return pd.Series(
        [str(value) for value in frame[RAINGAGE_COLUMN]],
        index=pd.Index([str(name) for name in frame.index], name="Name"),
        name=RAINGAGE_COLUMN,
        dtype="object",
    )


def read_options(model: sw.Model) -> dict[str, str]:
    """The ``[OPTIONS]`` section as an upper-cased key to raw value mapping.

    Values are returned as written. Only the caller knows whether a given option
    is a number, a duration or a keyword.
    """
    options = getattr(model.inp, "options", None)
    if options is None or len(options) == 0:
        return {}
    if "Value" not in options.columns:
        raise InpReadError(f"[OPTIONS] of {model.inp.path} has no 'Value' column; got {list(options.columns)}")
    return {str(key).strip().upper(): str(value).strip() for key, value in options["Value"].items()}


def link_offsets_mode(model: sw.Model) -> str:
    """Whether conduit offsets are depths above the node invert or elevations.

    A file that does not set ``LINK_OFFSETS`` means ``DEPTH``, which is SWMM's
    default, so the absent option is answered rather than refused.

    Raises:
        InpReadError: if the option carries a value SWMM does not define.
    """
    value = read_options(model).get(LINK_OFFSETS_OPTION, LINK_OFFSETS_DEPTH).upper()
    if value not in LINK_OFFSETS_MODES:
        raise InpReadError(
            f"[OPTIONS] {LINK_OFFSETS_OPTION} of {model.inp.path} is {value!r}; expected one of {list(LINK_OFFSETS_MODES)}"
        )
    return value


def read_nodes(model: sw.Model, include_rpt: bool = True) -> pd.DataFrame:
    """Return the node frame alone, without building the conduit frame.

    The physical view needs node inverts, manhole depths and the outfall columns
    but computes conduit geometry from ``[CONDUITS]`` and ``[XSECTIONS]``
    directly, so asking for the three frames of :func:`read_frames` would pay for
    a conduit merge it does not use.

    Raises:
        InpReadError: if report data was requested but the model has none, or the
            frame could not be built.
    """
    if include_rpt and not _has_valid_rpt(model):
        raise InpReadError(
            f"Model {model.inp.path} has no valid report; run SWMM on the working copy first, "
            "otherwise the result columns are absent"
        )
    return _frame(model.nodes.dataframe, "nodes", model)


def read_conduit_geometry(model: sw.Model) -> pd.DataFrame:
    """Conduit geometry as laid down in the file, offsets and barrels included.

    Joins ``[CONDUITS]`` with ``[XSECTIONS]`` on the link name, in the row order
    of ``[CONDUITS]``. This is deliberately not ``model.conduits()``: that frame
    drops ``InOffset``, ``OutOffset`` and ``Barrels`` and derives a slope from
    the node inverts alone, which is the same as assuming the offsets are zero.

    Returns:
        A frame indexed by conduit name with :data:`CONDUIT_GEOMETRY_COLUMNS`.
        Offsets are returned exactly as the file records them; converting them to
        a common meaning is :func:`offsets_as_depth`.

    Raises:
        InpReadError: if either section is missing, a conduit has no
            cross-section, a conduit name repeats, or a numeric field is not a
            number.
    """
    conduits = getattr(model.inp, "conduits", None)
    if conduits is None or len(conduits) == 0:
        raise InpReadError(f"{model.inp.path} declares no conduits")
    xsections = getattr(model.inp, "xsections", None)
    if xsections is None or len(xsections) == 0:
        raise InpReadError(f"{model.inp.path} declares no cross-sections, so its conduits have no geometry")

    frame = _frame(conduits, "conduits", model)
    sections = _frame(xsections, "xsections", model)
    names = [str(name) for name in frame.index]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise InpReadError(f"[CONDUITS] of {model.inp.path} names {duplicates} more than once")

    missing_columns = [column for column in ("InletNode", "OutletNode", "Length", "Roughness") if column not in frame.columns]
    if missing_columns:
        raise InpReadError(f"[CONDUITS] of {model.inp.path} lacks columns {missing_columns}; got {list(frame.columns)}")
    if "Geom1" not in sections.columns:
        raise InpReadError(f"[XSECTIONS] of {model.inp.path} lacks 'Geom1'; got {list(sections.columns)}")

    sections = sections.rename(index=str)
    without_section = [name for name in names if name not in sections.index]
    if without_section:
        raise InpReadError(
            f"{len(without_section)} conduit(s) of {model.inp.path} have no [XSECTIONS] row: {without_section[:5]}"
        )

    result = pd.DataFrame(index=pd.Index(names, name="Name"))
    for column in ("InletNode", "OutletNode"):
        result[column] = [str(value) for value in frame[column]]
    for column in ("Length", "Roughness"):
        result[column] = _numeric(frame[column].to_numpy(), column, model)
    for column in ("InOffset", "OutOffset"):
        # A file may omit the offsets entirely; SWMM reads an absent offset as 0.
        raw = frame[column].to_numpy() if column in frame.columns else np.zeros(len(names))
        result[column] = _numeric(raw, column, model)
    aligned = sections.loc[names]
    result["Shape"] = [str(value) for value in aligned["Shape"]] if "Shape" in aligned.columns else "UNKNOWN"
    result["Geom1"] = _numeric(aligned["Geom1"].to_numpy(), "Geom1", model)
    barrels = aligned[BARRELS_COLUMN].to_numpy() if BARRELS_COLUMN in aligned.columns else np.full(len(names), DEFAULT_BARRELS)
    result[BARRELS_COLUMN] = _numeric(barrels, BARRELS_COLUMN, model, default=DEFAULT_BARRELS)
    return result.loc[:, list(CONDUIT_GEOMETRY_COLUMNS)]


def offsets_as_depth(offsets: pd.Series, node_inverts: pd.Series, *, mode: str) -> pd.Series:
    """Conduit-end offsets expressed as a height above the node invert [m].

    ``DEPTH`` files already record that height. ``ELEVATION`` files record the
    invert elevation of the pipe end, whose height above the node invert is the
    difference. Both spellings therefore describe the same geometry, and this is
    the one place the two are reconciled -- everything downstream works with a
    single meaning.

    Args:
        offsets: the offsets as written, indexed by conduit name.
        node_inverts: the invert elevation of the node at that end of each
            conduit, on the same index.
        mode: the file's ``LINK_OFFSETS`` setting.

    Raises:
        InpReadError: if ``mode`` is not a SWMM offset mode.
    """
    if mode not in LINK_OFFSETS_MODES:
        raise InpReadError(f"Unknown LINK_OFFSETS mode {mode!r}; expected one of {list(LINK_OFFSETS_MODES)}")
    if mode == LINK_OFFSETS_DEPTH:
        return offsets.astype("float64")
    return (offsets.astype("float64") - node_inverts.astype("float64")).astype("float64")


def _numeric(values: object, column: str, model: sw.Model, *, default: float | None = None) -> np.ndarray:
    """Cast a raw INP column to float, naming the file and the column on failure.

    Raises:
        InpReadError: if a value is not a number and no default covers it. A
            silently coerced ``NaN`` would travel into a geometry computation and
            surface much later as an unexplained missing feature.
    """
    series = pd.to_numeric(pd.Series(values), errors="coerce").astype("float64")
    if default is not None:
        series = series.fillna(default)
    if series.isna().any():
        raise InpReadError(f"Column {column!r} of {model.inp.path} carries {int(series.isna().sum())} non-numeric value(s)")
    return series.to_numpy()


def _has_valid_rpt(model: sw.Model) -> bool:
    """Whether the model carries a parsable report with simulated results."""
    if getattr(model, "rpt", None) is None:
        return False
    return bool(model.rpt_is_valid())


def _frame(source: object, what: str, model: sw.Model) -> pd.DataFrame:
    frame = getattr(source, "dataframe", source)
    if not isinstance(frame, pd.DataFrame):
        raise InpReadError(f"swmmio returned {type(frame).__name__} instead of a DataFrame for {what} of {model.inp.path}")
    return frame.copy()
