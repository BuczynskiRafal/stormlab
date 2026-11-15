"""One bounded pass over a SWMM input file: what the upload gates and the structure check need.

The pass reads the file the way the engine does (see :mod:`sa.core.inp_safety`: lines
end at ``\\n``, ``;`` starts a comment, quoted tokens, a NUL ends a line, section
headers match by prefix and case-insensitively) and keeps only:

* the number of data rows in each section of interest;
* the ``[OPTIONS]`` lines, in file order (the engine lets a later line win);
* the names of the nodes, the two ends of every link and the shape of every cross-section.

It never builds a frame and never parses numbers, so a broken model gets as far as a
plain sentence instead of an exception from swmmio. Names are kept up to ``max_items``
in total; past that the survey is marked incomplete and only the counts stay exact, which
is enough for the element limit to refuse the file.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from .errors import AnalysisInputError
from .inp_safety import _lines, _tokens

SECTIONS = {
    b"[OPTION": "OPTIONS",
    b"[JUNCTION": "JUNCTIONS",
    b"[OUTFALL": "OUTFALLS",
    b"[STORAGE": "STORAGE",
    b"[DIVIDER": "DIVIDERS",
    b"[CONDUIT": "CONDUITS",
    b"[PUMP": "PUMPS",
    b"[ORIFICE": "ORIFICES",
    b"[WEIR": "WEIRS",
    b"[OUTLET": "OUTLETS",
    b"[XSECTION": "XSECTIONS",
    b"[SUBCATCHMENT": "SUBCATCHMENTS",
}
"""Header prefixes, as the engine spells them, of the sections read here. None is a
prefix of another, so the first match is the only one."""

NODE_SECTIONS = ("JUNCTIONS", "OUTFALLS", "STORAGE", "DIVIDERS")
LINK_SECTIONS = ("CONDUITS", "PUMPS", "ORIFICES", "WEIRS", "OUTLETS")

MAX_ITEMS = 200_000
"""Names and option lines kept in memory at most; far above any model the element limit admits."""

MAX_LISTED = 3
"""How many offending elements a message names before it says how many more there are."""


@dataclass
class InpSurvey:
    rows: Counter = field(default_factory=Counter)
    options: list[tuple[str, str | None]] = field(default_factory=list)
    nodes: set[str] = field(default_factory=set)
    links: list[tuple[str, str, str, str]] = field(default_factory=list)
    """``(section, name, inlet node, outlet node)`` of every link, in file order."""
    shapes: list[tuple[str, str]] = field(default_factory=list)
    """``(link name, shape keyword as written, upper-cased)`` of every ``[XSECTIONS]`` row."""
    complete: bool = True

    @property
    def conduits(self) -> int:
        return self.rows["CONDUITS"]

    @property
    def node_count(self) -> int:
        return sum(self.rows[section] for section in NODE_SECTIONS)

    @property
    def subcatchments(self) -> int:
        return self.rows["SUBCATCHMENTS"]

    @property
    def elements(self) -> int:
        """Conduits, nodes and subcatchments: the elements every analysis step works on."""
        return self.conduits + self.node_count + self.subcatchments


def _section(header: bytes) -> str | None:
    upper = header.upper()
    return next((name for prefix, name in SECTIONS.items() if upper.startswith(prefix)), None)


def _text(token: bytes) -> str:
    return token.decode("utf-8", errors="replace")


def survey_inp(chunks: Iterable[bytes], *, max_items: int = MAX_ITEMS) -> InpSurvey:
    """Survey the file given as byte chunks; memory stays bounded by ``max_items``."""
    survey = InpSurvey()
    section: str | None = None
    kept = 0
    for line in _lines(chunks):
        if line is None:
            continue  # a line the engine would split; the safety scan refuses such a file
        tokens = _tokens(line)
        if not tokens:
            continue
        if tokens[0].startswith(b"["):
            section = _section(tokens[0])
            continue
        if section is None:
            continue
        survey.rows[section] += 1
        if kept >= max_items:
            survey.complete = False
            continue
        kept += 1
        if section == "OPTIONS":
            survey.options.append((_text(tokens[0]).upper(), _text(tokens[1]) if len(tokens) > 1 else None))
        elif section in NODE_SECTIONS:
            survey.nodes.add(_text(tokens[0]))
        elif section in LINK_SECTIONS and len(tokens) >= 3:
            survey.links.append((section, _text(tokens[0]), _text(tokens[1]), _text(tokens[2])))
        elif section == "XSECTIONS" and len(tokens) >= 2:
            survey.shapes.append((_text(tokens[0]), _text(tokens[1]).upper()))
    return survey


def listing(items: list[str]) -> str:
    """``"A, B, C and 4 more"``: the first few of *items*, then how many are left."""
    shown = ", ".join(items[:MAX_LISTED])
    rest = len(items) - MAX_LISTED
    return f"{shown} and {rest} more" if rest > 0 else shown


def check_structure(survey: InpSurvey) -> None:
    """Refuse, in plain words, a model the analysis cannot be built on.

    Raises:
        AnalysisInputError: when the model has no conduits, no subcatchments, or a link
            whose end is a node the model does not define.
    """
    if survey.conduits == 0:
        raise AnalysisInputError("The model has no conduits; StormLab assesses the conduits of a storm sewer network.")
    if survey.subcatchments == 0:
        # The checks and the recommendation model read the runoff of the catchments,
        # and the engine writes no runoff results for a model without one.
        raise AnalysisInputError(
            "The model has no subcatchments; StormLab analyses networks loaded by rainfall runoff "
            "and needs at least one subcatchment."
        )
    if not survey.complete:
        return  # too large to list every name; the element limit refuses such a file anyway
    known = survey.nodes
    by_case = {name.casefold(): name for name in known}
    undefined: list[str] = []
    for _, link, *ends in survey.links:
        for node in ends:
            if node in known:
                continue
            spelled = by_case.get(node.casefold())
            hint = f" (the model spells it {spelled})" if spelled else ""
            undefined.append(f"{link} → {node}{hint}")
    if undefined:
        raise AnalysisInputError(f"These links refer to a node the model does not define: {listing(undefined)}.")


def check_file_structure(path: str, *, chunk_size: int = 64 * 1024) -> None:
    """:func:`check_structure` on the file at *path*."""
    with open(path, "rb") as handle:
        check_structure(survey_inp(iter(lambda: handle.read(chunk_size), b"")))
