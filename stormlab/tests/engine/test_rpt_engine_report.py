"""The sentences SWMM prints for an empty summary table, read from a real report.

``Conduit Surcharge Summary`` and ``Node Flooding Summary`` become one sentence
when nothing qualifies, and the parser recognises that sentence by its exact
text. The parser tests build their reports by hand, so only a report the engine
wrote can show that the text is still the one SWMM uses -- if it changed, an empty
section would read as a missing table.

None of the shipped networks is dry under its own rainfall, so the tests run a
copy of each with the rainfall halved. The original is simulated too, to show that
the same network yields populated tables, which keeps the empty result from being
an accident of the parser.

The engine writes ``.rpt``/``.out`` next to its input, so everything runs on
copies in ``tmp_path``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pyswmm import Simulation

from stormlab.io import rpt

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
NETWORKS = ("demo_network_coordinates.inp", "demo_network_schematic.inp")

RAINFALL_FACTOR = 0.5
_TIMESERIES_ROW = re.compile(r"(?m)^(test_series\s+\S+\s+)([0-9.]+)")

pytestmark = pytest.mark.slow


def _simulate(source: Path, target_dir: Path, *, rainfall_factor: float = 1.0) -> str:
    """Run a copy of ``source`` in ``target_dir`` and return the text of its report."""
    text = source.read_bytes().decode("utf-8")
    scaled, rows = _TIMESERIES_ROW.subn(lambda m: f"{m.group(1)}{float(m.group(2)) * rainfall_factor:.3f}", text)
    assert rows > 0, f"{source.name} has no rainfall rows to scale"

    inp = target_dir / source.name
    inp.write_bytes(scaled.encode("utf-8"))
    with Simulation(str(inp)) as simulation:
        for _ in simulation:
            pass
    return inp.with_suffix(".rpt").read_text(encoding="utf-8", errors="replace")


@pytest.mark.parametrize("network", NETWORKS)
def test_an_empty_exception_table_is_the_sentence_swmm_writes(network: str, tmp_path: Path) -> None:
    text = _simulate(EXAMPLES / network, tmp_path, rainfall_factor=RAINFALL_FACTOR)

    assert rpt.NO_CONDUITS_SURCHARGED in text
    assert rpt.NO_NODES_FLOODED in text


@pytest.mark.parametrize("network", NETWORKS)
def test_the_parser_reads_those_sections_as_empty_not_as_missing(network: str, tmp_path: Path) -> None:
    text = _simulate(EXAMPLES / network, tmp_path, rainfall_factor=RAINFALL_FACTOR)

    tables = rpt.parse_report(text, source=network)

    assert tables.conduit_surcharge.empty
    assert tables.node_flooding.empty
    assert list(tables.conduit_surcharge.columns) == list(rpt.CONDUIT_SURCHARGE_COLUMNS)
    assert list(tables.node_flooding.columns) == list(rpt.NODE_FLOODING_COLUMNS)
    assert not tables.link_flow.empty
    assert not tables.node_depth.empty


@pytest.mark.parametrize("network", NETWORKS)
def test_the_unscaled_network_has_populated_tables_and_no_sentences(network: str, tmp_path: Path) -> None:
    text = _simulate(EXAMPLES / network, tmp_path)

    tables = rpt.parse_report(text, source=network)

    assert rpt.NO_CONDUITS_SURCHARGED not in text
    assert rpt.NO_NODES_FLOODED not in text
    assert not tables.conduit_surcharge.empty
    assert not tables.node_flooding.empty
