"""Conduit offsets, the two ways of writing them, and the barrel count."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stormlab.io.inp import (
    DEFAULT_BARRELS,
    InpReadError,
    link_offsets_mode,
    load_model,
    offsets_as_depth,
    read_conduit_geometry,
    read_nodes,
    read_options,
)
from stormlab.tests.unit.physical_fixtures import write_inp

JUNCTIONS = (
    {"name": "J0", "invert": 100.0, "max_depth": 3.0},
    {"name": "J1", "invert": 99.0, "max_depth": 3.0},
)
OUTFALLS = ({"name": "O1", "invert": 98.0},)


def _network(
    path: Path,
    *,
    in_offset: float = 0.0,
    out_offset: float = 0.0,
    link_offsets: str = "DEPTH",
    barrels: int = 1,
    second: tuple[float, float] = (0.0, 0.0),
) -> Path:
    """A two-conduit chain whose first conduit carries the offsets under test.

    ``second`` is the offset pair of the downstream conduit. It is a parameter
    because an ELEVATION file has to state every end as an elevation: SWMM reads
    a literal ``0`` there as "the pipe end lies at sea level", not as "no offset".
    """
    return write_inp(
        path,
        junctions=list(JUNCTIONS),
        outfalls=list(OUTFALLS),
        conduits=[
            {
                "name": "C0",
                "inlet": "J0",
                "outlet": "J1",
                "length": 50.0,
                "diameter": 0.4,
                "in_offset": in_offset,
                "out_offset": out_offset,
                "barrels": barrels,
            },
            {
                "name": "C1",
                "inlet": "J1",
                "outlet": "O1",
                "length": 50.0,
                "diameter": 0.4,
                "in_offset": second[0],
                "out_offset": second[1],
            },
        ],
        link_offsets=link_offsets,
    )


def _pipe_inverts(path: Path) -> pd.DataFrame:
    """Both pipe-end inverts of every conduit, whichever way the file writes them."""
    model = load_model(path, include_rpt=False)
    geometry = read_conduit_geometry(model)
    inverts = read_nodes(model, include_rpt=False)["InvertElev"]
    inverts.index = [str(name) for name in inverts.index]
    mode = link_offsets_mode(model)
    inlet = geometry["InletNode"].astype(str).map(inverts)
    outlet = geometry["OutletNode"].astype(str).map(inverts)
    return pd.DataFrame(
        {
            "inlet_pipe_invert": inlet + offsets_as_depth(geometry["InOffset"], inlet, mode=mode),
            "outlet_pipe_invert": outlet + offsets_as_depth(geometry["OutOffset"], outlet, mode=mode),
        }
    )


class TestTheTwoWaysOfWritingAnOffset:
    def test_depth_and_elevation_describe_the_same_geometry(self, tmp_path: Path) -> None:
        as_depth = _network(tmp_path / "depth.inp", in_offset=0.3, out_offset=0.2, link_offsets="DEPTH")
        # The same pipe, written as elevations: every end is the node invert plus
        # the rise the depth file states (J0 100.0, J1 99.0, O1 98.0).
        as_elevation = _network(
            tmp_path / "elev.inp",
            in_offset=100.3,
            out_offset=99.2,
            link_offsets="ELEVATION",
            second=(99.0, 98.0),
        )

        pd.testing.assert_frame_equal(_pipe_inverts(as_depth), _pipe_inverts(as_elevation))

    def test_a_raised_end_lifts_the_pipe_without_moving_the_manhole(self, tmp_path: Path) -> None:
        flat = _pipe_inverts(_network(tmp_path / "flat.inp", in_offset=0.0, out_offset=0.0, link_offsets="DEPTH"))
        raised = _pipe_inverts(_network(tmp_path / "raised.inp", in_offset=0.3, out_offset=0.0, link_offsets="DEPTH"))
        depths = read_nodes(load_model(tmp_path / "raised.inp", include_rpt=False), include_rpt=False)["MaxDepth"]

        assert raised.loc["C0", "inlet_pipe_invert"] - flat.loc["C0", "inlet_pipe_invert"] == pytest.approx(0.3)
        assert raised.loc["C0", "outlet_pipe_invert"] == flat.loc["C0", "outlet_pipe_invert"]
        assert depths.loc["J0"] == 3.0

    def test_offsets_as_depth_refuses_a_mode_swmm_does_not_define(self) -> None:
        offsets = pd.Series([0.0], index=["C0"])

        with pytest.raises(InpReadError, match="Unknown LINK_OFFSETS mode"):
            offsets_as_depth(offsets, offsets, mode="RELATIVE")


class TestTheLinkOffsetsOption:
    def test_a_file_that_does_not_set_it_means_depth(self, tmp_path: Path) -> None:
        path = _network(tmp_path / "silent.inp", in_offset=0.0, out_offset=0.0, link_offsets="DEPTH")
        path.write_text(
            "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if "LINK_OFFSETS" not in line),
            encoding="utf-8",
        )

        assert link_offsets_mode(load_model(path, include_rpt=False)) == "DEPTH"

    def test_an_unknown_value_is_refused_rather_than_treated_as_a_depth(self, tmp_path: Path) -> None:
        path = _network(tmp_path / "odd.inp", in_offset=0.0, out_offset=0.0, link_offsets="DEPTH")
        path.write_text(path.read_text(encoding="utf-8").replace("LINK_OFFSETS         DEPTH", "LINK_OFFSETS         MIDDLE"))

        with pytest.raises(InpReadError, match="LINK_OFFSETS"):
            link_offsets_mode(load_model(path, include_rpt=False))

    def test_options_are_keyed_by_the_upper_cased_option_name(self, tmp_path: Path) -> None:
        model = load_model(_network(tmp_path / "o.inp", in_offset=0.0, out_offset=0.0, link_offsets="DEPTH"), include_rpt=False)

        options = read_options(model)

        assert options["FLOW_ROUTING"] == "KINWAVE"
        assert options["ROUTING_STEP"] == "0:01:00"


class TestBarrels:
    def test_a_double_barrel_conduit_keeps_its_two_barrels(self, tmp_path: Path) -> None:
        path = _network(tmp_path / "two.inp", in_offset=0.0, out_offset=0.0, link_offsets="DEPTH", barrels=2)

        geometry = read_conduit_geometry(load_model(path, include_rpt=False))

        assert geometry.loc["C0", "Barrels"] == 2.0
        assert geometry.loc["C1", "Barrels"] == DEFAULT_BARRELS

    def test_a_cross_section_line_without_a_barrel_field_means_one_barrel(self, tmp_path: Path) -> None:
        path = _network(tmp_path / "short.inp", in_offset=0.0, out_offset=0.0, link_offsets="DEPTH")
        without_barrels = "\n".join(
            line.rsplit(maxsplit=1)[0] if line.startswith(("C0 ", "C1 ")) and "CIRCULAR" in line else line
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        path.write_text(without_barrels, encoding="utf-8")

        geometry = read_conduit_geometry(load_model(path, include_rpt=False))

        assert list(geometry["Barrels"]) == [DEFAULT_BARRELS, DEFAULT_BARRELS]


class TestRefusals:
    def test_a_conduit_without_a_cross_section_is_named(self, tmp_path: Path) -> None:
        path = _network(tmp_path / "gap.inp", in_offset=0.0, out_offset=0.0, link_offsets="DEPTH")
        path.write_text(
            "\n".join(
                line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("C1               CIRCULAR")
            ),
            encoding="utf-8",
        )

        with pytest.raises(InpReadError, match="no \\[XSECTIONS\\] row"):
            read_conduit_geometry(load_model(path, include_rpt=False))

    def test_a_model_without_conduits_is_refused_by_name(self, tmp_path: Path) -> None:
        path = _network(tmp_path / "empty.inp", in_offset=0.0, out_offset=0.0, link_offsets="DEPTH")
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(line for line in lines if not line.startswith(("C0 ", "C1 "))), encoding="utf-8")

        with pytest.raises(InpReadError, match="declares no conduits"):
            read_conduit_geometry(load_model(path, include_rpt=False))
