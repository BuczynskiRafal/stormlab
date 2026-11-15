"""The bounded survey of a SWMM file and the structure check built on it."""

from __future__ import annotations

from pathlib import Path

import pytest

from sa.core.errors import AnalysisInputError
from sa.core.inp_structure import check_structure, survey_inp
from sa.core.services import engine_errors

EXAMPLES = Path(__file__).resolve().parents[4] / "examples"

NETWORK = b"""[TITLE]
;; a [CONDUITS] header in a comment does not count
[OPTIONS]
FLOW_UNITS CMS
END_DATE 06/17/2022 ; trailing comment
[SUBCATCHMENTS]
S1 RG J1 1.0 50 100 1 0
[JUNCTIONS]
;;Name Elev
J1 100 2
J2 99 2
[OUTFALLS]
O1 98 FREE
[CONDUITS]
C1 J1 J2 50 0.013 0 0
C2 J2 O1 50 0.013 0 0
[WEIRS]
W1 J2 O1 TRANSVERSE 0.5 3.33
[XSECTIONS]
C1 CIRCULAR 0.3 0 0 0 1
C2 "RECT_CLOSED" 1 1 0 0 1
W1 RECT_OPEN 1 1 0 0
"""


def survey(text: bytes, **kwargs):
    return survey_inp([text], **kwargs)


def test_counts_names_options_and_shapes():
    result = survey(NETWORK)

    assert (result.conduits, result.node_count, result.subcatchments, result.elements) == (2, 3, 1, 6)
    assert result.nodes == {"J1", "J2", "O1"}
    assert [link[:2] for link in result.links] == [("CONDUITS", "C1"), ("CONDUITS", "C2"), ("WEIRS", "W1")]
    assert result.shapes == [("C1", "CIRCULAR"), ("C2", "RECT_CLOSED"), ("W1", "RECT_OPEN")]
    assert result.options == [("FLOW_UNITS", "CMS"), ("END_DATE", "06/17/2022")]
    check_structure(result)


def test_crlf_lower_case_and_abbreviated_headers_are_read_as_the_engine_reads_them():
    text = NETWORK.replace(b"\n", b"\r\n").replace(b"[CONDUITS]", b"[conduit]").replace(b"[XSECTIONS]", b"[XSECTION]")

    result = survey(text)

    assert result.conduits == 2
    assert result.shapes[0] == ("C1", "CIRCULAR")


def test_every_shipped_example_passes_the_structure_check():
    for path in EXAMPLES.glob("*.inp"):
        check_structure(survey(path.read_bytes()))


def test_a_link_to_an_undefined_node_is_named():
    with pytest.raises(AnalysisInputError) as raised:
        check_structure(survey(NETWORK.replace(b"C2 J2 O1", b"C2 J2 JX")))

    assert str(raised.value) == "These links refer to a node the model does not define: C2 → JX."


def test_a_node_spelled_in_another_case_is_pointed_out():
    with pytest.raises(AnalysisInputError, match=r"C1 → j1 \(the model spells it J1\)"):
        check_structure(survey(NETWORK.replace(b"C1 J1 J2", b"C1 j1 J2")))


def test_many_undefined_nodes_are_listed_briefly():
    text = NETWORK + b"[CONDUITS]\n" + b"".join(b"X%d A%d B%d 1 0.013 0 0\n" % (i, i, i) for i in range(5))

    with pytest.raises(AnalysisInputError, match=r"X0 → A0, X0 → B0, X1 → A1 and 7 more\.$"):
        check_structure(survey(text))


def test_a_model_without_conduits_is_refused_plainly():
    with pytest.raises(AnalysisInputError, match="^The model has no conduits"):
        check_structure(survey(b"[OPTIONS]\nFLOW_UNITS CMS\n"))


def test_a_model_without_subcatchments_is_refused_plainly():
    text = NETWORK.replace(b"S1 RG J1 1.0 50 100 1 0\n", b"")

    with pytest.raises(AnalysisInputError, match="^The model has no subcatchments"):
        check_structure(survey(text))


def test_memory_is_bounded_and_counts_stay_exact():
    text = b"[JUNCTIONS]\n" + b"".join(b"J%d 1 1\n" % i for i in range(1000))

    result = survey(text, max_items=10)

    assert result.node_count == 1000
    assert len(result.nodes) == 10
    assert not result.complete


def test_engine_errors_are_read_in_order_without_paths(tmp_path):
    report = tmp_path / "model.rpt"
    report.write_text(
        "  EPA STORM WATER MANAGEMENT MODEL - VERSION 5.2\n"
        "  ERROR 133: Node J3 has more than one outlet link.\n"
        f"  ERROR 209: undefined object JX at line 12 of [CONDUITS] section:\n  K21 C7 JX\n"
        f"  ERROR 303: cannot open file {tmp_path}/model.inp\n"
    )

    lines = engine_errors(report, hide=(str(tmp_path) + "/",))

    assert lines == [
        "ERROR 133: Node J3 has more than one outlet link.",
        "ERROR 209: undefined object JX at line 12 of [CONDUITS] section",
        "ERROR 303: cannot open file model.inp",
    ]
    assert engine_errors(tmp_path / "missing.rpt") == []
