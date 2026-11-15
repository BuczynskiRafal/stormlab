"""How the physical extraction reads a subcatchment tag that names no land cover class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stormlab.config import load_config
from stormlab.features.physical import PhysicalFeatureError, extract_physical_features
from stormlab.tests.unit.physical_fixtures import write_inp, write_rpt

JUNCTIONS = (
    {"name": "J0", "invert": 100.0, "max_depth": 2.5},
    {"name": "J1", "invert": 99.5, "max_depth": 2.5},
)
OUTFALLS = ({"name": "O1", "invert": 99.0},)
CONDUITS = (
    {"name": "C0", "inlet": "J0", "outlet": "J1", "length": 50.0, "diameter": 0.4},
    {"name": "C1", "inlet": "J1", "outlet": "O1", "length": 50.0, "diameter": 0.4},
)
CONTEXT = ["sbc_category_code", "sbc_missing", "subcatchment_area_ha", "subcatchment_missing"]


def _case(tmp_path: Path, tag: str) -> tuple[Path, Path]:
    case = tmp_path / tag.strip().replace(" ", "_")
    case.mkdir()
    inp = write_inp(case / "net.inp", junctions=list(JUNCTIONS), outfalls=list(OUTFALLS), conduits=list(CONDUITS), tag=tag)
    rpt = write_rpt(
        case / "net.rpt",
        links={"C0": (0.05, 1.1, 0.4, 0.5), "C1": (0.05, 1.0, 0.4, 0.5)},
        nodes={"J0": (0.1, 0.3, 100.3, 0.3), "J1": (0.1, 0.3, 99.8, 0.3), "O1": (0.1, 0.2, 99.2, 0.2)},
    )
    return inp, rpt


def _extract(paths: tuple[Path, Path], **kwargs):
    return extract_physical_features(*paths, config=load_config(check_inputs=False), file_id="t", **kwargs)


def test_by_default_a_foreign_tag_refuses_the_case(tmp_path):
    with pytest.raises(PhysicalFeatureError, match="Osiedle_A"):
        _extract(_case(tmp_path, "Osiedle_A"))


def test_read_as_untagged_a_foreign_tag_sets_sbc_missing_like_no_tag(tmp_path):
    foreign = _extract(_case(tmp_path, "Osiedle_A"), foreign_tags_untagged=True)
    untagged = _extract(_case(tmp_path, "-"), foreign_tags_untagged=True)

    assert foreign["sbc_missing"].tolist() == [1, 1]
    assert np.isnan(foreign["sbc_category_code"]).all()
    assert foreign[CONTEXT].equals(untagged[CONTEXT])


def test_read_as_untagged_a_class_name_in_another_case_is_that_class(tmp_path):
    shouted = _extract(_case(tmp_path, "RURAL"), foreign_tags_untagged=True)
    canonical = _extract(_case(tmp_path, "rural"))

    assert shouted["sbc_missing"].tolist() == [0, 0]
    assert shouted[CONTEXT].equals(canonical[CONTEXT])
