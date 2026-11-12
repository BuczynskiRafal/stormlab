"""The shipped catchment classifier on the Rakowiec network.

``fixtures/catchment_classifier_rakowiec.npz`` holds, for
``examples/wroclaw_rakowiec.inp`` with its subcatchment tags removed, the twelve
model inputs per subcatchment, the reference class probabilities, the assigned
category and the category the file's own tag names.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sa.core.constants import SUBCATCHMENT_CATEGORIES

HAS_TF = importlib.util.find_spec("tensorflow") is not None
requires_tf = pytest.mark.skipif(not HAS_TF, reason="requires tensorflow")

CORE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = CORE_DIR / "catchment_classifier" / "model.keras"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "catchment_classifier_rakowiec.npz"
EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "wroclaw_rakowiec.inp"
MODEL_TOLERANCE = 1e-5
AGREEMENT_WITH_TAGS = 34


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return {key: data[key] for key in data.files}


def test_shipped_classifier_file_is_present() -> None:
    """Runs without TensorFlow, so a checkout missing the model fails even where the model tests skip."""
    assert MODEL_PATH.is_file(), f"catchment classifier missing: {MODEL_PATH}"


@requires_tf
def test_classifier_reproduces_the_reference_probabilities(reference: dict[str, np.ndarray]) -> None:
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH, compile=False)

    probabilities = model.predict(reference["features"], verbose=0)

    np.testing.assert_allclose(probabilities, reference["probabilities"], rtol=0.0, atol=MODEL_TOLERANCE)
    categories = [SUBCATCHMENT_CATEGORIES[index] for index in probabilities.argmax(axis=1)]
    assert categories == reference["category"].tolist()
    assert sum(a == b for a, b in zip(categories, reference["tag"].tolist())) == AGREEMENT_WITH_TAGS


def _without_subcatchment_tags(source: Path, target: Path) -> dict[str, str]:
    """Write ``source`` to ``target`` without subcatchment rows in ``[TAGS]``; return the removed tags."""
    tags: dict[str, str] = {}
    kept: list[str] = []
    section = ""
    for line in source.read_bytes().decode("utf-8").split("\r\n"):
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped.upper()
        if section == "[TAGS]" and stripped.startswith("Subcatch"):
            _, name, tag = stripped.split()
            tags[name] = tag
            continue
        kept.append(line)
    target.write_bytes("\r\n".join(kept).encode("utf-8"))
    return tags


@requires_tf
@pytest.mark.slow
def test_untagged_network_gets_its_categories_from_the_classifier(reference: dict[str, np.ndarray], tmp_path: Path) -> None:
    from sa.core.data import DataManager

    inp = tmp_path / EXAMPLE.name
    tags = _without_subcatchment_tags(EXAMPLE, inp)
    assert len(tags) == len(reference["names"])

    with DataManager(str(inp), zone=0.8) as dm:
        assigned = {str(name): str(category) for name, category in dm.dfs["category"].items()}

    assert assigned == dict(zip(reference["names"].tolist(), reference["category"].tolist()))
    assert sum(assigned[name] == tag for name, tag in tags.items()) == AGREEMENT_WITH_TAGS
