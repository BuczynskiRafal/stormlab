"""Parity of the shipped GNN recommendation model.

The model was trained on a labelled dataset of storm-sewer conduits and is served
as the mean of the class probabilities of five training runs. Two fixtures pin
its outputs, both recorded with the reference implementation the runs were
trained with:

* ``fixtures/gnn_demo_parity.npz``: for ``examples/demo_network_coordinates.inp``,
  the 66-column model input after imputation, the adjacency, each run's
  probabilities, their mean and the recommended class per conduit;
* ``fixtures/gnn_rakowiec_golden.npz``: for ``examples/wroclaw_rakowiec.inp``,
  the model input, the adjacency, the mean probabilities over the five runs and
  the recommended class per conduit.

The model-level tests need no simulation and hold on every platform. The ``slow``
tests run SWMM and the whole application pipeline. The simulation is not
bit-reproducible across platforms, so those tests compare recommended classes and
allow a few conduits with nearly tied classes to differ from the reference.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sa.core.enums import RecommendationCategory

HAS_TF = importlib.util.find_spec("tensorflow") is not None
requires_tf = pytest.mark.skipif(not HAS_TF, reason="requires tensorflow")

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EXAMPLES = Path(__file__).resolve().parents[4] / "examples"
RECOMMENDATIONS_DIR = Path(__file__).resolve().parents[1] / "recommendations"
MODEL_TOLERANCE = 1e-5
MAX_DIFFERING_CONDUITS = {"demo_network_coordinates.inp": 0, "wroclaw_rakowiec.inp": 0}
CLASS_ORDER = [category.value for category in RecommendationCategory]
N_RUNS = 5


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURES / name) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def demo() -> dict[str, np.ndarray]:
    return _load("gnn_demo_parity.npz")


@pytest.fixture(scope="module")
def rakowiec() -> dict[str, np.ndarray]:
    return _load("gnn_rakowiec_golden.npz")


@requires_tf
@pytest.mark.parametrize("run", range(1, N_RUNS + 1))
def test_each_shipped_run_reproduces_its_reference_probabilities(demo: dict[str, np.ndarray], run: int) -> None:
    from sa.core.graph_constructor import load_gnn_model_weights

    model = load_gnn_model_weights(str(RECOMMENDATIONS_DIR / f"gnn_recommender.run{run}.weights.h5"))
    assert model is not None, f"run {run} did not load"

    probabilities = model([demo["features"], demo["adjacency"]], training=False).numpy()

    np.testing.assert_allclose(probabilities, demo["member_probabilities"][run - 1], rtol=0.0, atol=MODEL_TOLERANCE)


@requires_tf
def test_served_model_is_the_mean_of_the_five_runs(demo: dict[str, np.ndarray]) -> None:
    from sa.core.graph_constructor import GnnEnsemble, load_gnn_model_weights

    model = load_gnn_model_weights()
    assert isinstance(model, GnnEnsemble)
    assert len(model.members) == N_RUNS
    assert model.feature_contract.name == "v4_physical"
    assert model.imputation_plan is not None and len(model.imputation_plan["fills"]) == 66

    probabilities = model([demo["features"], demo["adjacency"]], training=False).numpy()

    np.testing.assert_allclose(probabilities, demo["probabilities"], rtol=0.0, atol=MODEL_TOLERANCE)
    assert [CLASS_ORDER[index] for index in probabilities.argmax(axis=1)] == demo["recommendation"].tolist()


@requires_tf
def test_served_model_reproduces_the_reference_on_the_rakowiec_network(rakowiec: dict[str, np.ndarray]) -> None:
    from sa.core.graph_constructor import load_gnn_model_weights

    model = load_gnn_model_weights()

    probabilities = model([rakowiec["features"], rakowiec["adjacency"]], training=False).numpy()

    np.testing.assert_allclose(probabilities, rakowiec["probabilities"], rtol=0.0, atol=MODEL_TOLERANCE)
    assert [CLASS_ORDER[index] for index in probabilities.argmax(axis=1)] == rakowiec["recommendation"].tolist()


@requires_tf
def test_a_missing_run_refuses_to_serve_a_partial_ensemble(monkeypatch: pytest.MonkeyPatch) -> None:
    from sa.core import predictor
    from sa.core.graph_constructor import load_gnn_model_weights

    weights = (*predictor.GNN_CONFIG["weights"], "gnn_recommender.run_missing.weights.h5")
    monkeypatch.setitem(predictor.GNN_CONFIG, "weights", weights)

    assert load_gnn_model_weights() is None


def test_shipped_model_files_are_present() -> None:
    """Runs without TensorFlow, so a checkout missing a model file fails even where the model tests skip."""
    expected = [f"gnn_recommender.run{run}.weights.h5" for run in range(1, N_RUNS + 1)]
    expected += ["gnn_recommender.spec.json", "gnn_recommender.imputation.json"]

    missing = [name for name in expected if not (RECOMMENDATIONS_DIR / name).is_file()]

    assert not missing, f"shipped model files missing from {RECOMMENDATIONS_DIR}: {missing}"


def test_metadata_digest_covers_all_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    from sa.core.predictor import get_model_metadata

    monkeypatch.setenv("GNN_ENABLED", "true")
    member_digests = [
        hashlib.sha256((RECOMMENDATIONS_DIR / f"gnn_recommender.run{run}.weights.h5").read_bytes()).hexdigest()
        for run in range(1, N_RUNS + 1)
    ]

    metadata = get_model_metadata()

    assert metadata["model_kind"] == "gnn"
    assert metadata["model_id"] == "gnn_recommender_v1"
    assert metadata["weights_sha256"] == hashlib.sha256("".join(member_digests).encode("ascii")).hexdigest()


@pytest.fixture
def gnn_enabled_env():
    import sa.core.predictor as predictor_module

    original_env = os.environ.get("GNN_ENABLED")
    original_models_loaded = predictor_module._models_loaded
    original_gnn_recommendation = predictor_module._gnn_recommendation

    os.environ["GNN_ENABLED"] = "true"
    predictor_module._models_loaded = False
    predictor_module._gnn_recommendation = None

    yield

    if original_env is None:
        os.environ.pop("GNN_ENABLED", None)
    else:
        os.environ["GNN_ENABLED"] = original_env
    predictor_module._models_loaded = original_models_loaded
    predictor_module._gnn_recommendation = original_gnn_recommendation


def _serve(inp: Path, tmp_path: Path) -> tuple[Any, list[str], np.ndarray]:
    """Run the application pipeline on a copy of ``inp``; return it, the conduit order and the raw probabilities."""
    from sa.core.data import DataManager

    work_inp = tmp_path / inp.name
    shutil.copy2(inp, work_inp)

    dm = DataManager(str(work_inp), zone=0.8)
    dm.calculate()
    service = dm.recommendation_service
    assert service.model_name == "GNN"
    assert dm.fallback_reason is None

    model = service.model
    outputs: list[np.ndarray] = []

    def recording_model(inputs: Any, training: bool = False) -> Any:
        result = model(inputs, training=training)
        outputs.append(np.asarray(result.numpy()))
        return result

    recording_model.spec = model.spec  # type: ignore[attr-defined]
    recording_model.feature_contract = model.feature_contract  # type: ignore[attr-defined]
    recording_model.imputation_plan = model.imputation_plan  # type: ignore[attr-defined]
    service.model = recording_model

    dm.feature_engineering()
    dm.recommendations()
    assert len(outputs) == 1
    return dm, [str(name) for name in service.conduit_order], outputs[0]


def _served_classes(dm: Any) -> dict[str, str]:
    names = (dm.dfc["Name"] if "Name" in dm.dfc.columns else dm.dfc.index.to_series()).astype(str)
    return dict(zip(names, dm.dfc["recommendation"].astype(str)))


def _assert_pipeline_matches(network: str, reference: dict[str, np.ndarray], tmp_path: Path) -> Any:
    """Serve ``network`` end to end and compare the recommended classes with the reference."""
    dm, order, probabilities = _serve(EXAMPLES / network, tmp_path)

    expected_order = reference["conduit_order"].tolist()
    assert sorted(order) == sorted(expected_order)
    assert probabilities.shape == (len(order), len(CLASS_ORDER))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-5)

    position = {name: index for index, name in enumerate(order)}
    aligned = probabilities[[position[name] for name in expected_order]]
    served = _served_classes(dm)
    assert [served[name] for name in expected_order] == [CLASS_ORDER[index] for index in aligned.argmax(axis=1)]

    drift = np.abs(aligned - reference["probabilities"]).max(axis=1)
    differing = [
        f"{name}: served {served[name]}, reference {expected} (probability drift {drift[index]:.4f})"
        for index, (name, expected) in enumerate(zip(expected_order, reference["recommendation"].tolist()))
        if served[name] != expected
    ]
    assert len(differing) <= MAX_DIFFERING_CONDUITS[network], (
        f"{len(differing)} of {len(expected_order)} conduits differ from the reference "
        f"(largest probability drift {drift.max():.4f}, {int((drift > 1e-3).sum())} conduits above 0.001):\n"
        + "\n".join(differing)
    )
    return dm


@requires_tf
@pytest.mark.slow
def test_pipeline_serves_the_ensemble_on_the_demo_network(
    demo: dict[str, np.ndarray], gnn_enabled_env: None, tmp_path: Path
) -> None:
    from sa.core.predictor import get_model_metadata

    dm = _assert_pipeline_matches("demo_network_coordinates.inp", demo, tmp_path)

    metadata = get_model_metadata(dm.recommendation_service.model_name)
    assert metadata["model_kind"] == "gnn"
    assert metadata["model_id"] == "gnn_recommender_v1"


@requires_tf
@pytest.mark.slow
def test_pipeline_serves_the_ensemble_on_the_rakowiec_network(
    rakowiec: dict[str, np.ndarray], gnn_enabled_env: None, tmp_path: Path
) -> None:
    _assert_pipeline_matches("wroclaw_rakowiec.inp", rakowiec, tmp_path)
