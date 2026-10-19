"""Seeds in the configuration have to reach the random sources, not just the YAML."""

from __future__ import annotations

import random

import numpy as np
import pytest

from stormlab.determinism import DETERMINISM_KEYS, configure_determinism

SEEDS = {"python": 11, "numpy": 22, "tf": 33, "split": 44, "bootstrap": 55}


def test_the_python_random_source_is_seeded() -> None:
    configure_determinism(SEEDS)
    first = [random.random() for _ in range(5)]

    configure_determinism(SEEDS)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_the_numpy_random_source_is_seeded() -> None:
    configure_determinism(SEEDS)
    first = np.random.rand(5)

    configure_determinism(SEEDS)
    second = np.random.rand(5)

    np.testing.assert_array_equal(first, second)


def test_a_different_seed_gives_a_different_stream() -> None:
    """Also pins the ordering: Keras seeds `random` too, from its own seed."""
    configure_determinism(SEEDS)
    first = [random.random() for _ in range(5)]

    configure_determinism({**SEEDS, "python": 12})
    second = [random.random() for _ in range(5)]

    assert first != second


def test_it_reports_what_it_applied() -> None:
    applied = configure_determinism(SEEDS)

    assert applied["seeds"] == SEEDS
    assert set(applied) >= {"seeds", "tensorflow_available", "op_determinism"}


def test_an_incomplete_seed_set_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="tf"):
        configure_determinism({"python": 1, "numpy": 2, "split": 3})


def test_it_accepts_the_config_seeds_object() -> None:
    from stormlab.config import Seeds

    applied = configure_determinism(Seeds(python=1, numpy=2, tf=3, split=4, bootstrap=5))

    assert applied["seeds"] == {"python": 1, "numpy": 2, "tf": 3, "split": 4, "bootstrap": 5}


def test_it_rejects_something_that_is_neither() -> None:
    with pytest.raises(TypeError):
        configure_determinism(42)


def test_the_expected_seed_names_match_the_config_dataclass() -> None:
    from stormlab.config import Seeds

    assert set(DETERMINISM_KEYS) == set(Seeds(python=0, numpy=0, tf=0, split=0, bootstrap=0).as_dict())


def test_tensorflow_op_determinism_is_switched_on_when_tensorflow_is_there() -> None:
    """In an environment with TensorFlow this is the assertion that matters."""
    pytest.importorskip("tensorflow", reason="TensorFlow is optional; the seeds above are applied regardless")

    applied = configure_determinism(SEEDS)

    assert applied["tensorflow_available"] is True
    assert applied["op_determinism"] is True


def test_keras_is_seeded_so_two_runs_initialise_the_same_weights() -> None:
    keras = pytest.importorskip("keras", reason="TensorFlow/Keras is optional")

    def initial_weights() -> list:
        configure_determinism(SEEDS)
        layer = keras.layers.Dense(4)
        layer.build((None, 3))
        return [np.asarray(weight) for weight in layer.get_weights()]

    first, second = initial_weights(), initial_weights()

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_a_missing_tensorflow_is_reported_rather_than_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """The package must stay usable in the TensorFlow-free CI matrix."""
    import builtins

    real_import = builtins.__import__

    def without_tensorflow(name: str, *args: object, **kwargs: object) -> object:
        if name == "tensorflow":
            raise ImportError("no tensorflow in this environment")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", without_tensorflow)

    applied = configure_determinism(SEEDS)

    assert applied["tensorflow_available"] is False
    assert applied["op_determinism"] is False
    assert applied["seeds"] == SEEDS
