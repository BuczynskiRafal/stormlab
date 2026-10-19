"""One place that makes a run reproducible, and one record of what it did.

Seeds were configurable long before anything applied them, which is the worst of
both worlds: the YAML file documented a reproducibility guarantee that no line of
code delivered. :func:`configure_determinism` is called by :mod:`stormlab.cli`
before any experiment starts, and returns exactly what it applied so the same
facts can be written into the provenance record.

TensorFlow is imported lazily, inside the function: the whole package must stay
importable — and testable — in an environment without it, and importing it at
module scope would also make ``python -m stormlab --help`` pay for a framework it
may never use.

``PYTHONHASHSEED`` is deliberately *not* set here. It is read by the interpreter
at start-up, so a process cannot fix its own; the Makefile and the CI workflow
set it in the environment instead.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Mapping

logger = logging.getLogger(__name__)

DETERMINISM_KEYS: tuple[str, ...] = ("python", "numpy", "tf", "split", "bootstrap")
"""Seeds a configuration is expected to carry.

Only ``python``, ``numpy`` and ``tf`` are applied here. ``split`` is applied by
the splitter and ``bootstrap`` by the resampler that draws the confidence
intervals, each at the moment it draws — a seed applied globally at start-up
would make the interval depend on how much randomness ran before it. They are
listed all the same, because this tuple is what a run records as "how it was
seeded", and a seed that is used but unrecorded is the gap the record exists to
close."""


def configure_determinism(seeds: Mapping[str, int] | Any) -> dict[str, Any]:
    """Seed every random source this pipeline draws from.

    Args:
        seeds: either a mapping of seed name to value, or an object exposing
            ``as_dict()`` (:class:`stormlab.config.Seeds`).

    Returns:
        What was actually applied: the seed values, whether TensorFlow was
        available, and whether deterministic op selection was switched on. The
        return value goes into the provenance record, so a run that could not
        enable determinism says so rather than looking identical to one that did.
    """
    values = _as_mapping(seeds)
    missing = [key for key in DETERMINISM_KEYS if key not in values]
    if missing:
        raise ValueError(f"Seeds are missing {missing}; expected all of {list(DETERMINISM_KEYS)}")

    import numpy as np

    applied: dict[str, Any] = {
        "seeds": {key: int(values[key]) for key in DETERMINISM_KEYS},
        "tensorflow_available": False,
        "op_determinism": False,
    }

    # Keras first, then the two general-purpose sources. `keras.utils.set_random_seed`
    # seeds `random` and `numpy.random` as well as TensorFlow, all from its single
    # argument, so calling it last would silently overwrite the `python` and `numpy`
    # seeds with the `tf` one and make two of the four configured values inert.
    try:
        import tensorflow as tf
    except ImportError as error:
        logger.warning("TensorFlow is not importable (%s); only the python and numpy seeds were applied", error)
    else:
        import keras

        keras.utils.set_random_seed(values["tf"])
        tf.config.experimental.enable_op_determinism()
        applied["tensorflow_available"] = True
        applied["tensorflow_version"] = str(tf.__version__)
        applied["op_determinism"] = True

    random.seed(values["python"])
    np.random.seed(values["numpy"])
    return applied


def _as_mapping(seeds: Mapping[str, int] | Any) -> Mapping[str, int]:
    if hasattr(seeds, "as_dict"):
        return seeds.as_dict()
    if isinstance(seeds, Mapping):
        return seeds
    raise TypeError(f"Expected a seed mapping or an object with as_dict(), got {type(seeds).__name__}")


__all__ = ["DETERMINISM_KEYS", "configure_determinism"]
