"""stormlab — reproducible pipeline from SWMM INP files to thesis numbers.

The package is layered: ``contracts`` define the data shapes, ``hydraulics``
holds pure formulas, ``io`` touches the filesystem and the SWMM engine,
``features``/``labels``/``graph`` derive model inputs, ``datasets``/``models``
train and predict, and ``experiments``/``reports`` orchestrate and publish.
``config`` and ``provenance`` are cross-cutting and may be imported anywhere.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
