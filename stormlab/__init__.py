"""stormlab — the domain layer between SWMM INP files and the recommendation models.

The package is layered: ``contracts`` define the data shapes, ``hydraulics``
holds pure formulas, ``io`` reads INP and report files, ``features`` and ``graph``
derive model inputs, and ``models`` builds the graph network and serves it.
``config`` is cross-cutting and may be imported anywhere.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
