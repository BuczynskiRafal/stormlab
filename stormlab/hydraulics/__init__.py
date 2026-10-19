"""Pure hydraulic mathematics: no I/O, no DataFrames, no knowledge of SWMM files.

Three modules, deliberately not merged even though names repeat between them:

* :mod:`stormlab.hydraulics.round` — the Biedugnis-style geometry family. Its
  ``calc_velocity`` takes slope in per mille and its ``calc_u`` works in degrees.
* :mod:`stormlab.hydraulics.formulas` — the Manning family used by the feature
  pipeline. Its ``calc_velocity`` takes slope in m/m and its ``calc_u`` works in
  radians.
* :mod:`stormlab.hydraulics.validators` — boolean checks shared by both.

``round.calc_area`` and ``formulas.calc_area`` are *different functions* that
happen to share a name; deduplicating them would change results.
"""

from __future__ import annotations
