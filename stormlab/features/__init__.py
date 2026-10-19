"""Feature extraction from SWMM result frames.

Every public entry point here is a pure function: it takes frames, returns new
frames and never touches the filesystem. The conduit pipeline owns the one piece
of hidden knowledge the legacy code carried in its call sites — the order of the
extraction steps, which is load-bearing because later steps read columns written
by earlier ones.
"""

from __future__ import annotations
