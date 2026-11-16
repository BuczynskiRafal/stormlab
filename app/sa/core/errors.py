"""Errors whose text is written for the user.

An analysis can fail because of the model (the user can fix it) or because of the
application (the user can only report it). Only the first kind may reach the page and
the history: its ``str()`` is a plain sentence that names the elements involved and
never a path, a Python exception or a placeholder. Anything else is replaced by
:func:`unexpected_failure_message` and goes to the log with its traceback.
"""

from __future__ import annotations


class AnalysisInputError(Exception):
    """The model cannot be analysed as it stands; the message says why, for the user."""


class SwmmInputError(AnalysisInputError):
    """The SWMM engine refused the model; the message carries the engine's own error lines."""


def unexpected_failure_message(session_id: int) -> str:
    """The fixed sentence for a failure the user cannot fix, with the number to report."""
    return (
        "The analysis stopped because of an unexpected error in the application. "
        f"Please report it, quoting analysis number {session_id}."
    )
