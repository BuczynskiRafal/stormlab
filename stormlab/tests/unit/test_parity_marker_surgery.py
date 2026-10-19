"""``--run-parity`` must remove exactly one conjunct from the marker expression.

``addopts`` carries ``-m 'not parity'`` so an ordinary run never collects the
parity layer, and ``--run-parity`` undoes that. The undoing is a regex over the
marker expression — the only real logic in the whole opt-in mechanism — and it
had no test: ``test_parity_marker_policy`` covers the collection hook, which is
the second line of defence, not this. If this function cleared the expression
instead of one conjunct, ``--run-parity -m 'not slow'`` would silently pull the
slow end-to-end layer in as well, and a green run would mean something else than
it says.
"""

from __future__ import annotations

import pytest

from stormlab.tests.conftest import parity_exclusion_removed


class TestParityExclusionRemoved:
    def test_the_configured_expression_becomes_empty(self) -> None:
        assert parity_exclusion_removed("not parity") == ""

    def test_an_empty_expression_stays_empty(self) -> None:
        assert parity_exclusion_removed("") == ""

    @pytest.mark.parametrize(
        ("markexpr", "expected"),
        [
            ("not parity and not slow", "not slow"),
            ("not slow and not parity", "not slow"),
            ("not slow and not parity and not integration", "not slow and not integration"),
            ("(not parity) and not slow", "not slow"),
        ],
    )
    def test_only_the_parity_conjunct_is_dropped(self, markexpr: str, expected: str) -> None:
        """Everything the caller excluded on purpose has to survive."""
        assert parity_exclusion_removed(markexpr) == expected

    def test_an_expression_without_the_conjunct_is_returned_unchanged(self) -> None:
        assert parity_exclusion_removed("not slow") == "not slow"

    def test_a_marker_whose_name_merely_contains_parity_is_not_dropped(self) -> None:
        """``not parity_golden`` is a different marker and is none of this function's business."""
        assert parity_exclusion_removed("not parity_golden and not slow") == "not parity_golden and not slow"
