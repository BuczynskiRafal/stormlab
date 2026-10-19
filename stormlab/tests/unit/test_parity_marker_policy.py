"""Parity tests must stay opt-in even when the command line supplies its own -m.

pytest's ``-m`` is single-valued: a command line ``-m 'not slow'`` replaces any
expression configured in ``addopts`` instead of combining with it. The suite
therefore deselects the parity layer from a collection hook, and this test pins
that behaviour.
"""

from __future__ import annotations

from stormlab.tests.conftest import RUN_PARITY_OPTION, pytest_collection_modifyitems


class _FakeConfig:
    def __init__(self, run_parity: bool) -> None:
        self._run_parity = run_parity

    def getoption(self, name: str, default: bool = False) -> bool:
        assert name == RUN_PARITY_OPTION
        assert default is False, "the hook must tolerate the option being unregistered"
        return self._run_parity


class _FakeItem:
    def __init__(self, keywords: tuple[str, ...]) -> None:
        self.keywords = set(keywords)
        self.markers: list[object] = []

    def add_marker(self, marker: object) -> None:
        self.markers.append(marker)


def test_parity_items_are_skipped_by_default() -> None:
    parity_item = _FakeItem(("parity",))
    plain_item = _FakeItem(("unit",))

    pytest_collection_modifyitems(_FakeConfig(run_parity=False), [parity_item, plain_item])  # type: ignore[arg-type]

    assert len(parity_item.markers) == 1
    assert not plain_item.markers


def test_parity_items_run_when_explicitly_requested() -> None:
    parity_item = _FakeItem(("parity",))

    pytest_collection_modifyitems(_FakeConfig(run_parity=True), [parity_item])  # type: ignore[arg-type]

    assert not parity_item.markers
