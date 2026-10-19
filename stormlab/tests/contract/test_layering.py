"""Import-graph guard: dependencies only ever point down the layer stack.

The check is static — it parses every module under ``stormlab/`` with ``ast``
instead of importing it, so it also holds in environments without TensorFlow.
Three rules are enforced: no module may import a layer above its own, nothing
under ``stormlab/`` may import the legacy Django application, and every import is
absolute — a relative import does not name the layer it reaches, so it would slip
past the first rule unseen.

The rules live in :func:`check_module`, which both the sweep over the real package
and the negative tests call, so the negative tests exercise the shipped rule
rather than restating the rank table.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

LAYERS: tuple[str, ...] = (
    "contracts",
    "hydraulics",
    "io",
    "generate",
    "features",
    "labels",
    "graph",
    "datasets",
    "models",
    "experiments",
    "author_review",
    "reports",
    "preflight",
    "cli",
)

LAYER_RANK: dict[str, int] = {name: index for index, name in enumerate(LAYERS)}
# labels and graph are peers: either may import the other, neither may reach higher.
LAYER_RANK["graph"] = LAYER_RANK["labels"]

CROSS_CUTTING: frozenset[str] = frozenset({"config", "provenance", "determinism", "tests"})
"""Layers any module may import: they depend on nothing of the pipeline.

``cli`` and ``preflight`` used to be listed here and are not neutral — they are the
entry points, and an exemption for them meant the two modules that reach furthest
across the package were the two the check never looked at. They now sit at the top
of :data:`LAYERS` instead, which lets them import everything below and nothing
above, and is the same statement without the hole."""

FORBIDDEN_ROOTS: frozenset[str] = frozenset({"app", "sa"})


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _modules() -> list[Path]:
    return sorted(path for path in _package_root().rglob("*.py") if "tests" not in path.relative_to(_package_root()).parts)


def _layer_of(path: Path, *, root: Path | None = None) -> str:
    parts = path.relative_to(root if root is not None else _package_root()).parts
    return parts[0] if len(parts) > 1 else path.stem


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _imported_stormlab_layers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    layers: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "stormlab" and len(parts) > 1:
                layers.add(parts[1])
        # ``from stormlab import reports`` names the layer in the alias, not in the
        # module path, so reading only ``node.module`` let a whole layer through.
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "stormlab":
            layers.update(alias.name for alias in node.names)
    return layers


def _relative_imports(path: Path) -> list[str]:
    """Relative ``from . import x`` statements, which the layer check cannot resolve."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        "." * node.level + (node.module or "") for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level > 0
    ]


class LayerViolation(AssertionError):
    """Raised by :func:`check_module` when a module's imports break a layer rule."""


def check_module(module: Path, *, root: Path | None = None) -> None:
    """The production rule, applied to one module.

    Extracted so the parametrised sweep over the real package and the negative
    tests below run the *same* code. A negative test that only re-states the rank
    table proves the table, not the check.

    Raises:
        LayerViolation: if the module imports the legacy application, or a layer
            above its own.
    """
    package_root = root if root is not None else _package_root()

    relative = _relative_imports(module)
    if relative:
        raise LayerViolation(
            f"{module} uses relative import(s) {relative}; every import inside stormlab is absolute, because the "
            "layer check reads the import statement and a relative one hides which layer it reaches"
        )

    forbidden = _imported_roots(module) & FORBIDDEN_ROOTS
    if forbidden:
        raise LayerViolation(f"{module} imports {sorted(forbidden)}; stormlab must not depend on the Django application")

    layer = _layer_of(module, root=package_root)
    if layer in CROSS_CUTTING or layer not in LAYER_RANK:
        return
    for imported in _imported_stormlab_layers(module):
        if imported in CROSS_CUTTING:
            continue
        if imported not in LAYER_RANK:
            raise LayerViolation(f"{module} imports unknown layer {imported!r}")
        if LAYER_RANK[imported] > LAYER_RANK[layer]:
            raise LayerViolation(f"{module} (layer {layer!r}) imports {imported!r}, which sits above it in {' -> '.join(LAYERS)}")


def test_the_package_contains_modules_to_check() -> None:
    assert _modules()


@pytest.mark.parametrize("module", _modules(), ids=lambda path: str(path.relative_to(_package_root())))
def test_module_respects_the_layer_rules(module: Path) -> None:
    check_module(module)


def _write_module(root: Path, relative: str, source: str) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return target


def test_the_rule_rejects_an_upward_import(tmp_path: Path) -> None:
    module = _write_module(tmp_path, "datasets/build.py", "from stormlab.reports.tables import write\n")

    with pytest.raises(LayerViolation, match="sits above it"):
        check_module(module, root=tmp_path)


def test_the_rule_rejects_an_upward_import_written_as_a_package_alias(tmp_path: Path) -> None:
    """``from stormlab import reports`` names its layer in the alias, not the module path."""
    module = _write_module(tmp_path, "datasets/build.py", "from stormlab import reports\n")

    with pytest.raises(LayerViolation, match="sits above it"):
        check_module(module, root=tmp_path)


def test_the_entry_points_are_checked_rather_than_exempt(tmp_path: Path) -> None:
    """``cli`` sits at the top of the stack, so nothing is above it and nothing is unchecked."""
    module = _write_module(tmp_path, "cli.py", "from stormlab.reports.report import run\n")
    check_module(module, root=tmp_path)

    above = _write_module(tmp_path, "preflight.py", "from stormlab.cli import main\n")

    with pytest.raises(LayerViolation, match="sits above it"):
        check_module(above, root=tmp_path)


def test_the_rule_rejects_a_function_local_upward_import(tmp_path: Path) -> None:
    """A deferred import is still a dependency; hiding it inside a function is not an escape."""
    module = _write_module(
        tmp_path,
        "hydraulics/formulas.py",
        "def f():\n    from stormlab.models.registry import load_registry\n    return load_registry\n",
    )

    with pytest.raises(LayerViolation, match="models"):
        check_module(module, root=tmp_path)


def test_the_rule_rejects_an_unknown_layer(tmp_path: Path) -> None:
    module = _write_module(tmp_path, "features/conduits.py", "from stormlab.telemetry.sink import emit\n")

    with pytest.raises(LayerViolation, match="unknown layer"):
        check_module(module, root=tmp_path)


@pytest.mark.parametrize(
    ("relative", "source"),
    [
        ("features/conduits.py", "import app.sa.core.data\n"),
        ("features/nodes.py", "from sa.core.round import min_slope\n"),
    ],
)
def test_the_rule_rejects_the_legacy_application(tmp_path: Path, relative: str, source: str) -> None:
    module = _write_module(tmp_path, relative, source)

    with pytest.raises(LayerViolation, match="Django application"):
        check_module(module, root=tmp_path)


def test_the_rule_accepts_a_same_layer_and_a_downward_import(tmp_path: Path) -> None:
    same_layer = _write_module(tmp_path, "features/conduits.py", "from stormlab.features.subcatchments import encode\n")
    downward = _write_module(tmp_path, "graph/build.py", "from stormlab.contracts.features import FEATURE_COLUMNS\n")

    check_module(same_layer, root=tmp_path)
    check_module(downward, root=tmp_path)


def test_the_rule_accepts_a_cross_cutting_import_from_any_layer(tmp_path: Path) -> None:
    """``config`` is importable everywhere; the graph constants rely on that."""
    module = _write_module(tmp_path, "hydraulics/round.py", "from stormlab.config import KHOP_SCHEMES\n")

    check_module(module, root=tmp_path)


def test_the_rule_rejects_a_relative_import(tmp_path: Path) -> None:
    """A relative import is invisible to the layer check, so it is banned outright."""
    module = _write_module(tmp_path, "features/conduits.py", "from ..reports.tables import write\n")

    with pytest.raises(LayerViolation, match="relative import"):
        check_module(module, root=tmp_path)


def test_graph_and_labels_are_peers() -> None:
    assert LAYER_RANK["graph"] == LAYER_RANK["labels"]
