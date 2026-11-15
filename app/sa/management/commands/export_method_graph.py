"""Export the data behind the "What the model reads" figure of the documentation.

The figure draws the Rakowiec demo network and shows, for a selected conduit, the
conduits the graph model aggregates over. The aggregation matrix is built here with
the same functions and the same model specification the application uses when it
serves recommendations, so the weights in the figure are the production weights.
The specification is the one shipped in the repository
(``sa/core/recommendations/gnn_recommender.spec.json``); one substituted at run
time through the ``GNN_SPEC`` environment variable is not read here.

Output: ``app/static/data/method-graph.json``, byte-stable for a given network and
graph code. ``--check`` compares the committed file with a fresh build and fails on
any difference, which is how drift between the figure and the graph code is caught.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from django.conf import settings
from scipy.sparse import SparseEfficiencyWarning
from django.core.management.base import BaseCommand, CommandError

REPO_ROOT = Path(settings.BASE_DIR).parent
DEFAULT_INP = REPO_ROOT / "examples" / "wroclaw_rakowiec.inp"
DEFAULT_OUTPUT = Path(settings.BASE_DIR) / "static" / "data" / "method-graph.json"
MODEL_SPEC = Path(settings.BASE_DIR) / "sa" / "core" / "recommendations" / "gnn_recommender.spec.json"

FRAME = 1000  # longer side of the local coordinate frame
WEIGHT_DECIMALS = 4


def _load_conduits(inp_path: Path):
    """The swmmio model and its conduit frame, node columns cast to str as in production."""
    import swmmio

    model = swmmio.Model(str(inp_path), include_rpt=False)
    dfc = model.conduits()
    for column in ("InletNode", "OutletNode"):
        dfc[column] = dfc[column].astype(str)
    return model, dfc


def _aggregation_matrix(dfc, spec: dict):
    """The matrix the served model aggregates with, built as ``_select_recommendation_service`` builds it."""
    from stormlab.graph.build import build_graph
    from stormlab.graph.khop import operator_adjacency
    from stormlab.graph.order import canonical_order

    order = canonical_order(dfc, "inp")
    with warnings.catch_warnings():
        # The graph builder changes CSR sparsity in place; scipy's advice about it is noise here.
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        graph = build_graph(dfc, order, direction="downstream")
        source = graph.adjacency.transpose().tocsr() if spec["direction"] == "upstream" else graph.adjacency
        matrix = operator_adjacency(source, operator=spec["operator"], max_hops=spec["hops"])
    return list(graph.order), matrix.tocsr()


def _rows(matrix, order: list[str]) -> dict[str, list[list]]:
    """Each conduit's nonzero row entries as ``[name, weight]``, heaviest first, ties in network order."""
    position = {name: i for i, name in enumerate(order)}
    rows = {}
    for i, name in enumerate(order):
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        entries = [
            (order[j], round(float(w), WEIGHT_DECIMALS))
            for j, w in zip(matrix.indices[start:end], matrix.data[start:end], strict=True)
            if w != 0
        ]
        entries.sort(key=lambda item: (-item[1], position[item[0]]))
        rows[name] = [[other, weight] for other, weight in entries]
    return rows


def _frame(points: list[tuple[float, float]]):
    """A projection of map coordinates onto integers in a ``FRAME`` box, y pointing down."""
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    min_x, max_y = min(xs), max(ys)
    span = max(max(xs) - min_x, max_y - min(ys)) or 1.0
    scale = FRAME / span

    def project(x: float, y: float) -> list[int]:
        return [round((x - min_x) * scale), round((max_y - y) * scale)]

    view_box = [round((max(xs) - min_x) * scale), round((max_y - min(ys)) * scale)]
    return project, view_box


def build_method_graph(inp_path: Path | str = DEFAULT_INP) -> dict:
    """The figure's data for one network: geometry, one-step and two-step rows, default selection.

    ``step1`` is the row of the aggregation matrix P (exact weights of one graph
    step); ``step2`` is the row of P·P, the share of the two-step neighbourhood
    average before the model's non-linearities.
    """
    inp_path = Path(inp_path)
    spec = json.loads(MODEL_SPEC.read_text(encoding="utf-8"))
    model, dfc = _load_conduits(inp_path)
    order, matrix = _aggregation_matrix(dfc, spec)

    coordinates = model.inp.coordinates
    vertices = model.inp.vertices
    node_ids = list(dict.fromkeys(node for name in order for node in (dfc.at[name, "InletNode"], dfc.at[name, "OutletNode"])))
    missing = [node for node in node_ids if node not in coordinates.index]
    if missing:
        raise ValueError(f"Nodes without coordinates: {missing}")

    node_xy = {node: (float(coordinates.at[node, "X"]), float(coordinates.at[node, "Y"])) for node in node_ids}
    link_vertices = {
        name: [(float(x), float(y)) for x, y in vertices.loc[[name], ["X", "Y"]].itertuples(index=False)]
        for name in order
        if name in vertices.index
    }
    project, view_box = _frame(list(node_xy.values()) + [xy for points in link_vertices.values() for xy in points])

    outfalls = set(map(str, model.inp.outfalls.index))
    step1 = _rows(matrix, order)
    step2 = _rows((matrix @ matrix).tocsr(), order)
    # The widest two-step reach is the most telling first view; the earliest such conduit wins ties.
    default = max(order, key=lambda name: (len(step2[name]), -order.index(name)))

    nodes = []
    for node in node_ids:
        x, y = project(*node_xy[node])
        nodes.append({"id": node, "x": x, "y": y, "outfall": node in outfalls})

    conduits = []
    for name in order:
        inlet, outlet = dfc.at[name, "InletNode"], dfc.at[name, "OutletNode"]
        points = [node_xy[inlet], *link_vertices.get(name, []), node_xy[outlet]]
        conduits.append({"id": name, "from": inlet, "to": outlet, "path": [project(x, y) for x, y in points]})

    return {
        "network": inp_path.stem,
        "model": {"operator": spec["operator"], "hops": spec["hops"], "direction": spec["direction"]},
        "viewBox": view_box,
        "nodes": nodes,
        "conduits": conduits,
        "step1": step1,
        "step2": step2,
        "default": default,
    }


def render_method_graph(data: dict) -> str:
    """Serialise the figure data byte-stably: sorted keys, compact separators, trailing newline."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


class Command(BaseCommand):
    help = "Write the data of the documentation's receptive-field figure (static/data/method-graph.json)."

    def add_arguments(self, parser):
        parser.add_argument("--inp", default=str(DEFAULT_INP), help="SWMM input file of the network to draw.")
        parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write the JSON file.")
        parser.add_argument("--check", action="store_true", help="Fail if the file on disk differs from a fresh build.")

    def handle(self, *args, inp: str, output: str, check: bool, **options):
        inp_path, output_path = Path(inp), Path(output)
        if not inp_path.is_file():
            raise CommandError(f"Network file not found: {inp_path}")
        content = render_method_graph(build_method_graph(inp_path))

        if check:
            current = output_path.read_text(encoding="utf-8") if output_path.is_file() else None
            if current != content:
                raise CommandError(f"{output_path} is out of date; run `manage.py export_method_graph` to regenerate it.")
            self.stdout.write(f"{output_path} is up to date.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8", newline="\n")
        self.stdout.write(f"Wrote {output_path} ({len(content.encode('utf-8'))} bytes).")
