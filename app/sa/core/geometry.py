"""Drawable geometry of a SWMM network, extracted from the INP sections alone.

The module is deliberately free of Django and of SWMM: it reads an
``swmmio.Model.inp`` object (duck-typed — any object exposing the section
dataframes will do) and returns a plain dictionary that the persistence layer
stores and the browser draws.  Nothing here runs a simulation, touches the
database or writes a file.

Two shapes of payload come out of :func:`extract_geometry`, told apart by the
``layout`` key, and **both are complete**: every node and every link with known
endpoints carries a position.  ``"inp"`` means the coordinates come from the
file; ``"auto"`` means the file had no usable coordinates and the positions
come from the layered schematic of :func:`auto_layout`.  No consumer ever has
to handle a node without a point.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from typing import Any

NETMAP_VERSION = 2
"""Stamped into every payload; bump it when the contract below changes."""

COMPONENT_GAP = 2.0
"""Column units left blank between two components of the automatic schematic."""

NETMAP_MAX_ELEMENTS = 6000
"""Upper bound on ``counts.elements`` the browser is asked to draw as SVG.

Nothing in this module enforces it — extraction and storage are unbounded — and
the only reader is the serializer that fills the network-map panel: above this
many elements the payload stays in the database and the panel says so.
"""

NETMAP_MAX_POINTS = 120_000
"""Upper bound on ``counts.points``, the second half of the same guard.

An element says nothing about how much of the payload it is: a subcatchment is
one element whatever the length of its [Polygons] outline, and a link is two
whatever the number of [VERTICES] bends it carries, so a thousand detailed
outlines pass the element bound while weighing hundreds of thousands of
coordinate pairs.  One pair costs roughly twenty bytes of JSON (``[1234.56,
789.01],``), which puts this limit at about 2.4 MB of inlined payload — still
large, and already past what a page should carry.  Same contract as
:data:`NETMAP_MAX_ELEMENTS`: the extractor only counts, the serializer refuses.
"""

_COLUMN_SPACING = 1.0
"""Column units taken by one leaf of the schematic before scaling by ``dx``."""

_LAYER_SPACING = 1.0
"""Drawing units between two consecutive layers of the schematic."""

_ASPECT_TARGET = 0.6
"""Target width/height ratio the horizontal stretch of the schematic aims at."""

_COORD_DECIMALS = 2
"""Normalised coordinates are rounded here and nowhere else."""

_GAP_DECIMALS = 4
"""``min_gap`` keeps more digits than a coordinate — the browser divides by it."""

_NODE_SECTIONS: tuple[str, ...] = ("junctions", "outfalls", "storage", "dividers")
"""Order in which model nodes are collected; it fixes the ``i`` of every node."""

_NODE_KIND_PRIORITY: tuple[tuple[str, str], ...] = (
    ("outfalls", "outfall"),
    ("storage", "storage"),
    ("junctions", "junction"),
)
"""``k`` resolution order.  Anything else — [DIVIDERS] included — becomes ``"other"``."""

_LINK_SECTIONS: tuple[tuple[str, str], ...] = (
    ("conduits", "conduit"),
    ("pumps", "pump"),
    ("orifices", "orifice"),
    ("weirs", "weir"),
    ("outlets", "outlet"),
)
"""Order in which links are collected; it fixes the ``i`` of every link."""

_MIN_POLYGON_POINTS = 3
"""Fewer points than this is not a polygon, it is a line or a dot."""


def extract_geometry(inp: Any) -> dict:
    """Return the drawable geometry of the model behind *inp*.

    *inp* is an ``swmmio.Model.inp``; ends, kinds and coordinates are all read
    from its sections, so a pump or a weir carries a complete record even
    though the analysis pipeline never computes a result for it.

    The returned dictionary is the single definition of the network-map
    payload::

        {"version": 2,
         "origin": [x0, y_top],
         "bbox": {"w": float, "h": float},
         "min_gap": float | None,
         "layout": "inp" | "auto",
         "nodes": [{"n": str, "i": int, "p": [x, y], "k": str}],
         "links": [{"n": str, "i": int, "p": [[x, y], ...], "a": str, "b": str,
                    "kind": str}],
         "polys": [{"n": str, "i": int, "p": [[x, y], ...], "c": [x, y],
                    "o": [x, y], "on": str}],
         "skipped": {"links_dangling": int, "polys_degenerate": int,
                     "polys_no_outlet": int},
         "counts": {"elements": int, "points": int}}

    Version 2 added three keys to a polygon, and they come in two groups.  ``c``
    is the **area centroid** of the outline and is emitted for **every** polygon
    of the ``"inp"`` layout, whether or not its outlet was resolved: it is the
    anchor a subcatchment label hangs from, and a subcatchment nobody recognised
    the outlet of still has a name to show.  Computing it a second time in the
    browser is not an option — one point would then have two truths, and the
    fallback rule of :func:`_polygon_centroid` would have to exist in two
    languages.  ``o`` and ``on`` are the far end and the name of the receiver of
    the outlet line, and they are emitted **together or not at all**: without
    both there is nothing to draw.  They are absent in the ``"auto"`` layout
    (which invents no outlines), when the outlet names nothing the model knows,
    and when ``o`` lands on ``c`` after rounding — a line of zero length, which
    the browser would answer with an empty path and no element at all.
    ``polys_no_outlet`` counts the outlet lines that could not be drawn because
    the far end of the line is not a point of this drawing: an unknown name, an
    orphan outline no [SUBCATCHMENTS] row names, and — D18, confirmed in the
    final review — a target that *is* a row of [SUBCATCHMENTS] but whose own
    outline was dropped, so it has no centroid to aim at.  Counting that last
    case with the others is the rule and not an oversight: the counter answers
    "how many outlet lines are missing from the picture", and a target with
    nothing drawable on it is missing from the picture exactly as an unknown
    name is.  Two cases stay outside it: an outline no row names at all
    attempted no line, and the zero-length case above, where the target is
    known, drawable, and simply coincides with the source.  A payload of
    version 1 has none of this and is still served as it is: nothing backfills
    an old row.

    ``i`` is the 0-based position in the element's own list and serves only as
    a DOM key.  Coordinates are normalised so that the drawing starts at the
    origin and the Y axis points down: ``x' = round(x - x0, 2)`` and
    ``y' = round(y_top - y, 2)``; ``origin`` is kept unrounded so the
    normalisation can be undone.  The function never raises on a model that
    lacks a section and never learns how big a network is allowed to be — it
    only reports ``counts``: ``elements`` is what the browser draws and
    ``points`` is what that costs, the two halves of one guard the serializer
    alone applies.
    """
    node_names, node_kinds = _model_nodes(inp)
    if not node_names:
        return _empty_payload()

    coordinates = _coordinates(inp)
    layout = "inp" if all(name in coordinates for name in node_names) else "auto"

    known = set(node_names)
    all_links = _model_links(inp)
    links = [link for link in all_links if link[1] in known and link[2] in known]
    links_dangling = len(all_links) - len(links)

    if layout == "inp":
        # [VERTICES] and [Polygons] are read and grouped once, here, and the rule that
        # decides what a polygon is lives in this one line: the origin has to take in
        # exactly the outlines that end up drawn, and two copies of the rule would put
        # a subcatchment into the bounding box without drawing it, or the reverse.
        bends = _grouped_points(_frame(inp, "vertices"))
        groups = _grouped_points(_frame(inp, "polygons"))
        outlets = _subcatchment_outlets(inp)
        # The names of the **model's** subcatchments, which is not the same set as the
        # names [Polygons] carries (an outline proves nothing about the tables) and not
        # the keys of ``outlets`` either (a subcatchment that flows nowhere is missing
        # from those, and it is exactly the valid target of ``S1 -> S2``).
        subcatchment_names = set(_section_names(inp, "subcatchments"))
        outlines = {name: points for name, points in groups.items() if len(points) >= _MIN_POLYGON_POINTS}
        polys_degenerate = len(groups) - len(outlines)
        origin = _raw_origin(node_names, coordinates, links, bends, outlines)
        positions, link_points, polys, polys_no_outlet = _geometry_from_file(
            coordinates, links, origin, bends, outlines, known, outlets, subcatchment_names
        )
    else:
        origin = (0.0, 0.0)
        positions = auto_layout(node_names, [(a, b) for _, a, b, _ in links], _section_names(inp, "outfalls"))
        link_points = [[list(positions[a]), list(positions[b])] for _, a, b, _ in links]
        polys = []
        polys_degenerate = 0
        polys_no_outlet = 0

    drawn = [positions[name] for name in node_names]
    drawn.extend(point for points in link_points for point in points)
    drawn.extend(point for poly in polys for point in poly["p"])
    # A centroid rides along for every polygon and an outlet end for every line drawn;
    # both are coordinate pairs the payload carries, so both weigh on ``points``.  The
    # bounding box is unaffected by construction: ``o`` is either a node position (already
    # in the list) or another outline's centroid, and a centroid always lies inside the
    # box of its own vertices, which are in the list too.
    drawn.extend(poly["c"] for poly in polys)
    drawn.extend(poly["o"] for poly in polys if "o" in poly)
    outlet_lines = sum(1 for poly in polys if "o" in poly)

    node_records = [
        {"n": name, "i": index, "p": list(positions[name]), "k": node_kinds[name]} for index, name in enumerate(node_names)
    ]
    link_records = [
        {"n": name, "i": index, "p": points, "a": a, "b": b, "kind": kind}
        for index, ((name, a, b, kind), points) in enumerate(zip(links, link_points))
    ]

    return {
        "version": NETMAP_VERSION,
        "origin": [origin[0], origin[1]],
        "bbox": _bbox(drawn),
        "min_gap": _min_gap([positions[name] for name in node_names]),
        "layout": layout,
        "nodes": node_records,
        "links": link_records,
        "polys": polys,
        "skipped": {
            "links_dangling": links_dangling,
            "polys_degenerate": polys_degenerate,
            "polys_no_outlet": polys_no_outlet,
        },
        "counts": {
            # An outlet line is one drawable element; the centroid is not drawn at all.
            "elements": len(polys) + 2 * len(link_records) + len(node_records) + outlet_lines,
            # Every coordinate pair the payload carries, which is what it weighs:
            # ``drawn`` is already exactly that list, node positions included.
            "points": len(drawn),
        },
    }


def auto_layout(
    node_names: Any,
    links: Any,
    outfalls: Any,
) -> dict[str, tuple[float, float]]:
    """Layered schematic used when the file carries no usable coordinates.

    *node_names* is every node of the model, *links* an iterable of ``(a, b)``
    pairs of **all** link kinds — pumps and weirs hold the picture together
    just as conduits do — and *outfalls* the names that [OUTFALLS] lists.

    A node's layer is its hop distance to the nearest root, measured by a
    breadth-first search over the reversed graph; because the search visits
    every node once, a cycle is harmless and needs no detection.  Roots are the
    outfalls, or the sinks of the graph when the model has none, or the
    alphabetically first node when the graph is one big cycle.  Nodes that no
    root reaches become roots of their own and are laid out next to the rest,
    :data:`COMPONENT_GAP` column units away from the previous component.

    Within a layer the order comes from an iterative depth-first walk of the
    search forest — a leaf takes the next column, a parent the mean of its
    children — and the horizontal axis is then stretched by
    ``dx = max(1, 0.6 * layers / columns)`` so that a chain-shaped network does
    not come out as a ribbon.  Every loop runs over sorted names and nothing
    draws a random number, so the same model always yields the same result.

    The returned positions are already in screen space — the outfall sits at
    the largest ``y``, that is at the bottom — so the Y flip of the ``"inp"``
    branch must **not** be applied to them a second time.  The smallest ``x``
    and the smallest ``y`` are both ``0.0``.
    """
    # The three normalisations are defensive: every caller inside the module already
    # passes names that went through _name, so these are no-ops there.  They use the
    # same function all the same - a public entry point that stringified a numeric name
    # its own way would be the one place in the file where two names of one node could
    # still disagree.
    names = sorted({_name(name) for name in node_names})
    if not names:
        return {}
    known = set(names)

    successors: dict[str, set[str]] = {name: set() for name in names}
    predecessors: dict[str, set[str]] = {name: set() for name in names}
    for a, b in links:
        a, b = _name(a), _name(b)
        if a in known and b in known:
            successors[a].add(b)
            predecessors[b].add(a)

    roots = sorted({_name(name) for name in outfalls} & known)
    if not roots:
        roots = [name for name in names if not successors[name]]
    if not roots:
        # A model that is one cycle has neither an outfall nor a sink.
        roots = [names[0]]

    layer: dict[str, int] = {}
    children: dict[str, list[str]] = {name: [] for name in names}
    root_order: list[str] = []
    seeds = roots
    cursor = 0
    sinks_seeded = False
    while True:
        fresh = [seed for seed in seeds if seed not in layer]
        if fresh:
            _spread_layers(fresh, predecessors, layer, children)
            root_order.extend(fresh)
        while cursor < len(names) and names[cursor] in layer:
            cursor += 1
        if cursor == len(names):
            break
        # The same fallback order as for the first roots, applied to what is left:
        # a node nothing flows out of is the outfall of its own component.  Every
        # node still unvisited only flows into unvisited nodes, so a sink of the
        # whole graph is a sink of its component as well.
        # All of them are seeded in one pass; what is left after that lies on
        # cycles and is picked up one name at a time, without another scan.
        sinks = [] if sinks_seeded else [name for name in names[cursor:] if name not in layer and not successors[name]]
        sinks_seeded = True
        seeds = sinks or [names[cursor]]

    column = _columns(root_order, children)
    lowest_column = min(column.values())
    span = max(column.values()) - lowest_column + _COLUMN_SPACING
    deepest = max(layer.values())
    stretch = max(1.0, _ASPECT_TARGET * (deepest + 1) / span)
    return {
        name: (
            round((column[name] - lowest_column) * stretch, _COORD_DECIMALS),
            round((deepest - layer[name]) * _LAYER_SPACING, _COORD_DECIMALS),
        )
        for name in names
    }


def _spread_layers(
    seeds: list[str],
    predecessors: dict[str, set[str]],
    layer: dict[str, int],
    children: dict[str, list[str]],
) -> None:
    """Breadth-first search upstream from *seeds*, filling *layer* and *children*."""
    queue: deque[str] = deque()
    for seed in seeds:
        layer[seed] = 0
        queue.append(seed)
    while queue:
        node = queue.popleft()
        for upstream in sorted(predecessors[node]):
            if upstream in layer:
                continue
            layer[upstream] = layer[node] + 1
            children[node].append(upstream)
            queue.append(upstream)


def _columns(root_order: list[str], children: dict[str, list[str]]) -> dict[str, float]:
    """Assign a column to every node of the search forest, component by component.

    A leaf takes the next free column, a parent the mean of its children's
    columns ("tidy tree").  The walk uses an explicit stack, so a deep network
    cannot exhaust the interpreter's recursion limit.  Two components are
    separated by exactly :data:`COMPONENT_GAP` column units between the last
    column of one and the first column of the next.
    """
    column: dict[str, float] = {}
    next_column = 0.0
    for root in root_order:
        first_column = next_column
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            kids = children[node]
            if not kids:
                column[node] = next_column
                next_column += _COLUMN_SPACING
            elif not expanded:
                stack.append((node, True))
                for kid in reversed(kids):
                    stack.append((kid, False))
            else:
                column[node] = sum(column[kid] for kid in kids) / len(kids)
        last_column = max(first_column, next_column - _COLUMN_SPACING)
        next_column = last_column + COMPONENT_GAP
    return column


def _geometry_from_file(
    coordinates: dict[str, tuple[float, float]],
    links: list[tuple[str, str, str, str]],
    origin: tuple[float, float],
    bends: dict[str, list[tuple[float, float]]],
    outlines: dict[str, list[tuple[float, float]]],
    known: set[str],
    outlets: dict[str, str],
    subcatchment_names: set[str],
) -> tuple[dict[str, tuple[float, float]], list[list[list[float]]], list[dict], int]:
    """Normalise the coordinates the file carries into the payload's shapes.

    *bends* and *outlines* are the two grouped sections, already filtered by the
    caller: nothing here decides what a polygon is.  *known* is the set of the
    model's node names, *outlets* the ``{subcatchment: outlet}`` mapping of
    [SUBCATCHMENTS] and *subcatchment_names* the names the same section lists.
    The flip and the rounding live in this function, so the centroid ``c`` and
    the outlet end ``o`` have to be produced here as well and not by the caller.

    The outlet of a polygon resolves in one hop and never follows a chain: a
    target that is a **node of the model** gives that node's position, a target
    that is a **subcatchment of the model with an outline of its own** gives
    that outline's centroid, and anything else — an unknown name, an orphan
    outline that no [SUBCATCHMENTS] row names, the polygon itself — gives no
    line.  *known* is checked, never ``positions``: the latter holds every
    [COORDINATES] row, model node or not, and a line to a name that is not in
    the network would point at nothing.  Reading ``positions[target]`` after
    that check cannot raise, because the ``"inp"`` layout is chosen only when
    **every** node has a coordinate.

    The fourth returned value counts the outlet lines that could not be drawn:
    the target names nothing the model knows, or it names a subcatchment whose
    own outline was dropped and which therefore has no centroid the line could
    end on.  A polygon no [SUBCATCHMENTS] row names has no target and attempted
    no line, so it is not counted.  Neither is the degenerate case, where ``o``
    falls on ``c``
    after rounding: there the target is known and the line is merely of zero
    length, so the counter would misname it — and the pair is dropped here, in
    Python, rather than left for the browser to discard, because the drawing
    gate compares the number of outlet elements in the DOM with the number of
    polygons carrying ``o`` in the payload.  One rule keeps both sides equal;
    two rules on two sides do not.
    """
    x0, y_top = origin

    def flip(point: tuple[float, float]) -> tuple[float, float]:
        return round(point[0] - x0, _COORD_DECIMALS), round(y_top - point[1], _COORD_DECIMALS)

    positions = {name: flip(point) for name, point in coordinates.items()}
    link_points = [
        [list(positions[a]), *(list(flip(bend)) for bend in bends.get(name, ())), list(positions[b])] for name, a, b, _ in links
    ]

    # Every outline's centroid is computed once, before any target is resolved: a target
    # that is a subcatchment must work whatever the iteration order and even when that
    # subcatchment flows nowhere itself, and no centroid is worth computing twice.
    centroids = {name: flip(_polygon_centroid(points)) for name, points in outlines.items()}

    polys: list[dict] = []
    polys_no_outlet = 0
    for index, (name, points) in enumerate(outlines.items()):
        centroid = centroids[name]
        record: dict = {
            "n": name,
            "i": index,
            "p": [list(flip(point)) for point in points],
            "c": list(centroid),
        }
        target = outlets.get(name)
        outlet: tuple[float, float] | None = None
        if target is not None:
            if target in known:
                outlet = positions[target]
            elif target != name and target in subcatchment_names and target in outlines:
                outlet = centroids[target]
        if outlet is not None:
            if outlet != centroid:
                record["o"] = list(outlet)
                record["on"] = target
        elif target is not None:
            polys_no_outlet += 1
        polys.append(record)
    return positions, link_points, polys, polys_no_outlet


def _polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Area centroid of the outline *points*, with a mean-of-vertices fallback.

    The shoelace formula is right for a simple polygon and wrong for everything
    else, and the extractor accepts everything else: an outline is kept on the
    number of usable points alone (:data:`_MIN_POLYGON_POINTS`), so a
    self-intersecting ring, a ring of collinear points and a ring of huge
    coordinates around a tiny area all reach this function.  Three conditions
    therefore fall back to the mean of the vertices, which is deterministic and
    always inside the box of those vertices:

    1. the signed area is zero — three collinear points pass the point count;
    2. either coordinate of the result is not finite;
    3. the result lies outside the bounding box of the polygon's own vertices.

    Condition 3 is not caution, it is a fix.  For ``(0, 0), (10, 10), (0, 10),
    (8, 0)`` the signed area is 10, so condition 1 does not fire, and the
    formula answers ``(6, 20)`` — outside ``[0, 10] x [0, 10]``.  Normalised,
    that is ``y = -10``, which :func:`_bbox` (it measures maxima from zero)
    does not cover at all; mirrored vertically the same ring answers ``(6, -10)``
    and normalises to ``y = 20``, which would raise ``bbox.h`` from 10 to 20.
    Depending only on the orientation we would lose either the framing of the
    drawing or the promised immutability of the frame, and condition 3 closes
    both cases with one rule.  It closes numerical degeneration as well: a
    five-centimetre square around ``(6 500 000, 5 700 000)`` loses most of the
    significant digits of the shoelace sums.

    The fallback itself has to be finite for **every** finite input, because it
    is what the three conditions fall back *to*: see :func:`_vertex_mean`.
    """
    low_x = min(point[0] for point in points)
    high_x = max(point[0] for point in points)
    low_y = min(point[1] for point in points)
    high_y = max(point[1] for point in points)
    mean = _vertex_mean(points, (low_x, low_y, high_x, high_y))
    twice_area = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        cross = x1 * y2 - x2 * y1
        twice_area += cross
        sum_x += (x1 + x2) * cross
        sum_y += (y1 + y2) * cross
    if twice_area == 0.0:
        return mean
    centroid = (sum_x / (3.0 * twice_area), sum_y / (3.0 * twice_area))
    if not (math.isfinite(centroid[0]) and math.isfinite(centroid[1])):
        return mean
    if not (low_x <= centroid[0] <= high_x and low_y <= centroid[1] <= high_y):
        return mean
    return centroid


def _vertex_mean(
    points: list[tuple[float, float]],
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Mean of the vertices, computed so that it is finite for any finite input.

    The obvious ``sum(...) / len(points)`` is not: two vertices at 1e308 sum to
    ``inf`` and the division of an infinity is still an infinity, so the
    *fallback* of :func:`_polygon_centroid` — the value all three of its
    conditions return — would itself be non-finite and a NaN would reach the
    payload.  Each coordinate is therefore shifted to the low corner of the
    vertices' own bounding box first and divided *before* it is added: every
    term is then between 0 and the width of that box, and so is their sum.

    The shift cannot always save it: vertices at -1e308 and +1e308 have a box
    wider than a float, and the shifted coordinate overflows on its own.  Then
    the centre of that box answers, written ``low / 2 + high / 2`` and never
    ``(low + high) / 2``, which is the same overflow again.  The result is
    inside the box either way, which is what conditions 1 and 3 of
    :func:`_polygon_centroid` promise the caller.
    """
    low_x, low_y, high_x, high_y = box
    count = len(points)
    mean_x = low_x + math.fsum((point[0] - low_x) / count for point in points)
    mean_y = low_y + math.fsum((point[1] - low_y) / count for point in points)
    if math.isfinite(mean_x) and math.isfinite(mean_y):
        return min(max(mean_x, low_x), high_x), min(max(mean_y, low_y), high_y)
    return low_x / 2.0 + high_x / 2.0, low_y / 2.0 + high_y / 2.0


def _raw_origin(
    node_names: list[str],
    coordinates: dict[str, tuple[float, float]],
    links: list[tuple[str, str, str, str]],
    bends: dict[str, list[tuple[float, float]]],
    outlines: dict[str, list[tuple[float, float]]],
) -> tuple[float, float]:
    """Left edge and top edge of everything that will actually be drawn.

    All three sources are taken into account — [COORDINATES], the bends of
    [VERTICES] and the outlines of [Polygons] — because a conduit bent outside
    the hull of its nodes would otherwise normalise to a negative coordinate
    and be clipped away by the viewBox.  [MAP] is never consulted: swmmio does
    not expose it.
    """
    points = [coordinates[name] for name in node_names]
    for name, _, _, _ in links:
        points.extend(bends.get(name, ()))
    for outline in outlines.values():
        points.extend(outline)
    return min(point[0] for point in points), max(point[1] for point in points)


def _bbox(points: list) -> dict[str, float]:
    """Width and height of the normalised drawing, never zero."""
    width = max((point[0] for point in points), default=0.0)
    height = max((point[1] for point in points), default=0.0)
    return {"w": round(width, _COORD_DECIMALS) or 1.0, "h": round(height, _COORD_DECIMALS) or 1.0}


def _min_gap(positions: list) -> float | None:
    """Smallest distance between two **different** node positions, or *None*.

    Measured on the rounded positions, over the set of distinct points rather
    than over pairs of nodes: SWMM allows two nodes to sit on the same point,
    and rounding to two decimals can merge two points that were 0.004 apart.
    Either way the closest *pair of nodes* would be zero units apart and the
    density rule downstream would hold for any zoom.  The value returned is
    therefore either *None* or strictly positive.
    """
    distinct = sorted({(float(point[0]), float(point[1])) for point in positions})
    if len(distinct) < 2:
        return None
    return round(_closest_pair(distinct), _GAP_DECIMALS)


def _closest_pair(points: list[tuple[float, float]]) -> float:
    """Closest-pair sweep over points already sorted by ``(x, y)``.

    A plain double loop is quadratic, and the guard on the size of a network
    lives elsewhere, so the sweep keeps the walk near ``O(n log n)``: points
    further left than the best distance found so far leave the active set, and
    only the active points within that distance in ``y`` are measured.
    """
    best = math.inf
    active: list[tuple[float, float]] = []  # (y, x), sorted
    left = 0
    for x, y in points:
        while points[left][0] < x - best:
            active.pop(bisect.bisect_left(active, (points[left][1], points[left][0])))
            left += 1
        low = bisect.bisect_left(active, (y - best, -math.inf))
        high = bisect.bisect_right(active, (y + best, math.inf))
        for other_y, other_x in active[low:high]:
            best = min(best, math.hypot(x - other_x, y - other_y))
        bisect.insort(active, (y, x))
    return best


def _empty_payload() -> dict:
    """A model without a single node: nothing to draw, nothing to measure."""
    return {
        "version": NETMAP_VERSION,
        "origin": [0.0, 0.0],
        "bbox": {"w": 1.0, "h": 1.0},
        "min_gap": None,
        "layout": "inp",
        "nodes": [],
        "links": [],
        "polys": [],
        "skipped": {"links_dangling": 0, "polys_degenerate": 0, "polys_no_outlet": 0},
        "counts": {"elements": 0, "points": 0},
    }


def _frame(inp: Any, section: str) -> Any | None:
    """The dataframe of *section*, or *None* when the file has no such section.

    swmmio answers a missing section with an empty frame that has no columns at
    all, so every caller would otherwise have to guard twice.
    """
    try:
        frame = getattr(inp, section)
    except (AttributeError, KeyError, ValueError):
        return None
    if frame is None or frame.empty:
        return None
    return frame


def _name(value: Any) -> str:
    """The name in a cell or an index as text, whatever pandas made of the value.

    Every name in this module goes through this one function - the node sections,
    the endpoints of a link, [COORDINATES], the vertex sections and the outlet of
    a subcatchment - because a name is only ever used to look another one up, and
    two sides of a lookup that disagree draw nothing.  They disagree per *column*:
    an index of junctions called ``101``/``102`` arrives as int64 and ``str()``
    answers ``"102"``, but a single blank cell in the ``Outlet`` column of
    [SUBCATCHMENTS] promotes that column to float64, and there ``str()`` answers
    ``"102.0"``, which matches no node of the model.  The line is then dropped and
    counted as one with an unknown target, for a target the model does carry.  An
    integral float therefore goes back through :class:`int` first.

    Two consequences, both deliberate.  A name written ``001`` in the file is the
    integer ``1`` on **every** side of the lookup, so the sides still agree and the
    line is drawn - to the node the file calls ``1``, if the model has one as well;
    that ambiguity belongs to a file that names two elements with the same number
    and cannot be repaired here.  And a name that is literally ``102.0`` in the
    file is read as a float and written as ``"102"``: a SWMM name that is a decimal
    number is pathological enough that agreeing with every other section is worth
    more than its trailing zero.

    The test is ``isinstance(value, float)``, which holds for :class:`numpy.float64`
    (it subclasses :class:`float`) and not for :class:`numpy.float32`, which would
    stringify as ``"102.0"`` again.  pandas gives these sections ``float64``, so no
    path of this module reaches a 32-bit column; were one ever to, the fix belongs
    here and nowhere else, because every name of the module passes through this
    function.
    """
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value)


def _section_names(inp: Any, section: str) -> list[str]:
    """Index of *section* as plain strings; numeric names arrive as numpy scalars."""
    frame = _frame(inp, section)
    return [] if frame is None else [_name(name) for name in frame.index]


def _model_nodes(inp: Any) -> tuple[list[str], dict[str, str]]:
    """Every node of the model, in section order, with the symbol to draw it with."""
    names_by_section = {section: _section_names(inp, section) for section in _NODE_SECTIONS}
    members = {section: set(names) for section, names in names_by_section.items()}
    order: list[str] = []
    seen: set[str] = set()
    for section in _NODE_SECTIONS:
        for name in names_by_section[section]:
            if name not in seen:
                seen.add(name)
                order.append(name)
    kinds = {name: next((kind for section, kind in _NODE_KIND_PRIORITY if name in members[section]), "other") for name in order}
    return order, kinds


def _model_links(inp: Any) -> list[tuple[str, str, str, str]]:
    """Every link of the model as ``(name, from, to, kind)``, in section order."""
    links: list[tuple[str, str, str, str]] = []
    for section, kind in _LINK_SECTIONS:
        frame = _frame(inp, section)
        if frame is None or "InletNode" not in frame.columns or "OutletNode" not in frame.columns:
            continue
        for name, a, b in zip(frame.index, frame["InletNode"], frame["OutletNode"]):
            links.append((_name(name), _name(a), _name(b), kind))
    return links


def _subcatchment_outlets(inp: Any) -> dict[str, str]:
    """[SUBCATCHMENTS] as ``{subcatchment: outlet}``; a blank outlet is no outlet.

    Read exactly like :func:`_model_links` reads the link sections — the frame,
    a guard on the column, then a walk over ``zip(index, column)`` — because a
    file without the section, or with it but without the column, is an ordinary
    model and not an error.

    The module carries no pandas, so a missing cell arrives as a bare ``float``
    NaN and is recognised by ``value != value``.  That test has to come
    **before** :func:`_name`: afterwards the cell is the string ``"nan"``, which
    is a perfectly good name and would be written into the payload as one.  And
    that one blank cell is precisely what makes :func:`_name` necessary here: it
    promotes the whole column to float64, so a model with numeric node names
    would otherwise hand back ``"102.0"`` for the node called ``102``.
    """
    outlets: dict[str, str] = {}
    frame = _frame(inp, "subcatchments")
    if frame is None or "Outlet" not in frame.columns:
        return outlets
    for name, value in zip(frame.index, frame["Outlet"]):
        if value is None or value != value:
            continue
        target = _name(value)
        if not target.strip():
            continue
        outlets.setdefault(_name(name), target)
    return outlets


def _coordinates(inp: Any) -> dict[str, tuple[float, float]]:
    """[COORDINATES] as a mapping; a NaN or an infinity counts as no coordinate."""
    frame = _frame(inp, "coordinates")
    points: dict[str, tuple[float, float]] = {}
    if frame is None or "X" not in frame.columns or "Y" not in frame.columns:
        return points
    for name, x, y in zip(frame.index, frame["X"], frame["Y"]):
        key = _name(name)
        point = _finite_point(x, y)
        if point is not None and key not in points:
            points[key] = point
    return points


def _grouped_points(frame: Any | None) -> dict[str, list[tuple[float, float]]]:
    """Rows of a vertex-style section grouped by name, in the order of the file."""
    groups: dict[str, list[tuple[float, float]]] = {}
    if frame is None or "X" not in frame.columns or "Y" not in frame.columns:
        return groups
    for name, x, y in zip(frame.index, frame["X"], frame["Y"]):
        point = _finite_point(x, y)
        if point is not None:
            groups.setdefault(_name(name), []).append(point)
    return groups


def _finite_point(x: Any, y: Any) -> tuple[float, float] | None:
    """``(x, y)`` as plain floats, or *None* when either is missing or not finite."""
    try:
        point = (float(x), float(y))
    except (TypeError, ValueError):
        return None
    return point if math.isfinite(point[0]) and math.isfinite(point[1]) else None
