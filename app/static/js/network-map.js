/**
 * Network Map — static drawing of the analysed sewer network.
 *
 * Reads the geometry envelope embedded by the results page as
 * <script id="network-map-data" type="application/json"> and builds the whole SVG in
 * the browser: nothing of the drawing is rendered server side.  The module owns its
 * own IIFE and never touches the Results V2 widget engine; the only contract between
 * the two is the visibility of #r-map-panel, which this module watches with a
 * ResizeObserver instead of hooking into the tab machinery.
 *
 * Five groups are drawn, back to front: subcatchment outlines (nm-polys), the line
 * from each subcatchment to its outlet (nm-outlets), links (nm-links), nodes
 * (nm-nodes) and the element names (nm-labels, built lazily when they are switched on).  The outlet lines come before the pipes on purpose: the line is a hint
 * about where the runoff goes, not part of the network, so it runs *under* the pipes
 * and never over them.  Node symbols are sized in *screen* pixels — a
 * symbol drawn in drawing units grows with the zoom and buries the network under
 * circles — so every node is a <g> scaled by r / pxPerUnit and every stroke carries
 * vector-effect: non-scaling-stroke.
 */
(function () {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';

  const PAD_FRACTION = 0.03;
  /** Margin around the drawing, as a fraction of its longer side. */

  const PAD_MIN_PX = 8;
  /** ...but never thinner than this once converted to screen pixels. */

  const MAX_SYMBOL_PX = 3.5;
  const MIN_SYMBOL_PX = 1.5;
  /** A node symbol never grows past MAX and is hidden rather than drawn below MIN. */

  const DENSITY_DIVISOR = 3;
  /** A symbol may take up to a third of the closest gap between two node positions. */

  const DRAG_PX = 4;
  /** Past this distance a pointer gesture is a pan, and the click ending it selects nothing. */

  const ZOOM_STEP = 1.15;
  /** One notch of the wheel and one press of the + / - control. */

  const ZOOM_OUT_LIMIT = 2;
  const ZOOM_IN_LIMIT = 50;
  /** The viewBox never grows past twice the fitted box, nor shrinks below a fiftieth of it. */

  const HIT_STROKE_PX = 12;
  /** Screen width of the invisible stroke that catches a pointer on a link. */

  const HIT_RADIUS_PX = 8;
  /** Screen radius of the invisible circle that catches a pointer on a node. */

  const HINT_MS = 1500;
  /** How long the "use Ctrl" overlay stays up after an unmodified wheel. */

  const HINT_TEXT = 'Use Ctrl + scroll to zoom';
  /**
   * What the overlay says unless a caller says otherwise.
   *
   * It is a constant because the overlay is borrowed: the names button shows its own
   * sentence in the same element, and the default has to be restored on every show or
   * the next unmodified wheel would explain the wrong thing.
   */

  const PAN_STEP_PX = 40;
  /** One press of an arrow key, in screen pixels — the same unit a drag speaks. */

  const LABEL_FONT_PX = 11;
  const LABEL_LINE_PX = 1.2;
  /** Type of a name on the map: 11 px on a 1.2 line — the size E4b measures with. */

  const LABEL_HALO_PX = 1.5;
  /** Half of the 3 px halo a name is stroked with, so it counts on both sides. */

  const LABEL_MARGIN_PX = 2;
  /**
   * Breathing space between a name and whatever it stands next to.
   *
   * Owned by the caller, never by the solver: E4b adds it once, into the r of the
   * disks and the hw of the segments, and leaves opts.margin at 0.  Added in both
   * places the margin would simply count twice.
   */

  const LABEL_POLY_RING_PX = 10;
  /**
   * Radius of the ring the eight fallback anchors of a subcatchment are measured from.
   *
   * A subcatchment anchors on its centroid, which has no size (r = 0), and the
   * production margin is 0, so plain offsets would put every one of the nine
   * candidates on the edge of the centroid itself: a node sitting in that centroid
   * would collide with all nine and the name could never be drawn.  Ten is written
   * out rather than derived so a test can measure it, and it is not eight or nine by
   * accident: the largest node disk is MAX_SYMBOL_PX * sqrt(2) + 3 (the .nm-sel
   * stroke) + LABEL_MARGIN_PX ~ 9.95, and an anchor inside that disk collides too.
   */

  const LABEL_CELL_PX = 32;
  /** Side of one bucket of the obstacle grid, in screen pixels. */

  const LABEL_MAX = 600;
  /** How many elements are ever offered a name, selection aside. */

  const LABEL_MIN_GAP_PX = 6;
  /** Below this screen gap between two nodes the whole group of names is hidden (E4b). */

  const LABEL_EPS_PX = 1e-6;
  /**
   * Touching is not colliding.
   *
   * Every candidate is offset by exactly r (or hw), so the winning position is
   * regularly tangent to the obstacle it was moved away from, and 16.2 - 16.2 does
   * not always come back as zero in binary.  Without this tolerance the N anchor of
   * a horizontal pipe would fail by one ulp and the pipe would lose its name.
   */

  const LABEL_ANCHORS = ['E', 'W', 'N', 'S', 'NE', 'SE', 'NW', 'SW'];
  /** The eight positions, tried in this order; the first free one wins. */

  const LABEL_MAX_CELLS = 4096;
  /**
   * Past this many buckets an obstacle stops being bucketed and is simply tested
   * every time.  A superset of the candidates is always safe — the exact test
   * decides — and this keeps an unclipped, enormous segment from filling the grid.
   */

  const NOT_ANALYSED_CLASS = 'nm-na';
  /**
   * A drawn element the pipeline computed nothing for.
   *
   * "Not analysed" is decided by the **absence of a row in the Results V2 collection**
   * of that kind and by nothing else — never by ``k`` and never by ``kind``.  The two
   * would be a second copy of the pipeline's knowledge living in the map: ``dfn`` of
   * DataManager is junctions + outfalls + **storage**, so a [STORAGE] unit is analysed
   * in full and blocking it on its kind would be plain wrong, while a [DIVIDERS] node
   * gets no row and blocks itself under the same rule, with nothing written down here.
   */

  const PREFS_KEY = 'stormwater-network-map-preferences';
  /**
   * The map keeps its own preferences under its own key.
   *
   * Results V2 owns ``stormwater-results-v2-preferences`` and nothing here reads or
   * writes it: the two sets are independent, and one shared read would make a broken
   * value of either one break both.
   */

  const DEFAULT_MODE = 'recommendation';
  /** D7 - the map opens on the recommendation of the model, not on a hydraulic number. */

  const DEFAULT_LABELS = false;
  /**
   * D20 - the names of the elements start hidden.
   *
   * A drawing that opens with six hundred names written over it is unreadable before
   * the reader has done anything, and the gesture that brings them back is one button.
   */

  const NO_DATA_CLASS = 'nm-nodata';
  /** No results row, no usable value: a grey the three modes share. */

  /**
   * The nine recommendation codes of the model, each with the class that paints it
   * and the label the legend shows.
   *
   * A copy of ``ACTIONS`` of results-v2.js (line 56) and deliberately a copy: the two
   * files share no module system, and an import would make the map depend on the load
   * order of the table.  The class names are written out in full rather than built
   * from the code with a template, because a static test reads them out of this file
   * and matches them against the rules of network-map.css - a name assembled at run
   * time is invisible to that test, and an unpainted class is invisible on screen.
   */
  const REC_CLASSES = [
    ['valid', 'nm-c-valid', 'No changes'],
    ['diameter_increase', 'nm-c-diameter_increase', 'Increase diameter'],
    ['diameter_reduction', 'nm-c-diameter_reduction', 'Reduce diameter'],
    ['slope_increase', 'nm-c-slope_increase', 'Increase slope'],
    ['slope_reduction', 'nm-c-slope_reduction', 'Reduce slope'],
    ['depth_increase', 'nm-c-depth_increase', 'Increase depth'],
    ['pump', 'nm-c-pump', 'Pump station'],
    ['tank', 'nm-c-tank', 'Retention tank'],
    ['seepage_boxes', 'nm-c-seepage_boxes', 'Seepage boxes']
  ];

  /** Code -> class, built once from the table above. */
  const REC_CLASS_BY_CODE = REC_CLASSES.reduce((table, entry) => {
    table[entry[0]] = entry[1];
    return table;
  }, Object.create(null));

  /** How many of the seven checks the conduit passed, 0..7. */
  const VALIDITY_CLASSES = ['nm-v-0', 'nm-v-1', 'nm-v-2', 'nm-v-3', 'nm-v-4', 'nm-v-5', 'nm-v-6', 'nm-v-7'];

  /**
   * Filling, in five bands.  The last one is its own colour and not the end of a
   * ramp: at and above 100 % the pipe runs full, which is a state and not a degree.
   */
  const FILL_CLASSES = ['nm-f-0', 'nm-f-1', 'nm-f-2', 'nm-f-3', 'nm-f-4'];
  const FILL_BOUNDS = [25, 50, 75, 100];
  const FILL_LABELS = ['0-25 %', '25-50 %', '50-75 %', '75-100 %', 'Full (>= 100 %)'];

  /** The three colouring modes, in the order the switch shows them. */
  const MODES = [
    { key: 'recommendation', label: 'Recommendation' },
    { key: 'validity', label: 'Checks passed' },
    { key: 'filling', label: 'Filling' }
  ];

  /**
   * The two vocabularies of the map, translated in exactly one place.
   *
   * ``link|node|poly`` is the kind of *drawing*: it is what the geometry payload and
   * ``data-nm-kind`` speak.  ``conduits|nodes|catchments`` is the key of a Results V2
   * tab, schema and collection.  Every event this module sends out speaks the tab
   * vocabulary, translated here before dispatch, so the table never translates again.
   */
  const KIND_TO_TAB = {link: 'conduits', node: 'nodes', poly: 'catchments'};
  const TAB_TO_KIND = {conduits: 'link', nodes: 'node', catchments: 'poly'};

  const COLLECTIONS = [['link', 'links'], ['node', 'nodes'], ['poly', 'polys']];
  /** Drawing kind and the key its elements live under in the geometry payload. */

  /**
   * Two or three values of the results row, shown on the selection card of each kind:
   * ``[label, key, unit, factor]``, the factor optional.
   *
   * ``flood_volume`` is SWMM's Total Flood Volume in millions of litres (models.py:148),
   * so it is multiplied by 1000 and shown in cubic metres - the conversion the flooding tooltip
   * of the table makes.  Printed raw it reads "0.00" for a real overflow of four cubic
   * metres, on the card of a node the map has just painted as flooded.
   */
  const CARD_FIELDS = {
    link: [['Recommendation', 'recommendation_label', ''], ['Diameter', 'diameter', 'm'], ['Length', 'length', 'm'], ['Max. velocity', 'velocity', 'm/s']],
    node: [['Node depth', 'depth', 'm'], ['Max. water depth', 'waterDepth', 'm'], ['Flooding volume', 'floodVolume', 'm\u00b3', 1000]],
    poly: [['Area', 'area', 'ha'], ['Imperviousness', 'impervious', '%'], ['Runoff coefficient', 'coefficient', '']]
  };

  /**
   * The three counters of ``data.skipped``: key, singular, plural.
   *
   * They sit at the root of the payload, beside ``counts`` and not inside it, and they
   * count the elements that **cannot be drawn in any mode**: a link whose end is not a
   * node of the model, a subcatchment whose [Polygons] block is not a polygon, and an
   * outlet line whose target is not an object the model knows.  The wording says
   * exactly that, because "elements without geometry" would name something else
   * entirely — a model without coordinates puts the whole network into the automatic
   * layout and leaves all three at zero.  The third counter names the *line*, not the
   * subcatchment: the outline itself is drawn, only the line to its outlet is not.
   *
   * A payload of version 1 carries no ``polys_no_outlet`` key at all, and skippedLabel
   * reads a missing key as zero, so an older session gets no third counter and no
   * false zero either.
   */
  const SKIPPED_FIELDS = [
    ['links_dangling', 'link with an unknown endpoint', 'links with an unknown endpoint'],
    ['polys_degenerate', 'subcatchment without a polygon', 'subcatchments without a polygon'],
    ['polys_no_outlet', 'outlet line with an unknown target', 'outlet lines with an unknown target']
  ];

  // ---------------------------------------------------------------------------
  // Pure helpers — no DOM, unit-tested from Node by the harness
  // ---------------------------------------------------------------------------

  function isPositive(value) {
    return Number.isFinite(value) && value > 0;
  }

  /**
   * Number of drawable elements in a payload, counted exactly as the extractor does
   * (a link costs two: its ends carry the network, its body carries the picture, and
   * an outlet line costs one more; the centroid is not drawn and costs nothing).
   *
   * A version 1 payload has no ``o`` on any polygon, so the last term is zero there and
   * the formula gives exactly what it gave before.
   */
  function elementCount(data) {
    if (!data || typeof data !== 'object') return 0;
    const polyList = Array.isArray(data.polys) ? data.polys : [];
    const links = Array.isArray(data.links) ? data.links.length : 0;
    const nodes = Array.isArray(data.nodes) ? data.nodes.length : 0;
    const outlets = polyList.filter(poly => poly && typeof poly === 'object' && Array.isArray(poly.o)).length;
    return polyList.length + 2 * links + nodes + outlets;
  }

  /**
   * The sentence under the drawing, or ``null`` when there is nothing to say.
   *
   * A component of zero is left out of the brackets, and a total of zero gives ``null``
   * so that the caller builds no element at all: a footer reading "0 elements not
   * drawn" is noise on every healthy model, which is nearly all of them.  Everything
   * unusable — a missing object, a string, a negative or non-finite count — reads as
   * zero rather than reaching the page as "NaN elements not drawn".
   */
  function skippedLabel(skipped) {
    if (!skipped || typeof skipped !== 'object') return null;
    let total = 0;
    const parts = [];
    SKIPPED_FIELDS.forEach(field => {
      const raw = Object.prototype.hasOwnProperty.call(skipped, field[0]) ? Number(skipped[field[0]]) : 0;
      const count = Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 0;
      if (count <= 0) return;
      total += count;
      parts.push(`${count} ${count === 1 ? field[1] : field[2]}`);
    });
    if (total <= 0) return null;
    return `${total} ${total === 1 ? 'element' : 'elements'} not drawn (${parts.join(', ')}) \u2014 table only`;
  }

  /**
   * ``M x y L x y ...`` for a list of ``[x, y]`` pairs, skipping anything not finite.
   * Fewer than two usable points is not a line, and an empty ``d`` draws nothing.
   */
  function pointsToPath(points) {
    if (!Array.isArray(points)) return '';
    const usable = points.filter(p => Array.isArray(p) && Number.isFinite(p[0]) && Number.isFinite(p[1]));
    if (usable.length < 2) return '';
    return usable.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]} ${p[1]}`).join(' ');
  }

  /**
   * ``M cx cy L ox oy`` for the outlet line of *poly*, or ``''`` when there is none.
   *
   * The guards are the ones pointsToPath carries and they are there for the same
   * reason: a single unusable coordinate reaches the DOM as ``d="M NaN NaN ..."`` and
   * the browser drops the whole path silently.  A degenerate line — the centroid and
   * the outlet in the same place — is not a line either.  In practice the degenerate
   * case never gets this far, because the extractor emits the pair ``o``/``on`` only
   * when the two points differ; the guard stays as the defence against a version 2
   * payload built somewhere else.
   */
  function outletPath(poly) {
    if (!poly || typeof poly !== 'object') return '';
    const c = poly.c;
    const o = poly.o;
    if (!Array.isArray(c) || !Array.isArray(o)) return '';
    if (!Number.isFinite(c[0]) || !Number.isFinite(c[1])) return '';
    if (!Number.isFinite(o[0]) || !Number.isFinite(o[1])) return '';
    if (c[0] === o[0] && c[1] === o[1]) return '';
    return `M${c[0]} ${c[1]} L${o[0]} ${o[1]}`;
  }

  /**
   * The viewBox that shows the whole drawing inside a container of *sizePx*.
   *
   * The margin is the larger of PAD_FRACTION of the drawing and *padPx* screen pixels
   * converted into drawing units at the scale the drawing would be shown without any
   * margin — which is why the container size is needed at all.  Every degenerate input
   * (missing bbox, zero or non-finite side, container of zero size) falls back to a
   * finite box: a NaN here reaches the DOM as viewBox="NaN NaN NaN NaN" and the map
   * silently disappears.
   */
  function fitViewBox(bbox, padPx, sizePx) {
    const w = bbox && isPositive(bbox.w) ? bbox.w : 1;
    const h = bbox && isPositive(bbox.h) ? bbox.h : 1;
    let pad = PAD_FRACTION * Math.max(w, h);
    const width = sizePx && Number(sizePx.w);
    const height = sizePx && Number(sizePx.h);
    if (isPositive(width) && isPositive(height) && isPositive(padPx)) {
      const scale = Math.min(width / w, height / h);
      if (isPositive(scale)) pad = Math.max(pad, padPx / scale);
    }
    if (!Number.isFinite(pad) || pad < 0) pad = 0;
    return { x: -pad, y: -pad, w: w + 2 * pad, h: h + 2 * pad };
  }

  /**
   * *vb* zoomed by *k* about the drawing-unit point ``(ux, uy)``, which keeps that
   * point under the cursor.  A factor that is not a positive finite number, or a box
   * that is already degenerate, is returned unchanged rather than turned into NaN.
   *
   * *limits* is optional; given as ``{minW, maxW}`` it bounds the width of the result.
   * The factor is *reduced* to what still fits rather than the zoom being refused, so
   * a wheel at the limit lands exactly on the limit with the anchor still under the
   * cursor, instead of jumping back to where it started.
   */
  function zoomAt(vb, ux, uy, k, limits) {
    if (!vb || !isPositive(vb.w) || !isPositive(vb.h)) return vb;
    if (!isPositive(k) || !Number.isFinite(ux) || !Number.isFinite(uy)) return vb;
    let factor = k;
    if (limits) {
      const minW = Number(limits.minW);
      const maxW = Number(limits.maxW);
      if (isPositive(minW) && vb.w / factor < minW) factor = vb.w / minW;
      if (isPositive(maxW) && vb.w / factor > maxW) factor = vb.w / maxW;
    }
    if (!isPositive(factor)) return vb;
    return {
      x: ux - (ux - vb.x) / factor,
      y: uy - (uy - vb.y) / factor,
      w: vb.w / factor,
      h: vb.h / factor
    };
  }

  /**
   * Radius of a node symbol in screen pixels.
   *
   * The early returns are obligatory: ``null``, ``undefined``, ``NaN`` and ``Infinity``
   * all reach this function from a real payload (``min_gap`` is null for a model with a
   * single node position), and the arithmetic would turn them into 1.5 or into NaN —
   * the second of which ends up as r="NaN" in the SVG.  ``minGap === NaN`` is never a
   * usable test in JavaScript; Number.isFinite is.
   */
  function symbolRadiusPx(minGap, pxPerUnit) {
    if (!Number.isFinite(minGap) || minGap <= 0) return MAX_SYMBOL_PX;
    if (!Number.isFinite(pxPerUnit) || pxPerUnit <= 0) return MAX_SYMBOL_PX;
    return Math.min(MAX_SYMBOL_PX, Math.max(MIN_SYMBOL_PX, (minGap * pxPerUnit) / DENSITY_DIVISOR));
  }

  /**
   * The running reach of a pointer gesture: the largest distance from where it began,
   * never the length of its path.  A loop that comes back to its starting point stays
   * a drag, and measuring the two axes apart would turn the three-pixel move
   * (0,0) -> (3,0) -> (0,3) into 4.24 and refuse a legitimate click.
   */
  function trackDist(maxDist, x0, y0, x, y) {
    const previous = Number.isFinite(maxDist) ? maxDist : 0;
    if (!Number.isFinite(x0) || !Number.isFinite(y0)) return previous;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return previous;
    return Math.max(previous, Math.hypot(x - x0, y - y0));
  }

  /**
   * Was the gesture that ended in this click a drag?
   *
   * A pure threshold on the distance ``trackDist`` accumulated.  The browser sends a
   * click after every ``pointerdown -> move -> pointerup`` and the click event itself
   * carries no deltas, so the distance has to be handed in.  There is deliberately no
   * condition on duration: a slow, deliberate press is a click like any other, and the
   * time a gesture took says nothing about what the reader meant by it.
   */
  function isDrag(dist) {
    return Number.isFinite(dist) && dist > DRAG_PX;
  }

  /**
   * Where the selection card is allowed to stand inside the canvas.
   *
   * Pure arithmetic on six numbers, so the whole of it is testable from Node: the
   * wanted corner, the size of the card and the size of the canvas in, the corner the
   * card may really take out.  ``null`` means "this cannot be computed - leave the
   * position the stylesheet gives" and is the answer to every unusable argument: a
   * value that is not a finite number, and a width or a height that is zero or
   * negative.  ``card.offsetWidth === 0`` is a real production state - the map panel
   * is still hidden - and answering it with a corner would write a meaningless
   * position over the one the reader dragged the card to.
   *
   * A card wider or taller than the canvas clamps to 0 on that axis: it sticks to the
   * left or top edge instead of walking off into negative coordinates.  Its own size
   * is capped in the stylesheet (.nm-card, max-width/max-height), because arithmetic
   * here can move a card but never shrink one.
   */
  function clampCardPos(left, top, cardW, cardH, hostW, hostH) {
    const numbers = [left, top, cardW, cardH, hostW, hostH];
    if (!numbers.every(value => Number.isFinite(value))) return null;
    if (cardW <= 0 || cardH <= 0 || hostW <= 0 || hostH <= 0) return null;
    const maxLeft = Math.max(0, hostW - cardW);
    const maxTop = Math.max(0, hostH - cardH);
    return {
      left: Math.min(Math.max(left, 0), maxLeft),
      top: Math.min(Math.max(top, 0), maxTop)
    };
  }

  /**
   * What a wheel event over the map means.
   *
   * Without Ctrl (or Cmd) the map takes nothing: the event is neither cancelled nor
   * acted on, and whatever the page would have scrolled, it scrolls.  With the
   * modifier the map zooms and cancels the event, which also stops the browser's own
   * page zoom.  One gesture, one meaning, chosen by a modifier the reader can see —
   * not by an invisible "did you click the map first" state.
   */
  function wheelAction(event) {
    const modified = Boolean(event && (event.ctrlKey || event.metaKey));
    const delta = event ? Number(event.deltaY) : 0;
    let factor = 1;
    if (Number.isFinite(delta) && delta !== 0) factor = delta < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
    return {zoom: modified, prevent: modified, factor: factor};
  }

  /**
   * Look *key* up in one of the two dictionaries above, own keys only.
   *
   * A plain property read would answer ``kindToTab('toString')`` with a function off
   * Object.prototype, and every caller here tests the answer for truth before using
   * it — so an inherited member would pass for a translation.
   */
  function translate(table, key) {
    return Object.prototype.hasOwnProperty.call(table, key) ? table[key] : undefined;
  }

  /** Drawing kind -> Results V2 collection key; an unknown value gives undefined. */
  function kindToTab(kind) {
    return translate(KIND_TO_TAB, kind);
  }

  /** Results V2 collection key -> drawing kind; an unknown value gives undefined. */
  function tabToKind(tab) {
    return translate(TAB_TO_KIND, tab);
  }

  /**
   * The elements of one drawn collection that the table of that kind lists at all —
   * the other half of the dimming, and the one that keeps it honest.
   *
   * ``dimSet`` answers "which of these did the filter leave out?", and an element the
   * table never had a row for answers yes to that question on every render, filtered
   * or not: a pump, an orifice, a weir and an outlet are drawn as links but only
   * conduits reach the conduit table, a [DIVIDERS] node is drawn but analysed nowhere.
   * Fed the whole collection, the very first, unfiltered ``rv2:filterchange`` would
   * fade all of them to 15 % and take the pointer away from them for good — the same
   * elements ``markUnanalysed`` deliberately leaves clickable so the tooltip can say
   * why they open nothing.  So the collection is narrowed *before* the question is
   * asked: "not in the table" and "filtered out of the table" are different states and
   * only the second one dims.
   *
   * *isKnown* is called with the name of each element and answers whether the results
   * collection of that kind carries a row for it.  Anything unusable — a collection
   * that is not a list, a predicate that is not a function — gives an empty answer, and
   * an empty collection fades nothing.
   */
  function analysedElements(elements, isKnown) {
    if (!Array.isArray(elements) || typeof isKnown !== 'function') return [];
    return elements.filter(element => {
      if (!element || typeof element !== 'object') return false;
      return isKnown(String(element.n)) === true;
    });
  }

  /**
   * Which drawn elements a filtered table leaves out — the pure half of the dimming.
   *
   * *tab* is the collection the table has just rendered, in the table's own
   * vocabulary ('conduits' | 'nodes' | 'catchments'), because the event that carries
   * it is emitted by the table; it is translated here and nowhere else.  *visible*
   * are the names the table is showing and *collections* the drawn elements by
   * drawing kind, each of them ``{n: name, i: index}`` as the geometry payload
   * spells them.
   *
   * The answer is a list of indices **inside the one collection named by tab** and
   * can be nothing else: the names of the conduits a filter left standing say
   * nothing whatsoever about the nodes, so a set built across the three collections
   * would fade every node and every subcatchment the moment the conduit table was
   * filtered at all.
   *
   * An unknown tab, a missing collection or a ``visible`` that is not a list of names
   * all give an empty answer — "fade nothing" is the safe reading of a message that
   * cannot be understood.
   */
  function dimSet(tab, visible, collections) {
    const kind = tabToKind(tab);
    if (!kind || !collections || typeof collections !== 'object') return [];
    const elements = Object.prototype.hasOwnProperty.call(collections, kind) ? collections[kind] : null;
    if (!Array.isArray(elements) || !Array.isArray(visible)) return [];

    const shown = new Set();
    visible.forEach(name => {
      if (typeof name === 'string') shown.add(name);
    });

    const hidden = [];
    elements.forEach(element => {
      if (!element || typeof element !== 'object') return;
      if (!Number.isInteger(element.i)) return;
      if (!shown.has(String(element.n))) hidden.push(element.i);
    });
    return hidden;
  }

  /**
   * The Results V2 row of the element *name* of drawing kind *kind*, or ``null``.
   *
   * The collection is chosen by the kind and never searched across: a node and a link
   * may legally carry the same name, and one shared map from name to row would hand
   * out the wrong one.  Names are compared as they arrive — no trimming, no case
   * folding, no unescaping — so ``A/B``, ``P 1``, ``s#1`` and ``"x"`` each find their
   * own row and nothing else.
   */
  function findResult(payload, kind, name) {
    const tab = kindToTab(kind);
    if (!tab || !payload || typeof payload !== 'object') return null;
    const rows = Object.prototype.hasOwnProperty.call(payload, tab) ? payload[tab] : null;
    if (!Array.isArray(rows)) return null;
    for (let index = 0; index < rows.length; index += 1) {
      const row = rows[index];
      if (row && row.name === name) return row;
    }
    return null;
  }

  /**
   * The CSS class that paints one results row in one colouring mode.
   *
   * The contract is a *class name* and never a colour: the values live in
   * network-map.css next to the rest of the palette, an SVG inherits neither
   * ``color`` nor ``background-color``, and a HEX literal here would be a second
   * palette nobody maintains.  Everything unusable - no row, an unknown mode, an
   * unknown recommendation code, a missing or non-numeric value - answers with the
   * one grey class, so a pipe is always painted and never left unstyled.
   *
   * ``recommendation`` is normalised exactly as ``getActionLabel`` of results-v2.js
   * normalises it, because the payload carries the raw database value and real
   * sessions hold ``"VALID"`` as readily as ``"valid"``.
   */
  function pickColor(mode, row) {
    if (!row || typeof row !== 'object') return NO_DATA_CLASS;

    if (mode === 'recommendation') {
      const raw = row.recommendation === undefined ? row.action : row.recommendation;
      if (typeof raw !== 'string') return NO_DATA_CLASS;
      const code = raw.trim().toLowerCase();
      const found = REC_CLASS_BY_CODE[code];
      return found === undefined ? NO_DATA_CLASS : found;
    }

    if (mode === 'validity') {
      // typeof, not Number(): Number(null) is 0 and would paint a row with no
      // validation at all as a conduit that failed all seven checks.
      const passed = row.validation;
      if (typeof passed !== 'number' || !Number.isInteger(passed)) return NO_DATA_CLASS;
      if (passed < 0 || passed >= VALIDITY_CLASSES.length) return NO_DATA_CLASS;
      return VALIDITY_CLASSES[passed];
    }

    if (mode === 'filling') {
      const pct = row.fillPercent;
      if (typeof pct !== 'number' || !Number.isFinite(pct) || pct < 0) return NO_DATA_CLASS;
      for (let band = 0; band < FILL_BOUNDS.length; band += 1) {
        if (pct < FILL_BOUNDS[band]) return FILL_CLASSES[band];
      }
      // At and above 100 % the pipe runs full; everything above it is the same state.
      return FILL_CLASSES[FILL_CLASSES.length - 1];
    }

    return NO_DATA_CLASS;
  }

  /**
   * The legend state after a click on the entry *key*, as a new array of dimmed keys.
   *
   * A pure toggle: a key that is dimmed comes back, a key that is not gets dimmed, and
   * ``null`` (what Esc sends) clears everything.  The argument is never mutated - the
   * caller compares the two states to decide which elements have to change - and
   * anything that is not an array of strings is read as "nothing is dimmed".
   */
  function nextLegendState(state, key) {
    const current = Array.isArray(state) ? state.filter(item => typeof item === 'string') : [];
    if (key === null || key === undefined) return [];
    const wanted = String(key);
    if (current.indexOf(wanted) === -1) return current.concat([wanted]);
    return current.filter(item => item !== wanted);
  }

  /**
   * The stored map preferences, read from an injected storage-like object.
   *
   * The storage is an argument and not ``localStorage`` itself so that the function is
   * testable from Node without a DOM.  Every failure - no storage, a getItem that
   * throws (a browser with site data blocked does exactly that), a missing value,
   * broken JSON, a mode nobody knows - gives the default mode back.  There is no path
   * out of this function that throws, and none that returns an unknown mode.
   */
  function readMapPrefs(storageLike) {
    // A fresh object on every path, including every failure: the caller keeps what it
    // gets, and one shared fallback would let one map write into the next read.
    function fallback() {
      return { mode: DEFAULT_MODE, card: null, labels: DEFAULT_LABELS };
    }
    if (!storageLike || typeof storageLike.getItem !== 'function') return fallback();

    let raw;
    try {
      raw = storageLike.getItem(PREFS_KEY);
    } catch (err) {
      return fallback();
    }
    if (typeof raw !== 'string' || raw === '') return fallback();

    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      return fallback();
    }
    if (!parsed || typeof parsed !== 'object') return fallback();
    // One line per field, each field decided on its own.  The whole-object fallback this
    // replaces was a bug the users already have in their storage: {"mode":"filling"} is
    // what every reader of the map wrote before plan24, and a missing ``card`` (E2) or
    // ``labels`` (E4b) must cost the default of that one field, never the remembered
    // mode - just as a broken card may not cost the names beside it.
    const stored = parsed.card;
    const usable =
      Boolean(stored) &&
      typeof stored === 'object' &&
      !Array.isArray(stored) &&
      Number.isFinite(stored.left) &&
      Number.isFinite(stored.top) &&
      stored.left >= 0 &&
      stored.top >= 0;
    return {
      mode: isMode(parsed.mode) ? parsed.mode : DEFAULT_MODE,
      card: usable ? { left: stored.left, top: stored.top } : null,
      labels: parsed.labels === true
    };
  }

  /** Is *mode* one of the three the switch offers?  Own keys only, as ever. */
  function isMode(mode) {
    return MODES.some(entry => entry.key === mode);
  }

  /**
   * Store the whole preference object, or lose it silently: a preference is not worth
   * an error.
   *
   * It takes the object and never one field, and the module has exactly one caller for
   * it (``savePrefs``).  A writer per gesture is how "turn the names on, switch the
   * colouring, reload, names gone" happens: the last write wins and it carried one
   * field - and it is exactly how the dragged position of the card would disappear on
   * the next click of the mode switch.  All three fields travel together.
   */
  function writeMapPrefs(storageLike, prefs) {
    if (!storageLike || typeof storageLike.setItem !== 'function') return;
    const value = prefs && typeof prefs === 'object' ? prefs : {};
    try {
      storageLike.setItem(PREFS_KEY, JSON.stringify({ mode: value.mode, card: value.card || null, labels: value.labels === true }));
    } catch (err) {
      /* a full or blocked storage costs the preference and nothing else */
    }
  }

  /**
   * ``window.localStorage``, or null.
   *
   * Even *reading* the property throws a SecurityError where site data is blocked, so
   * the access itself - not only the call to getItem - has to sit in a try.
   */
  function mapStorage() {
    try {
      return typeof window !== 'undefined' ? window.localStorage : null;
    } catch (err) {
      return null;
    }
  }

  /** The legend of one mode: ``[key, label]`` pairs, the grey class last. */
  function legendEntries(mode) {
    if (mode === 'validity') {
      return VALIDITY_CLASSES.map((cls, passed) => [cls, `${passed} / 7 checks passed`]).concat([[NO_DATA_CLASS, 'No data']]);
    }
    if (mode === 'filling') {
      return FILL_CLASSES.map((cls, band) => [cls, FILL_LABELS[band]]).concat([[NO_DATA_CLASS, 'No data']]);
    }
    return REC_CLASSES.map(entry => [entry[1], entry[2]]).concat([[NO_DATA_CLASS, 'No data']]);
  }

  /** The symbol of one node kind, in units of its own radius around (0, 0). */
  function symbolFor(kind) {
    if (kind === 'outfall') {
      return { tag: 'path', attrs: { d: 'M0 -1.15 L1 0.8 L-1 0.8 Z' } };
    }
    if (kind === 'storage') {
      return { tag: 'rect', attrs: { x: '-1', y: '-1', width: '2', height: '2' } };
    }
    if (kind === 'junction') {
      return { tag: 'circle', attrs: { r: '1' } };
    }
    return { tag: 'path', attrs: { d: 'M0 -1 L1 0 L0 1 L-1 0 Z' } };
  }

  // ---------------------------------------------------------------------------
  // Drawing
  // ---------------------------------------------------------------------------

  function svgNode(tag, attrs, className) {
    const el = document.createElementNS(SVG_NS, tag);
    if (className) el.setAttribute('class', className);
    if (attrs) {
      Object.keys(attrs).forEach(name => el.setAttribute(name, attrs[name]));
    }
    return el;
  }

  /**
   * Give *el* an accessible name and a native tooltip, as data and never as markup.
   *
   * The ``<title>`` child is what a browser shows when the pointer rests on the shape
   * and what a screen reader announces; the floating tooltip of this module is a
   * second, faster channel, not a replacement for it.
   */
  function titled(el, name) {
    const title = document.createElementNS(SVG_NS, 'title');
    title.textContent = name;
    el.appendChild(title);
    return el;
  }

  /**
   * The invisible element that catches the pointer for one drawn element.
   *
   * It exists because a 2.5 px line and a 3.5 px symbol are not targets.  Identity
   * travels on it as three data attributes — ``(kind, index)`` is what the code uses,
   * the name is carried along as *data* only and never reaches a selector.
   */
  function hitElement(tag, attrs, kind, index, name) {
    const el = svgNode(tag, attrs, 'nm-hit');
    el.setAttribute('data-nm-kind', kind);
    el.setAttribute('data-nm-i', String(index));
    el.setAttribute('data-nm-name', String(name));
    return titled(el, String(name));
  }

  /** Build the SVG of *data*; the returned object carries the three indexes of its elements. */
  function buildSvg(data) {
    // role="group" and deliberately not role="application": the latter turns off the
    // browse mode of a screen reader over the whole drawing, which costs the reader
    // every key they navigate with and gives back four this module handles.  One
    // tabindex, on the container: the drawn elements take no focus of their own, the
    // arrows move the view instead of walking a list, and the complete, fully
    // accessible view of the same data is the table next door.
    const svg = svgNode('svg', {
      preserveAspectRatio: 'xMidYMid meet',
      role: 'group',
      tabindex: '0',
      'aria-label': 'Network map',
      'aria-describedby': 'nm-help'
    }, 'nm-svg');

    const polys = svgNode('g', null, 'nm-polys');
    // Between the outlines and the pipes: the line says where the runoff of a
    // subcatchment goes and is not part of the network, so it runs under every pipe it
    // crosses and never over one.
    const outlets = svgNode('g', null, 'nm-outlets');
    const links = svgNode('g', null, 'nm-links');
    const nodes = svgNode('g', null, 'nm-nodes');
    // The names go last, over everything they name, and out of the accessibility tree:
    // every drawn element already carries its name in a <title>, so a screen reader that
    // read this group as well would hear the whole network a second time.
    const labels = svgNode('g', { 'aria-hidden': 'true' }, 'nm-labels');
    svg.appendChild(polys);
    svg.appendChild(outlets);
    svg.appendChild(links);
    svg.appendChild(nodes);
    svg.appendChild(labels);

    // Index of the drawn element of each kind, by its position in the payload.  Every
    // later lookup goes through these lists: a selector built from an element name
    // would be both an injection and a DOMException waiting for a legal SWMM name.
    const els = {link: [], node: [], poly: []};
    // ...and its hit element, under the same index.  Dimming has to reach both
    // siblings at once: opacity alone still catches the pointer, so the class that
    // fades a category has to travel to the target as well as to the drawing.  A
    // subcatchment is its own target, so both lists hold the same element there.
    const hitEls = {link: [], node: [], poly: []};
    // ...and the outlet line of a subcatchment, under the index of *its polygon*.  Only
    // one kind has such a line, so the object carries one key; setDim reaches it through
    // the same hasOwnProperty guard as the other two indexes.
    const outletEls = {poly: []};
    // ...and the <text> that names it, under the same index, in the same three
    // collections.  The lists stay empty until ensureLabels() fills them: the names are
    // built the first time the reader asks for them and never before.
    const labelEls = {link: [], node: [], poly: []};

    (Array.isArray(data.polys) ? data.polys : []).forEach(poly => {
      const d = pointsToPath(poly.p);
      if (!d) return;
      // The outline is its own target: it is a filled shape, so the stylesheet lets it
      // catch the pointer inside its area and no second element is needed.
      const path = svgNode('path', { d: `${d} Z`, 'data-i': String(poly.i) }, 'nm-poly');
      path.setAttribute('data-nm-kind', 'poly');
      path.setAttribute('data-nm-i', String(poly.i));
      path.setAttribute('data-nm-name', String(poly.n));
      titled(path, String(poly.n));
      polys.appendChild(path);
      els.poly[poly.i] = path;
      hitEls.poly[poly.i] = path;

      // The line to the outlet: no <title>, no data-nm-* and no nm-hit, so it is not a
      // target of the pointer and the click goes through it to whatever lies below.
      // It is deliberately left out of markUnanalysed: that function only adds the
      // class nm-na, and every .nm-na rule of the stylesheet names a kind
      // (.nm-node.nm-na, .nm-link.nm-na, .nm-poly.nm-na), so the class on an
      // .nm-outlet would paint nothing at all - dead code frozen by a test.
      // The index is SPARSE on purpose: outletEls.poly[poly.i], never push.  (kind,
      // index) is the identity the whole module looks an element up by; a dense array
      // would shift every line by one as soon as a single subcatchment had no outlet,
      // and the table filter would then fade the line of a *different* subcatchment.
      const outletD = outletPath(poly);
      if (outletD) {
        const line = svgNode('path', { d: outletD, 'data-i': String(poly.i) }, 'nm-outlet');
        outlets.appendChild(line);
        outletEls.poly[poly.i] = line;
      }
    });

    (Array.isArray(data.links) ? data.links : []).forEach(link => {
      const d = pointsToPath(link.p);
      if (!d) return;
      const path = svgNode('path', { d: d, 'data-i': String(link.i), 'data-kind': String(link.kind) }, 'nm-link');
      links.appendChild(path);
      // The same geometry once more, under a transparent stroke twelve screen pixels
      // wide.  non-scaling-stroke keeps that width through every zoom, so the target
      // does not thin out as the reader zooms out.
      const linkHit = hitElement('path', {
        d: d,
        fill: 'none',
        stroke: 'transparent',
        'stroke-width': String(HIT_STROKE_PX),
        'vector-effect': 'non-scaling-stroke'
      }, 'link', link.i, link.n);
      links.appendChild(linkHit);
      els.link[link.i] = path;
      hitEls.link[link.i] = linkHit;
    });

    // Identity of a drawn element is (kind, index); names never enter a selector or a
    // markup string, so a node called ``a"b`` cannot break either.
    (Array.isArray(data.nodes) ? data.nodes : []).forEach(node => {
      const point = node.p;
      if (!Array.isArray(point) || !Number.isFinite(point[0]) || !Number.isFinite(point[1])) return;
      const group = svgNode('g', { 'data-i': String(node.i), 'data-k': String(node.k) }, 'nm-node');
      const symbol = symbolFor(node.k);
      group.appendChild(svgNode(symbol.tag, symbol.attrs));
      nodes.appendChild(group);
      // A sibling of the symbol group, not a child of it: the group is scaled by
      // r / pxPerUnit and a circle inside it would be scaled twice.  The radius is set
      // in rescale(); the target is never hidden, not even where the symbols are.
      const hit = hitElement('circle', {
        cx: String(point[0]),
        cy: String(point[1]),
        r: '0',
        fill: 'transparent'
      }, 'node', node.i, node.n);
      nodes.appendChild(hit);
      els.node[node.i] = group;
      hitEls.node[node.i] = hit;
    });

    return { svg: svg, els: els, hitEls: hitEls, outletEls: outletEls, labelEls: labelEls, labelGroup: labels };
  }

  function initNetworkMap() {
    const root = document.getElementById('storm-results-v2');
    if (!root) return;
    const host = root.querySelector('#nm-canvas');
    const script = document.getElementById('network-map-data');
    if (!host || !script) return;

    let envelope;
    try {
      envelope = JSON.parse(script.textContent || '{}');
    } catch (err) {
      console.error('network-map: failed to parse #network-map-data:', err);
      return;
    }
    if (!envelope || envelope.available !== true || !envelope.data) return;
    const data = envelope.data;

    // D3 - a drawing whose positions were computed from the topology is not the
    // geometry of the model, and the panel has to say so.  The mark is built here,
    // before the first draw and regardless of whether the panel is visible yet, so
    // that it is permanent: nothing removes it, nothing hides it after a while and it
    // is not aria-hidden.  aria-describedby makes a screen reader read both lines as
    // part of the panel's description.  In the "inp" layout the branch is not taken
    // and neither element exists in the DOM at all.
    const panel = host.parentNode;
    if (data.layout === 'auto') {
      const badge = document.createElement('p');
      badge.id = 'nm-schematic-mark';
      badge.className = 'nm-badge';
      badge.textContent = 'Schematic layout — not model geometry';
      const note = document.createElement('p');
      note.id = 'nm-schematic-scale';
      note.className = 'nm-legend-note';
      note.textContent = 'Positions are computed from network topology; distances and directions are not to scale.';
      if (panel) {
        panel.insertBefore(badge, host);
        panel.insertBefore(note, host.nextSibling);
        panel.setAttribute('aria-describedby', 'nm-schematic-mark nm-schematic-scale');
      }
    }

    // The map is one Tab stop and answers four keys, so the page has to say which -
    // and say it in the document, where a reader sees it and where aria-describedby of
    // the <svg> can point at it.  Built here and not in ensureDrawn: the drawing is
    // lazy, an aria-describedby pointing at an id that does not exist yet describes
    // nothing, and the sentence is true before the first pixel is drawn.
    const help = document.createElement('p');
    help.id = 'nm-help';
    help.className = 'nm-help';
    // One short line, and short on purpose: on a 390 px phone every pixel this sentence
    // takes comes out of the drawing, which is the one thing the panel is for - and a
    // reader on a phone has no keyboard to use it with.  The full, fully accessible view
    // of the same elements is the table next to this tab, which needs no explaining.
    help.textContent = 'Arrow keys pan, + and \u2212 zoom, Home fits, Esc closes.';
    if (panel) panel.appendChild(help);

    // The elements no drawing can show, counted at the root of the payload.  Nothing is
    // built when there are none: a line reading "0 elements not drawn" would stand under
    // every healthy model, which is nearly all of them.
    const skippedText = skippedLabel(data.skipped);
    if (skippedText && panel) {
      const skippedLine = document.createElement('p');
      skippedLine.id = 'nm-skipped';
      skippedLine.className = 'nm-skipped';
      skippedLine.textContent = skippedText;
      panel.appendChild(skippedLine);
    }

    let drawing = null;
    let viewBox = null;
    let fitted = null;     // the box that shows everything; the zoom limits rest on it
    let selection = null;  // {kind, index, name, el} — lives here and nowhere else
    let card = null;
    // The gesture that drags the card by its heading: {pointerId, handle, x0, y0,
    // startLeft, startTop} while the heading is held, null between gestures.  It needs
    // no "captured" flag: the capture is taken in pointerdown, before the record
    // exists, so having a record *means* owning the pointer.
    let cardDrag = null;
    let gesture = null;
    let hintTimer = null;

    // Labels (position 4).  All of it closure state: the <text> elements are built once
    // and lazily, the layer that measures them with them, and the deferred placement
    // reads the view of the frame that scheduled it instead of computing it again.
    let labelsBuilt = false;
    let labelFrame = null;
    let lastView = null;
    let labelCanvas = null;
    let labelCtx = null;
    let labelFonts = null;
    let labelButton = null;
    const labelMetrics = new Map();
    const labelAnchors = { link: [], node: [], poly: [] };

    /**
     * The one writer of the preferences (N02).
     *
     * Every gesture worth remembering calls this and never ``writeMapPrefs`` directly:
     * with a single write there is nowhere to lose a field.  The failure it closes is
     * real - the mode switch used to write the mode alone, so turning the names on and
     * then switching the colouring forgot the names, and the same write dropped the
     * corner the card had been dragged to.  It is called from ``setMode``, from the
     * names toggle and from both ends of the card gesture, and **never** from
     * ``pointermove`` - that one runs dozens of times a second and
     * ``localStorage.setItem`` is synchronous.
     */
    function savePrefs() {
      writeMapPrefs(mapStorage(), { mode: mode, card: cardPos, labels: labelsOn });
    }

    // Colouring (stage 6).  ``mode`` is the only thing that is remembered between
    // visits; the dimmed categories are a reading gesture, not a setting, and start
    // empty every time the page opens.
    const prefs = readMapPrefs(mapStorage());
    let mode = prefs.mode;
    // ...and the corner the reader dragged the card to, which is remembered with it
    // (D9): the card comes back where it was left, in this visit and the next one.
    // It is deliberately not cleared and not rewritten when the viewport changes -
    // applyCardPos passes it through clampCardPos on its way to the *style*, so a
    // position saved in a larger window lands against the edge of a smaller canvas
    // instead of outside it, and comes back where it was when the window grows again.
    // The only writers are the drag itself and settleCardPos at the end of it.
    let cardPos = prefs.card;
    // The one read feeds both fields.  Without this line the map has no source for "the
    // names were on when I left": the button would start off and the stored true would
    // reach nothing at all.
    let labelsOn = prefs.labels;
    let dimmed = [];
    const linkClass = [];  // index of a drawn link -> the colour class it carries now

    // Filtering (stage 7).  The second, independent reason an element may be faded:
    // the table it belongs to is showing a filtered set and everything outside that
    // set steps back.  One set per collection, because one filtered table speaks for
    // its own collection only; ``null`` means "this collection was never filtered".
    const filterDim = { link: null, node: null, poly: null };

    // The map reads the Results V2 payload for itself.  The two scripts share no
    // state and no globals: the selection card needs a couple of values from the row
    // of the clicked element, and a missing or broken payload only costs those values.
    let results = null;
    const resultsScript = document.getElementById('results-v2-data');
    if (resultsScript) {
      try {
        results = JSON.parse(resultsScript.textContent || '{}');
      } catch (err) {
        console.error('network-map: failed to parse #results-v2-data:', err);
      }
    }

    /**
     * Rows of one collection by name, built once and kept.
     *
     * The selection card can afford the linear scan of ``findResult`` - it runs on one
     * click - but colouring walks every link on every change of mode, and a scan per
     * link turns that into a quadratic pass over the whole model.
     */
    const rowIndexes = {};
    function rowsByName(kind) {
      if (Object.prototype.hasOwnProperty.call(rowIndexes, kind)) return rowIndexes[kind];
      const index = new Map();
      const tab = kindToTab(kind);
      const rows = tab && results && typeof results === 'object' && Array.isArray(results[tab]) ? results[tab] : [];
      rows.forEach(row => {
        if (row && typeof row.name === 'string' && !index.has(row.name)) index.set(row.name, row);
      });
      rowIndexes[kind] = index;
      return index;
    }

    function rowFor(kind, name) {
      const row = rowsByName(kind).get(name);
      return row === undefined ? null : row;
    }

    /**
     * Did the pipeline analyse this drawn element?  The one test, asked in one place.
     *
     * It is the presence of a row in the Results V2 collection of that kind and
     * nothing else — never ``k``, never ``kind`` (see NOT_ANALYSED_CLASS).  A payload
     * that carries no such collection at all is not evidence that nothing was
     * analysed, so everything of that kind counts as analysed there: a broken Results
     * V2 payload must not block the drawing.
     *
     * Both readers go through this function, which is what makes the two states
     * exclusive by construction: an element is marked "not analysed" exactly when it
     * is invisible to the filter dimming, so nothing can ever be both.
     */
    function isAnalysed(kind, name) {
      const tab = kindToTab(kind);
      const known = tab && results && typeof results === 'object'
        && Object.prototype.hasOwnProperty.call(results, tab) && Array.isArray(results[tab]);
      if (!known) return true;
      return rowFor(kind, name) !== null;
    }

    function ensureDrawn() {
      if (drawing) return;
      drawing = buildSvg(data);
      host.appendChild(drawing.svg);
      // The drawing's keys live on the SVG, the one focusable element of the drawing;
      // the drawn elements take no focus, so a key is never delivered to one of them.
      // Escape alone is handled on the document (see mapKeyAction).
      drawing.svg.addEventListener('keydown', onMapKey);
      markFlooded();
      markUnanalysed();
      paint();
    }

    /**
     * The alarm fill of a flooded node.
     *
     * The flag is read from the results row and nothing is derived here: the pipeline
     * already owns one definition of flooding (``node_is_flooded``), SWMM reports
     * ``Hours Flooded`` to two decimals, so a short overflow reads 0.00 h with a
     * non-zero volume, and a third definition living in the map would disagree with
     * the table on exactly those nodes.  Drawn once: flooding does not depend on the
     * colouring mode, and the node symbols stay constant across all three.
     */
    function markFlooded() {
      (Array.isArray(data.nodes) ? data.nodes : []).forEach(node => {
        const el = drawing.els.node[node.i];
        if (!el) return;
        const row = rowFor('node', String(node.n));
        if (row && row.is_flooded === true) el.classList.add('nm-flooded');
      });
    }

    /**
     * Grey out and block every drawn element ``isAnalysed`` says nothing was computed
     * for - the absence of a row in its Results V2 collection and nothing else.
     *
     * The class goes on the drawing *and* on its hit element: the first is what the
     * reader sees, the second is what carries the cursor and what the click handler
     * reads.  Pointer events stay on - ``pointer-events: none`` would take the tooltip
     * away with the click, and the tooltip is how the reader learns why nothing opens.
     * The filter dimming, which *does* take pointer events away, never reaches these
     * elements: it is fed the analysed ones only.
     */
    function markUnanalysed() {
      COLLECTIONS.forEach(pair => {
        const kind = pair[0];
        (Array.isArray(data[pair[1]]) ? data[pair[1]] : []).forEach(element => {
          if (isAnalysed(kind, String(element.n))) return;
          // Two lists, deliberately not three: drawing.outletEls is left out (D30).  This
          // function only adds the class nm-na, and every .nm-na rule of the stylesheet
          // names a kind (.nm-node.nm-na, .nm-link.nm-na, .nm-poly.nm-na), so the class on
          // an .nm-outlet would paint nothing at all - dead code frozen by a test.
          [drawing.els, drawing.hitEls].forEach(group => {
            const list = Object.prototype.hasOwnProperty.call(group, kind) ? group[kind] : null;
            const el = list ? list[element.i] : null;
            if (el) el.classList.add(NOT_ANALYSED_CLASS);
          });
        });
      });
    }

    /** A pair of finite drawing-unit coordinates, or null. */
    function finitePoint(point) {
      if (!Array.isArray(point) || !Number.isFinite(point[0]) || !Number.isFinite(point[1])) return null;
      return [point[0], point[1]];
    }

    /**
     * The middle of the longest segment of a polyline, or null.
     *
     * Not the middle of the bounding box: an elbowed conduit has its box centre off the
     * pipe entirely, and a name anchored there would be named after a pipe it does not
     * touch.  The longest segment is also the one with the most room along it.
     */
    function longestSegmentMid(points) {
      const list = Array.isArray(points) ? points : [];
      let best = null;
      let bestLen = -1;
      for (let n = 1; n < list.length; n += 1) {
        const a = finitePoint(list[n - 1]);
        const b = finitePoint(list[n]);
        if (!a || !b) continue;
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        if (len <= bestLen) continue;
        bestLen = len;
        best = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      }
      return best;
    }

    /**
     * The box one name takes on screen, measured once per (font, text) and kept.
     *
     * The key carries the font as well as the text (N01): the collections have separate
     * name spaces, so the same SWMM name is regularly a node *and* a subcatchment, and
     * the italic of a subcatchment is a different set of widths.  The box is the **ink**
     * box plus the halo on every side - ``measureText().width`` is an advance, and the
     * glyphs of a 'j' or of any italic reach outside it on both sides, which is exactly
     * the overhang the halo then strokes.
     */
    function labelBox(variant, text) {
      const font = labelFonts[variant];
      const key = font + '\u0000' + text;
      const cached = labelMetrics.get(key);
      if (cached) return cached;
      labelCtx.font = font;
      const m = labelCtx.measureText(text);
      // [Z]6 - a browser whose TextMetrics has no ink box gets an explicitly approximate
      // one.  It is an assumption and not a contract: for an italic it may well miss the
      // overhang on the left.
      const left = Number.isFinite(m.actualBoundingBoxLeft) ? m.actualBoundingBoxLeft : 0;
      const right = Number.isFinite(m.actualBoundingBoxRight) ? m.actualBoundingBoxRight : m.width;
      const ascent = Number.isFinite(m.actualBoundingBoxAscent) ? m.actualBoundingBoxAscent : 0.8 * LABEL_FONT_PX;
      const descent = Number.isFinite(m.actualBoundingBoxDescent) ? m.actualBoundingBoxDescent : 0.25 * LABEL_FONT_PX;
      const box = {
        left: left,
        ascent: ascent,
        w: left + right + 2 * LABEL_HALO_PX,
        h: ascent + descent + 2 * LABEL_HALO_PX
      };
      labelMetrics.set(key, box);
      return box;
    }

    /** Every built <text>, with the kind and the index it was built for. */
    function eachLabel(fn) {
      if (!drawing) return;
      COLLECTIONS.forEach(pair => {
        const kind = pair[0];
        const list = drawing.labelEls[kind];
        for (let n = 0; n < list.length; n += 1) {
          if (list[n]) fn(list[n], kind, n);
        }
      });
    }

    /** The names of the elements nothing was computed for, greyed like the elements. */
    function markUnanalysedLabels() {
      COLLECTIONS.forEach(pair => {
        const kind = pair[0];
        (Array.isArray(data[pair[1]]) ? data[pair[1]] : []).forEach(element => {
          if (isAnalysed(kind, String(element.n))) return;
          const el = drawing.labelEls[kind][element.i];
          if (el) el.classList.add(NOT_ANALYSED_CLASS);
        });
      });
    }

    /**
     * Build the <text> of every element, once, together with the layer that measures
     * them.  Idempotent, and guarded by ``labelsBuilt`` rather than by the length of a
     * list: on a payload with no nodes the length never stops being zero and this would
     * run on every wheel click and every step of a pan, which is the hot path.
     *
     * Lazy on purpose: at the cap of 6000 elements building the names in ensureDrawn
     * would be thousands of DOM nodes nobody asked for, the names being off by default.
     *
     * This is the only place in the module that measures text, and the only one allowed
     * to call getComputedStyle.  The font is read **per variant** - the upright
     * .nm-label and the italic .nm-label-poly have different widths and different
     * overhangs, and the map shows both collections at once - from a probe that has been
     * inserted into the live group first, because before insertion the rule does not
     * reach it.  The ``.font`` shorthand is not used: some browsers hand back an empty
     * string for it, so the four longhands are joined by hand, in that order.
     *
     * A browser with no 2D context leaves the map without names and with everything
     * else: ``labelsBuilt`` stays false, the group stays hidden and the deferred
     * placement walks out.  Nothing falls back on ``text.length * a constant`` - that
     * arithmetic is precisely what the measurement replaced.
     */
    function ensureLabels() {
      if (labelsBuilt || !drawing) return;
      if (!labelCanvas) {
        labelCanvas = document.createElement('canvas');
        labelCtx = typeof labelCanvas.getContext === 'function' ? labelCanvas.getContext('2d') : null;
      }
      if (!labelCtx) return;

      const probes = { node: svgNode('text', null, 'nm-label'), poly: svgNode('text', null, 'nm-label nm-label-poly') };
      drawing.labelGroup.appendChild(probes.node);
      drawing.labelGroup.appendChild(probes.poly);
      labelFonts = {};
      Object.keys(probes).forEach(variant => {
        const style = window.getComputedStyle(probes[variant]);
        labelFonts[variant] = [style.fontStyle, style.fontWeight, style.fontSize, style.fontFamily].join(' ');
      });
      drawing.labelGroup.removeChild(probes.node);
      drawing.labelGroup.removeChild(probes.poly);

      COLLECTIONS.forEach(pair => {
        const kind = pair[0];
        (Array.isArray(data[pair[1]]) ? data[pair[1]] : []).forEach(element => {
          // A node stands on its own point; a subcatchment on ``element.c``, the centroid
          // E3a put in the payload (D22) - never on one computed here, which would be a
          // second definition of the same point.  A payload written before E3a carries no
          // ``c``, and such an outline gets no <text> at all.
          let anchor = null;
          if (kind === 'node') anchor = finitePoint(element.p);
          else if (kind === 'poly') anchor = finitePoint(element.c);
          else anchor = longestSegmentMid(element.p);
          if (!anchor) return;
          const text = svgNode('text', null, kind === 'poly' ? 'nm-label nm-label-poly' : 'nm-label');
          // The name as data and never as markup: textContent, so a SWMM name holding a
          // '<' or a quote is a name and not a tag.
          text.textContent = String(element.n);
          text.style.display = 'none';
          drawing.labelGroup.appendChild(text);
          drawing.labelEls[kind][element.i] = text;
          labelAnchors[kind][element.i] = anchor;
        });
      });

      markUnanalysedLabels();
      // markUnanalysed() and applyDim() ran in ensureDrawn, before any of these elements
      // existed, so the names catch up here - over the label lists alone.  applyDim()
      // would walk the whole double loop of COLLECTIONS and end in updateLegendState(),
      // that is: enter the legend machinery from the label path.
      COLLECTIONS.forEach(pair => {
        const kind = pair[0];
        const hidden = filterDim[kind];
        (Array.isArray(data[pair[1]]) ? data[pair[1]] : []).forEach(element => {
          const byLegend = kind === 'link' && dimmed.indexOf(linkClass[element.i]) !== -1;
          const byFilter = hidden !== null && hidden.has(element.i);
          const el = drawing.labelEls[kind][element.i];
          if (el && (byLegend || byFilter)) el.classList.add('nm-dim');
        });
      });
      labelsBuilt = true;
    }

    /** Repaint every link in the current mode and re-apply the dimming on top. */
    function paint() {
      if (!drawing) return;
      (Array.isArray(data.links) ? data.links : []).forEach(link => {
        const el = drawing.els.link[link.i];
        if (!el) return;
        const previous = linkClass[link.i];
        if (previous) el.classList.remove(previous);
        const next = pickColor(mode, rowFor('link', String(link.n)));
        el.classList.add(next);
        linkClass[link.i] = next;
      });
      applyDim();
    }

    /**
     * Fade or restore one drawn element *and its hit element*.
     *
     * Always the pair: the drawing is what the reader sees, the target is what the
     * browser hits, and fading only the first leaves a faded pipe swallowing clicks
     * meant for what lies underneath it.  The outlet line of a subcatchment is faded
     * together with its polygon and has no state of its own: it is a property of the
     * outline, not a third element the reader can reach.
     */
    function setDim(kind, index, on) {
      if (!drawing) return;
      // ``drawing.outletEls`` (E3b) stands beside labelEls: an outlet line fades with the
      // subcatchment it belongs to, exactly as the name of it does.
      const lists = [drawing.els, drawing.hitEls, drawing.outletEls, drawing.labelEls];
      lists.forEach(group => {
        const list = Object.prototype.hasOwnProperty.call(group, kind) ? group[kind] : null;
        const el = list ? list[index] : null;
        if (!el) return;
        if (on) el.classList.add('nm-dim');
        else el.classList.remove('nm-dim');
      });
    }

    /**
     * Re-apply both reasons an element may be faded, over the whole drawing.
     *
     * A legend entry switched off (stage 6) and a table filter (stage 7) are separate
     * gestures that write the same class, so they are resolved together in one place:
     * an element fades when *either* says so and comes back only when both let go.
     * Two independent writers of .nm-dim would take turns undoing each other — the
     * first repaint after a filter would restore every link the legend had faded.
     */
    function applyDim() {
      if (!drawing) return;
      COLLECTIONS.forEach(pair => {
        const kind = pair[0];
        const hidden = filterDim[kind];
        (Array.isArray(data[pair[1]]) ? data[pair[1]] : []).forEach(element => {
          const byLegend = kind === 'link' && dimmed.indexOf(linkClass[element.i]) !== -1;
          const byFilter = hidden !== null && hidden.has(element.i);
          setDim(kind, element.i, byLegend || byFilter);
        });
      });
      updateLegendState();
    }

    /**
     * Fit the drawing to the container and re-size the node symbols.
     *
     * Called on every change of the viewBox, including the one that a hidden panel
     * becoming visible amounts to.  While the panel is hidden the container measures
     * 0x0, so the function returns before any division.
     */
    function rescale() {
      if (!drawing) return;
      const rect = host.getBoundingClientRect();
      if (!isPositive(rect.width) || !isPositive(rect.height)) return;

      if (!viewBox) {
        viewBox = fitViewBox(data.bbox, PAD_MIN_PX, { w: rect.width, h: rect.height });
        fitted = viewBox;
      }
      drawing.svg.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);

      const pxPerUnit = Math.min(rect.width / viewBox.w, rect.height / viewBox.h);
      if (!isPositive(pxPerUnit)) return;

      // The targets are sized in screen pixels for the same reason the symbols are:
      // a target in drawing units is unreachable zoomed out and covers half the model
      // zoomed in.  Eight pixels of radius is a finger, not a mouse point.
      const hitRadius = HIT_RADIUS_PX / pxPerUnit;

      const radiusPx = symbolRadiusPx(data.min_gap, pxPerUnit);
      const scale = radiusPx / pxPerUnit;
      const gap = data.min_gap;
      // Below the floor the symbols would merge into a blot; no symbol reads better
      // than a blot, and zooming in brings them back.
      const crowded = Number.isFinite(gap) && gap > 0 && (gap * pxPerUnit) / DENSITY_DIVISOR < MIN_SYMBOL_PX;
      // The payload is walked, exactly as ``markFlooded`` walks it, instead of a second
      // list of handles kept beside ``els``: the two lists are indexed differently — one
      // by ``node.i``, the other by the order the nodes were drawn in — and a node that
      // ``buildSvg`` skipped for a non-finite point simply has no entry here, which is
      // what the guard reads.
      (Array.isArray(data.nodes) ? data.nodes : []).forEach(node => {
        const el = drawing.els.node[node.i];
        const hit = drawing.hitEls.node[node.i];
        if (!el || !hit) return;
        // The targets are sized here and not with the symbols: both are screen-sized,
        // and one walk of the payload does the pair.
        hit.setAttribute('r', String(hitRadius));
        el.setAttribute('transform', `translate(${node.p[0]} ${node.p[1]}) scale(${scale})`);
        el.style.display = crowded ? 'none' : '';
      });

      // The frame, written once, for the deferred placement to read.  Computing
      // pxPerUnit, radiusPx or crowded a second time there would be two truths about one
      // frame, and the labels would be laid out against a view nobody is looking at.
      // The rectangle itself is **not** cached with them (F1): ``left``/``top`` of
      // getBoundingClientRect are the position of the container in the viewport, which
      // is not a property of the frame at all - a page that scrolls between this call
      // and the deferred placement changes them without changing anything the three
      // numbers above describe, and no listener here would see it (the module observes
      // size, never position).  placeLabelsNow reads the rectangle fresh instead.
      lastView = { pxPerUnit: pxPerUnit, radiusPx: radiusPx, crowded: crowded };

      // The preference came back as true and nobody pressed the button, so this is the
      // only place that can build the names at all.
      if (labelsOn && !labelsBuilt) ensureLabels();

      // Synchronous, with the viewBox and with the symbols: a transform written in the
      // deferred call would leave every name in the scale of the previous frame, and the
      // names would visibly drag behind the symbols they name.
      const labelScale = 1 / pxPerUnit;
      eachLabel((el, kind, index) => {
        const anchor = labelAnchors[kind][index];
        if (!anchor) return;
        el.setAttribute('transform', `translate(${anchor[0]} ${anchor[1]}) scale(${labelScale})`);
      });

      // Three ways out, all of them hiding the whole group: the names are off, the
      // symbols are too dense to draw at all (D25), or two node centres are closer than
      // LABEL_MIN_GAP_PX on screen (D24).  A null min_gap - the auto layout - hides
      // nothing: the culling and LABEL_MAX bound the work there.  The "zoom in" hint
      // belongs to toggleLabels and never here: this runs on every step of a pan.
      const gapPx = Number.isFinite(gap) ? gap * pxPerUnit : null;
      const hide = !labelsOn || !labelsBuilt || crowded || (gapPx !== null && gapPx < LABEL_MIN_GAP_PX);
      drawing.labelGroup.style.display = hide ? 'none' : '';
      if (hide) return;
      scheduleLabels();
    }

    /**
     * The one place in the module that asks for a frame (R07), and the only
     * requestAnimationFrame in the file.
     *
     * Four callers: rescale(), toggleLabels() and the two ends of a selection.  Neither
     * selectElement nor clearSelection calls rescale(), so without them a name that lost
     * a collision would not get the priority D27 promises it until the next pan, zoom or
     * resize; and .nm-sel changes the geometry of the obstacles themselves - a conduit
     * goes from stroke-width 2.5 to 4, a node from 1 to 3 - so the layout computed for
     * the previous selection is out of date, which no rule inside the solver can repair.
     */
    function scheduleLabels() {
      if (labelFrame) return;
      labelFrame = requestAnimationFrame(() => {
        labelFrame = null;
        placeLabelsNow();
      });
    }

    /** Hide every name of one collection; the placement reveals the ones it placed. */
    function hideLabels(list) {
      for (let n = 0; n < list.length; n += 1) {
        if (list[n]) list[n].style.display = 'none';
      }
    }

    /**
     * Lay the names out over the scale ``rescale`` last measured and the rectangle the
     * container occupies **now**, and write their positions.
     *
     * Every name goes down first.  A <text> is built once and lazily, the solver simply
     * omits an element it found no room for, and the culling drops everything outside the
     * visible rectangle - so a name left where the previous frame put it would sit on a
     * symbol or along a pipe as soon as the view moved, which is the failure the whole of
     * position 4 exists to prevent.
     *
     * Two spaces and two jobs, deliberately not merged: **which elements are considered**
     * is decided here, in drawing units, through the real CTM (clientToUnits), because
     * with preserveAspectRatio="xMidYMid meet" the content outside the viewBox still
     * renders in the side bands and culling by the viewBox would drop names the reader
     * can see; **how far an obstacle segment is followed** is decided by the solver, in
     * screen pixels, through opts.clip.
     *
     * The obstacles are built from the shape of the symbol and from the *current*
     * selection, and both come from data - ``selection`` and the results row - never from
     * the DOM.  ``radiusPx`` is the scale of a symbol and not the radius of a circle
     * around it: the corner of the square of a storage lies sqrt(2) further than its
     * side, so a rectangle that cleared the old disk could still lie on the outline
     * (N09).  A conduit's half width is computed once and handed to its segment *and* to
     * its own item as ``r``, so the candidates start off the centre line of its own pipe.
     * Subcatchment outlines are obstacles for nobody (D22/D31).
     */
    function placeLabelsNow() {
      if (!drawing || !labelsBuilt || !labelsOn || !lastView) return;
      hideLabels(drawing.labelEls.node);
      hideLabels(drawing.labelEls.link);
      hideLabels(drawing.labelEls.poly);
      if (drawing.labelGroup.style.display === 'none') return;

      const pxPerUnit = lastView.pxPerUnit;
      const radiusPx = lastView.radiusPx;
      // One layout read per frame, and it has to be this one: the corners below are
      // mapped through the *live* CTM, so a rectangle snapshotted in rescale() would mix
      // a stale page position with a current transform and displace the whole culling
      // frame by however far the page scrolled in between (F1).
      const rect = host.getBoundingClientRect();
      if (!isPositive(pxPerUnit) || !isPositive(rect.width) || !isPositive(rect.height)) return;

      const corner1 = clientToUnits(rect.left, rect.top);
      const corner2 = clientToUnits(rect.right, rect.bottom);
      let frame = null;
      if (corner1 && corner2) {
        frame = {
          x1: Math.min(corner1.x, corner2.x),
          y1: Math.min(corner1.y, corner2.y),
          x2: Math.max(corner1.x, corner2.x),
          y2: Math.max(corner1.y, corner2.y)
        };
      } else if (viewBox) {
        // No CTM means a hidden panel, in which nothing is visible anyway.
        frame = { x1: viewBox.x, y1: viewBox.y, x2: viewBox.x + viewBox.w, y2: viewBox.y + viewBox.h };
      }
      if (!frame) return;
      const spanX = frame.x2 - frame.x1;
      const spanY = frame.y2 - frame.y1;
      if (!isPositive(spanX) || !isPositive(spanY)) return;
      const scaleX = rect.width / spanX;
      const scaleY = rect.height / spanY;

      const selectedKey = selection ? selection.kind + ':' + selection.index : null;
      const items = [];
      const segments = [];
      const byKey = new Map();
      let reach = 0;

      (Array.isArray(data.nodes) ? data.nodes : []).forEach(node => {
        const anchor = labelAnchors.node[node.i];
        const el = drawing.labelEls.node[node.i];
        if (!anchor || !el) return;
        const x = (anchor[0] - frame.x1) * scaleX;
        const y = (anchor[1] - frame.y1) * scaleY;
        const box = labelBox('node', String(node.n));
        if (x < -box.w || x > rect.width + box.w || y < -box.h || y > rect.height + box.h) return;
        // k = 1 for the circle of a junction, sqrt(2) for every other kind: the square of
        // a storage is the widest of them, and the triangle of an outfall (1.281) and the
        // diamond of "other" (1.0) fit inside the same conservative allowance.  The stroke
        // is 3 px when selected, 2 px when flooded, 1 px otherwise - .nm-sel beats
        // .nm-flooded by source order at equal specificity, so it is asked first.
        const k = node.k === 'junction' ? 1 : Math.SQRT2;
        const row = rowFor('node', String(node.n));
        const strokePx = selectedKey === 'node:' + node.i ? 3 : (row && row.is_flooded === true ? 2 : 1);
        const r = k * radiusPx + strokePx + LABEL_MARGIN_PX;
        if (box.w > reach) reach = box.w;
        if (box.h > reach) reach = box.h;
        items.push({ key: 'node:' + node.i, kind: 'node', i: node.i, x: x, y: y, r: r, w: box.w, h: box.h });
        byKey.set('node:' + node.i, { el: el, box: box });
      });

      (Array.isArray(data.links) ? data.links : []).forEach(link => {
        // One number, two structures: the segment of the pipe and the r of its own item.
        // Two expressions would part company at the first change of a stroke width.
        const hw = (selectedKey === 'link:' + link.i ? 4 : 2.5) / 2 + LABEL_MARGIN_PX;
        const points = Array.isArray(link.p) ? link.p : [];
        for (let n = 1; n < points.length; n += 1) {
          const a = finitePoint(points[n - 1]);
          const b = finitePoint(points[n]);
          if (!a || !b) continue;
          segments.push({
            key: 'link:' + link.i,
            x1: (a[0] - frame.x1) * scaleX,
            y1: (a[1] - frame.y1) * scaleY,
            x2: (b[0] - frame.x1) * scaleX,
            y2: (b[1] - frame.y1) * scaleY,
            hw: hw
          });
        }
        const anchor = labelAnchors.link[link.i];
        const el = drawing.labelEls.link[link.i];
        if (!anchor || !el) return;
        const x = (anchor[0] - frame.x1) * scaleX;
        const y = (anchor[1] - frame.y1) * scaleY;
        const box = labelBox('node', String(link.n));
        if (x < -box.w || x > rect.width + box.w || y < -box.h || y > rect.height + box.h) return;
        if (box.w > reach) reach = box.w;
        if (box.h > reach) reach = box.h;
        items.push({ key: 'link:' + link.i, kind: 'link', i: link.i, x: x, y: y, r: hw, w: box.w, h: box.h });
        byKey.set('link:' + link.i, { el: el, box: box });
      });

      (Array.isArray(data.polys) ? data.polys : []).forEach(poly => {
        const anchor = labelAnchors.poly[poly.i];
        const el = drawing.labelEls.poly[poly.i];
        if (!anchor || !el) return;
        const x = (anchor[0] - frame.x1) * scaleX;
        const y = (anchor[1] - frame.y1) * scaleY;
        const box = labelBox('poly', String(poly.n));
        if (x < -box.w || x > rect.width + box.w || y < -box.h || y > rect.height + box.h) return;
        if (box.w > reach) reach = box.w;
        if (box.h > reach) reach = box.h;
        items.push({ key: 'poly:' + poly.i, kind: 'poly', i: poly.i, x: x, y: y, r: 0, w: box.w, h: box.h });
        byKey.set('poly:' + poly.i, { el: el, box: box });
      });

      // The clip is the visible rectangle in screen pixels, grown by one label on every
      // side: a pipe that leaves the frame still blocks the name of the element it passes
      // just inside it, and the margin is the widest name this frame considered.
      const clip = { x1: -reach, y1: -reach, x2: rect.width + reach, y2: rect.height + reach };
      const placed = placeLabels(items, segments, { margin: 0, selected: selectedKey, clip: clip });

      for (let n = 0; n < placed.length; n += 1) {
        const entry = placed[n];
        const target = byKey.get(entry.key);
        if (!target) continue;
        // The solver answers with the top-left corner of the rectangle; ``y`` of a <text>
        // is a baseline.  The insertion point is that corner plus the halo and plus how
        // far the glyphs reach outside the advance - measured (N01), never estimated, or
        // the ink would leave the rectangle the solver reserved, in both directions for a
        // name with a 'j' and for every italic.
        target.el.setAttribute('x', String(entry.dx + LABEL_HALO_PX + target.box.left));
        target.el.setAttribute('y', String(entry.dy + LABEL_HALO_PX + target.box.ascent));
        // The reserved rectangle, for the browser gate: it cannot read a closure and must
        // not patch the module it is there to measure.
        target.el.setAttribute('data-nm-box', `${entry.dx} ${entry.dy} ${target.box.w} ${target.box.h}`);
        target.el.style.display = '';
      }
    }

    // -------------------------------------------------------------------------
    // Mode switch and legend (stage 6)
    //
    // Both are built here and not in the template: the map is drawn in the browser,
    // the legend of a mode is a property of the code that paints it, and a template
    // carrying the nine recommendation rows would be a second list to keep in step.
    // -------------------------------------------------------------------------

    const tools = document.createElement('div');
    tools.className = 'nm-tools';

    const modeRow = document.createElement('div');
    modeRow.className = 'nm-modes';
    modeRow.setAttribute('role', 'group');
    modeRow.setAttribute('aria-label', 'Colour mode');
    tools.appendChild(modeRow);

    const legend = document.createElement('ul');
    legend.className = 'nm-legend';
    legend.setAttribute('aria-label', 'Legend');
    tools.appendChild(legend);

    const modeButtons = MODES.map(entry => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'nm-mode';
      button.textContent = entry.label;
      // aria-pressed and not aria-selected: these are toggle buttons in a group, not
      // tabs, and nothing here controls a panel.
      button.setAttribute('aria-pressed', 'false');
      button.addEventListener('click', () => setMode(entry.key));
      modeRow.appendChild(button);
      return { key: entry.key, el: button };
    });

    let legendButtons = [];

    /** The legend of the current mode, rebuilt from scratch on every change of mode. */
    function buildLegend() {
      while (legend.firstChild) legend.removeChild(legend.firstChild);
      legendButtons = legendEntries(mode).map(entry => {
        const item = document.createElement('li');
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'nm-legend-item';
        button.setAttribute('aria-pressed', 'false');
        const swatch = document.createElement('span');
        // The swatch wears the colour class itself, so one rule in the stylesheet
        // paints the pipe and the chip that stands for it.
        swatch.classList.add('nm-swatch', entry[0]);
        swatch.setAttribute('aria-hidden', 'true');
        const label = document.createElement('span');
        label.className = 'nm-legend-label';
        label.textContent = entry[1];
        button.appendChild(swatch);
        button.appendChild(label);
        button.addEventListener('click', () => toggleCategory(entry[0]));
        item.appendChild(button);
        legend.appendChild(item);
        return { key: entry[0], el: button };
      });
    }

    /**
     * Show which categories are faded.
     *
     * The entry of a faded category is only greyed - ``.nm-legend-off`` carries no
     * ``pointer-events: none`` - because it is the one control that brings the
     * category back.  ``aria-pressed`` says the same thing to a screen reader.
     */
    function updateLegendState() {
      legendButtons.forEach(item => {
        const off = dimmed.indexOf(item.key) !== -1;
        item.el.classList.toggle('nm-legend-off', off);
        item.el.setAttribute('aria-pressed', off ? 'true' : 'false');
      });
    }

    function toggleCategory(key) {
      dimmed = nextLegendState(dimmed, key);
      applyDim();
    }

    function clearLegend() {
      if (!dimmed.length) return false;
      dimmed = nextLegendState(dimmed, null);
      applyDim();
      return true;
    }

    /**
     * Switch the colouring mode.
     *
     * The dimmed categories belong to the mode that defined them - ``nm-v-3`` means
     * nothing among the recommendation classes - so the filter is dropped and every
     * element comes back before the new colours go on.
     */
    function setMode(next) {
      if (!isMode(next) || next === mode) return;
      mode = next;
      dimmed = [];
      savePrefs();
      buildLegend();
      updateModeButtons();
      paint();
    }

    function updateModeButtons() {
      modeButtons.forEach(item => {
        const on = item.key === mode;
        item.el.classList.toggle('nm-mode-on', on);
        item.el.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
    }

    buildLegend();
    updateModeButtons();
    if (host.parentNode) host.parentNode.insertBefore(tools, host);

    // -------------------------------------------------------------------------
    // View: zoom, pan and the three on-screen controls
    // -------------------------------------------------------------------------

    /**
     * The drawing-unit point under a client pixel, through the real CTM of the SVG.
     *
     * Not rect.width / viewBox.w: preserveAspectRatio="xMidYMid meet" letterboxes the
     * drawing whenever the container and the box differ in shape, and the anchor of
     * zoomAt only stays under the cursor if that offset is accounted for.  The CTM
     * knows it; arithmetic on the container size does not.
     */
    function clientToUnits(clientX, clientY) {
      if (!drawing || typeof drawing.svg.getScreenCTM !== 'function') return null;
      const ctm = drawing.svg.getScreenCTM();
      if (!ctm) return null;
      const point = drawing.svg.createSVGPoint();
      point.x = clientX;
      point.y = clientY;
      const local = point.matrixTransform(ctm.inverse());
      if (!Number.isFinite(local.x) || !Number.isFinite(local.y)) return null;
      return { x: local.x, y: local.y };
    }

    function zoomLimits() {
      if (!fitted || !isPositive(fitted.w)) return null;
      return { minW: fitted.w / ZOOM_IN_LIMIT, maxW: fitted.w * ZOOM_OUT_LIMIT };
    }

    /** Every zoom in this module goes through here, wheel and buttons alike. */
    function zoomAtClient(factor, clientX, clientY) {
      if (!viewBox) return;
      const anchor = clientToUnits(clientX, clientY);
      if (!anchor) return;
      viewBox = zoomAt(viewBox, anchor.x, anchor.y, factor, zoomLimits());
      rescale();
    }

    /** The + and - controls have no cursor to anchor to, so they use the centre. */
    function zoomByStep(factor) {
      const rect = host.getBoundingClientRect();
      if (!isPositive(rect.width) || !isPositive(rect.height)) return;
      zoomAtClient(factor, rect.left + rect.width / 2, rect.top + rect.height / 2);
    }

    function fitToView() {
      const rect = host.getBoundingClientRect();
      if (!isPositive(rect.width) || !isPositive(rect.height)) return;
      viewBox = fitViewBox(data.bbox, PAD_MIN_PX, { w: rect.width, h: rect.height });
      fitted = viewBox;
      rescale();
    }

    /** Move the view by a screen distance; panning changes no scale, so the CTM holds. */
    function panBy(dxPx, dyPx) {
      if (!viewBox || !drawing) return;
      const ctm = drawing.svg.getScreenCTM();
      if (!ctm || !isPositive(ctm.a) || !isPositive(ctm.d)) return;
      viewBox = { x: viewBox.x - dxPx / ctm.a, y: viewBox.y - dyPx / ctm.d, w: viewBox.w, h: viewBox.h };
      rescale();
    }

    /**
     * The keys of the map, every one of them going through the functions the buttons
     * and the wheel already use.
     *
     * There is no second implementation of a zoom, of a fit or of a pan here: an arrow
     * is ``panBy`` in screen pixels, ``+`` and ``-`` are ``zoomByStep`` with the very
     * step the wheel takes, and ``Home`` is the ``Fit`` button.  ``Enter`` activates the
     * link of the selection card - the same anchor, clicked the way the reader would
     * click it, so the browser opens it in a new tab on a modified click like any other
     * link.
     *
     * Arrows move the view and never move the focus: the map is one Tab stop, the drawn
     * elements are not focusable, and walking a network of sixty pipes with four keys is
     * not a view of anything - the table next to this tab is the equivalent and fully
     * accessible one.  A modified press is left to the browser and to the operating
     * system, which own Ctrl+Home, Alt+Arrow and the rest.
     *
     * ``Escape`` is deliberately absent: it is handled once, on the document, so that it
     * also clears the legend while the focus sits on a legend button.  Two handlers for
     * one key would be two definitions of what Escape means.
     */
    function mapKeyAction(event) {
      // Alt, Ctrl and Meta belong to the browser and to the operating system.  Shift
      // does not: on most layouts '+' *is* a shifted key, and refusing it there would
      // leave the keyboard with no way to zoom in.
      if (event.altKey || event.ctrlKey || event.metaKey) return false;
      const key = event.key;
      if (key === 'ArrowLeft') panBy(PAN_STEP_PX, 0);
      else if (key === 'ArrowRight') panBy(-PAN_STEP_PX, 0);
      else if (key === 'ArrowUp') panBy(0, PAN_STEP_PX);
      else if (key === 'ArrowDown') panBy(0, -PAN_STEP_PX);
      else if (key === '+' || key === '=') zoomByStep(ZOOM_STEP);
      else if (key === '-') zoomByStep(1 / ZOOM_STEP);
      else if (key === 'Home') fitToView();
      else if (key === 'Enter') {
        const open = card ? card.querySelector('.nm-card-open') : null;
        if (!open) return false;
        open.click();
      } else return false;
      return true;
    }

    /** The one keyboard listener of the module; ensureDrawn puts it on the <svg>. */
    function onMapKey(event) {
      // Taken means taken: an arrow that moved the view must not also scroll the page,
      // and Home must not jump to the top of the document behind the map.
      if (mapKeyAction(event)) event.preventDefault();
    }

    const controls = document.createElement('div');
    controls.className = 'nm-ctrls';

    /** A real button, because a div with role="button" is a keyboard trap in a costume. */
    function addControl(label, glyph, handler) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'nm-ctrl';
      button.setAttribute('aria-label', label);
      button.textContent = glyph;
      button.addEventListener('click', handler);
      controls.appendChild(button);
      return button;
    }

    addControl('Zoom in', '+', () => zoomByStep(ZOOM_STEP));
    addControl('Zoom out', '\u2212', () => zoomByStep(1 / ZOOM_STEP));
    addControl('Fit network to view', 'Fit', fitToView);
    // D19 - the fourth control, and the only one carrying a state.  addControl builds a
    // button and knows nothing about pressed, so updateLabelButton owns aria-pressed and
    // .nm-ctrl-on here, the way updateModeButtons owns them for the mode row.  Four
    // buttons of 32 px with three 6 px gaps are 146 px: a column of that height does NOT
    // fit the 114.4 px canvas the auto session draws on 390x844, so below 520 px the
    // stylesheet folds the same four into two columns of two, 70x70 px (F7).
    labelButton = addControl('Show element names', 'Aa', toggleLabels);
    updateLabelButton();
    host.appendChild(controls);

    /** Say that the names are on, to the reader and to a screen reader alike. */
    function updateLabelButton() {
      if (!labelButton) return;
      labelButton.classList.toggle('nm-ctrl-on', labelsOn);
      labelButton.setAttribute('aria-pressed', labelsOn ? 'true' : 'false');
    }

    /**
     * Turn the names on or off (D19, D25).
     *
     * The preference is written through savePrefs, so the gesture remembers every field
     * and not only its own.  rescale() then writes the transforms and takes one of its
     * three ways out, and scheduleLabels() asks for the frame that places them.  The
     * hint lives here and not in rescale(): it covers *both* ways the group can stay
     * hidden while the names are on - the density of the symbols and the gap threshold -
     * and on a 390x844 phone at the fitted view it is the first of them that trips, so a
     * hint tied to the threshold alone would explain nothing on a phone.
     *
     * There is a third way, and it is not one of those two: a browser that hands back no
     * 2D context measures no text and so builds no name at all (ensureLabels walks out,
     * ``labelsBuilt`` stays false).  Telling that reader to zoom in would be advice no
     * zoom can follow, so the gesture is refused instead (F4): the button goes back off,
     * the hint says why, and nothing is written - a stored ``labels: true`` would bring
     * the same dead state back on the next visit.  ``labelCanvas && !labelCtx`` is the
     * precise question: ``!labelsBuilt`` is also true for a panel that is still hidden,
     * and that one does build its names on the next rescale.
     */
    function toggleLabels() {
      labelsOn = !labelsOn;
      updateLabelButton();
      if (labelsOn) ensureLabels();
      if (labelsOn && labelCanvas && !labelCtx) {
        labelsOn = false;
        updateLabelButton();
        showHint('Names are not available in this browser');
        return;
      }
      savePrefs();
      rescale();
      scheduleLabels();
      if (labelsOn && drawing && drawing.labelGroup.style.display === 'none') {
        showHint('Zoom in to show names');
      }
    }

    const hint = document.createElement('div');
    hint.className = 'nm-hint';
    hint.setAttribute('aria-hidden', 'true');
    hint.hidden = true;
    hint.textContent = HINT_TEXT;
    host.appendChild(hint);

    /**
     * Show the overlay, with *text* or with the default sentence.
     *
     * The text is assigned on **every** show and not once at construction: the names
     * button borrows this overlay for "Zoom in to show names", and without the
     * assignment the next unmodified wheel would tell the reader to zoom in to show
     * names.  Every existing caller passes nothing and keeps the default.
     */
    function showHint(text) {
      hint.textContent = typeof text === 'string' && text !== '' ? text : HINT_TEXT;
      hint.hidden = false;
      if (hintTimer) clearTimeout(hintTimer);
      hintTimer = setTimeout(() => {
        hint.hidden = true;
        hintTimer = null;
      }, HINT_MS);
    }

    // -------------------------------------------------------------------------
    // Tooltip — one element, reused, never a target of its own
    // -------------------------------------------------------------------------

    const tip = document.createElement('div');
    tip.className = 'nm-tip';
    tip.setAttribute('aria-hidden', 'true');
    tip.hidden = true;
    host.appendChild(tip);

    function hitUnder(target) {
      return target && typeof target.closest === 'function' ? target.closest('[data-nm-kind]') : null;
    }

    function hideTip() {
      tip.hidden = true;
    }

    /** Show the name of *hit* near the pointer, bounced off the edges of the container. */
    function showTip(hit, clientX, clientY) {
      const rect = host.getBoundingClientRect();
      const name = hit.getAttribute('data-nm-name') || '';
      // The name stays: it is the reason the tooltip exists.  The blocked element only
      // says, in the same breath, why a click on it opens nothing - and the extra class
      // lets that sentence wrap, because the one-line tooltip would cut it off.
      const blocked = hit.classList.contains(NOT_ANALYSED_CLASS);
      tip.classList.toggle('nm-tip-na', blocked);
      tip.textContent = blocked ? `${name} \u2014 element not analysed` : name;
      tip.hidden = false;
      const box = tip.getBoundingClientRect();
      let x = clientX - rect.left + 12;
      let y = clientY - rect.top + 12;
      if (x + box.width > rect.width) x = Math.max(0, clientX - rect.left - box.width - 12);
      if (y + box.height > rect.height) y = Math.max(0, clientY - rect.top - box.height - 12);
      tip.style.left = `${x}px`;
      tip.style.top = `${y}px`;
    }

    // -------------------------------------------------------------------------
    // Selection and the selection card (D11)
    // -------------------------------------------------------------------------

    /**
     * One results value, or null when the row has nothing usable under that key.
     *
     * *factor* converts the stored unit into the shown one (millions of litres into
     * cubic metres) and is applied only to a finite number: a string value is passed
     * through as it stands.
     */
    function formatValue(value, unit, factor) {
      if (typeof value === 'number' && Number.isFinite(value)) {
        const scaled = Number.isFinite(factor) && factor > 0 ? value * factor : value;
        if (!Number.isFinite(scaled)) return null;
        const text = Math.abs(scaled) >= 100 ? scaled.toFixed(0) : scaled.toFixed(2);
        return unit ? `${text} ${unit}` : text;
      }
      if (typeof value === 'string' && value !== '') return value;
      return null;
    }

    /**
     * Put the card where ``cardPos`` says, or give it back to the stylesheet.
     *
     * The only place that writes the position of the card into its style, and it is
     * called from three places: at the end of ``buildCard`` (so a new card opens where
     * the reader left the last one), while the heading is being dragged, and from the
     * ResizeObserver (so a canvas that changed size does not leave the card outside
     * it).  Never from ``rescale``: that one runs on every wheel notch and every
     * pointermove of a pan, and it belongs to the viewBox of the <svg>, not to an HTML
     * sibling of it.
     *
     * Three things it has to do, in this order:
     *
     * 1. Return on ``!card``.  The observer calls this unconditionally and fires
     *    already at ``.observe(host)`` - on entering the map tab, when nothing has ever
     *    been selected and ``card`` is null.  Without the guard every opening of the map
     *    would put a TypeError in the console.
     * 2. ``cardPos === null`` (nothing was ever dragged) clears the three properties
     *    again, and ``left: 10px; bottom: 10px`` of the stylesheet takes over.
     * 3. Otherwise clamp against the canvas *now* - **into the style and into nothing
     *    else**.  ``cardPos`` is the corner the reader chose and this function never
     *    writes it (F3): the ResizeObserver calls this on every size change, so a write
     *    back would let a narrow window quietly replace a position chosen in a wide one,
     *    and the next savePrefs - the mode row, the names button - would persist the
     *    replacement.  Widening the window back could then restore nothing.  ``null``
     *    back from clampCardPos means the geometry is not readable yet (a hidden panel
     *    measures 0) and nothing is written at all.
     *
     * The clamp for *storage* is a different question and lives in ``settleCardPos``,
     * which runs only where a gesture ends.
     *
     * ``bottom: 'auto'`` is not optional: the stylesheet pins the card with
     * ``left + bottom``, so a ``top`` added next to a live ``bottom`` stretches the card
     * over the whole height of the canvas.
     */
    function applyCardPos() {
      if (!card) return;
      if (cardPos === null) {
        card.style.left = '';
        card.style.top = '';
        card.style.bottom = '';
        return;
      }
      const placed = clampedCardCorner();
      if (!placed) return;
      card.style.left = `${placed.left}px`;
      card.style.top = `${placed.top}px`;
      card.style.bottom = 'auto';
    }

    /**
     * The corner ``cardPos`` clamps to against the canvas as it measures right now, or
     * ``null`` when neither the card nor the canvas has a size yet.
     *
     * One reading of the geometry for the two questions that ask it - where to draw the
     * card (applyCardPos) and what to remember of it (settleCardPos) - because a second
     * expression would part company with the first at the next change of the clamp.
     */
    function clampedCardCorner() {
      if (!card || cardPos === null) return null;
      const rect = host.getBoundingClientRect();
      return clampCardPos(cardPos.left, cardPos.top, card.offsetWidth, card.offsetHeight, rect.width, rect.height);
    }

    /**
     * Bring ``cardPos`` inside the canvas the gesture ended in, and only then.
     *
     * ``moveCardDrag`` writes the raw pointer arithmetic, which may well be outside the
     * canvas, and ``applyCardPos`` no longer normalises it (F3).  This is the one moment
     * at which the clamped number *is* what the reader meant, because it is where the
     * card came to rest under their finger; a clamp at any other moment is the viewport
     * speaking, not the reader.  It therefore runs at both ends of the gesture - the
     * pointerup/pointercancel end and the Escape end - immediately before savePrefs.
     */
    function settleCardPos() {
      const placed = clampedCardCorner();
      if (placed) cardPos = placed;
    }

    /**
     * Take the card by its heading.
     *
     * The pointer is captured **here**, in pointerdown, and not after a threshold: from
     * that moment the browser delivers every move, up and cancel of this pointer to the
     * heading, wherever the pointer goes - off the heading, off the card, off the
     * canvas, out of the window.  There is no window in which an event could land
     * somewhere else, so a gesture that starts one pixel from an edge and leaves the
     * handle with its first move works by construction.
     *
     * Capturing on a *container* would be wrong for the reason written at the map's own
     * pointerdown: the capture retargets the resulting click as well.  The handle is a
     * bare <p> with the name in it - no button, no link, nothing that a click does
     * anything with - which is what makes the early capture safe here.
     *
     * The guard against a second gesture is unconditional on purpose.  Owning the
     * pointer is the *reason* the record exists, so a record that is there always
     * belongs to a pointer that is still down; a second finger on the same heading has
     * nothing to add and its press is simply ignored.
     */
    function beginCardDrag(event) {
      if (event.button !== 0) return;
      if (cardDrag) return;
      // The handle suppresses the browser's own text selection, which is what dragging
      // a <p> means to it; the values below the heading keep theirs.
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      cardDrag = {
        pointerId: event.pointerId,
        handle: event.currentTarget,
        x0: event.clientX,
        y0: event.clientY,
        // offsetLeft/offsetTop are already in the coordinates of the canvas: .nm-canvas
        // is position: relative and so is the offsetParent of the card.  They read the
        // right numbers even while the card still stands on ``bottom: 10px`` and has
        // never had a ``left`` of its own in its inline style.
        startLeft: card ? card.offsetLeft : 0,
        startTop: card ? card.offsetTop : 0
      };
    }

    /**
     * Move the card with the pointer that owns the gesture, and nothing else.
     *
     * The arithmetic is raw - the pointer may well have left the canvas - and it stays
     * raw in ``cardPos``: applyCardPos clamps what it draws, settleCardPos clamps what
     * the end of the gesture remembers (F3).
     */
    function moveCardDrag(event) {
      if (!cardDrag || event.pointerId !== cardDrag.pointerId) return;
      cardPos = {
        left: cardDrag.startLeft + (event.clientX - cardDrag.x0),
        top: cardDrag.startTop + (event.clientY - cardDrag.y0)
      };
      applyCardPos();
      // The first move of the gesture takes the tooltip away; the calls after it find
      // ``tip.hidden`` already true and set the same flag again, which costs nothing.
      // Nothing is written to localStorage here - see savePrefs.
      hideTip();
    }

    /**
     * End the gesture - on pointerup and on pointercancel alike.
     *
     * The owner test comes **first**, before anything is released or saved: a second
     * finger put on the same heading is ignored by the guard in beginCardDrag, but the
     * pointerup of that second finger still arrives here, and without this line it
     * would end the gesture of the first one, leaving a finger down that no longer
     * drags anything.
     *
     * The release sits in a try: the browser may have dropped the capture already (the
     * element left the document, the gesture was cancelled), and then the call throws
     * although the gesture is over either way.  Cancel saves like up does, because a
     * cancelled gesture still leaves the card where it had been moved to.
     */
    function endCardDrag(event) {
      if (!cardDrag || event.pointerId !== cardDrag.pointerId) return;
      try {
        cardDrag.handle.releasePointerCapture(cardDrag.pointerId);
      } catch (err) {
        /* already released by the browser; the gesture is finished all the same */
      }
      settleCardPos();
      savePrefs();
      cardDrag = null;
    }

    function clearSelection() {
      if (selection && selection.el) selection.el.classList.remove('nm-sel');
      selection = null;
      // The gesture goes, the position stays.  Esc during a drag removes the card with
      // a captured pointer on it, so the record has to go with it; ``cardPos`` is the
      // point of this stage and survives every selection, on purpose.
      //
      // And it ends the gesture as surely as a pointerup does, so it saves like one
      // (F2).  Escape does not reach endCardDrag - the heading that holds the capture
      // is removed a few lines below, and the browser answers that with
      // lostpointercapture, which nothing listens to - so without these lines the card
      // would stay where the last move left it for the rest of the visit while
      // localStorage still held the corner before the gesture.  The release repeats
      // endCardDrag's tail rather than sharing a helper with it, because the order
      // there - owner test, release, settle, save - is asserted on that function's own
      // body and a gesture that ends with its card is a different order: there is no
      // second pointer to refuse.
      if (cardDrag) {
        try {
          cardDrag.handle.releasePointerCapture(cardDrag.pointerId);
        } catch (err) {
          /* already released by the browser; the gesture is finished all the same */
        }
        settleCardPos();
        savePrefs();
      }
      cardDrag = null;
      if (card) {
        card.remove();
        card = null;
      }
      // The strokes of the obstacles shrink back as .nm-sel goes, and this function calls
      // no rescale(), so the names are laid out again from here (R07).
      scheduleLabels();
    }

    function buildCard(kind, name) {
      const row = findResult(results, kind, name);

      card = document.createElement('div');
      card.className = 'nm-card';
      card.setAttribute('role', 'group');
      card.setAttribute('aria-label', 'Selected element');

      const close = document.createElement('button');
      close.type = 'button';
      close.className = 'nm-card-close';
      close.setAttribute('aria-label', 'Close selection');
      close.textContent = '\u00d7';
      close.addEventListener('click', clearSelection);
      card.appendChild(close);

      // textContent, always: the name is user input from the model file.
      const heading = document.createElement('p');
      heading.className = 'nm-card-name';
      heading.textContent = name;
      // Four listeners, all four on the heading and none of them on the card or on the
      // container.  The card is built again on every selection, so nothing leaks and
      // nothing has to be detached.  The heading is the whole handle (D7): the values
      // stay selectable and the buttons stay clickable, which they would not be if the
      // gesture sat on .nm-card.
      heading.addEventListener('pointerdown', beginCardDrag);
      heading.addEventListener('pointermove', moveCardDrag);
      heading.addEventListener('pointerup', endCardDrag);
      heading.addEventListener('pointercancel', endCardDrag);
      card.appendChild(heading);

      if (row) {
        const values = document.createElement('dl');
        values.className = 'nm-card-values';
        const fields = Object.prototype.hasOwnProperty.call(CARD_FIELDS, kind) ? CARD_FIELDS[kind] : [];
        fields.forEach(field => {
          const text = formatValue(row[field[1]], field[2], field[3]);
          if (text === null) return;
          const term = document.createElement('dt');
          term.textContent = field[0];
          const value = document.createElement('dd');
          value.textContent = text;
          values.appendChild(term);
          values.appendChild(value);
        });
        if (values.childNodes.length) card.appendChild(values);
      }

      // An element without a detail page is a normal state, not an error: a node that
      // ends no conduit has no panel to open.  The anchor is then not created at all —
      // a dead link is worse than a sentence saying there is nothing to open — while
      // the element still selects and still shows its values.  The href itself always
      // comes from the server's reverse() in the payload; nothing here builds a URL,
      // and nothing here assigns to window.location.
      const href = row && row.detail_url ? row.detail_url : null;
      if (href) {
        const open = document.createElement('a');
        open.className = 'nm-card-open';
        open.href = href;
        open.textContent = 'Open details';
        card.appendChild(open);
      } else {
        const missing = document.createElement('p');
        missing.className = 'nm-card-none';
        missing.textContent = 'No detail page for this element.';
        card.appendChild(missing);
      }

      const toTable = document.createElement('button');
      toTable.type = 'button';
      toTable.className = 'nm-card-table';
      toTable.textContent = 'Show in table';
      toTable.addEventListener('click', () => {
        // The only event this module ever sends out, and the only way a gesture on the
        // map changes the tab.  Selecting an element emits nothing: if it did, one
        // click would switch to the table and this card would never be seen.
        root.dispatchEvent(new CustomEvent('nm:select', {
          bubbles: true,
          detail: { kind: kindToTab(kind), name: name }
        }));
      });
      card.appendChild(toTable);

      // Right after the <svg> in the DOM, so that one Tab out of the map reaches the
      // card and not the floating view controls.  The card is positioned absolutely
      // and sits above them either way, so the order costs nothing on screen.
      host.insertBefore(card, drawing ? drawing.svg.nextSibling : host.firstChild);
      // After the insertion, never before it: only a card that is in the document has
      // an offsetWidth, and the clamp needs the size of *this* card.
      applyCardPos();
    }

    function selectElement(hit) {
      const kind = hit.getAttribute('data-nm-kind');
      const index = Number(hit.getAttribute('data-nm-i'));
      const name = hit.getAttribute('data-nm-name') || '';
      const list = drawing && drawing.els && Object.prototype.hasOwnProperty.call(drawing.els, kind) ? drawing.els[kind] : null;
      const el = list && Number.isInteger(index) ? list[index] : null;
      clearSelection();
      if (!el) return;
      el.classList.add('nm-sel');
      selection = { kind: kind, index: index, name: name, el: el };
      buildCard(kind, name);
      // The selected element goes to the front of the placement order (D27) and its
      // outline is now wider; neither fact reaches the names without this call.
      scheduleLabels();
    }

    // -------------------------------------------------------------------------
    // Pointer and wheel
    // -------------------------------------------------------------------------

    host.addEventListener('wheel', event => {
      const action = wheelAction(event);
      if (!action.zoom) {
        // Nothing is cancelled and nothing moves: the wheel still belongs to whatever
        // the reader was scrolling.  The overlay says how to zoom instead.
        showHint();
        return;
      }
      event.preventDefault();
      zoomAtClient(action.factor, event.clientX, event.clientY);
    }, { passive: false });

    host.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      // The pointer is deliberately NOT captured here.  Capturing redirects the
      // following events — the resulting click included — to the capturing element,
      // so a container that captured on pointerdown would see every click with itself
      // as the target and nothing on the map could ever be selected.  Capture happens
      // in pointermove, once the gesture is known to be a pan.
      //
      // Every press starts a fresh gesture, including one that lands on the card or on
      // a control: a click handler reading a stale target would re-select whatever the
      // reader clicked before.
      //
      // ...but a *second* finger does not, and this is the one press that is ignored.
      // The canvas sets touch-action: none, so both contacts of a two-finger gesture
      // arrive here; replacing the record of the first one loses the fact that its
      // pointer is captured, and from then on its moves match no gesture (the map
      // freezes mid-pan) while ``endGesture`` returns on the pointerId mismatch and
      // never releases the capture it no longer knows about.  The first finger owns
      // the gesture until it ends it.
      if (gesture && gesture.active && event.pointerId !== gesture.pointerId) return;
      gesture = {
        pointerId: event.pointerId,
        x0: event.clientX,
        y0: event.clientY,
        lastX: event.clientX,
        lastY: event.clientY,
        maxDist: 0,
        target: event.target,
        pannable: Boolean(drawing && drawing.svg.contains(event.target)),
        captured: false,
        active: true
      };
    });

    host.addEventListener('pointermove', event => {
      // ``active`` and not merely ``gesture``: the state outlives pointerup so that the
      // click can read it, and a move made after the button came up is a hover, not a
      // continuation of the pan.  Without the flag the map kept panning under a moving
      // cursor and the tooltip never appeared again.
      // A release outside the container never reaches this element, so a gesture could
      // otherwise stay "active" and start panning again on a plain hover.  No button
      // down (buttons === 0 covers mouse, pen and touch) means the gesture is over.
      if (gesture && gesture.active && event.buttons === 0) gesture.active = false;
      if (gesture && gesture.active && event.pointerId === gesture.pointerId) {
        gesture.maxDist = trackDist(gesture.maxDist, gesture.x0, gesture.y0, event.clientX, event.clientY);
        if (gesture.pannable && isDrag(gesture.maxDist)) {
          if (!gesture.captured && typeof host.setPointerCapture === 'function') {
            host.setPointerCapture(event.pointerId);
            gesture.captured = true;
            hideTip();
          }
          panBy(event.clientX - gesture.lastX, event.clientY - gesture.lastY);
        }
        gesture.lastX = event.clientX;
        gesture.lastY = event.clientY;
        if (gesture.captured) return;
      }
      const hit = hitUnder(event.target);
      if (hit) showTip(hit, event.clientX, event.clientY);
      else hideTip();
    });

    function endGesture(event) {
      if (!gesture || event.pointerId !== gesture.pointerId) return;
      if (gesture.captured && typeof host.hasPointerCapture === 'function' && host.hasPointerCapture(event.pointerId)) {
        host.releasePointerCapture(event.pointerId);
      }
      gesture.captured = false;
      gesture.active = false;
      // The gesture state is NOT cleared: click arrives after pointerup and still has
      // to read maxDist and target from it.  The click that follows consumes it, and a
      // press no click ever follows is replaced by the next pointerdown.
    }

    host.addEventListener('pointerup', endGesture);
    host.addEventListener('pointercancel', event => {
      endGesture(event);
      gesture = null;
    });
    host.addEventListener('pointerleave', hideTip);

    host.addEventListener('click', event => {
      // A click belongs to exactly one gesture, so the gesture is *consumed* here.
      // Kept, it outlives the press that made it and is read again by the next click
      // that has no press of its own — which is every click a keyboard makes: Enter on
      // the card's close button produces one, the stale target is still the element
      // selected by the last click on the map, and the card would be rebuilt by the
      // very button that closes it.  A drag leaves it behind in the same way.
      //
      // A gesture still ``active`` belongs to a pointer that is still down: this click
      // came from somewhere else (the second finger of a two-finger touch, the
      // keyboard) and must neither select anything nor eat the gesture under the
      // finger that is still panning.
      const gone = gesture;
      if (gone && gone.active) return;
      gesture = null;
      if (gone && isDrag(gone.maxDist)) return;
      // A click that starts inside the card or the view controls is never a selection,
      // whatever it carries: those are real controls with listeners of their own.
      if (event.target && typeof event.target.closest === 'function' && event.target.closest('.nm-card, .nm-ctrls')) return;
      // The element of the pointerdown, not of the click: past the drag threshold the
      // second one is the container, because the pointer was captured.  Below the
      // threshold the two are the same element anyway.
      const source = gone && gone.target ? gone.target : event.target;
      const hit = hitUnder(source);
      if (!hit) return;
      // The element has no results row: there is nothing to put on a card and no page
      // to open.  Only the click is refused - the pointer still hovers it and the
      // tooltip still says what it is and why it is grey.
      if (hit.classList.contains(NOT_ANALYSED_CLASS)) return;
      selectElement(hit);
    });

    /**
     * The table filtered itself; everything it no longer lists steps back (stage 7).
     *
     * The only message the map listens for, and the only reason the two scripts talk
     * at all in this direction.  The set is kept even while nothing is drawn yet — the
     * panel may still be hidden — because ``ensureDrawn`` ends in ``paint``, which
     * ends in ``applyDim`` and will find it there.
     *
     * The event is emitted on every render, filtered or not, and the first one comes
     * from merely opening the tab — which is why only the elements the table actually
     * lists are ever offered to ``dimSet``.  Everything else is drawn but analysed
     * nowhere (a pump, a weir, a [DIVIDERS] node, a storage outline), is absent from
     * ``visible`` on every render, and would be faded to 15 % and made unclickable by
     * a table nobody has filtered.  ``analysedElements`` narrows the collections with
     * the same ``isAnalysed`` that greys those elements, so the two states stay
     * exclusive.
     *
     * The very first render of the table happens before this listener exists:
     * results-v2.js is loaded first and its initial ``renderTable`` runs inside its
     * own init.  That event is not worth chasing — the filters are not restored from
     * storage, so it lists everything and asks for nothing to be faded.
     */
    root.addEventListener('rv2:filterchange', event => {
      const detail = event && event.detail ? event.detail : null;
      if (!detail) return;
      const kind = tabToKind(detail.kind);
      if (!kind) return;
      const collections = {};
      COLLECTIONS.forEach(pair => {
        collections[pair[0]] = analysedElements(data[pair[1]], name => isAnalysed(pair[0], name));
      });
      filterDim[kind] = new Set(dimSet(detail.kind, detail.visible, collections));
      applyDim();
    });

    document.addEventListener('keydown', event => {
      // Esc undoes every reading of the map: it puts the tooltip away, it drops the
      // selection with its card and it brings back every category the legend faded.
      //
      // On the document and not on the <svg>, and that is the whole reason this one
      // listener is not on the container with the others: the categories are faded from
      // the legend, so the focus sits on a legend button exactly when Esc has most to
      // undo, and a listener on the drawing would never see that key.  The early return
      // keeps Esc free for the rest of the page when the map has nothing to undo.
      if (event.key !== 'Escape') return;
      hideTip();
      if (!selection && !dimmed.length) return;
      clearSelection();
      clearLegend();
    });

    if (typeof ResizeObserver !== 'undefined') {
      // One mechanism for both cases: the observer reports the container's size when
      // it starts observing (the map tab is active on load) and again when the panel
      // stops being hidden (0x0 -> real size).  Drawing stays lazy either way.
      new ResizeObserver(() => {
        const rect = host.getBoundingClientRect();
        if (!isPositive(rect.width) || !isPositive(rect.height)) return;
        ensureDrawn();
        rescale();
        // After rescale, and here rather than inside it: a canvas that changed size can
        // leave the card hanging outside itself, and clampCardPos pulls it back in.
        applyCardPos();
      }).observe(host);
    } else {
      ensureDrawn();
      rescale();
    }
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initNetworkMap);
    } else {
      initNetworkMap();
    }
  }

  // Node sees the pure helpers; the browser skips this line.  It has to stay inside
  // the IIFE — beyond the closing brace the functions are out of scope.
  /** Distance from a point to an axis-aligned rectangle; zero anywhere inside it. */
  function rectPointDist(rect, x, y) {
    const dx = Math.max(rect.x1 - x, 0, x - rect.x2);
    const dy = Math.max(rect.y1 - y, 0, y - rect.y2);
    return Math.hypot(dx, dy);
  }

  /** Distance from a point to a finite segment, measured on the segment, not on its line. */
  function pointSegDist(x, y, seg) {
    const dx = seg.x2 - seg.x1;
    const dy = seg.y2 - seg.y1;
    const len2 = dx * dx + dy * dy;
    let t = len2 > 0 ? ((x - seg.x1) * dx + (y - seg.y1) * dy) / len2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    return Math.hypot(x - (seg.x1 + t * dx), y - (seg.y1 + t * dy));
  }

  /**
   * Liang-Barsky: the part of a segment that lies inside a rectangle, or null when
   * none of it does.
   *
   * It answers two questions with one piece of arithmetic — which part of an obstacle
   * is inside the frame, and whether a segment touches a candidate rectangle at all —
   * and it gets the case a naive "drop it when both ends are outside" loses: a
   * collector that enters the frame on one side and leaves on the other has both of
   * its ends outside and still crosses everything.
   */
  function clipSegment(x1, y1, x2, y2, rect) {
    if (!rect) return { x1: x1, y1: y1, x2: x2, y2: y2 };
    const dx = x2 - x1;
    const dy = y2 - y1;
    const p = [-dx, dx, -dy, dy];
    const q = [x1 - rect.x1, rect.x2 - x1, y1 - rect.y1, rect.y2 - y1];
    let t0 = 0;
    let t1 = 1;
    for (let n = 0; n < 4; n += 1) {
      if (p[n] === 0) {
        if (q[n] < 0) return null;
        continue;
      }
      const t = q[n] / p[n];
      if (p[n] < 0) {
        if (t > t1) return null;
        if (t > t0) t0 = t;
      } else {
        if (t < t0) return null;
        if (t < t1) t1 = t;
      }
    }
    return { x1: x1 + t0 * dx, y1: y1 + t0 * dy, x2: x1 + t1 * dx, y2: y1 + t1 * dy };
  }

  /** Distance from a rectangle to a segment: zero when they meet, otherwise the closest pair. */
  function rectSegDist(rect, seg) {
    if (clipSegment(seg.x1, seg.y1, seg.x2, seg.y2, rect)) return 0;
    let best = Math.min(rectPointDist(rect, seg.x1, seg.y1), rectPointDist(rect, seg.x2, seg.y2));
    best = Math.min(best, pointSegDist(rect.x1, rect.y1, seg));
    best = Math.min(best, pointSegDist(rect.x2, rect.y1, seg));
    best = Math.min(best, pointSegDist(rect.x2, rect.y2, seg));
    best = Math.min(best, pointSegDist(rect.x1, rect.y2, seg));
    return best;
  }

  /** Does the candidate rectangle bite into an obstacle?  Touching it does not count. */
  function rectHits(rect, obstacle) {
    if (obstacle.t === 'disk') return rectPointDist(rect, obstacle.x, obstacle.y) < obstacle.r - LABEL_EPS_PX;
    if (obstacle.t === 'seg') return rectSegDist(rect, obstacle) < obstacle.hw - LABEL_EPS_PX;
    return (
      rect.x1 < obstacle.x2 - LABEL_EPS_PX &&
      obstacle.x1 < rect.x2 - LABEL_EPS_PX &&
      rect.y1 < obstacle.y2 - LABEL_EPS_PX &&
      obstacle.y1 < rect.y2 - LABEL_EPS_PX
    );
  }

  /**
   * The offset of the top-left corner of a label rectangle for one anchor.
   *
   * ``reach`` is how far the rectangle starts from the anchor point: r + margin for a
   * node or a conduit, LABEL_POLY_RING_PX for the eight fallback positions of a
   * subcatchment.  C centres the rectangle on the anchor and exists for subcatchments
   * only — on a node or a conduit it would land on the symbol or on the pipe.
   */
  function anchorOffset(name, reach, w, h) {
    if (name === 'C') return { dx: -w / 2, dy: -h / 2 };
    if (name === 'E') return { dx: reach, dy: -h / 2 };
    if (name === 'W') return { dx: -reach - w, dy: -h / 2 };
    if (name === 'N') return { dx: -w / 2, dy: -reach - h };
    if (name === 'S') return { dx: -w / 2, dy: reach };
    if (name === 'NE') return { dx: reach, dy: -reach - h };
    if (name === 'SE') return { dx: reach, dy: reach };
    if (name === 'NW') return { dx: -reach - w, dy: -reach - h };
    if (name === 'SW') return { dx: -reach - w, dy: reach };
    return null;
  }

  /** node before link before subcatchment, whatever the input order was. */
  function kindRank(kind) {
    if (kind === 'node') return 0;
    if (kind === 'link') return 1;
    return 2;
  }

  /** The records of ``items`` that carry a usable geometry, copied, never the originals. */
  function labelItems(items) {
    const out = [];
    if (!Array.isArray(items)) return out;
    for (let n = 0; n < items.length; n += 1) {
      const raw = items[n];
      if (!raw || typeof raw !== 'object') continue;
      if (raw.kind !== 'node' && raw.kind !== 'link' && raw.kind !== 'poly') continue;
      if (typeof raw.key !== 'string' || !raw.key) continue;
      if (!Number.isFinite(raw.x) || !Number.isFinite(raw.y) || !Number.isFinite(raw.i)) continue;
      if (!Number.isFinite(raw.w) || !Number.isFinite(raw.h) || raw.w < 0 || raw.h < 0) continue;
      out.push({
        key: raw.key,
        kind: raw.kind,
        i: raw.i,
        x: raw.x,
        y: raw.y,
        r: Number.isFinite(raw.r) && raw.r > 0 ? raw.r : 0,
        w: raw.w,
        h: raw.h
      });
    }
    return out;
  }

  /**
   * A uniform bucket grid over screen space.
   *
   * Without it every candidate would be tried against every obstacle: eight anchors
   * times (nodes + segments) is 10^7 to 10^8 tests for one frame of a pan, which no
   * cap on the number of labels repairs.  A query walks the buckets of the candidate
   * rectangle grown by the widest stroke in the drawing, so an obstacle whose centre
   * line sits two buckets away but whose outline reaches in is still offered to the
   * exact test.  Nothing here iterates over the map itself, so the answer never
   * depends on the order the buckets were filled in.
   */
  function labelGrid(cell) {
    const cells = new Map();
    const loose = [];
    let visit = 0;

    function put(cx, cy, obstacle) {
      const key = cx + ',' + cy;
      const bucket = cells.get(key);
      if (bucket) bucket.push(obstacle);
      else cells.set(key, [obstacle]);
    }

    function putBox(x1, y1, x2, y2, obstacle) {
      const cx1 = Math.floor(x1 / cell);
      const cx2 = Math.floor(x2 / cell);
      const cy1 = Math.floor(y1 / cell);
      const cy2 = Math.floor(y2 / cell);
      if (!Number.isFinite(cx1) || !Number.isFinite(cx2) || !Number.isFinite(cy1) || !Number.isFinite(cy2)) {
        loose.push(obstacle);
        return;
      }
      if ((cx2 - cx1 + 1) * (cy2 - cy1 + 1) > LABEL_MAX_CELLS) {
        loose.push(obstacle);
        return;
      }
      for (let cx = cx1; cx <= cx2; cx += 1) {
        for (let cy = cy1; cy <= cy2; cy += 1) put(cx, cy, obstacle);
      }
    }

    /** Every bucket the centre line of a segment passes through (Amanatides-Woo). */
    function putSegment(seg) {
      const dx = seg.x2 - seg.x1;
      const dy = seg.y2 - seg.y1;
      let cx = Math.floor(seg.x1 / cell);
      let cy = Math.floor(seg.y1 / cell);
      const ex = Math.floor(seg.x2 / cell);
      const ey = Math.floor(seg.y2 / cell);
      if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(ex) || !Number.isFinite(ey)) {
        loose.push(seg);
        return;
      }
      let steps = Math.abs(ex - cx) + Math.abs(ey - cy);
      if (steps + 1 > LABEL_MAX_CELLS) {
        loose.push(seg);
        return;
      }
      const stepX = dx > 0 ? 1 : dx < 0 ? -1 : 0;
      const stepY = dy > 0 ? 1 : dy < 0 ? -1 : 0;
      const deltaX = stepX === 0 ? Infinity : Math.abs(cell / dx);
      const deltaY = stepY === 0 ? Infinity : Math.abs(cell / dy);
      let nextX = stepX === 0 ? Infinity : (stepX > 0 ? (cx + 1) * cell - seg.x1 : seg.x1 - cx * cell) / Math.abs(dx);
      let nextY = stepY === 0 ? Infinity : (stepY > 0 ? (cy + 1) * cell - seg.y1 : seg.y1 - cy * cell) / Math.abs(dy);
      for (;;) {
        put(cx, cy, seg);
        if ((cx === ex && cy === ey) || steps <= 0) return;
        steps -= 1;
        if (nextX < nextY) {
          cx += stepX;
          nextX += deltaX;
        } else {
          cy += stepY;
          nextY += deltaY;
        }
      }
    }

    return {
      addDisk: function (disk) {
        putBox(disk.x - disk.r, disk.y - disk.r, disk.x + disk.r, disk.y + disk.r, disk);
      },
      addRect: function (rect) {
        putBox(rect.x1, rect.y1, rect.x2, rect.y2, rect);
      },
      addSegment: putSegment,
      /** True as soon as one obstacle of the grown box answers the test. */
      any: function (rect, pad, test) {
        visit += 1;
        const cx1 = Math.floor((rect.x1 - pad) / cell);
        const cx2 = Math.floor((rect.x2 + pad) / cell);
        const cy1 = Math.floor((rect.y1 - pad) / cell);
        const cy2 = Math.floor((rect.y2 + pad) / cell);
        for (let cx = cx1; cx <= cx2; cx += 1) {
          for (let cy = cy1; cy <= cy2; cy += 1) {
            const bucket = cells.get(cx + ',' + cy);
            if (!bucket) continue;
            for (let n = 0; n < bucket.length; n += 1) {
              const obstacle = bucket[n];
              if (obstacle.seen === visit) continue;
              obstacle.seen = visit;
              if (test(obstacle)) return true;
            }
          }
        }
        for (let n = 0; n < loose.length; n += 1) {
          const obstacle = loose[n];
          if (obstacle.seen === visit) continue;
          obstacle.seen = visit;
          if (test(obstacle)) return true;
        }
        return false;
      }
    };
  }

  /**
   * Where the name of every drawn element may stand.
   *
   * The whole placement problem as a pure function: no document, no measurement, no
   * randomness, no clock.  It takes the elements already converted to *screen*
   * pixels, the obstacles they have to keep off, and answers with the offset of the
   * top-left corner of each label rectangle from the anchor of its element.  An
   * element with no free position is simply **absent** from the answer: a name is
   * never forced into a position it collides in.
   *
   *   items     [{key, kind, i, x, y, r, w, h}]   kind: 'node' | 'link' | 'poly'
   *   segments  [{x1, y1, x2, y2, hw, key}]       hw = half the drawn stroke plus the margin
   *   opts      {anchors, cell, max, margin, selected, clip}
   *   ->        [{key, dx, dy}]
   *
   * ``key`` is compound — 'node:3', 'link:12' — because a node and a conduit may
   * legally carry the same SWMM name, so a name alone would give two elements one
   * key and the order would stop being decidable; ``kind`` and ``i`` travel beside it
   * so nothing has to parse the string.  ``w``/``h`` are the **measured** glyph box
   * with the halo on every side, handed in by E4b: this function never derives a text
   * width from a string length, because ``text.length * 0.6 * fontPx`` understates the
   * proportional face the map inherits by a third ('WWWWWWWW' at 11 px is 52.8
   * guessed against 85.7 measured) and the name would then be parked on top of the
   * very symbol it was meant to miss.  ``getBBox()`` stays banned as well — it forces
   * a synchronous layout and does not exist in Node.
   *
   * ``dx``/``dy`` locate the **corner of the rectangle**, not the ``y`` attribute of a
   * ``<text>``: that element writes a baseline, and E4b turns the corner into
   * ``x = dx + LABEL_HALO_PX + actualBoundingBoxLeft``,
   * ``y = dy + LABEL_HALO_PX + actualBoundingBoxAscent``.  This is the only place the
   * collision rectangle is defined, so it has to be said here.
   *
   * Three rules carry the behaviour, and none of them is an accident:
   *
   * 1. **Own obstacle.**  Only the disk of the element's **own node** is skipped, and
   *    only as a safeguard, since the candidates start at r + margin anyway.  The
   *    segment of a link's **own** conduit is deliberately *not* skipped: were it
   *    skipped, the name of a horizontal pipe would pass in the E anchor while lying
   *    exactly along the pipe.  A conduit therefore carries ``r = hw`` — the same
   *    number as in its ``segments`` entry, never zero — so E and W fall of their own
   *    accord and N or S wins, with no special case anywhere.
   * 2. **Subcatchment outlines and outlet lines are obstacles for nobody.**  Do not
   *    "fix" this: a subcatchment carries its own name *inside* its outline (the C
   *    anchor), so were polygons obstacles no subcatchment name could stand at all.
   *    The obstacles are node disks, conduit segments and the labels already placed.
   * 3. **One total order, written once.**  The selected element first — named by
   *    ``opts.selected``, a compound key, not by a flag in ``items`` — then ``kind``
   *    (node 0, link 1, poly 2), then ``i`` ascending.  The input order means
   *    nothing: this function sorts for itself, or ``opts.max`` and determinism would
   *    say two different things about a shuffled input.  Sorting on the key as a
   *    string is forbidden, 'link:12' standing lexicographically before 'node:3'.
   *
   * ``opts.max`` caps how many elements are *offered* a position, taken from the head
   * of that order; ``opts.selected`` is put in front and only *added* when it did not
   * make the cap, so it displaces nobody and the answer may hold max + 1 entries.  A
   * selection matching no key is ignored without complaint.
   *
   * A subcatchment anchors on its centroid with ``r = 0`` and gets a ninth anchor, C,
   * **first** in its list: the middle of an outline is the best place for its name,
   * not the worst.  Its other eight are measured from a circle of radius
   * LABEL_POLY_RING_PX rather than from the point itself — see that constant.  A node
   * or a conduit never gets C.  ``opts.anchors`` replaces the eight fallback
   * positions, but not that rule: C is prepended for a subcatchment and dropped for
   * everything else, whatever the list says.
   *
   * The obstacles live in a uniform bucket grid (side ``opts.cell``).  Segments are
   * cut to ``opts.clip`` with Liang-Barsky **before** they are rasterised, or one long
   * collector would paint hundreds of buckets outside the frame while neither the
   * bucket side nor LABEL_MAX bounds the work.  Every placed label joins the same
   * grid, which buys the label-to-label test for nothing.
   */
  function placeLabels(items, segments, opts) {
    const options = opts && typeof opts === 'object' ? opts : {};
    const cell = Number.isFinite(options.cell) && options.cell > 0 ? options.cell : LABEL_CELL_PX;
    const max = Number.isFinite(options.max) && options.max >= 0 ? Math.floor(options.max) : LABEL_MAX;
    const margin = Number.isFinite(options.margin) && options.margin > 0 ? options.margin : 0;
    const anchors = Array.isArray(options.anchors) && options.anchors.length ? options.anchors : LABEL_ANCHORS;
    // C belongs to subcatchments alone, so it is stripped from whatever list the caller
    // hands in and prepended once for poly — never twice, never for a node or a pipe.
    const plainAnchors = anchors.indexOf('C') >= 0 ? anchors.filter(name => name !== 'C') : anchors;
    const polyAnchors = ['C'].concat(plainAnchors);
    const selected = typeof options.selected === 'string' ? options.selected : '';
    const clip = labelClipRect(options.clip);

    const list = labelItems(items);
    list.sort(function (a, b) {
      const rank = kindRank(a.kind) - kindRank(b.kind);
      return rank !== 0 ? rank : a.i - b.i;
    });

    const grid = labelGrid(cell);
    let pad = 0;

    for (let n = 0; n < list.length; n += 1) {
      const item = list[n];
      if (item.kind === 'node' && item.r > 0) {
        grid.addDisk({ t: 'disk', key: item.key, x: item.x, y: item.y, r: item.r, seen: 0 });
      }
    }

    if (Array.isArray(segments)) {
      for (let n = 0; n < segments.length; n += 1) {
        const raw = segments[n];
        if (!raw || typeof raw !== 'object') continue;
        if (!Number.isFinite(raw.x1) || !Number.isFinite(raw.y1)) continue;
        if (!Number.isFinite(raw.x2) || !Number.isFinite(raw.y2)) continue;
        if (!Number.isFinite(raw.hw) || raw.hw <= 0) continue;
        const cut = clipSegment(raw.x1, raw.y1, raw.x2, raw.y2, clip);
        if (!cut) continue;
        const seg = { t: 'seg', key: typeof raw.key === 'string' ? raw.key : '', x1: cut.x1, y1: cut.y1, x2: cut.x2, y2: cut.y2, hw: raw.hw, seen: 0 };
        if (seg.hw > pad) pad = seg.hw;
        grid.addSegment(seg);
      }
    }

    // The head of the order is what fits under the cap; the selection joins it in
    // front, and only adds an entry when the cap had already left it out.
    const queue = list.slice(0, max);
    if (selected) {
      for (let n = 0; n < list.length; n += 1) {
        if (list[n].key !== selected) continue;
        const at = queue.indexOf(list[n]);
        if (at >= 0) queue.splice(at, 1);
        queue.unshift(list[n]);
        break;
      }
    }

    const placed = [];
    for (let n = 0; n < queue.length; n += 1) {
      const item = queue[n];
      const poly = item.kind === 'poly';
      const reach = poly ? LABEL_POLY_RING_PX : item.r + margin;
      const names = poly ? polyAnchors : plainAnchors;
      for (let a = 0; a < names.length; a += 1) {
        const offset = anchorOffset(names[a], reach, item.w, item.h);
        if (!offset) continue;
        const rect = {
          t: 'rect',
          x1: item.x + offset.dx,
          y1: item.y + offset.dy,
          x2: item.x + offset.dx + item.w,
          y2: item.y + offset.dy + item.h,
          seen: 0
        };
        const blocked = grid.any(rect, pad, function (obstacle) {
          if (obstacle.t === 'disk' && obstacle.key === item.key) return false;
          return rectHits(rect, obstacle);
        });
        if (blocked) continue;
        grid.addRect(rect);
        placed.push({ key: item.key, dx: offset.dx, dy: offset.dy });
        break;
      }
    }
    return placed;
  }

  /** ``opts.clip`` as an ordered rectangle, or null when there is nothing to clip to. */
  function labelClipRect(clip) {
    if (!clip || typeof clip !== 'object') return null;
    if (!Number.isFinite(clip.x1) || !Number.isFinite(clip.y1)) return null;
    if (!Number.isFinite(clip.x2) || !Number.isFinite(clip.y2)) return null;
    return {
      x1: Math.min(clip.x1, clip.x2),
      y1: Math.min(clip.y1, clip.y2),
      x2: Math.max(clip.x1, clip.x2),
      y2: Math.max(clip.y1, clip.y2)
    };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = {fitViewBox, zoomAt, pointsToPath, outletPath, elementCount, symbolRadiusPx, isDrag, trackDist, wheelAction, kindToTab, tabToKind, analysedElements, dimSet, findResult, pickColor, nextLegendState, readMapPrefs, clampCardPos, writeMapPrefs, placeLabels};
})();
