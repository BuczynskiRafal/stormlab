/**
 * Empirical harness of the pure layer of the network map (plan23, stages 5 and 6).
 *
 * It tests **only the exported functions** of ``app/static/js/network-map.js`` and
 * builds no DOM at all — not even a mock one.  A mocked SVG proves that the mock
 * agrees with the code that drives it and nothing about a browser, so everything
 * that needs a document (the view controls, the selection card, Esc, the tooltip,
 * the cursor, the pointer-events layer) is checked by the browser gate
 * ``rev/findings/plan23-checks/shots_map.py`` instead.
 *
 * ``require`` works because the module assigns ``module.exports`` inside its IIFE and
 * skips its own initialisation when there is no ``document``.
 *
 * Runnable on its own:  node app/sa/tests/harness_network_map.js  ->  exit code 0.
 */
const assert = require('assert');
const path = require('path');

const nm = require(path.resolve(__dirname, '../../static/js/network-map.js'));

const {
  fitViewBox,
  zoomAt,
  pointsToPath,
  outletPath,
  elementCount,
  symbolRadiusPx,
  isDrag,
  trackDist,
  wheelAction,
  kindToTab,
  tabToKind,
  analysedElements,
  dimSet,
  findResult,
  pickColor,
  nextLegendState,
  readMapPrefs,
  writeMapPrefs,
  clampCardPos,
  placeLabels
} = nm;

// The exported surface is a contract between the stages: eleven names after stage 5,
// fourteen after stage 6, fifteen after stage 7, sixteen once the filter dimming
// learned to leave the unanalysed elements alone, eighteen when the selection card
// learned to be dragged (plan24 E2: clampCardPos, and writeMapPrefs, which was written
// long before but never exported - the preferences grew a second field and the writer
// had to become testable from here), nineteen with the outlet line of plan24 E3b
// (outletPath) and twenty once plan24 E4a added the placement solver (placeLabels).
// A name that quietly stops being exported takes its whole group with it and the run
// would still end green without this line.
assert.strictEqual(Object.keys(nm).length, 20, 'network-map.js must export exactly twenty names');

let passed = 0;

function group(title, body) {
  body();
  passed += 1;
  console.log(`✓ ${title}`);
}

/** Equality for numbers a floating-point division is allowed to move a little. */
function near(actual, expected, tolerance, message) {
  assert.ok(
    Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${message}: expected ${expected} +- ${tolerance}, got ${actual}`
  );
}

// ---------------------------------------------------------------------------
// 1. fitViewBox — proportions, the minimum margin and a degenerate box
// ---------------------------------------------------------------------------
group('fitViewBox keeps the drawing inside the box, margin included', () => {
  const box = fitViewBox({ w: 200, h: 100 }, 8, { w: 800, h: 400 });
  assert.ok(box.w >= 200 && box.h >= 100, 'the fitted box must contain the drawing');
  near(box.x, -box.w / 2 + 100, 1e-9, 'the drawing must sit centred in x');
  near(box.y, -box.h / 2 + 50, 1e-9, 'the drawing must sit centred in y');
  near(box.w - 200, box.h - 100, 1e-9, 'the margin must be the same on both axes');

  // 3 % of the longer side is 6 drawing units; at 4 px per unit 8 screen pixels are
  // 2 units, so the fraction wins here and the floor wins on a small container.
  near(box.w, 212, 1e-9, 'the fractional margin must win on a large container');
  const tight = fitViewBox({ w: 200, h: 100 }, 8, { w: 80, h: 40 });
  near(tight.w, 200 + 2 * (8 / 0.4), 1e-9, 'the pixel floor must win on a small container');

  // Every degenerate input has to come back finite: a NaN here reaches the DOM as
  // viewBox="NaN NaN NaN NaN" and the map silently disappears.
  const unit = fitViewBox({ w: 1, h: 1 }, 8, { w: 300, h: 300 });
  assert.ok(Number.isFinite(unit.w) && unit.w > 1, 'a 1x1 bbox must still give a finite, padded box');
  near(unit.w, unit.h, 1e-9, 'a 1x1 bbox must stay square');
  [undefined, null, {}, { w: 0, h: 0 }, { w: NaN, h: 3 }].forEach(bbox => {
    const fallback = fitViewBox(bbox, 8, { w: 300, h: 300 });
    ['x', 'y', 'w', 'h'].forEach(key => {
      assert.ok(Number.isFinite(fallback[key]), `fitViewBox(${JSON.stringify(bbox)}).${key} must be finite`);
    });
    assert.ok(fallback.w > 0 && fallback.h > 0, 'a degenerate bbox must still give a drawable box');
  });
});

// ---------------------------------------------------------------------------
// 2. zoomAt — the anchor stays put, the limits bound the box
// ---------------------------------------------------------------------------
group('zoomAt holds the anchor and respects the limits', () => {
  const start = { x: 0, y: 0, w: 100, h: 50 };

  // The point that was at (25, 10) before the zoom is at (25, 10) after it: the
  // fraction of the box lying to its left and above it does not change.
  [1.15, 1 / 1.15, 4].forEach(factor => {
    const zoomed = zoomAt(start, 25, 10, factor);
    near((25 - zoomed.x) / zoomed.w, (25 - start.x) / start.w, 1e-12, 'the anchor must not move in x');
    near((10 - zoomed.y) / zoomed.h, (10 - start.y) / start.h, 1e-12, 'the anchor must not move in y');
    near(zoomed.w, start.w / factor, 1e-12, 'the width must follow the factor');
  });

  const limits = { minW: 20, maxW: 200 };
  const deepIn = zoomAt(start, 25, 10, 100, limits);
  near(deepIn.w, 20, 1e-12, 'zooming in must stop at minW');
  near((25 - deepIn.x) / deepIn.w, (25 - start.x) / start.w, 1e-12, 'the anchor must hold at the limit too');
  const farOut = zoomAt(start, 25, 10, 1 / 100, limits);
  near(farOut.w, 200, 1e-12, 'zooming out must stop at maxW');
  near(zoomAt({ x: 0, y: 0, w: 20, h: 10 }, 5, 5, 2, limits).w, 20, 1e-12, 'a box already at minW must not shrink further');
  near(zoomAt({ x: 0, y: 0, w: 200, h: 100 }, 5, 5, 0.5, limits).w, 200, 1e-12, 'a box already at maxW must not grow further');

  // Nothing degenerate may leak out as NaN; the box comes back unchanged instead.
  assert.strictEqual(zoomAt(start, 25, 10, 0), start, 'a zero factor must change nothing');
  assert.strictEqual(zoomAt(start, 25, 10, NaN), start, 'a NaN factor must change nothing');
  assert.strictEqual(zoomAt(start, NaN, 10, 2), start, 'a NaN anchor must change nothing');
  assert.strictEqual(zoomAt({ w: 0, h: 0, x: 0, y: 0 }, 1, 1, 2).w, 0, 'a degenerate box must come back as it was');
});

// ---------------------------------------------------------------------------
// 3. pointsToPath
// ---------------------------------------------------------------------------
group('pointsToPath writes M... L... and refuses what is not a line', () => {
  assert.strictEqual(pointsToPath([[0, 0], [10, 5], [20, 0]]), 'M0 0 L10 5 L20 0');
  assert.strictEqual(pointsToPath([[1.5, -2.25], [3, 4]]), 'M1.5 -2.25 L3 4');
  // A point that is not finite is dropped, and fewer than two usable points is not a
  // line: an empty ``d`` draws nothing instead of drawing a NaN.
  assert.strictEqual(pointsToPath([[0, 0], [NaN, 1], [10, 10]]), 'M0 0 L10 10');
  assert.strictEqual(pointsToPath([[0, 0]]), '');
  assert.strictEqual(pointsToPath([[0, 0], [Infinity, 1]]), '');
  assert.strictEqual(pointsToPath([]), '');
  assert.strictEqual(pointsToPath(null), '');
  assert.strictEqual(pointsToPath('0 0'), '');
});

// ---------------------------------------------------------------------------
// 4. elementCount — the same formula the extractor stamps into counts.elements
// ---------------------------------------------------------------------------
group('elementCount counts a link twice, as the extractor does', () => {
  assert.strictEqual(elementCount({ polys: [1, 2], links: [1, 2, 3], nodes: [1, 2, 3, 4] }), 2 + 6 + 4);
  assert.strictEqual(elementCount({ polys: [], links: [], nodes: [] }), 0);
  assert.strictEqual(elementCount({ nodes: [1] }), 1);
  assert.strictEqual(elementCount(null), 0);
  assert.strictEqual(elementCount('links'), 0);

  // plan24: an outlet line is one more drawable element, and only a polygon that
  // carries the point `o` has one.  The centroid `c` is never drawn and costs nothing.
  assert.strictEqual(
    elementCount({ polys: [{ i: 0, c: [0, 0], o: [1, 1] }, { i: 1, c: [2, 2] }], links: [1, 2], nodes: [1, 2, 3] }),
    2 + 4 + 3 + 1,
    'a polygon with an outlet costs one element more than one without'
  );
  // A version 1 payload has no `o` anywhere, so the formula gives exactly what it gave
  // before the stage: backward compatibility of the counter, asserted and not assumed.
  assert.strictEqual(
    elementCount({ polys: [{ i: 0 }, { i: 1 }], links: [1, 2, 3], nodes: [1, 2, 3, 4] }),
    2 + 6 + 4,
    'a version 1 payload must count exactly as it did before plan24'
  );
});

// ---------------------------------------------------------------------------
// 4b. outletPath - the line from the centroid of a subcatchment to its outlet
// ---------------------------------------------------------------------------
group('outletPath draws the line only for a usable pair of points', () => {
  assert.strictEqual(outletPath({ c: [0, 0], o: [10, 5] }), 'M0 0 L10 5');
  assert.strictEqual(outletPath({ i: 3, n: 'S1', c: [-2.5, 7], o: [1, 1.25] }), 'M-2.5 7 L1 1.25');

  // Everything a payload can be missing or get wrong ends as an empty `d`, which draws
  // nothing, and never as an exception or as d="M NaN NaN ...".
  [
    { c: [0, 0] },
    { o: [1, 1] },
    {},
    { c: '0 0', o: [1, 1] },
    { c: [0, 0], o: '1 1' },
    { c: null, o: [1, 1] },
    { c: [0, 0], o: null },
    { c: [NaN, 0], o: [1, 1] },
    { c: [0, Infinity], o: [1, 1] },
    { c: [0, 0], o: [NaN, 1] },
    { c: [0, 0], o: [1, -Infinity] },
    { c: [0, 0], o: [0, 0] }
  ].forEach(poly => {
    assert.strictEqual(outletPath(poly), '', `an unusable polygon (${JSON.stringify(poly)}) must give no line`);
  });
  [null, undefined, 'poly', 5, []].forEach(bad => {
    assert.strictEqual(outletPath(bad), '', `an unusable argument (${String(bad)}) must give no line`);
  });
});

// ---------------------------------------------------------------------------
// 5. symbolRadiusPx — five regimes, and every unusable min_gap
// ---------------------------------------------------------------------------
group('symbolRadiusPx covers the five regimes and every unusable min_gap', () => {
  // Crowded: a third of the gap is below the floor, so the caller is told 1.5 px and
  // hides the symbols rather than drawing a blot.
  near(symbolRadiusPx(1, 2), 1.5, 1e-12, 'a crowded network must clamp to the floor');
  near(symbolRadiusPx(2, 3), 2, 1e-12, 'in between, the radius is a third of the gap');
  near(symbolRadiusPx(5, 10), 3.5, 1e-12, 'a sparse network must clamp to the ceiling');
  near(symbolRadiusPx(2, 2.25), 1.5, 1e-12, 'exactly at the floor the floor is returned');
  near(symbolRadiusPx(10.5, 1), 3.5, 1e-12, 'exactly at the ceiling the ceiling is returned');

  // null, 0, negative, NaN and Infinity all arrive from real payloads (min_gap is
  // null for a model with a single node position).  Each one means "no density
  // information", which is the ceiling and never the vanishing of the symbols.
  [null, undefined, 0, -3, NaN, Infinity, -Infinity, 'x'].forEach(gap => {
    const radius = symbolRadiusPx(gap, 4);
    assert.ok(!Number.isNaN(radius), `symbolRadiusPx(${String(gap)}, 4) must not be NaN`);
    near(radius, 3.5, 1e-12, `an unusable min_gap (${String(gap)}) must give the ceiling`);
  });
  [null, 0, -1, NaN, Infinity].forEach(scale => {
    near(symbolRadiusPx(2, scale), 3.5, 1e-12, `an unusable pxPerUnit (${String(scale)}) must give the ceiling`);
  });
});

// ---------------------------------------------------------------------------
// 6. isDrag / trackDist — the threshold and the accumulator that feeds it
// ---------------------------------------------------------------------------
group('isDrag blocks a drag, passes a click, and time never enters into it', () => {
  assert.strictEqual(isDrag(5), true, '5 px is a drag');
  assert.strictEqual(isDrag(2), false, '2 px is a click');
  assert.strictEqual(isDrag(0), false, 'a motionless press is a click');
  assert.strictEqual(isDrag(4), false, 'exactly the threshold is still a click');
  assert.strictEqual(isDrag(4.001), true, 'just past the threshold is a drag');
  assert.strictEqual(isDrag(NaN), false, 'an unusable distance must not swallow a click');
  // There is no duration argument to pass: a press held for a second and a half is
  // the same click as a quick one, which is the whole point of the threshold.
  assert.strictEqual(isDrag.length, 1, 'isDrag takes the distance and nothing else');

  // The accumulator measures the *reach* of the gesture, so a loop that comes back to
  // its starting point is still a drag...
  let maxDist = 0;
  [[0, 30], [0, 0]].forEach(point => {
    maxDist = trackDist(maxDist, 0, 0, point[0], point[1]);
  });
  near(maxDist, 30, 1e-12, 'a loop must keep the reach it had');
  assert.strictEqual(isDrag(maxDist), true, 'a gesture that looped back must stay a drag');

  // ...while (0,0) -> (3,0) -> (0,3) reaches 3 px and stays a click.  Per-axis maxima
  // would give hypot(3, 3) = 4.24 here and refuse a legitimate selection.
  let small = 0;
  [[3, 0], [0, 3]].forEach(point => {
    small = trackDist(small, 0, 0, point[0], point[1]);
  });
  near(small, 3, 1e-12, 'the reach of the zig-zag is 3 px, not 4.24');
  assert.strictEqual(isDrag(small), false, 'the zig-zag must still select the element');

  near(trackDist(7, 0, 0, 1, 1), 7, 1e-12, 'the accumulator never goes down');
  near(trackDist(NaN, 0, 0, 3, 4), 5, 1e-12, 'an unusable accumulator restarts at 0');
  near(trackDist(2, 0, 0, NaN, 4), 2, 1e-12, 'an unusable point leaves the accumulator alone');
});

// ---------------------------------------------------------------------------
// 7. wheelAction — the modifier decides, and only the modifier
// ---------------------------------------------------------------------------
group('wheelAction zooms only with Ctrl or Cmd', () => {
  const inCtrl = wheelAction({ ctrlKey: true, metaKey: false, deltaY: -120 });
  assert.strictEqual(inCtrl.zoom, true, 'Ctrl + wheel must zoom');
  assert.strictEqual(inCtrl.prevent, true, 'Ctrl + wheel must be cancelled, page zoom included');
  assert.ok(inCtrl.factor > 1, 'a wheel away from the reader must zoom in');

  const outMeta = wheelAction({ ctrlKey: false, metaKey: true, deltaY: 120 });
  assert.strictEqual(outMeta.zoom, true, 'Cmd + wheel must zoom as well');
  assert.strictEqual(outMeta.prevent, true, 'Cmd + wheel must be cancelled as well');
  assert.ok(outMeta.factor < 1, 'a wheel towards the reader must zoom out');
  near(outMeta.factor * wheelAction({ ctrlKey: true, deltaY: -1 }).factor, 1, 1e-12, 'the two directions must undo each other');

  const plain = wheelAction({ ctrlKey: false, metaKey: false, deltaY: -120 });
  assert.strictEqual(plain.zoom, false, 'a bare wheel must not zoom');
  assert.strictEqual(plain.prevent, false, 'a bare wheel must not be cancelled: the page keeps it');

  assert.strictEqual(wheelAction({ ctrlKey: true, deltaY: 0 }).factor, 1, 'no delta, no movement');
  assert.strictEqual(wheelAction(null).zoom, false, 'a missing event must not zoom');
});

// ---------------------------------------------------------------------------
// 8. kindToTab / tabToKind — one translation, defined once
// ---------------------------------------------------------------------------
group('kindToTab and tabToKind are inverse on the three pairs', () => {
  [['link', 'conduits'], ['node', 'nodes'], ['poly', 'catchments']].forEach(pair => {
    assert.strictEqual(kindToTab(pair[0]), pair[1], `${pair[0]} must map to ${pair[1]}`);
    assert.strictEqual(tabToKind(pair[1]), pair[0], `${pair[1]} must map back to ${pair[0]}`);
    assert.strictEqual(tabToKind(kindToTab(pair[0])), pair[0], 'the round trip must return the kind');
    assert.strictEqual(kindToTab(tabToKind(pair[1])), pair[1], 'the round trip must return the tab');
  });
  ['conduits', 'map', 'info', '', 'Link', 'toString', undefined].forEach(unknown => {
    assert.strictEqual(kindToTab(unknown), undefined, `kindToTab(${String(unknown)}) must be undefined`);
  });
  ['link', 'map', 'info', '', 'Nodes', 'toString', undefined].forEach(unknown => {
    assert.strictEqual(tabToKind(unknown), undefined, `tabToKind(${String(unknown)}) must be undefined`);
  });
});

// ---------------------------------------------------------------------------
// 9. findResult — names survive the data layer, and the collections stay apart
// ---------------------------------------------------------------------------
group('findResult searches one collection and returns names bit for bit', () => {
  const awkward = ['A/B', 'P 1', 's#1', '"x"', 'zlewnia-ąęś', "o'brien", 'a\\b'];
  const payload = {
    conduits: awkward.map((name, index) => ({ name: name, detail_url: `/c/${index}/` })),
    nodes: [{ name: 'shared', detail_url: '/n/0/' }, { name: 'N2', detail_url: null }],
    catchments: [{ name: 'S1', detail_url: '/s/0/' }]
  };
  // The same name on a link and on a node is legal in SWMM; one shared map from name
  // to row would hand out whichever of the two was written last.
  payload.conduits.push({ name: 'shared', detail_url: '/c/shared/' });

  awkward.forEach((name, index) => {
    const row = findResult(payload, 'link', name);
    assert.ok(row, `the row of ${JSON.stringify(name)} must be found`);
    assert.strictEqual(row.name, name, 'the name must come back exactly as it went in');
    assert.strictEqual(row.detail_url, `/c/${index}/`, 'the row must be the right one');
  });

  assert.strictEqual(findResult(payload, 'link', 'shared').detail_url, '/c/shared/', 'a link must be found among links');
  assert.strictEqual(findResult(payload, 'node', 'shared').detail_url, '/n/0/', 'a node must be found among nodes');
  assert.strictEqual(findResult(payload, 'poly', 'S1').detail_url, '/s/0/', 'a subcatchment must be found among catchments');

  // A row without a detail page is a row, not a miss: the card shows its values and
  // simply builds no link.
  const noPage = findResult(payload, 'node', 'N2');
  assert.ok(noPage, 'a row whose detail_url is null must still be found');
  assert.strictEqual(noPage.detail_url, null, 'the missing URL must come through as it is');

  // A short overflow: SWMM writes Hours Flooded with two decimals, so a real flood of a
  // few seconds reads 0.00 h while the volume is not zero.  The flag comes from the
  // serializer (node_is_flooded), the map only carries it through, and the row a click
  // finds has to say "flooded" here - a map that derived its own verdict from the hours
  // would disagree with the table on exactly these nodes.
  const flooded = { name: 'J-short', hours_flooded: 0, flood_volume: 0.004, is_flooded: true, floodVolume: 0.004 };
  const floodPayload = { nodes: [flooded, { name: 'J-dry', hours_flooded: 0, flood_volume: 0, is_flooded: false }] };
  assert.strictEqual(findResult(floodPayload, 'node', 'J-short').is_flooded, true, 'a 0.00 h flood with volume must still be flooded');
  assert.strictEqual(findResult(floodPayload, 'node', 'J-dry').is_flooded, false, 'a dry node must not be flooded');

  assert.strictEqual(findResult(payload, 'link', 'nothing like this'), null, 'an unknown name must give null');
  assert.strictEqual(findResult(payload, 'link', 'a/b'), null, 'the comparison must not fold case');
  assert.strictEqual(findResult(payload, 'link', ' A/B '), null, 'the comparison must not trim');
  assert.strictEqual(findResult(payload, 'poly', 'shared'), null, 'a collection must not be searched across');
  assert.strictEqual(findResult(payload, 'conduits', 'A/B'), null, 'findResult speaks the drawing vocabulary');
  assert.strictEqual(findResult(null, 'link', 'A/B'), null, 'a missing payload must give null');
  assert.strictEqual(findResult({}, 'link', 'A/B'), null, 'a payload without the collection must give null');
  assert.strictEqual(findResult({ conduits: 'A/B' }, 'link', 'A/B'), null, 'a collection that is not a list must give null');
});

// ---------------------------------------------------------------------------
// 10. pickColor — a class name for every mode, every code and every edge value
// ---------------------------------------------------------------------------
group('pickColor names a class for every mode and falls back to nm-nodata', () => {
  const codes = [
    ['valid', 'nm-c-valid'],
    ['diameter_increase', 'nm-c-diameter_increase'],
    ['diameter_reduction', 'nm-c-diameter_reduction'],
    ['slope_increase', 'nm-c-slope_increase'],
    ['slope_reduction', 'nm-c-slope_reduction'],
    ['depth_increase', 'nm-c-depth_increase'],
    ['pump', 'nm-c-pump'],
    ['tank', 'nm-c-tank'],
    ['seepage_boxes', 'nm-c-seepage_boxes']
  ];
  assert.strictEqual(codes.length, 9, 'the nine classes of ACTIONS must all be covered');

  codes.forEach(pair => {
    assert.strictEqual(pickColor('recommendation', { recommendation: pair[0] }), pair[1], `${pair[0]} must paint ${pair[1]}`);
    // The payload carries the raw database value: real sessions hold "VALID" and
    // " Tank " as readily as the normalised code.
    assert.strictEqual(pickColor('recommendation', { recommendation: pair[0].toUpperCase() }), pair[1], 'the code must be folded to lower case');
    assert.strictEqual(pickColor('recommendation', { recommendation: `  ${pair[0]}  ` }), pair[1], 'the code must be trimmed');
  });

  // Never a colour, always a class: a HEX value here would be a second palette.
  codes.forEach(pair => {
    const answer = pickColor('recommendation', { recommendation: pair[0] });
    assert.ok(/^nm-[a-z]/.test(answer) && !answer.includes('#'), 'pickColor must return a class name, never a colour');
  });

  ['unknown_code', '', '-', 'toString', 'constructor'].forEach(code => {
    assert.strictEqual(pickColor('recommendation', { recommendation: code }), 'nm-nodata', `${JSON.stringify(code)} must fall back to nm-nodata`);
  });
  assert.strictEqual(pickColor('recommendation', {}), 'nm-nodata', 'a row without a recommendation must fall back');
  assert.strictEqual(pickColor('recommendation', { recommendation: null }), 'nm-nodata', 'a null recommendation must fall back');

  // validity: the number of checks the conduit passed, 0..7.
  for (let passed = 0; passed <= 7; passed += 1) {
    assert.strictEqual(pickColor('validity', { validation: passed }), `nm-v-${passed}`, `${passed} passed checks must paint nm-v-${passed}`);
  }
  [-1, 8, 3.5, NaN, Infinity, null, undefined, '5'].forEach(value => {
    assert.strictEqual(pickColor('validity', { validation: value }), 'nm-nodata', `an unusable count (${String(value)}) must fall back`);
  });
  assert.strictEqual(pickColor('validity', {}), 'nm-nodata', 'a row with no count must fall back');

  // filling: four bands and a colour of its own for a pipe running full.
  [[0, 'nm-f-0'], [24.99, 'nm-f-0'], [25, 'nm-f-1'], [49.99, 'nm-f-1'], [50, 'nm-f-2'], [74.99, 'nm-f-2'], [75, 'nm-f-3'], [99.99, 'nm-f-3'], [100, 'nm-f-4'], [180.5, 'nm-f-4']].forEach(pair => {
    assert.strictEqual(pickColor('filling', { fillPercent: pair[0] }), pair[1], `${pair[0]} % must paint ${pair[1]}`);
  });
  [null, undefined, NaN, Infinity, -1, '50', ''].forEach(value => {
    assert.strictEqual(pickColor('filling', { fillPercent: value }), 'nm-nodata', `an unusable filling (${String(value)}) must fall back`);
  });
  assert.strictEqual(pickColor('filling', {}), 'nm-nodata', 'a row with no fillPercent must fall back');

  // No row and no mode: both are normal states of the map, neither may throw.
  ['recommendation', 'validity', 'filling'].forEach(mode => {
    assert.strictEqual(pickColor(mode, null), 'nm-nodata', 'a missing row must fall back');
    assert.strictEqual(pickColor(mode, 'row'), 'nm-nodata', 'a row that is not an object must fall back');
  });
  [undefined, null, '', 'colour', 'Recommendation'].forEach(mode => {
    assert.strictEqual(pickColor(mode, { recommendation: 'valid', validation: 7, fillPercent: 10 }), 'nm-nodata', `an unknown mode (${String(mode)}) must fall back`);
  });
});

// ---------------------------------------------------------------------------
// 11. readMapPrefs — one key, and every field decided on its own
//
// Three fields after plan24: ``mode`` (plan23), ``card`` (E2) and ``labels`` (E4b).
// Every assertion here names all three, because the point of the shape is that a
// broken one may not cost the other two.  The round trip of the writer keeps its own
// group below (16), where E2 put it: no new group, the count stays what E4a left.
// ---------------------------------------------------------------------------
group('readMapPrefs reads its own key and decides every field of the preferences on its own', () => {
  const KEY = 'stormwater-network-map-preferences';

  /** The smallest object that answers getItem/setItem: no DOM is needed to test either. */
  function store(values, options) {
    const config = options || {};
    return {
      asked: [],
      values: values,
      getItem(key) {
        this.asked.push(key);
        if (config.throws) throw new Error('site data blocked');
        return Object.prototype.hasOwnProperty.call(this.values, key) ? this.values[key] : null;
      },
      setItem(key, value) {
        if (config.setThrows) throw new Error('quota exceeded');
        this.values[key] = value;
      }
    };
  }

  const good = store({ [KEY]: JSON.stringify({ mode: 'filling', labels: true }) });
  assert.deepStrictEqual(readMapPrefs(good), { mode: 'filling', card: null, labels: true }, 'the stored fields must come back, and the one that was never stored takes its default');
  assert.deepStrictEqual(good.asked, [KEY], 'the map must read its own key and nothing else');
  assert.ok(!good.asked.includes('stormwater-results-v2-preferences'), 'the map must not read the Results V2 key');
  assert.deepStrictEqual(
    readMapPrefs(store({ [KEY]: '{"mode":"validity","labels":false}' })),
    { mode: 'validity', card: null, labels: false },
    'the second mode must come back too'
  );

  // The shape every reader of the map has in their browser today, from before the card
  // had a position and before the names could be turned on: it must keep its mode and
  // take the default of the two fields it never carried.  Answering it with the whole
  // fallback would reset the mode of every existing reader.
  assert.deepStrictEqual(
    readMapPrefs(store({ [KEY]: JSON.stringify({ mode: 'filling' }) })),
    { mode: 'filling', card: null, labels: false },
    'the old {mode} shape must survive, with card and labels defaulted'
  );
  // ...and the shape E2 writes, from before the names existed: the card stands and only
  // ``labels`` defaults.
  assert.deepStrictEqual(
    readMapPrefs(store({ [KEY]: JSON.stringify({ mode: 'filling', card: { left: 40, top: 30 } }) })),
    { mode: 'filling', card: { left: 40, top: 30 }, labels: false },
    'the {mode, card} shape must survive, with labels defaulted'
  );
  // All three at once, and then each of them broken on its own: a bad mode may not cost
  // the position or the names, and a bad position may not cost the other two.
  assert.deepStrictEqual(
    readMapPrefs(store({ [KEY]: JSON.stringify({ mode: 'filling', card: { left: 40, top: 30 }, labels: true }) })),
    { mode: 'filling', card: { left: 40, top: 30 }, labels: true },
    'all three stored fields must come back'
  );
  assert.deepStrictEqual(
    readMapPrefs(store({ [KEY]: JSON.stringify({ mode: 'rainbow', card: { left: 5, top: 5 }, labels: true }) })),
    { mode: 'recommendation', card: { left: 5, top: 5 }, labels: true },
    'an unknown mode must take neither the position nor the names with it'
  );
  // ``labels`` is true and nothing else is: a string and a number are not a flag.
  assert.deepStrictEqual(
    readMapPrefs(store({ [KEY]: JSON.stringify({ labels: 'yes' }) })),
    { mode: 'recommendation', card: null, labels: false },
    'a string must not pass for true'
  );
  assert.deepStrictEqual(
    readMapPrefs(store({ [KEY]: JSON.stringify({ labels: 1 }) })),
    { mode: 'recommendation', card: null, labels: false },
    'a number must not pass for true'
  );

  // Broken, unknown, missing, absent and throwing: five ways to fail, one whole shape.
  const whole = { mode: 'recommendation', card: null, labels: false };
  assert.deepStrictEqual(readMapPrefs(store({ [KEY]: '{mode: filling' })), whole, 'broken JSON must give the whole default');
  assert.deepStrictEqual(readMapPrefs(store({ [KEY]: '"filling"' })), whole, 'a value that is not an object must give the whole default');
  assert.deepStrictEqual(readMapPrefs(store({ [KEY]: JSON.stringify({ mode: 'rainbow' }) })), whole, 'an unknown mode must give the whole default');
  assert.deepStrictEqual(readMapPrefs(store({ [KEY]: JSON.stringify({ mode: 'map' }) })), whole, 'a tab name is not a mode');
  assert.deepStrictEqual(readMapPrefs(store({})), whole, 'nothing stored must give the whole default');
  assert.deepStrictEqual(readMapPrefs(store({}, { throws: true })), whole, 'a getItem that throws must give the whole default');
  assert.deepStrictEqual(readMapPrefs(null), whole, 'no storage at all must give the whole default');
  assert.deepStrictEqual(readMapPrefs({}), whole, 'an object without getItem must give the whole default');
  assert.deepStrictEqual(readMapPrefs(store({ [KEY]: JSON.stringify({ mode: 'recommendation' }) })), whole, 'the default itself must survive a round trip');

  // Six ways for the position alone to be unusable; every one of them gives card: null
  // and leaves the mode and the names standing.
  [
    { left: NaN, top: 10 },
    { left: 10, top: -1 },
    { left: '10', top: 10 },
    { left: 10 },
    [10, 20],
    null
  ].forEach(card => {
    assert.deepStrictEqual(
      readMapPrefs(store({ [KEY]: JSON.stringify({ mode: 'filling', card: card, labels: true }) })),
      { mode: 'filling', card: null, labels: true },
      `an unusable card (${JSON.stringify(card)}) must give card: null and keep the mode and the names`
    );
  });

  // A fresh object on every path: the caller keeps what it gets, and one shared fallback
  // would let one map write into the next read.
  const first = readMapPrefs(null);
  first.mode = 'filling';
  assert.deepStrictEqual(readMapPrefs(null), whole, 'the default must be a new object every time');
});

// ---------------------------------------------------------------------------
// 12. nextLegendState — a toggle that gives back a new array every time
// ---------------------------------------------------------------------------
group('nextLegendState toggles, clears on null and never mutates its argument', () => {
  const empty = [];
  const one = nextLegendState(empty, 'nm-c-valid');
  assert.deepStrictEqual(one, ['nm-c-valid'], 'the first click must dim the category');
  assert.deepStrictEqual(empty, [], 'the argument must not be mutated');

  const two = nextLegendState(one, 'nm-c-pump');
  assert.deepStrictEqual(two, ['nm-c-valid', 'nm-c-pump'], 'a second category must join the first');
  assert.deepStrictEqual(one, ['nm-c-valid'], 'the argument must still not be mutated');

  // The second click on the same entry restores it, and only it.
  assert.deepStrictEqual(nextLegendState(two, 'nm-c-valid'), ['nm-c-pump'], 'the second click must restore the category');
  assert.deepStrictEqual(nextLegendState(nextLegendState(two, 'nm-c-valid'), 'nm-c-valid'), ['nm-c-pump', 'nm-c-valid'], 'a third click must dim it again');

  // Esc sends null: everything comes back at once.
  assert.deepStrictEqual(nextLegendState(two, null), [], 'null must clear every dimmed category');
  assert.deepStrictEqual(nextLegendState(two, undefined), [], 'undefined must clear as well');
  assert.deepStrictEqual(two, ['nm-c-valid', 'nm-c-pump'], 'clearing must not mutate the argument either');

  // Anything that is not a list of keys reads as "nothing is dimmed".
  assert.deepStrictEqual(nextLegendState(null, 'nm-v-3'), ['nm-v-3'], 'a missing state must start empty');
  assert.deepStrictEqual(nextLegendState('nm-v-3', 'nm-v-3'), ['nm-v-3'], 'a state that is not an array must start empty');
  assert.deepStrictEqual(nextLegendState([1, {}, 'nm-f-4'], 'nm-f-0'), ['nm-f-4', 'nm-f-0'], 'only string keys survive');
  assert.deepStrictEqual(nextLegendState(['nm-nodata'], 'nm-nodata'), [], 'the grey category toggles like any other');
});

// ---------------------------------------------------------------------------
// 13. dimSet — one collection, the names the table is not showing
// ---------------------------------------------------------------------------
group('dimSet fades what one filtered table left out and touches no other collection', () => {
  // A node and a subcatchment deliberately carry the name of a link: SWMM allows it,
  // and a set built across the collections would take the conduit filter as licence
  // to fade them.  Indices overlap between the collections for the same reason - an
  // answer of [1] means nothing unless it is known which list it indexes.
  const collections = {
    link: [{ n: 'C1', i: 0 }, { n: 'C2', i: 1 }, { n: 'A/B', i: 2 }],
    node: [{ n: 'C2', i: 0 }, { n: 'N1', i: 1 }],
    poly: [{ n: 'C1', i: 0 }, { n: 'S 1', i: 1 }]
  };

  assert.deepStrictEqual(dimSet('conduits', ['C1'], collections), [1, 2], 'every link the table does not list must fade');
  assert.deepStrictEqual(dimSet('conduits', ['C1', 'C2', 'A/B'], collections), [], 'an unfiltered table must fade nothing');
  assert.deepStrictEqual(dimSet('conduits', [], collections), [0, 1, 2], 'a table showing nothing must fade the whole collection');

  // The same call, read against the other two lists: the answer indexes links only.
  assert.deepStrictEqual(dimSet('nodes', ['C2'], collections), [1], 'the node table must speak for the nodes');
  assert.deepStrictEqual(dimSet('catchments', ['C1'], collections), [1], 'the catchment table must speak for the catchments');
  assert.deepStrictEqual(dimSet('nodes', ['N1'], collections), [0], 'a node named like a link must be judged by the node table');

  // Names are compared as they stand: no trimming, no case folding, no unescaping.
  const odd = { link: [{ n: 'P 1', i: 0 }, { n: 's#1', i: 1 }, { n: '"x"', i: 2 }, { n: 'a/b', i: 3 }] };
  assert.deepStrictEqual(dimSet('conduits', ['P 1', '"x"'], odd), [1, 3], 'a name with a space or a quote must match itself');
  assert.deepStrictEqual(dimSet('conduits', ['p 1', ' s#1', 'A/B'], odd), [0, 1, 2, 3], 'a near miss is not a match');

  // A number that arrived as a name still matches the drawn element: the payload
  // carries ``n`` as the model wrote it, and a purely numeric name is legal.
  const numeric = { link: [{ n: 12, i: 0 }, { n: '13', i: 1 }] };
  assert.deepStrictEqual(dimSet('conduits', ['12'], numeric), [1], 'a numeric name must compare as its text');

  // Everything that cannot be understood fades nothing at all.
  [undefined, null, '', 'link', 'map', 'info', 'toString', 'constructor', 0, {}].forEach(tab => {
    assert.deepStrictEqual(dimSet(tab, [], collections), [], `an unusable tab (${String(tab)}) must fade nothing`);
  });
  [undefined, null, 'C1', 0, {}].forEach(visible => {
    assert.deepStrictEqual(dimSet('conduits', visible, collections), [], `an unusable visible list (${String(visible)}) must fade nothing`);
  });
  [undefined, null, 'links', 5].forEach(bad => {
    assert.deepStrictEqual(dimSet('conduits', [], bad), [], 'an unusable payload must fade nothing');
  });
  assert.deepStrictEqual(dimSet('conduits', [], {}), [], 'a payload without that collection must fade nothing');
  assert.deepStrictEqual(dimSet('conduits', [], { link: 'C1' }), [], 'a collection that is not a list must fade nothing');

  // A drawn element without a usable index cannot be addressed and is skipped rather
  // than returned as undefined, which setDim would silently turn into a no-op.
  const broken = { link: [null, 'C1', { n: 'C2' }, { n: 'C3', i: 1.5 }, { n: 'C4', i: '2' }, { n: 'C5', i: 3 }] };
  assert.deepStrictEqual(dimSet('conduits', [], broken), [3], 'only an element with an integer index may be faded');

  // The argument is never touched: the caller keeps the geometry payload itself here.
  const before = JSON.stringify(collections);
  dimSet('conduits', ['C1'], collections);
  assert.strictEqual(JSON.stringify(collections), before, 'dimSet must not mutate the collections it reads');
});

// ---------------------------------------------------------------------------
// 14. analysedElements — what may be offered to dimSet in the first place
// ---------------------------------------------------------------------------
group('analysedElements keeps the filter away from elements the table never listed', () => {
  // The drawn links of a real model: four conduits the analysis produced rows for and
  // a pump, a weir and an outlet it did not.  The same holds for a [DIVIDERS] node and
  // for the outline of a storage unit, which is why the helper takes the predicate
  // rather than a kind.
  const links = [
    { n: 'C1', i: 0 },
    { n: 'PUMP1', i: 1 },
    { n: 'C2', i: 2 },
    { n: 'W1', i: 3 },
    { n: 'OUT1', i: 4 }
  ];
  const rows = new Set(['C1', 'C2']);
  const isKnown = name => rows.has(name);

  assert.deepStrictEqual(
    analysedElements(links, isKnown).map(element => element.n),
    ['C1', 'C2'],
    'only the elements the results collection carries a row for may be dimmed'
  );

  // The whole point, end to end: the table renders unfiltered and lists every conduit
  // it has.  Fed the drawn collection, dimSet answers with the pump, the weir and the
  // outlet — which the filter would fade to 15 % and make unclickable although nobody
  // filtered anything.  Fed the narrowed one, it answers with nothing.
  const visible = ['C1', 'C2'];
  assert.deepStrictEqual(dimSet('conduits', visible, { link: links }), [1, 3, 4], 'the unnarrowed collection is the bug');
  assert.deepStrictEqual(
    dimSet('conduits', visible, { link: analysedElements(links, isKnown) }),
    [],
    'an unfiltered table must fade nothing, whatever else is drawn'
  );

  // ...and a real filter still dims, on the very same narrowed collection.
  assert.deepStrictEqual(
    dimSet('conduits', ['C1'], { link: analysedElements(links, isKnown) }),
    [2],
    'a conduit with a row that the filter left out must still fade'
  );

  // Indices are the ones the payload carries, not the positions in the narrowed list:
  // setDim addresses els[kind][i] with them.
  assert.deepStrictEqual(analysedElements(links, isKnown).map(element => element.i), [0, 2], 'the payload index must survive');

  // Names are compared as text, as everywhere else in the module.
  assert.deepStrictEqual(
    analysedElements([{ n: 12, i: 0 }, { n: '13', i: 1 }], name => name === '12').map(element => element.i),
    [0],
    'a numeric name must be offered to the predicate as its text'
  );

  // A predicate that answers anything but true is a no, and anything unusable gives an
  // empty collection — which fades nothing, the safe reading everywhere in this module.
  assert.deepStrictEqual(analysedElements(links, () => 1).length, 0, 'only a literal true may pass an element through');
  [undefined, null, 'links', 5, {}].forEach(bad => {
    assert.deepStrictEqual(analysedElements(bad, isKnown), [], `an unusable collection (${String(bad)}) must give nothing`);
  });
  [undefined, null, 'x', {}].forEach(bad => {
    assert.deepStrictEqual(analysedElements(links, bad), [], 'an unusable predicate must give nothing');
  });
  assert.deepStrictEqual(analysedElements([null, 'C1', { n: 'C2', i: 0 }], () => true).length, 1, 'only real records survive');

  // The payload itself is never touched.
  const untouched = JSON.stringify(links);
  analysedElements(links, isKnown);
  assert.strictEqual(JSON.stringify(links), untouched, 'analysedElements must not mutate the collection it reads');
});

// ---------------------------------------------------------------------------
// 15. clampCardPos - the corner the dragged card may really take
// ---------------------------------------------------------------------------
group('clampCardPos keeps the card inside the canvas and refuses what it cannot compute', () => {
  assert.deepStrictEqual(clampCardPos(40, 30, 200, 120, 800, 500), { left: 40, top: 30 }, 'a corner inside the canvas is kept as it is');
  assert.deepStrictEqual(clampCardPos(-50, -10, 200, 120, 800, 500), { left: 0, top: 0 }, 'a negative corner clamps to the top left');
  assert.deepStrictEqual(clampCardPos(9999, 9999, 200, 120, 800, 500), { left: 600, top: 380 }, 'a corner past the canvas clamps to the far edge');
  assert.deepStrictEqual(clampCardPos(600, 380, 200, 120, 800, 500), { left: 600, top: 380 }, 'exactly against the edge is still inside');
  // A card larger than the canvas sticks to the left/top edge instead of walking off
  // into negative coordinates: the arithmetic can move it, only the stylesheet can
  // make it smaller.
  assert.deepStrictEqual(clampCardPos(50, 50, 900, 600, 800, 500), { left: 0, top: 0 }, 'a card larger than the canvas sticks to the top left');

  const answer = clampCardPos(40, 30, 200, 120, 800, 500);
  assert.deepStrictEqual(Object.keys(answer), ['left', 'top'], 'the answer carries exactly left and top');
  assert.ok(Number.isFinite(answer.left) && Number.isFinite(answer.top), 'neither coordinate may be NaN');

  // Every argument, one at a time, in each of the four unusable forms.
  const usable = [40, 30, 200, 120, 800, 500];
  [NaN, Infinity, undefined, '10'].forEach(bad => {
    for (let index = 0; index < usable.length; index += 1) {
      const args = usable.slice();
      args[index] = bad;
      assert.strictEqual(clampCardPos.apply(null, args), null, `argument ${index} = ${String(bad)} must give null`);
    }
  });

  // A size of zero or less, of the canvas and of the card alike.  The card is the case
  // that happens in production: card.offsetWidth is 0 while the map panel is still
  // hidden, and a corner computed then would be written over the position the reader
  // dragged the card to.
  [[0, 120, 800, 500], [200, 0, 800, 500], [-1, 120, 800, 500], [200, -1, 800, 500],
   [200, 120, 0, 500], [200, 120, 800, 0], [200, 120, -1, 500], [200, 120, 800, -1]].forEach(sizes => {
    assert.strictEqual(clampCardPos(40, 30, sizes[0], sizes[1], sizes[2], sizes[3]), null, `sizes ${JSON.stringify(sizes)} must give null`);
  });
});

// ---------------------------------------------------------------------------
// 16. writeMapPrefs - the whole object goes in, and comes back out
// ---------------------------------------------------------------------------
group('writeMapPrefs stores the whole object and survives a round trip', () => {
  const KEY = 'stormwater-network-map-preferences';

  /** A storage that really keeps what it is given, so the two functions can meet. */
  function memory(options) {
    const config = options || {};
    return {
      values: {},
      setItem(key, value) {
        if (config.throws) throw new Error('quota exceeded');
        this.values[key] = value;
      },
      getItem(key) {
        return Object.prototype.hasOwnProperty.call(this.values, key) ? this.values[key] : null;
      }
    };
  }

  const store = memory();
  const written = { mode: 'filling', card: { left: 40, top: 30 }, labels: true };
  writeMapPrefs(store, written);
  assert.deepStrictEqual(Object.keys(store.values), [KEY], 'the writer must use the map key and no other');
  assert.deepStrictEqual(readMapPrefs(store), written, 'all three fields must survive the round trip');

  // The mode is validated on the way out, not on the way in: a value the switch does
  // not know lands on the default and the two fields beside it are untouched.
  const unknown = memory();
  writeMapPrefs(unknown, { mode: 'rainbow', card: { left: 5, top: 5 }, labels: true });
  assert.deepStrictEqual(readMapPrefs(unknown), { mode: 'recommendation', card: { left: 5, top: 5 }, labels: true }, 'a mode off the list must read back as the default');

  const noCard = memory();
  writeMapPrefs(noCard, { mode: 'validity', card: null, labels: false });
  assert.deepStrictEqual(readMapPrefs(noCard), { mode: 'validity', card: null, labels: false }, 'a card that was never dragged must round-trip as null');

  // A field the caller forgot is written as its default, never as undefined: the
  // stored JSON has to stay a whole object or the next read would have to guess.
  const partial = memory();
  writeMapPrefs(partial, { mode: 'validity' });
  assert.deepStrictEqual(readMapPrefs(partial), { mode: 'validity', card: null, labels: false }, 'a missing field is written as its default');

  // A preference is not worth an error: neither a storage that refuses to write nor a
  // missing one may throw out of the module.
  assert.doesNotThrow(() => writeMapPrefs(memory({ throws: true }), written), 'a setItem that throws must cost the preference and nothing else');
  assert.doesNotThrow(() => writeMapPrefs(null, written), 'no storage at all must be survivable');
  assert.doesNotThrow(() => writeMapPrefs({}, written), 'an object without setItem must be survivable');
  assert.doesNotThrow(() => writeMapPrefs(memory(), null), 'no preferences at all must be survivable');
});


// ---------------------------------------------------------------------------
// placeLabels — the placement solver (plan24, stage E4a).
//
// Everything below tests the pure solver: no DOM, no measurement, no randomness.
// The reference implementation that group 3 compares the bucket grid against lives
// here on purpose — an O(n^2) loop over every obstacle is the only thing that can
// say whether the acceleration structure lost a candidate, and keeping it beside
// the module keeps the two from sharing a bug.
// ---------------------------------------------------------------------------

/** Mirrors LABEL_POLY_RING_PX; asserted literally so a change in the module shows up here. */
const RING = 10;

/** Deterministic pseudo-random numbers: a seeded LCG, never Math.random. */
function lcg(seed) {
  let state = seed >>> 0;
  return function () {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function refPointRect(rect, x, y) {
  const dx = Math.max(rect.x1 - x, 0, x - rect.x2);
  const dy = Math.max(rect.y1 - y, 0, y - rect.y2);
  return Math.hypot(dx, dy);
}

function refPointSeg(x, y, seg) {
  const dx = seg.x2 - seg.x1;
  const dy = seg.y2 - seg.y1;
  const len2 = dx * dx + dy * dy;
  let t = len2 > 0 ? ((x - seg.x1) * dx + (y - seg.y1) * dy) / len2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.hypot(x - (seg.x1 + t * dx), y - (seg.y1 + t * dy));
}

/** The part of a segment inside the frame, written independently of the module's clipper. */
function refClip(seg, clip) {
  if (!clip) return seg;
  const dx = seg.x2 - seg.x1;
  const dy = seg.y2 - seg.y1;
  let lo = 0;
  let hi = 1;
  const axes = [[seg.x1, dx, clip.x1, clip.x2], [seg.y1, dy, clip.y1, clip.y2]];
  for (let n = 0; n < axes.length; n += 1) {
    const start = axes[n][0];
    const delta = axes[n][1];
    const low = axes[n][2];
    const high = axes[n][3];
    if (delta === 0) {
      if (start < low || start > high) return null;
      continue;
    }
    let a = (low - start) / delta;
    let b = (high - start) / delta;
    if (a > b) {
      const swap = a;
      a = b;
      b = swap;
    }
    if (a > lo) lo = a;
    if (b < hi) hi = b;
  }
  if (lo > hi) return null;
  return { x1: seg.x1 + lo * dx, y1: seg.y1 + lo * dy, x2: seg.x1 + hi * dx, y2: seg.y1 + hi * dy, hw: seg.hw, key: seg.key };
}

function refSegRect(rect, seg) {
  if (refClip(seg, rect)) return 0;
  let best = Math.min(refPointRect(rect, seg.x1, seg.y1), refPointRect(rect, seg.x2, seg.y2));
  const corners = [[rect.x1, rect.y1], [rect.x2, rect.y1], [rect.x2, rect.y2], [rect.x1, rect.y2]];
  for (let n = 0; n < corners.length; n += 1) best = Math.min(best, refPointSeg(corners[n][0], corners[n][1], seg));
  return best;
}

const REF_EPS = 1e-6;
const REF_ANCHORS = ['E', 'W', 'N', 'S', 'NE', 'SE', 'NW', 'SW'];

function refOffset(name, reach, w, h) {
  switch (name) {
    case 'C': return { dx: -w / 2, dy: -h / 2 };
    case 'E': return { dx: reach, dy: -h / 2 };
    case 'W': return { dx: -reach - w, dy: -h / 2 };
    case 'N': return { dx: -w / 2, dy: -reach - h };
    case 'S': return { dx: -w / 2, dy: reach };
    case 'NE': return { dx: reach, dy: -reach - h };
    case 'SE': return { dx: reach, dy: reach };
    case 'NW': return { dx: -reach - w, dy: -reach - h };
    case 'SW': return { dx: -reach - w, dy: reach };
    default: return null;
  }
}

/**
 * The same contract, solved by brute force: every candidate against every obstacle.
 * Only the acceleration structure differs, so any disagreement with placeLabels is a
 * hole in the bucket grid.
 */
function refPlace(items, segments, opts) {
  const options = opts || {};
  const max = Number.isFinite(options.max) ? Math.floor(options.max) : 600;
  const margin = Number.isFinite(options.margin) && options.margin > 0 ? options.margin : 0;
  const selected = typeof options.selected === 'string' ? options.selected : '';
  const clip = options.clip || null;
  const anchors = Array.isArray(options.anchors) && options.anchors.length ? options.anchors : REF_ANCHORS;
  const plainAnchors = anchors.filter(name => name !== 'C');
  const polyAnchors = ['C'].concat(plainAnchors);

  const list = items.slice().map(item => Object.assign({}, item, { r: item.r > 0 ? item.r : 0 }));
  const rank = kind => (kind === 'node' ? 0 : kind === 'link' ? 1 : 2);
  list.sort((a, b) => (rank(a.kind) - rank(b.kind)) || (a.i - b.i));

  const disks = list.filter(item => item.kind === 'node' && item.r > 0);
  const segs = (segments || [])
    .filter(seg => seg.hw > 0)
    .map(seg => refClip(seg, clip))
    .filter(seg => seg !== null);

  const queue = list.slice(0, max);
  if (selected) {
    const chosen = list.filter(item => item.key === selected)[0];
    if (chosen) {
      const at = queue.indexOf(chosen);
      if (at >= 0) queue.splice(at, 1);
      queue.unshift(chosen);
    }
  }

  const rects = [];
  const out = [];
  queue.forEach(item => {
    const poly = item.kind === 'poly';
    const reach = poly ? RING : item.r + margin;
    const names = poly ? polyAnchors : plainAnchors;
    for (let a = 0; a < names.length; a += 1) {
      const offset = refOffset(names[a], reach, item.w, item.h);
      if (!offset) continue;
      const rect = {
        x1: item.x + offset.dx,
        y1: item.y + offset.dy,
        x2: item.x + offset.dx + item.w,
        y2: item.y + offset.dy + item.h
      };
      const hitDisk = disks.some(disk => disk.key !== item.key && refPointRect(rect, disk.x, disk.y) < disk.r - REF_EPS);
      if (hitDisk) continue;
      const hitSeg = segs.some(seg => refSegRect(rect, seg) < seg.hw - REF_EPS);
      if (hitSeg) continue;
      const hitRect = rects.some(
        other =>
          rect.x1 < other.x2 - REF_EPS &&
          other.x1 < rect.x2 - REF_EPS &&
          rect.y1 < other.y2 - REF_EPS &&
          other.y1 < rect.y2 - REF_EPS
      );
      if (hitRect) continue;
      rects.push(rect);
      out.push({ key: item.key, dx: offset.dx, dy: offset.dy });
      return;
    }
  });
  return out;
}

/** Runs both implementations on the same input and returns the answer they agree on. */
function agree(what, items, segments, opts) {
  const fast = placeLabels(items, segments, opts);
  const slow = refPlace(items, segments, opts);
  assert.deepStrictEqual(fast, slow, `${what}: the bucket grid must answer exactly what brute force does`);
  return fast;
}

/** The rectangle a returned entry describes, for the assertions that measure it. */
function rectOf(item, entry) {
  return { x1: item.x + entry.dx, y1: item.y + entry.dy, x2: item.x + entry.dx + item.w, y2: item.y + entry.dy + item.h };
}

function entryFor(result, key) {
  return result.filter(entry => entry.key === key)[0];
}

/**
 * A thousand elements two hundred pixels apart: far enough that every one of them
 * takes its first anchor, so a cap can be read off the answer without a collision
 * muddying it.  Handed back shuffled, because the input order must mean nothing.
 */
function spacedScene() {
  const items = [];
  for (let n = 0; n < 1000; n += 1) {
    const kind = n < 500 ? 'node' : 'link';
    const i = n < 500 ? n : n - 500;
    items.push({ key: `${kind}:${i}`, kind: kind, i: i, x: (n % 40) * 200, y: Math.floor(n / 40) * 200, r: 0, w: 50, h: 16 });
  }
  const random = lcg(31337);
  const shuffled = items.slice();
  for (let n = shuffled.length - 1; n > 0; n -= 1) {
    const m = Math.floor(random() * (n + 1));
    const swap = shuffled[n];
    shuffled[n] = shuffled[m];
    shuffled[m] = swap;
  }
  return { items: items, shuffled: shuffled };
}

/** A scene of 200 elements and 200 conduits, drawn from a seeded LCG. */
function randomScene(seed, spread) {
  const random = lcg(seed);
  const items = [];
  const segments = [];
  for (let n = 0; n < 200; n += 1) {
    const kind = n % 5 === 0 ? 'poly' : n % 2 === 0 ? 'node' : 'link';
    const hw = 1.25 + random() * 3;
    items.push({
      key: `${kind}:${n}`,
      kind: kind,
      i: n,
      x: random() * spread,
      y: random() * (spread * 0.66),
      r: kind === 'node' ? 2 + random() * 8 : kind === 'link' ? hw : 0,
      w: 20 + random() * 70,
      h: 16.2
    });
  }
  for (let n = 0; n < 200; n += 1) {
    const x = random() * spread;
    const y = random() * (spread * 0.66);
    segments.push({
      key: `link:${n}`,
      x1: x,
      y1: y,
      x2: x + (random() - 0.5) * spread,
      y2: y + (random() - 0.5) * spread,
      hw: 1.25 + random() * 3
    });
  }
  return { items: items, segments: segments };
}

// -- 15 -----------------------------------------------------------------------
group('placeLabels never puts a name on an obstacle', () => {
  // Two nodes ten pixels apart and names wider than the gap: whatever comes back,
  // it stands clear of the other symbol.
  const items = [
    { key: 'node:0', kind: 'node', i: 0, x: 0, y: 0, r: 4, w: 60, h: 16 },
    { key: 'node:1', kind: 'node', i: 1, x: 10, y: 0, r: 4, w: 60, h: 16 }
  ];
  const segments = [{ key: 'link:0', x1: -200, y1: 60, x2: 200, y2: 60, hw: 3 }];
  const result = agree('two crowded nodes', items, segments, {});
  assert.strictEqual(result.length, 2, 'both names must find somewhere to stand');
  result.forEach(entry => {
    const own = items.filter(item => item.key === entry.key)[0];
    const rect = rectOf(own, entry);
    items.forEach(other => {
      if (other.key === own.key) return;
      assert.ok(refPointRect(rect, other.x, other.y) >= other.r - REF_EPS, `${entry.key} must clear the disk of ${other.key}`);
    });
    segments.forEach(seg => {
      assert.ok(refSegRect(rect, seg) >= seg.hw - REF_EPS, `${entry.key} must clear the conduit ${seg.key}`);
    });
  });

  // R01: the width is MEASURED and handed in, never guessed from the string length.
  // 'WWWWWWWW' at 11 px measures 85 px with the halo; the deleted formula
  // length * 0.6 * fontPx said 52.8.  The obstacle sits at 60 px — inside the real
  // rectangle, outside the guessed one — so on the old arithmetic this is green
  // with the name lying on the symbol, and that is the whole point of the case.
  const wide = [
    { key: 'node:0', kind: 'node', i: 0, x: 0, y: 0, r: 0, w: 85, h: 16.2 },
    { key: 'node:1', kind: 'node', i: 1, x: 60, y: 0, r: 1, w: 0, h: 0 }
  ];
  const measured = agree('a measured width', wide, [], {});
  const first = entryFor(measured, 'node:0');
  assert.ok(first, 'the wide name must still find a position');
  assert.strictEqual(first.dx, -85, 'E is blocked at the measured width, so W wins');
});

// -- 16 -----------------------------------------------------------------------
group('placeLabels is deterministic', () => {
  const scene = randomScene(20260922, 1200);
  const opts = { clip: { x1: 0, y1: 0, x2: 1200, y2: 800 } };
  const once = placeLabels(JSON.parse(JSON.stringify(scene.items)), JSON.parse(JSON.stringify(scene.segments)), opts);
  const twice = placeLabels(JSON.parse(JSON.stringify(scene.items)), JSON.parse(JSON.stringify(scene.segments)), opts);
  assert.deepStrictEqual(once, twice, 'two runs on the same input must answer identically');
  assert.ok(once.length > 0, 'the scene must actually place something');

  // The input order carries no meaning: the function sorts for itself.
  const random = lcg(7);
  const shuffled = scene.items.slice();
  for (let n = shuffled.length - 1; n > 0; n -= 1) {
    const m = Math.floor(random() * (n + 1));
    const swap = shuffled[n];
    shuffled[n] = shuffled[m];
    shuffled[m] = swap;
  }
  assert.deepStrictEqual(placeLabels(shuffled, scene.segments, opts), once, 'a shuffled input must give the same answer');

  // Nothing handed in is written to.
  const untouched = JSON.stringify(scene.items);
  placeLabels(scene.items, scene.segments, opts);
  assert.strictEqual(JSON.stringify(scene.items), untouched, 'placeLabels must not mutate what it reads');
});

// -- 17 -----------------------------------------------------------------------
group('placeLabels with a bucket grid agrees with brute force', () => {
  const scene = randomScene(20260922, 1200);
  const answer = agree('200 elements and 200 conduits', scene.items, scene.segments, { clip: { x1: 0, y1: 0, x2: 1200, y2: 800 } });
  // A crowded frame: some names stand, some have nowhere to go — a scene where
  // everything or nothing fitted would compare the two implementations on nothing.
  assert.ok(answer.length > 20, `the scene must place a realistic number of names, got ${answer.length}`);
  assert.ok(answer.length < 200, `the scene must also crowd some names out, got ${answer.length}`);

  // (a) A collector with BOTH ends outside the frame, crossing it.  The naive
  // "drop it when both ends are outside" loses it, and clipping it away entirely
  // would be just as wrong — so one name has to be blocked by it and one, past the
  // clipped end, has to be free.
  const clip = { x1: 0, y1: 0, x2: 200, y2: 200 };
  const crossing = [{ key: 'link:9', x1: -500, y1: 100, x2: 700, y2: 100, hw: 5 }];
  const across = [
    { key: 'node:0', kind: 'node', i: 0, x: 100, y: 100, r: 6, w: 50, h: 16 },
    { key: 'node:1', kind: 'node', i: 1, x: 300, y: 100, r: 6, w: 50, h: 16 }
  ];
  const clipped = agree('a segment crossing the frame', across, crossing, { clip: clip });
  assert.deepStrictEqual(entryFor(clipped, 'node:0'), { key: 'node:0', dx: -25, dy: -22 }, 'inside the frame the conduit must still block E and W');
  assert.deepStrictEqual(entryFor(clipped, 'node:1'), { key: 'node:1', dx: 6, dy: -8 }, 'past the clipped end the conduit must be gone');

  // (b) A stroke thicker than a bucket, crossing a bucket boundary: the name stands
  // two buckets away from the centre line and must still see it.
  const thick = [{ key: 'link:0', x1: 0, y1: 64, x2: 400, y2: 64, hw: 40 }];
  const near = [{ key: 'node:0', kind: 'node', i: 0, x: 200, y: 110, r: 0, w: 50, h: 16 }];
  const fat = agree('a stroke thicker than a bucket', near, thick, {});
  assert.deepStrictEqual(fat, [{ key: 'node:0', dx: -25, dy: 0 }], 'E, W and N lie inside the thick outline, S does not');

  // (c) The disk of a selected storage, sized by the formula of E4b (N09):
  // sqrt(2) * 3.5 + 3 + 2 = 9.9497.  The rectangle (4.8, 4.8)-(54.8, 21) is 6.788
  // away from the centre, so it passed the old 6.5 disk while crossing the corner
  // of the drawn square (7.071 away).  On the new radius it has to fail.
  const storage = Math.SQRT2 * 3.5 + 3 + 2;
  const corner = [
    { key: 'node:0', kind: 'node', i: 0, x: 4.8, y: 12.9, r: 0, w: 50, h: 16.2 },
    { key: 'node:1', kind: 'node', i: 1, x: 0, y: 0, r: 6.5, w: 0, h: 0 }
  ];
  const near6 = agree('the old 6.5 disk', corner, [], {});
  assert.strictEqual(entryFor(near6, 'node:0').dx, 0, 'on the old disk the rectangle passed at E');
  corner[1].r = storage;
  const near10 = agree('the disk of a selected storage', corner, [], {});
  const moved = entryFor(near10, 'node:0');
  assert.deepStrictEqual(moved, { key: 'node:0', dx: -25, dy: 0 }, 'E, W and N now cross the disk, S does not');
  assert.ok(refPointRect(rectOf(corner[0], moved), 0, 0) >= storage - REF_EPS, 'the winning rectangle must clear the disk');

  // (d) High zoom: hundreds of pixels between elements and conduits longer than the frame.
  const zoomed = randomScene(4242, 9000);
  agree('a deeply zoomed view', zoomed.items, zoomed.segments, { clip: { x1: 0, y1: 0, x2: 1440, y2: 900 } });
});

// -- 18 -----------------------------------------------------------------------
group('placeLabels hides a name with nowhere to go', () => {
  // One element and an obstacle in the middle of each of its eight candidates.
  const item = { key: 'node:0', kind: 'node', i: 0, x: 0, y: 0, r: 0, w: 40, h: 16 };
  const items = [item];
  [[20, 0], [-20, 0], [0, -8], [0, 8], [20, -8], [20, 8], [-20, -8], [-20, 8]].forEach((point, n) => {
    items.push({ key: `node:${n + 1}`, kind: 'node', i: n + 1, x: point[0], y: point[1], r: 2, w: 0, h: 0 });
  });
  const result = agree('eight blocked anchors', items, [], {});
  assert.strictEqual(entryFor(result, 'node:0'), undefined, 'an element with no free anchor must be absent, never forced');
});

// -- 19 -----------------------------------------------------------------------
group('placeLabels skips its own node but not its own conduit', () => {
  const lone = [{ key: 'node:0', kind: 'node', i: 0, x: 0, y: 0, r: 5, w: 50, h: 16 }];
  assert.deepStrictEqual(agree('a lone node', lone, [], {}), [{ key: 'node:0', dx: 5, dy: -8 }], 'a lone node takes the first anchor, E');

  // A conduit carries r = hw, the very number its segment carries.  With r = 0 the
  // candidates would start on the centre line and collide with the pipe in all eight
  // anchors, and the name would never be drawn — which is why an empty answer here
  // is a failure, not a detail.
  const flat = [{ key: 'link:0', kind: 'link', i: 0, x: 0, y: 0, r: 3, w: 50, h: 16 }];
  const flatSeg = [{ key: 'link:0', x1: -100, y1: 0, x2: 100, y2: 0, hw: 3 }];
  const horizontal = agree('a lone horizontal conduit', flat, flatSeg, {});
  assert.strictEqual(horizontal.length, 1, 'a lone conduit must get a name');
  assert.deepStrictEqual(horizontal[0], { key: 'link:0', dx: -25, dy: -19 }, 'E and W lie on the pipe, so N wins');
  assert.ok(refSegRect(rectOf(flat[0], horizontal[0]), flatSeg[0]) >= 3 - REF_EPS, 'the name must not sit on its own pipe');

  const upright = [{ key: 'link:0', kind: 'link', i: 0, x: 0, y: 0, r: 3, w: 50, h: 16 }];
  const uprightSeg = [{ key: 'link:0', x1: 0, y1: -100, x2: 0, y2: 100, hw: 3 }];
  assert.deepStrictEqual(agree('a lone vertical conduit', upright, uprightSeg, {}), [{ key: 'link:0', dx: 3, dy: -8 }], 'a vertical pipe leaves E free');
});

// -- 20 -----------------------------------------------------------------------
group('placeLabels respects opts.max', () => {
  const scene = spacedScene();
  const capped = agree('a cap of 600', scene.shuffled, [], { max: 600 });
  assert.strictEqual(capped.length, 600, 'the cap is a cap, whatever order the input came in');
  const expected = [];
  for (let n = 0; n < 500; n += 1) expected.push(`node:${n}`);
  for (let n = 0; n < 100; n += 1) expected.push(`link:${n}`);
  assert.deepStrictEqual(capped.map(entry => entry.key), expected, 'the cap keeps the head of the comparator, not the head of the input');
});

// -- 21 -----------------------------------------------------------------------
group('placeLabels lifts opts.selected out of opts.max', () => {
  const scene = spacedScene();
  const plain = agree('a cap of 600', scene.shuffled, [], { max: 600 });

  const lifted = agree('a selection past the cap', scene.shuffled, [], { max: 600, selected: 'link:499' });
  assert.strictEqual(lifted.length, 601, 'the selected element is exempt from the cap');
  assert.strictEqual(lifted[0].key, 'link:499', 'the selected element comes first');
  assert.deepStrictEqual(lifted.slice(1), plain, 'the selection must push nobody out');

  const inside = agree('a selection already inside the cap', scene.shuffled, [], { max: 600, selected: 'node:5' });
  assert.strictEqual(inside.length, 600, 'a selection that already fitted does not add an entry');
  assert.strictEqual(inside[0].key, 'node:5', 'it only moves to the front');
  assert.deepStrictEqual(inside.slice(1), plain.filter(entry => entry.key !== 'node:5'), 'and nothing else moves');

  const nobody = agree('a selection matching nothing', scene.shuffled, [], { max: 600, selected: 'node:99999' });
  assert.deepStrictEqual(nobody, plain, 'a selection matching no key is ignored without complaint');
});

// -- 22 -----------------------------------------------------------------------
group('placeLabels puts a subcatchment name on its centroid', () => {
  const lone = [{ key: 'poly:0', kind: 'poly', i: 0, x: 0, y: 0, r: 0, w: 50, h: 16 }];
  assert.deepStrictEqual(agree('a lone subcatchment', lone, [], {}), [{ key: 'poly:0', dx: -25, dy: -8 }], 'C centres the rectangle on the centroid');

  // A node sitting exactly in the centroid.  C falls, and the answer still exists
  // because the other eight are measured from a ring, not from the point: on the
  // geometry of v1.3 all nine collided and no subcatchment could be named at all.
  // The node carries no name of its own here (w = h = 0) so that the only obstacle
  // under test is its disk.
  const shared = [
    { key: 'poly:0', kind: 'poly', i: 0, x: 0, y: 0, r: 0, w: 50, h: 16 },
    { key: 'node:0', kind: 'node', i: 0, x: 0, y: 0, r: 5.5, w: 0, h: 0 }
  ];
  const ringed = entryFor(agree('a node in the centroid', shared, [], {}), 'poly:0');
  assert.ok(ringed, 'a subcatchment with a node in its centroid must still be named');
  assert.strictEqual(ringed.dx, RING, 'the fallback anchors start on the ring, not on the point');
  assert.strictEqual(ringed.dy, -8, 'and E keeps the rectangle centred vertically');
  assert.ok(refPointRect(rectOf(shared[0], ringed), 0, 0) >= 5.5 - REF_EPS, 'the ring anchor must clear the disk');

  // The ninth anchor belongs to subcatchments alone: the same element drawn as a node
  // takes E, never C, because C would land on its own symbol.
  const asNode = [{ key: 'node:0', kind: 'node', i: 0, x: 0, y: 0, r: 0, w: 50, h: 16 }];
  assert.deepStrictEqual(agree('the same element as a node', asNode, [], {}), [{ key: 'node:0', dx: 0, dy: -8 }], 'a node never centres its name on itself');

  // Under a cap the subcatchments yield to both other collections (kindRank poly = 2).
  const mixed = [
    { key: 'poly:0', kind: 'poly', i: 0, x: 0, y: 0, r: 0, w: 50, h: 16 },
    { key: 'link:0', kind: 'link', i: 0, x: 400, y: 0, r: 0, w: 50, h: 16 },
    { key: 'node:0', kind: 'node', i: 0, x: 800, y: 0, r: 0, w: 50, h: 16 }
  ];
  assert.deepStrictEqual(
    agree('a cap of two', mixed, [], { max: 2 }).map(entry => entry.key),
    ['node:0', 'link:0'],
    'a subcatchment name is the first thing a cap drops'
  );

  // opts.anchors replaces the eight fallback positions, never the rule: a caller who
  // asks for C explicitly still does not get it on a node, because C on a node lands
  // on the symbol.  Asked for C alone, a node is left with no candidate at all.
  const disk = [{ key: 'node:0', kind: 'node', i: 0, x: 0, y: 0, r: 5, w: 50, h: 16 }];
  assert.deepStrictEqual(agree('C asked for on a node', disk, [], { anchors: ['C'] }), [], 'C is not an anchor a node can be given');
  assert.deepStrictEqual(
    agree('C and E asked for on a node', asNode, [], { anchors: ['C', 'E'] }),
    [{ key: 'node:0', dx: 0, dy: -8 }],
    'the rest of the list still counts, C is only dropped from it'
  );
  assert.deepStrictEqual(
    agree('C asked for twice on a subcatchment', lone, [], { anchors: ['C', 'E'] }),
    [{ key: 'poly:0', dx: -25, dy: -8 }],
    'a subcatchment that lists C itself still gets C, and gets it first'
  );
});

console.log('------------------------------------------------------------');
console.log(`ALL NETWORK MAP PURE-LAYER CASES VERIFIED (${passed} / 25 PASSED)`);
console.log('------------------------------------------------------------');
assert.strictEqual(passed, 25, 'every group must have run');
