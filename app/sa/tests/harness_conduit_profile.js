/**
 * Harness of the longitudinal profile of the conduit panel (``renderProfile`` in
 * ``app/static/js/conduit-panel.js``).
 *
 * It checks the one thing a screenshot is poor at: the geometry of the joins between
 * the pipe and the two manholes. The pipe runs from the inlet shaft's outer wall to the
 * outlet shaft's wall, its crown and invert are open lines (no end strokes), and each
 * shaft outline leaves an opening exactly where the pipe enters. It also pins the water
 * tooltips (pipe = max conduit depth, shaft = max node depth) and the caption text.
 *
 * The svg is a stub (innerHTML + setAttribute): the checks read the markup string that
 * renderProfile writes, so they prove what is drawn, not how a browser paints it - the
 * painted joins are checked on the live page (Playwright) and by eye on the crops.
 *
 * ``require`` works because the module assigns ``module.exports`` inside its IIFE and
 * wires nothing when there is no ``document``.
 *
 * Runnable on its own:  node app/sa/tests/harness_conduit_profile.js  ->  exit code 0.
 */
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const cp = require(path.resolve(__dirname, '../../static/js/conduit-panel.js'));

assert.deepStrictEqual(Object.keys(cp), ['renderProfile', 'profileCaption'], 'conduit-panel.js exports the drawing and its caption');

// Half the wall stroke (app.css: stroke-width 2): the shaft outline ends this far inside
// the opening, on the inner edge of the crown / invert stroke.
const WALL_HALF = 1;
const SHAFT_RX = 3;
const EPS = 0.011;  // the path strings are rounded to two decimals

let passed = 0;
let total = 0;

function group(title, body) {
  total += 1;
  body();
  passed += 1;
  console.log(`✓ ${title}`);
}

function stubSvg() {
  return {
    innerHTML: '',
    attrs: {},
    setAttribute(name, value) { this.attrs[name] = value; },
  };
}

function render(data) {
  const svg = stubSvg();
  const exaggeration = cp.renderProfile(svg, data);
  return { svg, html: svg.innerHTML, exaggeration };
}

/** Every element carrying the given data-role, in document order, as attribute maps. */
function elements(html, role) {
  const re = new RegExp('<(\\w+) data-role="' + role + '"([^>]*?)/?>', 'g');
  const out = [];
  let m;
  while ((m = re.exec(html)) !== null) {
    const attrs = { tag: m[1] };
    const attrRe = /([\w-]+)="([^"]*)"/g;
    let a;
    while ((a = attrRe.exec(m[2])) !== null) {
      attrs[a[1]] = a[2];
    }
    out.push(attrs);
  }
  return out;
}

/** Tokenise a path string into [command, numbers[]] pairs. */
function commands(d) {
  const out = [];
  const re = /([MLHVAZmlhvaz])([^MLHVAZmlhvaz]*)/g;
  let m;
  while ((m = re.exec(d)) !== null) {
    const nums = m[2].trim() ? m[2].trim().split(/[\s,]+/).map(Number) : [];
    out.push([m[1], nums]);
  }
  return out;
}

/**
 * Walk a path of absolute M/L/H/V/A commands and return its straight segments and the
 * visited points (the arcs only move the pen - they are the rounded bottom corners).
 */
function walk(d) {
  const segments = [];
  const points = [];
  let x = NaN;
  let y = NaN;
  for (const [c, n] of commands(d)) {
    let nx = x;
    let ny = y;
    if (c === 'M' || c === 'L') { nx = n[0]; ny = n[1]; }
    else if (c === 'H') { nx = n[0]; }
    else if (c === 'V') { ny = n[0]; }
    else if (c === 'A') { nx = n[5]; ny = n[6]; }
    else { throw new Error(`unexpected path command ${c} in ${d}`); }
    if (c === 'L' || c === 'H' || c === 'V') {
      segments.push({ x1: x, y1: y, x2: nx, y2: ny });
    }
    x = nx;
    y = ny;
    points.push([x, y]);
  }
  return { segments, points };
}

function near(actual, expected, message) {
  assert.ok(Math.abs(actual - expected) <= EPS, `${message}: ${actual} != ${expected}`);
}

/** The band path of quad(): M xa ya L xb yb L xb yb' L xa ya' Z -> four corners. */
function band(el) {
  const cmds = commands(el.d);
  assert.deepStrictEqual(cmds.map((c) => c[0]).join(''), 'MLLLZ', `band path shape: ${el.d}`);
  const p = cmds.slice(0, 4).map((c) => c[1]);
  return { xa: p[0][0], topA: p[0][1], xb: p[1][0], topB: p[1][1], botB: p[2][1], botA: p[3][1], xb2: p[2][0], xa2: p[3][0] };
}

function base(overrides) {
  return Object.assign({
    name: 'C1', inletNode: 'N1', outletNode: 'N2',
    inletMaxDepth: null, outletMaxDepth: null,
    inletWaterDepth: null, outletWaterDepth: null,
    inletFloodHours: 0, outletFloodHours: 0, inletFloodVolume: 0, outletFloodVolume: 0,
    inletFlooded: false, outletFlooded: false,
    inletCover: null, outletCover: null, slope: null,
  }, overrides);
}

// Synthetic inputs modelled on the four conduits of the acceptance shots.
const CASES = {
  // 9 m, flat, widest shafts (W = 32), inlet flooded, water in both shafts.
  flatWide: base({
    length: 8.96, diameter: 0.3, filling: 0.3,
    inletElevation: 119.37, outletElevation: 119.37, inletInvert: 115.97, outletInvert: 115.97,
    inletWaterDepth: 3.4, outletWaterDepth: 0.3, inletFloodHours: 0.4, inletFloodVolume: 0.011,
    inletFlooded: true,
  }),
  // 3.5 m at 133 permille: the crown and invert at each wall differ from the axis.
  steep: base({
    length: 3.54, diameter: 0.3, filling: 0.05,
    inletElevation: 118.46, outletElevation: 118.36, inletInvert: 117.43, outletInvert: 116.96,
    inletWaterDepth: 0.3, outletWaterDepth: 0.18,
  }),
  // 83 m: the manhole width clamps to W_MIN = 14; both nodes flooded.
  long: base({
    length: 83.2, diameter: 0.3, filling: 0.3,
    inletElevation: 118.29, outletElevation: 118.4, inletInvert: 117.48, outletInvert: 117.45,
    inletWaterDepth: 0.81, outletWaterDepth: 0.95, inletFlooded: true, outletFlooded: true,
    inletFloodHours: 0.2, outletFloodHours: 0.3,
  }),
  // A large diameter with a partly filled pipe.
  large: base({
    length: 64.61, diameter: 0.6, filling: 0.19,
    inletElevation: 119.77, outletElevation: 119.67, inletInvert: 115.9, outletInvert: 115.85,
    inletWaterDepth: 0.19, outletWaterDepth: 0.19,
  }),
};

/** The full set of join checks for one drawable input; returns the measured geometry. */
function checkJoins(label, data, expectW) {
  const { html, exaggeration } = render(data);
  assert.ok(exaggeration >= 1, `${label}: drawable`);

  assert.strictEqual(elements(html, 'pipe').length, 1, `${label}: exactly one data-role="pipe"`);
  const shafts = elements(html, 'shaft');
  assert.strictEqual(shafts.length, 2, `${label}: two shaft bodies`);
  const W = Number(shafts[0].width);
  if (expectW !== undefined) {
    assert.strictEqual(W, expectW, `${label}: shaft width`);
  }
  const inlet = { x: Number(shafts[0].x), top: Number(shafts[0].y) };
  const outlet = { x: Number(shafts[1].x), top: Number(shafts[1].y) };
  inlet.sump = inlet.top + Number(shafts[0].height);
  outlet.sump = outlet.top + Number(shafts[1].height);
  const R = inlet.x + W;   // inlet shaft, outer (right) wall
  const L = outlet.x;      // outlet shaft, left wall

  // Pipe body and water: both span exactly wall to wall.
  const pipe = band(elements(html, 'pipe')[0]);
  const water = band(elements(html, 'water')[0]);
  for (const [name, b] of [['pipe', pipe], ['water', water]]) {
    near(b.xa, R, `${label}: ${name} starts at the inlet wall`);
    near(b.xa2, R, `${label}: ${name} closes at the inlet wall`);
    near(b.xb, L, `${label}: ${name} ends at the outlet wall`);
    near(b.xb2, L, `${label}: ${name} closes at the outlet wall`);
  }
  // Water never leaves the pipe: same invert, top at or below the crown.
  near(water.botA, pipe.botA, `${label}: water stands on the invert (inlet)`);
  near(water.botB, pipe.botB, `${label}: water stands on the invert (outlet)`);
  assert.ok(water.topA >= pipe.topA - EPS && water.topB >= pipe.topB - EPS, `${label}: water below the crown`);

  // Crown and invert: two open straight lines, wall to wall, on the band's edges.
  const walls = elements(html, 'pipe-wall');
  assert.strictEqual(walls.length, 2, `${label}: crown and invert lines`);
  for (const w of walls) {
    assert.ok(!/[Zz]/.test(w.d), `${label}: pipe-wall is open (no Z): ${w.d}`);
    assert.ok(/^M[-\d.]+ [-\d.]+L[-\d.]+ [-\d.]+$/.test(w.d), `${label}: pipe-wall is one straight segment, no end stroke: ${w.d}`);
  }
  const crown = commands(walls[0].d).map((c) => c[1]);
  const invert = commands(walls[1].d).map((c) => c[1]);
  near(crown[0][0], R, `${label}: crown starts at the inlet wall`);
  near(crown[1][0], L, `${label}: crown ends at the outlet wall`);
  near(invert[0][0], R, `${label}: invert starts at the inlet wall`);
  near(invert[1][0], L, `${label}: invert ends at the outlet wall`);
  near(crown[0][1], pipe.topA, `${label}: crown on the band (inlet)`);
  near(crown[1][1], pipe.topB, `${label}: crown on the band (outlet)`);
  near(invert[0][1], pipe.botA, `${label}: invert on the band (inlet)`);
  near(invert[1][1], pipe.botB, `${label}: invert on the band (outlet)`);

  // Shaft outlines: open exactly between crown and invert on the pipe side.
  const shaftWalls = elements(html, 'shaft-wall');
  assert.strictEqual(shaftWalls.length, 2, `${label}: two shaft outlines`);
  const sides = [
    { name: 'inlet', el: shaftWalls[0], wx: R, farX: inlet.x, crownY: crown[0][1], invertY: invert[0][1], s: inlet },
    { name: 'outlet', el: shaftWalls[1], wx: L, farX: outlet.x + W, crownY: crown[1][1], invertY: invert[1][1], s: outlet },
  ];
  for (const side of sides) {
    const tag = `${label} ${side.name}`;
    assert.ok(!/[Zz]/.test(side.el.d), `${tag}: shaft outline is open: ${side.el.d}`);
    const { segments, points } = walk(side.el.d);
    const first = points[0];
    const last = points[points.length - 1];
    // Starts at the crown and ends at the invert, on the wall facing the pipe, each
    // WALL_HALF inside the opening (the inner edge of the crown / invert stroke).
    near(first[0], side.wx, `${tag}: outline starts on the pipe-side wall`);
    near(last[0], side.wx, `${tag}: outline ends on the pipe-side wall`);
    near(first[1], side.crownY + WALL_HALF, `${tag}: opening starts at the crown`);
    near(last[1], side.invertY - WALL_HALF, `${tag}: opening ends at the invert`);
    // No wall segment on the pipe side crosses the opening.
    const lo = side.crownY + WALL_HALF;
    const hi = side.invertY - WALL_HALF;
    for (const sg of segments) {
      if (Math.abs(sg.x1 - side.wx) < EPS && Math.abs(sg.x2 - side.wx) < EPS) {
        const a = Math.min(sg.y1, sg.y2);
        const b = Math.max(sg.y1, sg.y2);
        assert.ok(b <= lo + EPS || a >= hi - EPS, `${tag}: a wall segment ${a}..${b} crosses the opening ${lo}..${hi}`);
      }
    }
    // The rest of the outline goes round the shaft: far wall, lid line, sump.
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    near(Math.min(...xs), Math.min(side.wx, side.farX), `${tag}: outline spans the shaft (left)`);
    near(Math.max(...xs), Math.max(side.wx, side.farX), `${tag}: outline spans the shaft (right)`);
    near(Math.min(...ys), side.s.top, `${tag}: outline reaches the lid line`);
    near(Math.max(...ys), side.s.sump, `${tag}: outline reaches the sump`);
    assert.strictEqual(commands(side.el.d).filter((c) => c[0] === 'A').length, 2, `${tag}: two rounded bottom corners`);
  }

  // Draw order of spec item 5.
  const order = ['"shaft"', '"shaft-water"', '"pipe"', '"water"', '"pipe-wall"', '"shaft-wall"', '"flow-arrow"', '"lid"'];
  const at = order.map((role) => html.indexOf('data-role=' + role));
  for (let k = 0; k < at.length; k++) {
    assert.ok(at[k] >= 0, `${label}: ${order[k]} drawn`);
    if (k > 0) {
      assert.ok(at[k - 1] < at[k], `${label}: ${order[k - 1]} before ${order[k]}`);
    }
  }
  return { W, R, L, html };
}

/** Shaft water as absolute rectangles: parses M x y h w V y h -w Z (+ an optional tab). */
function shaftWaterRects(html) {
  return elements(html, 'shaft-water').map((el) => {
    const parts = el.d.split('Z').filter(Boolean);
    return parts.map((p) => {
      const m = /^M([-\d.]+) ([-\d.]+)h([-\d.]+)V([-\d.]+)h([-\d.]+)$/.exec(p);
      assert.ok(m, `shaft water sub-path shape: ${p}`);
      const x = Number(m[1]);
      const dx = Number(m[3]);
      return { x1: Math.min(x, x + dx), x2: Math.max(x, x + dx), y1: Number(m[2]), y2: Number(m[4]) };
    });
  });
}

group('flat conduit, widest shafts: pipe, water, crown and invert run wall to wall', () => {
  const g = checkJoins('flatWide', CASES.flatWide, 32);
  assert.ok(g.L - g.R > 300, 'the pipe spans nearly the whole drawing');
});

group('steep conduit: the openings follow crown and invert at each wall, not at the axis', () => {
  const g = checkJoins('steep', CASES.steep);
  const crown = commands(elements(g.html, 'pipe-wall')[0].d).map((c) => c[1]);
  // On this slope the crown drops by tens of units between the two walls, so its height
  // at the wall differs from its height at the manhole axis (half a shaft away) -
  // the openings checked in checkJoins are the ones at the walls.
  const fall = (crown[1][1] - crown[0][1]) / (g.L - g.R);
  assert.ok(crown[1][1] - crown[0][1] > 30, 'the crown falls between the walls');
  assert.ok(fall * g.W / 2 > 1, 'crown at the wall and at the axis differ by more than one unit');
});

group('long conduit: the manhole width clamps to W_MIN and the joins still hold', () => {
  checkJoins('long', CASES.long, 14);
});

group('large diameter, partly filled: the joins hold and the water stays under the crown', () => {
  checkJoins('large', CASES.large);
});

group('shaft water meets the pipe water in the opening and never leaves the pipe band', () => {
  const { html } = render(CASES.flatWide);
  const shafts = elements(html, 'shaft');
  const W = Number(shafts[0].width);
  const pipe = band(elements(html, 'pipe')[0]);
  const rects = shaftWaterRects(html);
  assert.strictEqual(rects.length, 2, 'water in both shafts');
  const walls = [
    { x: Number(shafts[0].x), wx: Number(shafts[0].x) + W, crown: pipe.topA, invert: pipe.botA },
    { x: Number(shafts[1].x), wx: Number(shafts[1].x), crown: pipe.topB, invert: pipe.botB },
  ];
  rects.forEach((parts, i) => {
    const w = walls[i];
    // The body: the full shaft width, so it reaches the wall line at the opening.
    near(parts[0].x1, w.x, `shaft ${i}: water from the left wall`);
    near(parts[0].x2, w.x + W, `shaft ${i}: water to the right wall`);
    // The tab: outside the shaft by WALL_HALF only, and only between crown and invert.
    if (parts.length > 1) {
      const tab = parts[1];
      assert.ok(Math.abs(tab.x1 - w.wx) < EPS || Math.abs(tab.x2 - w.wx) < EPS, `shaft ${i}: tab on the pipe-side wall`);
      near(tab.x2 - tab.x1, WALL_HALF, `shaft ${i}: tab width`);
      assert.ok(tab.y1 >= w.crown - EPS && tab.y2 <= w.invert + EPS, `shaft ${i}: tab inside the pipe band`);
    }
  });
});

group('a steep drop is clamped: the opening never leaves the shaft body', () => {
  // A steep conduit whose crown pokes out of the ground at both manholes: 0.05 m of
  // shaft under a 0.6 m pipe at the inlet, and at the outlet the crown is 0.2 m above the
  // ground at the axis and higher still at the wall (the pipe rises towards the inlet).
  // Without the clamp the outlet outline would start some 6 units above its lid line; at
  // the inlet the fall brings the crown back to just under a unit above it. (Below the sump it
  // cannot go with these constants: the steepest drawable fall moves the invert by at
  // most ~5 units over half a shaft, SUMP is 9; the clamp covers it all the same.)
  const { html } = render(base({
    length: 1, diameter: 0.6, filling: 0.1,
    inletElevation: 112, outletElevation: 101, inletInvert: 111.95, outletInvert: 100.6,
  }));
  const shafts = elements(html, 'shaft');
  const shaftWalls = elements(html, 'shaft-wall');
  assert.strictEqual(shaftWalls.length, 2);
  shaftWalls.forEach((el, i) => {
    const top = Number(shafts[i].y);
    const sump = top + Number(shafts[i].height);
    for (const [, y] of walk(el.d).points) {
      assert.ok(y >= top - EPS && y <= sump + EPS, `shaft ${i}: outline point ${y} outside ${top}..${sump}`);
    }
    const { points } = walk(el.d);
    const first = points[0][1];
    const last = points[points.length - 1][1];
    assert.ok(first <= last + EPS, `shaft ${i}: the opening never turns inside out (${first} > ${last})`);
    assert.ok(last <= sump - SHAFT_RX + EPS, `shaft ${i}: the opening stays above the bottom corner`);
  });
});

group('no data: the message only, no pipe or shaft outline', () => {
  const { html, exaggeration, svg } = render(base({ length: 0, diameter: 0.3, filling: 0.1 }));
  assert.strictEqual(exaggeration, 0);
  assert.ok(html.includes('No data to draw the profile'));
  for (const role of ['pipe', 'pipe-wall', 'shaft', 'shaft-wall', 'water', 'shaft-water']) {
    assert.strictEqual(elements(html, role).length, 0, `no ${role} without data`);
  }
  assert.ok(/no data/.test(svg.attrs['aria-label']));
});

/** The <title> texts inside every element with the given data-role, in document order. */
function titles(html, role) {
  const re = new RegExp('<(\\w+) data-role="' + role + '"[^>]*?(/?)>', 'g');
  const out = [];
  let m;
  while ((m = re.exec(html)) !== null) {
    if (m[2] === '/') {
      out.push([]);
      continue;
    }
    const close = html.indexOf('</' + m[1] + '>', re.lastIndex);
    const inner = html.slice(re.lastIndex, close);
    out.push([...inner.matchAll(/<title>([^<]*)<\/title>/g)].map((t) => t[1]));
  }
  return out;
}

group('water tooltips: the pipe shows the conduit maximum, each shaft its node maximum', () => {
  // The two levels are different SWMM maxima and are drawn as they are; the titles say
  // which is which, with the numbers the drawing uses (two decimals).
  const { html } = render(CASES.large);
  assert.deepStrictEqual(titles(html, 'water'), [['Max conduit depth 0.19 m (32 %)']]);
  assert.deepStrictEqual(titles(html, 'shaft-water'), [
    ['Max water depth at N1: 0.19 m'],
    ['Max water depth at N2: 0.19 m'],
  ]);
  const flat = render(CASES.flatWide).html;
  assert.deepStrictEqual(titles(flat, 'water'), [['Max conduit depth 0.30 m (100 %)']]);
  assert.deepStrictEqual(titles(flat, 'shaft-water'), [
    ['Max water depth at N1: 3.40 m'],
    ['Max water depth at N2: 0.30 m'],
  ]);
  // The flood marker keeps its own title; the pipe body and the walls carry none.
  assert.deepStrictEqual(titles(flat, 'flood'), [['Flooding: 0.40 h, 0.011 × 10⁶ l']]);
  for (const role of ['pipe', 'pipe-wall', 'shaft', 'shaft-wall']) {
    assert.ok(titles(flat, role).every((t) => t.length === 0), `${role}: no title`);
  }
});

group('water tooltips: a flooded node without a depth gets water but no title; names are escaped', () => {
  const { html } = render(Object.assign({}, CASES.long, { inletWaterDepth: null, inletNode: 'A<&>"B' }));
  const shaftTitles = titles(html, 'shaft-water');
  assert.strictEqual(shaftTitles.length, 2, 'both flooded shafts hold water');
  assert.deepStrictEqual(shaftTitles[0], [], 'no depth, no title');
  assert.deepStrictEqual(shaftTitles[1], ['Max water depth at N2: 0.95 m']);
  const esc = render(Object.assign({}, CASES.large, { inletNode: 'A<&>"B' })).html;
  assert.deepStrictEqual(titles(esc, 'shaft-water')[0], ['Max water depth at A&lt;&amp;&gt;&quot;B: 0.19 m']);
});

group('caption: the water legend always, the exaggeration only above 1, nothing without a drawing', () => {
  const legend = 'Water: manhole = max node depth · pipe = max conduit depth';
  assert.strictEqual(cp.profileCaption(0), '');
  assert.strictEqual(cp.profileCaption(1), legend);
  assert.strictEqual(cp.profileCaption(12), legend + ' · vertical exaggeration ×12');
});

group('app.css: the bodies are fill-only, only the wall roles stroke', () => {
  const css = fs.readFileSync(path.resolve(__dirname, '../../static/css/app.css'), 'utf8');
  const rule = (role) => {
    const m = new RegExp('\\.conduit-profile \\[data-role="' + role + '"\\] \\{([^}]*)\\}').exec(css);
    assert.ok(m, `rule for ${role}`);
    return m[1];
  };
  assert.ok(!/stroke/.test(rule('pipe')), 'the pipe body has no stroke (open ends)');
  assert.ok(!/stroke/.test(rule('shaft')), 'the shaft body has no stroke (the opening)');
  for (const role of ['pipe-wall', 'shaft-wall']) {
    const body = rule(role);
    assert.ok(/fill:\s*none/.test(body), `${role}: no fill`);
    assert.ok(/stroke-width:\s*2;/.test(body), `${role}: stroke 2 = 2 * WALL_HALF`);
    assert.ok(/stroke-linecap:\s*butt/.test(body), `${role}: butt caps`);
  }
});

console.log(`ALL CONDUIT PROFILE CASES VERIFIED (${passed} / ${total} PASSED)`);
