/**
 * Automated Empirical Test Harness for Results V2 Interactive Workspace.
 *
 * Verifies all Visual & Interaction Acceptance Criteria:
 * 1. Tabbed workspace (Conduits, Nodes, Subcatchments with count badges)
 * 2. 3 summary cards with real session metrics
 * 3. Search and filtering across columns (multi-field, case-insensitive, recommendation/category dropdowns, failed checks / flooding toggles)
 * 4. Sortable column headers (asc/desc natural & numeric)
 * 5. Column visibility presets (Overview, Technical/Hydraulics, All columns) & pinned columns
 * 6. Inline validation expansion rows (7-check grid with actual values vs thresholds)
 * 7. CSV export downloads correct data with UTF-8 BOM
 * 8. Clicking conduit name navigates to conduit_detail URL
 *
 * Runs headless in Node.js with a complete simulated DOM environment.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

// -----------------------------------------------------------------------------
// 1. Minimal Simulated DOM Environment
// -----------------------------------------------------------------------------

/**
 * Split a descendant selector on the whitespace that separates its steps.
 *
 * A plain ``split(/\s+/)`` tears ``[data-row="P\ 1"]`` in half: the space inside the
 * brackets is part of the value, escaped by CSS.escape precisely so that it is not a
 * separator.  Whitespace counts only outside brackets and never right after a
 * backslash.
 */
function splitSelector(selector) {
  const parts = [];
  let current = '';
  let depth = 0;
  for (let index = 0; index < selector.length; index += 1) {
    const ch = selector[index];
    if (ch === '\\') {
      current += ch + (selector[index + 1] || '');
      index += 1;
      continue;
    }
    if (ch === '[') depth += 1;
    if (ch === ']') depth -= 1;
    if (depth === 0 && /\s/.test(ch)) {
      if (current) parts.push(current);
      current = '';
      continue;
    }
    current += ch;
  }
  if (current) parts.push(current);
  return parts;
}

/**
 * ``CSS.escape`` — enough of the specification for the names a model file carries.
 *
 * Node has no ``CSS`` object, and the engine now builds a row selector out of an
 * element name, so the harness has to provide the function the browser provides.
 * Every character outside ``[A-Za-z0-9_-]`` and outside the Latin-1 supplement and
 * beyond is escaped with a backslash; a leading digit is escaped as a code point, as
 * the specification requires, because ``\1`` alone would read as the start of one.
 */
function cssEscape(value) {
  const text = String(value);
  let out = '';
  for (let index = 0; index < text.length; index += 1) {
    const ch = text[index];
    const code = text.charCodeAt(index);
    if (code === 0) {
      out += '\uFFFD';
    } else if (index === 0 && code >= 0x0030 && code <= 0x0039) {
      out += '\\' + code.toString(16) + ' ';
    } else if (/[a-zA-Z0-9_-]/.test(ch) || code >= 0x0080) {
      out += ch;
    } else {
      out += '\\' + ch;
    }
  }
  return out;
}

/**
 * Undo the escaping ``CSS.escape`` applies to a value inside ``[attr="..."]``.
 *
 * Two forms exist and both are produced by the names of a model file: ``\\x`` for a
 * character that only needed a backslash, and ``\\31 `` - a code point in hexadecimal,
 * closed by one optional space - for a leading digit, which cannot be escaped the
 * short way because ``\\1`` would read as the start of a code point.
 */
function unescapeSelectorValue(raw) {
  return raw.replace(/\\([0-9a-fA-F]{1,6}) ?|\\([\s\S])/g, (whole, hex, plain) =>
    hex === undefined ? plain : String.fromCodePoint(parseInt(hex, 16))
  );
}

class MockElement {
  constructor(tagName, id = '', className = '') {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.className = className;
    this.children = [];
    this.parentElement = null;
    this.attributes = new Map();
    this.listeners = new Map();
    this.dataset = {};
    this.style = {
      setProperty: (k, v) => { this.style[k] = v; },
      getPropertyValue: (k) => this.style[k] || ''
    };
    this._value = '';
    this.checked = false;
    this.hidden = false;
    this.disabled = false;
    this.placeholder = '';
    this._textContent = '';
    this._innerHTML = '';
    this.scrollLeft = 0;
    this.scrollTop = 0;
    this.scrollWidth = 1000;
    this.clientWidth = 800;
    this.href = '';
    this.download = '';
  }

  get value() {
    return this._value;
  }

  set value(v) {
    this._value = String(v);
  }

  get textContent() {
    if (this._textContent !== '') return this._textContent;
    return this.children.map(c => c.textContent).join('');
  }

  set textContent(v) {
    this._textContent = String(v);
    this.children = [];
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set innerHTML(html) {
    this._innerHTML = String(html);
    this._parseHTML(html);
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
    if (name.startsWith('data-')) {
      const key = name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      this.dataset[key] = String(value);
    }
    if (name === 'id') this.id = String(value);
    if (name === 'class') this.className = String(value);
    if (name === 'hidden') this.hidden = true;
    if (name === 'disabled') this.disabled = true;
    if (name === 'checked') this.checked = true;
    if (name === 'value') this._value = String(value);
    if (name === 'href') this.href = String(value);
    if (name === 'download') this.download = String(value);
  }

  getAttribute(name) {
    if (name.startsWith('data-')) {
      const key = name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      if (this.dataset[key] !== undefined) return this.dataset[key];
    }
    if (name === 'id') return this.id || null;
    if (name === 'class') return this.className || null;
    if (name === 'hidden') return this.hidden ? '' : null;
    if (name === 'disabled') return this.disabled ? '' : null;
    if (name === 'checked') return this.checked ? '' : null;
    if (name === 'value') return this._value;
    if (name === 'href') return this.href || null;
    if (name === 'download') return this.download || null;
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }

  hasAttribute(name) {
    return this.getAttribute(name) !== null;
  }

  removeAttribute(name) {
    this.attributes.delete(name);
    if (name.startsWith('data-')) {
      const key = name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      delete this.dataset[key];
    }
    if (name === 'hidden') this.hidden = false;
    if (name === 'disabled') this.disabled = false;
    if (name === 'checked') this.checked = false;
  }

  addEventListener(type, handler) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, []);
    }
    this.listeners.get(type).push(handler);
  }

  removeEventListener(type, handler) {
    if (!this.listeners.has(type)) return;
    const list = this.listeners.get(type).filter(h => h !== handler);
    this.listeners.set(type, list);
  }

  dispatchEvent(event) {
    let cur = this;
    while (cur) {
      if (cur.listeners.has(event.type)) {
        for (const handler of cur.listeners.get(event.type)) {
          handler.call(cur, event);
          if (event._propagationStopped) return;
        }
      }
      cur = cur.parentElement;
    }
  }

  click() {
    const event = {
      type: 'click',
      target: this,
      currentTarget: null,
      defaultPrevented: false,
      _propagationStopped: false,
      preventDefault: () => { event.defaultPrevented = true; },
      stopPropagation: () => { event._propagationStopped = true; }
    };
    this.dispatchEvent(event);
  }

  focus() {}
  scrollTo() {}

  /**
   * The row the map pointed at asks to be brought into view; the harness only has to
   * remember that it was asked, and with which options.
   */
  scrollIntoView(options) {
    this.scrolledIntoView = options === undefined ? true : options;
  }

  /**
   * ``classList`` over the single ``className`` string.
   *
   * Built on demand rather than stored, because ``className`` is also written
   * directly (by setAttribute and by the HTML parser) and a cached token list would
   * quietly disagree with it.
   */
  get classList() {
    const owner = this;
    const tokens = () => owner.className.split(/\s+/).filter(Boolean);
    return {
      contains: name => tokens().includes(name),
      add: (...names) => {
        const list = tokens();
        names.forEach(name => { if (name && !list.includes(name)) list.push(name); });
        owner.className = list.join(' ');
      },
      remove: (...names) => {
        owner.className = tokens().filter(t => !names.includes(t)).join(' ');
      },
      toggle: (name, force) => {
        const has = tokens().includes(name);
        const on = force === undefined ? !has : Boolean(force);
        if (on) owner.classList.add(name); else owner.classList.remove(name);
        return on;
      }
    };
  }

  appendChild(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }

  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx !== -1) {
      this.children.splice(idx, 1);
      child.parentElement = null;
    }
    return child;
  }

  remove() {
    if (this.parentElement) {
      this.parentElement.removeChild(this);
    }
  }

  closest(selector) {
    let cur = this;
    while (cur) {
      if (cur._matchesSelector(selector)) return cur;
      cur = cur.parentElement;
    }
    return null;
  }

  _matchesSelector(selector) {
    selector = selector.trim();
    if (!selector) return false;

    // Split compound selector: tag, #id, .class, [attr=val]
    let remaining = selector;
    let tagMatch = remaining.match(/^[a-z0-9]+/i);
    if (tagMatch) {
      if (this.tagName.toLowerCase() !== tagMatch[0].toLowerCase()) return false;
      remaining = remaining.slice(tagMatch[0].length);
    }

    while (remaining.length > 0) {
      if (remaining.startsWith('#')) {
        const idMatch = remaining.match(/^#([a-z0-9_-]+)/i);
        if (!idMatch || this.id !== idMatch[1]) return false;
        remaining = remaining.slice(idMatch[0].length);
      } else if (remaining.startsWith('.')) {
        const clsMatch = remaining.match(/^\.([a-z0-9_-]+)/i);
        if (!clsMatch || !this.className.split(/\s+/).includes(clsMatch[1])) return false;
        remaining = remaining.slice(clsMatch[0].length);
      } else if (remaining.startsWith('[')) {
        // A backslash escapes the character after it, so a value may legally contain
        // a space, a quote or a bracket: this is what CSS.escape produces for the
        // element names SWMM allows (``A/B``, ``P 1``, ``s#1``, ``"x"``).
        const attrMatch = remaining.match(/^\[([a-z0-9_-]+)(?:=((?:\\.|[^\]])+))?\]/i);
        if (!attrMatch) return false;
        const attrName = attrMatch[1];
        const rawExpected = attrMatch[2];
        if (rawExpected === undefined) {
          if (!this.hasAttribute(attrName)) return false;
        } else {
          const expected = unescapeSelectorValue(rawExpected.replace(/^["']|["']$/g, ''));
          if (this.getAttribute(attrName) !== expected) return false;
        }
        remaining = remaining.slice(attrMatch[0].length);
      } else {
        break;
      }
    }
    return true;
  }

  querySelector(selector) {
    const all = this.querySelectorAll(selector);
    return all.length ? all[0] : null;
  }

  querySelectorAll(selector) {
    const parts = splitSelector(selector);
    let currentMatches = [this];
    for (const part of parts) {
      const nextMatches = [];
      for (const cur of currentMatches) {
        cur._walk(el => {
          if (el !== cur && el._matchesSelector(part) && !nextMatches.includes(el)) {
            nextMatches.push(el);
          }
        });
      }
      currentMatches = nextMatches;
    }
    return currentMatches;
  }

  _walk(fn) {
    fn(this);
    for (const child of this.children) {
      child._walk(fn);
    }
  }

  _parseHTML(html) {
    this.children = [];
    const tokens = [];
    const tokenizer = /<(\/)?([a-z0-9]+)([^>]*)>|([^<]+)/gi;
    let t;
    while ((t = tokenizer.exec(html)) !== null) {
      if (t[4]) {
        tokens.push({ type: 'text', content: t[4] });
      } else if (t[1]) {
        tokens.push({ type: 'close', tag: t[2].toLowerCase() });
      } else {
        const selfClosing = t[3].endsWith('/') || ['input', 'meta', 'link', 'br', 'hr', 'img'].includes(t[2].toLowerCase());
        tokens.push({
          type: selfClosing ? 'self' : 'open',
          tag: t[2].toLowerCase(),
          attrs: t[3].replace(/\/$/, '')
        });
      }
    }

    const stack = [this];
    for (const token of tokens) {
      const parent = stack[stack.length - 1];
      if (token.type === 'text') {
        const text = token.content.trim();
        if (text && parent) {
          parent._textContent = (parent._textContent ? parent._textContent + ' ' : '') + text;
        }
      } else if (token.type === 'open' || token.type === 'self') {
        const el = new MockElement(token.tag);
        const attrRegex = /([a-z0-9_-]+)(?:=(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/gi;
        let a;
        while ((a = attrRegex.exec(token.attrs)) !== null) {
          const name = a[1];
          const val = a[2] !== undefined ? a[2] : (a[3] !== undefined ? a[3] : (a[4] !== undefined ? a[4] : ''));
          el.setAttribute(name, val);
        }
        if (parent) {
          parent.appendChild(el);
        }
        if (token.type === 'open') {
          stack.push(el);
        }
      } else if (token.type === 'close') {
        if (stack.length > 1 && stack[stack.length - 1].tagName.toLowerCase() === token.tag) {
          stack.pop();
        }
      }
    }
  }
}

class MockDocument {
  constructor() {
    this.body = new MockElement('BODY');
    this.elementsById = new Map();
    this.readyState = 'complete';
  }

  getElementById(id) {
    if (this.elementsById.has(id)) {
      return this.elementsById.get(id);
    }
    let found = null;
    this.body._walk(el => {
      if (!found && el.id === id) found = el;
    });
    return found;
  }

  createElement(tag) {
    return new MockElement(tag);
  }

  addEventListener() {}
}

// -----------------------------------------------------------------------------
// 2. Realistic Session Data Payload Generator
// -----------------------------------------------------------------------------

function generateMockSessionData() {
  return {
    session: {
      id: 42,
      model_file: 'wroclaw_test_model.inp',
      created_at: '2026-09-21 07:00',
      frost_zone: 0.8,
      status: 'completed'
    },
    summary: {
      conduits_count: 5,
      conduits_changes_count: 3,
      nodes_count: 4,
      nodes_flooding_count: 2,
      catchments_count: 3
    },
    conduits: [
      {
        id: 'C1',
        name: 'C1',
        conduit_name: 'C1',
        inlet: 'N1',
        outlet: 'N2',
        catchment: 'S1',
        category: 'residential',
        diameter: 0.4,
        geom1: 0.4,
        length: 50.0,
        slope: 4.5,
        slope_per_mile: 4.5,
        coverIn: 1.2,
        coverOut: 1.4,
        minDiameter: 0.3,
        minSlope: 2.5,
        maxSlope: 54.0,
        flow: 0.12,
        max_q: 0.12,
        velocity: 1.25,
        max_v: 1.25,
        filling: 0.22,
        fillPercent: 55.0,
        validation: 7,
        failed_checks_count: 0,
        flags: [true, true, true, true, true, true, true],
        action: 'valid',
        recommendation: 'valid',
        rule: 'valid',
        rule_label: 'valid',
        confidence: 94.5,
        panel_url: '/analysis/results/42/conduit/C1/'
      },
      {
        id: 'C2',
        name: 'C2',
        conduit_name: 'C2',
        inlet: 'N2',
        outlet: 'N3',
        catchment: 'S1',
        category: 'residential',
        diameter: 0.3,
        geom1: 0.3,
        length: 40.0,
        slope: 2.0,
        slope_per_mile: 2.0,
        coverIn: 0.7,
        coverOut: 0.75,
        minDiameter: 0.4,
        minSlope: 3.5,
        maxSlope: 54.0,
        flow: 0.28,
        max_q: 0.28,
        velocity: 0.55,
        max_v: 0.55,
        filling: 0.29,
        fillPercent: 96.7,
        validation: 3,
        failed_checks_count: 4,
        flags: [false, true, false, true, false, true, false],
        action: 'diameter_increase',
        recommendation: 'diameter_increase',
        rule: 'depth_increase',
        rule_label: 'depth_increase',
        confidence: 82.0,
        panel_url: '/analysis/results/42/conduit/C2/'
      },
      {
        id: 'C3',
        name: 'C3',
        conduit_name: 'C3',
        inlet: 'N3',
        outlet: 'N4',
        catchment: 'S2',
        category: 'industrial',
        diameter: 0.6,
        geom1: 0.6,
        length: 80.0,
        slope: 5.5,
        slope_per_mile: 5.5,
        coverIn: 1.5,
        coverOut: 1.6,
        minDiameter: 0.5,
        minSlope: 2.0,
        maxSlope: 54.0,
        flow: 0.85,
        max_q: 0.85,
        velocity: 5.8,
        max_v: 5.8,
        filling: 0.42,
        fillPercent: 70.0,
        validation: 6,
        failed_checks_count: 1,
        flags: [true, false, true, true, true, true, true],
        action: 'slope_reduction',
        recommendation: 'slope_reduction',
        rule: 'slope_reduction',
        rule_label: 'slope_reduction',
        confidence: 88.0,
        panel_url: '/analysis/results/42/conduit/C3/'
      },
      {
        id: 'C4',
        name: 'C4',
        conduit_name: 'C4',
        inlet: 'N4',
        outlet: 'N1',
        catchment: 'S3',
        category: 'commercial',
        diameter: 0.5,
        geom1: 0.5,
        length: 60.0,
        slope: 60.0,
        slope_per_mile: 60.0,
        coverIn: 1.8,
        coverOut: 1.9,
        minDiameter: 0.4,
        minSlope: 2.0,
        maxSlope: 54.0,
        flow: 0.35,
        max_q: 0.35,
        velocity: 2.4,
        max_v: 2.4,
        filling: 0.25,
        fillPercent: 50.0,
        validation: 6,
        failed_checks_count: 1,
        flags: [true, true, true, false, true, true, true],
        action: 'tank',
        recommendation: 'tank',
        rule: 'tank',
        rule_label: 'tank',
        confidence: 91.0,
        panel_url: '/analysis/results/42/conduit/C4/'
      },
      {
        id: 'C5',
        name: 'C5',
        conduit_name: 'C5',
        inlet: 'N1',
        outlet: 'N3',
        catchment: 'S2',
        category: 'industrial',
        diameter: 0.4,
        geom1: 0.4,
        length: 45.0,
        slope: 3.8,
        slope_per_mile: 3.8,
        coverIn: 1.1,
        coverOut: 1.2,
        minDiameter: 0,
        minSlope: 229.72,
        maxSlope: 54.0,
        flow: 0.18,
        max_q: 0.18,
        velocity: 1.4,
        max_v: 1.4,
        filling: 0.20,
        fillPercent: 50.0,
        validation: 7,
        failed_checks_count: 0,
        flags: [true, true, true, true, true, true, true],
        action: 'valid',
        recommendation: 'valid',
        rule: 'valid',
        rule_label: 'valid',
        confidence: 97.0,
        panel_url: '/analysis/results/42/conduit/C5/'
      }
    ],
    nodes: [
      {
        id: 'N1',
        name: 'N1',
        node_name: 'N1',
        invert: 118.0,
        invert_elevation: 118.0,
        ground: 120.5,
        ground_elevation: 120.5,
        depth: 2.5,
        max_depth: 2.5,
        waterDepth: 1.1,
        max_water_depth: 1.1,
        floodHours: 0.0,
        hours_flooded: 0.0,
        floodVolume: 0.0,
        flood_volume: 0.0,
        is_flooded: false,
        panel_url: '/analysis/results/42/conduit/C1/#node-inlet'
      },
      {
        id: 'N2',
        name: 'N2',
        node_name: 'N2',
        invert: 117.5,
        invert_elevation: 117.5,
        ground: 120.0,
        ground_elevation: 120.0,
        depth: 2.5,
        max_depth: 2.5,
        waterDepth: 2.6,
        max_water_depth: 2.6,
        floodHours: 1.5,
        hours_flooded: 1.5,
        floodVolume: 4.8,
        flood_volume: 4.8,
        is_flooded: true,
        panel_url: '/analysis/results/42/conduit/C2/#node-inlet'
      },
      {
        id: 'N3',
        name: 'N3',
        node_name: 'N3',
        invert: 116.8,
        invert_elevation: 116.8,
        ground: 119.5,
        ground_elevation: 119.5,
        depth: 2.7,
        max_depth: 2.7,
        waterDepth: 2.8,
        max_water_depth: 2.8,
        floodHours: 0.5,
        hours_flooded: 0.5,
        floodVolume: 1.2,
        flood_volume: 1.2,
        is_flooded: true,
        panel_url: '/analysis/results/42/conduit/C3/#node-inlet'
      },
      {
        id: 'N4',
        name: 'N4',
        node_name: 'N4',
        invert: 116.0,
        invert_elevation: 116.0,
        ground: 118.8,
        ground_elevation: 118.8,
        depth: 2.8,
        max_depth: 2.8,
        waterDepth: 1.5,
        max_water_depth: 1.5,
        floodHours: 0.0,
        hours_flooded: 0.0,
        floodVolume: 0.0,
        flood_volume: 0.0,
        is_flooded: false,
        panel_url: '/analysis/results/42/conduit/C4/#node-inlet'
      }
    ],
    catchments: [
      {
        id: 'S1',
        name: 'S1',
        subcatchment_name: 'S1',
        outlet: 'N1',
        area: 1.85,
        impervious: 55.0,
        perc_imperv: 55.0,
        slope: 2.2,
        perc_slope: 2.2,
        category: 'residential',
        peak: 0.35,
        peak_runoff: 0.35,
        runoff: 14.2,
        total_runoff_mg: 14.2,
        coefficient: 0.45,
        runoff_coeff: 0.45,
        panel_url: '/analysis/results/42/conduit/C1/#catchment-1'
      },
      {
        id: 'S2',
        name: 'S2',
        subcatchment_name: 'S2',
        outlet: 'N3',
        area: 3.10,
        impervious: 75.0,
        perc_imperv: 75.0,
        slope: 1.8,
        perc_slope: 1.8,
        category: 'industrial',
        peak: 0.88,
        peak_runoff: 0.88,
        runoff: 32.5,
        total_runoff_mg: 32.5,
        coefficient: 0.65,
        runoff_coeff: 0.65,
        panel_url: '/analysis/results/42/conduit/C3/#catchment-2'
      },
      {
        id: 'S3',
        name: 'S3',
        subcatchment_name: 'S3',
        outlet: 'N4',
        area: 2.40,
        impervious: 60.0,
        perc_imperv: 60.0,
        slope: 2.5,
        perc_slope: 2.5,
        category: 'commercial',
        peak: 0.52,
        peak_runoff: 0.52,
        runoff: 20.1,
        total_runoff_mg: 20.1,
        coefficient: 0.52,
        runoff_coeff: 0.52,
        panel_url: '/analysis/results/42/conduit/C4/#catchment-3'
      }
    ]
  };
}

// -----------------------------------------------------------------------------
// 3. Template Skeletons
// -----------------------------------------------------------------------------

/**
 * The markup of #storm-results-v2, mirroring analysis_results.html.
 *
 * ``withMap`` switches between the two fixtures the engine has to survive: the page
 * as it renders without stored geometry (no map tab, no map panel, no map payload —
 * Conduits active) and the page as it renders with geometry (map tab first and
 * already selected, table panel hidden by the server, exactly as Django renders it).
 */
function workspaceMarkup(options) {
  const withMap = Boolean(options && options.withMap);
  const mapSelected = withMap ? 'true' : 'false';
  const conduitsSelected = withMap ? 'false' : 'true';
  const tableHidden = withMap ? ' hidden' : '';
  if (!withMap) {
    return `
    <div id="r-overview">
      <div class="r-workspace">
        <div class="r-tabs" role="tablist">
          <div class="r-tabs-list">
            <button type="button" class="r-tab" id="r-tab-conduits" data-tab="conduits" role="tab" aria-selected="true">
              Conduits <span class="r-count" id="r-count-conduits">0</span>
            </button>
            <button type="button" class="r-tab" id="r-tab-nodes" data-tab="nodes" role="tab" aria-selected="false">
              Nodes <span class="r-count" id="r-count-nodes">0</span>
            </button>
            <button type="button" class="r-tab" id="r-tab-catchments" data-tab="catchments" role="tab" aria-selected="false">
              Subcatchments <span class="r-count" id="r-count-catchments">0</span>
            </button>
            <button type="button" class="r-tab" id="r-tab-info" data-tab="info" role="tab" aria-selected="false">
              <i class="fas fa-info-circle"></i> Info
            </button>
          </div>
          <div class="r-tabs-actions">
            <a href="/analysis/" class="r-action-btn r-action-btn-primary">New Analysis</a>
            <a href="/history/" class="r-action-btn r-action-btn-secondary">History</a>
          </div>
        </div>
        <section id="r-table-panel" role="tabpanel">
          <div class="r-toolbar">
            <div class="r-filters">
              <label class="r-field r-search">
                <span>Search</span>
                <span class="r-inputwrap">
                  <input type="search" id="r-search" placeholder="Search...">
                </span>
              </label>
              <label class="r-field" id="r-select-field">
                <span id="r-select-label">Recommendation</span>
                <select id="r-filter-select"></select>
              </label>
              <label class="r-checkbox" id="r-failed-label">
                <input type="checkbox" id="r-failed">
                <span id="r-failed-text">Failed checks only</span>
              </label>
            </div>
            <div class="r-tools">
              <button type="button" class="r-button" id="r-columns" aria-expanded="false">
                Columns <small id="r-column-count"></small>
              </button>
              <button type="button" class="r-button" id="r-export">CSV</button>
            </div>
          </div>
          <div class="r-viewline">
            <span id="r-filter-summary"></span>
            <button type="button" class="r-reset" id="r-clear-filters" hidden>Clear filters</button>
          </div>
          <div class="r-selection" id="r-column-selection" hidden>
            <div class="r-selectionhead">
              <h3>Visible columns</h3>
              <div class="r-presets">
                <button type="button" class="r-preset" data-preset="overview">Overview</button>
                <button type="button" class="r-preset" data-preset="technical" id="r-technical-preset">Hydraulics</button>
                <button type="button" class="r-preset" data-preset="all">All columns</button>
                <button type="button" class="r-close" id="r-close-columns">&times;</button>
              </div>
            </div>
            <div class="r-groups" id="r-column-groups"></div>
          </div>
          <div class="r-tableframe" id="r-tableframe">
            <div class="r-overflow" id="r-overflow">
              <table id="r-table"></table>
            </div>
          </div>
          <footer class="r-foot">
            <span id="r-row-count"></span>
            <span class="r-scrollhint" id="r-scrollhint" hidden>
              <button type="button" id="r-scroll-columns">More columns</button>
            </span>
            <span id="r-foot-columns"></span>
          </footer>
        </section>
        <section id="r-info-panel" class="r-info" role="tabpanel" hidden>
          <div class="r-info-grid">
            <div class="r-info-item">
              <dt>Model file</dt>
              <dd>model.inp</dd>
            </div>
            <div class="r-info-item">
              <dt>Simulation date</dt>
              <dd>21 Sep 2026, 07:00</dd>
            </div>
            <div class="r-info-item">
              <dt>Frost zone</dt>
              <dd>0.80 m</dd>
            </div>
          </div>
        </section>
      </div>
    </div>
    <span class="r-sr" id="r-announcement"></span>
  `;
  }
  return `
    <div id="r-overview">
      <div class="r-workspace">
        <div class="r-tabs" role="tablist">
          <div class="r-tabs-list">
            <button type="button" class="r-tab" id="r-tab-map" data-tab="map" role="tab" aria-selected="${mapSelected}">
              Map
            </button>
            <button type="button" class="r-tab" id="r-tab-conduits" data-tab="conduits" role="tab" aria-selected="${conduitsSelected}">
              Conduits <span class="r-count" id="r-count-conduits">0</span>
            </button>
            <button type="button" class="r-tab" id="r-tab-nodes" data-tab="nodes" role="tab" aria-selected="false">
              Nodes <span class="r-count" id="r-count-nodes">0</span>
            </button>
            <button type="button" class="r-tab" id="r-tab-catchments" data-tab="catchments" role="tab" aria-selected="false">
              Subcatchments <span class="r-count" id="r-count-catchments">0</span>
            </button>
            <button type="button" class="r-tab" id="r-tab-info" data-tab="info" role="tab" aria-selected="false">
              <i class="fas fa-info-circle"></i> Info
            </button>
          </div>
          <div class="r-tabs-actions">
            <a href="/analysis/" class="r-action-btn r-action-btn-primary">New Analysis</a>
            <a href="/history/" class="r-action-btn r-action-btn-secondary">History</a>
          </div>
        </div>
        <section id="r-table-panel" role="tabpanel" aria-labelledby="r-tab-conduits"${tableHidden}>
          <div class="r-toolbar">
            <div class="r-filters">
              <label class="r-field r-search">
                <span>Search</span>
                <span class="r-inputwrap">
                  <input type="search" id="r-search" placeholder="Search...">
                </span>
              </label>
              <label class="r-field" id="r-select-field">
                <span id="r-select-label">Recommendation</span>
                <select id="r-filter-select"></select>
              </label>
              <label class="r-checkbox" id="r-failed-label">
                <input type="checkbox" id="r-failed">
                <span id="r-failed-text">Failed checks only</span>
              </label>
            </div>
            <div class="r-tools">
              <button type="button" class="r-button" id="r-columns" aria-expanded="false">
                Columns <small id="r-column-count"></small>
              </button>
              <button type="button" class="r-button" id="r-export">CSV</button>
            </div>
          </div>
          <div class="r-viewline">
            <span id="r-filter-summary"></span>
            <button type="button" class="r-reset" id="r-clear-filters" hidden>Clear filters</button>
          </div>
          <div class="r-selection" id="r-column-selection" hidden>
            <div class="r-selectionhead">
              <h3>Visible columns</h3>
              <div class="r-presets">
                <button type="button" class="r-preset" data-preset="overview">Overview</button>
                <button type="button" class="r-preset" data-preset="technical" id="r-technical-preset">Hydraulics</button>
                <button type="button" class="r-preset" data-preset="all">All columns</button>
                <button type="button" class="r-close" id="r-close-columns">&times;</button>
              </div>
            </div>
            <div class="r-groups" id="r-column-groups"></div>
          </div>
          <div class="r-tableframe" id="r-tableframe">
            <div class="r-overflow" id="r-overflow">
              <table id="r-table"></table>
            </div>
          </div>
          <footer class="r-foot">
            <span id="r-row-count"></span>
            <span class="r-scrollhint" id="r-scrollhint" hidden>
              <button type="button" id="r-scroll-columns">More columns</button>
            </span>
            <span id="r-foot-columns"></span>
          </footer>
        </section>
        <section id="r-info-panel" class="r-info" role="tabpanel" hidden>
          <div class="r-info-grid">
            <div class="r-info-item">
              <dt>Model file</dt>
              <dd>model.inp</dd>
            </div>
            <div class="r-info-item">
              <dt>Simulation date</dt>
              <dd>21 Sep 2026, 07:00</dd>
            </div>
            <div class="r-info-item">
              <dt>Frost zone</dt>
              <dd>0.80 m</dd>
            </div>
          </div>
        </section>
        <section id="r-map-panel" role="tabpanel" aria-labelledby="r-tab-map">
          <div class="nm-canvas" id="nm-canvas"></div>
        </section>
      </div>
    </div>
    <span class="r-sr" id="r-announcement"></span>
  `;
}
/**
 * The two browser globals the engine reached for in stage 7, in their simplest form.
 *
 * They are installed once, for both fixtures: ``renderTable`` now dispatches
 * ``rv2:filterchange`` on every render, so a fixture without ``CustomEvent`` would
 * fail on the first paint rather than in the test that cares about the event.  Node
 * has a ``CustomEvent`` of its own these days, but it is not the one the mock
 * dispatcher understands, and it has no ``CSS`` at all.
 */
global.CSS = { escape: cssEscape };

global.CustomEvent = class MockCustomEvent {
  constructor(type, init) {
    this.type = String(type);
    this.detail = init && 'detail' in init ? init.detail : null;
    this.bubbles = Boolean(init && init.bubbles);
    this.target = null;
    this.defaultPrevented = false;
    this._propagationStopped = false;
  }

  preventDefault() { this.defaultPrevented = true; }
  stopPropagation() { this._propagationStopped = true; }
};

// -----------------------------------------------------------------------------
// 4. Harness Setup & Runner
// -----------------------------------------------------------------------------

function runEmpiricalTests() {
  console.log('------------------------------------------------------------');
  console.log('Results V2 Empirical Acceptance Criteria Harness (Node.js)');
  console.log('------------------------------------------------------------');

  const doc = new MockDocument();
  const root = new MockElement('DIV', 'storm-results-v2');
  doc.body.appendChild(root);

  // Template skeleton inside #storm-results-v2 matching analysis_results.html
  root.innerHTML = workspaceMarkup({});

  // Attach data script
  const dataScript = new MockElement('SCRIPT', 'results-v2-data');
  dataScript.setAttribute('type', 'application/json');
  const mockPayload = generateMockSessionData();
  dataScript.textContent = JSON.stringify(mockPayload);
  doc.body.appendChild(dataScript);

  // Storage mock
  const storageMap = new Map();
  const localStorageMock = {
    getItem: k => storageMap.get(k) || null,
    setItem: (k, v) => storageMap.set(k, String(v)),
    clear: () => storageMap.clear()
  };

  // Mock global URLs & blobs
  let exportedBlobContent = null;
  let exportedDownloadName = null;

  global.document = doc;
  global.window = {
    addEventListener: () => {},
    matchMedia: () => ({ matches: false }),
    ResizeObserver: class { observe() {} }
  };
  global.ResizeObserver = global.window.ResizeObserver;
  global.localStorage = localStorageMock;
  global.URL = {
    createObjectURL: (blob) => {
      exportedBlobContent = blob._parts ? blob._parts.join('') : '';
      return 'blob:mock-export-url';
    },
    revokeObjectURL: () => {}
  };

  // Custom Blob implementation to capture payload
  global.Blob = class MockBlob {
    constructor(parts, options) {
      this._parts = parts;
      this.options = options;
    }
  };

  // Intercept anchor clicks for download verification
  const origCreateElement = doc.createElement.bind(doc);
  doc.createElement = (tag) => {
    const el = origCreateElement(tag);
    if (tag.toLowerCase() === 'a') {
      const origClick = el.click.bind(el);
      el.click = () => {
        exportedDownloadName = el.download;
        origClick();
      };
    }
    return el;
  };

  // Load and execute results-v2.js
  const jsFilePath = path.resolve(__dirname, '../../static/js/results-v2.js');
  const jsSource = fs.readFileSync(jsFilePath, 'utf8');
  eval(jsSource);

  console.log('✓ Script initialized without errors.');

  // Helper query functions
  const $ = s => root.querySelector(s);
  const $$ = s => root.querySelectorAll(s);

  // ---------------------------------------------------------------------------
  // Test 1: Tabbed Workspace & Badges
  // ---------------------------------------------------------------------------
  console.log('Testing Criteria 1: Tabbed Workspace & Badges...');
  assert.strictEqual($('#r-count-conduits').textContent.trim(), '5', 'Conduits count badge must show 5');
  assert.strictEqual($('#r-count-nodes').textContent.trim(), '4', 'Nodes count badge must show 4');
  assert.strictEqual($('#r-count-catchments').textContent.trim(), '3', 'Subcatchments count badge must show 3');

  const conduitsTab = $('#r-tab-conduits');
  const nodesTab = $('#r-tab-nodes');
  const catchmentsTab = $('#r-tab-catchments');

  assert.strictEqual(conduitsTab.getAttribute('aria-selected'), 'true', 'Conduits tab must initially be active');
  assert.strictEqual(nodesTab.getAttribute('aria-selected'), 'false', 'Nodes tab must initially be inactive');
  assert.strictEqual(catchmentsTab.getAttribute('aria-selected'), 'false', 'Catchments tab must initially be inactive');

  // Switch to Nodes tab
  nodesTab.click();
  assert.strictEqual(nodesTab.getAttribute('aria-selected'), 'true', 'Nodes tab must become active');
  assert.strictEqual(conduitsTab.getAttribute('aria-selected'), 'false', 'Conduits tab must become inactive');
  assert.strictEqual($('#r-row-count').textContent.trim(), '4 of 4 nodes', 'Footer must display 4 of 4 nodes');

  // Switch to Subcatchments tab
  catchmentsTab.click();
  assert.strictEqual(catchmentsTab.getAttribute('aria-selected'), 'true', 'Catchments tab must become active');
  assert.strictEqual($('#r-row-count').textContent.trim(), '3 of 3 subcatchments', 'Footer must display 3 of 3 subcatchments');

  // Switch to Info tab
  const infoTab = $('#r-tab-info');
  assert.ok(infoTab, 'Info tab button must exist');
  infoTab.click();
  assert.strictEqual(infoTab.getAttribute('aria-selected'), 'true', 'Info tab must become active');
  assert.strictEqual(conduitsTab.getAttribute('aria-selected'), 'false', 'Conduits tab must become inactive');
  assert.strictEqual($('#r-table-panel').hidden, true, 'Table panel must be hidden on Info tab');
  assert.strictEqual($('#r-info-panel').hidden, false, 'Info panel must be visible on Info tab');

  // Return to Conduits tab
  conduitsTab.click();
  assert.strictEqual(conduitsTab.getAttribute('aria-selected'), 'true', 'Conduits tab must return to active');
  assert.strictEqual($('#r-table-panel').hidden, false, 'Table panel must become visible again');
  assert.strictEqual($('#r-info-panel').hidden, true, 'Info panel must become hidden again');
  console.log('✓ Criteria 1 Passed: Tabbed workspace (including Info tab) and dynamic count badges work correctly.');

  // ---------------------------------------------------------------------------
  // Test 2: Recommendation / Flooding Filters and Clear Filters
  // ---------------------------------------------------------------------------
  console.log('Testing Criteria 2: Recommendation and flooding filters...');

  // "Changes recommended" narrows the conduits to those the model wants altered.
  const recSelect = $('#r-filter-select');
  recSelect.value = 'changes';
  recSelect.dispatchEvent({ type: 'change', target: recSelect });
  assert.strictEqual($('#r-row-count').textContent.trim(), '3 of 5 conduits', 'Only 3 conduits requiring changes should be listed');

  $('#r-clear-filters').click();
  assert.strictEqual($('#r-row-count').textContent.trim(), '5 of 5 conduits', 'All 5 conduits restored');

  // "Flooding only" narrows the nodes to those that reported an overflow.
  nodesTab.click();
  const floodOnly = $('#r-failed');
  floodOnly.checked = true;
  floodOnly.dispatchEvent({ type: 'change', target: floodOnly });
  assert.strictEqual($('#r-row-count').textContent.trim(), '2 of 4 nodes', 'Only 2 flooded nodes should be listed');

  // The flood mark reports the volume in cubic metres: flood_volume is SWMM's
  // Total Flood Volume in 10^6 litres, and 1 x 10^6 l = 1000 m3 (N2: 4.8 -> 4800.0 m3).
  const floodMarks = $$('#r-table .r-floodmark');
  assert.strictEqual(floodMarks.length, 2, 'Both flooded nodes must carry a flood mark');
  const floodTitles = floodMarks.map(m => m.getAttribute('title'));
  assert.ok(
    floodTitles.includes('Flooding reported: 4,800.0 m\u00b3') || floodTitles.includes('Flooding reported: 4800.0 m\u00b3'),
    'Flood mark must convert 4.8 x 10^6 l into 4800 m3, got: ' + floodTitles.join(' | ')
  );
  assert.ok(
    !floodTitles.some(t => /\b4\.80 m/.test(t)),
    'Flood mark must not print the 10^6 l figure as cubic metres'
  );

  $('#r-clear-filters').click();
  assert.strictEqual($('#r-failed').checked, false, 'Clear filters must uncheck flooding checkbox');
  assert.strictEqual($('#r-row-count').textContent.trim(), '4 of 4 nodes', 'All 4 nodes restored');

  conduitsTab.click();
  console.log('✓ Criteria 2 Passed: Recommendation and flooding filters narrow and reset correctly.');

  // ---------------------------------------------------------------------------
  // Test 3: Search and Filtering Across Columns
  // ---------------------------------------------------------------------------
  console.log('Testing Criteria 3: Search and Filtering...');
  const searchInput = $('#r-search');
  const filterSelect = $('#r-filter-select');
  const failedCheckbox = $('#r-failed');

  // Text search on Conduit ID
  searchInput.value = 'c2';
  searchInput.dispatchEvent({ type: 'input', target: searchInput });
  assert.strictEqual($('#r-row-count').textContent.trim(), '1 of 5 conduits', 'Search "c2" must match exactly 1 conduit');

  // Text search on Subcatchment identifier
  searchInput.value = 's2';
  searchInput.dispatchEvent({ type: 'input', target: searchInput });
  assert.strictEqual($('#r-row-count').textContent.trim(), '2 of 5 conduits', 'Search "s2" must match conduits C3 and C5');

  // Text search on Category
  searchInput.value = 'industrial';
  searchInput.dispatchEvent({ type: 'input', target: searchInput });
  assert.strictEqual($('#r-row-count').textContent.trim(), '2 of 5 conduits', 'Search "industrial" must match 2 conduits');

  // Clear search
  searchInput.value = '';
  searchInput.dispatchEvent({ type: 'input', target: searchInput });
  assert.strictEqual($('#r-row-count').textContent.trim(), '5 of 5 conduits', 'Clearing search restores 5 conduits');

  // Dropdown filter: recommendation = diameter_increase
  filterSelect.value = 'diameter_increase';
  filterSelect.dispatchEvent({ type: 'change', target: filterSelect });
  assert.strictEqual($('#r-row-count').textContent.trim(), '1 of 5 conduits', 'Filtering by diameter_increase matches C2');

  // Checkbox: Failed checks only
  filterSelect.value = 'all';
  filterSelect.dispatchEvent({ type: 'change', target: filterSelect });
  failedCheckbox.checked = true;
  failedCheckbox.dispatchEvent({ type: 'change', target: failedCheckbox });
  // C1 has 7/7, C2 has 3/7, C3 has 6/7, C4 has 6/7, C5 has 7/7 -> failed checks: C2, C3, C4 (3 conduits)
  assert.strictEqual($('#r-row-count').textContent.trim(), '3 of 5 conduits', 'Failed checks only filter matches 3 conduits');

  // Clear all filters
  $('#r-clear-filters').click();
  assert.strictEqual($('#r-row-count').textContent.trim(), '5 of 5 conduits', 'Filters reset successfully');
  console.log('✓ Criteria 3 Passed: Multi-field search, dropdown, and checkbox filters operate correctly.');

  // ---------------------------------------------------------------------------
  // Test 4: Sortable Column Headers (Asc / Desc Natural & Numeric)
  // ---------------------------------------------------------------------------
  console.log('Testing Criteria 4: Column Header Sorting...');
  const getRenderedRowIds = () => $$('#r-table .r-row').map(tr => tr.getAttribute('data-row'));

  // Sort by Diameter (numeric)
  const diaSortBtn = $('#r-table [data-sort="diameter"]');
  assert.ok(diaSortBtn, 'Diameter sort header button must exist');

  // Click ascending
  diaSortBtn.click();
  let rowIds = getRenderedRowIds();
  // Diameters: C2 (0.3), C1 (0.4), C5 (0.4), C4 (0.5), C3 (0.6)
  assert.strictEqual(rowIds[0], 'C2', 'Smallest diameter C2 (0.3) must be first in ascending sort');
  assert.strictEqual(rowIds[rowIds.length - 1], 'C3', 'Largest diameter C3 (0.6) must be last in ascending sort');
  assert.strictEqual($('#r-table th[aria-sort="ascending"] button').getAttribute('data-sort'), 'diameter');

  // Click descending
  diaSortBtn.click();
  rowIds = getRenderedRowIds();
  assert.strictEqual(rowIds[0], 'C3', 'Largest diameter C3 (0.6) must be first in descending sort');
  assert.strictEqual(rowIds[rowIds.length - 1], 'C2', 'Smallest diameter C2 (0.3) must be last in descending sort');
  assert.strictEqual($('#r-table th[aria-sort="descending"] button').getAttribute('data-sort'), 'diameter');

  // Sort by Velocity (numeric)
  const velSortBtn = $('#r-table [data-sort="velocity"]');
  velSortBtn.click(); // ascending
  rowIds = getRenderedRowIds();
  // Velocities: C2 (0.55), C1 (1.25), C5 (1.4), C4 (2.4), C3 (5.8)
  assert.strictEqual(rowIds[0], 'C2', 'Lowest velocity C2 (0.55 m/s) must be first');
  assert.strictEqual(rowIds[rowIds.length - 1], 'C3', 'Highest velocity C3 (5.8 m/s) must be last');

  // Sort by Conduit ID (natural string)
  const idSortBtn = $('#r-table [data-sort="id"]');
  idSortBtn.click();
  rowIds = getRenderedRowIds();
  assert.deepStrictEqual(rowIds, ['C1', 'C2', 'C3', 'C4', 'C5'], 'Conduit ID natural sort must order C1..C5');
  console.log('✓ Criteria 4 Passed: Column header sorting performs natural & numeric ordering accurately.');

  // ---------------------------------------------------------------------------
  // Test 5: Column Visibility Presets & Pinned Columns
  // ---------------------------------------------------------------------------
  console.log('Testing Criteria 5: Column Visibility Presets...');
  const columnsDrawerBtn = $('#r-columns');
  const drawer = $('#r-column-selection');

  // Open drawer
  columnsDrawerBtn.click();
  assert.strictEqual(drawer.hidden, false, 'Column drawer must open on button click');
  assert.strictEqual(columnsDrawerBtn.getAttribute('aria-expanded'), 'true');

  // Check Technical/Hydraulics preset
  const techPresetBtn = $('#r-technical-preset');
  assert.strictEqual(techPresetBtn.textContent.trim(), 'Hydraulics', 'Preset label must be Hydraulics on Conduits tab');
  techPresetBtn.click();

  let visibleThSortKeys = $$('#r-table th button').map(b => b.getAttribute('data-sort'));
  // Technical preset: ['id', 'diameter', 'slope', 'flow', 'velocity', 'filling', 'fillPercent', 'validation', 'minDiameter', 'minSlope']
  assert.ok(visibleThSortKeys.includes('slope'), 'Slope column must be visible in Technical preset');
  assert.ok(visibleThSortKeys.includes('filling'), 'Filling depth column must be visible in Technical preset');
  assert.ok(!visibleThSortKeys.includes('action'), 'Recommendation column must NOT be visible in Technical preset');

  // Check All columns preset
  const allPresetBtn = $('[data-preset="all"]');
  allPresetBtn.click();
  visibleThSortKeys = $$('#r-table th button').map(b => b.getAttribute('data-sort'));
  assert.strictEqual(visibleThSortKeys.length, 20, 'All 20 columns must be visible in All Columns preset');

  // Check Overview preset
  const overviewPresetBtn = $('[data-preset="overview"]');
  overviewPresetBtn.click();
  visibleThSortKeys = $$('#r-table th button').map(b => b.getAttribute('data-sort'));
  assert.deepStrictEqual(
    visibleThSortKeys,
    ['id', 'diameter', 'flow', 'velocity', 'fillPercent', 'category', 'validation', 'action', 'rule'],
    'Overview preset columns must match specification'
  );

  // Pinned column enforcement: Conduit ID input must be disabled
  const idColCheckbox = $('input[data-col="id"]');
  assert.strictEqual(idColCheckbox.disabled, true, 'Primary identifier column must be pinned and disabled');

  // Close drawer
  $('#r-close-columns').click();
  assert.strictEqual(drawer.hidden, true, 'Drawer must close on close button click');
  console.log('✓ Criteria 5 Passed: Column presets and pinned columns function reliably.');

  // ---------------------------------------------------------------------------
  // Test 6: Inline 7-Check Validation Expansion Rows
  // ---------------------------------------------------------------------------
  console.log('Testing Criteria 6: Inline 7-Check Validation Expansion...');
  const valBtnC2 = $('#r-table [data-validation="C2"]');
  assert.ok(valBtnC2, 'Validation button for C2 must exist');
  assert.strictEqual(valBtnC2.textContent.includes('3/7'), true, 'C2 must show 3/7 passed checks');

  // Click to expand validation row
  valBtnC2.click();
  let expandedRow = $('#r-checks-C2');
  assert.ok(expandedRow, 'Validation row #r-checks-C2 must be inserted into the DOM');
  const liveValBtnC2 = $('#r-table [data-validation="C2"]');
  assert.strictEqual(liveValBtnC2.getAttribute('aria-expanded'), 'true');

  // Verify 7-check grid inside expanded panel
  const checkCards = $$('#r-checks-C2 .r-check');
  assert.strictEqual(checkCards.length, 7, 'There must be exactly 7 validation check cards');

  // Check card titles and values:
  // 0: Max filling: actual 0.290 m, limit <= 0.827 * 0.3 = 0.248 m (FAILED)
  assert.strictEqual(checkCards[0].textContent.includes('Max filling'), true);
  assert.strictEqual(checkCards[0].textContent.includes('Failed'), true);
  assert.strictEqual(checkCards[0].textContent.includes('0.290 m · ≤ 0.248 m'), true);

  // 1: Max velocity: actual 0.55 m/s, limit <= 5.00 m/s (PASSED)
  assert.strictEqual(checkCards[1].textContent.includes('Max velocity'), true);
  assert.strictEqual(checkCards[1].textContent.includes('Passed'), true);
  assert.strictEqual(checkCards[1].textContent.includes('0.55 m/s · ≤ 5.00 m/s'), true);

  // 2: Min velocity: actual 0.55 m/s, limit >= 0.70 m/s (FAILED)
  assert.strictEqual(checkCards[2].textContent.includes('Min velocity'), true);
  assert.strictEqual(checkCards[2].textContent.includes('Failed'), true);
  assert.strictEqual(checkCards[2].textContent.includes('0.55 m/s · ≥ 0.70 m/s'), true);

  // 3: Max slope: actual 2.00 ‰, limit <= 54.00 ‰ (PASSED)
  assert.strictEqual(checkCards[3].textContent.includes('Max slope'), true);
  assert.strictEqual(checkCards[3].textContent.includes('Passed'), true);
  assert.strictEqual(checkCards[3].textContent.includes('2.00 ‰ · ≤ 54.00 ‰'), true);

  // 4: Min slope: actual 2.00 ‰, limit >= 3.50 ‰ (FAILED)
  assert.strictEqual(checkCards[4].textContent.includes('Min slope'), true);
  assert.strictEqual(checkCards[4].textContent.includes('Failed'), true);
  assert.strictEqual(checkCards[4].textContent.includes('2.00 ‰ · ≥ 3.50 ‰'), true);

  // 5: Depth: actual max depth, limit <= 8.00 m (PASSED)
  assert.strictEqual(checkCards[5].textContent.includes('Depth'), true);
  assert.strictEqual(checkCards[5].textContent.includes('Passed'), true);

  // 6: Coverage: min cover 0.70 m, limit >= 0.80 m (FAILED)
  assert.strictEqual(checkCards[6].textContent.includes('Coverage'), true);
  assert.strictEqual(checkCards[6].textContent.includes('Failed'), true);
  assert.strictEqual(checkCards[6].textContent.includes('0.70 m · ≥ 0.80 m'), true);

  // Close validation row via close button
  const closeChecksBtn = $('#r-checks-C2 [data-close-checks]');
  closeChecksBtn.click();
  expandedRow = $('#r-checks-C2');
  assert.strictEqual(expandedRow, null, 'Validation row must be removed from DOM after close');

  // Verify dry conduit (minSlope > maxSlope, minDiameter === 0) rendering on C5
  $('#r-technical-preset').click();
  const c5Row = $('#r-table [data-row="C5"]');
  assert.ok(c5Row.textContent.includes('—'), 'Dry conduit with minDiameter === 0 must render — instead of 0.000');
  $('#r-table [data-validation="C5"]').click();
  const c5CheckCards = $$('#r-checks-C5 .r-check');
  assert.ok(
    c5CheckCards[4].textContent.includes('no feasible slope (min 229.72 > max 54.00 ‰)'),
    'Dry conduit with minSlope > maxSlope must report no feasible slope'
  );
  $('#r-checks-C5 [data-close-checks]').click();
  $('[data-preset="overview"]').click();

  console.log('✓ Criteria 6 Passed: Inline 7-check validation grid renders actual values vs thresholds accurately.');

  // ---------------------------------------------------------------------------
  // Test 7: CSV Export Downloads Correct Data
  // ---------------------------------------------------------------------------
  console.log('Testing Criteria 7: CSV Export...');
  const exportBtn = $('#r-export');

  // Trigger CSV export
  exportBtn.click();

  assert.ok(exportedBlobContent, 'CSV blob content must be generated');
  assert.strictEqual(exportedDownloadName, 'session-42-conduits.csv', 'Download filename must match session-42-conduits.csv');
  assert.ok(exportedBlobContent.startsWith('\uFEFF'), 'CSV must start with UTF-8 BOM');

  const lines = exportedBlobContent.replace(/^\uFEFF/, '').trim().split('\r\n');
  assert.strictEqual(lines.length, 6, 'CSV must contain 1 header line + 5 data rows');

  const headerCols = lines[0].split(',');
  assert.ok(headerCols[0].includes('Conduit'), 'Header must contain Conduit');
  assert.ok(headerCols[1].includes('Diameter [m]'), 'Header must include unit in brackets');

  console.log('✓ Criteria 7 Passed: CSV export produces valid UTF-8 BOM data with accurate columns and filename.');

  // ---------------------------------------------------------------------------
  // Test 8: Clicking Conduit Name Navigates to conduit_detail URL
  // ---------------------------------------------------------------------------
  console.log('Testing Criteria 8: Conduit Detail Navigation Link...');
  const c1Link = $('#r-table a.r-id');
  assert.ok(c1Link, 'Conduit C1 must render as a clickable link');
  assert.strictEqual(c1Link.getAttribute('href'), '/analysis/results/42/conduit/C1/', 'Conduit link must target conduit_detail URL');

  // Ensure clicking the link triggers standard link navigation without preventDefault
  let navigationPrevented = false;
  const clickEvt = {
    type: 'click',
    target: c1Link,
    currentTarget: null,
    defaultPrevented: false,
    _propagationStopped: false,
    preventDefault: () => { navigationPrevented = true; },
    stopPropagation: () => {}
  };
  c1Link.dispatchEvent(clickEvt);
  assert.strictEqual(navigationPrevented, false, 'Link click must NOT prevent default navigation');
  console.log('✓ Criteria 8 Passed: Conduit identifier links directly to sa:conduit_detail URL.');

}

// -----------------------------------------------------------------------------
// 5. Second Fixture: the page as it renders with a stored network geometry
// -----------------------------------------------------------------------------

/**
 * The same engine against the other shape of the page.
 *
 * With geometry the server hands over a document whose Map tab is already selected
 * and whose table panel is already hidden, and 'map' is a tab like any other in TABS.
 * What is checked here is the split of roles the engine depends on: ``currentTab`` is
 * what the user sees, ``active`` is which table schema is loaded, applyTabState is the
 * only writer of aria-selected, and neither filtering nor sorting may move the
 * selection while the map is up.  ``active`` is not reachable from outside the IIFE,
 * so it is observed through the one thing derived from it — the table panel's
 * aria-labelledby.
 */
function runMapFixtureTests() {
  console.log('------------------------------------------------------------');
  console.log('Second fixture: workspace with a network map tab');
  console.log('------------------------------------------------------------');

  const doc = new MockDocument();
  const root = new MockElement('DIV', 'storm-results-v2');
  doc.body.appendChild(root);
  root.innerHTML = workspaceMarkup({ withMap: true });

  const dataScript = new MockElement('SCRIPT', 'results-v2-data');
  dataScript.setAttribute('type', 'application/json');
  dataScript.textContent = JSON.stringify(generateMockSessionData());
  doc.body.appendChild(dataScript);

  // The geometry envelope lives beside the results payload, outside the root, exactly
  // as json_script renders it.
  const mapScript = new MockElement('SCRIPT', 'network-map-data');
  mapScript.setAttribute('type', 'application/json');
  mapScript.textContent = JSON.stringify({
    available: true,
    reason: null,
    data: {
      version: 1,
      origin: [0, 0],
      bbox: { w: 20, h: 10 },
      min_gap: 10,
      layout: 'inp',
      nodes: [
        { n: 'N1', i: 0, p: [0, 0], k: 'junction' },
        { n: 'N2', i: 1, p: [20, 10], k: 'outfall' }
      ],
      links: [{ n: 'C1', i: 0, p: [[0, 0], [20, 10]], a: 'N1', b: 'N2', kind: 'conduit' }],
      polys: [],
      skipped: { links_dangling: 0, polys_degenerate: 0 },
      counts: { elements: 4 }
    }
  });
  doc.body.appendChild(mapScript);

  const storageMap = new Map();
  global.document = doc;
  global.window = {
    addEventListener: () => {},
    matchMedia: () => ({ matches: false }),
    ResizeObserver: class { observe() {} }
  };
  global.ResizeObserver = global.window.ResizeObserver;
  global.localStorage = {
    getItem: k => storageMap.get(k) || null,
    setItem: (k, v) => storageMap.set(k, String(v)),
    clear: () => storageMap.clear()
  };

  const jsSource = fs.readFileSync(path.resolve(__dirname, '../../static/js/results-v2.js'), 'utf8');
  eval(jsSource);

  const $ = s => root.querySelector(s);
  const $$ = s => root.querySelectorAll(s);
  const selected = () => $$('[data-tab]').filter(t => t.getAttribute('aria-selected') === 'true').map(t => t.id);
  const labelledBy = () => $('#r-table-panel').getAttribute('aria-labelledby');

  // -- On load: the map is the visible tab and nothing else claims to be ------------
  assert.deepStrictEqual(selected(), ['r-tab-map'], 'Exactly the map tab must be selected on load');
  assert.strictEqual($('#r-table-panel').hidden, true, 'Table panel must stay hidden while the map is up');
  assert.strictEqual($('#r-map-panel').hidden, false, 'Map panel must be visible on load');
  assert.strictEqual($('#r-info-panel').hidden, true, 'Info panel must be hidden on load');
  assert.strictEqual(labelledBy(), 'r-tab-conduits', "``active`` must still be a table type, not 'map'");
  assert.strictEqual($('#r-count-conduits').textContent.trim(), '5', 'Badges must be filled even with the map up');

  // -- Map -> Conduits: the table comes back --------------------------------------
  $('#r-tab-conduits').click();
  assert.deepStrictEqual(selected(), ['r-tab-conduits'], 'Selection must move to Conduits');
  assert.strictEqual($('#r-table-panel').hidden, false, 'Table panel must be revealed');
  assert.strictEqual($('#r-map-panel').hidden, true, 'Map panel must be hidden');
  assert.strictEqual($('#r-row-count').textContent.trim(), '5 of 5 conduits', 'Table must render its rows');

  // -- Back to the map, then straight to Nodes: the label follows ``active`` -------
  $('#r-tab-map').click();
  assert.deepStrictEqual(selected(), ['r-tab-map'], 'Map must be selectable again');
  $('#r-tab-nodes').click();
  assert.deepStrictEqual(selected(), ['r-tab-nodes'], 'Exactly one tab may be selected after Map -> Nodes');
  assert.strictEqual(labelledBy(), 'r-tab-nodes', 'Table panel must be labelled by the tab whose schema it shows');
  assert.strictEqual($('#r-row-count').textContent.trim(), '4 of 4 nodes', 'Nodes table must render');

  // -- Map does not disturb the table state it left behind ------------------------
  $('#r-tab-map').click();
  assert.strictEqual(labelledBy(), 'r-tab-nodes', "Showing the map must not change ``active``");
  assert.strictEqual($('#r-map-panel').hidden, false, 'Map panel must be visible again');

  // -- Filtering and sorting while the map is up must not steal the selection ------
  const searchInput = $('#r-search');
  searchInput.value = 'n2';
  searchInput.dispatchEvent({ type: 'input', target: searchInput });
  assert.deepStrictEqual(selected(), ['r-tab-map'], 'Filtering must not move aria-selected off the map');
  const sortBtn = $('#r-table [data-sort]');
  if (sortBtn) sortBtn.click();
  assert.deepStrictEqual(selected(), ['r-tab-map'], 'Sorting must not move aria-selected off the map');
  assert.strictEqual($('#r-table-panel').hidden, true, 'The table panel must stay hidden throughout');

  console.log('✓ Map fixture passed: map tab default, panel visibility and ARIA state have a single owner.');
  // ===========================================================================
  // Stage 7: the map points at a row, the table says what it is showing
  // ===========================================================================

  /** Every row of the rendered table that carries the highlight class. */
  const highlighted = () => $$('#r-table .r-row-hl').map(row => row.getAttribute('data-row'));

  /** The selectors the engine asked the document for, newest last. */
  const asked = [];
  const realQuerySelector = root.querySelector.bind(root);
  root.querySelector = selector => {
    asked.push(selector);
    return realQuerySelector(selector);
  };

  /** ``nm:select`` exactly as the Show in table button of the card sends it. */
  function select(kind, name) {
    root.dispatchEvent(new global.CustomEvent('nm:select', { bubbles: true, detail: { kind: kind, name: name } }));
  }

  // Every render announces what is on screen, whether filtered or not.
  const filterEvents = [];
  root.addEventListener('rv2:filterchange', e => filterEvents.push(e.detail));

  // -- The map is up; the card asks for a conduit ----------------------------------
  // The cases above left a query on the *nodes* filter (they were on the nodes schema
  // when they typed it), and a filter survives a change of tab by design.  Clearing it
  // here keeps the assertions below about stage 7 and not about that leftover.
  $('#r-tab-nodes').click();
  const leftover = $('#r-search');
  leftover.value = '';
  leftover.dispatchEvent({ type: 'input', target: leftover });
  $('#r-tab-map').click();
  assert.deepStrictEqual(selected(), ['r-tab-map'], 'the map must be the visible tab before the hand-over');
  select('conduits', 'C2');

  assert.deepStrictEqual(selected(), ['r-tab-conduits'], 'nm:select must switch to the tab of the element');
  assert.strictEqual($('#r-table-panel').hidden, false, 'the table panel must be revealed by the hand-over');
  assert.strictEqual($('#r-map-panel').hidden, true, 'the map panel must give way to the table');
  assert.deepStrictEqual(highlighted(), ['C2'], 'exactly the named row must carry the highlight');
  assert.strictEqual($('#r-table [data-row="C2"]').scrolledIntoView.block, 'nearest',
    'the row must be scrolled into view with block: nearest');

  // The class is not put on by hand after the switch: it is derived inside the render
  // that changeTab runs, which is why the state is stored first.  Sorting rebuilds the
  // whole <tbody> - the proof is that the highlight is on the new row too.
  const firstSort = $('#r-table [data-sort]');
  firstSort.click();
  assert.deepStrictEqual(highlighted(), ['C2'], 'the highlight must survive a sort');
  firstSort.click();
  assert.deepStrictEqual(highlighted(), ['C2'], 'the highlight must survive the reversed sort too');

  // -- A filter that hides the row costs the state nothing --------------------------
  const search = $('#r-search');
  search.value = 'C4';
  search.dispatchEvent({ type: 'input', target: search });
  assert.deepStrictEqual(highlighted(), [], 'a row the filter hides cannot be highlighted');
  assert.strictEqual($('#r-row-count').textContent.trim(), '1 of 5 conduits', 'the filter must have hidden the row');
  search.value = '';
  search.dispatchEvent({ type: 'input', target: search });
  assert.deepStrictEqual(highlighted(), ['C2'], 'clearing the filter must bring the highlight back');

  // -- rv2:filterchange: the tab vocabulary and the names on screen -----------------
  filterEvents.length = 0;
  search.value = 'C4';
  search.dispatchEvent({ type: 'input', target: search });
  const filtered = filterEvents[filterEvents.length - 1];
  assert.strictEqual(filtered.kind, 'conduits', 'the table must speak the tab vocabulary');
  assert.deepStrictEqual(filtered.visible, ['C4'], 'visible must carry the names of the filtered rows');
  search.value = '';
  search.dispatchEvent({ type: 'input', target: search });
  assert.deepStrictEqual(filterEvents[filterEvents.length - 1].visible, ['C1', 'C2', 'C3', 'C4', 'C5'],
    'an unfiltered table must list everything, so that the map can come back');

  // Another collection, another vocabulary: the event follows the schema on screen.
  filterEvents.length = 0;
  $('#r-tab-nodes').click();
  const nodeEvent = filterEvents[filterEvents.length - 1];
  assert.strictEqual(nodeEvent.kind, 'nodes', 'the nodes table must announce itself as nodes');
  // Sorted as the table happens to be sorted - the order of ``visible`` is the order on
  // screen and carries no meaning for the map, which only asks whether a name is in it.
  assert.deepStrictEqual([...nodeEvent.visible].sort(), ['N1', 'N2', 'N3', 'N4'], 'the nodes table must list node names');
  assert.deepStrictEqual(highlighted(), [], 'a conduit highlight may not leak into the nodes table');

  // -- A highlight of its own collection, and one that names nothing ----------------
  select('nodes', 'N3');
  assert.deepStrictEqual(highlighted(), ['N3'], 'a node must be highlighted in the nodes table');
  select('conduits', 'C5');
  assert.deepStrictEqual(highlighted(), ['C5'], 'the hand-over must move both the tab and the highlight');

  select('catchments', 'nothing-by-that-name');
  assert.deepStrictEqual(selected(), ['r-tab-catchments'], 'a name with no row must still switch the tab');
  assert.deepStrictEqual(highlighted(), [], 'a name with no row highlights nothing and throws nothing');
  assert.strictEqual($('#r-row-count').textContent.trim(), '3 of 3 subcatchments', 'the table must render as usual');

  // A message that cannot be acted on is ignored whole: the tab does not move.
  [undefined, null, {}, { kind: 'map', name: 'C1' }, { kind: 'info', name: 'C1' },
   { kind: 'conduits' }, { kind: 'conduits', name: 42 }, { kind: 'toString', name: 'C1' }].forEach(detail => {
    root.dispatchEvent(new global.CustomEvent('nm:select', { bubbles: true, detail: detail }));
    assert.deepStrictEqual(selected(), ['r-tab-catchments'], `nm:select ${JSON.stringify(detail)} must be ignored`);
  });

  // -- The row selector is built with CSS.escape, and holds for the awkward names ----
  // The browser is the authority on a selector; what is asserted here is that the
  // engine escapes the name instead of pasting it in, and that the result addresses
  // the row and that row alone.
  ['A/B', 'P 1', 's#1', '"x"', '1st'].forEach(name => {
    asked.length = 0;
    select('conduits', name);
    const rowSelectors = asked.filter(sel => sel.indexOf('[data-row=') !== -1);
    assert.ok(rowSelectors.length >= 1, `a lookup must have been made for ${name}`);
    rowSelectors.forEach(sel => {
      assert.strictEqual(sel, `#r-table [data-row="${cssEscape(name)}"]`, `the selector for ${name} must be escaped`);
      assert.ok(sel.indexOf(name) === -1 || name === cssEscape(name), `${name} must not appear raw in a selector`);
    });

    // The escaped selector still names exactly the row it was built for.
    const table = $('#r-table');
    const row = new MockElement('TR', '', 'r-row');
    row.setAttribute('data-row', name);
    table.appendChild(row);
    const found = $$(rowSelectors[0]);
    assert.strictEqual(found.length, 1, `the selector for ${name} must address one row`);
    assert.strictEqual(found[0], row, `the selector for ${name} must address its own row`);
    table.removeChild(row);
  });

  root.querySelector = realQuerySelector;
  console.log('✓ Stage 7 passed: nm:select highlights and scrolls, the highlight survives every render, rv2:filterchange lists what is on screen.');
}

runEmpiricalTests();
runMapFixtureTests();

console.log('------------------------------------------------------------');
console.log('ALL EMPIRICAL ACCEPTANCE CRITERIA VERIFIED (8 / 8 PASSED)');
console.log('------------------------------------------------------------');
