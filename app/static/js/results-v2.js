/**
 * Results V2 Interactive Workspace Widget Engine
 *
 * Scoped strictly within #storm-results-v2 container.
 * Reads data cleanly from <script id="results-v2-data" type="application/json">.
 * Implements:
 *   - Tab state management (Conduits, Nodes, Subcatchments with counts).
 *   - Summary KPI cards with interactive filtering.
 *   - Multi-field search across visible columns and primary identifiers.
 *   - Dropdown filters (recommendations, category, failed checks toggle, flooding toggle).
 *   - Column sorting with ascending/descending indicators and natural alphanumeric ordering.
 *   - Column visibility chooser with presets (Overview, Technical/Hydraulics/Geometry/Runoff, All columns).
 *   - Inline 7-check validation expansion rows for conduits with actual vs threshold values.
 *   - Direct conduit navigation to panel_url.
 *   - Client-side CSV export with UTF-8 BOM of currently visible/filtered rows.
 *   - Pure Font Awesome 6 (fas fa-*) icons throughout.
 */
(function () {
  'use strict';

  function initResultsV2() {
    const root = document.getElementById('storm-results-v2');
    if (!root) {
      return;
    }
    if (document.body && document.body.classList) {
      document.body.classList.add('results-v2-page');
    }

    const dataScript = document.getElementById('results-v2-data');
    if (!dataScript) {
      console.warn('results-v2: Script tag #results-v2-data not found.');
      return;
    }

    let rawData;
    try {
      rawData = JSON.parse(dataScript.textContent || '{}');
    } catch (err) {
      console.error('results-v2: Failed to parse JSON from #results-v2-data:', err);
      return;
    }

    const $ = selector => root.querySelector(selector);
    const $$ = selector => Array.from(root.querySelectorAll(selector));

    const TYPES = ['conduits', 'nodes', 'catchments'];
    const TABS = ['map', 'conduits', 'nodes', 'catchments', 'info'];
    const CHECKS = [
      'Max filling',
      'Max velocity',
      'Min velocity',
      'Max slope',
      'Min slope',
      'Depth',
      'Coverage'
    ];

    const ACTIONS = {
      valid: 'No changes',
      diameter_increase: 'Increase diameter',
      diameter_reduction: 'Reduce diameter',
      slope_increase: 'Increase slope',
      slope_reduction: 'Reduce slope',
      depth_increase: 'Increase depth',
      pump: 'Pump station',
      tank: 'Retention tank',
      seepage_boxes: 'Seepage boxes'
    };

    function getActionLabel(code) {
      if (!code || code === '-') return '-';
      const key = String(code).trim().toLowerCase();
      return ACTIONS[key] || String(code);
    }

    const numberFormats = new Map();
    const numberFormat = digits => {
      let nf = numberFormats.get(digits);
      if (!nf) {
        nf = new Intl.NumberFormat('en-GB', {
          minimumFractionDigits: digits,
          maximumFractionDigits: digits
        });
        numberFormats.set(digits, nf);
      }
      return nf;
    };

    const fmt = (v, p = 2) => {
      if (v === null || v === undefined || !Number.isFinite(Number(v))) {
        return '—';
      }
      return numberFormat(p).format(Number(v));
    };

    // One collator for every string comparison; constructing one per comparison
    // dominates the cost of sorting a large table.
    const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

    const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[c]));

    // Extract raw datasets with fallback aliases
    const rawConduits = Array.isArray(rawData.conduits) ? rawData.conduits : [];
    const rawNodes = Array.isArray(rawData.nodes) ? rawData.nodes : [];
    const rawCatchments = Array.isArray(rawData.catchments)
      ? rawData.catchments
      : (Array.isArray(rawData.subcatchments) ? rawData.subcatchments : []);

    // Column definition helper
    const C = (key, label, unit, group, kind = 'number', digits = 2) => ({
      key,
      label,
      unit,
      group,
      kind,
      digits
    });

    const schemas = {
      conduits: {
        name: 'Conduits',
        rows: rawConduits,
        cols: [
          C('id', 'Conduit', '', 'Identity', 'id'),
          C('inlet', 'Inlet node', '', 'Identity', 'text'),
          C('outlet', 'Outlet node', '', 'Identity', 'text'),
          C('catchment', 'Model subcatchment', '', 'Identity', 'text'),
          C('diameter', 'Diameter', 'm', 'Geometry', 'number', 3),
          C('length', 'Length', 'm', 'Geometry', 'number', 2),
          C('slope', 'Slope', '‰', 'Geometry', 'number', 2),
          C('coverIn', 'Inlet cover', 'm', 'Geometry', 'number', 2),
          C('coverOut', 'Outlet cover', 'm', 'Geometry', 'number', 2),
          C('minDiameter', 'Min. diameter', 'm', 'Geometry', 'number', 3),
          C('minSlope', 'Min. slope', '‰', 'Geometry', 'number', 2),
          C('flow', 'Max. flow', 'm³/s', 'Hydraulics', 'number', 3),
          C('velocity', 'Max. velocity', 'm/s', 'Hydraulics', 'number', 2),
          C('filling', 'Water depth', 'm', 'Hydraulics', 'number', 3),
          C('fillPercent', 'Filling', '% of diameter', 'Hydraulics', 'fill', 0),
          C('category', 'Catchment category', '', 'Assessment', 'category'),
          C('validation', 'Validation', 'passed / 7', 'Assessment', 'checks'),
          C('action', 'Recommendation', '', 'Assessment', 'action'),
          C('confidence', 'Model confidence', '%', 'Assessment', 'number', 1),
          C('rule', 'Rule label', 'Wrocław reference', 'Assessment', 'rule')
        ],
        overview: ['id', 'diameter', 'flow', 'velocity', 'fillPercent', 'category', 'validation', 'action', 'rule'],
        technical: ['id', 'diameter', 'slope', 'flow', 'velocity', 'filling', 'fillPercent', 'validation', 'minDiameter', 'minSlope']
      },
      nodes: {
        name: 'Nodes',
        rows: rawNodes,
        cols: [
          C('id', 'Node', '', 'Identity', 'id'),
          C('invert', 'Node invert', 'm', 'Geometry', 'number', 2),
          C('ground', 'Ground elevation', 'm', 'Geometry', 'number', 2),
          C('depth', 'Node depth', 'm', 'Geometry', 'number', 2),
          C('waterDepth', 'Max. water depth', 'm', 'Simulation', 'number', 2),
          C('floodHours', 'Flooding duration', 'h', 'Flooding', 'number', 2),
          C('floodVolume', 'Flooding volume', '10⁶ l', 'Flooding', 'number', 3)
        ],
        overview: ['id', 'invert', 'depth', 'waterDepth', 'floodHours', 'floodVolume'],
        technical: ['id', 'ground', 'invert', 'depth', 'waterDepth']
      },
      catchments: {
        name: 'Subcatchments',
        rows: rawCatchments,
        cols: [
          C('id', 'Subcatchment', '', 'Identity', 'id'),
          C('outlet', 'Outlet node', '', 'Identity', 'text'),
          C('area', 'Area', 'ha', 'Characteristics', 'number', 3),
          C('impervious', 'Imperviousness', '%', 'Characteristics', 'number', 1),
          C('slope', 'Slope', '%', 'Characteristics', 'number', 2),
          C('category', 'Category', '', 'Classification', 'category'),
          C('peak', 'Peak runoff', 'm³/s', 'Runoff', 'number', 3),
          C('runoff', 'Total runoff', '10⁶ L', 'Runoff', 'number', 3),
          C('coefficient', 'Runoff coefficient', '—', 'Runoff', 'number', 3)
        ],
        overview: ['id', 'outlet', 'area', 'impervious', 'category', 'peak', 'runoff', 'coefficient'],
        technical: ['id', 'area', 'peak', 'runoff', 'coefficient']
      }
    };

    // State variables
    // ``active`` is the key of the table schema being shown and is therefore always one
    // of TYPES - never 'map' and never 'info', which have no schema, no filters and no
    // sort.  ``currentTab`` is the tab the user is looking at, which may be either.
    let active = 'conduits';
    const mapAvailable = (function () {
      const script = document.getElementById('network-map-data');
      if (!script) return false;
      try {
        return JSON.parse(script.textContent || '{}').available === true;
      } catch (_) {
        return false;
      }
    })();
    let currentTab = mapAvailable ? 'map' : 'conduits';
    // The element the map asked the table to show, as ``{kind, name}`` — ``kind`` is a
    // tab key, because a name is unique inside one collection and nowhere else.  The
    // state lives here and the class .r-row-hl is derived from it on every render:
    // a class put on the row by hand would vanish on the first sort, which rebuilds
    // the whole <tbody>.
    let highlight = null;
    let openColumns = false;
    let expanded = null;
    let lastExportURL = null;

    const design = { density: 'compact', stripe: false };
    const prefs = { columns: {}, filters: {}, sorts: {} };

    TYPES.forEach(k => {
      prefs.columns[k] = [...schemas[k].overview];
      prefs.filters[k] = { query: '', select: 'all', failed: false };
      prefs.sorts[k] = { key: 'id', dir: 1 };
    });

    const storageKey = 'stormwater-results-v2-preferences';

    function announce(text) {
      const el = $('#r-announcement');
      if (el) el.textContent = text;
    }

    function savePreferences() {
      try {
        localStorage.setItem(storageKey, JSON.stringify({
          columns: prefs.columns,
          density: design.density,
          stripe: design.stripe
        }));
      } catch (_) {}
    }

    function restorePreferences() {
      try {
        const saved = JSON.parse(localStorage.getItem(storageKey) || 'null');
        if (saved && typeof saved === 'object') {
          TYPES.forEach(k => {
            if (Array.isArray(saved.columns?.[k])) {
              const validKeys = schemas[k].cols.map(c => c.key);
              const restored = saved.columns[k].filter(key => key !== 'id' && validKeys.includes(key));
              prefs.columns[k] = restored.length ? ['id', ...restored] : [...schemas[k].overview];
            }
          });
          if (['compact', 'comfortable'].includes(saved.density)) {
            design.density = saved.density;
          }
          design.stripe = Boolean(saved.stripe);
        }
      } catch (_) {}
    }

    restorePreferences();

    function columns() {
      return schemas[active].cols.filter(c => prefs.columns[active].includes(c.key));
    }

    function sortValue(row, key) {
      if (key === 'action' || key === 'recommendation') {
        return getActionLabel(row[key]);
      }
      if (key === 'rule' || key === 'rule_label') {
        return getActionLabel(row[key]);
      }
      let v = row[key];
      if (v === undefined) {
        // Safe property alias resolution
        const aliases = {
          coverIn: 'inlet_ground_cover',
          coverOut: 'outlet_ground_cover',
          minDiameter: 'min_diameter',
          minSlope: 'min_required_slope',
          flow: 'max_q',
          velocity: 'max_v',
          waterDepth: 'max_water_depth',
          floodHours: 'hours_flooded',
          floodVolume: 'flood_volume',
          runoff: 'total_runoff_mg',
          peak: 'peak_runoff',
          coefficient: 'runoff_coeff',
          impervious: 'perc_imperv',
          action: 'recommendation',
          rule: 'rule_label',
          category: 'sbc_category'
        };
        if (aliases[key]) v = row[aliases[key]];
      }
      return v;
    }

    function rowMatchesSearch(r, q, cols) {
      if (!q) return true;

      // 1. Search visible columns
      for (const c of cols) {
        let val = r[c.key];
        if (val === undefined) {
          val = sortValue(r, c.key);
        }
        if (c.kind === 'action' || c.kind === 'rule') {
          val = getActionLabel(val);
        } else if (c.kind === 'checks') {
          val = (r.validation ?? '') + '/7';
        } else if (c.kind === 'fill') {
          val = (r.fillPercent ?? '') + '%';
        }
        if (val !== null && val !== undefined && String(val).toLowerCase().includes(q)) {
          return true;
        }
      }

      // 2. Also search core identifiers regardless of column visibility
      const extraKeys = ['id', 'name', 'inlet', 'inlet_node', 'outlet', 'outlet_node', 'catchment', 'subcatchment', 'category', 'sbc_category'];
      for (const k of extraKeys) {
        const val = r[k];
        if (val !== null && val !== undefined && String(val).toLowerCase().includes(q)) {
          return true;
        }
      }

      return false;
    }

    function rows() {
      const f = prefs.filters[active];
      const q = f.query.trim().toLowerCase();
      const cols = columns();
      const sort = prefs.sorts[active];

      return schemas[active].rows.filter(r => {
        // Text Search
        if (q && !rowMatchesSearch(r, q, cols)) {
          return false;
        }

        // Conduits tab filters
        if (active === 'conduits') {
          const valCount = r.validation ?? (r.flags ? r.flags.filter(Boolean).length : 7);
          if (f.failed && valCount === 7) {
            return false;
          }
          const recNorm = String(r.action || r.recommendation || '').trim().toLowerCase();
          if (f.select === 'changes' && recNorm === 'valid') {
            return false;
          }
          if (f.select !== 'all' && f.select !== 'changes') {
            if (recNorm !== f.select.toLowerCase()) {
              return false;
            }
          }
        }

        // Nodes tab filters
        if (active === 'nodes') {
          const floodVol = r.floodVolume ?? r.flood_volume ?? 0;
          const floodHrs = r.floodHours ?? r.hours_flooded ?? 0;
          const isFlooded = r.is_flooded ?? (floodVol > 0 || floodHrs > 0);
          if (f.failed && !isFlooded) {
            return false;
          }
        }

        // Subcatchments tab filters
        if (active === 'catchments') {
          if (f.select !== 'all') {
            const cat = r.category || r.sbc_category || '';
            if (cat !== f.select) {
              return false;
            }
          }
        }

        return true;
      }).sort((a, b) => {
        const x = sortValue(a, sort.key);
        const y = sortValue(b, sort.key);

        if ((x === null || x === undefined) && (y === null || y === undefined)) return 0;
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;

        if (typeof x === 'number' && typeof y === 'number') {
          return sort.dir * (x - y);
        }
        return sort.dir * collator.compare(String(x), String(y));
      });
    }

    function activeFilters() {
      const f = prefs.filters[active];
      return Boolean(f.query.trim() || f.select !== 'all' || f.failed);
    }

    function renderControls() {
      const f = prefs.filters[active];
      const searchInput = $('#r-search');
      if (searchInput) {
        searchInput.value = f.query;
        searchInput.placeholder = active === 'conduits'
          ? 'Conduit, node or category…'
          : active === 'nodes'
            ? 'Search node name…'
            : 'Subcatchment or category…';
      }

      const failedLabel = $('#r-failed-label');
      const failedCheckbox = $('#r-failed');
      const failedText = $('#r-failed-text');

      if (failedLabel) failedLabel.hidden = (active === 'catchments');
      if (failedCheckbox) failedCheckbox.checked = f.failed;
      if (failedText) {
        failedText.textContent = active === 'nodes' ? 'Flooding only' : 'Failed checks only';
      }

      const selectField = $('#r-select-field');
      const selectLabel = $('#r-select-label');
      const filterSelect = $('#r-filter-select');

      if (selectField) selectField.hidden = (active === 'nodes');

      if (selectLabel) {
        selectLabel.textContent = active === 'conduits' ? 'Recommendation' : 'Category';
      }

      if (filterSelect) {
        if (active === 'conduits') {
          let opts = '<option value="all">All recommendations</option><option value="changes">Changes recommended</option>';
          opts += Object.entries(ACTIONS).map(([k, label]) => `<option value="${k}">${esc(label)}</option>`).join('');
          filterSelect.innerHTML = opts;
        } else if (active === 'catchments') {
          const uniqueCats = Array.from(
            new Set(schemas.catchments.rows.map(r => r.category || r.sbc_category).filter(Boolean))
          ).sort();

          let opts = '<option value="all">All categories</option>';
          opts += uniqueCats.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('');
          filterSelect.innerHTML = opts;
        }

        filterSelect.value = f.select;
        if (filterSelect.value !== f.select) {
          f.select = 'all';
          filterSelect.value = 'all';
        }
      }
    }

    function sameArrays(a, b) {
      return a.length === b.length && a.every(x => b.includes(x));
    }

    function syncColumns() {
      $$('[data-col]').forEach(input => {
        input.checked = prefs.columns[active].includes(input.dataset.col);
      });

      $$('[data-preset]').forEach(b => {
        const presetKey = b.dataset.preset;
        const keys = presetKey === 'all'
          ? schemas[active].cols.map(c => c.key)
          : schemas[active][presetKey];
        b.setAttribute('aria-pressed', String(sameArrays(prefs.columns[active], keys)));
      });

      const colSelection = $('#r-column-selection');
      if (colSelection) colSelection.hidden = !openColumns;

      const colBtn = $('#r-columns');
      if (colBtn) colBtn.setAttribute('aria-expanded', String(openColumns));
    }

    function renderColumns() {
      const s = schemas[active];
      const groups = [...new Set(s.cols.map(c => c.group))];

      const groupsContainer = $('#r-column-groups');
      if (groupsContainer) {
        groupsContainer.innerHTML = groups.map(g => `
          <fieldset>
            <legend>${esc(g)}</legend>
            ${s.cols.filter(c => c.group === g).map(c => `
              <label class="r-option">
                <input type="checkbox" data-col="${c.key}" ${prefs.columns[active].includes(c.key) ? 'checked' : ''} ${c.key === 'id' ? 'disabled' : ''}>
                <span>${esc(c.label)}${c.key === 'id' ? ' · pinned' : ''}</span>
              </label>
            `).join('')}
          </fieldset>
        `).join('');
      }

      const techPreset = $('#r-technical-preset');
      if (techPreset) {
        techPreset.textContent = active === 'conduits'
          ? 'Hydraulics'
          : (active === 'nodes' ? 'Geometry' : 'Runoff');
      }

      syncColumns();
    }

    function validationInfo(r) {
      const diameter = r.diameter ?? r.geom1 ?? 0;
      const filling = r.filling ?? r.filling_m ?? 0;
      const velocity = r.velocity ?? r.max_v ?? 0;
      const slope = r.slope ?? r.slope_per_mile ?? 0;
      const minSlope = r.minSlope ?? r.min_required_slope ?? 0;
      const maxSlope = r.maxSlope ?? r.max_allowable_slope ?? 0;
      const depth = r.depth ?? Math.max(r.inlet_max_depth || 0, r.outlet_max_depth || 0);
      const coverIn = r.coverIn ?? r.inlet_ground_cover ?? 0;
      const coverOut = r.coverOut ?? r.outlet_ground_cover ?? 0;
      const minCover = Math.min(coverIn, coverOut);
      const frostZone = rawData.session?.frost_zone ?? 0.80;
      const slopeRangeEmpty = Number.isFinite(Number(minSlope)) && Number.isFinite(Number(maxSlope)) && Number(minSlope) > Number(maxSlope);
      const minSlopeThreshold = slopeRangeEmpty
        ? 'no feasible slope (min ' + fmt(minSlope, 2) + ' > max ' + fmt(maxSlope, 2) + ' ‰)'
        : '≥ ' + fmt(minSlope, 2) + ' ‰';

      return [
        [fmt(filling, 3) + ' m', '≤ ' + fmt(0.827 * diameter, 3) + ' m'],
        [fmt(velocity, 2) + ' m/s', '≤ 5.00 m/s'],
        [fmt(velocity, 2) + ' m/s', '≥ 0.70 m/s'],
        [fmt(slope, 2) + ' ‰', '≤ ' + fmt(maxSlope, 2) + ' ‰'],
        [fmt(slope, 2) + ' ‰', minSlopeThreshold],
        [fmt(depth, 2) + ' m', '≤ 8.00 m'],
        [fmt(minCover, 2) + ' m', '≥ ' + fmt(frostZone, 2) + ' m']
      ];
    }

    function checkGrid(r) {
      const measures = validationInfo(r);
      const flags = r.flags || [
        r.val_max_fill == 1,
        r.val_max_v == 1,
        r.val_min_v == 1,
        r.val_max_slope == 1,
        r.val_min_slope == 1,
        r.val_depth == 1,
        r.val_coverage == 1
      ];

      return `
        <div class="r-checkgrid">
          ${CHECKS.map((name, i) => {
            const isPassed = Boolean(flags[i]);
            return `
              <div class="r-check ${isPassed ? '' : 'r-failed'}">
                <strong>
                  <i class="fas ${isPassed ? 'fa-check' : 'fa-exclamation-triangle'} me-1" aria-hidden="true"></i>
                  ${isPassed ? 'Passed' : 'Failed'}
                </strong>
                <span>${esc(name)}</span>
                <small>${measures[i][0]} · ${measures[i][1]}</small>
              </div>
            `;
          }).join('')}
        </div>
      `;
    }

    function cell(c, r) {
      const v = r[c.key];

      if (c.kind === 'id') {
        const panelUrl = r.panel_url || r.panelUrl || r.detail_url || r.link;
        if (active === 'conduits') {
          return panelUrl
            ? `<a class="r-id" href="${esc(panelUrl)}" title="Open conduit ${esc(r.id)} panel">${esc(r.id)}</a>`
            : `<span class="r-id">${esc(r.id)}</span>`;
        }

        if (active === 'nodes') {
          const floodVol = r.floodVolume ?? r.flood_volume ?? 0;
          const floodHrs = r.floodHours ?? r.hours_flooded ?? 0;
          const isFlooded = r.is_flooded ?? (floodVol > 0 || floodHrs > 0);

          let linkHtml = panelUrl
            ? `<a class="r-id" href="${esc(panelUrl)}" title="View conduit for node ${esc(r.id)}">${esc(r.id)}</a>`
            : `<span class="r-id">${esc(r.id)}</span>`;

          if (isFlooded) {
            // flood_volume is SWMM's Total Flood Volume in 10⁶ litres; 1 × 10⁶ l = 1000 m³.
            const volTitle = floodVol > 0
              ? `Flooding reported: ${fmt(floodVol * 1000, 1)} m³`
              : 'Flooding reported';
            linkHtml += ` <span class="r-floodmark" title="${volTitle}" aria-label="Flooding reported"><i class="fas fa-exclamation-triangle" aria-hidden="true"></i></span>`;
          }
          return linkHtml;
        }

        if (active === 'catchments') {
          return panelUrl
            ? `<a class="r-id" href="${esc(panelUrl)}" title="View conduit for subcatchment ${esc(r.id)}">${esc(r.id)}</a>`
            : `<span class="r-id">${esc(r.id)}</span>`;
        }
      }

      if (c.kind === 'checks') {
        const valCount = r.validation ?? (r.flags ? r.flags.filter(Boolean).length : 7);
        const isExp = expanded === r.id;
        const failed = valCount < 7;
        return `
          <button type="button" class="r-checkbutton ${failed ? 'r-failed' : ''}" data-validation="${esc(r.id)}" aria-expanded="${String(isExp)}" aria-controls="r-checks-${esc(r.id)}" aria-label="Validation ${esc(r.id)}: ${valCount} of 7 checks passed. ${isExp ? 'Hide' : 'Show'} checks">
            <i class="fas ${failed ? 'fa-exclamation-circle' : 'fa-check-circle'} me-1" aria-hidden="true"></i>${valCount}/7
            <span aria-hidden="true">${isExp ? '−' : '+'}</span>
          </button>
        `;
      }

      if (c.kind === 'fill') {
        const fillPct = v ?? r.fillPercent ?? r.fill_percent ?? 0;
        const fillingM = r.filling ?? r.filling_m ?? 0;
        const diaM = r.diameter ?? r.geom1 ?? 0;
        const isBad = r.flags ? !r.flags[0] : (fillingM > 0.827 * diaM);
        const clamped = Math.max(0, Math.min(fillPct, 100));
        return `
          <span class="r-fillcell ${isBad ? 'r-fillbad' : ''}" title="${fmt(fillingM, 3)} m / ${fmt(diaM, 3)} m · ${isBad ? 'Filling limit exceeded (>82.7%)' : 'Filling check passed'}">
            <span class="r-meter" aria-hidden="true"><span style="width:${clamped}%"></span></span>
            <b>${isBad ? '! ' : ''}${fmt(fillPct, 0)}%</b>
          </span>
        `;
      }

      if (c.kind === 'action') {
        const act = v ?? r.action ?? r.recommendation;
        const isNormValid = String(act).trim().toLowerCase() === 'valid';
        return `<span class="r-action ${isNormValid ? 'r-valid' : ''}">${esc(getActionLabel(act))}</span>`;
      }

      if (c.kind === 'rule') {
        const ruleVal = v ?? r.rule ?? r.rule_label;
        if (!ruleVal || ruleVal === '-') {
          return `<span class="r-muted">—</span>`;
        }
        const actVal = r.action ?? r.recommendation;
        const isMatch = String(ruleVal).trim().toLowerCase() === String(actVal).trim().toLowerCase();
        return `
          <span class="r-rule">
            <span>${esc(getActionLabel(ruleVal))}</span>
            <span class="${isMatch ? 'r-agreement' : 'r-disagreement'}" title="${isMatch ? 'Matches model recommendation' : 'Differs from model recommendation'}" aria-label="${isMatch ? 'Matches model' : 'Differs from model'}">
              <i class="fas ${isMatch ? 'fa-check' : 'fa-times'}" aria-hidden="true"></i>
            </span>
          </span>
        `;
      }

      if (c.kind === 'category') {
        const cat = v ?? r.category ?? r.sbc_category;
        return `<span class="r-category">${esc(cat || '—')}</span>`;
      }

      if (c.kind === 'number') {
        const numVal = sortValue(r, c.key);
        if (active === 'conduits' && c.key === 'minDiameter' && Number(numVal) === 0) {
          return '<span>—</span>';
        }
        const isFloodNum = active === 'nodes' && ['floodHours', 'floodVolume'].includes(c.key) && numVal > 0;
        return `<span class="${isFloodNum ? 'r-floodnum' : ''}">${fmt(numVal, c.digits)}</span>`;
      }

      return esc(v !== null && v !== undefined ? v : '—');
    }

    function updateOverflow() {
      const area = $('#r-overflow');
      if (!area) return;

      root.style.setProperty('--r-viewport', Math.max(0, area.clientWidth - 2) + 'px');
      const hasOverflow = area.scrollWidth > area.clientWidth + 2;
      const canScrollRight = area.scrollLeft + area.clientWidth < area.scrollWidth - 3;

      const frame = $('#r-tableframe');
      if (frame) frame.dataset.right = String(hasOverflow && canScrollRight);

      const hint = $('#r-scrollhint');
      if (hint) hint.hidden = !hasOverflow;

      const scrollBtn = $('#r-scroll-columns');
      if (scrollBtn) {
        scrollBtn.textContent = canScrollRight ? 'More columns to the right' : 'Back to first columns';
      }
    }

    /**
     * Put .r-row-hl back on the row the map pointed at, if that row is on screen.
     *
     * Called at the very end of ``renderTable`` and from nowhere else: every sort,
     * every filter and every change of columns writes a new <tbody>, so the class has
     * to be derived from the stored state each time rather than applied once.  The
     * name comes from the model file, so it reaches the selector through
     * ``CSS.escape`` — ``A/B`` and ``P 1`` are legal SWMM names and a raw selector
     * built from either of them throws a DOMException.  A highlight belonging to
     * another collection, or naming a row the current filter hides, simply shows
     * nothing; it is kept, because clearing the filter has to bring it back.
     */
    function applyHighlight() {
      if (!highlight || highlight.kind !== active) return;
      const row = $(`#r-table [data-row="${CSS.escape(highlight.name)}"]`);
      if (row) row.classList.add('r-row-hl');
    }

    function renderTable(preserveScroll = true) {
      const area = $('#r-overflow');
      const sx = area ? area.scrollLeft : 0;
      const sy = area ? area.scrollTop : 0;

      const cols = columns();
      const filtered = rows();
      const sort = prefs.sorts[active];

      const table = $('#r-table');
      if (!table) return;

      const captionText = `${schemas[active].name} simulation results`;

      const theadHtml = `
        <caption class="r-sr">${captionText}</caption>
        <thead>
          <tr>
            ${cols.map(c => {
              const isSorted = sort.key === c.key;
              const ariaSort = isSorted ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none';
              const sortIcon = isSorted
                ? (sort.dir === 1 ? '<i class="fas fa-arrow-up" aria-hidden="true"></i>' : '<i class="fas fa-arrow-down" aria-hidden="true"></i>')
                : '<i class="fas fa-sort" aria-hidden="true"></i>';

              return `
                <th scope="col" class="${c.key === 'id' ? 'r-pin' : ''}${['number', 'fill'].includes(c.kind) ? ' r-num' : ''}" aria-sort="${ariaSort}">
                  <button type="button" class="r-sort" data-sort="${c.key}" aria-label="Sort by ${esc(c.label)}">
                    <span>
                      ${esc(c.label)}
                      ${c.unit ? `<span class="r-unit">${esc(c.unit)}</span>` : ''}
                    </span>
                    <span class="r-sortmark" aria-hidden="true">${sortIcon}</span>
                  </button>
                </th>
              `;
            }).join('')}
          </tr>
        </thead>
      `;

      let tbodyHtml = '';
      if (!filtered.length) {
        tbodyHtml = `
          <tbody>
            <tr>
              <td class="r-empty" colspan="${cols.length}">
                <strong>No matching ${schemas[active].name.toLowerCase()}</strong>
                <p>Try refining your search or clear active filters.</p>
                <button type="button" class="r-button" data-clear>
                  <i class="fas fa-filter-circle-xmark me-1" aria-hidden="true"></i>Clear filters
                </button>
              </td>
            </tr>
          </tbody>
        `;
      } else {
        tbodyHtml = '<tbody>' + filtered.map(r => {
          const rowHtml = `
            <tr class="r-row" data-row="${esc(r.id)}">
              ${cols.map(c => `
                <td class="${c.key === 'id' ? 'r-pin' : ''}${['number', 'fill'].includes(c.kind) ? ' r-num' : ''}">
                  ${cell(c, r)}
                </td>
              `).join('')}
            </tr>
          `;

          if (active === 'conduits' && expanded === r.id) {
            const valCount = r.validation ?? (r.flags ? r.flags.filter(Boolean).length : 7);
            const valRowHtml = `
              <tr class="r-validationrow" id="r-checks-${esc(r.id)}">
                <td colspan="${cols.length}">
                  <div class="r-checkpanel">
                    <div class="r-checkpanelhead">
                      <h3>
                        <i class="fas fa-clipboard-check me-2" aria-hidden="true"></i>
                        ${esc(r.id)} · Technical validation · ${valCount}/7 passed
                      </h3>
                      <button type="button" class="r-close" data-close-checks aria-label="Close validation details">×</button>
                    </div>
                    ${checkGrid(r)}
                  </div>
                </td>
              </tr>
            `;
            return rowHtml + valRowHtml;
          }

          return rowHtml;
        }).join('') + '</tbody>';
      }

      table.innerHTML = theadHtml + tbodyHtml;

      if (area) {
        area.scrollLeft = preserveScroll ? sx : 0;
        area.scrollTop = preserveScroll ? sy : 0;
      }

      const colCountEl = $('#r-column-count');
      if (colCountEl) colCountEl.textContent = `${cols.length}/${schemas[active].cols.length}`;

      const footColsEl = $('#r-foot-columns');
      if (footColsEl) footColsEl.textContent = `${cols.length} of ${schemas[active].cols.length} columns`;

      const rowCountEl = $('#r-row-count');
      if (rowCountEl) {
        rowCountEl.textContent = `${filtered.length} of ${schemas[active].rows.length} ${schemas[active].name.toLowerCase()}`;
      }

      const summaryEl = $('#r-filter-summary');
      if (summaryEl) {
        summaryEl.textContent = activeFilters()
          ? `Filtered results · ${filtered.length} matches`
          : '';
      }

      const clearBtn = $('#r-clear-filters');
      if (clearBtn) clearBtn.hidden = !activeFilters();

      updateOverflow();

      // The two things every render owes the rest of the page, in this order.  The
      // highlight is re-derived from the stored state, because the <tbody> above is
      // brand new.  The table then says which names it is showing, so that the map
      // can fade everything else; ``kind`` is a tab key, the vocabulary the table
      // speaks, and the map translates it on its side.  The event is emitted on every
      // render, filtered or not — "everything is visible" is the message that brings
      // a faded network back.
      applyHighlight();
      root.dispatchEvent(new CustomEvent('rv2:filterchange', {
        detail: { kind: active, visible: filtered.map(r => r.id) }
      }));

      return filtered;
    }

    function renderAll(resetScroll = false) {
      root.dataset.density = design.density;
      root.dataset.stripe = String(design.stripe);

      TYPES.forEach(k => {
        const badge = $(`#r-count-${k}`);
        if (badge) badge.textContent = schemas[k].rows.length;
      });
      // Tab selection and panel visibility belong to applyTabState alone: a second
      // writer here would light up Conduits on every filter change while the map is up.

      renderControls();
      renderColumns();
      renderTable(!resetScroll);
    }

    /**
     * Single owner of which panel is visible and which tab reads as selected.
     *
     * Every element is guarded: a page without the map tab or the map panel (an older
     * template, or the harness fixture that has neither) must not throw here.  ARIA
     * state goes through setAttribute, because assigning ``el['aria-selected']`` only
     * creates a property and leaves the DOM untouched.
     */
    function applyTabState(k) {
      const tablePanel = $('#r-table-panel');
      const infoPanel = $('#r-info-panel');
      const mapPanel = $('#r-map-panel');
      if (tablePanel) tablePanel.hidden = (k === 'info' || k === 'map');
      if (infoPanel) infoPanel.hidden = (k !== 'info');
      if (mapPanel) mapPanel.hidden = (k !== 'map');
      TABS.forEach(t => {
        const tab = $(`#r-tab-${t}`);
        if (tab) tab.setAttribute('aria-selected', String(t === k));
      });
      if (tablePanel) tablePanel.setAttribute('aria-labelledby', `r-tab-${active}`);
    }

    function changeTab(k) {
      if (!TABS.includes(k)) return;

      currentTab = k;
      // ``active`` has to be current before applyTabState runs: the label of the table
      // panel is derived from it, and nothing else fixes it afterwards.
      if (TYPES.includes(k)) active = k;
      applyTabState(k);

      // 'info' and 'map' own no table state, so they touch neither prefs nor schemas.
      if (TYPES.includes(k)) {
        expanded = null;
        openColumns = false;
        renderAll(true);
      }

      let message;
      if (k === 'info') {
        message = 'Showing session info';
      } else if (k === 'map') {
        message = 'Showing network map';
      } else {
        message = `Showing ${schemas[k].name}`;
      }
      announce(message);
    }

    function clearFilters() {
      prefs.filters[active] = { query: '', select: 'all', failed: false };
      expanded = null;
      renderControls();
      renderTable(false);
      announce('Filters cleared');
    }

    function exportCSV() {
      const cols = columns();
      const data = rows();
      const sanitizeFormula = s => {
        if (!s || s === '-') return s;
        const first = s[0];
        if (first === '=' || first === '+' || first === '@' || first === '\t' || first === '\r') {
          return "'" + s;
        }
        if (first === '-' && !Number.isFinite(Number(s))) {
          return "'" + s;
        }
        return s;
      };
      const quote = v => '"' + sanitizeFormula(String(v ?? '')).replace(/"/g, '""') + '"';

      function csvValue(c, r) {
        if (c.kind === 'id') {
          return r.id || r.name;
        }
        if (c.kind === 'action' || c.kind === 'rule') {
          return getActionLabel(sortValue(r, c.key));
        }
        if (c.kind === 'checks') {
          const valCount = r.validation ?? (r.flags ? r.flags.filter(Boolean).length : 7);
          const failed = CHECKS.filter((_, i) => r.flags && !r.flags[i]);
          return `${valCount}/7; failed: ${failed.length ? failed.join(', ') : 'none'}`;
        }
        if (c.kind === 'fill') {
          const fillPct = r[c.key] ?? r.fillPercent ?? r.fill_percent;
          return fillPct !== null && fillPct !== undefined && Number.isFinite(Number(fillPct))
            ? Number(fillPct).toFixed(0) + '%'
            : '';
        }
        if (c.kind === 'number') {
          const v = sortValue(r, c.key);
          return v !== null && v !== undefined && Number.isFinite(Number(v))
            ? Number(v).toFixed(c.digits)
            : '';
        }
        if (c.kind === 'category') {
          return r[c.key] ?? r.category ?? r.sbc_category ?? '';
        }
        const v = r[c.key];
        return v !== null && v !== undefined ? String(v) : '';
      }

      const header = cols.map(c => quote(c.label + (c.unit ? ` [${c.unit}]` : ''))).join(',');
      const bodyLines = data.map(r => cols.map(c => quote(csvValue(c, r))).join(','));
      const csvContent = '\uFEFF' + [header, ...bodyLines].join('\r\n');

      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      if (lastExportURL) {
        URL.revokeObjectURL(lastExportURL);
      }
      lastExportURL = URL.createObjectURL(blob);

      const sessionId = rawData.session?.id || 'session';
      const a = document.createElement('a');
      a.href = lastExportURL;
      a.download = `session-${sessionId}-${active}.csv`;
      a.hidden = true;
      document.body.appendChild(a);
      a.click();
      a.remove();

      announce(`CSV exported: ${data.length} rows and ${cols.length} columns.`);
    }

    // Event Bindings
    $$('[data-tab]').forEach(b => {
      b.addEventListener('click', () => changeTab(b.dataset.tab));
      b.addEventListener('keydown', e => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
        e.preventDefault();
        const allTabs = TABS;
        const curTab = b.dataset.tab;
        const curIdx = allTabs.indexOf(curTab);
        const nextIdx = e.key === 'Home'
          ? 0
          : e.key === 'End'
            ? allTabs.length - 1
            : (curIdx + (e.key === 'ArrowRight' ? 1 : -1) + allTabs.length) % allTabs.length;
        changeTab(allTabs[nextIdx]);
        const nextTab = $(`#r-tab-${allTabs[nextIdx]}`);
        if (nextTab) nextTab.focus();
      });
    });


    const searchEl = $('#r-search');
    if (searchEl) {
      searchEl.addEventListener('input', e => {
        prefs.filters[active].query = e.target.value;
        expanded = null;
        const filtered = renderTable(false);
        announce(`${filtered.length} matching results`);
      });
    }

    const filterSelectEl = $('#r-filter-select');
    if (filterSelectEl) {
      filterSelectEl.addEventListener('change', e => {
        prefs.filters[active].select = e.target.value;
        expanded = null;
          const filtered = renderTable(false);
        announce(`${filtered.length} matching results`);
      });
    }

    const failedEl = $('#r-failed');
    if (failedEl) {
      failedEl.addEventListener('change', e => {
        prefs.filters[active].failed = e.target.checked;
        expanded = null;
          const filtered = renderTable(false);
        announce(`${filtered.length} matching results`);
      });
    }

    const clearFiltersBtn = $('#r-clear-filters');
    if (clearFiltersBtn) {
      clearFiltersBtn.addEventListener('click', clearFilters);
    }

    const columnsBtn = $('#r-columns');
    if (columnsBtn) {
      columnsBtn.addEventListener('click', () => {
        openColumns = !openColumns;
        syncColumns();
      });
    }

    const closeColumnsBtn = $('#r-close-columns');
    if (closeColumnsBtn) {
      closeColumnsBtn.addEventListener('click', () => {
        openColumns = false;
        syncColumns();
        if (columnsBtn) columnsBtn.focus();
      });
    }

    const columnGroupsContainer = $('#r-column-groups');
    if (columnGroupsContainer) {
      columnGroupsContainer.addEventListener('change', e => {
        const key = e.target.dataset.col;
        if (!key || key === 'id') return;

        const isChecked = e.target.checked;
        if (isChecked && !prefs.columns[active].includes(key)) {
          prefs.columns[active].push(key);
        } else if (!isChecked) {
          prefs.columns[active] = prefs.columns[active].filter(c => c !== key);
        }

        if (key === 'validation' && !isChecked) {
          expanded = null;
        }

        syncColumns();
        renderTable();
        savePreferences();

        const colDef = schemas[active].cols.find(c => c.key === key);
        if (colDef) {
          announce((isChecked ? 'Shown: ' : 'Hidden: ') + colDef.label);
        }
      });
    }

    $$('[data-preset]').forEach(b => {
      b.addEventListener('click', () => {
        const presetKey = b.dataset.preset;
        prefs.columns[active] = [
          ...(presetKey === 'all'
            ? schemas[active].cols.map(c => c.key)
            : schemas[active][presetKey])
        ];
        expanded = null;
        syncColumns();
        renderTable(false);
        savePreferences();
        announce(`Column preset: ${b.textContent}`);
      });
    });

    const exportBtn = $('#r-export');
    if (exportBtn) {
      exportBtn.addEventListener('click', exportCSV);
    }

    const tableEl = $('#r-table');
    if (tableEl) {
      tableEl.addEventListener('click', e => {
        const sortBtn = e.target.closest('[data-sort]');
        const validateBtn = e.target.closest('[data-validation]');
        const closeChecksBtn = e.target.closest('[data-close-checks]');
        const clearBtn = e.target.closest('[data-clear]');

        if (sortBtn) {
          const sortKey = sortBtn.dataset.sort;
          const cur = prefs.sorts[active];
          prefs.sorts[active] = {
            key: sortKey,
            dir: cur.key === sortKey ? -cur.dir : 1
          };
          renderTable(false);
          const updatedBtn = $(`#r-table [data-sort="${sortKey}"]`);
          if (updatedBtn) updatedBtn.focus();
        } else if (validateBtn) {
          const conduitId = validateBtn.dataset.validation;
          expanded = (expanded === conduitId) ? null : conduitId;
          renderTable();
          const targetBtn = $(`#r-table [data-validation="${conduitId}"]`);
          if (targetBtn) targetBtn.focus();
          announce(expanded ? `Validation details for ${expanded}` : 'Validation details closed');
        } else if (closeChecksBtn) {
          const prior = expanded;
          expanded = null;
          renderTable();
          if (prior) {
            const priorBtn = $(`#r-table [data-validation="${prior}"]`);
            if (priorBtn) priorBtn.focus();
          }
        } else if (clearBtn) {
          clearFilters();
        }
      });
    }

    /**
     * The map asked for an element to be shown in the table (stage 7, D10 variant a).
     *
     * The only direction there is: under the tabbed layout the map and the table are
     * never on screen together, so nothing sends the table's selection back to a map
     * nobody can see.  The event is sent by the ``Show in table`` button of the
     * selection card and never by selecting an element, and its ``kind`` is already a
     * tab key.
     *
     * The order of the three steps is the whole point.  The state is stored *first*,
     * so that the ``renderTable`` which ``changeTab`` runs synchronously ends on an
     * ``applyHighlight`` that can already see it; only then is the row looked up, to
     * be scrolled to.  Applying the class by hand after ``changeTab`` would work once
     * and disappear on the reader's first sort.
     */
    root.addEventListener('nm:select', e => {
      const detail = e && e.detail ? e.detail : null;
      if (!detail || typeof detail.name !== 'string' || !TYPES.includes(detail.kind)) return;

      highlight = { kind: detail.kind, name: detail.name };
      changeTab(detail.kind);

      // A name with no row is a normal state — the reader may have left a filter on —
      // and it costs the highlight nothing: the class comes back when the filter does.
      const row = $(`#r-table [data-row="${CSS.escape(detail.name)}"]`);
      if (row && typeof row.scrollIntoView === 'function') row.scrollIntoView({ block: 'nearest' });
    });

    root.addEventListener('keydown', e => {
      if (e.key === 'Escape' && openColumns) {
        openColumns = false;
        syncColumns();
        if (columnsBtn) columnsBtn.focus();
      }
    });

    const overflowArea = $('#r-overflow');
    if (overflowArea) {
      overflowArea.addEventListener('scroll', updateOverflow, { passive: true });
    }

    const scrollColsBtn = $('#r-scroll-columns');
    if (scrollColsBtn && overflowArea) {
      scrollColsBtn.addEventListener('click', () => {
        const canRight = overflowArea.scrollLeft + overflowArea.clientWidth < overflowArea.scrollWidth - 3;
        overflowArea.scrollTo({
          left: canRight
            ? Math.min(overflowArea.scrollWidth, overflowArea.scrollLeft + overflowArea.clientWidth * 0.65)
            : 0,
          behavior: window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches
            ? 'auto'
            : 'smooth'
        });
      });
    }

    renderAll();
    applyTabState(currentTab);

    if (overflowArea && window.ResizeObserver) {
      new ResizeObserver(updateOverflow).observe(overflowArea);
    }
    window.addEventListener('resize', updateOverflow);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initResultsV2);
  } else {
    initResultsV2();
  }
})();
