/*
 * method-graph.js - the "What the model reads" figure of the documentation.
 *
 * Progressive enhancement of every [data-method-graph] figure: it fetches the
 * figure data from data-src (written by `manage.py export_method_graph`), draws
 * the network as an inline SVG and keeps a readout beside it. Until the data has
 * arrived the figure shows its static fallback sentence; if the fetch fails, that
 * sentence simply stays.
 *
 * Data: conduits with their paths in a local frame (y pointing down), and for
 * every conduit two rows of weights - `step1`, the row of the aggregation matrix
 * P (one graph step), and `step2`, the row of P·P (two steps).
 *
 * View: a neighbourhood is a few links long, so the map shows a window of the
 * network around the selection (a fixed zoom, widened only when a neighbourhood
 * does not fit) and pans when the selection changes. An overview of the whole
 * network sits in a corner of the map and marks the window.
 *
 * Interaction: click or tap a conduit (on the map or in the overview), pick it in
 * the <select>, or focus the map (one tab stop) and move along the network with
 * the arrow keys. The selection persists; hovering only names the conduit under
 * the pointer.
 *
 * Geometry is laid out in CSS pixels of the current map width, so strokes, arrows
 * and labels keep their size on every screen.
 */

(function () {
    'use strict';

    var SVG_NS = 'http://www.w3.org/2000/svg';
    var ZOOM = 2.4; // window width = network width / ZOOM
    var WINDOW_MARGIN = 34; // frame units kept around a neighbourhood
    var EDGE = 30; // frame units the window may extend past the network
    var PAN_MS = 320;
    var HIT_RADIUS = 12; // px: pointer distance that still picks a conduit
    var HIT_RADIUS_TOUCH = 22;
    var ARROW_MIN_LENGTH = 18; // px: shorter conduits get no arrowhead
    var LIST_LIMIT = 8;
    var LABEL_HEIGHT = 14;
    var LABEL_PAD_X = 3;
    var LABEL_RADII = [5, 12, 20, 30, 42, 56];
    var LABEL_ANGLES = 16;
    var INSET_GAP = 8; // px between the overview and the edge of the map

    // ------------------------------------------------------------------
    // Small helpers
    // ------------------------------------------------------------------

    function svgEl(name, attrs, cls) {
        var el = document.createElementNS(SVG_NS, name);
        if (cls) {
            el.setAttribute('class', cls);
        }
        if (attrs) {
            Object.keys(attrs).forEach(function (key) {
                el.setAttribute(key, attrs[key]);
            });
        }
        return el;
    }

    function htmlEl(name, cls, text) {
        var el = document.createElement(name);
        if (cls) {
            el.className = cls;
        }
        if (text !== undefined) {
            el.textContent = text;
        }
        return el;
    }

    function clear(el) {
        while (el.firstChild) {
            el.removeChild(el.firstChild);
        }
    }

    function round(value) {
        return Math.round(value * 10) / 10;
    }

    function formatShare(weight) {
        var pct = weight * 100;
        return pct > 0 && pct < 0.1 ? '<0.1%' : pct.toFixed(1) + '%';
    }

    function plural(count, one, many) {
        return count + ' ' + (count === 1 ? one : many);
    }

    function pathData(points) {
        return points.map(function (p, i) {
            return (i ? 'L' : 'M') + round(p[0]) + ' ' + round(p[1]);
        }).join(' ');
    }

    function prefersReducedMotion() {
        return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    }

    // Squared distance from point (px, py) to segment a-b.
    function segmentDistance2(px, py, ax, ay, bx, by) {
        var dx = bx - ax;
        var dy = by - ay;
        var len2 = dx * dx + dy * dy;
        var t = len2 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
        t = Math.max(0, Math.min(1, t));
        var qx = ax + t * dx - px;
        var qy = ay + t * dy - py;
        return qx * qx + qy * qy;
    }

    // Liang-Barsky: does segment [x0, y0, x1, y1] touch the box?
    function segmentHitsBox(seg, box) {
        var x0 = seg[0];
        var y0 = seg[1];
        var dx = seg[2] - x0;
        var dy = seg[3] - y0;
        var p = [-dx, dx, -dy, dy];
        var q = [x0 - box.x, box.x + box.w - x0, y0 - box.y, box.y + box.h - y0];
        var t0 = 0;
        var t1 = 1;
        for (var i = 0; i < 4; i++) {
            if (p[i] === 0) {
                if (q[i] < 0) {
                    return false;
                }
            } else {
                var r = q[i] / p[i];
                if (p[i] < 0) {
                    t0 = Math.max(t0, r);
                } else {
                    t1 = Math.min(t1, r);
                }
                if (t0 > t1) {
                    return false;
                }
            }
        }
        return true;
    }

    function segmentsCross(s, t) {
        function orient(ax, ay, bx, by, cx, cy) {
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        }
        var d1 = orient(t[0], t[1], t[2], t[3], s[0], s[1]);
        var d2 = orient(t[0], t[1], t[2], t[3], s[2], s[3]);
        var d3 = orient(s[0], s[1], s[2], s[3], t[0], t[1]);
        var d4 = orient(s[0], s[1], s[2], s[3], t[2], t[3]);
        return d1 * d2 < 0 && d3 * d4 < 0;
    }

    function boxesOverlap(a, b, margin) {
        return a.x < b.x + b.w + margin && b.x < a.x + a.w + margin && a.y < b.y + b.h + margin && b.y < a.y + a.h + margin;
    }

    function inflate(box, by) {
        return { x: box.x - by, y: box.y - by, w: box.w + 2 * by, h: box.h + 2 * by };
    }

    // Point on a polyline at half its length, with the unit direction there.
    function midpointOf(points) {
        var total = 0;
        var i;
        for (i = 1; i < points.length; i++) {
            total += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
        }
        var half = total / 2;
        for (i = 1; i < points.length; i++) {
            var a = points[i - 1];
            var b = points[i];
            var len = Math.hypot(b[0] - a[0], b[1] - a[1]);
            if (half <= len || i === points.length - 1) {
                var t = len ? Math.min(1, half / len) : 0;
                return {
                    x: a[0] + (b[0] - a[0]) * t,
                    y: a[1] + (b[1] - a[1]) * t,
                    ux: len ? (b[0] - a[0]) / len : 1,
                    uy: len ? (b[1] - a[1]) / len : 0,
                    length: total
                };
            }
            half -= len;
        }
        return { x: points[0][0], y: points[0][1], ux: 1, uy: 0, length: 0 };
    }

    function hitRadius(pointerType) {
        return pointerType === 'mouse' ? HIT_RADIUS : HIT_RADIUS_TOUCH;
    }

    function setText(el, text) {
        if (el.textContent !== text) {
            el.textContent = text;
        }
    }

    function easeInOut(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    // ------------------------------------------------------------------
    // Label placement: greedy, in priority order. Each label tries positions
    // around its anchor - nearest ring first, directions closest to the
    // conduit's normal first - and takes the first one that stays inside the
    // map and clear of every line, obstacle box, earlier label and earlier
    // leader. A label that finds no place is left out; the readout still lists
    // every conduit in view (the tail of a long list under "+ n more").
    // ------------------------------------------------------------------

    function placeLabels(items, segments, boxes, width, height) {
        var placed = [];
        var leaders = [];

        function fits(box) {
            if (box.x < 2 || box.y < 2 || box.x + box.w > width - 2 || box.y + box.h > height - 2) {
                return false;
            }
            var grown = inflate(box, 2);
            return placed.every(function (other) {
                return !boxesOverlap(box, other.box, 3);
            }) && boxes.every(function (other) {
                return !boxesOverlap(box, other, 2);
            }) && segments.every(function (seg) {
                return !segmentHitsBox(seg, grown);
            }) && leaders.every(function (seg) {
                return !segmentHitsBox(seg, grown);
            });
        }

        function leaderFor(item, box) {
            var ex = Math.max(box.x, Math.min(item.x, box.x + box.w));
            var ey = Math.max(box.y, Math.min(item.y, box.y + box.h));
            var len = Math.hypot(ex - item.x, ey - item.y) || 1;
            var leader = [item.x + (ex - item.x) * (4 / len), item.y + (ey - item.y) * (4 / len), ex, ey];
            var free = placed.every(function (other) {
                return !segmentHitsBox(leader, inflate(other.box, 1));
            }) && segments.every(function (seg) {
                return seg.id === item.id || !segmentsCross(leader, seg);
            }) && leaders.every(function (seg) {
                return !segmentsCross(leader, seg);
            });
            return free ? leader : null;
        }

        items.forEach(function (item) {
            var normal = Math.atan2(item.ny, item.nx);
            var angles = [];
            for (var k = 0; k < LABEL_ANGLES; k++) {
                var a = (k / LABEL_ANGLES) * 2 * Math.PI;
                var diff = Math.abs(Math.atan2(Math.sin(a - normal), Math.cos(a - normal)));
                angles.push({ a: a, cost: Math.min(diff, Math.PI - diff) });
            }
            angles.sort(function (p, q) {
                return p.cost - q.cost || p.a - q.a;
            });

            for (var r = 0; r < LABEL_RADII.length; r++) {
                var radius = LABEL_RADII[r];
                for (var j = 0; j < angles.length; j++) {
                    var cos = Math.cos(angles[j].a);
                    var sin = Math.sin(angles[j].a);
                    // The box touches the candidate point with its side facing the anchor.
                    var box = {
                        x: item.x + cos * radius + (cos - 1) * item.w / 2,
                        y: item.y + sin * radius + (sin - 1) * item.h / 2,
                        w: item.w,
                        h: item.h
                    };
                    if (!fits(box)) {
                        continue;
                    }
                    var leader = null;
                    if (r > 0) {
                        leader = leaderFor(item, box);
                        if (!leader) {
                            continue;
                        }
                    }
                    placed.push({ item: item, box: box, leader: leader });
                    if (leader) {
                        leaders.push(leader);
                    }
                    return;
                }
            }
        });
        return placed;
    }

    // ------------------------------------------------------------------
    // One figure
    // ------------------------------------------------------------------

    function MethodGraph(root, data) {
        this.root = root;
        this.data = data;
        this.svg = root.querySelector('[data-method-graph-svg]');
        this.mapMount = root.querySelector('[data-method-graph-map]');
        this.readoutMount = root.querySelector('[data-method-graph-readout]');
        this.select = root.querySelector('[data-method-graph-select]');
        this.stepInputs = Array.prototype.slice.call(root.querySelectorAll('[data-method-graph-steps]'));
        this.conduits = data.conduits;
        this.byId = {};
        this.position = {};
        this.upstream = {};
        this.downstream = {};
        this.textWidths = {};
        this.strokes = {};
        this.width = 0;
        this.view = null;
        this.hoverId = null;
        this.pointer = null; // last pointer position over the map, for hover after a pan
        this.gesturePointer = 'mouse'; // pointer type of the press that ends in a click
        this.panFrame = null;
        this.readoutKey = null;

        var checked = this.stepInputs.filter(function (input) {
            return input.checked;
        })[0];
        this.steps = checked && checked.value === '1' ? 1 : 2;

        this.indexNetwork();
        this.selected = this.byId[data.default] ? data.default : this.conduits[0].id;
    }

    MethodGraph.prototype.indexNetwork = function () {
        var self = this;
        var byInlet = {};
        this.conduits.forEach(function (c, i) {
            self.byId[c.id] = c;
            self.position[c.id] = i;
            (byInlet[c.from] = byInlet[c.from] || []).push(c.id);
            self.upstream[c.id] = [];
        });
        this.conduits.forEach(function (c) {
            self.downstream[c.id] = byInlet[c.to] || [];
            self.downstream[c.id].forEach(function (next) {
                self.upstream[next].push(c.id);
            });
        });
    };

    // Swap the static fallback for the interactive parts, or back again. The
    // stylesheet keeps the room of both reserved, so nothing moves when it flips.
    MethodGraph.prototype.setReady = function (ready) {
        var fallback = this.root.querySelector('[data-method-graph-fallback]');
        var vb = this.data.viewBox;
        this.root.style.setProperty('--mg-map-aspect', vb[0] + ' / ' + vb[1]);
        this.root.classList.toggle('is-ready', ready);
        if (fallback) {
            fallback.hidden = ready;
        }
    };

    MethodGraph.prototype.focusIsOnMap = function () {
        return !!document.activeElement && this.hitLayer.contains(document.activeElement);
    };

    MethodGraph.prototype.start = function () {
        var self = this;
        this.buildSvg();
        this.buildReadout();
        this.buildSelect();

        this.stepInputs.forEach(function (input) {
            input.addEventListener('change', function () {
                if (input.checked) {
                    self.steps = input.value === '1' ? 1 : 2;
                    self.refresh(false);
                }
            });
        });

        // A click uses the radius of the pointer that made it, the same one hover
        // feedback uses, so it never picks a conduit that was not highlighted.
        this.svg.addEventListener('pointerdown', function (event) {
            self.gesturePointer = event.pointerType || 'mouse';
        });
        this.svg.addEventListener('click', function (event) {
            var id = self.conduitAt(event.clientX, event.clientY, hitRadius(self.gesturePointer));
            if (id) {
                self.choose(id, self.focusIsOnMap());
            }
        });
        this.svg.addEventListener('pointermove', function (event) {
            self.pointer = { x: event.clientX, y: event.clientY, type: event.pointerType || 'mouse' };
            self.hoverFromPointer();
        });
        this.svg.addEventListener('pointerleave', function () {
            self.pointer = null;
            self.setHover(null);
        });

        this.hitLayer.addEventListener('keydown', function (event) {
            self.onKey(event);
        });
        // Focus listeners sit on the figure, not on an SVG element: Chromium makes any
        // SVG element with focus listeners a tab stop of its own.
        this.root.addEventListener('focusin', function (event) {
            var visible = false;
            if (self.hitLayer.contains(event.target)) {
                try {
                    visible = event.target.matches(':focus-visible');
                } catch (e) {
                    visible = true;
                }
            }
            self.root.classList.toggle('is-map-focused', visible);
        });
        this.root.addEventListener('focusout', function (event) {
            if (self.hitLayer.contains(event.target)) {
                self.root.classList.remove('is-map-focused');
            }
        });

        this.setReady(true);
        this.layout();
        this.statusEl.setAttribute('aria-live', 'polite');

        if (typeof ResizeObserver === 'function') {
            var pending = false;
            new ResizeObserver(function () {
                if (!pending) {
                    pending = true;
                    window.requestAnimationFrame(function () {
                        pending = false;
                        self.layout();
                    });
                }
            }).observe(this.mapMount);
        } else {
            window.addEventListener('resize', function () {
                self.layout();
            });
        }
        if (document.fonts && document.fonts.ready) {
            document.fonts.ready.then(function () {
                self.textWidths = {};
                self.finish();
            });
        }
    };

    // ---- SVG skeleton --------------------------------------------------

    MethodGraph.prototype.buildSvg = function () {
        var self = this;
        var svg = this.svg;
        clear(svg);

        // Everything drawn is decoration for assistive technology; the conduits are
        // exposed once, as the radios of the hit layer.
        var hidden = { 'aria-hidden': 'true' };
        this.lineLayer = svg.appendChild(svgEl('g', hidden, 'method-graph__lines'));
        this.arrowLayer = svg.appendChild(svgEl('g', hidden, 'method-graph__arrows'));
        this.nodeLayer = svg.appendChild(svgEl('g', hidden, 'method-graph__nodes'));
        this.focusRing = svg.appendChild(svgEl('rect', { rx: 6, ry: 6, 'aria-hidden': 'true' }, 'method-graph__focus-ring'));
        this.labelLayer = svg.appendChild(svgEl('g', hidden, 'method-graph__labels'));
        this.hoverLabel = svg.appendChild(svgEl('text', hidden, 'method-graph__hover-label'));
        this.inset = svg.appendChild(svgEl('g', hidden, 'method-graph__inset'));
        this.hitLayer = svg.appendChild(svgEl('g', { role: 'radiogroup', 'aria-label': 'Conduits' }, 'method-graph__hits'));
        // A <desc> child describes its parent: the radio group, and only it.
        var desc = this.hitLayer.appendChild(svgEl('desc'));
        desc.textContent = 'Arrow keys move the selection to the neighbouring conduit in that direction on the map.';

        this.halo = svgEl('path', null, 'method-graph__halo');
        this.els = {};
        this.conduits.forEach(function (c) {
            var els = {
                line: self.lineLayer.appendChild(svgEl('path', null, 'method-graph__line')),
                arrow: self.arrowLayer.appendChild(svgEl('path', { d: 'M -3 -3 L 3 0 L -3 3 Z' }, 'method-graph__arrow')),
                hit: self.hitLayer.appendChild(svgEl('path', {
                    role: 'radio',
                    tabindex: '-1',
                    'aria-label': 'Conduit ' + c.id,
                    'aria-checked': 'false'
                }, 'method-graph__hit'))
            };
            self.els[c.id] = els;
        });
        this.nodeEls = this.data.nodes.map(function (n) {
            var cls = n.outfall ? 'method-graph__outfall' : 'method-graph__node';
            var dot = self.nodeLayer.appendChild(svgEl('circle', null, cls));
            return { node: n, el: dot };
        });

        this.insetBg = this.inset.appendChild(svgEl('rect', { rx: 4, ry: 4 }, 'method-graph__inset-bg'));
        this.insetBase = this.inset.appendChild(svgEl('path', null, 'method-graph__inset-network'));
        this.insetMarks = this.inset.appendChild(svgEl('g', null, 'method-graph__inset-marks'));
        this.insetWindow = this.inset.appendChild(svgEl('rect', null, 'method-graph__inset-window'));
    };

    // ---- Geometry ------------------------------------------------------

    // Size of the map for the current width; the overview's own geometry.
    MethodGraph.prototype.layout = function () {
        var width = Math.floor(this.mapMount.clientWidth);
        if (!width || width === this.width) {
            return;
        }
        var vb = this.data.viewBox;
        this.width = width;
        this.height = Math.round(width * vb[1] / vb[0]);
        this.svg.setAttribute('viewBox', '0 0 ' + width + ' ' + this.height);
        this.svg.setAttribute('width', width);
        this.svg.setAttribute('height', this.height);

        var insetW = Math.max(72, Math.min(112, Math.round(width * 0.2)));
        var insetPad = 6;
        this.insetBox = { w: insetW, h: Math.round(insetW * vb[1] / vb[0]), pad: insetPad };
        this.insetScale = (insetW - 2 * insetPad) / vb[0];
        var scale = this.insetScale;
        this.insetPoints = {};
        var parts = [];
        var self = this;
        this.conduits.forEach(function (c) {
            var points = c.path.map(function (p) {
                return [insetPad + p[0] * scale, insetPad + p[1] * scale];
            });
            self.insetPoints[c.id] = points;
            parts.push(pathData(points));
        });
        this.insetBg.setAttribute('width', insetW);
        this.insetBg.setAttribute('height', this.insetBox.h);
        this.insetBase.setAttribute('d', parts.join(' '));

        this.stopPan();
        this.view = this.windowFor(this.selected);
        this.refresh(false);
    };

    // The part of the network (in frame units) the map shows for a selection:
    // the two-step neighbourhood and the first conduit downstream, centred.
    MethodGraph.prototype.windowFor = function (id) {
        var vb = this.data.viewBox;
        var aspect = vb[1] / vb[0];
        var ids = (this.data.step2[id] || []).map(function (entry) {
            return entry[0];
        }).concat([id], this.downstream[id]);
        var xs = [];
        var ys = [];
        var self = this;
        ids.forEach(function (other) {
            self.byId[other].path.forEach(function (p) {
                xs.push(p[0]);
                ys.push(p[1]);
            });
        });
        var minX = Math.min.apply(null, xs);
        var maxX = Math.max.apply(null, xs);
        var minY = Math.min.apply(null, ys);
        var maxY = Math.max.apply(null, ys);

        var w = Math.max(vb[0] / ZOOM, maxX - minX + 2 * WINDOW_MARGIN, (maxY - minY + 2 * WINDOW_MARGIN) / aspect);
        var h = w * aspect;

        function place(center, size, extent) {
            var lo = -EDGE;
            var hi = extent + EDGE - size;
            if (hi < lo) {
                return (extent - size) / 2;
            }
            return Math.max(lo, Math.min(hi, center - size / 2));
        }

        return {
            x: place((minX + maxX) / 2, w, vb[0]),
            y: place((minY + maxY) / 2, h, vb[1]),
            w: w,
            h: h
        };
    };

    // Project every conduit, node and overlay through a window.
    MethodGraph.prototype.draw = function (view) {
        var self = this;
        var scale = this.width / view.w;
        function px(p) {
            return [(p[0] - view.x) * scale, (p[1] - view.y) * scale];
        }

        this.geom = {};
        this.segments = [];
        this.conduits.forEach(function (c) {
            var points = c.path.map(px);
            var d = pathData(points);
            var mid = midpointOf(points);
            var els = self.els[c.id];
            self.geom[c.id] = { points: points, mid: mid };
            for (var i = 1; i < points.length; i++) {
                var seg = [points[i - 1][0], points[i - 1][1], points[i][0], points[i][1]];
                seg.id = c.id;
                self.segments.push(seg);
            }
            els.line.setAttribute('d', d);
            els.hit.setAttribute('d', d);

            var k = ((self.strokes[c.id] || 1.5) + 4) / 6;
            var angle = Math.atan2(mid.uy, mid.ux) * 180 / Math.PI;
            els.arrow.setAttribute('transform', 'translate(' + round(mid.x) + ' ' + round(mid.y) + ') rotate(' +
                round(angle) + ') scale(' + Math.round(k * 100) / 100 + ')');
            els.arrow.classList.toggle('is-hidden', mid.length < ARROW_MIN_LENGTH);
        });
        this.halo.setAttribute('d', this.els[this.selected].line.getAttribute('d'));

        this.nodeEls.forEach(function (entry) {
            var p = px([entry.node.x, entry.node.y]);
            entry.px = p;
            entry.el.setAttribute('cx', round(p[0]));
            entry.el.setAttribute('cy', round(p[1]));
            entry.el.setAttribute('r', entry.node.outfall ? 4.5 : 2);
        });

        var sel = this.geom[this.selected].points;
        var xs = sel.map(function (p) {
            return p[0];
        });
        var ys = sel.map(function (p) {
            return p[1];
        });
        var pad = 7;
        var x0 = Math.min.apply(null, xs) - pad;
        var y0 = Math.min.apply(null, ys) - pad;
        this.focusBox = { x: x0, y: y0, w: Math.max.apply(null, xs) + pad - x0, h: Math.max.apply(null, ys) + pad - y0 };
        this.focusRing.setAttribute('x', round(x0));
        this.focusRing.setAttribute('y', round(y0));
        this.focusRing.setAttribute('width', round(this.focusBox.w));
        this.focusRing.setAttribute('height', round(this.focusBox.h));

        var s = this.insetScale;
        var ip = this.insetBox.pad;
        this.insetWindow.setAttribute('x', round(ip + view.x * s));
        this.insetWindow.setAttribute('y', round(ip + view.y * s));
        this.insetWindow.setAttribute('width', round(view.w * s));
        this.insetWindow.setAttribute('height', round(view.h * s));
    };

    // The pointer's conduit: in the overview by its own scale, on the map otherwise.
    MethodGraph.prototype.conduitAt = function (clientX, clientY, radius) {
        if (!this.geom) {
            return null;
        }
        var rect = this.svg.getBoundingClientRect();
        if (!rect.width) {
            return null;
        }
        var k = this.width / rect.width;
        var x = (clientX - rect.left) * k;
        var y = (clientY - rect.top) * k;
        var best = null;
        var bestD = radius * radius;
        var spot = this.insetSpot;

        if (spot && x >= spot.x && x <= spot.x + spot.w && y >= spot.y && y <= spot.y + spot.h) {
            var ix = x - spot.x;
            var iy = y - spot.y;
            var self = this;
            bestD = 100; // 10 px in the overview
            this.conduits.forEach(function (c) {
                var pts = self.insetPoints[c.id];
                for (var i = 1; i < pts.length; i++) {
                    var d = segmentDistance2(ix, iy, pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1]);
                    if (d < bestD) {
                        bestD = d;
                        best = c.id;
                    }
                }
            });
            return best;
        }

        this.segments.forEach(function (seg) {
            var d = segmentDistance2(x, y, seg[0], seg[1], seg[2], seg[3]);
            if (d < bestD) {
                bestD = d;
                best = seg.id;
            }
        });
        return best;
    };

    // ---- State ---------------------------------------------------------

    MethodGraph.prototype.rows = function (id) {
        var one = {};
        var two = {};
        (this.data.step1[id] || []).forEach(function (entry) {
            one[entry[0]] = entry[1];
        });
        (this.data.step2[id] || []).forEach(function (entry) {
            two[entry[0]] = entry[1];
        });
        return { one: one, two: two };
    };

    MethodGraph.prototype.roleOf = function (id, rows) {
        if (id === this.selected) {
            return 'selected';
        }
        if (id in rows.one) {
            return 'step1';
        }
        if (this.steps === 2 && id in rows.two) {
            return 'step2';
        }
        return 'dim';
    };

    // Kept in step with the stroke widths in method-graph.css.
    MethodGraph.prototype.strokeWidth = function (role, weight) {
        if (role === 'selected') {
            return 4;
        }
        if (role === 'dim') {
            return 1.5;
        }
        return 1.75 + weight * 7;
    };

    MethodGraph.prototype.choose = function (id, moveFocus) {
        if (!this.byId[id]) {
            return;
        }
        var changed = id !== this.selected;
        this.selected = id;
        this.refresh(changed);
        if (moveFocus) {
            this.els[id].hit.focus({ preventScroll: true });
        }
    };

    // Repaint after a change of selection or step count; pan if asked to.
    MethodGraph.prototype.refresh = function (pan) {
        if (!this.width) {
            return;
        }
        this.paintState();
        var target = this.windowFor(this.selected);
        if (pan && this.view && !prefersReducedMotion()) {
            this.panTo(target);
            return;
        }
        this.stopPan();
        this.view = target;
        this.draw(this.view);
        this.finish();
    };

    MethodGraph.prototype.panTo = function (target) {
        var self = this;
        var from = this.view;
        var startTime = null;
        this.stopPan();
        clear(this.labelLayer);
        this.setHover(null);
        this.root.classList.add('is-panning');

        function frame(now) {
            if (startTime === null) {
                startTime = now;
            }
            var t = Math.min(1, (now - startTime) / PAN_MS);
            var e = easeInOut(t);
            self.view = {
                x: from.x + (target.x - from.x) * e,
                y: from.y + (target.y - from.y) * e,
                w: from.w + (target.w - from.w) * e,
                h: from.h + (target.h - from.h) * e
            };
            self.draw(self.view);
            if (t < 1) {
                self.panFrame = window.requestAnimationFrame(frame);
            } else {
                self.panFrame = null;
                self.view = target;
                self.root.classList.remove('is-panning');
                self.finish();
                self.hoverFromPointer();
            }
        }
        this.panFrame = window.requestAnimationFrame(frame);
    };

    MethodGraph.prototype.stopPan = function () {
        if (this.panFrame !== null) {
            window.cancelAnimationFrame(this.panFrame);
            this.panFrame = null;
            this.root.classList.remove('is-panning');
        }
    };

    MethodGraph.prototype.setHover = function (id) {
        if (id === this.hoverId) {
            return;
        }
        if (this.hoverId) {
            this.els[this.hoverId].line.classList.remove('is-hover');
        }
        this.hoverId = id;
        this.svg.classList.toggle('is-pointing', !!id);
        if (id) {
            this.els[id].line.classList.add('is-hover');
        }
        this.paintHoverLabel();
    };

    // Hover follows the last pointer position; nothing is hovered mid-pan.
    MethodGraph.prototype.hoverFromPointer = function () {
        var p = this.pointer;
        if (!p || this.panFrame !== null) {
            this.setHover(null);
            return;
        }
        this.setHover(this.conduitAt(p.x, p.y, hitRadius(p.type)));
    };

    MethodGraph.prototype.onKey = function (event) {
        var dirs = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
        var dir = dirs[event.key];
        // Modified arrows belong to the browser and assistive technology.
        if (!dir || !this.geom || event.altKey || event.ctrlKey || event.metaKey) {
            return;
        }
        event.preventDefault();
        var next = this.neighbourTowards(this.selected, dir[0], dir[1]);
        if (next) {
            this.choose(next, true);
        }
    };

    // The conduit an arrow key moves to: a connected conduit roughly in that
    // direction if there is one, otherwise the nearest conduit that way.
    MethodGraph.prototype.neighbourTowards = function (id, dx, dy) {
        var self = this;
        var from = this.geom[id].mid;

        function best(candidates, minCos) {
            var pick = null;
            var pickScore = Infinity;
            candidates.forEach(function (other) {
                var m = self.geom[other].mid;
                var vx = m.x - from.x;
                var vy = m.y - from.y;
                var dist = Math.hypot(vx, vy);
                var cos = dist ? (vx * dx + vy * dy) / dist : -1;
                var score = dist * (2 - cos);
                if (other !== id && cos >= minCos && score < pickScore) {
                    pickScore = score;
                    pick = other;
                }
            });
            return pick;
        }

        return best(this.upstream[id].concat(this.downstream[id]), 0.35) || best(Object.keys(this.byId), 0.5);
    };

    // ---- Painting ------------------------------------------------------

    // Classes, weights and draw order of every conduit; the readout.
    MethodGraph.prototype.paintState = function () {
        var self = this;
        var rows = this.rows(this.selected);
        var weights = this.steps === 2 ? rows.two : rows.one;
        var rank = { dim: 0, step2: 1, step1: 2, selected: 3 };
        var layered = [];
        var inView = {};

        this.conduits.forEach(function (c) {
            var els = self.els[c.id];
            var role = self.roleOf(c.id, rows);
            var weight = weights[c.id] || 0;
            [els.line, els.arrow].forEach(function (el) {
                var cl = el.classList;
                cl.toggle('is-selected', role === 'selected');
                cl.toggle('is-step1', role === 'step1');
                cl.toggle('is-step2', role === 'step2');
                cl.toggle('is-dim', role === 'dim');
            });
            els.line.style.setProperty('--mg-weight', String(weight));
            els.hit.setAttribute('aria-checked', role === 'selected' ? 'true' : 'false');
            els.hit.setAttribute('tabindex', role === 'selected' ? '0' : '-1');
            self.strokes[c.id] = self.strokeWidth(role, weight);
            layered.push({ id: c.id, rank: rank[role] });
            if (role !== 'dim') {
                inView[c.from] = true;
                inView[c.to] = true;
            }
        });

        // Draw order: dimmed lines at the bottom, the selection on top with its halo.
        layered.sort(function (a, b) {
            return a.rank - b.rank || self.position[a.id] - self.position[b.id];
        });
        layered.forEach(function (entry) {
            if (entry.id === self.selected) {
                self.lineLayer.appendChild(self.halo);
            }
            self.lineLayer.appendChild(self.els[entry.id].line);
            self.arrowLayer.appendChild(self.els[entry.id].arrow);
        });

        this.nodeEls.forEach(function (entry) {
            entry.el.classList.toggle('is-in-view', !!inView[entry.node.id]);
        });

        clear(this.insetMarks);
        layered.forEach(function (entry) {
            if (entry.rank > 0) {
                var role = self.roleOf(entry.id, rows);
                var d = pathData(self.insetPoints[entry.id]);
                self.insetMarks.appendChild(svgEl('path', { d: d }, 'method-graph__inset-mark is-' + role));
            }
        });

        this.paintReadout(rows);
        if (this.select && this.select.value !== this.selected) {
            this.select.value = this.selected;
        }
    };

    // Everything that depends on the settled view: overview corner, labels.
    MethodGraph.prototype.finish = function () {
        if (!this.geom || this.panFrame !== null) {
            return;
        }
        this.placeInset();
        this.paintLabels();
        this.paintHoverLabel();
    };

    // Put the overview in the corner that hides the fewest lines; the previous
    // corner wins ties, so it does not wander between similar views.
    MethodGraph.prototype.placeInset = function () {
        var box = this.insetBox;
        var self = this;
        var corners = [
            [INSET_GAP, INSET_GAP],
            [this.width - box.w - INSET_GAP, INSET_GAP],
            [INSET_GAP, this.height - box.h - INSET_GAP],
            [this.width - box.w - INSET_GAP, this.height - box.h - INSET_GAP]
        ];
        var previous = this.insetCorner || 0;
        var scores = corners.map(function (corner, i) {
            var spot = inflate({ x: corner[0], y: corner[1], w: box.w, h: box.h }, 6);
            var hidden = self.segments.filter(function (seg) {
                return segmentHitsBox(seg, spot);
            }).length;
            return { i: i, score: hidden * 10 + (i === previous ? 0 : 1) + i * 0.1 };
        });
        scores.sort(function (a, b) {
            return a.score - b.score;
        });
        this.insetCorner = scores[0].i;
        var c = corners[this.insetCorner];
        this.insetSpot = { x: c[0], y: c[1], w: box.w, h: box.h };
        this.inset.setAttribute('transform', 'translate(' + c[0] + ' ' + c[1] + ')');
    };

    MethodGraph.prototype.measure = function (text, cls) {
        var key = cls + '|' + text;
        if (!(key in this.textWidths)) {
            var probe = this.labelLayer.appendChild(svgEl('text', null, cls));
            probe.textContent = text;
            var width = 0;
            try {
                width = probe.getComputedTextLength();
            } catch (e) {
                width = 0;
            }
            this.labelLayer.removeChild(probe);
            this.textWidths[key] = width || text.length * 7;
        }
        return this.textWidths[key];
    };

    MethodGraph.prototype.paintLabels = function () {
        var self = this;
        var layer = this.labelLayer;
        var rows = this.rows(this.selected);
        var weights = this.steps === 2 ? rows.two : rows.one;
        var items = [];
        clear(layer);

        function visible(x, y) {
            return x >= 0 && y >= 0 && x <= self.width && y <= self.height;
        }

        function add(id, text, role, mid) {
            if (!visible(mid.x, mid.y)) {
                return;
            }
            var cls = 'method-graph__label-text is-' + role;
            items.push({
                id: id,
                text: text,
                cls: cls,
                role: role,
                x: mid.x,
                y: mid.y,
                nx: -mid.uy,
                ny: mid.ux,
                w: self.measure(text, cls) + 2 * LABEL_PAD_X,
                h: LABEL_HEIGHT
            });
        }

        add(this.selected, this.selected, 'selected', this.geom[this.selected].mid);
        Object.keys(weights).filter(function (id) {
            return id !== self.selected;
        }).sort(function (a, b) {
            return weights[b] - weights[a] || self.position[a] - self.position[b];
        }).forEach(function (id) {
            add(id, id, self.roleOf(id, rows), self.geom[id].mid);
        });
        this.nodeEls.forEach(function (entry) {
            if (entry.node.outfall) {
                add(null, 'Outfall', 'outfall', { x: entry.px[0], y: entry.px[1], ux: 1, uy: 0 });
            }
        });

        // Labels also keep off the focus ring's box, shown or not, so focus moves no label.
        var obstacles = [inflate(this.insetSpot, 2), this.focusBox];
        placeLabels(items, this.segments, obstacles, this.width, this.height).forEach(function (spot) {
            var item = spot.item;
            if (spot.leader) {
                layer.appendChild(svgEl('line', {
                    x1: round(spot.leader[0]),
                    y1: round(spot.leader[1]),
                    x2: round(spot.leader[2]),
                    y2: round(spot.leader[3])
                }, 'method-graph__leader is-' + item.role));
            }
            var text = layer.appendChild(svgEl('text', {
                x: round(spot.box.x + LABEL_PAD_X),
                y: round(spot.box.y + spot.box.h / 2)
            }, item.cls));
            text.textContent = item.text;
        });
    };

    MethodGraph.prototype.paintHoverLabel = function () {
        var label = this.hoverLabel;
        var id = this.hoverId;
        // Only for a conduit that carries no label already, and not mid-pan.
        var labelled = Array.prototype.some.call(this.labelLayer.querySelectorAll('text'), function (t) {
            return t.textContent === id;
        });
        var shown = !!id && this.panFrame === null && !labelled;
        label.classList.toggle('is-visible', shown);
        if (!shown) {
            return;
        }
        var m = this.geom[id].mid;
        label.textContent = id;
        label.setAttribute('x', round(Math.min(Math.max(m.x, 16), this.width - 16)));
        label.setAttribute('y', round(m.y > 24 ? m.y - 11 : m.y + 13));
    };

    // ---- Readout -------------------------------------------------------

    MethodGraph.prototype.buildReadout = function () {
        var mount = this.readoutMount;
        clear(mount);

        // One short sentence for screen readers; the visible summary is not live.
        // It becomes live after the first paint, so loading the page announces nothing.
        this.statusEl = mount.appendChild(htmlEl('p', 'method-graph__status'));
        this.statusEl.setAttribute('aria-atomic', 'true');

        var summary = mount.appendChild(htmlEl('div', 'method-graph__summary'));
        summary.appendChild(htmlEl('p', 'method-graph__eyebrow', 'Selected conduit'));
        this.nameEl = summary.appendChild(htmlEl('p', 'method-graph__name'));
        var counts = summary.appendChild(htmlEl('dl', 'method-graph__counts'));
        this.countEls = {};
        [[1, 'One step'], [2, 'Two steps']].forEach(function (pair) {
            var group = counts.appendChild(htmlEl('div', 'method-graph__count'));
            group.appendChild(htmlEl('dt', null, pair[1]));
            this.countEls[pair[0]] = { group: group, value: group.appendChild(htmlEl('dd')) };
        }, this);

        var panel = mount.appendChild(htmlEl('div', 'method-graph__panel'));
        this.listTitle = panel.appendChild(htmlEl('p', 'method-graph__list-title'));
        this.listEl = panel.appendChild(htmlEl('ol', 'method-graph__list'));
        // The tail of a long list, in place under a native disclosure.
        this.moreEl = panel.appendChild(htmlEl('details', 'method-graph__more'));
        this.moreSummary = this.moreEl.appendChild(htmlEl('summary', 'method-graph__more-toggle'));
        this.restEl = this.moreEl.appendChild(htmlEl('ol', 'method-graph__list method-graph__list--rest'));
        this.noteEl = panel.appendChild(htmlEl('p', 'method-graph__note'));

        var legend = mount.appendChild(htmlEl('ul', 'method-graph__legend'));
        legend.setAttribute('aria-label', 'Legend');
        [
            ['selected', 'Selected conduit'],
            ['step1', 'First step (width = weight)'],
            ['step2', 'Added by the second step'],
            ['dim', 'Not read, incl. downstream'],
            ['outfall', 'Outfall']
        ].forEach(function (pair) {
            var item = legend.appendChild(htmlEl('li', 'method-graph__legend-item is-' + pair[0]));
            item.appendChild(htmlEl('span', 'method-graph__swatch')).setAttribute('aria-hidden', 'true');
            item.appendChild(htmlEl('span', null, pair[1]));
            if (pair[0] === 'step2') {
                this.legendStep2 = item;
            }
        }, this);
    };

    MethodGraph.prototype.paintReadout = function (rows) {
        var id = this.selected;
        var nOne = Object.keys(rows.one).length;
        var nTwo = Object.keys(rows.two).length;

        // Arrow keys on the map already announce the newly focused conduit.
        if (!this.focusIsOnMap()) {
            setText(this.statusEl, id + ': ' + plural(nOne, 'conduit', 'conduits') + ' in one step, ' + nTwo + ' in two.');
        }

        // Resizes repaint too; the readout only changes with the selection or the step count.
        var key = id + '|' + this.steps;
        if (key === this.readoutKey) {
            return;
        }
        this.readoutKey = key;

        var row = (this.steps === 2 ? this.data.step2[id] : this.data.step1[id]) || [];
        var isHead = this.upstream[id].length === 0;
        setText(this.nameEl, id);
        setText(this.countEls[1].value, plural(nOne, 'conduit', 'conduits'));
        setText(this.countEls[2].value, plural(nTwo, 'conduit', 'conduits'));
        this.countEls[1].group.classList.toggle('is-active', this.steps === 1);
        this.countEls[2].group.classList.toggle('is-active', this.steps === 2);
        this.legendStep2.classList.toggle('is-inactive', this.steps === 1);
        setText(this.listTitle, this.steps === 2 ? 'Share of the two-step average' : 'Weights of one graph step');

        var top = row.length ? row[0][1] : 1;
        this.fillList(this.listEl, row.slice(0, LIST_LIMIT), rows, top);
        this.fillList(this.restEl, row.slice(LIST_LIMIT), rows, top);
        this.restEl.setAttribute('start', String(LIST_LIMIT + 1));
        var rest = row.length - LIST_LIMIT;
        this.moreEl.hidden = rest <= 0;
        setText(this.moreSummary, rest > 0 ? '+ ' + rest + ' more' : '');

        var note;
        if (isHead) {
            note = 'Nothing drains into ' + id + '. It is a head conduit: the model sees only the conduit itself.';
        } else if (this.steps === 2) {
            note = 'Indicative reach of both steps, before the model\u2019s non-linearities; not an importance score.';
        } else {
            note = 'Exact weights of one step: the conduit itself and the conduits up to two links upstream.';
        }
        setText(this.noteEl, note);
        this.noteEl.classList.toggle('is-head', isHead);
    };

    MethodGraph.prototype.fillList = function (list, entries, rows, top) {
        var self = this;
        clear(list);
        entries.forEach(function (entry) {
            var li = list.appendChild(htmlEl('li', 'method-graph__row is-' + self.roleOf(entry[0], rows)));
            var name = li.appendChild(htmlEl('span', 'method-graph__row-name', entry[0]));
            if (entry[0] === self.selected) {
                name.appendChild(htmlEl('span', 'method-graph__row-self', 'itself'));
            }
            var bar = li.appendChild(htmlEl('span', 'method-graph__bar'));
            bar.setAttribute('aria-hidden', 'true');
            var fill = bar.appendChild(htmlEl('span', 'method-graph__bar-fill'));
            fill.style.setProperty('--mg-share', String(top ? entry[1] / top : 0));
            li.appendChild(htmlEl('span', 'method-graph__row-value', formatShare(entry[1])));
        });
    };

    // ---- Select --------------------------------------------------------

    MethodGraph.prototype.buildSelect = function () {
        var self = this;
        if (!this.select) {
            return;
        }
        clear(this.select);
        this.conduits.forEach(function (c) {
            var label = self.upstream[c.id].length ? c.id : c.id + ' (head)';
            var option = self.select.appendChild(htmlEl('option', null, label));
            option.value = c.id;
        });
        this.select.value = this.selected;
        this.select.addEventListener('change', function () {
            self.choose(self.select.value, false);
        });
    };

    // ------------------------------------------------------------------
    // Boot
    // ------------------------------------------------------------------

    function isUsable(data) {
        return !!data && Array.isArray(data.conduits) && data.conduits.length > 0 && Array.isArray(data.nodes) &&
            Array.isArray(data.viewBox) && !!data.step1 && !!data.step2;
    }

    function init(root) {
        var src = root.getAttribute('data-src');
        if (!src || !root.querySelector('[data-method-graph-svg]') || !root.querySelector('[data-method-graph-map]') ||
                !root.querySelector('[data-method-graph-readout]') || typeof window.fetch !== 'function') {
            return;
        }
        window.fetch(src, { credentials: 'same-origin' })
            .then(function (response) {
                if (!response.ok) {
                    throw new Error('HTTP ' + response.status);
                }
                return response.json();
            })
            .then(function (data) {
                if (!isUsable(data)) {
                    throw new Error('unexpected figure data');
                }
                var figure = new MethodGraph(root, data);
                try {
                    figure.start();
                } catch (error) {
                    figure.setReady(false);
                    throw error;
                }
            })
            .catch(function (error) {
                // The static fallback sentence stays in place.
                if (window.console && window.console.warn) {
                    window.console.warn('method-graph: figure not drawn:', error);
                }
            });
    }

    function boot() {
        Array.prototype.forEach.call(document.querySelectorAll('[data-method-graph]'), init);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
})();
