/*
 * conduit-panel.js - the conduit panel: longitudinal profile and the details tabs.
 *
 * Draws the inlet and outlet manholes, the ground line, the soil wash, the pipe with
 * its water level and the cover / length dimensions into the <svg class="conduit-profile">
 * of the conduit panel. No dependencies: plain DOM plus one SVG string.
 *
 * The pipe follows the long-section convention: it runs from the inlet manhole's outer
 * wall to the outlet manhole's wall, and each wall facing the pipe is open between the
 * crown and the invert. Fills and outlines are separate elements for that reason - the
 * pipe and the shafts are fill-only bodies, their outlines ("pipe-wall", "shaft-wall")
 * are open paths drawn on top, so no stroke closes an end.
 *
 * Colours come from app.css (block "conduit-panel"), matched on the data-role attributes
 * and the helper classes below - this file never names a colour.
 *
 * Horizontal distances and the assumed manhole width share one scale, subject to
 * legibility bounds; the fixed frame means that scale changes with conduit length. The
 * vertical scale is exaggerated. renderProfile returns that factor and the caller puts it
 * in the caption under the drawing.
 *
 * The two water levels are different SWMM results and are drawn as they are: in a manhole
 * the node's maximum water depth, in the pipe the conduit's maximum filling. The maxima
 * fall at different times, so the levels may not meet at the wall; the <title> of each
 * water element and the caption under the drawing say which is which.
 */
(function () {
    'use strict';

    // Drawing frame, in viewBox units (viewBox is "0 0 580 256": the length label ends near y 246).
    const AX1 = 97;    // inlet manhole axis
    const AX2 = 483;   // outlet manhole axis
    // SWMM junctions carry no chamber diameter; assume a standard 1.2 m manhole.
    const MANHOLE_DIAMETER_M = 1.2;
    const W_MIN = 14;  // keeps the lid, water and rounded outline legible
    const YTOP = 60;   // highest ground elevation lands here
    const YBOT = 187;  // lowest invert lands here
    const SUMP = 9;    // manhole bottom below the pipe invert
    const X0 = 25;     // left edge of the grid
    const X1 = 555;    // right edge of the grid
    const XM = (AX1 + AX2) / 2;
    const GTOP = 12;   // top of the grid
    const GBOT = 222;  // bottom of the grid
    const HB = 8;      // thickness of the hatched band under the ground line
    // Half the stroke width of the pipe and shaft walls (app.css: stroke-width 2). A wall
    // ends this far inside the opening so that its butt end covers the outer corner of the
    // join with the crown / invert line instead of leaving a notch there.
    const WALL_HALF = 1;
    const SHAFT_RX = 3;  // rounded bottom corners of the manhole outline

    const NO_DATA_TEXT = 'No data to draw the profile';

    /** Fixed-point number with a decimal point (the application is English). */
    function nf(value, digits) {
        return Number(value).toFixed(digits);
    }

    /** Round to two decimals so the path strings stay short. */
    function r(value) {
        return Math.round(value * 100) / 100;
    }

    /** Straight line between the two manhole axes, read at x. */
    function lerp(a, b, x) {
        return a + (b - a) * (x - AX1) / (AX2 - AX1);
    }

    /** Nothing leaves the grid. */
    function clampY(y) {
        return Math.min(GBOT, Math.max(GTOP, y));
    }

    /** Points as the "x y" pairs of a path, joined with L. */
    function poly(points) {
        return points.map(function (p) { return r(p[0]) + ' ' + r(p[1]); }).join('L');
    }

    /** A missing value renders as an em dash rather than the string "null". */
    function orDash(value) {
        return value === null || value === undefined ? '—' : value;
    }

    function esc(value) {
        return String(value).replace(/[&<>"]/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
        });
    }

    /** Shorten by code points (SWMM names may contain astral characters); null -> ''. */
    function short(value) {
        const MAX = 12;
        const chars = Array.from(String(value === null || value === undefined ? '' : value));
        return chars.length > MAX ? chars.slice(0, MAX - 1).join('') + '…' : chars.join('');
    }

    function showMessage(svg, text, label) {
        svg.innerHTML = '<text x="290" y="128" text-anchor="middle">' + esc(text) + '</text>';
        svg.setAttribute('aria-label', label);
    }

    /**
     * Render the profile into `svg`.
     * Returns the vertical exaggeration factor, or 0 when there is nothing to draw.
     */
    function renderProfile(svg, raw) {
        const d = Object.assign({}, raw);
        d.name = orDash(d.name);
        d.inletNode = orDash(d.inletNode);
        d.outletNode = orDash(d.outletNode);

        // Invert from the node record; failing that, ground level minus cover minus
        // diameter, which is the relation the cover itself is measured by.
        const invert = function (stored, ground, cover) {
            if (stored !== null && stored !== undefined) {
                return Number(stored);
            }
            return cover === null || cover === undefined ? NaN : ground - cover - d.diameter;
        };
        const inv1 = invert(d.inletInvert, d.inletElevation, d.inletCover);
        const inv2 = invert(d.outletInvert, d.outletElevation, d.outletCover);

        const needed = [d.length, d.diameter, d.filling, d.inletElevation, d.outletElevation, inv1, inv2];
        const drawable =
            needed.every(Number.isFinite) &&
            d.length > 0 &&
            d.diameter > 0 &&
            // A manhole whose invert sits at or above its ground level would come out as a
            // negative-height rectangle; show the message instead of a wrong drawing.
            inv1 < d.inletElevation &&
            inv2 < d.outletElevation;
        if (!drawable) {
            showMessage(svg, NO_DATA_TEXT, 'Longitudinal profile of conduit ' + d.name + ': no data.');
            return 0;
        }

        const eMax = Math.max(d.inletElevation, d.outletElevation);
        const eMin = Math.min(inv1, inv2);
        const sv = (YBOT - YTOP) / Math.max(eMax - eMin, 1e-6);  // px per metre, vertical
        const sh = (AX2 - AX1) / Math.max(d.length, 1e-6);       // px per metre, horizontal
        const exaggeration = Math.max(1, Math.round(sv / sh));
        const W = Math.min(32, Math.max(W_MIN, Math.round(MANHOLE_DIAMETER_M * sh)));

        const Y = function (elevation) {
            return YTOP + (eMax - elevation) * sv;
        };
        const gY = function (x) {                                 // ground away from the manholes
            return lerp(Y(d.inletElevation), Y(d.outletElevation), x);
        };
        const iY = function (x) {                                 // pipe invert
            return lerp(Y(inv1), Y(inv2), x);
        };

        const sx1 = AX1 - W / 2;
        const sx2 = AX2 - W / 2;
        const g1 = Y(d.inletElevation);
        const g2 = Y(d.outletElevation);
        const pipeH = d.diameter * sv;
        const waterH = Math.min(Math.max(d.filling, 0), d.diameter) * sv;  // water stays inside the pipe
        // The pipe joins the manholes at their walls: from the inlet's outer wall to the
        // outlet's wall, never into a shaft.
        const pa = sx1 + W;
        const pb = sx2;
        const cy = function (x) {                                 // pipe axis
            return iY(x) - pipeH / 2;
        };
        const sump1 = iY(AX1) + SUMP;                             // manhole bottom, inlet
        const sump2 = iY(AX2) + SUMP;                             // manhole bottom, outlet

        // Ground line: grid edge to grid edge, flat over each manhole.
        const GP = [
            [X0, clampY(gY(X0))],
            [sx1, g1],
            [sx1 + W, g1],
            [sx2, g2],
            [sx2 + W, g2],
            [X1, clampY(gY(X1))],
        ];
        /** y of the DRAWN ground line at x - the cover dimension must end on it. */
        const gLine = function (x) {
            for (let k = 1; k < GP.length; k++) {
                const a = GP[k - 1];
                const b = GP[k];
                if (x <= b[0]) {
                    return a[1] + (b[1] - a[1]) * (x - a[0]) / Math.max(b[0] - a[0], 1e-6);
                }
            }
            return GP[GP.length - 1][1];
        };

        const groundD = 'M' + poly(GP);
        const soilD = groundD + 'L' + X1 + ' ' + GBOT + 'L' + X0 + ' ' + GBOT + 'Z';
        const hatchD = groundD + 'L' + poly(GP.map(function (p) {
            return [p[0], Math.min(GBOT, p[1] + HB)];
        }).reverse()) + 'Z';

        // Conduit label: above the pipe; with little cover it moves inside a tall pipe,
        // otherwise below it.
        const coverPx = (iY(XM) - pipeH) - gY(XM);
        const labelInside = coverPx < 34 && pipeH >= 22;
        const labelY = coverPx >= 34 ? iY(XM) - pipeH - 9 : labelInside ? cy(XM) + 4 : iY(XM) + 17;

        // Flow arrow along the pipe axis.
        const dx = 65;
        const ax = labelInside ? 370 : 255;
        const bx = ax + dx;
        const ay = cy(ax);
        const by = cy(bx);
        const len = Math.hypot(dx, by - ay);
        const ux = dx / len;
        const uy = (by - ay) / len;
        const hx = bx - 7 * ux;
        const hy = by - 7 * uy;
        const arrowD = 'M' + r(ax) + ' ' + r(ay) + 'L' + r(bx) + ' ' + r(by)
            + 'M' + r(hx + 4 * uy) + ' ' + r(hy - 4 * ux) + 'L' + r(bx) + ' ' + r(by)
            + 'L' + r(hx - 4 * uy) + ' ' + r(hy + 4 * ux);

        // Shaft body: fill only, the outline is shaftWall below.
        const shaft = function (x, y, h) {
            return '<rect data-role="shaft" x="' + x + '" y="' + r(y) + '" width="' + W
                + '" height="' + r(h) + '" rx="' + SHAFT_RX + '"/>';
        };
        // Where the pipe enters a shaft: the right wall of the inlet shaft (side > 0), the
        // left wall of the outlet shaft (side < 0). Crown and invert are read at that wall -
        // on a steep conduit they differ from their values at the axis - and kept inside the
        // straight part of the shaft wall (lid line to the start of the bottom corner).
        const opening = function (x, ground, sump, side) {
            const wx = side > 0 ? x + W : x;
            const inside = function (y) {
                return Math.min(sump - SHAFT_RX, Math.max(ground, y));
            };
            return { x: wx, crown: inside(iY(wx) - pipeH), invert: inside(iY(wx)) };
        };
        // Shaft outline, open between the pipe's crown and invert. The path starts and ends
        // at the opening and goes round the shaft the long way: square top (the lid covers
        // it), rounded bottom corners. Both ends reach WALL_HALF into the opening, onto the
        // inner edge of the crown / invert stroke, so each join is a clean corner.
        const shaftWall = function (x, ground, sump, side) {
            const gap = opening(x, ground, sump, side);
            const wx = gap.x;
            const fx = side > 0 ? x : x + W;      // far wall
            const k = side > 0 ? 1 : -1;          // from the far wall towards the pipe wall
            const sweep = side > 0 ? 0 : 1;
            const top = Math.min(gap.crown + WALL_HALF, gap.invert);
            const bottom = Math.max(top, gap.invert - WALL_HALF);
            return '<path data-role="shaft-wall" d="M' + wx + ' ' + r(top) + 'V' + r(ground)
                + 'H' + fx + 'V' + r(sump - SHAFT_RX)
                + 'A' + SHAFT_RX + ' ' + SHAFT_RX + ' 0 0 ' + sweep + ' ' + (fx + k * SHAFT_RX) + ' ' + r(sump)
                + 'H' + (wx - k * SHAFT_RX)
                + 'A' + SHAFT_RX + ' ' + SHAFT_RX + ' 0 0 ' + sweep + ' ' + wx + ' ' + r(sump - SHAFT_RX)
                + 'V' + r(bottom) + '"/>';
        };
        // Shaft water over the full shaft width: the walls drawn over it hide the outer
        // pixel. Inside the opening it also reaches WALL_HALF under the pipe body, which is
        // opaque and drawn later, so the anti-aliased edge where shaft water meets pipe water
        // does not show a light seam; the tab never leaves the pipe band.
        const shaftWater = function (x, ground, sump, inv, depth, flooded, side, name) {
            if (!flooded && !(Number.isFinite(depth) && depth > 0)) {
                return '';
            }
            const top = flooded ? ground + 1 : Math.max(ground + 1, Y(inv + depth));
            const bottom = sump - 1;
            if (top >= bottom) {
                return '';
            }
            const gap = opening(x, ground, sump, side);
            const tabTop = Math.max(top, gap.crown);
            const tabBottom = Math.min(bottom, gap.invert);
            const tab = tabTop < tabBottom
                ? 'M' + gap.x + ' ' + r(tabTop) + 'h' + side * WALL_HALF + 'V' + r(tabBottom)
                    + 'h' + -side * WALL_HALF + 'Z'
                : '';
            // A flooded node may have no recorded depth: the water is drawn, the title is not.
            const title = Number.isFinite(depth)
                ? '<title>Max water depth at ' + esc(name) + ': ' + nf(depth, 2) + ' m</title>'
                : '';
            return '<path data-role="shaft-water" d="M' + x + ' ' + r(top) + 'h' + W
                + 'V' + r(bottom) + 'h' + -W + 'Z' + tab + (title ? '">' + title + '</path>' : '"/>');
        };
        // Manhole lid: horizontal, exactly as wide as the shaft, 4 px thick, axis one pixel
        // below the ground elevation so its top edge sits on the ground line.
        const lid = function (x, y) {
            return '<path data-role="lid" d="M' + x + ' ' + r(y + 1) + 'H' + (x + W) + '"/>';
        };
        // A band of the given height standing on the invert line, from xa to xb; `title`
        // (already escaped markup) becomes its tooltip.
        const quad = function (role, xa, xb, height, title) {
            return '<path data-role="' + role + '" d="M' + xa + ' ' + r(iY(xa) - height)
                + 'L' + xb + ' ' + r(iY(xb) - height)
                + 'L' + xb + ' ' + r(iY(xb))
                + 'L' + xa + ' ' + r(iY(xa)) + 'Z' + (title ? '"><title>' + title + '</title></path>' : '"/>');
        };
        // Crown and invert: two open lines, so the pipe has no end strokes.
        const pipeWalls = function (xa, xb) {
            const line = function (lift) {
                return '<path data-role="pipe-wall" d="M' + xa + ' ' + r(iY(xa) - lift)
                    + 'L' + xb + ' ' + r(iY(xb) - lift) + '"/>';
            };
            return line(pipeH) + line(0);
        };
        // SWMM rounds the flooded hours to two decimals, so the server decides whether the
        // node flooded; a short flood reads 0.00 h here and is shown as "< 0.01 h".
        const hoursText = function (hours) {
            if (!Number.isFinite(hours)) {
                return '';
            }
            return hours < 0.005 ? '&lt; 0.01 h' : nf(hours, 2) + ' h';
        };
        const floodMarker = function (x, y, flooded, hours, volume) {
            if (!flooded) {
                return '';
            }
            const parts = [hoursText(hours), Number.isFinite(volume) ? nf(volume, 3) + ' × 10⁶ l' : ''];
            const detail = parts.filter(Boolean).join(', ');
            return '<path data-role="flood" d="M' + r(x - 5) + ' ' + r(y - 8) + 'L' + x + ' '
                + r(y - 2) + 'L' + r(x + 5) + ' ' + r(y - 8) + 'Z"><title>Flooding'
                + (detail ? ': ' + detail : '') + '</title></path>';
        };
        // Node name (shortened, full name in <title>) and ground elevation. Flooded
        // nodes reserve the space immediately above the lid for the marker.
        const nodeTxt = function (x, y, name, elevation, flooded) {
            const lift = flooded ? 13 : 0;
            return '<text x="' + x + '" y="' + r(y - 21 - lift) + '" text-anchor="middle">'
                + '<title>' + esc(name) + '</title>' + esc(short(name)) + '</text>'
                + '<text data-role="elev-ground" class="cp-small" x="' + x + '" y="' + r(y - 8 - lift)
                + '" text-anchor="middle">' + nf(elevation, 2) + ' m</text>';
        };
        const invertTxt = function (x, yBottom, elevation) {
            return '<text data-role="elev-invert" class="cp-small" x="' + x + '" y="' + r(yBottom + 11)
                + '" text-anchor="middle">invert ' + nf(elevation, 2) + ' m</text>';
        };
        const coverValue = function (stored, ground, inv) {
            return Number.isFinite(stored) ? stored : ground - inv - d.diameter;
        };
        // Cover dimension: a vertical arrowed line from the drawn ground line to the crown.
        const coverDim = function (x, value, side) {
            const yt = gLine(x);
            const yb = iY(x) - pipeH;
            const h = yb - yt;
            if (!Number.isFinite(value) || h < 11) {
                return '';
            }
            const tx = side > 0 ? x + 4 : x - 4;
            // Too little room for the arrows: the value alone, just above the crown.
            const cramped = h < 14;
            const line = cramped ? '' : '<path data-role="cover-dim" d="M' + r(x) + ' ' + r(yt) + 'V' + r(yb)
                + 'M' + r(x - 4) + ' ' + r(yt + 5) + 'L' + r(x) + ' ' + r(yt) + 'L' + r(x + 4) + ' ' + r(yt + 5)
                + 'M' + r(x - 4) + ' ' + r(yb - 5) + 'L' + r(x) + ' ' + r(yb) + 'L' + r(x + 4) + ' ' + r(yb - 5)
                + '"/>';
            return line + '<text data-role="cover-dim" class="cp-small" x="' + r(tx)
                + '" y="' + r(cramped ? yb - 3 : (yt + yb) / 2 + 3)
                + '" text-anchor="' + (side > 0 ? 'start' : 'end') + '">' + nf(value, 2) + ' m</text>';
        };

        const slope = Number.isFinite(d.slope) ? d.slope : 1000 * (inv1 - inv2) / d.length;
        const fillingPercent = 100 * Math.min(Math.max(d.filling, 0), d.diameter) / d.diameter;
        const waterTitle = 'Max conduit depth ' + nf(Math.min(Math.max(d.filling, 0), d.diameter), 2)
            + ' m (' + nf(fillingPercent, 0) + ' %)';
        const diameterMm = Math.round(d.diameter * 1000);
        const lengthText = nf(d.length, 1);
        const inletFlooded = d.inletFlooded === true;
        const outletFlooded = d.outletFlooded === true;

        svg.innerHTML = [
            '<defs>',
            '<pattern id="cp-grid" width="25" height="25" patternUnits="userSpaceOnUse">',
            '<path class="cp-grid-line" d="M25 0H0V25"/></pattern>',
            '<pattern id="cp-soil-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">',
            '<path class="cp-hatch-line" d="M0 0V7"/></pattern>',
            '</defs>',
            '<rect x="' + X0 + '" y="' + GTOP + '" width="' + (X1 - X0) + '" height="' + (GBOT - GTOP)
                + '" fill="url(#cp-grid)"/>',
            '<path data-role="soil" d="' + soilD + '"/>',
            '<path data-role="hatch" d="' + hatchD + '" fill="url(#cp-soil-hatch)"/>',
            '<path data-role="ground" d="' + groundD + '"/>',
            '<text x="' + XM + '" y="' + r(gY(XM) - 9 - Math.abs(g2 - g1) / (AX2 - AX1) * 42)
                + '" text-anchor="middle">Ground level</text>',
            shaft(sx1, g1, sump1 - g1),
            shaft(sx2, g2, sump2 - g2),
            shaftWater(sx1, g1, sump1, inv1, d.inletWaterDepth, inletFlooded, 1, d.inletNode),
            shaftWater(sx2, g2, sump2, inv2, d.outletWaterDepth, outletFlooded, -1, d.outletNode),
            quad('pipe', pa, pb, pipeH),
            quad('water', pa, pb, waterH, waterTitle),
            pipeWalls(pa, pb),
            // After the pipe walls, so each join reads as one outline.
            shaftWall(sx1, g1, sump1, 1),
            shaftWall(sx2, g2, sump2, -1),
            '<path data-role="flow-arrow" d="' + arrowD + '"/>',
            coverDim(sx1 + W + 16, coverValue(d.inletCover, d.inletElevation, inv1), 1),
            coverDim(sx2 - 16, coverValue(d.outletCover, d.outletElevation, inv2), -1),
            lid(sx1, g1),
            lid(sx2, g2),
            floodMarker(AX1, g1, inletFlooded, d.inletFloodHours, d.inletFloodVolume),
            floodMarker(AX2, g2, outletFlooded, d.outletFloodHours, d.outletFloodVolume),
            nodeTxt(AX1, g1, d.inletNode, d.inletElevation, inletFlooded),
            nodeTxt(AX2, g2, d.outletNode, d.outletElevation, outletFlooded),
            invertTxt(AX1, sump1, inv1),
            invertTxt(AX2, sump2, inv2),
            '<text class="cp-accent" x="' + XM + '" y="' + r(labelY) + '" text-anchor="middle">'
                + 'Ø ' + diameterMm + ' mm · ' + nf(slope, 2) + ' ‰</text>',
            '<path data-role="length-dim" d="M' + AX1 + ' 229V241M' + AX2 + ' 229V241M'
                + AX1 + ' 235H' + AX2 + '"/>',
            '<rect data-role="length-label-bg" x="' + (XM - 36) + '" y="224" width="72" height="22"/>',
            '<text x="' + XM + '" y="239" text-anchor="middle">' + lengthText + ' m</text>',
        ].join('');

        const nodeState = function (position, depth, flooded, hours, volume) {
            const parts = [];
            if (Number.isFinite(depth)) {
                parts.push(position + ' manhole water depth ' + nf(depth, 2) + ' m');
            }
            if (flooded) {
                let flooding = position + ' node flooded';
                if (Number.isFinite(hours)) {
                    flooding += hours < 0.005 ? ' for less than 0.01 h' : ' for ' + nf(hours, 2) + ' h';
                }
                if (Number.isFinite(volume)) {
                    flooding += ', flood volume ' + nf(volume, 3) + ' million litres';
                }
                parts.push(flooding);
            } else if (Number.isFinite(hours) || Number.isFinite(volume)) {
                parts.push(position + ' node not flooded');
            }
            return parts;
        };
        const nodeStates = nodeState('Inlet', d.inletWaterDepth, inletFlooded, d.inletFloodHours, d.inletFloodVolume)
            .concat(nodeState('Outlet', d.outletWaterDepth, outletFlooded, d.outletFloodHours, d.outletFloodVolume));

        svg.setAttribute(
            'aria-label',
            'Longitudinal profile of conduit ' + d.name + ' between nodes ' + d.inletNode
            + ' and ' + d.outletNode + ', length ' + lengthText + ' m, diameter '
            + diameterMm + ' mm, filling ' + nf(fillingPercent, 0)
            + ' %. Vertical exaggeration \u00d7' + exaggeration + '.'
            + (nodeStates.length ? ' ' + nodeStates.join('. ') + '.' : '')
        );
        return exaggeration;
    }

    /**
     * Caption under the drawing: which result each water level shows and, only when the
     * vertical scale is stretched, by how much. Empty when nothing was drawn (0).
     */
    function profileCaption(exaggeration) {
        if (!exaggeration) {
            return '';
        }
        return 'Water: manhole = max node depth · pipe = max conduit depth'
            + (exaggeration > 1 ? ' · vertical exaggeration ×' + exaggeration : '');
    }

    function init() {
        const svg = document.querySelector('.conduit-profile');
        if (!svg) {
            return;
        }
        // The drawing is wider than the viewport on a narrow screen, so its box is a
        // focusable scroll region and the left/right arrows belong to it. main.js has a
        // document-level shortcut that moves between conduits on the same keys, so
        // without this the arrows would navigate away instead of scrolling - the same
        // collision the tablist handler below resolves the same way. Attached before the
        // data is read: the box scrolls on the "no data" path too (the svg keeps its
        // min-width).
        const scroller = svg.closest('.conduit-profile-scroll');
        if (scroller) {
            scroller.addEventListener('keydown', function (event) {
                if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                    event.stopPropagation();
                }
            });
        }

        const dataElement = document.getElementById('conduit-data');

        let data;
        try {
            // A missing data element throws here too and ends in the same message.
            data = JSON.parse(dataElement.textContent);
        } catch (error) {
            // json_script emits a bare NaN for a non-finite float, which JSON.parse rejects.
            showMessage(svg, NO_DATA_TEXT, 'Longitudinal profile: no data.');
            console.error('conduit-profile: cannot parse the diagram data', error);
            return;
        }

        const exaggeration = renderProfile(svg, data);
        const note = document.getElementById('conduit-profile-note');
        if (note) {
            note.textContent = profileCaption(exaggeration);
            note.hidden = !note.textContent;
        }
    }

    // Under Node (app/sa/tests/harness_conduit_profile.js) there is no document: the
    // drawing is exported for the harness and nothing is wired. The browser takes only
    // the first branch; `module` does not exist there.
    if (typeof document !== 'undefined') {
        document.addEventListener('DOMContentLoaded', init);
    }
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = { renderProfile: renderProfile, profileCaption: profileCaption };
    }
})();

/*
 * Node anchors. The node cards sit in a <details> under the drawing and the results
 * tables link straight at #node-inlet / #node-outlet. A target inside a closed <details>
 * is not rendered, so the browser cannot scroll to it: open the disclosure first, then
 * scroll. (The tabs handler below only looks inside #conduit-details, so it never sees
 * these two.)
 */
(function () {
    'use strict';

    function followHash() {
        const hash = window.location.hash;
        if (hash !== '#node-inlet' && hash !== '#node-outlet') {
            return;
        }
        const target = document.getElementById(hash.slice(1));
        if (!target) {
            return;
        }
        const disclosure = target.closest('details');
        if (disclosure) {
            disclosure.open = true;
        }
        target.scrollIntoView();
    }

    if (typeof document === 'undefined') {
        return;  // required by the Node harness: nothing to wire
    }
    document.addEventListener('DOMContentLoaded', followHash);
    window.addEventListener('hashchange', followHash);
})();

/*
 * Details tabs. The server renders all three panels; without this script they are all
 * visible one under another, which is the accessible fallback. The script hides the
 * inactive ones, wires the roving tab index and the left/right arrow keys, and opens
 * the panel a fragment link points into (a hidden target gets no native scroll).
 */
(function () {
    'use strict';

    function init() {
        const container = document.getElementById('conduit-details');
        if (!container) {
            return;
        }
        const tabs = Array.from(container.querySelectorAll('[role="tab"]'));
        const panels = tabs.map(function (tab) {
            return document.getElementById(tab.getAttribute('aria-controls'));
        });
        if (!tabs.length || panels.includes(null)) {
            return;
        }

        function select(index, moveFocus) {
            // Focus first: taking the roving tabindex away from the tab that currently
            // has focus makes the browser drop the focus to the document body.
            if (moveFocus) {
                tabs[index].focus();
            }
            tabs.forEach(function (tab, i) {
                const active = i === index;
                tab.setAttribute('aria-selected', String(active));
                tab.tabIndex = active ? 0 : -1;
                panels[i].hidden = !active;
            });
        }

        tabs.forEach(function (tab, index) {
            tab.addEventListener('click', function () {
                select(index, false);
            });
            tab.addEventListener('keydown', function (event) {
                if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') {
                    return;
                }
                event.preventDefault();
                // main.js moves between conduits on the arrow keys from document level;
                // inside the tablist the arrows belong to the tabs.
                event.stopPropagation();
                const step = event.key === 'ArrowRight' ? 1 : -1;
                select((index + step + tabs.length) % tabs.length, true);
            });
        });

        /** Open the panel that contains the current fragment target and scroll to it. */
        function followHash() {
            const hash = window.location.hash;
            if (hash.length < 2) {
                return false;
            }
            const target = document.getElementById(hash.slice(1));
            if (!target || !container.contains(target)) {
                return false;
            }
            const index = panels.findIndex(function (panel) {
                return panel.contains(target);
            });
            if (index < 0) {
                return false;
            }
            select(index, false);
            target.scrollIntoView();
            return true;
        }

        if (!followHash()) {
            select(0, false);
        }
        window.addEventListener('hashchange', followHash);
    }

    if (typeof document !== 'undefined') {  // absent under the Node harness
        document.addEventListener('DOMContentLoaded', init);
    }
})();
