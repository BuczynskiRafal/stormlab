/*
 * docs.js - the table of contents of the documentation pages (About, Method).
 * Loaded from {% block extra_js %} of the page with the defer attribute, so the
 * document is parsed when it runs and the script ends before DOMContentLoaded.
 *
 * Two jobs:
 *   1. --docs-anchor-offset: the distance of an anchor from the top of the window,
 *      computed from the MEASURED height of .site-header, not from the --header-h
 *      token. The header is 72 px high at every viewport, but the measurement
 *      stays: a wrapped or, one day, taller bar must not hide the target of an
 *      anchor under the header. The property has to be set before the browser
 *      jumps to the URL fragment, which is why it happens in top-level code and
 *      not in DOMContentLoaded.
 *   2. The active entry of the table of contents: computed deterministically on
 *      the scroll event. With two levels of targets and a short last section an
 *      intersection observer would give a result that depends on how fast the
 *      page is scrolled.
 *
 * Smooth scrolling and scroll-margin-top belong to docs.css - there is no
 * preventDefault and no window.scrollTo here.
 */

(function () {
    'use strict';

    var ANCHOR_GAP = 16; // 1rem below the navigation bar
    var ACTIVE_BAND = 24; // a target counts as passed when it is just below the bar
    var BOTTOM_EPSILON = 2; // tolerance for a sub-pixel document height

    var nav = document.querySelector('.docs-nav');
    if (!nav) {
        return;
    }

    var list = nav.querySelector('.docs-nav-list');
    var siteHeader = document.querySelector('.site-header');

    // (link, target) pairs in the order of the table of contents. The targets are
    // exactly the elements the hrefs point to: the id of a <section> for a
    // main entry, the id of an <h3> for a sub-entry.
    var entries = [];
    Array.prototype.forEach.call(nav.querySelectorAll('a[href^="#"]'), function (link) {
        var id = decodeURIComponent(link.getAttribute('href').slice(1));
        var target = id ? document.getElementById(id) : null;
        if (target) {
            entries.push({ link: link, target: target });
        }
    });

    if (!entries.length) {
        return;
    }

    function headerHeight() {
        return siteHeader ? siteHeader.getBoundingClientRect().height : 0;
    }

    function setAnchorOffset() {
        document.documentElement.style.setProperty('--docs-anchor-offset', headerHeight() + ANCHOR_GAP + 'px');
    }

    // The active chip has to be visible in the bar that scrolls sideways on a phone.
    // scrollLeft is set by hand: scrollIntoView would scroll the page as well.
    function revealInList(link) {
        if (!list || list.scrollWidth <= list.clientWidth) {
            return;
        }
        var listBox = list.getBoundingClientRect();
        var linkBox = link.getBoundingClientRect();
        if (linkBox.left < listBox.left) {
            list.scrollLeft -= listBox.left - linkBox.left + 12;
        } else if (linkBox.right > listBox.right) {
            list.scrollLeft += linkBox.right - listBox.right + 12;
        }
    }

    var activeIndex = -1;
    // An entry the reader clicked stays active until they scroll by themselves: a
    // target near the end of the page cannot reach the header line, so the position
    // alone would name a later entry.
    var pinnedIndex = -1;

    function currentIndex() {
        if (pinnedIndex >= 0) {
            return pinnedIndex;
        }
        var top = headerHeight() + ACTIVE_BAND;
        // Over the last screen of the page the activation line slides from below the
        // header to the bottom of the viewport, so the last sections, which can never
        // scroll up to the header, still become active one after another.
        var remaining = document.documentElement.scrollHeight - (window.scrollY + window.innerHeight);
        var reach = Math.min(1, Math.max(0, 1 - remaining / window.innerHeight));
        var line = top + (window.innerHeight - top - ACTIVE_BAND) * reach;
        var index = 0; // no target above the line (top of the page) => the first entry
        for (var i = 0; i < entries.length; i++) {
            if (entries[i].target.getBoundingClientRect().top <= line) {
                index = i;
            }
        }
        var scrollBottom = window.scrollY + window.innerHeight;
        if (scrollBottom >= document.documentElement.scrollHeight - BOTTOM_EPSILON) {
            index = entries.length - 1; // end of the page => the last entry
        }
        return index;
    }

    function updateActive() {
        var index = currentIndex();
        if (index === activeIndex) {
            return;
        }
        activeIndex = index;
        for (var i = 0; i < entries.length; i++) {
            var link = entries[i].link;
            if (i === index) {
                link.classList.add('active');
                link.setAttribute('aria-current', 'true');
            } else {
                link.classList.remove('active');
                link.removeAttribute('aria-current');
            }
        }
        revealInList(entries[index].link);
    }

    // A deep link from the address bar: the browser jumps to the fragment before the
    // images and the fonts settle the height of the document (measured at 390 px:
    // -233 px between DOMContentLoaded and load), so the target lands a dozen or so
    // pixels too high - under the navigation bar. After `load` it is moved into
    // place, but only while the reader has not moved the page.
    var userMoved = false;

    function markUserMoved() {
        userMoved = true;
    }

    function unpin() {
        if (pinnedIndex >= 0) {
            pinnedIndex = -1;
            schedule();
        }
    }

    ['wheel', 'touchstart', 'keydown'].forEach(function (name) {
        window.addEventListener(name, unpin, { passive: true });
    });

    entries.forEach(function (entry, index) {
        entry.link.addEventListener('click', function () {
            pinnedIndex = index;
            updateActive();
        });
    });

    ['wheel', 'touchstart', 'keydown'].forEach(function (name) {
        window.addEventListener(name, markUserMoved, { passive: true, once: true });
    });

    function alignHashTarget() {
        if (userMoved || !location.hash) {
            return;
        }
        var id = decodeURIComponent(location.hash.slice(1));
        var target = id ? document.getElementById(id) : null;
        if (!target) {
            return;
        }
        var delta = target.getBoundingClientRect().top - (headerHeight() + ANCHOR_GAP);
        if (Math.abs(delta) >= 1) {
            // behavior: 'auto' overrides scroll-behavior: smooth - the correction is meant
            // to be invisible, not animated half a second after the page opened.
            window.scrollBy({ top: delta, left: 0, behavior: 'auto' });
        }
    }

    var scheduled = false;

    function schedule() {
        if (scheduled) {
            return;
        }
        scheduled = true;
        window.requestAnimationFrame(function () {
            scheduled = false;
            updateActive();
        });
    }

    function refresh() {
        setAnchorOffset();
        updateActive();
    }

    setAnchorOffset();
    updateActive();

    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', refresh);
    window.addEventListener('hashchange', schedule);
    document.addEventListener('DOMContentLoaded', refresh);
    // On a narrow screen the height of the bar depends on how the navigation wraps,
    // and that on the loaded fonts - compute the offset once more after `load`.
    window.addEventListener('load', function () {
        setAnchorOffset();
        alignHashTarget();
        updateActive();
    });
})();

/*
 * Wide figures of the Method page (.method-fig-scroll). On a narrow screen a drawing
 * keeps a legible size and scrolls sideways; without a hint half of it went unnoticed.
 * While a scroller overflows it gets .is-scrollable (edge fades in method-figures.css)
 * and .is-at-start / .is-at-end, and the "Scroll to see the rest" line under the figure
 * is shown until the reader has scrolled it.
 */
(function () {
    'use strict';

    var SCROLLED_PX = 24;
    var scrollers = document.querySelectorAll('.method-fig-scroll');
    if (!scrollers.length) {
        return;
    }

    Array.prototype.forEach.call(scrollers, function (scroller) {
        var figure = scroller.closest('figure');
        var hint = figure ? figure.querySelector('.method-fig-hint') : null;
        var scrolled = false;

        function sync() {
            var overflow = scroller.scrollWidth - scroller.clientWidth > 1;
            var left = scroller.scrollLeft;
            if (left > SCROLLED_PX) {
                scrolled = true;
            }
            scroller.classList.toggle('is-scrollable', overflow);
            scroller.classList.toggle('is-at-start', left <= 1);
            scroller.classList.toggle('is-at-end', left >= scroller.scrollWidth - scroller.clientWidth - 1);
            if (hint) {
                hint.hidden = !overflow || scrolled;
            }
        }

        scroller.addEventListener('scroll', sync, { passive: true });
        window.addEventListener('resize', sync);
        sync();
    });
})();
