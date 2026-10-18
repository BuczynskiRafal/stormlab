/*
 * docs.js - spis treści strony dokumentacji (/about/).
 * Ładowany z {% block extra_js %} w about.html z atrybutem defer, więc w chwili
 * wykonania dokument jest sparsowany, a skrypt kończy się przed DOMContentLoaded.
 *
 * Dwa zadania:
 *   1. --docs-anchor-offset: odstęp kotwic od góry okna, liczony ze ZMIERZONEJ
 *      wysokości .site-header, nie z tokena --header-h. Od Etapu 7 pasek ma
 *      72 px na każdym viewporcie, ale pomiar zostaje: zawinięta lub w
 *      przyszłości wyższa belka nie schowa celu kotwicy pod nagłówkiem.
 *      Ustawienie właściwości musi nastąpić przed skokiem
 *      przeglądarki do fragmentu URL - dlatego dzieje się w kodzie najwyższego
 *      poziomu, nie w DOMContentLoaded.
 *   2. Aktywna pozycja spisu treści: liczona deterministycznie na zdarzeniu
 *      scroll. Obserwator przecięć dawałby przy dwóch poziomach celów i krótkiej
 *      ostatniej sekcji wynik zależny od prędkości przewijania.
 *
 * Płynne przewijanie i scroll-margin-top należą do docs.css - tutaj nie ma
 * żadnego preventDefault ani window.scrollTo.
 */

(function () {
    'use strict';

    var ANCHOR_GAP = 16; // 1rem pod paskiem nawigacji
    var ACTIVE_BAND = 24; // cel liczy się jako minięty, gdy jest tuż pod paskiem
    var BOTTOM_EPSILON = 2; // tolerancja dla subpikselowej wysokości dokumentu

    var nav = document.querySelector('.docs-nav');
    if (!nav) {
        return;
    }

    var list = nav.querySelector('.docs-nav-list');
    var siteHeader = document.querySelector('.site-header');

    // Pary (link, cel) w kolejności występowania w spisie treści. Cele to
    // dokładnie te elementy, na które wskazują href-y: id na <section> dla
    // pozycji głównych, id na <h3> dla podpozycji.
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

    // Aktywny chip musi być widoczny w poziomo przewijanym pasku na mobile.
    // Ustawiamy scrollLeft ręcznie: scrollIntoView przewinęłoby także stronę.
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

    function currentIndex() {
        var line = headerHeight() + ACTIVE_BAND;
        var index = 0; // żaden cel nad linią (początek strony) => pierwsza pozycja
        for (var i = 0; i < entries.length; i++) {
            if (entries[i].target.getBoundingClientRect().top <= line) {
                index = i;
            }
        }
        var scrollBottom = window.scrollY + window.innerHeight;
        if (scrollBottom >= document.documentElement.scrollHeight - BOTTOM_EPSILON) {
            index = entries.length - 1; // koniec strony => ostatnia pozycja
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

    // Deep link z paska adresu: przeglądarka skacze do fragmentu, zanim obrazki
    // i kroje pisma ustalą wysokość dokumentu (zmierzone przy 390 px: -233 px
    // między DOMContentLoaded a load), więc cel ląduje kilkanaście pikseli za
    // wysoko - pod paskiem nawigacji. Po `load` dosuwamy go, ale tylko dopóki
    // użytkownik sam nie ruszył strony.
    var userMoved = false;

    function markUserMoved() {
        userMoved = true;
    }

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
            // behavior: 'auto' bije scroll-behavior: smooth - poprawka ma być
            // niewidoczna, a nie animowana pół sekundy po otwarciu strony.
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
    // Wysokość paska na wąskim ekranie zależy od zawijania nawigacji, a to od
    // wczytanych krojów pisma - po `load` przelicz offset jeszcze raz.
    window.addEventListener('load', function () {
        setAnchorOffset();
        alignHashTarget();
        updateActive();
    });
})();
