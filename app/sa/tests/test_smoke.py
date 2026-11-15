"""Smoke tests for every view the stylesheet contract touches.

They are deliberately shallow - status code plus a couple of markers of the
stylesheet contract - because their job is to be the regression gate for the CSS
and template work, not to test view logic.
"""

from __future__ import annotations

import re

from django.core.exceptions import PermissionDenied, SuspiciousOperation
from django.template import loader
from django.test import TestCase, override_settings
from django.urls import path as url_path
from django.urls import reverse

from storm_analysis.urls import urlpatterns as site_urlpatterns

from .factories import VALIDATION_FLAGS, make_conduit, make_session, make_swmm_model, make_user


def _raise(exception):
    def view(request):
        raise exception

    return view


# ROOT_URLCONF of ErrorPageTests: the site plus three routes that fail on purpose.
urlpatterns = [
    url_path("test-errors/400/", _raise(SuspiciousOperation("bad"))),
    url_path("test-errors/403/", _raise(PermissionDenied())),
    url_path("test-errors/500/", _raise(RuntimeError("boom"))),
    *site_urlpatterns,
]


class PublicPageTests(TestCase):
    """Pages that do not require a login."""

    def test_index_renders_for_anonymous_visitor(self):
        response = self.client.get(reverse("sa:index"))

        self.assertEqual(response.status_code, 200)

    def test_about_renders_for_anonymous_visitor(self):
        response = self.client.get(reverse("sa:about"))

        self.assertEqual(response.status_code, 200)

    def test_method_renders_for_anonymous_visitor(self):
        response = self.client.get(reverse("sa:method"))

        self.assertEqual(response.status_code, 200)

    def test_history_redirects_anonymous_visitor_to_login(self):
        response = self.client.get(reverse("sa:history"))

        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])


class SiteChromeContractTests(TestCase):
    """base.html carries the one navigation the whole site shares."""

    def test_header_offers_a_single_bootstrap_toggler_for_the_mobile_menu(self):
        response = self.client.get(reverse("sa:index"))

        self.assertContains(response, "navbar-toggler", count=1)
        self.assertContains(response, 'data-bs-target="#siteNavCollapse"')
        self.assertContains(response, 'id="siteNavCollapse"')

    def test_skip_link_is_the_first_focusable_element_and_targets_main(self):
        html = self.client.get(reverse("sa:index")).content.decode()
        body = html[html.index("<body>") :]

        first_link = re.search(r"<(a|button)\b[^>]*>", body).group(0)
        self.assertIn('class="skip-link" href="#main-content"', first_link)
        self.assertIn('<main class="containerwide content-wrapper" id="main-content" tabindex="-1">', body)

    def test_brand_is_named_stormlab_in_text_and_for_screen_readers(self):
        response = self.client.get(reverse("sa:index"))

        self.assertContains(response, 'aria-label="StormLab, home"', count=2)
        self.assertContains(response, '<span class="site-wordmark" aria-hidden="true">StormLab</span>')
        self.assertNotContains(response, "Stormwater analysis - home")

    def test_every_title_names_the_page_then_the_product(self):
        self.assertContains(
            self.client.get(reverse("sa:index")), "<title>StormLab · Decision support for storm sewer design</title>"
        )
        for name, title in (("sa:about", "About"), ("sa:method", "Method"), ("accounts:login", "Log in")):
            with self.subTest(page=name):
                self.assertContains(self.client.get(reverse(name)), f"<title>{title} · StormLab</title>")

    def test_auth_actions_are_two_word_verbs(self):
        anonymous = self.client.get(reverse("sa:index"))
        self.assertContains(anonymous, ">Log in</a>")
        self.assertContains(anonymous, ">Sign up</a>")
        self.assertNotContains(anonymous, ">Login<")
        self.assertNotContains(anonymous, "Sign-up")

        self.client.force_login(make_user(email="chrome@example.com"))
        member = self.client.get(reverse("sa:index"))
        self.assertContains(member, ">Log out</button>")
        self.assertNotContains(member, ">Logout<")

    def test_square_icons_are_linked(self):
        response = self.client.get(reverse("sa:index"))

        self.assertContains(response, 'rel="icon" type="image/png" sizes="32x32" href="/static/img/favicon-32.png"')
        self.assertContains(response, 'rel="apple-touch-icon" sizes="180x180" href="/static/img/apple-touch-icon.png"')
        self.assertNotContains(response, 'rel="icon" type="image/png" href="/static/img/logo.png"')

    def test_icon_files_are_square_and_small(self):
        from pathlib import Path

        from PIL import Image

        img_dir = Path(__file__).resolve().parents[2] / "static" / "img"
        for name, size in (("favicon-32.png", 32), ("apple-touch-icon.png", 180), ("icon-512.png", 512)):
            with self.subTest(icon=name), Image.open(img_dir / name) as icon:
                self.assertEqual(icon.size, (size, size))
                self.assertLess((img_dir / name).stat().st_size, 16 * 1024)

    def test_login_page_renders_the_auth_card_without_a_green_submit(self):
        response = self.client.get(reverse("accounts:login"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "auth-card")
        self.assertNotContains(response, "btn-success")


class MessagesAreRenderedTests(TestCase):
    """base.html shows the django.contrib.messages of the previous request under the header.

    Before, no template looped over ``messages``: every confirmation and every reason
    for a redirect was queued and silently dropped.
    """

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="messages@example.com")
        cls.session = make_session(cls.user, n_conduits=1, n_nodes=2, n_subcatchments=1)

    def setUp(self):
        self.client.force_login(self.user)

    def test_missing_conduit_redirect_explains_itself_as_an_alert(self):
        url = reverse("sa:conduit_detail", args=[self.session.id, "NOPE"])

        response = self.client.get(url, follow=True)

        self.assertEqual(response.redirect_chain[-1][0], reverse("sa:analysis_results", args=[self.session.id]))
        self.assertContains(response, 'class="containerwide site-messages"')
        self.assertContains(response, "This analysis has no conduit named “NOPE”.")
        self.assertContains(response, 'site-message--error alert alert-dismissible fade show"\n         role="alert"')

    def test_failed_session_redirect_explains_itself_on_the_analysis_form(self):
        failed = make_session(self.user, status="failed")

        response = self.client.get(reverse("sa:analysis_results", args=[failed.id]), follow=True)

        self.assertEqual(response.redirect_chain[-1][0], reverse("sa:analysis"))
        self.assertContains(response, "site-message--warning")
        self.assertContains(response, 'id="uploadForm"')

    def test_success_is_announced_politely_and_can_be_dismissed(self):
        response = self.client.post(
            reverse("accounts:user_profile"),
            {"user_profile_forms": "0", "email": self.user.email, "first_name": "Ada", "last_name": ""},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "site-message--success")
        self.assertContains(response, 'role="status"')
        self.assertContains(response, 'data-bs-dismiss="alert"')

    def test_a_page_without_messages_renders_no_strip(self):
        response = self.client.get(reverse("sa:history"))

        self.assertNotContains(response, "site-messages")


class AboutAndHomeWordingTests(TestCase):
    """The shared vocabulary: design checks, action labels, frost zones, heading levels."""

    def test_home_cards_are_level_two_and_name_the_design_checks(self):
        response = self.client.get(reverse("sa:index"))

        self.assertContains(response, '<h2 class="feature-title">Design checks</h2>')
        self.assertNotContains(response, "Validation")
        self.assertNotContains(response, "<h3")

    def test_about_names_the_design_checks_and_keeps_the_anchor(self):
        response = self.client.get(reverse("sa:about"))

        self.assertContains(response, '<h3 id="verification">Design checks</h3>')
        self.assertContains(response, 'href="#verification">Design checks</a>')
        self.assertNotContains(response, "Hydraulic verification")

    def test_about_action_table_shows_labels_not_raw_ids(self):
        html = self.client.get(reverse("sa:about")).content.decode()
        labels = re.findall(r"<tr><td>([^<]+)</td><td>[^<]+</td></tr>", html[html.index("<th>Action</th>") :])[:9]

        self.assertEqual(
            labels,
            [
                "No changes",
                "Increase diameter",
                "Reduce diameter",
                "Increase slope",
                "Reduce slope",
                "Increase depth",
                "Retention tank",
                "Seepage boxes",
                "Pump station",
            ],
        )
        for raw in ("<td>valid</td>", "<td>diameter_increase</td>", "<td>seepage_boxes</td>"):
            self.assertNotIn(raw, html)

    def test_about_frost_zones_use_the_shared_labels(self):
        response = self.client.get(reverse("sa:about"))

        for zone, depth in (("I", "0.8"), ("II", "1.0"), ("III", "1.2"), ("IV", "1.4")):
            self.assertContains(response, f"<tr><td>Zone {zone}</td><td>{depth} m</td></tr>")


class DocsPageStyleTests(TestCase):
    """docs.css and docs.js: chips in a row, readable code, no dead space before the footer."""

    @classmethod
    def setUpTestData(cls):
        from pathlib import Path

        static = Path(__file__).resolve().parents[2] / "static"
        cls.css = (static / "css" / "docs.css").read_text(encoding="utf-8")
        cls.js = (static / "js" / "docs.js").read_text(encoding="utf-8")

    def test_stat_chips_are_not_indented_or_capped_by_the_section_list_rule(self):
        self.assertIn(".docs-section .docs-stats,\n.docs-stats {", self.css)
        self.assertIn("max-width: none;", self.css)

    def test_inline_code_uses_ink_instead_of_bootstraps_pink(self):
        rule = self.css[self.css.index(".docs-section code {") :]
        self.assertIn("color: var(--ink);", rule[: rule.index("}")])

    def test_print_drops_the_site_chrome_and_keeps_blocks_whole(self):
        printed = self.css[self.css.index("@media print {") :]

        for selector in (".site-header,", ".site-footer,", "#backToTop {"):
            self.assertIn(selector, printed)
        self.assertIn("break-after: avoid;", printed)
        self.assertIn("break-inside: avoid;", printed)

    def test_about_serves_versioned_docs_assets(self):
        response = self.client.get(reverse("sa:about"))

        self.assertContains(response, "/static/css/docs.css?v=")
        self.assertContains(response, "/static/js/docs.js?v=")

    def test_no_half_screen_padding_before_the_footer(self):
        self.assertNotIn("50vh", self.css)
        self.assertIn("pinnedIndex = index;", self.js)
        self.assertIn("var reach = Math.min(1, Math.max(0, 1 - remaining / window.innerHeight));", self.js)


class PhoneLayoutContractTests(TestCase):
    """One 16 px gutter on phones; the back-to-top button stays clear of the footer."""

    @classmethod
    def setUpTestData(cls):
        from pathlib import Path

        static = Path(__file__).resolve().parents[2] / "static"
        cls.css = (static / "css" / "app.css").read_text(encoding="utf-8")
        cls.js = (static / "js" / "main.js").read_text(encoding="utf-8")

    def test_page_gutter_drops_to_16px_below_576px(self):
        self.assertIn("padding-left: var(--page-gutter);", self.css)
        self.assertRegex(self.css, r"@media \(max-width: 575\.98px\) \{\s*:root \{\s*--page-gutter: 1rem;")

    def test_full_bleed_conduit_panel_follows_the_gutter(self):
        block = self.css[self.css.index("/* BEGIN page-gutter */") : self.css.index("/* END page-gutter */")]
        self.assertIn("margin-left: calc(-1 * var(--page-gutter));", block)
        self.assertGreater(self.css.index("/* BEGIN page-gutter */"), self.css.index("/* END conduit-panel */"))

    def test_back_to_top_is_lifted_above_the_footer_and_hidden_while_reading_on_phones(self):
        self.assertIn("bottom: calc(1.5rem + var(--back-to-top-lift, 0px));", self.css)
        self.assertIn("window.innerHeight - footer.getBoundingClientRect().top", self.js)
        self.assertIn("(!narrow.matches || scrollingUp)", self.js)


class StylesheetContractTests(TestCase):
    """base.html must serve exactly one stylesheet and no unused font family."""

    def test_page_links_the_consolidated_stylesheet(self):
        response = self.client.get(reverse("sa:index"))

        self.assertContains(response, "css/app.css")
        self.assertNotContains(response, "css/style.css")
        self.assertNotContains(response, "css/academic-theme.css")

    def test_page_does_not_load_the_unused_ibm_plex_font(self):
        response = self.client.get(reverse("sa:index"))

        self.assertNotContains(response, "IBM+Plex")

    def test_about_serves_its_styles_and_script_as_static_files(self):
        response = self.client.get(reverse("sa:about"))

        self.assertContains(response, "css/docs.css")
        self.assertContains(response, "js/docs.js")

    def test_about_has_no_inline_style_block(self):
        response = self.client.get(reverse("sa:about"))

        self.assertNotContains(response, "<style")

    def test_header_and_footer_link_to_the_method_page(self):
        response = self.client.get(reverse("sa:index"))

        self.assertContains(response, f'href="{reverse("sa:method")}"', count=3)

    def test_method_link_is_marked_active_on_the_method_page_only(self):
        active = f'href="{reverse("sa:method")}" class="nav-link is-active"'

        self.assertContains(self.client.get(reverse("sa:method")), active, count=1)
        self.assertNotContains(self.client.get(reverse("sa:about")), active)


class MethodPageContractTests(TestCase):
    """The Method page: its assets, its figures and the anchors other pages link to."""

    @classmethod
    def setUpTestData(cls):
        cls.url = reverse("sa:method")

    def setUp(self):
        self.response = self.client.get(self.url)

    def test_serves_its_styles_and_scripts_as_static_files(self):
        for asset in ("css/docs.css", "css/method.css", "js/docs.js", "js/method-graph.js"):
            self.assertContains(self.response, asset)

    def test_has_no_inline_style_or_script_block(self):
        self.assertNotContains(self.response, "<style")
        self.assertNotContains(self.response, "<script>")

    def test_every_entry_of_the_table_of_contents_has_its_target(self):
        """A link without a target would leave docs.js with a dead entry and the reader with a dead click."""
        html = self.response.content.decode()
        anchors = re.findall(r'class="docs-nav-link[^"]*" href="#([^"]+)"', html)

        self.assertGreaterEqual(len(anchors), 10)
        for anchor in anchors:
            self.assertEqual(html.count(f'id="{anchor}"'), 1, f"#{anchor} must exist exactly once")

    def test_anchors_linked_from_the_about_page_exist(self):
        about = self.client.get(reverse("sa:about")).content.decode()
        linked = set(re.findall(re.escape(self.url) + r"#([a-z-]+)", about))

        self.assertEqual(linked, {"graph", "experiments"})
        for anchor in linked:
            self.assertContains(self.response, f'id="{anchor}"')

    def test_figures_are_rendered_with_text_alternatives(self):
        html = self.response.content.decode()

        self.assertEqual(html.count('<figure class="method-fig '), 3, "three static figures, drawn in the template")
        self.assertEqual(html.count("<title"), 4, "the page title and one title per static figure")
        self.assertEqual(html.count("<desc"), 3, "every static figure describes itself to a screen reader")
        # Figure 2 is two drawings (stacked on phones); only the first is announced, its
        # description covers both panels.
        self.assertEqual(html.count('role="img" aria-labelledby="mf-'), 3)
        self.assertEqual(html.count("<figcaption"), 5)
        self.assertContains(self.response, "data-method-graph")
        self.assertContains(self.response, "data/method-graph.json")

    def test_all_five_figures_are_numbered_in_page_order(self):
        html = self.response.content.decode()

        self.assertEqual(re.findall(r"<strong>Figure (\d)\.</strong>", html), ["1", "2", "3", "4", "5"])

    def test_figure_two_stacks_its_panels_instead_of_scrolling(self):
        html = self.response.content.decode()
        figure = html[html.index('<figure class="method-fig method-fig--conduit-graph">') :]
        figure = figure[: figure.index("</figure>")]

        self.assertNotIn("method-fig-scroll", figure)
        self.assertIn('class="method-fig-panel method-fig-panel--graph" viewBox="392 0 328 334"', figure)

    def test_wide_figures_scroll_with_a_keyboard_focus_and_a_hint(self):
        self.assertContains(self.response, 'class="method-fig-scroll" tabindex="0" role="region"', count=2)
        self.assertContains(self.response, '<p class="method-fig-hint" hidden>Scroll to see the rest', count=2)

    def test_results_table_stacks_on_phones_with_labelled_values(self):
        self.assertContains(self.response, 'class="docs-table method-stack method-results"')
        self.assertContains(self.response, '<td data-label="MLP + neighbourhood">0.886</td>')

    def test_about_limitations_anchor_linked_from_the_method_page_exists(self):
        self.assertContains(self.response, f"{reverse('sa:about')}#limitations")
        self.assertContains(self.client.get(reverse("sa:about")), 'id="limitations"')


class AuthenticatedPageTests(TestCase):
    """Pages behind the login, on a session with one conduit, two nodes, one subcatchment."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user()
        cls.session = make_session(cls.user, n_conduits=1, n_nodes=2, n_subcatchments=1)

    def setUp(self):
        self.client.force_login(self.user)

    def test_history_renders_for_logged_in_user(self):
        response = self.client.get(reverse("sa:history"))

        self.assertEqual(response.status_code, 200)

    def test_analysis_results_renders_for_completed_session(self):
        response = self.client.get(reverse("sa:analysis_results", args=[self.session.id]))

        self.assertEqual(response.status_code, 200)

    def test_analysis_results_does_not_serve_jquery_and_datatables(self):
        response = self.client.get(reverse("sa:analysis_results", args=[self.session.id]))

        self.assertNotContains(response, "code.jquery.com")
        self.assertNotContains(response, "cdn.datatables.net")

    def test_analysis_upload_form_does_not_serve_jquery_and_datatables(self):
        response = self.client.get(reverse("sa:analysis"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "code.jquery.com")
        self.assertNotContains(response, "cdn.datatables.net")

    def test_conduit_detail_renders(self):
        response = self.client.get(reverse("sa:conduit_detail", args=[self.session.id, "C1"]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Results and design criteria")

    def test_node_and_subcatchment_pages_are_gone(self):
        """Nodes and subcatchments live on the conduit panel now; the old routes do not resolve."""
        for path in (
            f"/analysis/results/{self.session.id}/node/N1/",
            f"/analysis/results/{self.session.id}/subcatchment/S1/",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)

    def test_conduit_detail_counts_failed_validation_checks(self):
        failing = dict.fromkeys(VALIDATION_FLAGS, 1)
        failing["val_max_fill"] = 0
        failing["val_coverage"] = 0
        make_conduit(self.session, conduit_name="C-fail", **failing)

        response = self.client.get(reverse("sa:conduit_detail", args=[self.session.id, "C-fail"]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "2 of 7 checks failed")


@override_settings(DEBUG=False, ROOT_URLCONF=__name__)
class ErrorPageTests(TestCase):
    """With DEBUG off, errors answer with pages in the site's style instead of Django's plain text."""

    def setUp(self):
        self.client.raise_request_exception = False

    def assert_error_page(self, response, status, heading):
        self.assertEqual(response.status_code, status)
        self.assertContains(response, f'<p class="error-code">Error {status}</p>', status_code=status)
        self.assertContains(response, f'<h1 class="error-title">{heading}</h1>', status_code=status)
        self.assertContains(response, 'href="/">Home</a>', status_code=status)
        self.assertContains(response, 'href="/analysis/">Analysis</a>', status_code=status)

    def test_unknown_address_gets_the_404_page_with_the_site_header(self):
        response = self.client.get("/no-such-page/")

        self.assert_error_page(response, 404, "Page not found")
        self.assertContains(response, "<title>Page not found · StormLab</title>", status_code=404)
        self.assertContains(response, 'class="site-header"', status_code=404)

    def test_permission_denied_gets_the_403_page(self):
        self.assert_error_page(self.client.get("/test-errors/403/"), 403, "Access denied")

    def test_suspicious_request_gets_the_400_page(self):
        self.assert_error_page(self.client.get("/test-errors/400/"), 400, "Bad request")

    def test_server_error_gets_the_500_page(self):
        response = self.client.get("/test-errors/500/")

        self.assert_error_page(response, 500, "Something went wrong")
        self.assertContains(response, "<title>Server error · StormLab</title>", status_code=500)

    def test_500_page_renders_without_a_request_or_context_processors(self):
        """Django renders 500.html with an empty context; it must not depend on the user, the database or {% url %}."""
        html = loader.get_template("500.html").render()

        self.assertIn("Something went wrong", html)
        self.assertIn("/static/css/app.css", html)
        source = open(loader.get_template("500.html").origin.name, encoding="utf-8").read()
        for dependency in ("{% extends", "{% url", "user.", "registration_open", "messages"):
            self.assertNotIn(dependency, source.split("{% endcomment %}")[1], dependency)


@override_settings(DEBUG=False, ROOT_URLCONF=__name__)
class PageTitleTests(TestCase):
    """Every <title> is "<page> · StormLab", the product named once, at the end.

    base.html appends the suffix; a page that writes it as well doubles it. The home
    page alone leads with the product ("StormLab · Decision support ...").
    """

    HOME_TITLE = "StormLab · Decision support for storm sewer design"

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="titles@example.com")
        cls.session = make_session(cls.user, n_conduits=1, n_nodes=2, n_subcatchments=1)

    def setUp(self):
        self.client.raise_request_exception = False

    def title_of(self, response) -> str:
        found = re.findall(r"<title>(.*?)</title>", response.content.decode(), re.DOTALL)
        self.assertEqual(len(found), 1, "one <title> per page")
        return " ".join(found[0].split())

    def assert_titles(self, pages):
        for url, page in pages:
            with self.subTest(url=url):
                title = self.title_of(self.client.get(url))
                self.assertEqual(title, f"{page} · StormLab")
                self.assertEqual(title.count("StormLab"), 1)

    def test_public_pages(self):
        self.assertEqual(self.title_of(self.client.get(reverse("sa:index"))), self.HOME_TITLE)
        self.assert_titles(
            [
                (reverse("sa:about"), "About"),
                (reverse("sa:method"), "Method"),
                (reverse("accounts:login"), "Log in"),
                (reverse("accounts:register"), "Sign up"),
                ("/no-such-page/", "Page not found"),
                ("/test-errors/400/", "Bad request"),
                ("/test-errors/403/", "Access denied"),
                ("/test-errors/500/", "Server error"),
            ]
        )

    def test_pages_behind_the_login(self):
        self.client.force_login(self.user)
        file_name = self.session.swmm_model.file.name.rsplit("/", 1)[-1]
        self.assertEqual(self.title_of(self.client.get(reverse("sa:index"))), self.HOME_TITLE)
        self.assert_titles(
            [
                (reverse("sa:analysis"), "Analysis"),
                (reverse("sa:history"), "History"),
                (reverse("sa:analysis_results", args=[self.session.id]), f"{file_name} · Results"),
                (reverse("sa:conduit_detail", args=[self.session.id, "C1"]), f"Conduit C1 · {file_name}"),
                (reverse("accounts:user_profile"), "Your account"),
                (reverse("accounts:logout"), "Log out"),
            ]
        )


class OneDateAndZoneFormatTests(TestCase):
    """History, the results page (header and Info tab) and the account page write a date
    and a frost zone in one way, through the utc_datetime and zone_label filters."""

    @classmethod
    def setUpTestData(cls):
        import datetime

        from accounts.models import User
        from sa.models import CalculationSession

        stamp = datetime.datetime(2026, 3, 5, 4, 43, tzinfo=datetime.timezone.utc)
        cls.user = make_user(email="formats@example.com")
        User.objects.filter(pk=cls.user.pk).update(date_joined=stamp)
        cls.session = make_session(cls.user, swmm_model=make_swmm_model(cls.user, zone=3), n_conduits=1)
        CalculationSession.objects.filter(pk=cls.session.pk).update(created_at=stamp)

    def setUp(self):
        self.client.force_login(self.user)

    def test_every_page_writes_the_same_date_without_a_leading_zero(self):
        results = self.client.get(reverse("sa:analysis_results", args=[self.session.id])).content.decode()
        history = self.client.get(reverse("sa:history")).content.decode()
        account = self.client.get(reverse("accounts:user_profile")).content.decode()

        self.assertEqual(results.count(">5 Mar 2026, 04:43 UTC</time>"), 2, "the header and the Info tab")
        self.assertIn('<span class="history-date-day">5 Mar 2026</span>', history)
        self.assertIn('<span class="history-date-time">04:43 UTC</span>', history)
        self.assertIn('data-session-date="5 Mar 2026, 04:43 UTC"', history)
        self.assertIn("Member since 5 Mar 2026</p>", account)
        for html in (results, history, account):
            self.assertNotIn("05 Mar", html)

    def test_every_page_names_the_zone_the_same_way(self):
        results = self.client.get(reverse("sa:analysis_results", args=[self.session.id])).content.decode()
        history = self.client.get(reverse("sa:history")).content.decode()

        self.assertEqual(results.count("Zone III (1.2 m)"), 2, "the header and the Info tab")
        self.assertIn('data-label="Frost zone">Zone III (1.2 m)</td>', history)
        self.assertNotIn("Frost zone 3", results)

    def test_no_template_formats_a_date_or_a_zone_by_hand(self):
        from pathlib import Path

        roots = [
            Path(__file__).resolve().parents[2] / sub
            for sub in ("sa/templates", "accounts/templates", "storm_analysis/templates")
        ]
        for template in (path for root in roots for path in root.rglob("*.html")):
            source = template.read_text(encoding="utf-8")
            with self.subTest(template=template.name):
                # Only the machine-readable datetime attribute may use the date filter.
                self.assertEqual(re.findall(r"\|date:(?!'c')", source), [])
                self.assertNotIn("get_frost_zone_value", source)
                self.assertNotIn("frost_zones", source)
