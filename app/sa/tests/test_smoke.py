"""Smoke tests for every view the stylesheet contract touches.

They are deliberately shallow - status code plus a couple of markers of the
stylesheet contract - because their job is to be the regression gate for the CSS
and template work, not to test view logic.
"""

from __future__ import annotations

import re

from django.test import TestCase
from django.urls import reverse

from .factories import VALIDATION_FLAGS, make_conduit, make_session, make_user


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

        self.assertEqual(html.count('<svg viewBox="'), 3, "three static figures, drawn in the template")
        self.assertEqual(html.count("<title"), 4, "the page title and one title per static figure")
        self.assertEqual(html.count("<desc"), 3, "every static figure describes itself to a screen reader")
        self.assertEqual(html.count("<figcaption"), 4)
        self.assertContains(self.response, "data-method-graph")
        self.assertContains(self.response, "data/method-graph.json")

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
