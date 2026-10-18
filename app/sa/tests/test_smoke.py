"""Smoke tests for every view the plan15 restyling touches.

They are deliberately shallow - status code plus a couple of markers of the
stylesheet contract - because their job is to be the regression gate for the CSS
and template work, not to test view logic.
"""

from __future__ import annotations

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
        # The .data-table on /analysis/ only exists once a run produced `data`;
        # the bare upload form has nothing for DataTables to enhance.
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
