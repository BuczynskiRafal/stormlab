"""Results page links into the conduit panel; the conduit view's error paths and prev/next."""

from __future__ import annotations

from django.contrib.messages import get_messages
from django.contrib.messages.storage.fallback import FallbackStorage
from django.test import RequestFactory
from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from django.db.models import Prefetch

from sa.models import CalculationSession, ConduitData, SubcatchmentData
from sa.services import CalculationRetrievalService
from sa.views import AnalysisView

from .factories import make_conduit, make_node, make_session, make_subcatchment, make_user


class PanelLinksTests(TestCase):
    """Nodes and subcatchments on the results page link to the conduit panel that shows them."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="links@example.com")
        cls.session = make_session(cls.user)
        for name in ("N1", "N2", "N3", "N9"):
            make_node(cls.session, node_name=name, subcatchment="-")
        # C1: N1 -> N2, C2: N2 -> N3; N9 touches no conduit.
        make_conduit(cls.session, conduit_name="C1", inlet_node="N1", outlet_node="N2", subcatchment="S1")
        make_conduit(cls.session, conduit_name="C2", inlet_node="N2", outlet_node="N3", subcatchment="-")
        cls.s1 = make_subcatchment(cls.session, subcatchment_name="S1", outlet="N1")  # inlet of C1
        cls.s2 = make_subcatchment(cls.session, subcatchment_name="S2", outlet="N3")  # only an outlet (C2)
        cls.s3 = make_subcatchment(cls.session, subcatchment_name="S3", outlet="N9")  # no conduit at all

    def setUp(self):
        self.client.force_login(self.user)

    def _panel(self, conduit_name):
        return reverse("sa:conduit_detail", args=[self.session.id, conduit_name])

    def test_panel_links_targets(self):
        links = CalculationRetrievalService.panel_links(self.session)

        self.assertEqual(links["Nodes Data"]["N1"], self._panel("C1") + "#node-inlet")
        self.assertEqual(links["Nodes Data"]["N2"], self._panel("C2") + "#node-inlet")  # inlet wins over outlet
        self.assertEqual(links["Nodes Data"]["N3"], self._panel("C2") + "#node-outlet")
        self.assertNotIn("N9", links["Nodes Data"])
        self.assertEqual(links["Subcatchments Data"]["S1"], self._panel("C1") + f"#catchment-{self.s1.pk}")
        self.assertEqual(links["Subcatchments Data"]["S2"], self._panel("C2") + "#node-outlet")
        self.assertNotIn("S3", links["Subcatchments Data"])

    def test_panel_links_use_two_queries(self):
        with CaptureQueriesContext(connection) as ctx:
            CalculationRetrievalService.panel_links(self.session)

        self.assertLessEqual(len(ctx.captured_queries), 2, "\n".join(q["sql"] for q in ctx.captured_queries))

    def test_panel_links_pick_the_lowest_pk_conduit_whatever_the_row_order(self):
        """ "First by id wins" must not depend on what order the database returns rows in."""
        session = make_session(self.user)
        make_node(session, node_name="M1")
        make_node(session, node_name="M2")
        make_node(session, node_name="M3")
        first = make_conduit(session, conduit_name="K-first", inlet_node="M1", outlet_node="M2", subcatchment="-")
        make_conduit(session, conduit_name="K-second", inlet_node="M1", outlet_node="M3", subcatchment="-")
        low_pk_sbc = make_subcatchment(session, subcatchment_name="T1", outlet="M1")
        make_subcatchment(session, subcatchment_name="T2", outlet="M1")

        # A prefetch that hands the rows back newest first: only a Python-side sort
        # by pk can still resolve the link to the lowest-pk conduit.
        reversed_session = CalculationSession.objects.prefetch_related(
            Prefetch("conduits", queryset=ConduitData.objects.order_by("-id")),
            Prefetch("subcatchments", queryset=SubcatchmentData.objects.order_by("-id")),
        ).get(pk=session.pk)

        with CaptureQueriesContext(connection) as ctx:
            links = CalculationRetrievalService.panel_links(reversed_session)

        expected = reverse("sa:conduit_detail", args=[session.id, first.conduit_name])
        self.assertEqual(links["Nodes Data"]["M1"], expected + "#node-inlet")
        self.assertEqual(links["Subcatchments Data"]["T1"], expected + f"#catchment-{low_pk_sbc.pk}")
        self.assertEqual(
            len(ctx.captured_queries),
            0,
            "panel_links must read the prefetch cache:\n" + "\n".join(q["sql"] for q in ctx.captured_queries),
        )

    def test_results_page_renders_the_links_and_plain_names(self):
        response = self.client.get(reverse("sa:analysis_results", args=[self.session.id]))
        html = response.content.decode()

        self.assertEqual(response.status_code, 200)
        # Panel links are now embedded in the JSON data payload, not as HTML anchors.
        self.assertIn(f"{self._panel('C1')}#node-inlet", html)
        self.assertIn(f"{self._panel('C2')}#node-outlet", html)
        self.assertIn(f"{self._panel('C1')}#catchment-{self.s1.pk}", html)
        self.assertNotIn("/node/", html)
        self.assertNotIn("/subcatchment/", html)
        # S3 and N9 appear in the data but without links.
        self.assertIn("S3", html)
        self.assertIn("N9", html)

    def test_fresh_analysis_page_links_into_the_panel_too(self):
        """``AnalysisView._render_success`` renders analysis.html with the same tables."""
        request = RequestFactory().get(reverse("sa:analysis"))
        request.user = self.user
        request.session = {}
        request._messages = FallbackStorage(request)

        response = AnalysisView()._render_success(request, self.session)
        html = response.content.decode()

        self.assertIn(f'href="{self._panel("C1")}#node-inlet"', html)
        self.assertIn(f'href="{self._panel("C1")}#catchment-{self.s1.pk}"', html)
        self.assertNotIn("/node/", html)

    def test_names_with_a_slash_route_everywhere(self):
        """A legal SWMM name may contain "/": the panel, its neighbours and the results page all render."""
        session = make_session(self.user)
        make_node(session, node_name="X1")
        make_node(session, node_name="X2")
        make_node(session, node_name="X3")
        make_conduit(session, conduit_name="A/B", inlet_node="X1", outlet_node="X2", subcatchment="-")
        make_conduit(session, conduit_name="C2", inlet_node="X2", outlet_node="X3", subcatchment="-")
        sbc = make_subcatchment(session, subcatchment_name="S/1", outlet="X1")

        links = CalculationRetrievalService.panel_links(session)
        slash_panel = reverse("sa:conduit_detail", args=[session.id, "A/B"])
        self.assertEqual(links["Subcatchments Data"]["S/1"], slash_panel + f"#catchment-{sbc.pk}")

        self.assertEqual(self.client.get(reverse("sa:analysis_results", args=[session.id])).status_code, 200)
        self.assertEqual(self.client.get(slash_panel).status_code, 200)
        neighbour = self.client.get(reverse("sa:conduit_detail", args=[session.id, "C2"]))
        self.assertEqual(neighbour.status_code, 200)
        self.assertContains(neighbour, f'href="{slash_panel}"')

    def test_old_element_routes_are_gone(self):
        for path in (
            f"/analysis/results/{self.session.id}/node/N1/",
            f"/analysis/results/{self.session.id}/subcatchment/S1/",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)


class ConduitPanelAccessTests(TestCase):
    """Error paths of the standalone conduit view and its prev/next navigation."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="access@example.com")
        cls.other = make_user(email="other@example.com")
        cls.session = make_session(cls.user, n_conduits=3, n_nodes=4, n_subcatchments=3)

    def setUp(self):
        self.client.force_login(self.user)

    def _url(self, name, session=None):
        return reverse("sa:conduit_detail", args=[(session or self.session).id, name])

    def test_missing_conduit_redirects_to_results_with_a_message(self):
        response = self.client.get(self._url("C-none"))

        self.assertRedirects(response, reverse("sa:analysis_results", args=[self.session.id]))
        texts = [m.message for m in get_messages(response.wsgi_request)]
        self.assertTrue(any("C-none" in text for text in texts), texts)

    def test_another_users_session_is_404(self):
        foreign = make_session(self.other, n_conduits=1, n_nodes=2)

        self.assertEqual(self.client.get(self._url("C1", foreign)).status_code, 404)

    def test_unfinished_session_redirects_to_analysis(self):
        pending = make_session(self.user, status="processing", n_conduits=1, n_nodes=2)

        self.assertRedirects(self.client.get(self._url("C1", pending)), reverse("sa:analysis"))

    def test_anonymous_is_sent_to_login(self):
        self.client.logout()
        response = self.client.get(self._url("C1"))

        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])

    def test_prev_next_for_first_middle_and_last_conduit(self):
        first = self.client.get(self._url("C1")).context["navigation"]
        middle = self.client.get(self._url("C2")).context["navigation"]
        last = self.client.get(self._url("C3")).context["navigation"]

        self.assertIsNone(first["previous"]["url"])
        self.assertEqual(first["next"]["url"], self._url("C2"))
        self.assertEqual(middle["previous"]["url"], self._url("C1"))
        self.assertEqual(middle["next"]["url"], self._url("C3"))
        self.assertEqual(last["previous"]["url"], self._url("C2"))
        self.assertIsNone(last["next"]["url"])
