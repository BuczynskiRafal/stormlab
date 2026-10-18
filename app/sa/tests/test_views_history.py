"""Tests for the history list and the single-session delete endpoint.

They cover the contracts the page relies on: every control (search, status, sort,
page size) travels as a GET parameter, the pagination links carry the whole set
forward, the row counts come from subqueries rather than from prefetched relations,
and the delete endpoint answers JSON only for a session the requesting user owns.
"""

from __future__ import annotations

import json
import re

from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.db import connection
from django.urls import reverse

from sa.models import CalculationSession

from .factories import make_session, make_swmm_model, make_user

ROW_MARKER = '<tr data-session-id="'
PAGE_LINK_RE = re.compile(r'class="page-link" href="([^"]*)"')
SESSION_ID_RE = re.compile(r'<tr data-session-id="(\d+)"')


def count_rows(response) -> int:
    """Number of session rows in the rendered table (the checkbox carries the id too)."""
    return response.content.decode().count(ROW_MARKER)


def rendered_session_ids(response) -> list[int]:
    """Session ids in the order the table renders them."""
    return [int(value) for value in SESSION_ID_RE.findall(response.content.decode())]


def page_link_hrefs(response) -> list[str]:
    """The href of every pagination link (the disabled ones are <span>, not <a>)."""
    return PAGE_LINK_RE.findall(response.content.decode())


class HistoryPaginationTests(TestCase):
    """``per_page`` reaches ``get_paginate_by`` through the query string."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="history@example.com")
        for _ in range(25):
            make_session(cls.user)

    def setUp(self):
        self.client.force_login(self.user)

    def test_defaults_to_ten_rows_per_page(self):
        response = self.client.get(reverse("sa:history"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(count_rows(response), 10)

    def test_per_page_20_renders_twenty_rows(self):
        response = self.client.get(reverse("sa:history"), {"per_page": 20})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(count_rows(response), 20)
        self.assertEqual(len(response.context["sessions"]), 20)

    def test_per_page_outside_the_offered_options_falls_back_to_ten(self):
        response = self.client.get(reverse("sa:history"), {"per_page": 7})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(count_rows(response), 10)
        self.assertEqual(response.context["current_per_page"], 10)

    def test_pagination_links_carry_the_current_per_page(self):
        response = self.client.get(reverse("sa:history"), {"per_page": 20})

        # {% querystring %} preserves the order of the incoming query string and its
        # output is autoescaped, so the link is asserted parameter by parameter.
        href = page_link_hrefs(response)[0]
        self.assertIn("page=2", href)
        self.assertIn("per_page=20", href)

    def test_per_page_select_is_a_named_control_of_a_get_form(self):
        response = self.client.get(reverse("sa:history"), {"per_page": 20})

        self.assertContains(response, 'id="historyFilters"')
        self.assertContains(response, 'name="per_page"')
        self.assertContains(response, '<option value="20" selected>')


class HistoryControlsWithoutPaginationTests(TestCase):
    """The per-page select must survive when a single page holds every session."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="short-history@example.com")
        for _ in range(3):
            make_session(cls.user)

    def setUp(self):
        self.client.force_login(self.user)

    def test_per_page_select_is_rendered_although_there_is_one_page(self):
        response = self.client.get(reverse("sa:history"))

        self.assertFalse(response.context["is_paginated"])
        self.assertContains(response, 'id="perPageSelect"')
        self.assertContains(response, 'id="historyFilters"')


class DeleteSessionViewTests(TestCase):
    """``POST /session/<id>/delete/`` - the endpoint behind the confirmation modal."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="owner@example.com")
        cls.other_user = make_user(email="stranger@example.com")

    def setUp(self):
        self.session = make_session(self.user, n_conduits=1, n_nodes=2, n_subcatchments=1)
        self.client.force_login(self.user)

    def test_owner_deletes_own_session_and_gets_json_success(self):
        response = self.client.post(reverse("sa:delete_session", args=[self.session.id]))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(json.loads(response.content)["success"])
        self.assertFalse(CalculationSession.objects.filter(id=self.session.id).exists())

    def test_deleted_session_disappears_from_the_history_table(self):
        self.client.post(reverse("sa:delete_session", args=[self.session.id]))

        response = self.client.get(reverse("sa:history"))

        self.assertNotContains(response, f'{ROW_MARKER}{self.session.id}"')
        self.assertEqual(count_rows(response), 0)

    def test_another_users_session_is_not_found(self):
        foreign_session = make_session(self.other_user)

        response = self.client.post(reverse("sa:delete_session", args=[foreign_session.id]))

        self.assertEqual(response.status_code, 404)
        self.assertTrue(CalculationSession.objects.filter(id=foreign_session.id).exists())

    def test_anonymous_request_is_redirected_to_the_login_form(self):
        self.client.logout()

        response = self.client.post(reverse("sa:delete_session", args=[self.session.id]))

        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])
        self.assertTrue(CalculationSession.objects.filter(id=self.session.id).exists())


class HistorySearchAndFilterTests(TestCase):
    """``q``, ``status`` and ``sort`` are applied by the queryset, not by JavaScript."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="filters@example.com")
        cls.alpha = make_session(cls.user, swmm_model=make_swmm_model(cls.user, file="user_models/alpha.inp"))
        cls.beta = make_session(cls.user, swmm_model=make_swmm_model(cls.user, file="user_models/beta.inp"))
        cls.broken = make_session(
            cls.user,
            status="failed",
            error_message="SWMM run failed",
            swmm_model=make_swmm_model(cls.user, file="user_models/gamma.inp"),
        )

    def setUp(self):
        self.client.force_login(self.user)

    def test_q_keeps_only_sessions_whose_model_file_matches(self):
        response = self.client.get(reverse("sa:history"), {"q": "alpha"})

        self.assertEqual(rendered_session_ids(response), [self.alpha.id])

    def test_q_is_case_insensitive_and_matches_a_fragment(self):
        response = self.client.get(reverse("sa:history"), {"q": "BET"})

        self.assertEqual(rendered_session_ids(response), [self.beta.id])

    def test_q_without_a_match_renders_the_filtered_empty_state(self):
        response = self.client.get(reverse("sa:history"), {"q": "nothing-like-this"})

        self.assertEqual(count_rows(response), 0)
        self.assertTrue(response.context["filters_active"])
        self.assertContains(response, "No sessions match these filters")
        self.assertContains(response, "Clear filters")
        # The controls stay on the page, otherwise the filter cannot be undone.
        self.assertContains(response, 'id="historyFilters"')

    def test_status_failed_keeps_only_failed_sessions(self):
        response = self.client.get(reverse("sa:history"), {"status": "failed"})

        self.assertEqual(rendered_session_ids(response), [self.broken.id])
        self.assertEqual(response.context["current_status"], "failed")

    def test_unknown_status_is_ignored(self):
        with self.assertLogs("sa.views", level="WARNING") as logs:
            response = self.client.get(reverse("sa:history"), {"status": "exploded"})

        self.assertEqual(count_rows(response), 3)
        self.assertEqual(response.context["current_status"], "")
        self.assertFalse(response.context["filters_active"])
        self.assertIn("Invalid status value", logs.output[0])

    def test_sort_oldest_reverses_the_default_order(self):
        newest_first = self.client.get(reverse("sa:history"))
        oldest_first = self.client.get(reverse("sa:history"), {"sort": "oldest"})

        self.assertEqual(rendered_session_ids(oldest_first), list(reversed(rendered_session_ids(newest_first))))

    def test_sort_name_orders_by_the_model_file(self):
        response = self.client.get(reverse("sa:history"), {"sort": "name"})

        self.assertEqual(rendered_session_ids(response), [self.alpha.id, self.beta.id, self.broken.id])

    def test_unknown_sort_falls_back_to_newest(self):
        with self.assertLogs("sa.views", level="WARNING") as logs:
            response = self.client.get(reverse("sa:history"), {"sort": "xyz"})

        self.assertEqual(rendered_session_ids(response), rendered_session_ids(self.client.get(reverse("sa:history"))))
        self.assertEqual(response.context["current_sort"], "newest")
        self.assertIn("Invalid sort value", logs.output[0])

    def test_empty_status_and_sort_are_not_reported_as_invalid(self):
        # A native GET form submits every control, so blanks arrive on every request.
        with self.assertNoLogs("sa.views", level="WARNING"):
            response = self.client.get(reverse("sa:history"), {"q": "", "status": "", "sort": ""})

        self.assertEqual(count_rows(response), 3)
        self.assertFalse(response.context["filters_active"])

    def test_controls_echo_the_current_query_string(self):
        response = self.client.get(reverse("sa:history"), {"q": "abc", "status": "failed", "per_page": 20})

        self.assertContains(response, 'value="abc"')
        self.assertContains(response, '<option value="failed" selected>')
        self.assertContains(response, '<option value="20" selected>')

    def test_a_filtered_page_never_shows_the_no_sessions_yet_copy(self):
        response = self.client.get(reverse("sa:history"), {"q": "nothing-like-this"})

        self.assertNotContains(response, "You haven&#x27;t performed any analyses yet.")


class HistoryEmptyAccountTests(TestCase):
    """An account without a single session gets the other empty state."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="empty@example.com")

    def setUp(self):
        self.client.force_login(self.user)

    def test_no_sessions_yet_state_invites_a_first_analysis(self):
        response = self.client.get(reverse("sa:history"))

        self.assertFalse(response.context["filters_active"])
        self.assertContains(response, "No analysis history")
        self.assertContains(response, "Start your first analysis")
        self.assertNotContains(response, "No sessions match these filters")
        # Nothing to filter: the bar would only be noise.
        self.assertNotContains(response, 'id="historyFilters"')


class HistoryPaginationQueryStringTests(TestCase):
    """Paging must not drop the filters the user is looking at."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="paging-filters@example.com")
        for index in range(25):
            make_session(cls.user, swmm_model=make_swmm_model(cls.user, file=f"user_models/model{index:02d}.inp"))

    def setUp(self):
        self.client.force_login(self.user)

    def test_pagination_links_keep_q_status_sort_and_per_page(self):
        response = self.client.get(
            reverse("sa:history"),
            {"q": "model", "status": "completed", "sort": "oldest", "per_page": 20},
        )

        hrefs = page_link_hrefs(response)
        self.assertTrue(hrefs)
        for href in hrefs:
            self.assertIn("q=model", href)
            self.assertIn("status=completed", href)
            self.assertIn("sort=oldest", href)
            self.assertIn("per_page=20", href)
            self.assertIn("page=", href)

    def test_a_filter_narrows_the_list_and_the_range_line_follows(self):
        response = self.client.get(reverse("sa:history"), {"q": "model1"})

        # model10..model19 - ten of the twenty-five sessions.
        self.assertEqual(count_rows(response), 10)
        self.assertContains(response, "of 10 sessions")

    def test_the_second_page_of_a_filtered_list_stays_filtered(self):
        first = self.client.get(reverse("sa:history"), {"q": "model", "per_page": 10, "page": 1})
        second = self.client.get(reverse("sa:history"), {"q": "model", "per_page": 10, "page": 2})

        expected = list(
            CalculationSession.objects.filter(user=self.user, swmm_model__file__icontains="model")
            .order_by("-created_at", "-pk")
            .values_list("id", flat=True)
        )
        self.assertEqual(rendered_session_ids(first), expected[:10])
        self.assertEqual(rendered_session_ids(second), expected[10:20])
        self.assertContains(second, "Showing 11-20 of 25 sessions")
        for href in page_link_hrefs(second):
            self.assertIn("q=model", href)


class HistoryRowCountsTests(TestCase):
    """The Conduits/Nodes/Subcatchments columns come from annotated subqueries."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="counts@example.com")
        cls.populated = make_session(cls.user, n_conduits=3, n_nodes=4, n_subcatchments=2)
        cls.empty = make_session(cls.user)

    def setUp(self):
        self.client.force_login(self.user)

    def test_counts_are_annotated_on_every_session(self):
        response = self.client.get(reverse("sa:history"))
        by_id = {session.id: session for session in response.context["sessions"]}

        self.assertEqual(
            (by_id[self.populated.id].n_conduits, by_id[self.populated.id].n_nodes, by_id[self.populated.id].n_subcatchments),
            (3, 4, 2),
        )

    def test_a_session_without_objects_renders_a_zero_not_a_blank(self):
        response = self.client.get(reverse("sa:history"))
        by_id = {session.id: session for session in response.context["sessions"]}

        self.assertEqual(
            (by_id[self.empty.id].n_conduits, by_id[self.empty.id].n_nodes, by_id[self.empty.id].n_subcatchments),
            (0, 0, 0),
        )
        self.assertContains(response, '<span class="count-chip">0</span>', count=3)

    def test_the_status_pill_is_the_only_badge_in_a_row(self):
        # main.js reads row.querySelector('.badge') as the status - the counts must
        # not be badges, whatever column order the table ends up with.
        response = self.client.get(reverse("sa:history"))

        row = response.content.decode().split(ROW_MARKER)[1]
        self.assertEqual(row.count('class="badge'), 1)


class HistoryQueryBudgetTests(TestCase):
    """Counting through subqueries must not reintroduce a per-row query."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="budget@example.com")
        for _ in range(30):
            make_session(cls.user, n_conduits=2, n_nodes=3, n_subcatchments=1)

    def setUp(self):
        self.client.force_login(self.user)

    def test_the_page_costs_at_most_five_queries(self):
        with CaptureQueriesContext(connection) as captured:
            response = self.client.get(reverse("sa:history"), {"per_page": 20})

        self.assertEqual(count_rows(response), 20)
        self.assertLessEqual(
            len(captured),
            5,
            "queries: " + "\n".join(query["sql"] for query in captured),
        )


class HistoryContextTests(TestCase):
    """The statistics tiles and the "Latest Analysis Session" card are gone."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="context@example.com")
        make_session(cls.user, n_conduits=1)

    def setUp(self):
        self.client.force_login(self.user)

    def test_stats_and_latest_session_are_not_in_the_context(self):
        response = self.client.get(reverse("sa:history"))

        self.assertNotIn("stats", response.context)
        self.assertNotIn("latest_session", response.context)

    def test_the_removed_cards_are_not_rendered(self):
        response = self.client.get(reverse("sa:history"))

        self.assertNotContains(response, "Latest Analysis Session")
        self.assertNotContains(response, "Total Sessions")
        self.assertNotContains(response, "Total Conduits Analyzed")


class HistorySortStabilityTests(TestCase):
    """Paging a non-unique sort key must not repeat or lose a row."""

    @classmethod
    def setUpTestData(cls):
        cls.user = make_user(email="stable-sort@example.com")
        # Every session shares one file name, so "name" alone cannot order them.
        model = make_swmm_model(cls.user, file="user_models/same.inp")
        for _ in range(12):
            make_session(cls.user, swmm_model=model)

    def setUp(self):
        self.client.force_login(self.user)

    def test_sorting_by_name_pages_through_every_session_exactly_once(self):
        pages = [self.client.get(reverse("sa:history"), {"sort": "name", "per_page": 10, "page": page}) for page in (1, 2)]
        seen = [session_id for page in pages for session_id in rendered_session_ids(page)]

        self.assertEqual(len(seen), 12)
        self.assertEqual(sorted(seen), sorted(CalculationSession.objects.filter(user=self.user).values_list("id", flat=True)))
