"""The messages the views set say what happened and what to do, in plain sentences."""

from __future__ import annotations

import json
from unittest import mock

from django.contrib.messages import constants, get_messages
from django.test import TestCase
from django.urls import reverse

from sa.models import CalculationSession

from .factories import make_session, make_user


class ViewMessageTests(TestCase):
    def setUp(self) -> None:
        self.user = make_user()
        self.client.force_login(self.user)

    def message(self, response) -> tuple[int, str]:
        # Messages are only consumed when displayed, so the newest one is this response's.
        newest = list(get_messages(response.wsgi_request))[-1]
        return newest.level, newest.message

    def test_a_failed_analysis_says_it_has_no_results_and_where_to_look(self):
        session = make_session(self.user, status="failed", error_message="The model has no conduits.")
        for url in (
            reverse("sa:analysis_results", args=[session.id]),
            reverse("sa:conduit_detail", args=[session.id, "C1"]),
        ):
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertRedirects(response, reverse("sa:analysis"), fetch_redirect_response=False)
                self.assertEqual(
                    self.message(response),
                    (constants.WARNING, "This analysis failed, so it has no results. History shows why."),
                )

    def test_a_running_analysis_says_when_to_come_back(self):
        session = make_session(self.user, status="processing")

        response = self.client.get(reverse("sa:analysis_results", args=[session.id]))

        self.assertEqual(
            self.message(response),
            (constants.INFO, "This analysis is still running. Open it from History when it has finished."),
        )

    def test_an_unknown_conduit_is_named(self):
        session = make_session(self.user, n_conduits=1)

        response = self.client.get(reverse("sa:conduit_detail", args=[session.id, "NOPE"]))

        self.assertRedirects(response, reverse("sa:analysis_results", args=[session.id]), fetch_redirect_response=False)
        self.assertEqual(self.message(response), (constants.ERROR, "This analysis has no conduit named “NOPE”."))

    def test_delete_failures_say_to_try_again_and_keep_the_json_contract(self):
        session = make_session(self.user)
        with (
            mock.patch("sa.views.SessionDeletionService.delete", side_effect=RuntimeError("locked")),
            self.assertLogs("sa.views", level="ERROR"),
        ):
            single = self.client.post(reverse("sa:delete_session", args=[session.id]))
            bulk = self.client.post(
                reverse("sa:bulk_delete_sessions"), json.dumps({"session_ids": [session.id]}), content_type="application/json"
            )

        self.assertEqual(single.json(), {"success": False, "message": "The session could not be deleted. Try again."})
        self.assertEqual(bulk.json(), {"success": False, "message": "The sessions could not be deleted. Try again."})

    def test_a_bulk_delete_of_sessions_that_are_gone_says_to_reload(self):
        response = self.client.post(
            reverse("sa:bulk_delete_sessions"), json.dumps({"session_ids": [987654]}), content_type="application/json"
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json(), {"success": False, "message": "None of the selected sessions exist any more. Reload the page."}
        )

    def test_successful_deletes_keep_the_fields_the_history_script_reads(self):
        first, second = make_session(self.user), make_session(self.user)

        single = self.client.post(reverse("sa:delete_session", args=[first.id])).json()
        bulk = self.client.post(
            reverse("sa:bulk_delete_sessions"), json.dumps({"session_ids": [second.id]}), content_type="application/json"
        ).json()

        self.assertIs(single["success"], True)
        self.assertEqual((bulk["success"], bulk["deleted_count"]), (True, 1))
        self.assertFalse(CalculationSession.objects.exists())
