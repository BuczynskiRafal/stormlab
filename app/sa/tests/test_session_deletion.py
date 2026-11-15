"""Deleting sessions takes the uploaded-model rows nothing refers to any more with them."""

from __future__ import annotations

import json
from importlib import import_module

from django.apps import apps
from django.test import TestCase
from django.urls import reverse

from sa.models import CalculationSession, SWMMModel

from .factories import make_session, make_swmm_model, make_user


class OrphanModelTests(TestCase):
    def setUp(self) -> None:
        self.user = make_user()
        self.client.force_login(self.user)
        self.model = make_swmm_model(self.user)

    def _bulk(self, ids):
        return self.client.post(
            reverse("sa:bulk_delete_sessions"), json.dumps({"session_ids": ids}), content_type="application/json"
        )

    def test_deleting_the_last_session_of_a_model_deletes_the_model(self):
        first = make_session(self.user, swmm_model=self.model)
        second = make_session(self.user, swmm_model=self.model)

        self.client.post(reverse("sa:delete_session", args=[first.id]))
        self.assertTrue(SWMMModel.objects.filter(pk=self.model.pk).exists(), "still used by the second session")

        self.client.post(reverse("sa:delete_session", args=[second.id]))
        self.assertFalse(SWMMModel.objects.filter(pk=self.model.pk).exists())

    def test_bulk_delete_removes_only_the_models_left_without_a_session(self):
        shared = make_swmm_model(self.user)
        doomed = [make_session(self.user, swmm_model=self.model), make_session(self.user, swmm_model=shared)]
        kept = make_session(self.user, swmm_model=shared)

        response = self._bulk([session.id for session in doomed])

        self.assertEqual(response.json()["deleted_count"], 2)
        self.assertFalse(SWMMModel.objects.filter(pk=self.model.pk).exists())
        self.assertTrue(SWMMModel.objects.filter(pk=shared.pk).exists())
        self.assertTrue(CalculationSession.objects.filter(pk=kept.pk).exists())

    def test_another_users_session_and_model_are_left_alone(self):
        stranger = make_user(email="stranger@example.com")
        foreign = make_session(stranger)

        response = self._bulk([foreign.id])

        self.assertEqual(response.status_code, 404)
        self.assertTrue(CalculationSession.objects.filter(pk=foreign.pk).exists())
        self.assertTrue(SWMMModel.objects.filter(pk=foreign.swmm_model_id).exists())

    def test_the_data_migration_deletes_existing_orphans_only(self):
        used = make_session(self.user).swmm_model
        orphan = make_swmm_model(self.user)

        migration = import_module("sa.migrations.0012_delete_orphan_swmm_models")
        migration.delete_orphan_swmm_models(apps, None)

        self.assertFalse(SWMMModel.objects.filter(pk__in=[orphan.pk, self.model.pk]).exists())
        self.assertTrue(SWMMModel.objects.filter(pk=used.pk).exists())


class BulkDeleteValidationTests(TestCase):
    def setUp(self) -> None:
        self.user = make_user()
        self.client.force_login(self.user)
        self.session = make_session(self.user)

    def _post(self, body: str | bytes):
        return self.client.post(reverse("sa:bulk_delete_sessions"), body, content_type="application/json")

    def test_malformed_bodies_are_refused_with_400_and_delete_nothing(self):
        for body in (
            "[1,2]",
            '{"session_ids":"abc"}',
            '["x"]',
            "5",
            "[{}]",
            '{"session_ids":[{}]}',
            '{"session_ids":["1"]}',
            '{"session_ids":[0]}',
            '{"session_ids":[-3]}',
            '{"session_ids":[1.5]}',
            '{"session_ids":[true]}',
            '{"session_ids":null}',
            "{}",
            "not json",
            bytes([0xFF, 0xFE]),
        ):
            with self.subTest(body=body):
                response = self._post(body)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json(), {"success": False, "message": "Invalid request data."})
        self.assertTrue(CalculationSession.objects.filter(pk=self.session.pk).exists())

    def test_an_empty_or_too_long_list_is_refused_with_its_reason(self):
        self.assertEqual(self._post('{"session_ids":[]}').json()["message"], "No sessions selected.")
        response = self._post(json.dumps({"session_ids": list(range(1, 502))}))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["message"], "Select at most 500 sessions at a time.")

    def test_a_valid_list_still_deletes_only_the_users_sessions(self):
        response = self._post(json.dumps({"session_ids": [self.session.id, 999999]}))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted_count"], 1)
