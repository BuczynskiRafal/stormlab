"""Deployment plumbing: environment parsing helpers and the health endpoint."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from django.core.exceptions import ImproperlyConfigured
from django.db import DatabaseError
from django.test import SimpleTestCase, TestCase, override_settings

from storm_analysis.settings import envvars

GOOD_KEY = "k3Y-" + "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ"


class EnvHelperTests(SimpleTestCase):
    def test_blank_values_count_as_unset(self) -> None:
        env = {"X": "  "}
        self.assertTrue(envvars.env_bool("X", True, env))
        self.assertEqual(envvars.env_int("X", 7, env), 7)
        self.assertEqual(envvars.env_path("X", "/default", env), Path("/default"))
        self.assertEqual(envvars.env_list("X", env), [])

    def test_bool_accepts_common_spellings_and_rejects_others(self) -> None:
        for raw in ("1", "true", "Yes", " ON "):
            self.assertTrue(envvars.env_bool("X", False, {"X": raw}))
        for raw in ("0", "false", "No", "off"):
            self.assertFalse(envvars.env_bool("X", True, {"X": raw}))
        with self.assertRaisesMessage(ImproperlyConfigured, "X='maybe'"):
            envvars.env_bool("X", True, {"X": "maybe"})

    def test_int_parses_and_enforces_the_minimum(self) -> None:
        self.assertEqual(envvars.env_int("X", 0, {"X": " 42 "}), 42)
        with self.assertRaises(ImproperlyConfigured):
            envvars.env_int("X", 0, {"X": "5MB"})
        with self.assertRaises(ImproperlyConfigured):
            envvars.env_int("X", 1, {"X": "0"}, minimum=1)

    def test_path_overrides_the_default(self) -> None:
        self.assertEqual(envvars.env_path("X", "/default", {"X": "/data/db.sqlite3"}), Path("/data/db.sqlite3"))

    def test_list_strips_items_and_drops_empty_ones(self) -> None:
        self.assertEqual(envvars.env_list("X", {"X": "a.com, b.com,,"}), ["a.com", "b.com"])

    def test_required_list_rejects_missing_or_empty_values(self) -> None:
        for env in ({}, {"X": ""}, {"X": " , "}):
            with self.assertRaises(ImproperlyConfigured):
                envvars.required_list("X", env)


class SecretKeyTests(SimpleTestCase):
    def test_accepts_a_long_random_key(self) -> None:
        self.assertEqual(envvars.secret_key({"SECRET_KEY": GOOD_KEY}), GOOD_KEY)

    def test_rejects_missing_short_placeholder_and_insecure_keys(self) -> None:
        rejected = [
            {},
            {"SECRET_KEY": "  "},
            {"SECRET_KEY": "short-key"},
            {"SECRET_KEY": "a" * 60},
            {"SECRET_KEY": "your-secret-key-here" + GOOD_KEY},
            {"SECRET_KEY": "django-insecure-" + GOOD_KEY},
        ]
        for env in rejected:
            with self.subTest(env=env), self.assertRaises(ImproperlyConfigured):
                envvars.secret_key(env)


class HealthEndpointTests(TestCase):
    def test_reports_ok_when_the_database_answers(self) -> None:
        response = self.client.get("/healthz/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertIn("no-cache", response["Cache-Control"])

    def test_reports_503_when_the_database_fails(self) -> None:
        with mock.patch("storm_analysis.health.connection.cursor", side_effect=DatabaseError("down")):
            response = self.client.get("/healthz/")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "unavailable"})

    def test_rejects_writes(self) -> None:
        self.assertEqual(self.client.post("/healthz/").status_code, 405)

    @override_settings(SECURE_SSL_REDIRECT=True)
    def test_is_exempt_from_the_https_redirect(self) -> None:
        self.assertEqual(self.client.get("/healthz/").status_code, 200)
        self.assertEqual(self.client.get("/").status_code, 301)
