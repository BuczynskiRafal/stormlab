"""Tests for the template tags of ``sa``: ``{% querystring %}`` and ``{% versioned_static %}``."""

from __future__ import annotations

import os

from django.contrib.staticfiles import finders
from django.template import Context, Template
from django.test import RequestFactory, SimpleTestCase


def render(query_string: str, arguments: str = "", **context) -> str:
    """Render ``{% querystring <arguments> %}`` against a request carrying ``query_string``."""
    request = RequestFactory().get(f"/history/?{query_string}" if query_string else "/history/")
    template = Template("{% load sa_query %}{% querystring " + arguments + " %}")
    return template.render(Context({"request": request, **context}))


class QuerystringTagTests(SimpleTestCase):
    def test_an_override_replaces_the_existing_value(self):
        self.assertEqual(render("page=2", "page=3"), "?page=3")

    def test_an_override_keeps_the_other_parameters(self):
        rendered = render("q=alpha&status=failed&per_page=20", "page=3")

        for expected in ("q=alpha", "status=failed", "per_page=20", "page=3"):
            self.assertIn(expected, rendered)

    def test_a_new_key_is_added_to_an_empty_query_string(self):
        self.assertEqual(render("", "page=2"), "?page=2")

    def test_empty_incoming_parameters_are_dropped(self):
        # A native GET form submits every control, blanks included; the links must not
        # carry "?q=&status=" around.
        rendered = render("q=&status=&sort=newest", "page=2")

        self.assertNotIn("status=", rendered)
        self.assertNotIn("q=", rendered)
        self.assertIn("sort=newest", rendered)
        self.assertIn("page=2", rendered)

    def test_an_override_set_to_none_removes_the_key(self):
        rendered = render("page=2&q=alpha", "page=nothing", nothing=None)

        self.assertEqual(rendered, "?q=alpha")

    def test_special_characters_are_percent_encoded(self):
        rendered = render("", "q=value", value="a b&c=d")

        self.assertEqual(rendered, "?q=a+b%26c%3Dd")

    def test_the_separator_is_escaped_so_the_output_is_a_valid_attribute(self):
        # simple_tag output goes through conditional_escape; browsers decode &amp;.
        self.assertIn("&amp;", render("q=alpha", "page=2"))

    def test_a_query_string_that_ends_up_empty_renders_nothing(self):
        self.assertEqual(render("q=", ""), "")

    def test_a_request_free_context_renders_nothing(self):
        rendered = Template("{% load sa_query %}{% querystring %}").render(Context({}))

        self.assertEqual(rendered, "")


class DictGetFilterTests(SimpleTestCase):
    """``dict_get`` chains through nested dicts and never raises on missing keys."""

    def test_returns_the_value_for_a_present_key(self):
        from sa.templatetags.sa_query import dict_get

        self.assertEqual(dict_get({"a": {"b": 1}}, "a"), {"b": 1})

    def test_returns_none_for_a_missing_key_or_a_non_dict(self):
        from sa.templatetags.sa_query import dict_get

        self.assertIsNone(dict_get({"a": 1}, "zzz"))
        self.assertIsNone(dict_get(None, "a"))
        self.assertIsNone(dict_get(dict_get({"a": 1}, "zzz"), "b"))


class VersionedStaticTagTests(SimpleTestCase):
    @staticmethod
    def _render(path: str) -> str:
        return Template("{% load sa_static %}{% versioned_static path %}").render(Context({"path": path}))

    def test_a_known_file_gets_its_modification_time_as_the_version(self):
        source = finders.find("css/app.css")
        self.assertIsNotNone(source)

        self.assertEqual(self._render("css/app.css"), f"/static/css/app.css?v={int(os.path.getmtime(source))}")

    def test_an_unknown_file_keeps_the_bare_static_url(self):
        self.assertEqual(self._render("css/no-such-file.css"), "/static/css/no-such-file.css")
