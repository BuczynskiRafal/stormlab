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


class LabelFilterTests(SimpleTestCase):
    """``{% load sa_labels %}``: display labels, timestamps and file names."""

    @staticmethod
    def render(source: str, **context) -> str:
        return Template("{% load sa_labels %}" + source).render(Context(context))

    def test_codes_become_their_labels(self):
        self.assertEqual(self.render("{{ v|action_label }}", v="slope_increase"), "Increase slope")
        self.assertEqual(self.render("{{ v|category_label }}", v="urban_weakly_impervious"), "Urban, weakly impervious")
        self.assertEqual(self.render("{{ v|category_label }}", v="-"), "—")

    def test_timestamps_are_written_in_utc(self):
        import datetime

        moment = datetime.datetime(2026, 11, 15, 5, 43, tzinfo=datetime.timezone(datetime.timedelta(hours=1)))
        self.assertEqual(self.render("{{ v|utc_datetime }}", v=moment), "15 Nov 2026, 04:43 UTC")
        early = datetime.datetime(2026, 3, 5, 9, 7, tzinfo=datetime.timezone.utc)
        self.assertEqual(self.render("{{ v|utc_datetime }}", v=early), "5 Mar 2026, 09:07 UTC")
        self.assertEqual(self.render("{{ v|utc_datetime }}", v="yesterday"), "")

    def test_a_layout_can_set_the_date_and_the_time_apart_in_the_same_format(self):
        import datetime

        early = datetime.datetime(2026, 3, 5, 9, 7, tzinfo=datetime.timezone.utc)
        self.assertEqual(self.render('{{ v|utc_datetime:"date" }}', v=early), "5 Mar 2026")
        self.assertEqual(self.render('{{ v|utc_datetime:"time" }}', v=early), "09:07 UTC")
        self.assertEqual(self.render('{{ v|utc_datetime:"date" }}', v=None), "")

    def test_zones_are_named_with_their_numeral_and_depth(self):
        from types import SimpleNamespace

        depths = {1: 0.8, 2: 1.0, 3: 1.2, 4: 1.4}
        for zone, label in ((1, "Zone I (0.8 m)"), (2, "Zone II (1.0 m)"), (3, "Zone III (1.2 m)"), (4, "Zone IV (1.4 m)")):
            model = SimpleNamespace(zone=zone, get_frost_zone_value=lambda zone=zone: depths[zone])
            with self.subTest(zone=zone):
                self.assertEqual(self.render("{{ v|zone_label }}", v=model), label)
        self.assertEqual(self.render("{{ v|zone_label }}", v=None), "")

    def test_file_label_is_the_bare_name(self):
        self.assertEqual(self.render("{{ v|file_label }}", v="uploads/wroclaw_rakowiec.inp"), "wroclaw_rakowiec.inp")
        self.assertEqual(self.render("{{ v|file_label }}", v="C:\\models\\net.inp"), "net.inp")
        self.assertEqual(self.render("{{ v|file_label }}", v=""), "model.inp")
