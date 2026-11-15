"""Uploads shaped like the models designers actually have, run end to end through ``POST /analysis/``.

Each test edits a copy of the demo network the way a real file differs from it and
checks what the user gets: the results page, or a plain sentence above the form.
"""

from __future__ import annotations

import re
from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from sa.core.constants import SUBCATCHMENT_CATEGORIES
from sa.models import CalculationSession

from .factories import make_user

DEMO_NETWORK = Path(__file__).resolve().parents[3] / "examples" / "demo_network_coordinates.inp"


def demo_text() -> str:
    return DEMO_NETWORK.read_text(encoding="utf-8")


def without_section(text: str, name: str) -> str:
    """``text`` without the section ``[name]``, its header included."""
    pattern = re.compile(r"^\[" + re.escape(name) + r"\][^\[]*", flags=re.MULTILINE)
    assert pattern.search(text), name
    return pattern.sub("", text)


class RealWorldUploadTestCase(TestCase):
    def setUp(self) -> None:
        self.user = make_user()
        self.client.force_login(self.user)

    def upload(self, content: str | bytes, name: str = "network.inp"):
        payload = content.encode("utf-8") if isinstance(content, str) else content
        upload = SimpleUploadedFile(name, payload, content_type="application/octet-stream")
        return self.client.post(reverse("sa:analysis"), {"file": upload, "zone": 1})

    def assert_completed(self, response) -> CalculationSession:
        session = CalculationSession.objects.get(user=self.user)
        self.assertEqual(session.status, "completed", session.error_message)
        self.assertRedirects(response, reverse("sa:analysis_results", args=[session.id]), fetch_redirect_response=False)
        return session


class ModelWithoutTagsTests(RealWorldUploadTestCase):
    def test_model_without_a_tags_section_is_analysed_and_classified(self):
        response = self.upload(without_section(demo_text(), "TAGS"))

        session = self.assert_completed(response)
        categories = set(session.subcatchments.values_list("category", flat=True))
        self.assertEqual(session.subcatchments.count(), 7)
        self.assertTrue(categories <= set(SUBCATCHMENT_CATEGORIES), categories)
