"""The ``REGISTRATION_OPEN`` switch: open by default, 404 and no sign-up links when closed."""

from __future__ import annotations

from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import User

SIGNUP_DATA = {"email": "new@example.com", "password1": "a-Long-pass-9271", "password2": "a-Long-pass-9271"}


class RegistrationOpenTests(TestCase):
    def test_register_page_and_links_are_shown_by_default(self) -> None:
        register_url = reverse("accounts:register")

        self.assertEqual(self.client.get(register_url).status_code, 200)
        self.assertContains(self.client.get(reverse("accounts:login")), f'href="{register_url}"', count=2)

    def test_post_creates_and_logs_in_a_user_by_default(self) -> None:
        response = self.client.post(reverse("accounts:register"), SIGNUP_DATA)

        self.assertRedirects(response, "/", fetch_redirect_response=False)
        self.assertTrue(User.objects.filter(email=SIGNUP_DATA["email"]).exists())


@override_settings(REGISTRATION_OPEN=False)
class RegistrationClosedTests(TestCase):
    def test_register_page_answers_404(self) -> None:
        self.assertEqual(self.client.get(reverse("accounts:register")).status_code, 404)

    def test_post_creates_no_user(self) -> None:
        response = self.client.post(reverse("accounts:register"), SIGNUP_DATA)

        self.assertEqual(response.status_code, 404)
        self.assertFalse(User.objects.filter(email=SIGNUP_DATA["email"]).exists())

    def test_sign_up_links_are_hidden(self) -> None:
        response = self.client.get(reverse("accounts:login"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, reverse("accounts:register"))
