"""Tests for the ``accounts`` app.

The avatar tests pin the fallback that keeps the site header from rendering a broken
image: the ``avatar`` field default names a file that lives in static files, not under
``MEDIA_ROOT``. The page tests pin the auth card of Etap 7: the class the stylesheet
hooks on, the water-coloured submit that replaced the green one, and the fact that a
rejected login comes back with a rendered error block instead of an empty form.

They deliberately assert nothing about where a *successful* login lands: that target
comes from ``LOGIN_REDIRECT_URL``, not from this app's templates.
"""

from __future__ import annotations

from tempfile import TemporaryDirectory

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import DEFAULT_AVATAR_STATIC_PATH, User

DEFAULT_AVATAR_URL = f"/static/{DEFAULT_AVATAR_STATIC_PATH}"


class GetAvatarUrlTests(TestCase):
    def test_user_without_upload_gets_the_static_placeholder(self) -> None:
        user = User.objects.create_user(email="no-avatar@example.com", password="avatar-pass-123")

        self.assertEqual(user.get_avatar_url(), DEFAULT_AVATAR_URL)

    def test_cleared_avatar_gets_the_static_placeholder(self) -> None:
        user = User.objects.create_user(email="cleared@example.com", password="avatar-pass-123", avatar="")

        self.assertEqual(user.get_avatar_url(), DEFAULT_AVATAR_URL)

    def test_uploaded_avatar_missing_from_storage_gets_the_static_placeholder(self) -> None:
        user = User.objects.create_user(email="gone@example.com", password="avatar-pass-123", avatar="avatars/deleted.jpg")

        self.assertEqual(user.get_avatar_url(), DEFAULT_AVATAR_URL)

    def test_avatar_name_escaping_the_media_root_gets_the_static_placeholder(self) -> None:
        # A relative name with ".." reaches FileSystemStorage.exists(), which raises
        # SuspiciousFileOperation; the header must fall back, not return a 500.
        user = User.objects.create_user(email="escape@example.com", password="avatar-pass-123", avatar="avatars/../../etc/passwd")

        self.assertEqual(user.get_avatar_url(), DEFAULT_AVATAR_URL)

    def test_stored_avatar_is_served_from_media(self) -> None:
        with TemporaryDirectory() as media_root, override_settings(MEDIA_ROOT=media_root, MEDIA_URL="/media/"):
            user = User.objects.create_user(email="portrait@example.com", password="avatar-pass-123")
            user.avatar.save("portrait.jpg", SimpleUploadedFile("portrait.jpg", b"jpeg-bytes"), save=True)

            url = user.get_avatar_url()

            self.assertTrue(url.startswith("/media/avatars/"), url)
            self.assertNotEqual(url, DEFAULT_AVATAR_URL)


class HeaderAvatarTests(TestCase):
    def test_history_page_renders_the_static_placeholder_for_a_user_without_an_avatar(self) -> None:
        user = User.objects.create_user(email="header@example.com", password="avatar-pass-123")
        self.client.force_login(user)

        response = self.client.get(reverse("sa:history"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, DEFAULT_AVATAR_URL)
        self.assertNotContains(response, "/media/img/default_user.jpg")


class ProfilePageTests(TestCase):
    """The read-only profile page names the account by its USERNAME_FIELD."""

    def test_heading_shows_the_email_because_the_model_has_no_username(self) -> None:
        # AbstractBaseUser has no `username` attribute, so the old {{ user.username }}
        # rendered an empty heading on every profile page.
        user = User.objects.create_user(email="profile@example.com", password="avatar-pass-123")
        self.client.force_login(user)

        response = self.client.get(reverse("accounts:profile", args=[user.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "profile@example.com")


class LogoutTests(TestCase):
    """LogoutView redirects; it never renders a template (registration/logout.html is gone)."""

    def test_post_logs_out_and_redirects_to_the_index(self) -> None:
        user = User.objects.create_user(email="bye@example.com", password="avatar-pass-123")
        self.client.force_login(user)

        response = self.client.post(reverse("accounts:logout"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], "/")
        self.assertNotIn("_auth_user_id", self.client.session)


class LoginPageTests(TestCase):
    """The login form is the 420 px card of Etap 7, not the bare crispy form."""

    def test_get_renders_the_card_and_the_water_coloured_submit(self) -> None:
        response = self.client.get(reverse("accounts:login"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "auth-card")
        self.assertContains(response, "btn btn-primary auth-submit")
        self.assertNotContains(response, "btn-success")

    def test_get_links_only_to_a_route_that_exists(self) -> None:
        response = self.client.get(reverse("accounts:login"))

        self.assertContains(response, f'href="{reverse("accounts:register")}"')

    def test_wrong_credentials_come_back_with_the_error_block(self) -> None:
        User.objects.create_user(email="member@example.com", password="right-pass-123")

        response = self.client.post(
            reverse("accounts:login"),
            {"username": "member@example.com", "password": "wrong-pass-123"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["form"].non_field_errors())
        # crispy's bootstrap4 pack renders non-field errors as .alert.alert-danger,
        # which app.css paints with --bad on --bad-wash inside .auth-card.
        self.assertContains(response, "alert-danger")
        self.assertContains(response, "auth-card")


class RegisterPageTests(TestCase):
    """The register form shares the card and links back to the login page."""

    def test_get_renders_the_card_without_a_green_submit(self) -> None:
        response = self.client.get(reverse("accounts:register"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "auth-card")
        self.assertContains(response, "btn btn-primary auth-submit")
        self.assertNotContains(response, "btn-success")

    def test_get_links_only_to_a_route_that_exists(self) -> None:
        response = self.client.get(reverse("accounts:register"))

        self.assertContains(response, f'href="{reverse("accounts:login")}"')

    def test_mismatched_passwords_come_back_with_the_field_error(self) -> None:
        response = self.client.post(
            reverse("accounts:register"),
            {"email": "new@example.com", "password1": "long-pass-123", "password2": "other-pass-123"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("password2", response.context["form"].errors)
        self.assertFalse(User.objects.filter(email="new@example.com").exists())


class AvatarChangeFormTests(TestCase):
    """Avatar uploads exceeding 5 MB are rejected by UserAvatarForm."""

    def test_oversized_avatar_is_rejected_with_validation_error(self) -> None:
        from accounts.forms import MAX_AVATAR_BYTES, UserAvatarForm

        user = User.objects.create_user(email="big-avatar@example.com", password="avatar-pass-123")
        oversized = SimpleUploadedFile(
            "huge.png",
            b"\x89PNG\r\n\x1a\n" + b"0" * (MAX_AVATAR_BYTES + 1),
            content_type="image/png",
        )
        form = UserAvatarForm(data={}, files={"avatar": oversized}, instance=user)
        self.assertFalse(form.is_valid())
        self.assertIn("avatar", form.errors)


