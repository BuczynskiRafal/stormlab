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

    def test_post_confirms_the_logout_on_the_next_page(self):
        self.client.force_login(User.objects.create_user(email="bye2@example.com", password="avatar-pass-123"))

        response = self.client.post(reverse("accounts:logout"), follow=True)

        self.assertContains(response, "You have logged out.")
        self.assertContains(response, 'role="status"')

    def test_get_offers_the_logout_button_instead_of_a_blank_405(self):
        user = User.objects.create_user(email="get-out@example.com", password="avatar-pass-123")
        self.client.force_login(user)

        response = self.client.get(reverse("accounts:logout"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<title>Log out · StormLab</title>")
        self.assertContains(response, f'<form method="post" action="{reverse("accounts:logout")}">')
        # A GET must not change state: the user is still logged in.
        self.assertEqual(self.client.session["_auth_user_id"], str(user.pk))

    def test_get_without_a_login_goes_to_the_home_page(self):
        response = self.client.get(reverse("accounts:logout"))

        self.assertRedirects(response, "/", fetch_redirect_response=False)


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


class AuthFieldTests(TestCase):
    """Login and sign-up fields tell browsers and password managers what they hold."""

    def test_login_email_field_is_an_email_input_with_autocomplete(self):
        response = self.client.get(reverse("accounts:login"))

        self.assertContains(response, 'type="email" name="username"')
        self.assertContains(response, 'autocomplete="username"')
        self.assertContains(response, 'autocomplete="current-password"')
        self.assertEqual(response.context["form"].fields["username"].label, "E-mail")

    def test_register_fields_carry_autocomplete_hints(self):
        response = self.client.get(reverse("accounts:register"))

        self.assertContains(response, 'type="email" name="email"')
        self.assertContains(response, 'autocomplete="email"')
        self.assertContains(response, 'autocomplete="new-password"', count=2)

    def test_duplicate_email_is_explained_in_a_sentence(self):
        User.objects.create_user(email="taken@example.com", password="right-pass-123")

        response = self.client.post(
            reverse("accounts:register"),
            {"email": "taken@example.com", "password1": "a-Long-pass-9271", "password2": "a-Long-pass-9271"},
        )

        self.assertContains(
            response, "An account with this e-mail address already exists. Log in instead, or use another address."
        )
        self.assertNotContains(response, "User with this Email already exists.")

    def test_wrong_password_message_names_the_fields_plainly(self):
        User.objects.create_user(email="member2@example.com", password="right-pass-123")

        response = self.client.post(reverse("accounts:login"), {"username": "member2@example.com", "password": "nope-123456"})

        self.assertContains(response, "The e-mail address and password do not match an account.")

    def test_sign_up_page_does_not_promise_to_keep_uploaded_models(self):
        response = self.client.get(reverse("accounts:register"))

        self.assertNotContains(response, "keeps every model you upload")
        self.assertContains(response, "The model files you upload are not kept")
        self.assertContains(response, ">Sign up</button>")

    def test_sign_up_greets_the_new_account(self):
        response = self.client.post(
            reverse("accounts:register"),
            {"email": "fresh@example.com", "password1": "a-Long-pass-9271", "password2": "a-Long-pass-9271"},
            follow=True,
        )

        self.assertContains(response, "Your account is ready.")


class LoginReasonTests(TestCase):
    """A visitor sent to the login page from a page behind it reads why."""

    def lede(self, next_url=None):
        query = {"next": next_url} if next_url else {}
        return self.client.get(reverse("accounts:login"), query).context["login_lede"]

    def test_analysis_link_explains_that_an_analysis_needs_a_login(self):
        self.assertEqual(self.lede(reverse("sa:analysis")), "Log in to run an analysis.")

    def test_history_link_explains_that_history_needs_a_login(self):
        self.assertEqual(self.lede(reverse("sa:history")), "Log in to see the analyses you have run.")

    def test_following_the_nav_link_lands_on_the_explained_login_page(self):
        response = self.client.get(reverse("sa:history"), follow=True)

        self.assertContains(response, "Log in to see the analyses you have run.")

    def test_plain_or_foreign_next_falls_back_to_the_general_lede(self):
        general = "Log in to run an analysis and to return to its results later."
        self.assertEqual(self.lede(), general)
        self.assertEqual(self.lede("https://evil.example.com/history/"), general)
        self.assertEqual(self.lede("/no-such-page/"), general)
