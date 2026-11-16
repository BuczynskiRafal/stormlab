from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm, UserCreationForm
from django.core.files.uploadedfile import UploadedFile

User = get_user_model()


class UserForm(forms.ModelForm):
    """The "Account" section of the account page: e-mail and name.

    Phone, bio and the website and social-network links are model columns the
    application never uses, so they are no longer offered (the columns stay; dropping
    them needs a migration of its own).
    """

    class Meta:
        model = get_user_model()
        fields = ["email", "first_name", "last_name"]
        labels = {"email": "E-mail", "first_name": "First name", "last_name": "Last name"}
        widgets = {
            "email": forms.EmailInput(attrs={"autocomplete": "email", "autocapitalize": "none"}),
            "first_name": forms.TextInput(attrs={"autocomplete": "given-name"}),
            "last_name": forms.TextInput(attrs={"autocomplete": "family-name"}),
        }
        error_messages = {
            "email": {"unique": "Another account already uses this e-mail address."},
        }

    def clean_first_name(self):
        return (self.cleaned_data.get("first_name") or "").strip()

    def clean_last_name(self):
        return (self.cleaned_data.get("last_name") or "").strip()


MAX_AVATAR_BYTES = 5 * 1024 * 1024
AVATAR_EXTENSIONS = ("jpg", "jpeg", "png", "gif", "webp")
AVATAR_HINT = "JPG, PNG, GIF or WebP, up to 5 MB."


class UserAvatarForm(forms.ModelForm):
    class Meta:
        model = get_user_model()
        fields = [
            "avatar",
        ]
        labels = {"avatar": "Image"}
        help_texts = {"avatar": AVATAR_HINT}
        widgets = {"avatar": forms.FileInput(attrs={"accept": ",".join(f".{ext}" for ext in AVATAR_EXTENSIONS)})}
        error_messages = {
            "avatar": {
                "required": "Choose an image to upload.",
                "invalid_image": "This file is not an image the site can read. Choose a JPG, PNG, GIF or WebP file.",
            },
        }

    def clean_avatar(self):
        avatar = self.cleaned_data.get("avatar")
        # Without an upload the field falls back to the stored value, which for most
        # accounts is the placeholder "/img/default_user.jpg"; asking that for its
        # size makes the storage raise SuspiciousFileOperation (a 400). This form only
        # uploads, so no new file is simply a missing image.
        if not isinstance(avatar, UploadedFile):
            raise forms.ValidationError(self.fields["avatar"].error_messages["required"], code="required")
        extension = avatar.name.rsplit(".", 1)[-1].lower() if "." in avatar.name else ""
        if extension not in AVATAR_EXTENSIONS:
            raise forms.ValidationError("Choose a JPG, PNG, GIF or WebP file.", code="extension")
        if avatar.size > MAX_AVATAR_BYTES:
            raise forms.ValidationError(
                "The image is %(size).1f MB; the limit is 5 MB.",
                code="too_large",
                params={"size": avatar.size / (1024 * 1024)},
            )
        return avatar


class AccountPasswordChangeForm(PasswordChangeForm):
    """Django's password change form with the labels the account page uses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["old_password"].label = "Current password"
        self.fields["new_password1"].label = "New password"
        self.fields["new_password2"].label = "New password, again"
        # The page opens on the Account section; only a failed submit moves focus here.
        self.fields["old_password"].widget.attrs.pop("autofocus", None)


class EmailAuthenticationForm(AuthenticationForm):
    """The login form with an e-mail field: ``type=email`` and the autocomplete hints password managers read."""

    username = forms.EmailField(
        label="E-mail",
        widget=forms.EmailInput(attrs={"autofocus": True, "autocomplete": "username", "autocapitalize": "none"}),
    )

    error_messages = {
        **AuthenticationForm.error_messages,
        "invalid_login": "The e-mail address and password do not match an account.",
    }

    def __init__(self, request=None, *args, **kwargs):
        super().__init__(request, *args, **kwargs)
        # AuthenticationForm.__init__ resets the label from USERNAME_FIELD ("Email").
        self.fields["username"].label = "E-mail"


class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = get_user_model()
        fields = [
            "email",
        ]
        labels = {"email": "E-mail"}
        widgets = {"email": forms.EmailInput(attrs={"autocomplete": "email", "autocapitalize": "none", "autofocus": True})}
        error_messages = {
            "email": {"unique": "An account with this e-mail address already exists. Log in instead, or use another address."},
        }
