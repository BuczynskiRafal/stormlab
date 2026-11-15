from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm

User = get_user_model()


class UserForm(forms.ModelForm):
    class Meta:
        model = get_user_model()
        fields = ["first_name", "last_name", "email", "phone"]


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = get_user_model()
        fields = ["bio", "website_url", "facebook_url", "github_url", "twitter_url"]


MAX_AVATAR_BYTES = 5 * 1024 * 1024


class UserAvatarForm(forms.ModelForm):
    class Meta:
        model = get_user_model()
        fields = [
            "avatar",
        ]

    def clean_avatar(self):
        avatar = self.cleaned_data.get("avatar")
        if avatar and hasattr(avatar, "size") and avatar.size > MAX_AVATAR_BYTES:
            raise forms.ValidationError(
                f"Avatar file is too large ({avatar.size} bytes; maximum allowed is {MAX_AVATAR_BYTES} bytes)."
            )
        return avatar


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
