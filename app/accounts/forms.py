from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm

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



class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = get_user_model()
        fields = [
            "email",
        ]

    def __init__(self, *args, **kwargs):
        super(CustomUserCreationForm, self).__init__(*args, **kwargs)

    def save(self, commit=True):
        user = super().save(commit=False)
        if commit:
            user.save()
        return user
