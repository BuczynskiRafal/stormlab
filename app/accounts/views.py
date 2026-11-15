from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model, login, update_session_auth_hash
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import redirect, render
from django.urls import Resolver404, resolve

from accounts.forms import (
    AVATAR_HINT,
    AccountPasswordChangeForm,
    CustomUserCreationForm,
    EmailAuthenticationForm,
    UserAvatarForm,
    UserForm,
)

User = get_user_model()


def register(request):
    if not settings.REGISTRATION_OPEN:
        raise Http404("Registration is closed.")
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Your account is ready. Run your first analysis from the Analysis page.")
            return redirect("/")
    else:
        form = CustomUserCreationForm()
    return render(request, "registration/register.html", {"form": form})


# Why the login page was shown, by the URL name of the page that sent the visitor here.
LOGIN_REASONS = {
    "analysis": "Log in to run an analysis.",
    "analysis_results": "Log in to open the results of your analysis.",
    "conduit_detail": "Log in to open the results of your analysis.",
    "history": "Log in to see the analyses you have run.",
}
DEFAULT_LOGIN_LEDE = "Log in to run an analysis and to return to its results later."


class LoginView(auth_views.LoginView):
    """The login card; when ``next`` names a page behind the login, the lede says why the visitor is here."""

    template_name = "registration/login.html"
    authentication_form = EmailAuthenticationForm

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["login_lede"] = self.login_reason() or DEFAULT_LOGIN_LEDE
        return context

    def login_reason(self) -> str | None:
        # get_redirect_url() returns "" for an unsafe or missing ``next``.
        target = self.get_redirect_url()
        if not target:
            return None
        try:
            match = resolve(target.split("?", 1)[0])
        except Resolver404:
            return None
        return LOGIN_REASONS.get(match.url_name) if match.namespace == "sa" else None


class LogoutView(auth_views.LogoutView):
    """Log out on POST; answer a GET with a confirmation page instead of a blank 405.

    Django's view accepts only POST, so a typed or bookmarked ``/accounts/logout/``
    showed an empty "Method Not Allowed" page. Logging out stays a POST (a GET must not
    change state), but the GET now offers the button, or sends a visitor who is not
    logged in to the home page.
    """

    http_method_names = ["get", "post", "options"]
    next_page = "/"

    def get(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect(self.next_page)
        return render(request, "registration/logout.html")

    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        messages.info(request, "You have logged out.")
        return response


@login_required(login_url="/accounts/login/")
def user_profile(request):
    """The account page: three always-open sections, each a form of its own.

    ``user_profile_forms`` names the section that was submitted: 0 account details,
    1 password, 3 avatar (2, the removed "additional information" form, is refused).
    A rejected form comes back with its errors inline; ``error_section`` lets the
    page move focus to it.
    """
    user = request.user
    user_form = UserForm(instance=user)
    password_form = AccountPasswordChangeForm(user=user)
    avatar_form = UserAvatarForm(instance=user)
    error_section = None

    if request.method == "POST":
        action = request.POST.get("user_profile_forms")

        if action == "0":
            user_form = UserForm(request.POST, instance=user)
            if user_form.is_valid():
                user_form.save()
                messages.success(request, "Your account details were saved.")
                return redirect("accounts:user_profile")
            error_section = "account"

        elif action == "1":
            password_form = AccountPasswordChangeForm(data=request.POST, user=user)
            if password_form.is_valid():
                password_form.save()
                update_session_auth_hash(request, password_form.user)
                messages.success(request, "Your password was changed.")
                return redirect("accounts:user_profile")
            error_section = "password"

        elif action == "3":
            avatar_form = UserAvatarForm(request.POST, request.FILES, instance=user)
            if avatar_form.is_valid():
                avatar_form.save()
                messages.success(request, "Your avatar was updated.")
                return redirect("accounts:user_profile")
            error_section = "avatar"

        else:
            messages.error(request, "The form could not be read. Reload the page and try again.")
            return redirect("accounts:user_profile")

        # A ModelForm writes the rejected values onto its instance while validating;
        # reload it so the heading and the header avatar still show what is saved.
        user.refresh_from_db()

    context = {
        "user_form": user_form,
        "password_form": password_form,
        "avatar_form": avatar_form,
        "avatar_hint": AVATAR_HINT,
        "error_section": error_section,
    }
    return render(request, "accounts/user_profile.html", context)
