from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model, login, update_session_auth_hash
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from django.http import Http404
from django.shortcuts import redirect, render

from accounts.forms import CustomUserCreationForm, UserAvatarForm, UserForm, UserProfileForm

User = get_user_model()


def register(request):
    if not settings.REGISTRATION_OPEN:
        raise Http404("Registration is closed.")
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("/")
    else:
        form = CustomUserCreationForm()
    return render(request, "registration/register.html", {"form": form})


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
    user_form = UserForm(instance=request.user)
    profile_form = UserProfileForm(instance=request.user)
    password_form = PasswordChangeForm(user=request.user)
    avatar_form = UserAvatarForm(instance=request.user)

    if request.method == "POST":
        action = request.POST.get("user_profile_forms")

        if action == "0":
            user_form = UserForm(request.POST, request.FILES, instance=request.user)
            if user_form.is_valid():
                user_form.save()
                messages.success(request, "Your profile was successfully updated!")
                return redirect("accounts:user_profile")
            messages.error(request, "Please correct the errors in the profile form.")

        elif action == "1":
            password_form = PasswordChangeForm(data=request.POST, user=request.user)
            if password_form.is_valid():
                password_form.save()
                update_session_auth_hash(request, password_form.user)
                messages.success(request, "Your password was successfully updated!")
                return redirect("accounts:user_profile")
            messages.error(request, "Please correct the errors in the password form.")

        elif action == "2":
            profile_form = UserProfileForm(request.POST, request.FILES, instance=request.user)
            if profile_form.is_valid():
                profile_form.save()
                messages.success(request, "Your additional information was successfully updated!")
                return redirect("accounts:user_profile")
            messages.error(request, "Please correct the errors in the additional information form.")

        elif action == "3":
            avatar_form = UserAvatarForm(request.POST, request.FILES, instance=request.user)
            if avatar_form.is_valid():
                avatar_form.save()
                messages.success(request, "Your avatar was successfully updated!")
                return redirect("accounts:user_profile")
            messages.error(request, "Please correct the errors in the avatar upload.")

        else:
            messages.error(request, "Invalid form submission.")
            return redirect("accounts:user_profile")

    context = {
        "user_form": user_form,
        "profile_form": profile_form,
        "password_form": password_form,
        "avatar_form": avatar_form,
    }

    return render(request, "accounts/user_profile.html", context)


@login_required(login_url="/accounts/login/")
def profile(request, id):
    user = request.user
    context = {"user": user}
    return render(request, "accounts/profile.html", context)
