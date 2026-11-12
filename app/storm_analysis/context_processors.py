from django.conf import settings


def registration(request):
    """Expose the self-registration switch so templates can hide sign-up links."""
    return {"registration_open": settings.REGISTRATION_OPEN}
