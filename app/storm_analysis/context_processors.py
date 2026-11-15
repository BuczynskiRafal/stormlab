from django.conf import settings


def registration(request):
    """Site-wide values the templates read.

    ``registration_open`` hides the sign-up links when self-registration is closed;
    ``max_inp_bytes`` is the upload limit the analysis form states. (The name predates
    the second key; renaming it means editing TEMPLATES in the settings.)
    """
    return {
        "registration_open": settings.REGISTRATION_OPEN,
        "max_inp_bytes": settings.MAX_INP_BYTES,
    }
