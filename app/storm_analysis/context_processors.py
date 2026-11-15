from django.conf import settings

ROMAN_NUMERALS = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}


def frost_zone_labels():
    """``[(zone, "Zone I (0.8 m)"), ...]`` from ``SWMMModel.ZONE_CHOICES``.

    Display only: the zone numbers and their frost depths are read, never changed, so
    the form, History and About print the same label for the same zone.
    """
    from sa.models import SWMMModel  # imported lazily: context processors load before the apps

    return [(zone, f"Zone {ROMAN_NUMERALS.get(zone, zone)} ({depth:.1f} m)") for zone, depth in SWMMModel.ZONE_CHOICES]


def registration(request):
    """Site-wide values the templates read.

    ``registration_open`` hides the sign-up links when self-registration is closed;
    ``max_inp_bytes`` is the upload limit the analysis form states; ``frost_zones``
    are the zone labels shared by the analysis form and History. (The name predates the
    last two keys; renaming it means editing TEMPLATES in the settings.)
    """
    return {
        "registration_open": settings.REGISTRATION_OPEN,
        "max_inp_bytes": settings.MAX_INP_BYTES,
        "frost_zones": frost_zone_labels(),
    }
