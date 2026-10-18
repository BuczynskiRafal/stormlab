import io

from django import forms
from django.core.exceptions import ValidationError

from .core.inp_options import read_flow_units
from .models import SWMMModel

MAX_INP_BYTES = 50 * 1024 * 1024
"""Upper bound for an uploaded model; the Wrocław network is 60 kB."""

MAX_SCANNED_BYTES = 1024 * 1024
"""How much of the file the FLOW_UNITS gate reads; [OPTIONS] is always at the top."""


class SWMMModelForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Update zone field to show frost zone values in labels
        zone_choices = []
        for zone_num, frost_value in SWMMModel.ZONE_CHOICES:
            zone_choices.append((zone_num, f"Zone {zone_num} (Frost depth: {frost_value}m)"))
        self.fields["zone"].choices = zone_choices
        self.fields["zone"].widget.attrs.update({"class": "form-select"})
        self.fields["file"].widget.attrs.update({"class": "form-control", "accept": ".inp"})

    class Meta:
        model = SWMMModel
        fields = ["file", "zone"]
        labels = {"file": "SWMM Model File (.inp)", "zone": "Frost Zone"}
        help_texts = {
            "file": "Upload your SWMM model file with .inp extension",
            "zone": "Select the appropriate frost zone for your region",
        }

    def clean_file(self):
        uploaded = self.cleaned_data.get("file")
        if not uploaded:
            return uploaded
        if uploaded.size > MAX_INP_BYTES:
            raise ValidationError(
                "SWMM file is too large (%(size)d bytes; the limit is %(limit)d).",
                code="too_large",
                params={"size": uploaded.size, "limit": MAX_INP_BYTES},
            )
        wrapper = io.TextIOWrapper(uploaded.file, encoding="utf-8", errors="replace")
        try:
            # [OPTIONS] opens every SWMM file; one megabyte bounds the memory a hostile
            # single-line upload can cost (the line cap alone does not bound bytes).
            units = read_flow_units(wrapper.read(MAX_SCANNED_BYTES).splitlines())
        finally:
            wrapper.detach()
            uploaded.seek(0)
        if units is None:
            raise ValidationError(
                "SWMM file has no FLOW_UNITS in [OPTIONS]; the app requires CMS.",
                code="flow_units_missing",
            )
        if units != "CMS":
            raise ValidationError(
                "This app analyses SI models only: FLOW_UNITS must be CMS (file has %(units)s).",
                code="flow_units",
                params={"units": units},
            )
        return uploaded
