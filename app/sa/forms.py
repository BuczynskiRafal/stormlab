from collections.abc import Iterable

from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError

from .core.inp_encoding import utf8_chunks
from .core.inp_options import OptionValueError, read_flow_units, simulation_period
from .core.inp_safety import find_unsafe_input
from .core.inp_structure import listing, non_circular_conduits, survey_inp
from .models import INP_EXTENSION_MESSAGE, SWMMModel, has_inp_extension

MAX_SCANNED_BYTES = 1024 * 1024
"""How much of the file the FLOW_UNITS gate reads; [OPTIONS] is always at the top."""


def _head(chunks: Iterable[bytes], limit: int) -> str:
    """The first *limit* bytes of *chunks* as text; a character cut at the end is replaced."""
    head = bytearray()
    for chunk in chunks:
        head += chunk[: limit - len(head)]
        if len(head) >= limit:
            break
    return head.decode("utf-8", errors="replace")


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
        # The name first: any other file would only be told that its [OPTIONS] lack FLOW_UNITS.
        if not has_inp_extension(uploaded.name):
            raise ValidationError(INP_EXTENSION_MESSAGE, code="extension")
        limit = settings.MAX_INP_BYTES
        if uploaded.size > limit:
            raise ValidationError(
                "SWMM file is too large (%(size)d bytes; the limit is %(limit)d).",
                code="too_large",
                params={"size": uploaded.size, "limit": limit},
            )
        try:
            # [OPTIONS] opens every SWMM file; one megabyte bounds the memory a hostile
            # single-line upload can cost (the line cap alone does not bound bytes).
            units = read_flow_units(_head(utf8_chunks(uploaded.chunks), MAX_SCANNED_BYTES).splitlines())
        finally:
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
        try:
            # Every scan reads the UTF-8 bytes the view will write for the engine: a file
            # that is not UTF-8 grows when transcoded, and the line limit is about those bytes.
            unsafe = find_unsafe_input(utf8_chunks(uploaded.chunks))
        finally:
            uploaded.seek(0)
        if unsafe is not None:
            raise ValidationError(
                "SWMM file was rejected: line %(line)d has %(reason)s. "
                "Uploaded models must be self-contained, without references to other files.",
                code="unsafe_input",
                params={"line": unsafe.line, "reason": unsafe.reason},
            )
        try:
            survey = survey_inp(utf8_chunks(uploaded.chunks))
        finally:
            uploaded.seek(0)
        self._check_cost(survey)
        other_shapes = non_circular_conduits(survey)
        if other_shapes:
            raise ValidationError(
                "StormLab assesses circular conduits only; these conduits have another shape: %(conduits)s.",
                code="conduit_shape",
                params={"conduits": listing(other_shapes)},
            )
        return uploaded

    @staticmethod
    def _check_cost(survey) -> None:
        """Refuse a model whose run would outgrow the request or the temporary disk."""
        elements, limit = survey.elements, settings.MAX_MODEL_ELEMENTS
        if elements > limit:
            raise ValidationError(
                "SWMM file has %(count)d elements (conduits, nodes and subcatchments); the limit is %(limit)d.",
                code="too_many_elements",
                params={"count": elements, "limit": limit},
            )
        try:
            period = simulation_period(survey.options)
        except OptionValueError as error:
            raise ValidationError(
                "StormLab cannot read %(option)s %(value)s in [OPTIONS]; write dates as MM/DD/YYYY, "
                "times as HH:MM:SS and the report step as HH:MM:SS longer than zero.",
                code="option_value",
                params={"option": error.option, "value": error.value},
            ) from None
        hours, limit = period.duration_hours, settings.MAX_SIMULATION_HOURS
        if hours > limit:
            raise ValidationError(
                "SWMM file simulates %(hours)s hours (START to END in [OPTIONS]); the limit is %(limit)d hours.",
                code="simulation_too_long",
                params={"hours": f"{hours:g}" if hours < 1e6 else f"{hours:.0f}", "limit": limit},
            )
        steps, limit = period.report_steps, settings.MAX_REPORT_STEPS
        if steps > limit:
            raise ValidationError(
                "SWMM file asks for %(steps)d report steps (reporting period / REPORT_STEP); the limit is "
                "%(limit)d. Lengthen REPORT_STEP or shorten the simulation.",
                code="too_many_report_steps",
                params={"steps": steps, "limit": limit},
            )
