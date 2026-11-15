from accounts.models import User
from django.core.exceptions import ValidationError
from django.db import models
from django.utils.translation import gettext_lazy as _


INP_EXTENSION_MESSAGE = "Choose a SWMM input file (.inp)."


def has_inp_extension(name: str) -> bool:
    """Whether *name* ends in ``.inp`` in any letter case (``MODEL.INP`` is a SWMM file too)."""
    return name.lower().endswith(".inp")


def validate_file_extension(file):
    if not has_inp_extension(file.name):
        raise ValidationError(_(INP_EXTENSION_MESSAGE), code="extension")


ZONE_NUMERALS = {1: "I", 2: "II", 3: "III", 4: "IV"}


def zone_label(zone: int, depth: float) -> str:
    """ "Zone II (1.0 m)": the Roman numeral of the zone and its frost depth, as the site writes them."""
    return f"Zone {ZONE_NUMERALS.get(zone, zone)} ({depth:.1f} m)"


class SWMMModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to="user_models/", validators=[validate_file_extension])
    # The second element is the frost depth the analysis uses (get_frost_zone_value), not a
    # label; labels come from zone_label().
    ZONE_CHOICES = (
        (1, 0.8),
        (2, 1.0),
        (3, 1.2),
        (4, 1.4),
    )
    zone = models.IntegerField(choices=ZONE_CHOICES, null=False, blank=False)
    created_at = models.DateTimeField(auto_now_add=True, null=True)

    def __str__(self):
        return f"{self.user.get_username()} - {self.file.name}"

    def get_frost_zone_value(self):
        """Return the actual frost zone value based on the selected zone."""
        zone_mapping = dict(self.ZONE_CHOICES)
        return zone_mapping.get(self.zone, 0.8)  # Default to 0.8 if not found

    def get_zone_display_name(self):
        """The zone as the site names it: "Zone I (0.8 m)" ... "Zone IV (1.4 m)"."""
        return zone_label(self.zone, self.get_frost_zone_value())


class CalculationSession(models.Model):
    STATUS_CHOICES = [("processing", "Processing"), ("completed", "Completed"), ("failed", "Failed")]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    swmm_model = models.ForeignKey(SWMMModel, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    frost_zone = models.FloatField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="processing")
    error_message = models.TextField(blank=True, null=True)
    model_metadata = models.JSONField(
        null=True,
        blank=True,
        help_text="The recommendation model that produced the results, as get_model_metadata() described it "
        "when the analysis completed; empty for sessions completed before it was recorded",
    )

    class Meta:
        ordering = ["-created_at"]
        # Serves the stale-session sweep the analysis form and history page run (sa.housekeeping).
        indexes = [models.Index(fields=["status", "updated_at"], name="sa_session_status_updated")]

    def __str__(self):
        return f"{self.user.email} - {self.created_at.strftime('%Y-%m-%d %H:%M')} - {self.status}"


class ConduitData(models.Model):
    session = models.ForeignKey(CalculationSession, on_delete=models.CASCADE, related_name="conduits")
    conduit_name = models.CharField(max_length=100, db_index=True)

    # Geometric and flow data
    geom1 = models.FloatField(help_text="Diameter [m]")
    max_v = models.FloatField(help_text="Maximum velocity [m/s]")
    max_q = models.FloatField(help_text="Maximum flow [m³/s]")
    filling = models.FloatField(help_text="Filling height [m]")
    slope_per_mile = models.FloatField(help_text="Slope per mile")
    length = models.FloatField(help_text="Length [m]")

    # Validation flags
    val_max_fill = models.IntegerField(help_text="Valid maximum filling (0/1)")
    val_max_v = models.IntegerField(help_text="Valid maximum velocity (0/1)")
    val_min_v = models.IntegerField(help_text="Valid minimum velocity (0/1)")
    val_max_slope = models.IntegerField(help_text="Valid maximum slope (0/1)")
    val_min_slope = models.IntegerField(help_text="Valid minimum slope (0/1)")
    val_depth = models.IntegerField(help_text="Valid depth (0/1)")
    val_coverage = models.IntegerField(help_text="Valid coverage (0/1)")

    # Diameter analysis; NULL when the diameter is outside 0.2-2.0 m and was not assessed
    min_diameter = models.FloatField(help_text="Minimum required diameter [m]", null=True, blank=True)
    is_min_diameter = models.IntegerField(help_text="Is minimum diameter (0/1)", null=True, blank=True)
    increase_dia = models.IntegerField(help_text="Should increase diameter (0/1)", null=True, blank=True)
    reduce_dia = models.IntegerField(help_text="Can reduce diameter (0/1)", null=True, blank=True)

    # Slope analysis; NULL when the diameter is outside 0.2-2.0 m and was not assessed
    min_required_slope = models.FloatField(help_text="Minimum required slope [‰]", null=True, blank=True)
    increase_slope = models.IntegerField(help_text="Should increase slope (0/1)", null=True, blank=True)
    max_allowable_slope = models.FloatField(help_text="Maximum allowable slope [‰]", null=True, blank=True)
    reduce_slope = models.IntegerField(help_text="Should reduce slope (0/1)", null=True, blank=True)

    # Node information
    inlet_node = models.CharField(max_length=100)
    outlet_node = models.CharField(max_length=100)
    inlet_max_depth = models.FloatField(help_text="Inlet node max depth [m]")
    outlet_max_depth = models.FloatField(help_text="Outlet node max depth [m]")
    inlet_ground_elevation = models.FloatField(help_text="Inlet ground elevation [m]")
    outlet_ground_elevation = models.FloatField(help_text="Outlet ground elevation [m]")
    inlet_ground_cover = models.FloatField(help_text="Inlet ground cover [m]")
    outlet_ground_cover = models.FloatField(help_text="Outlet ground cover [m]")

    # Subcatchment information
    subcatchment = models.CharField(max_length=100, default="-")
    sbc_category = models.CharField(max_length=100, default="-")

    # Recommendation
    recommendation = models.CharField(max_length=50, help_text="AI-generated recommendation")

    # Confidence scores from neural network model
    confidence_pump = models.FloatField(help_text="Confidence for pump recommendation [0-1]", null=True, blank=True)
    confidence_tank = models.FloatField(help_text="Confidence for tank recommendation [0-1]", null=True, blank=True)
    confidence_seepage_boxes = models.FloatField(
        help_text="Confidence for seepage boxes recommendation [0-1]", null=True, blank=True
    )
    confidence_diameter_increase = models.FloatField(
        help_text="Confidence for diameter increase recommendation [0-1]", null=True, blank=True
    )
    confidence_diameter_reduction = models.FloatField(
        help_text="Confidence for diameter reduction recommendation [0-1]", null=True, blank=True
    )
    confidence_slope_increase = models.FloatField(
        help_text="Confidence for slope increase recommendation [0-1]", null=True, blank=True
    )
    confidence_slope_reduction = models.FloatField(
        help_text="Confidence for slope reduction recommendation [0-1]", null=True, blank=True
    )
    confidence_depth_increase = models.FloatField(
        help_text="Confidence for depth increase recommendation [0-1]", null=True, blank=True
    )
    confidence_valid = models.FloatField(help_text="Confidence for valid (no changes needed) [0-1]", null=True, blank=True)

    class Meta:
        unique_together = ["session", "conduit_name"]
        indexes = [
            models.Index(fields=["session", "conduit_name"]),
            models.Index(fields=["recommendation"]),
        ]

    def __str__(self):
        return f"{self.session} - {self.conduit_name}"


class NodeData(models.Model):
    session = models.ForeignKey(CalculationSession, on_delete=models.CASCADE, related_name="nodes")
    node_name = models.CharField(max_length=100, db_index=True)

    # Node properties
    max_depth = models.FloatField(help_text="Maximum depth [m]", null=True, blank=True)
    invert_elevation = models.FloatField(help_text="Invert elevation [m]", null=True, blank=True)
    max_water_depth = models.FloatField(help_text="Maximum simulated water depth above invert [m]", null=True, blank=True)
    hours_flooded = models.FloatField(help_text="Hours flooded [h]", null=True, blank=True)
    flood_volume = models.FloatField(help_text="Total flood volume [10^6 l]", null=True, blank=True)

    # Subcatchment information
    subcatchment = models.CharField(max_length=100, default="-")
    sbc_category = models.CharField(max_length=100, default="-")

    class Meta:
        unique_together = ["session", "node_name"]
        indexes = [
            models.Index(fields=["session", "node_name"]),
        ]

    def __str__(self):
        return f"{self.session} - {self.node_name}"


class SubcatchmentData(models.Model):
    session = models.ForeignKey(CalculationSession, on_delete=models.CASCADE, related_name="subcatchments")
    subcatchment_name = models.CharField(max_length=100, db_index=True)

    # Subcatchment properties
    area = models.FloatField(help_text="Area [ha]")
    perc_imperv = models.FloatField(help_text="Percent impervious [%]")
    perc_slope = models.FloatField(help_text="Percent slope [%]")
    outlet = models.CharField(max_length=100, help_text="Outlet node")

    # Flow data
    total_runoff_mg = models.FloatField(help_text="Total runoff [10⁶ l when FLOW_UNITS=CMS]", null=True, blank=True)
    peak_runoff = models.FloatField(help_text="Peak runoff [m³/s when FLOW_UNITS=CMS]", null=True, blank=True)
    runoff_coeff = models.FloatField(help_text="Runoff coefficient", null=True, blank=True)

    # Category
    category = models.CharField(max_length=100, default="-", help_text="Land use category")

    class Meta:
        unique_together = ["session", "subcatchment_name"]
        indexes = [
            models.Index(fields=["session", "subcatchment_name"]),
            models.Index(fields=["category"]),
        ]

    def __str__(self):
        return f"{self.session} - {self.subcatchment_name}"


class SessionGeometry(models.Model):
    session = models.OneToOneField(CalculationSession, on_delete=models.CASCADE, related_name="geometry")
    data = models.JSONField(help_text="Network drawing geometry captured from the INP at analysis time")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session} - geometry"
