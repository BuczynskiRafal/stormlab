from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.core.exceptions import SuspiciousFileOperation
from django.db import models
from django.templatetags.static import static
from django.utils import timezone

# The placeholder portrait ships with the application, not with the user uploads:
# it lives in ``static/img/`` and is addressed through ``STATIC_URL``.
DEFAULT_AVATAR_STATIC_PATH = "img/default_user.jpg"


class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        """
        Creates and returns a user with a given email address and password.
        """
        if not email:
            raise ValueError("Adres email jest wymagany")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        """
        Creates and returns a superuser with a given email address and password.
        """
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser musi mieć is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser musi mieć is_superuser=True.")

        return self.create_user(email, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=12, blank=True, null=True)
    first_name = models.CharField(max_length=30, null=True, blank=True)
    last_name = models.CharField(max_length=150, null=True, blank=True)
    bio = models.TextField(null=True, blank=True)
    avatar = models.ImageField(upload_to="avatars/", null=True, default="/img/default_user.jpg")
    date_joined = models.DateTimeField(default=timezone.now)

    website_url = models.URLField(blank=True, null=True)
    facebook_url = models.URLField(blank=True, null=True)
    github_url = models.URLField(blank=True, null=True)
    twitter_url = models.URLField(blank=True, null=True)

    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    groups = models.ManyToManyField(
        "auth.Group",
        verbose_name="groups",
        blank=True,
        help_text="The groups this user belongs to. A user will get all permissions granted to each of their groups.",
        related_name="user_custom",
        related_query_name="user",
    )
    user_permissions = models.ManyToManyField(
        "auth.Permission",
        verbose_name="user permissions",
        blank=True,
        help_text="Specific permissions for this user.",
        related_name="user_custom_perm",
        related_query_name="user",
    )

    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    EMAIL_FIELD = "email"

    def __str__(self):
        return self.email

    def get_avatar_url(self) -> str:
        """Return the URL of the profile picture, or of the bundled placeholder.

        The historical field default ``"/img/default_user.jpg"`` names a file that has
        never existed under ``MEDIA_ROOT`` - the picture is shipped in ``static/img/`` -
        so every account created without an upload rendered a broken image. Any absolute
        name is treated as such a placeholder and answered from static files without
        reaching the storage.

        A relative name still reaches ``FileSystemStorage.exists()``, which raises
        ``SuspiciousFileOperation`` for anything that escapes ``MEDIA_ROOT`` (``..``
        segments). Such a name cannot address a servable file either, so it falls back
        to the placeholder instead of propagating a 500 to every page with a header.
        """
        name = self.avatar.name if self.avatar else ""
        if not name or name.startswith("/") or name == DEFAULT_AVATAR_STATIC_PATH:
            return static(DEFAULT_AVATAR_STATIC_PATH)
        try:
            exists = self.avatar.storage.exists(name)
        except SuspiciousFileOperation:
            return static(DEFAULT_AVATAR_STATIC_PATH)
        if not exists:
            return static(DEFAULT_AVATAR_STATIC_PATH)
        return self.avatar.url
