"""Mark analysis sessions that stopped while running as failed.

An analysis runs inside the web request. If the worker process is killed (timeout,
out of memory, restart), its session stays ``processing`` forever. This command
moves such sessions to ``failed`` so the history page tells the user to run the
analysis again. Results and uploads are left untouched.
"""

from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from sa.models import CalculationSession

STALE_MESSAGE = "The analysis did not finish because the server stopped while it was running. Please run it again."


class Command(BaseCommand):
    help = "Mark sessions stuck in 'processing' for longer than --minutes as failed."

    def add_arguments(self, parser):
        parser.add_argument("--minutes", type=int, default=30, help="Age in minutes after which a session counts as stuck.")
        parser.add_argument("--dry-run", action="store_true", help="Only report the sessions that would be changed.")

    def handle(self, *args, minutes: int, dry_run: bool, **options):
        if minutes < 1:
            raise CommandError("--minutes must be at least 1.")
        now = timezone.now()
        stale = CalculationSession.objects.filter(status="processing", updated_at__lt=now - timedelta(minutes=minutes))
        if dry_run:
            ids = list(stale.values_list("id", flat=True))
            self.stdout.write(f"Would mark {len(ids)} stale session(s) as failed: {ids}")
            return
        count = stale.update(status="failed", error_message=STALE_MESSAGE, updated_at=now)
        self.stdout.write(f"Marked {count} stale session(s) as failed.")
