"""Mark analysis sessions that stopped while running as failed, and remove their files.

An analysis runs inside the web request. If the worker process is killed (timeout,
out of memory, restart), its session stays ``processing`` and its temporary files stay
on disk. This command moves such sessions to ``failed`` so the history page tells the
user to run the analysis again, and removes analysis files of the same age from the
application's temporary directory. Results and uploads are left untouched.

The analysis form and the history page run the same clean-up with the default age (see
``sa.housekeeping``); the container runs this command at start with ``--all-temp-files``,
before any worker exists.
"""

from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from sa.housekeeping import STALE_MESSAGE, STALE_MINUTES, reap_stale_sessions, remove_stale_analysis_files
from sa.models import CalculationSession

__all__ = ["Command", "STALE_MESSAGE"]


class Command(BaseCommand):
    help = "Mark sessions stuck in 'processing' for longer than --minutes as failed and remove their temporary files."

    def add_arguments(self, parser):
        parser.add_argument(
            "--minutes", type=int, default=STALE_MINUTES, help="Age in minutes after which a session counts as stuck."
        )
        parser.add_argument("--dry-run", action="store_true", help="Only report the sessions that would be changed.")
        parser.add_argument(
            "--all-temp-files",
            action="store_true",
            help="Remove every leftover analysis file, whatever its age; only safe while no analysis runs.",
        )

    def handle(self, *args, minutes: int, dry_run: bool, all_temp_files: bool, **options):
        if minutes < 1:
            raise CommandError("--minutes must be at least 1.")
        now = timezone.now()
        if dry_run:
            stale = CalculationSession.objects.filter(status="processing", updated_at__lt=now - timedelta(minutes=minutes))
            ids = list(stale.values_list("id", flat=True))
            self.stdout.write(f"Would mark {len(ids)} stale session(s) as failed: {ids}")
            return
        count = reap_stale_sessions(minutes, now=now)
        self.stdout.write(f"Marked {count} stale session(s) as failed.")
        files = remove_stale_analysis_files(None if all_temp_files else timedelta(minutes=minutes))
        self.stdout.write(f"Removed {files} leftover analysis file(s).")
