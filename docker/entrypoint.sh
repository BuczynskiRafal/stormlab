#!/bin/sh
# Prepare the database before the server starts; other commands run unchanged.
set -eu

if [ "${1:-}" = "gunicorn" ]; then
    python manage.py migrate --noinput
    # No worker runs yet, so every file left in the analysis directory belongs to a dead one.
    python manage.py reap_stale_sessions --all-temp-files
fi

exec "$@"
