#!/bin/sh
# Prepare the database before the server starts; other commands run unchanged.
set -eu

if [ "${1:-}" = "gunicorn" ]; then
    python manage.py migrate --noinput
    python manage.py reap_stale_sessions
fi

exec "$@"
