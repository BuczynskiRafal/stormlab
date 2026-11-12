# Deployment

The production setup is one Docker Compose project: `web` (Django on gunicorn, built from the root `Dockerfile`) behind `proxy` (Caddy, which obtains TLS certificates and serves uploaded avatars). Data lives in SQLite on a Docker volume.

## Requirements

- A Linux host with Docker Engine and the Compose plugin.
- A DNS name pointing at the host, with ports 80 and 443 open.
- About 1 GB of RAM per gunicorn worker (TensorFlow and the models are loaded in each worker), plus the proxy.

## Configuration

Create `.env` in the repository root (it is not tracked). Compose reads it for its own variables and passes it to `web`.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `DOMAIN` | yes | – | Public host name. Caddy requests its certificate; also the default for `ALLOWED_HOSTS`. |
| `SECRET_KEY` | yes | – | At least 50 random characters. Startup fails on a missing, short or placeholder key. |
| `ALLOWED_HOSTS` | no | `DOMAIN` | Comma-separated host names, if the site answers on more than one. |
| `CSRF_TRUSTED_ORIGINS` | no | empty | Comma-separated origins such as `https://example.com`; only needed if a proxy rewrites the Host header. |
| `SECURE_HSTS_SECONDS` | no | `3600` | HSTS lifetime. Raise it once HTTPS is known to work. |
| `SECURE_HSTS_INCLUDE_SUBDOMAINS`, `SECURE_HSTS_PRELOAD` | no | `false` | HSTS opt-ins; preloading is hard to undo. |
| `REGISTRATION_OPEN` | no | `true` | `false` hides the sign-up links and makes the sign-up page answer 404. |
| `MAX_INP_BYTES` | no | `52428800` (50 MB) | Largest accepted `.inp` upload. |
| `MAX_REQUEST_BODY` | no | `55MB` | Request size limit at the proxy; keep it a little above `MAX_INP_BYTES`. |
| `ADMIN_IPS` | no | `127.0.0.1/32` | Space-separated IP ranges allowed to open `/admin/`; everyone else gets 403. |
| `WEB_CONCURRENCY` | no | `2` | Number of gunicorn worker processes. |

The image sets `DJANGO_ENV=production`, `SQLITE_PATH=/data/db/db.sqlite3`, `MEDIA_ROOT=/data/media` and `STATIC_ROOT=/srv/static`; there is no need to change them. `app/.env.example` lists every application variable.

Generate a key with:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(50))"
```

## First start

```bash
docker compose build
docker compose run --rm web python manage.py check --deploy --fail-level WARNING
docker compose up -d
docker compose exec web python manage.py createsuperuser
```

On start, `web` applies database migrations, marks analyses stuck in processing for over 30 minutes as failed, and starts gunicorn. `docker compose logs -f web` shows the output.

## Updating

```bash
git pull
docker compose build
docker compose up -d
```

Static files are collected into the image at build time, and migrations run when the container starts.

## Backup

The database is one SQLite file on the `db` volume; avatars are on the `media` volume. Back up the database with SQLite's online backup, which is safe while the site runs:

```bash
docker compose exec web python -c "import sqlite3; sqlite3.connect('/data/db/db.sqlite3').backup(sqlite3.connect('/data/db/backup.sqlite3'))"
docker compose cp web:/data/db/backup.sqlite3 ./backup-$(date +%F).sqlite3
```

## Health check

`GET /healthz/` returns `200 {"status": "ok"}` when the database answers and `503` otherwise. It is served over plain HTTP inside the container (no HTTPS redirect), and Docker uses it for the container's health status (`docker compose ps`).

## Operational notes

- Each worker runs one simulation at a time: the SWMM engine allows a single simulation per process, so gunicorn uses sync workers with one thread. Concurrent analyses are limited by `WEB_CONCURRENCY`.
- The first analysis in each worker takes several seconds longer, because TensorFlow and the models load on first use.
- Analyses run inside the request; the gunicorn timeout is 180 seconds. If a worker is killed mid-analysis, its session stays in processing until `python manage.py reap_stale_sessions` (run at startup; `--minutes` sets the age, default 30) marks it as failed.
- `/admin/` is blocked at the proxy except for `ADMIN_IPS`. Management commands work without it via `docker compose exec web python manage.py ...`.
- Set `REGISTRATION_OPEN=false` to stop new sign-ups on a public instance; existing accounts keep working.
- Uploads larger than `MAX_INP_BYTES` are refused by the form, and larger than `MAX_REQUEST_BODY` by the proxy. Uploads that refer to other files on the server (for example a `[FILES]` section) are refused.
- The `web` container has a read-only root file system; only `/tmp` (in memory) and the two volumes are writable.

For local development with Docker, run `docker compose -f docker-compose.dev.yaml up --build` (development server on `http://localhost:8000`, repository mounted).
