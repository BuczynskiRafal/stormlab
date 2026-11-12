# Production image. Build from the repository root: docker build -t stormlab-web .
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    KERAS_HOME=/tmp/keras \
    MPLCONFIGDIR=/tmp/matplotlib

# libgomp1: OpenMP runtime used by the SWMM engine and TensorFlow.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin app \
    && install -d -o app -g app /data/db /data/media

# The repository layout is kept, so the app finds stormlab/ and examples/ next to it.
WORKDIR /srv/stormlab
COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt
COPY stormlab/ stormlab/
COPY examples/ examples/
COPY app/ app/
COPY docker/entrypoint.sh docker/healthcheck.py /usr/local/bin/

WORKDIR /srv/stormlab/app
ENV DJANGO_ENV=production \
    DJANGO_SETTINGS_MODULE=storm_analysis.settings \
    SQLITE_PATH=/data/db/db.sqlite3 \
    MEDIA_ROOT=/data/media \
    STATIC_ROOT=/srv/static \
    WEB_CONCURRENCY=2

# Settings need a key to load; this one exists only for this build step.
RUN SECRET_KEY=collectstatic-only-not-a-secret-0123456789abcdefghijklmnop ALLOWED_HOSTS=localhost \
    python manage.py collectstatic --noinput

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "/usr/local/bin/healthcheck.py"]
ENTRYPOINT ["/bin/sh", "/usr/local/bin/entrypoint.sh"]
# Sync workers with one thread each: the SWMM engine runs one simulation per process.
# The worker count comes from WEB_CONCURRENCY; the timeout covers a slow first analysis.
# The control socket is off because the root file system is read-only in production.
CMD ["gunicorn", "storm_analysis.wsgi:application", "--bind", "0.0.0.0:8000", "--worker-class", "sync", "--threads", "1", "--timeout", "180", "--graceful-timeout", "30", "--no-control-socket", "--access-logfile", "-"]
