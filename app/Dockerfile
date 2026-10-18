FROM python:3.10.13-slim-bullseye

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV DJANGO_SETTINGS_MODULE=storm_analysis.settings

# NOTE: DJANGO_ENV=production should be set at deploy time (e.g. via -e or an
# env file). It is deliberately not baked in here because the production
# settings hard-require SECRET_KEY (and ALLOWED_HOSTS) from the environment,
# which would make the container unbootable without them.
ENV DJANGO_ENV=development

RUN python manage.py migrate --noinput

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
