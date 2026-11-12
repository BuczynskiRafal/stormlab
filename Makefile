PYTHON ?= ./venv/bin/python
MANAGE = $(PYTHON) app/manage.py

.PHONY: install migrate run test test-core test-stormlab test-django lint typecheck check-deploy collectstatic docker-build docker-up

install:
	$(PYTHON) -m pip install -r requirements.txt -r app/requirements.txt

migrate:
	DJANGO_ENV=development $(MANAGE) migrate

run:
	DJANGO_ENV=development $(MANAGE) runserver

test-core:
	$(PYTHON) -m pytest app/sa/core/tests -p no:cacheprovider

test-stormlab:
	$(PYTHON) -m pytest stormlab/tests -p no:cacheprovider

test-django:
	DJANGO_ENV=testing $(MANAGE) test accounts sa

test: test-core test-stormlab test-django

lint:
	ruff check .
	ruff format --check app/sa/core/ stormlab/ conftest.py

typecheck:
	$(PYTHON) -m pyright

# Production checks; SECRET_KEY and ALLOWED_HOSTS must be set in the environment or app/.env.
check-deploy:
	DJANGO_ENV=production $(MANAGE) check --deploy --fail-level WARNING

collectstatic:
	DJANGO_ENV=production $(MANAGE) collectstatic --noinput

docker-build:
	docker compose build

docker-up:
	docker compose up -d
