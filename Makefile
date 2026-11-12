PYTHON ?= ./venv/bin/python
MANAGE = $(PYTHON) app/manage.py

.PHONY: install migrate run test test-core test-stormlab test-django lint typecheck check-deploy collectstatic docker-build docker-up

# The SWMM wheel for macOS ships libraries with invalid code signatures, and the
# system kills any process that loads them until they are re-signed.
install:
	$(PYTHON) -m pip install -r requirements.txt
ifeq ($(shell uname -s),Darwin)
	$(PYTHON) -c "import glob, subprocess, sysconfig; d = sysconfig.get_paths()['platlib'] + '/swmm/toolkit/'; subprocess.run(['codesign', '--force', '--sign', '-', *glob.glob(d + '*.dylib'), *glob.glob(d + '*.so')], check=True)"
endif

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
