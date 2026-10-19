PYTHON ?= ./venv/bin/python
MANAGE = $(PYTHON) app/manage.py

.PHONY: install migrate run test test-core test-stormlab test-django lint typecheck

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
