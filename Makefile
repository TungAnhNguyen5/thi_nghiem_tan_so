PY=python3
VENV=.venv
PIP=$(VENV)/bin/pip
PYTEST=$(VENV)/bin/pytest

.PHONY: venv install test clean

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

test: install
	$(PYTEST)

clean:
	rm -rf $(VENV) .pytest_cache dist build
