.PHONY: help install install-dev test test-cov run-api run-dashboard lint format clean init-db docker-build docker-up

PYTHON ?= python
PIP := $(PYTHON) -m pip
UVICORN := $(PYTHON) -m uvicorn
STREAMLIT := $(PYTHON) -m streamlit
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
BLACK := $(PYTHON) -m black

help:
	@echo "ConfTest Developer Commands:"
	@echo "  make install        - Install runtime dependencies"
	@echo "  make install-dev    - Install runtime & dev dependencies and local package"
	@echo "  make init-db        - Initialize SQLite database tables"
	@echo "  make test           - Run Pytest test suite"
	@echo "  make test-cov       - Run Pytest with test coverage report"
	@echo "  make run-api        - Start FastAPI backend server (port 8000)"
	@echo "  make run-dashboard  - Start Streamlit interactive dashboard (port 8501)"
	@echo "  make lint           - Run Ruff code quality checks"
	@echo "  make format         - Format with Black and apply Ruff fixes"
	@echo "  make docker-build   - Build Docker container image"
	@echo "  make docker-up      - Run API and Dashboard via Docker Compose"
	@echo "  make clean          - Remove caches and temporary build artifacts"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

init-db:
	$(PYTHON) -m conftest.db.init_db

test:
	$(PYTEST) -v tests/

test-cov:
	$(PYTEST) -v --cov=src/conftest --cov-report=term-missing --cov-report=html tests/

run-api:
	$(UVICORN) conftest.api.main:app --host 127.0.0.1 --port 8000 --reload

run-dashboard:
	$(STREAMLIT) run dashboard/app.py

lint:
	$(RUFF) check src/ tests/ dashboard/

format:
	$(BLACK) src/ tests/ dashboard/
	$(RUFF) check --fix src/ tests/ dashboard/

docker-build:
	docker build -t conftest:latest .

docker-up:
	docker compose up -d

clean:
	rm -rf .pytest_cache .coverage htmlcov __pycache__ *.egg-info build dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
