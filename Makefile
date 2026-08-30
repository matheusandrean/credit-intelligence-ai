\
.PHONY: install data validate features train test api app lint format typecheck security docker docker-up mlflow-ui clean

VENV := .venv
PY := $(VENV)/Scripts/python.exe

install:
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m pre_commit install

data:
	$(PY) -m src.data.generate_synthetic_credit_data

validate:
	$(PY) -m src.data.validation

features:
	$(PY) -m src.features.build_features

train:
	$(PY) -m src.models.train

test:
	$(PY) -m pytest --cov=src --cov=api --cov-report=term-missing

eval:
	$(PY) -m src.evaluation.run_golden_eval

sql-reports:
	$(PY) -m src.analytics.run_sql_reports

api:
	$(PY) -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

app:
	$(PY) -m streamlit run app/Home.py

lint:
	$(PY) -m ruff check .
	$(PY) -m black --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m black .

typecheck:
	$(PY) -m mypy src api

security:
	$(PY) -m detect_secrets scan --all-files > .secrets.baseline.tmp
	$(PY) -m pip_audit

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri file:./mlruns

docker:
	docker compose build

docker-up:
	docker compose up

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
