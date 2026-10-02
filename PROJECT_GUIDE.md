# EconoCausal — Internship Project Guide

## Goal
Estimate the causal effect of price on demand using cross-fitted Double Machine Learning (DML), then demonstrate price scenarios in a Streamlit dashboard.

## Local setup
Use Python 3.11 when possible.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Run and validate

```powershell
python -m src.pipeline
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

Expected automated test result: 5 tests passed.

## Repository contents
- `src/` — data generation, DML estimator and pipeline
- `tests/` — automated tests
- `data/raw/` — reproducible synthetic dataset
- `models/` — trained model artifact
- `reports/` — report, presentation, metrics, predictions and EDA figures
- `app.py` — Streamlit demonstration

## Validation note
The project pipeline and unit tests were validated before delivery. The dashboard must be launched in the user's local environment after installing the declared dependencies.

## Confidentiality
Do not upload the internship assignment letter itself. Keep confidential internship documents outside the public repository.
