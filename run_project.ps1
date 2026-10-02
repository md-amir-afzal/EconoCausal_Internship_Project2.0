$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Running ML pipeline..."
python -m src.pipeline

Write-Host "Running tests..."
python -m unittest discover -s tests -v

Write-Host "Starting Streamlit dashboard..."
python -m streamlit run app.py
