# EconoCausal: Dynamic Pricing via Double Machine Learning

An internship-level Data Science & Machine Learning project that estimates the causal relationship between price and demand using **Double Machine Learning (DML)**.

## What this project demonstrates

- Reproducible synthetic retail pricing data generation
- Data cleaning and exploratory data analysis
- Cross-fitted Double Machine Learning
- Random Forest nuisance models
- Price elasticity estimation with uncertainty interval
- Model evaluation and out-of-fold predictions
- Interactive Streamlit demonstration
- Automated unit tests
- Internship-ready report and presentation

## Project structure

```text
EconoCausal_Internship_Project/
├── app.py
├── PROJECT_GUIDE.md
├── README.md
├── requirements.txt
├── run_project.ps1
├── run_project.bat
├── data/raw/pricing_demand.csv
├── models/dml_model.joblib
├── src/
│   ├── generate_data.py
│   ├── dml_model.py
│   └── pipeline.py
├── tests/
│   └── test_pipeline.py
└── reports/
    ├── Project_Report.md
    ├── EconoCausal_Presentation.pptx
    ├── model_metrics.json
    ├── predictions.csv
    ├── test_results.txt
    └── figures/
```

## Run in VS Code

Open the project folder and create/activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the complete pipeline:

```powershell
python -m src.pipeline
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Launch the dashboard:

```powershell
python -m streamlit run app.py
```

Or use the included PowerShell script:

```powershell
.\run_project.ps1
```

## Main result from the validated run

Using the reproducible 8,000-row synthetic dataset:

| Metric | Result |
|---|---:|
| Estimated price elasticity | -1.1194 |
| Standard error | 0.0258 |
| 95% bootstrap interval | [-1.1650, -1.0714] |
| Outcome RMSE | 0.2438 |
| Outcome R² | 0.5456 |
| Treatment RMSE | 0.1005 |
| Treatment R² | 0.9817 |
| Observations | 8,000 |

The data generator uses a known benchmark elasticity of **-1.25**. This benchmark is used only to sanity-check the controlled synthetic experiment; it is not a claim about a real market.

## Validation

The project was tested before delivery:

- Full pipeline: **PASS**
- Unit tests: **5/5 PASS**
- Python compilation check: **PASS**
- Presentation generated successfully
- Dashboard dependency is declared in `requirements.txt`

The isolated build environment used for validation did not have Streamlit installed and had no network access to install it, so the Streamlit server itself could not be launched there. After running `pip install -r requirements.txt` locally, launch it with `python -m streamlit run app.py`.

## Internship deliverables

The internship deliverables include source code, project documentation/report, presentation, model evaluation results and a final demonstration. This repository includes those project artifacts.

## Important note

The dataset is synthetic because no real business dataset was supplied. The dashboard is an educational demonstration, not a production pricing engine.

Do not publish the supplied internship assignment letter itself to GitHub. It is a confidential document and should not be disclosed publicly.
