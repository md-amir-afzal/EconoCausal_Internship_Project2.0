# EconoCausal

**An internship portfolio project in causal machine learning and pricing analytics.** EconoCausal demonstrates how an analyst can frame a retail pricing question, estimate price elasticity with cross-fitted Double Machine Learning, and communicate a transparent what-if scenario through an interactive dashboard.

> The included retail data is synthetic. Results are educational examples, not findings about a real company or market.

## What the project demonstrates

- A reproducible 50,000-row synthetic retail dataset with business drivers such as promotions, competitor prices, inventory, advertising, and customer segments.
- A Double Machine Learning pipeline with cross-fitted ExtraTrees models and a bootstrap confidence interval.
- A company-style dashboard with live scenario controls, price response curve, baseline comparison, and retail context charts.
- A data explorer with filters, visual analysis, and CSV export.
- An optional AI economics tutor that explains the current model and scenario using aggregate metrics, with chat history limited to the current Streamlit session.
- Clear communication of assumptions, limitations, and the next steps required before using real company data.

## Run locally

From the project folder in PowerShell:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`. Keep the terminal open while using the dashboard.

If the virtual environment does not exist yet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## How to explore

1. Choose a product category, region, and customer segment in the sidebar.
2. Enter the selling price and optional promotion, demand, ad spend, rating, and inventory values to see the scenario update.
3. Review the estimated units, revenue, and demand curve on the dashboard.
4. Open **Explore data** to filter observations and download a CSV sample.
5. Open **Methodology** to review the model, evaluation metrics, assumptions, and responsible-use notes.
6. Open **AI tutor** to ask questions. Configure an OpenAI API key in `.streamlit/secrets.toml` or enter it for the current session. API use can incur charges; no row-level data is sent to the tutor.

## Method summary

The treatment is `log(price)` and the outcome is `log(1 + units_sold)`. Cross-fitted nuisance models predict each using observed product, calendar, promotion, cost, competitor, demand, inventory, advertising, rating, and competition features. Regressing the residualized outcome on residualized treatment yields the estimated elasticity. The dashboard uses that estimated elasticity for the price change; other scenario controls use the known coefficients from the synthetic data generator so their contribution is transparent.

## Limitations and next steps

Synthetic data has known structure and does not capture every source of bias in real retail data. A company deployment would require data validation, a clearly defined pricing intervention, robust sensitivity analysis, and preferably randomized price experiments or another credible identification strategy. Scenario outputs should not be used as automated pricing recommendations.

The AI tutor calls the OpenAI Responses API and does not fine-tune or train a model on chat. Streamlit is the current application format: this repository does not yet produce native Android APK/AAB, iOS, or Microsoft Store packages. Store releases need a dedicated mobile/desktop client, store-specific packaging, signing, privacy disclosures, and developer accounts.

## Project layout

```text
app.py                 Streamlit dashboard and scenario interface
src/data_generator.py  Reproducible synthetic retail data
src/dml_model.py       Cross-fitted DML estimator
tests/                 Existing project checks
reports/               Project reports, metrics, and presentation material
```
