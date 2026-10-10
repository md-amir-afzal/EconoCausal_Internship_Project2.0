# EconoCausal

**An internship portfolio project in causal machine learning and pricing analytics.** EconoCausal demonstrates how an analyst can frame a retail pricing question, estimate price elasticity with cross-fitted Double Machine Learning, validate the input data, and communicate a transparent what-if scenario.

> Synthetic data is provided for demonstration. Uploaded business data remains observational, and estimates are not automatically causal findings or pricing recommendations.

## What the project demonstrates

- A reproducible 50,000-row synthetic retail dataset.
- Cross-fitted ExtraTrees nuisance models with fold-local preprocessing, date-aware validation, robust standard errors, and row or cluster bootstrap intervals.
- CSV upload with schema checks, outcome validation, date coverage, missing-value and duplicate reporting, and confounder coverage warnings.
- Interactive price scenarios, dashboard diagnostics, cohort filters, and CSV export.
- An optional AI economics tutor that uses aggregate model and scenario context, never uploaded row-level records.

## Run locally

From the project folder in PowerShell:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

If the virtual environment does not exist yet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Use the dashboard

1. Choose **Synthetic demo** or **Upload a CSV** in the sidebar.
2. For a CSV, include `price`, `units_sold`, and at least one supported control. Common controls are `date`, `category`, `region`, `customer_segment`, `promotion`, `cost`, `competitor_price`, `demand_index`, `inventory`, `ad_spend`, `rating`, and `competitor_count`. `unit_price`, `selling_price`, `quantity`, and `units` are accepted aliases.
3. Enter a selling price to view the estimated scenario. Synthetic mode also supports promotion, demand, advertising, rating, and inventory controls. Uploaded data uses a price-only scenario; synthetic generator coefficients are never applied to business data.
4. Review the estimate, uncertainty interval, support warning, validation strategy, and data-quality diagnostics.
5. Use **Explore data** to filter and download a sample. Use **AI tutor** for explanations. API keys belong in `.streamlit/secrets.toml` or environment variables, never in source control. API usage may incur charges.

## Method summary

The treatment is `log(price)` and the outcome is `log(1 + units_sold)`. Cross-fitted nuisance models predict each using available product, store, calendar, promotion, cost, competitor, demand, inventory, advertising, rating, and competition features. Preprocessing is learned inside each fold. When complete dates span enough months, the estimator trains on earlier month blocks and predicts later blocks; otherwise it labels and uses randomized cross-fitting. The residualized outcome is regressed on residualized price to estimate elasticity. The app reports heteroskedasticity-robust standard errors and a bootstrap interval, clustering the bootstrap by an available store or product identifier when enough clusters exist.

## Limitations

Validation checks do not prove that all confounders have been controlled. Real data requires review of the pricing decision process, repeated observations, missingness, and the business meaning of every field. If stores or products repeat over time, include their identifiers for more appropriate uncertainty estimates. Prefer randomized pricing experiments or another credible identification strategy where feasible. Scenario outputs are educational and must not be used as automated pricing recommendations.

The tutor calls the OpenAI Responses API and does not fine-tune or train a model on chat. Streamlit is the current application format; native Android, iOS, and Microsoft Store packages are not included.

## Project layout

```text
app.py                 Streamlit dashboard, upload flow, and scenario interface
src/data_generator.py  Reproducible synthetic retail data
src/data_validation.py CSV normalization and quality checks
src/dml_model.py       Cross-fitted DML estimator and diagnostics
tests/                 Existing project checks
reports/               Project reports, metrics, and presentation material
```
