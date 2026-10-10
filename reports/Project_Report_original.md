# EconoCausal: Dynamic Pricing via Double Machine Learning

## 1. Executive summary

EconoCausal is an internship-level Data Science and Machine Learning project that estimates the causal relationship between product price and demand. The core method is Double Machine Learning (DML), which uses machine learning models to remove observed confounding before estimating the treatment effect.

The supplied assignment specifies the project name **“EconoCausal: Dynamic Pricing via Double Machine Learning”** and asks for practical exposure to data analysis, ML modeling, evaluation and industry-standard documentation. It also lists source code, project documentation/report, presentation, model evaluation results and final demonstration among the expected deliverables.

This implementation uses a reproducible synthetic retail dataset because no real business dataset was provided with the assignment. The synthetic data intentionally includes observed confounding so the DML method has a meaningful causal-inference task.

## 2. Business problem

A simple correlation between price and sales can be misleading because companies do not set prices randomly. For example, a business may increase price when demand is high or inventory is low. A pricing model should therefore distinguish:

- observed demand drivers,
- the price decision/treatment,
- resulting demand,
- the causal effect of changing price.

The project estimates price elasticity of demand and provides an interactive scenario demonstration.

## 3. Dataset

The generated dataset contains 8,000 daily product observations with:

- category and region,
- day of week and holiday flag,
- promotion flag,
- unit cost,
- competitor price,
- demand index,
- inventory,
- price,
- units sold.

The data generator uses a known causal elasticity of -1.25 to create a controlled benchmark. The benchmark is not a claim about any real company or market.

## 4. Methodology

### 4.1 Treatment and outcome

- Treatment: `log(price)`
- Outcome: `log1p(units_sold)`
- Covariates: category, region, calendar variables, promotion, cost, competitor price, demand index and inventory.

Using logarithms makes the estimated coefficient interpretable as an approximate percentage elasticity.

### 4.2 Double Machine Learning

For each cross-validation fold:

1. Fit a Random Forest outcome model for E[Y|X].
2. Fit a Random Forest treatment model for E[D|X].
3. Calculate residualized outcome and treatment.
4. Estimate the coefficient from residualized outcome on residualized treatment.
5. Repeat over folds to avoid using the same observations for nuisance-model training and effect estimation.

A bootstrap procedure is also used to provide a 95% empirical interval.

## 5. Exploratory data analysis

The pipeline creates:

- price vs demand scatter plot,
- demand by category chart,
- price distribution histogram.

These are stored under `reports/figures/`.

## 6. Evaluation

The pipeline reports:

- outcome RMSE,
- outcome R²,
- treatment RMSE,
- treatment R²,
- estimated elasticity,
- standard error,
- 95% bootstrap interval,
- sample size.

These metrics evaluate the nuisance prediction tasks and quantify uncertainty around the causal estimate.

\n### 6.1 Main run results\n\nThe tested pipeline produced the following results on the reproducible 8,000-row synthetic dataset:\n\n| Metric | Result |\n|---|---:|\n| Estimated price elasticity | -1.1194 |\n| Standard error | 0.0258 |\n| 95% bootstrap interval | [-1.1650, -1.0714] |\n| Outcome RMSE | 0.2438 |\n| Outcome R² | 0.5456 |\n| Treatment RMSE | 0.1005 |\n| Treatment R² | 0.9817 |\n| Observations | 8000 |\n\nThe synthetic generator uses a true elasticity of -1.25 as a benchmark; the DML estimate is close to that benchmark, which provides a useful sanity check for this controlled demonstration.\n
## 7. Limitations

1. The dataset is synthetic.
2. DML only controls for observed covariates included in the model.
3. Unmeasured confounding can still bias the estimate.
4. The interactive application is a demonstration, not an automated production pricing engine.
5. The causal interpretation depends on the identification assumptions of the design.

## 8. Reproducibility

The project uses random seed 42 for the main pipeline. All major steps are executable from the VS Code terminal.

## 9. Run commands

```powershell
.venv\Scripts\Activate
pip install -r requirements.txt
python -m src.pipeline
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

## 10. Deliverables

- Source code: repository root and `src/`
- Documentation: `README.md`
- Project report: this file
- Presentation: `reports/EconoCausal_Presentation.pptx`
- Model evaluation: `reports/model_metrics.json`
- Predictions: `reports/predictions.csv`
- Demonstration: `app.py`


## 10. Validation record

The project was validated before delivery:

- `python -m src.pipeline` completed successfully on the reproducible 8,000-row dataset.
- `python -m unittest discover -s tests -v` passed all 5 automated tests.
- `python -m compileall -q src tests app.py` completed without syntax errors.
- The isolated build environment did not contain Streamlit, so the dashboard process itself could not be launched there; `streamlit` remains declared in `requirements.txt` and is covered by the documented local run command.

## 11. Final internship checklist

The repository contains source code, documentation, project report, presentation, model evaluation results, predictions, EDA figures, automated tests, and the Streamlit demonstration. The supplied assignment identifies these categories as expected project deliverables.
