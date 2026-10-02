# EconoCausal: Dynamic Pricing via Double Machine Learning

## Executive summary
EconoCausal is an internship-level Data Science and Machine Learning project for estimating the causal relationship between product price and demand. The implementation uses a cross-fitted Double Machine Learning estimator with Random Forest nuisance models.

## Business problem
Observed price and sales can be confounded by demand, inventory, promotions, costs and competitor prices. The project therefore residualizes both demand and price against observed covariates before estimating price elasticity.

## Dataset
The project uses a reproducible synthetic retail dataset with 8,000 daily observations. Features include category, region, day of week, holiday, promotion, cost, competitor price, demand index, inventory, price and units sold. The generator uses a benchmark elasticity of -1.25 for a controlled validation experiment.

## Method
Treatment is `log(price)` and outcome is `log1p(units_sold)`. Five-fold cross-fitting is used. Random Forest models estimate the conditional outcome and treatment components. The causal coefficient is estimated from residualized treatment and outcome, with a bootstrap interval for uncertainty.

## Validated results
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

## EDA and application
The pipeline produces price-vs-demand, demand-by-category and price-distribution figures. The Streamlit dashboard allows controlled scenario inputs and displays the model elasticity, scenario demand and evaluation metrics.

## Testing
Five automated unit tests cover dataset shape, positive prices, finite elasticity, expected elasticity direction and finite metrics. The full pipeline and static compilation checks were also validated before delivery.

## Limitations
The dataset is synthetic and the estimate controls only for observed variables included in the design. Results should not be interpreted as a real-market pricing recommendation. The dashboard is an educational demonstration rather than a production pricing engine.
