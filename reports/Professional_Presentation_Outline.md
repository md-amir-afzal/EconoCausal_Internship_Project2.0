# EconoCausal — Internship Presentation

## Slide 1 — Title
**EconoCausal: Dynamic Pricing via Double Machine Learning**  
Data Science & Machine Learning Internship Project

Subtitle: Estimating the causal effect of price on demand.

## Slide 2 — Business Problem
- Price and sales correlation can be misleading.
- Pricing decisions are affected by demand, inventory, costs and competitors.
- Goal: separate observed confounding from the price effect.
- Output: an interpretable price elasticity and controlled what-if scenario.

## Slide 3 — Dataset
- Expanded synthetic retail dataset: 30,000 observations in the upgraded dashboard.
- Product/category and regional information.
- Customer segment.
- Calendar and promotion signals.
- Cost, competitor price, demand index and inventory.
- Advertising spend, rating and competitor count.
- Treatment: `log(price)`.
- Outcome: `log1p(units_sold)`.

## Slide 4 — Causal ML Method
Visual flow:
**Observed business context → ML nuisance models → cross-fitting → residualized price & demand → elasticity estimate → bootstrap uncertainty**

Explain that the DML approach follows the existing project structure: Random-Forest/ensemble nuisance prediction, residualization and partially-linear effect estimation.

## Slide 5 — Exploratory Data Analysis
Recommended visuals:
1. Price vs units sold scatter plot.
2. Units sold by category.
3. Price distribution.
4. Promotion vs non-promotion demand comparison.

Use captions that distinguish descriptive patterns from causal conclusions.

## Slide 6 — Model Evaluation
Show:
- Estimated elasticity
- Standard error
- 95% bootstrap interval
- Outcome RMSE / R²
- Treatment RMSE / R²
- Number of observations

For the original validated run, the supplied results were elasticity **-1.1194**, standard error **0.0258**, 95% interval **[-1.1650, -1.0714]**, outcome R² **0.5456**, treatment R² **0.9817**, on 8,000 observations.

## Slide 7 — Interactive Dashboard
Screenshot/visuals:
- KPI cards
- category and customer segment selectors
- price-change slider
- baseline vs scenario comparison
- model-quality section
- data explorer

## Slide 8 — Scenario Analysis
Example framing:
“If price changes by X%, the model applies the estimated elasticity to calculate an illustrative demand multiplier.”

Do not present this as a guaranteed forecast.

## Slide 9 — Validation & Reproducibility
- Pipeline test: PASS
- Unit tests: 5 PASS in the original supplied validation record
- Static compilation: PASS
- Seed: 42
- Streamlit is listed as a required dependency.

## Slide 10 — Limitations
- Synthetic data.
- Observed confounding only.
- Unmeasured confounding may remain.
- Causal interpretation depends on identification assumptions.
- Dashboard is a demonstration, not a production pricing engine.

## Slide 11 — Key Takeaways
- DML provides a practical framework for estimating treatment effects with rich covariates.
- Cross-fitting reduces overfitting leakage in nuisance prediction.
- Bootstrap uncertainty makes the result more transparent.
- The dashboard connects causal ML methodology to an understandable business scenario.

## Slide 12 — Thank You
**EconoCausal**  
Dynamic Pricing • Causal ML • Python • Scikit-learn • Streamlit
