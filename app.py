import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))
from src.data_generator import generate_data
from src.dml_model import estimate_dml

st.set_page_config(
    page_title="EconoCausal | Pricing intelligence",
    page_icon=":material/query_stats:",
    layout="wide",
)


@st.cache_data(show_spinner="Preparing the synthetic retail data…", max_entries=2)
def load_data():
    return generate_data(50000, 42)


@st.cache_data(show_spinner="Estimating the causal pricing model…", max_entries=2)
def train_model(data):
    return estimate_dml(data, folds=5, seed=42)


df = load_data()
metrics = train_model(df)

with st.container(border=True):
    st.title("EconoCausal", icon=":material/query_stats:")
    st.markdown("### Understand the demand impact of a pricing decision")
    st.write("An interactive causal analytics case study for retail pricing. Adjust a business scenario, inspect the expected demand response, and review how the estimate was built.")
    st.badge("Internship portfolio project", icon=":material/workspace_premium:", color="blue")
    st.badge("Synthetic demonstration data", icon=":material/science:", color="gray")

with st.sidebar:
    st.header("Scenario controls", anchor=False)
    st.caption("Choose a cohort, then enter the actual scenario values. Results update with your inputs.")
    category = st.selectbox("Product category", sorted(df.category.unique()))
    region = st.selectbox("Region", ["All regions", *sorted(df.region.unique())])
    segment = st.selectbox("Customer segment", sorted(df.customer_segment.unique()))
    st.caption("All data is synthetic and generated for demonstration.")

scope = df[(df.category == category) & (df.customer_segment == segment)]
if region != "All regions":
    scope = scope[scope.region == region]
if scope.empty:
    scope = df[(df.category == category) & (df.customer_segment == segment)]

base_price = float(scope.price.median())
base_units = float(scope.units_sold.median())
base_promo_rate = float(scope.promotion.mean())
base_demand = float(scope.demand_index.median())
base_ad_spend = float(scope.ad_spend.median())
base_rating = float(scope.rating.median())
base_inventory = float(scope.inventory.median())

with st.sidebar:
    st.subheader("Enter scenario values", anchor=False)
    scenario_price = st.number_input("Selling price (₹)", min_value=0.01, max_value=10_000_000.0, value=round(base_price, 2), step=1.0, format="%.2f", help="Enter the actual price you want to evaluate.")
    with st.expander("More inputs", icon=":material/tune:"):
        scenario_promo_pct = st.number_input("Promotion rate (%)", min_value=0.0, max_value=100.0, value=round(base_promo_rate * 100, 1), step=1.0)
        demand_limit = max(300.0, float(df.demand_index.quantile(0.99) * 2))
        ad_spend_limit = max(1000.0, float(df.ad_spend.quantile(0.99) * 5))
        scenario_demand = st.number_input("Demand index", min_value=0.0, max_value=demand_limit, value=round(base_demand, 1), step=1.0)
        scenario_ad_spend = st.number_input("Ad spend (₹)", min_value=0.0, max_value=ad_spend_limit, value=round(base_ad_spend, 2), step=10.0)
        scenario_rating = st.number_input("Product rating", min_value=1.0, max_value=5.0, value=round(base_rating, 1), step=0.1, format="%.1f")
        inventory_limit = max(10000.0, float(df.inventory.quantile(0.99) * 10))
        scenario_inventory = st.number_input("Available inventory (units)", min_value=0.0, max_value=inventory_limit, value=round(base_inventory, 1), step=10.0)

price_change = (scenario_price / base_price - 1) * 100 if base_price else 0.0
price_factor = (scenario_price / base_price) ** metrics["elasticity"]
other_factor = np.exp(
    0.18 * (scenario_promo_pct / 100 - base_promo_rate)
    + 0.0035 * (scenario_demand - base_demand)
    + 0.004 * (scenario_ad_spend - base_ad_spend)
    + 0.10 * (scenario_rating - base_rating)
    + 0.15 * np.log((1 + scenario_inventory) / (1 + base_inventory))
)
scenario_units = base_units * price_factor * other_factor
base_revenue = base_price * base_units
scenario_revenue = scenario_price * scenario_units
revenue_delta = (scenario_revenue / base_revenue - 1) * 100 if base_revenue else 0
price_low, price_high = scope.price.quantile([0.01, 0.99])

tabs = st.tabs(["Dashboard", "What-if analysis", "Explore data", "AI tutor", "Methodology"])

with tabs[0]:
    st.subheader("Decision overview", divider="gray")
    st.caption(f"{category} · {segment} customers · {region.lower()}")
    with st.container(horizontal=True):
        st.metric("Estimated units", f"{scenario_units:,.0f}", f"{scenario_units - base_units:+,.0f} vs baseline", border=True)
        st.metric("Scenario price", f"₹{scenario_price:,.2f}", f"{price_change:+.1f}% vs baseline", border=True)
        st.metric("Estimated revenue", f"₹{scenario_revenue:,.0f}", f"{revenue_delta:+.1f}% vs baseline", border=True)
        st.metric("Causal price elasticity", f"{metrics['elasticity']:.2f}", f"95% CI {metrics['ci_low']:.2f} to {metrics['ci_high']:.2f}", border=True)

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Demand across price points")
            changes = np.arange(-30, 31, 5)
            curve = pd.DataFrame({
                "Price change": changes,
                "Estimated units": base_units * ((1 + changes / 100) ** metrics["elasticity"]) * other_factor,
            })
            st.line_chart(curve, x="Price change", y="Estimated units", alt="Estimated demand at different price changes")
            st.caption("Other assumptions stay at the values selected in the sidebar.")
    with right:
        with st.container(border=True):
            st.subheader("Baseline vs scenario")
            compare = pd.DataFrame({
                "Units index": [100, scenario_units / base_units * 100 if base_units else 100],
                "Revenue index": [100, scenario_revenue / base_revenue * 100 if base_revenue else 100],
            }, index=["Baseline", "Scenario"])
            st.bar_chart(compare, alt="Baseline and scenario units and revenue, each indexed to baseline equals 100")
            st.caption("Each measure is indexed separately; baseline = 100.")

    st.subheader("Retail performance context", divider="gray")
    monthly = scope.assign(month_start=scope.date.dt.to_period("M").dt.to_timestamp()).groupby("month_start").agg(
        median_price=("price", "median"), median_units=("units_sold", "median"),
    ).reset_index().tail(18)
    context_left, context_right = st.columns(2, gap="large")
    with context_left:
        with st.container(border=True):
            st.markdown("**Median selling price**")
            st.line_chart(monthly, x="month_start", y="median_price", alt="Monthly median selling price in the selected cohort")
    with context_right:
        with st.container(border=True):
            st.markdown("**Median units sold**")
            st.line_chart(monthly, x="month_start", y="median_units", alt="Monthly median units sold in the selected cohort")

    st.subheader("Model snapshot")
    with st.container(horizontal=True):
        st.metric("Training observations", f"{metrics['n_observations']:,}", border=True)
        st.metric("Outcome model R²", f"{metrics['outcome_r2']:.3f}", border=True)
        st.metric("Median price in selection", f"₹{base_price:,.2f}", border=True)
        st.metric("Promotion rate in selection", f"{scope.promotion.mean() * 100:.1f}%", border=True)
    st.caption("Scenario demand combines the estimated causal price elasticity with known coefficients from this synthetic data generator for the other controls. This is a learning case study, not a live forecast or pricing recommendation.")
    if scenario_price < price_low or scenario_price > price_high:
        st.warning(f"Your price is outside the selected cohort's observed 1st–99th percentile range (₹{price_low:,.2f}–₹{price_high:,.2f}); the estimate is an extrapolation and may be unreliable.")

with tabs[1]:
    st.subheader("Build a business scenario", divider="gray")
    st.write("Enter an actual selling price and, if needed, actual promotion, demand, advertising, rating, and inventory values in the sidebar. All dashboard metrics and charts recalculate from your inputs.")
    with st.container(horizontal=True):
        st.metric("Reference units", f"{base_units:,.0f}", "Selected cohort median", border=True)
        st.metric("Scenario units", f"{scenario_units:,.0f}", f"{scenario_units - base_units:+,.0f} vs reference", border=True)
        st.metric("Revenue change", f"{revenue_delta:+.1f}%", "Scenario vs reference", border=True)
    factors = pd.DataFrame({
        "Input": ["Selling price", "Promotion rate", "Demand index", "Ad spend", "Product rating", "Available inventory"],
        "Cohort reference": [f"₹{base_price:,.2f}", f"{base_promo_rate * 100:.1f}%", f"{base_demand:,.1f}", f"₹{base_ad_spend:,.2f}", f"{base_rating:.1f} / 5", f"{base_inventory:,.1f}"],
        "Your scenario": [f"₹{scenario_price:,.2f}", f"{scenario_promo_pct:.1f}%", f"{scenario_demand:,.1f}", f"₹{scenario_ad_spend:,.2f}", f"{scenario_rating:.1f} / 5", f"{scenario_inventory:,.1f}"],
    })
    st.dataframe(factors, hide_index=True, width="stretch", alt="Cohort reference values and user-entered scenario values")
    st.info("Price uses the model's estimated elasticity. Other driver changes use known relationships from the synthetic data generator. This makes the controls transparent for a portfolio demonstration; validate every relationship on real company data before using it for decisions.")

with tabs[2]:
    st.subheader("Explore the synthetic retail dataset", divider="gray")
    st.write("Filter the generated retail observations, inspect how price and demand move together, and download a sample for further analysis.")
    f1, f2 = st.columns(2)
    with f1:
        categories = st.multiselect("Categories", sorted(df.category.unique()), default=sorted(df.category.unique()))
    with f2:
        regions = st.multiselect("Regions", sorted(df.region.unique()), default=sorted(df.region.unique()))
    filtered = df[df.category.isin(categories) & df.region.isin(regions)]
    st.caption(f"Showing up to 500 of {len(filtered):,} matching rows.")
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        by_category = filtered.groupby("category", as_index=False).agg(median_units=("units_sold", "median")).sort_values("median_units", ascending=False)
        st.bar_chart(by_category, x="category", y="median_units", alt="Median units sold by selected product category")
    with chart_right:
        st.scatter_chart(filtered.sample(min(2500, len(filtered)), random_state=42), x="price", y="units_sold", color="category", alt="Price and units sold for a sample of filtered retail observations")
    cols = ["date", "category", "region", "customer_segment", "price", "units_sold", "promotion", "competitor_price", "demand_index", "inventory", "ad_spend", "rating"]
    st.dataframe(filtered[cols].head(500), width="stretch", height=420, hide_index=True, alt="Filtered sample of synthetic retail data")
    st.download_button("Download filtered sample CSV", filtered[cols].head(500).to_csv(index=False), "econocausal_sample.csv", "text/csv", icon=":material/download:")

with tabs[3]:
    st.subheader("Learn causal pricing with an AI tutor", divider="gray")
    st.write("Ask about elasticity, Double Machine Learning, the charts, or the assumptions behind your current scenario. The tutor receives only the selected cohort summary and scenario metrics shown below—not the row-level dataset.")
    st.caption("This is an optional AI feature. Add your key in `.streamlit/secrets.toml` as `OPENAI_API_KEY` or enter it for this session. API usage may incur charges.")
    try:
        saved_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    except Exception:
        saved_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API key (optional)", value=saved_key, type="password", help="Stored only in this Streamlit session unless configured in secrets or the environment.")
    model_name = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-5-mini"), help="Change this if your API project uses a different model.")
    st.json({
        "cohort": f"{category} / {segment} / {region}",
        "observations_in_model": metrics["n_observations"],
        "elasticity": round(metrics["elasticity"], 4),
        "elasticity_95_percent_interval": [round(metrics["ci_low"], 4), round(metrics["ci_high"], 4)],
        "price_change_percent": price_change,
        "scenario_price_inr": round(scenario_price, 2),
        "promotion_rate_percent": round(scenario_promo_pct, 1),
        "demand_index": round(scenario_demand, 1),
        "ad_spend_inr": round(scenario_ad_spend, 2),
        "rating": round(scenario_rating, 1),
        "inventory_units": round(scenario_inventory, 1),
        "estimated_units": round(scenario_units, 2),
        "estimated_revenue_inr": round(scenario_revenue, 2),
        "data_type": "synthetic demonstration data",
    })
    if "tutor_messages" not in st.session_state:
        st.session_state.tutor_messages = []
    for message in st.session_state.tutor_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if question := st.chat_input("Ask a question about the model or scenario"):
        st.session_state.tutor_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        if not api_key:
            answer = "Add an OpenAI API key above to enable AI answers. You can still explore the Methodology tab for a concise explanation of the estimator."
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            try:
                from openai import OpenAI
                conversation = st.session_state.tutor_messages[-12:]
                response = OpenAI(api_key=api_key).responses.create(
                    model=model_name.strip() or "gpt-5-mini",
                    instructions=(
                        "You are a patient economics and causal inference tutor for a beginner. "
                        "Explain concepts in plain language, use the supplied app context, and "
                        "ask a short check-for-understanding question when useful. Clearly say "
                        "that this project's data is synthetic and its estimates are educational, "
                        "not real-world evidence or pricing advice. Never claim this app trains "
                        "or updates the underlying language model from the user's chat.\n\n"
                        f"Current app context (aggregate only): cohort={category}/{segment}/{region}; "
                        f"model_rows={metrics['n_observations']}; elasticity={metrics['elasticity']:.4f}; "
                        f"elasticity_95_ci=[{metrics['ci_low']:.4f}, {metrics['ci_high']:.4f}]; "
                        f"scenario_price_inr={scenario_price:.2f}; promotion_rate_percent={scenario_promo_pct:.1f}; "
                        f"demand_index={scenario_demand:.1f}; ad_spend_inr={scenario_ad_spend:.2f}; "
                        f"rating={scenario_rating:.1f}; inventory_units={scenario_inventory:.1f}; "
                        f"price_change_percent={price_change:.2f}; estimated_units={scenario_units:.2f}; "
                        f"estimated_revenue_inr={scenario_revenue:.2f}."
                    ),
                    input=conversation,
                    store=False,
                )
                answer = response.output_text or "I couldn't produce a text answer. Please try again."
                st.session_state.tutor_messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
            except ImportError:
                st.error("The OpenAI package is not installed. Install the project requirements and restart the app.")
            except Exception:
                st.error("The AI tutor request failed. Check the key, model name, network connection, and API account access, then try again.")
    if st.session_state.tutor_messages and st.button("Clear tutor conversation"):
        st.session_state.tutor_messages = []
        st.rerun()

with tabs[4]:
    st.subheader("Methodology and responsible use", divider="gray")
    st.write("The project demonstrates a Double Machine Learning workflow: flexible models account for observed confounders before estimating the price effect.")
    st.markdown("""
**Treatment:** log(price)  
**Outcome:** log(1 + units sold)  
**Controls:** product, region, segment, calendar, promotion, cost, competitor price, demand, inventory, advertising, rating, and competition.  
**Cross-fitting:** ExtraTrees nuisance models predict treatment and outcome out of fold.  
**Effect:** residualized outcome is regressed on residualized treatment. The coefficient is the estimated price elasticity.  
**Uncertainty:** bootstrap interval around the estimate.
""")
    st.subheader("Model evaluation")
    evaluation = pd.DataFrame({
        "Metric": ["Elasticity", "Standard error", "95% CI low", "95% CI high", "Outcome RMSE", "Outcome R²", "Treatment RMSE", "Treatment R²"],
        "Value": [metrics[k] for k in ["elasticity", "standard_error", "ci_low", "ci_high", "outcome_rmse", "outcome_r2", "treatment_rmse", "treatment_r2"]],
    })
    st.dataframe(evaluation, hide_index=True, width="stretch", alt="Causal model evaluation metrics")
    st.subheader("How to present this project")
    st.markdown("""
- **Business question:** How does price relate to product demand after accounting for observed business drivers?
- **My approach:** Generate a reproducible retail dataset, cross-fit nuisance models, estimate a causal elasticity, and expose assumptions through an interactive scenario dashboard.
- **What I would do next at a company:** Replace synthetic data with validated sales, promotion, inventory, cost, competitor, and calendar data; define the intervention; test robustness; and monitor results through an experiment.
""")
    st.warning("The dataset is synthetic, and the model does not establish that every real-world confounder has been controlled. The scenario is a portfolio demonstration, not a production pricing system.")
