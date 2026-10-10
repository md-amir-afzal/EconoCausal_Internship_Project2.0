import sys
import os
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))
from src.data_generator import generate_data
from src.dml_model import estimate_dml
from src.data_validation import normalize_retail_data

st.set_page_config(
    page_title="EconoCausal | Pricing intelligence",
    page_icon=":material/query_stats:",
    layout="wide",
)


@st.cache_data(show_spinner="Preparing the synthetic retail data…", max_entries=2)
def load_data():
    return generate_data(50000, 42)


def load_uploaded_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Estimating the causal pricing model…", max_entries=2)
def train_model(data):
    return estimate_dml(data, folds=5, seed=42)


with st.sidebar:
    st.header("Data and scenario", anchor=False)
    data_source = st.radio("Dataset", ["Synthetic demo", "Upload a CSV"], help="Uploaded rows stay in this app session and are not sent to the AI tutor.")
    uploaded_file = None
    if data_source == "Upload a CSV":
        uploaded_file = st.file_uploader(
            "Retail sales CSV",
            type="csv",
            max_upload_size=25,
            help="Required: price and units_sold. Add available controls such as date, category, promotion, cost, competitor_price, demand_index, inventory, ad_spend, and rating.",
        )

if data_source == "Upload a CSV":
    if uploaded_file is None:
        st.info("Upload a CSV with price, units_sold, and at least one supported control column to start.")
        st.stop()
    try:
        raw_df = load_uploaded_csv(uploaded_file.getvalue())
        df, data_quality = normalize_retail_data(raw_df)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(f"The CSV could not be used: {exc}")
        st.stop()
    is_synthetic = False
    dataset_label = f"Uploaded data · {uploaded_file.name}"
else:
    df, data_quality = normalize_retail_data(load_data())
    is_synthetic = True
    dataset_label = "Synthetic demonstration data"

try:
    metrics = train_model(df)
except (ValueError, TypeError) as exc:
    st.error(f"The model could not estimate a price effect from this dataset: {exc}")
    st.stop()

with st.container(border=True):
    st.title("EconoCausal", icon=":material/query_stats:")
    st.markdown("### Understand the demand impact of a pricing decision")
    st.write("An interactive causal analytics case study for retail pricing. Adjust a business scenario, inspect the expected demand response, and review how the estimate was built.")
    st.badge("Internship portfolio project", icon=":material/workspace_premium:", color="blue")
    st.badge(dataset_label, icon=":material/science:" if is_synthetic else ":material/upload_file:", color="gray" if is_synthetic else "green")

with st.sidebar:
    st.subheader("Cohort", anchor=False)
    category_values = sorted(df["category"].dropna().astype(str).unique()) if "category" in df else []
    region_values = sorted(df["region"].dropna().astype(str).unique()) if "region" in df else []
    segment_values = sorted(df["customer_segment"].dropna().astype(str).unique()) if "customer_segment" in df else []
    category = st.selectbox("Product category", ["All categories", *category_values]) if category_values else "All categories"
    region = st.selectbox("Region", ["All regions", *region_values]) if region_values else "All regions"
    segment = st.selectbox("Customer segment", ["All segments", *segment_values]) if segment_values else "All segments"

scope = df.copy()
if category != "All categories" and "category" in scope:
    scope = scope[scope.category.astype(str) == category]
if segment != "All segments" and "customer_segment" in scope:
    scope = scope[scope.customer_segment.astype(str) == segment]
if region != "All regions" and "region" in scope:
    scope = scope[scope.region.astype(str) == region]
if scope.empty:
    scope = df.copy()

base_price = float(scope.price.median())
base_units = float(scope.units_sold.median())
base_promo_rate = float(scope.promotion.mean()) if "promotion" in scope else 0.0
base_demand = float(scope.demand_index.median()) if "demand_index" in scope else 0.0
base_ad_spend = float(scope.ad_spend.median()) if "ad_spend" in scope else 0.0
base_rating = float(scope.rating.median()) if "rating" in scope else 0.0
base_inventory = float(scope.inventory.median()) if "inventory" in scope else 0.0

with st.sidebar:
    st.subheader("Enter scenario values", anchor=False)
    scenario_price = st.number_input("Selling price (₹)", min_value=0.01, max_value=10_000_000.0, value=round(base_price, 2), step=1.0, format="%.2f", help="Enter the actual price you want to evaluate.")
    if is_synthetic:
        with st.expander("More inputs", icon=":material/tune:"):
            scenario_promo_pct = st.number_input("Promotion rate (%)", min_value=0.0, max_value=100.0, value=round(base_promo_rate * 100, 1), step=1.0)
            demand_limit = max(300.0, float(df.demand_index.quantile(0.99) * 2))
            ad_spend_limit = max(1000.0, float(df.ad_spend.quantile(0.99) * 5))
            scenario_demand = st.number_input("Demand index", min_value=0.0, max_value=demand_limit, value=round(base_demand, 1), step=1.0)
            scenario_ad_spend = st.number_input("Ad spend (₹)", min_value=0.0, max_value=ad_spend_limit, value=round(base_ad_spend, 2), step=10.0)
            scenario_rating = st.number_input("Product rating", min_value=1.0, max_value=5.0, value=round(base_rating, 1), step=0.1, format="%.1f")
            inventory_limit = max(10000.0, float(df.inventory.quantile(0.99) * 10))
            scenario_inventory = st.number_input("Available inventory (units)", min_value=0.0, max_value=inventory_limit, value=round(base_inventory, 1), step=10.0)
    else:
        st.caption("Uploaded-data scenarios change price only; synthetic coefficients are not applied to real business data.")

price_change = (scenario_price / base_price - 1) * 100 if base_price else 0.0
price_factor = (scenario_price / base_price) ** metrics["elasticity"]
other_factor = np.exp(
    0.18 * (scenario_promo_pct / 100 - base_promo_rate)
    + 0.0035 * (scenario_demand - base_demand)
    + 0.004 * (scenario_ad_spend - base_ad_spend)
    + 0.10 * (scenario_rating - base_rating)
    + 0.15 * np.log((1 + scenario_inventory) / (1 + base_inventory))
) if is_synthetic else 1.0
scenario_units = base_units * price_factor * other_factor
base_revenue = base_price * base_units
scenario_revenue = scenario_price * scenario_units
revenue_delta = (scenario_revenue / base_revenue - 1) * 100 if base_revenue else 0
price_low, price_high = scope.price.quantile([0.01, 0.99])

tabs = st.tabs(["Dashboard", "What-if analysis", "Explore data", "Data quality", "AI tutor", "Methodology"])

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
    if "date" in scope and scope["date"].notna().any():
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
    else:
        st.info("Upload a date column to see time trends.")

    st.subheader("Model snapshot")
    with st.container(horizontal=True):
        st.metric("Training observations", f"{metrics['n_observations']:,}", border=True)
        st.metric("Outcome model R²", f"{metrics['outcome_r2']:.3f}", border=True)
        st.metric("Median price in selection", f"₹{base_price:,.2f}", border=True)
        st.metric("Promotion rate in selection", f"{scope.promotion.mean() * 100:.1f}%" if "promotion" in scope else "Not supplied", border=True)
        st.metric("Validation strategy", metrics["validation_strategy"], border=True)
    if is_synthetic:
        st.caption("Scenario demand combines estimated price elasticity with known synthetic-generator relationships for the other controls. This is a learning case study, not a live forecast.")
    else:
        st.caption("This estimate is observational. Its causal interpretation depends on the quality of the supplied confounders and study design.")
    if scenario_price < price_low or scenario_price > price_high:
        st.warning(f"Your price is outside the selected cohort's observed 1st–99th percentile range (₹{price_low:,.2f}–₹{price_high:,.2f}); the estimate is an extrapolation and may be unreliable.")

with tabs[1]:
    st.subheader("Build a business scenario", divider="gray")
    st.write("Enter a selling price in the sidebar. Synthetic demo data also lets you vary promotion, demand, advertising, rating, and inventory; uploaded business data uses a price-only scenario unless those effects are estimated from that data.")
    with st.container(horizontal=True):
        st.metric("Reference units", f"{base_units:,.0f}", "Selected cohort median", border=True)
        st.metric("Scenario units", f"{scenario_units:,.0f}", f"{scenario_units - base_units:+,.0f} vs reference", border=True)
        st.metric("Revenue change", f"{revenue_delta:+.1f}%", "Scenario vs reference", border=True)
    if is_synthetic:
        factors = pd.DataFrame({
            "Input": ["Selling price", "Promotion rate", "Demand index", "Ad spend", "Product rating", "Available inventory"],
            "Cohort reference": [f"₹{base_price:,.2f}", f"{base_promo_rate * 100:.1f}%", f"{base_demand:,.1f}", f"₹{base_ad_spend:,.2f}", f"{base_rating:.1f} / 5", f"{base_inventory:,.1f}"],
            "Your scenario": [f"₹{scenario_price:,.2f}", f"{scenario_promo_pct:.1f}%", f"{scenario_demand:,.1f}", f"₹{scenario_ad_spend:,.2f}", f"{scenario_rating:.1f} / 5", f"{scenario_inventory:,.1f}"],
        })
    else:
        factors = pd.DataFrame({
            "Input": ["Selling price"],
            "Cohort reference": [f"₹{base_price:,.2f}"],
            "Your scenario": [f"₹{scenario_price:,.2f}"],
        })
    st.dataframe(factors, hide_index=True, width="stretch", alt="Cohort reference values and user-entered scenario values")
    if is_synthetic:
        st.info("Price uses the estimated elasticity. Other drivers use known synthetic-generator relationships for this demonstration.")
    else:
        st.info("The user CSV has not been used to estimate separate causal effects for other scenario inputs, so the app changes price only. Validate identification assumptions before treating this as causal evidence.")

with tabs[2]:
    st.subheader("Explore the selected dataset", divider="gray")
    st.write("Filter observations, inspect price and demand patterns, and download a sample for further analysis.")
    filter_columns = [column for column in ["category", "region"] if column in df]
    selections = {}
    if filter_columns:
        filter_cols = st.columns(len(filter_columns))
        for column, container in zip(filter_columns, filter_cols):
            values = sorted(df[column].dropna().astype(str).unique())
            with container:
                selections[column] = st.multiselect(
                    "Categories" if column == "category" else "Regions",
                    values,
                    default=values,
                    key=f"explore_{column}",
                )
    filtered = df.copy()
    for column, values in selections.items():
        filtered = filtered[filtered[column].astype(str).isin(values)]
    st.caption(f"{len(filtered):,} matching rows · showing up to 500 below.")
    if filtered.empty:
        st.info("No rows match those filters. Select at least one value to display data.")
    else:
        chart_left, chart_right = st.columns(2, gap="large")
        group_column = "category" if "category" in filtered else "region" if "region" in filtered else None
        with chart_left:
            if group_column:
                grouped = filtered.groupby(group_column, as_index=False).agg(median_units=("units_sold", "median")).sort_values("median_units", ascending=False)
                st.bar_chart(grouped, x=group_column, y="median_units", alt="Median units sold by available product or region group")
            else:
                st.metric("Median units sold", f"{filtered.units_sold.median():,.0f}")
        with chart_right:
            plot_data = filtered.sample(min(2500, len(filtered)), random_state=42)
            chart_options = {"x": "price", "y": "units_sold", "alt": "Price and units sold for a sample of selected observations"}
            if "category" in plot_data:
                chart_options["color"] = "category"
            st.scatter_chart(plot_data, **chart_options)
        preferred_columns = ["date", "category", "region", "customer_segment", "price", "units_sold", "promotion", "competitor_price", "demand_index", "inventory", "ad_spend", "rating"]
        columns_to_show = [column for column in preferred_columns if column in filtered]
        st.dataframe(filtered[columns_to_show].head(500), width="stretch", height=420, hide_index=True, alt="Filtered sample of the selected retail dataset")
        st.download_button("Download filtered sample CSV", filtered[columns_to_show].head(500).to_csv(index=False), "econocausal_sample.csv", "text/csv", icon=":material/download:")

with tabs[3]:
    st.subheader("Data quality and model diagnostics", divider="gray")
    with st.container(horizontal=True):
        st.metric("Rows", f"{data_quality['rows']:,}", border=True)
        st.metric("Columns", f"{data_quality['columns']:,}", border=True)
        st.metric("Duplicate rows", f"{data_quality['duplicate_rows']:,}", border=True)
        st.metric("Missing cells", f"{data_quality['missing_cells']:,}", border=True)
    st.write(f"**Dataset:** {dataset_label}")
    st.write(f"**Date coverage:** {' to '.join(data_quality['date_span']) if data_quality['date_span'] else 'No usable date column'}")
    st.write(f"**Controls used by the estimator:** {', '.join(metrics['features'])}")
    if data_quality["rows"] < 1000:
        st.warning("This dataset has fewer than 1,000 rows. The elasticity and its interval may be unstable.")
    if not is_synthetic:
        missing_controls = data_quality["controls_missing"]
        missing_preview = ", ".join(missing_controls[:8])
        if len(missing_controls) > 8:
            missing_preview += f", and {len(missing_controls) - 8} more"
        st.warning(
            "Causal estimates can be biased when important pricing confounders are not recorded. "
            f"Supported controls not supplied: {missing_preview or 'none'}."
        )
        if not data_quality["cluster_columns"]:
            st.info("No store or product identifier was found. If rows repeat the same stores or products over time, the reported row bootstrap may understate uncertainty.")
        elif metrics["uncertainty_strategy"] == "row bootstrap":
            st.info(f"Only {metrics['n_clusters']:,} clusters were available; the interval uses row bootstrap because cluster bootstrap needs at least 30 clusters.")
    if data_quality["duplicate_rows"]:
        st.warning("Exact duplicate rows were found. Review them before interpreting the estimate.")
    if data_quality["missing_by_column"]:
        missing_table = pd.DataFrame([
            {"Column": column, "Missing rows": count, "Missing percent": count / len(df) * 100}
            for column, count in data_quality["missing_by_column"].items()
        ]).sort_values("Missing rows", ascending=False)
        st.dataframe(missing_table, hide_index=True, width="stretch", alt="Missing-value counts by column")
        st.caption("Missing control values are median or most-frequent imputed inside each training fold. Missing price or outcome values are rejected.")
    st.subheader("Out-of-fold diagnostics")
    diag = pd.DataFrame({
        "Diagnostic": ["Validation strategy", "Uncertainty method", "Clusters", "Rows in effect estimate", "Total rows", "Controls used", "Residualized price SD", "Elasticity 95% interval"],
        "Value": [metrics["validation_strategy"], metrics["uncertainty_strategy"], f"{metrics['n_clusters']:,}", f"{metrics['n_observations']:,}", f"{metrics['n_rows_total']:,}", str(metrics["n_features"]), f"{metrics['residualized_price_sd']:.4f}", f"[{metrics['ci_low']:.3f}, {metrics['ci_high']:.3f}]"],
    })
    st.dataframe(diag, hide_index=True, width="stretch", alt="Cross-fit sample and price-effect diagnostics")
    if metrics["validation_strategy"] == "random cross-fitting" and "date" in df:
        st.warning("Date data was missing or too sparse for forward month-block validation, so randomized cross-fitting was used.")

with tabs[4]:
    st.subheader("Learn causal pricing with an AI tutor", divider="gray")
    st.write("Ask about elasticity, Double Machine Learning, the charts, or the assumptions behind your current scenario. The tutor receives only the selected cohort summary and scenario metrics shown below—not the row-level dataset.")
    st.caption("This is an optional AI feature. Add your key in `.streamlit/secrets.toml` as `OPENAI_API_KEY` or enter it for this session. API usage may incur charges.")
    try:
        saved_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    except Exception:
        saved_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API key (optional)", value=saved_key, type="password", help="Stored only in this Streamlit session unless configured in secrets or the environment.")
    model_name = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-5-mini"), help="Change this if your API project uses a different model.")
    tutor_context = {
        "cohort": f"{category} / {segment} / {region}",
        "dataset": dataset_label,
        "observations_in_model": metrics["n_observations"],
        "elasticity": round(metrics["elasticity"], 4),
        "elasticity_95_percent_interval": [round(metrics["ci_low"], 4), round(metrics["ci_high"], 4)],
        "price_change_percent": price_change,
        "scenario_price_inr": round(scenario_price, 2),
        "estimated_units": round(scenario_units, 2),
        "estimated_revenue_inr": round(scenario_revenue, 2),
        "validation_strategy": metrics["validation_strategy"],
        "data_quality": {"rows": data_quality["rows"], "duplicate_rows": data_quality["duplicate_rows"], "missing_cells": data_quality["missing_cells"]},
    }
    if is_synthetic:
        tutor_context.update({
            "promotion_rate_percent": round(scenario_promo_pct, 1),
            "demand_index": round(scenario_demand, 1),
            "ad_spend_inr": round(scenario_ad_spend, 2),
            "rating": round(scenario_rating, 1),
            "inventory_units": round(scenario_inventory, 1),
        })
    st.json(tutor_context)
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
                        f"Current aggregate app context only (no uploaded rows): {tutor_context}."
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

with tabs[5]:
    st.subheader("Methodology and responsible use", divider="gray")
    st.write("The project demonstrates a Double Machine Learning workflow: flexible models account for observed confounders before estimating the price effect.")
    st.markdown("""
- **Treatment:** `log(price)`.
- **Outcome:** `log(1 + units sold)`.
- **Controls:** available product, store, region, calendar, promotion, cost, competitor price, demand, inventory, advertising, rating, and competition fields.
- **Cross-fitting:** ExtraTrees nuisance models are fit inside each fold. When usable dates exist, models train on earlier months and predict later months.
- **Effect:** residualized outcome is regressed on residualized treatment. The coefficient is the estimated price elasticity.
- **Uncertainty:** heteroskedasticity-robust standard error plus a bootstrap interval; when enough store/product clusters are supplied, uncertainty uses a cluster bootstrap.
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
    st.warning("Synthetic results are a portfolio demonstration. Uploaded data remains observational: the estimator cannot prove that every confounder is controlled, and the scenario is not a production pricing recommendation.")
