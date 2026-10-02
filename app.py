from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from src.dml_model import DoubleMLPricingModel

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "dml_model.joblib"
DATA_PATH = ROOT / "data" / "raw" / "pricing_demand.csv"

st.set_page_config(page_title="EconoCausal Dynamic Pricing", page_icon="📈", layout="centered")
st.title("📈 EconoCausal — Dynamic Pricing")
st.caption("Internship demonstration: Double Machine Learning for price elasticity")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.pipeline` first.")
    st.stop()

model = DoubleMLPricingModel.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

metrics = model.metrics_dict()
elasticity = metrics["elasticity"]

st.subheader("Estimated causal price elasticity")
st.metric("Elasticity", f"{elasticity:.3f}")
st.write(
    f"Interpretation: a 1% increase in price is associated with approximately "
    f"{elasticity:.2f}% change in expected demand, under the modeled assumptions."
)
st.info("This is a synthetic internship demonstration. It is not a live pricing recommendation system.")

st.subheader("Scenario")
col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Category", sorted(df["category"].unique()))
    current_price = st.number_input("Current price", min_value=1.0, value=float(df["price"].median()), step=1.0)
    competitor_price = st.number_input("Competitor price", min_value=1.0, value=float(df["competitor_price"].median()), step=1.0)
with col2:
    cost = st.number_input("Unit cost", min_value=0.1, value=float(df["cost"].median()), step=1.0)
    demand_index = st.slider("Demand index", 45, 170, int(df["demand_index"].median()))
    inventory = st.slider("Inventory", 25, 500, int(df["inventory"].median()))

promotion = st.checkbox("Promotion active", value=False)
price_change = st.slider("Price change (%)", -20.0, 20.0, 0.0, 0.5)

new_price = current_price * (1 + price_change / 100)
predicted_demand_multiplier = (new_price / current_price) ** elasticity
baseline = float(df.loc[df["category"] == category, "units_sold"].median())
scenario_demand = baseline * predicted_demand_multiplier
margin_per_unit = new_price - cost
scenario_profit = scenario_demand * margin_per_unit

st.subheader("Scenario output")
c1, c2, c3 = st.columns(3)
c1.metric("Scenario price", f"₹{new_price:,.2f}")
c2.metric("Estimated demand", f"{scenario_demand:,.0f} units")
c3.metric("Illustrative gross contribution", f"₹{scenario_profit:,.0f}")

st.subheader("Model evaluation")
eval_df = pd.DataFrame({
    "Metric": ["Outcome RMSE", "Outcome R²", "Treatment RMSE", "Treatment R²", "95% CI low", "95% CI high"],
    "Value": [
        metrics["outcome_rmse"], metrics["outcome_r2"],
        metrics["treatment_rmse"], metrics["treatment_r2"],
        metrics["ci_low"], metrics["ci_high"]
    ]
})
st.dataframe(eval_df, hide_index=True, use_container_width=True)
