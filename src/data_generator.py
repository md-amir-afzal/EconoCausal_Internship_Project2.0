
import numpy as np
import pandas as pd

def generate_data(n=50000, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=730, freq="D")
    df = pd.DataFrame({
        "date": rng.choice(dates, n),
        "category": rng.choice(["Electronics","Grocery","Home","Beauty","Sports","Apparel"], n),
        "region": rng.choice(["North","South","East","West","Central"], n),
        "customer_segment": rng.choice(["Budget","Mass","Premium"], n, p=[.35,.45,.20]),
    })
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["holiday"] = ((df["month"] == 10) & (df["day_of_week"] >= 4)).astype(int)
    df["promotion"] = rng.binomial(1, .22, n)
    df["competitor_price"] = rng.lognormal(np.log(100), .25, n)
    df["cost"] = df["competitor_price"] * rng.uniform(.52,.82,n)
    df["demand_index"] = rng.normal(100, 18, n) + 8*df["promotion"] + 4*df["weekend"]
    df["inventory"] = rng.lognormal(np.log(180), .65, n)
    df["ad_spend"] = rng.gamma(3.0, 35.0, n)
    df["rating"] = np.clip(rng.normal(4.1, .45, n), 2.5, 5.0)
    df["competitor_count"] = rng.integers(1, 12, n)
    category_factor = df["category"].map({
        "Electronics":1.30,"Grocery":.75,"Home":1.0,
        "Beauty":.85,"Sports":1.05,"Apparel":1.15
    })
    segment_factor = df["customer_segment"].map({"Budget":.85,"Mass":1.0,"Premium":1.22})
    seasonal = 1 + .08*np.sin(2*np.pi*df["month"]/12)
    latent_demand = df["demand_index"]*category_factor*segment_factor*seasonal
    # Price decision depends on observed business variables (confounding).
    df["price"] = np.maximum(
        df["cost"] * (1.25 + .002*latent_demand + .025*df["promotion"])
        + .12*df["competitor_price"] + rng.normal(0, 10, n), 1
    )
    # Controlled benchmark elasticity used only for data-generation validation.
    true_elasticity = -1.25
    log_units = (
        3.7 + true_elasticity*np.log(df["price"])
        + .0035*df["demand_index"] + .004*df["ad_spend"]
        + .18*df["promotion"] + .08*df["weekend"]
        + .06*df["holiday"] + .15*np.log1p(df["inventory"])
        + .10*df["rating"] - .025*df["competitor_count"]
        + rng.normal(0, .23, n)
    )
    df["units_sold"] = np.maximum(np.round(np.expm1(log_units)), 0).astype(int)
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("reports/expanded_retail_data.csv", index=False)
    print(f"Generated {len(df):,} rows")
