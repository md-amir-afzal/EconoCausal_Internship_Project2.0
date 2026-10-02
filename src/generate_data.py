from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42

def generate_pricing_data(n: int = 8000, seed: int = SEED) -> pd.DataFrame:
    """Generate a reproducible synthetic retail pricing dataset."""
    rng = np.random.default_rng(seed)

    categories = np.array(["Electronics", "Grocery", "Home", "Beauty"])
    regions = np.array(["North", "South", "East", "West"])

    category = rng.choice(categories, n, p=[0.25, 0.35, 0.22, 0.18])
    region = rng.choice(regions, n)
    day_of_week = rng.integers(0, 7, n)
    holiday = rng.binomial(1, 0.10, n)
    promotion = rng.binomial(1, 0.28, n)

    base_cost = np.select(
        [category == "Electronics", category == "Grocery",
         category == "Home", category == "Beauty"],
        [180, 35, 75, 28],
        default=50,
    )
    cost = np.maximum(5, base_cost * rng.lognormal(0, 0.12, n))

    competitor_price = cost * rng.uniform(1.15, 2.00, n)
    demand_index = np.clip(rng.normal(100, 18, n) + 7 * holiday + 5 * promotion, 45, 170)
    inventory = np.clip(rng.normal(220, 65, n), 25, 500)

    # Pricing policy creates confounding: high demand/low inventory tends to
    # increase price, while promotions lower price.
    raw_price = (
        1.28 * cost
        + 0.16 * competitor_price
        + 0.32 * demand_index
        - 0.10 * inventory
        - 8 * promotion
        + rng.normal(0, 10, n)
    )
    price = np.clip(raw_price, cost * 1.03, competitor_price * 1.12)
    price = np.maximum(price, 3.0)

    true_elasticity = -1.25
    category_demand = np.select(
        [category == "Electronics", category == "Grocery",
         category == "Home", category == "Beauty"],
        [42, 70, 55, 62],
        default=50,
    )
    regional_effect = np.select(
        [region == "North", region == "South", region == "East", region == "West"],
        [3, 6, -2, 1],
        default=0,
    )
    expected_log_demand = (
        np.log(category_demand)
        + 0.55 * np.log(demand_index / 100)
        + 0.12 * promotion
        + 0.08 * holiday
        + 0.10 * np.log(competitor_price / (cost + 1))
        + 0.03 * regional_effect
        + true_elasticity * np.log(price / (cost + 1))
    )
    units_sold = np.maximum(
        0,
        np.round(np.exp(expected_log_demand + rng.normal(0, 0.20, n)) - 1)
    ).astype(int)

    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "category": category,
        "region": region,
        "day_of_week": day_of_week,
        "holiday": holiday,
        "promotion": promotion,
        "cost": cost.round(2),
        "competitor_price": competitor_price.round(2),
        "demand_index": demand_index.round(2),
        "inventory": inventory.round(2),
        "price": price.round(2),
        "units_sold": units_sold,
    })

def save_dataset(path: str | Path, n: int = 8000, seed: int = SEED) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_pricing_data(n=n, seed=seed)
    df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    save_dataset("data/raw/pricing_demand.csv")
    print("Generated data/raw/pricing_demand.csv")
