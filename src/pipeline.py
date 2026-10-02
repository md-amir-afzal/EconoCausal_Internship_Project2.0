from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .generate_data import save_dataset
from .dml_model import DoubleMLPricingModel, save_metrics

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pricing_demand.csv"
MODEL = ROOT / "models" / "dml_model.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
PRED = REPORTS / "predictions.csv"
METRICS = REPORTS / "model_metrics.json"

def make_eda(df: pd.DataFrame):
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(df["price"], df["units_sold"], s=7, alpha=0.25)
    plt.xlabel("Price")
    plt.ylabel("Units sold")
    plt.title("Price vs Units Sold")
    plt.tight_layout()
    plt.savefig(FIGURES / "price_vs_demand.png", dpi=160)
    plt.close()

    grouped = df.groupby("category", as_index=False)["units_sold"].mean()
    plt.figure(figsize=(8, 5))
    plt.bar(grouped["category"], grouped["units_sold"])
    plt.ylabel("Average units sold")
    plt.title("Average Demand by Category")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES / "demand_by_category.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["price"], bins=40)
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.title("Price Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES / "price_distribution.png", dpi=160)
    plt.close()

def run():
    if not RAW.exists():
        df = save_dataset(RAW, n=8000, seed=42)
    else:
        df = pd.read_csv(RAW)

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().reset_index(drop=True)
    make_eda(df)

    model = DoubleMLPricingModel(n_splits=5, random_state=42).fit(df)
    model.save(MODEL)
    save_metrics(model, METRICS)
    model.oof_predictions_.to_csv(PRED, index=False)

    print("\n=== EconoCausal DML Results ===")
    for k, v in model.metrics_dict().items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\nSaved model: {MODEL}")
    print(f"Saved metrics: {METRICS}")
    print(f"Saved predictions: {PRED}")
    print(f"Saved figures: {FIGURES}")
    return model

if __name__ == "__main__":
    run()
