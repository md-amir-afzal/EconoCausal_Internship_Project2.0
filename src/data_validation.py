"""CSV intake and quality checks for the EconoCausal dashboard."""

import re

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "category",
    "region",
    "customer_segment",
    "product_id",
    "sku_id",
    "store_id",
    "day_of_week",
    "month",
    "quarter",
    "weekend",
    "holiday",
    "promotion",
    "cost",
    "competitor_price",
    "demand_index",
    "inventory",
    "ad_spend",
    "rating",
    "competitor_count",
)

ALIASES = {
    "unit_price": "price",
    "selling_price": "price",
    "quantity": "units_sold",
    "units": "units_sold",
    "order_date": "date",
    "transaction_date": "date",
    "promo": "promotion",
    "adspend": "ad_spend",
    "store": "store_id",
    "sku": "sku_id",
    "product": "product_id",
}


def normalize_retail_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize supported names and reject invalid price/outcome data."""
    if data.empty:
        raise ValueError("The CSV has no data rows.")

    clean = data.copy()
    normalized = [
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        for column in clean.columns
    ]
    clean.columns = [ALIASES.get(name, name) for name in normalized]
    if len(set(clean.columns)) != len(clean.columns):
        raise ValueError("Two or more columns map to the same name. Rename them so every column is unique.")

    required = {"price", "units_sold"}
    missing = required.difference(clean.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing)) +
            ". Include price and units_sold (or supported aliases such as unit_price and quantity)."
        )

    for column in ("price", "units_sold"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    invalid_outcomes = clean[["price", "units_sold"]].isna().any(axis=1)
    if invalid_outcomes.any():
        raise ValueError(f"{int(invalid_outcomes.sum()):,} rows have missing or non-numeric price/units_sold values.")
    if not np.isfinite(clean[["price", "units_sold"]].to_numpy(dtype=float)).all():
        raise ValueError("price and units_sold must contain finite numbers.")
    if (clean["price"] <= 0).any():
        raise ValueError("Every price must be greater than zero.")
    if (clean["units_sold"] < 0).any():
        raise ValueError("units_sold cannot be negative.")

    if "date" in clean.columns:
        clean["date"] = pd.to_datetime(clean["date"], errors="coerce", utc=True).dt.tz_convert(None)
        clean["day_of_week"] = clean["date"].dt.dayofweek
        clean["month"] = clean["date"].dt.month
        clean["quarter"] = clean["date"].dt.quarter
        clean["weekend"] = (clean["day_of_week"] >= 5).astype(float)

    controls = [column for column in FEATURE_COLUMNS if column in clean.columns]
    if not controls:
        raise ValueError(
            "Add at least one supported control column (for example date, category, promotion, cost, "
            "competitor_price, demand_index, inventory, or ad_spend) so the model can adjust for context."
        )

    for column in controls:
        if column not in {"category", "region", "customer_segment"}:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

    date_span = None
    if "date" in clean.columns and clean["date"].notna().any():
        date_span = [clean["date"].min().date().isoformat(), clean["date"].max().date().isoformat()]

    report = {
        "rows": int(len(clean)),
        "columns": int(len(clean.columns)),
        "duplicate_rows": int(clean.duplicated().sum()),
        "missing_cells": int(clean.isna().sum().sum()),
        "missing_by_column": {str(k): int(v) for k, v in clean.isna().sum().items() if v},
        "date_span": date_span,
        "controls_used": controls,
        "controls_missing": [column for column in FEATURE_COLUMNS if column not in clean.columns],
        "cluster_columns": [column for column in ("store_id", "product_id", "sku_id") if column in clean.columns],
    }
    return clean, report
