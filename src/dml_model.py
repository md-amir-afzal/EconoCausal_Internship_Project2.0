"""Cross-fitted Double Machine Learning for retail price elasticity."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURES = [
    "category", "region", "customer_segment", "product_id", "sku_id", "store_id",
    "day_of_week", "month", "quarter",
    "weekend", "holiday", "promotion", "cost", "competitor_price",
    "demand_index", "inventory", "ad_spend", "rating", "competitor_count",
]
CATEGORICAL = {"category", "region", "customer_segment", "product_id", "sku_id", "store_id"}


def _make_pipeline(categorical, numeric, seed):
    transforms = []
    if categorical:
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])
        transforms.append(("categorical", categorical_pipe, categorical))
    if numeric:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ])
        transforms.append(("numeric", numeric_pipe, numeric))
    preprocessor = ColumnTransformer(transforms, remainder="drop")
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=8,
            max_features=1.0,
            random_state=seed,
            n_jobs=1,
        )),
    ])


def _crossfit_splits(work, folds, seed):
    """Use forward month blocks when complete dates exist, otherwise random folds."""
    if "date" in work.columns:
        dates = pd.to_datetime(work["date"], errors="coerce")
        periods = dates.dt.to_period("M")
        unique_periods = pd.Index(periods.dropna().unique()).sort_values()
        if dates.notna().all() and len(unique_periods) > folds:
            splitter = TimeSeriesSplit(n_splits=folds)
            splits = []
            for train_month_idx, test_month_idx in splitter.split(unique_periods):
                train_months = unique_periods[train_month_idx]
                test_months = unique_periods[test_month_idx]
                train_idx = np.flatnonzero(periods.isin(train_months).to_numpy())
                test_idx = np.flatnonzero(periods.isin(test_months).to_numpy())
                splits.append((train_idx, test_idx))
            return splits, "forward month-block cross-fitting"

    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(work)), "random cross-fitting"


def estimate_dml(df, folds=5, seed=42):
    """Estimate price elasticity with date-aware cross-fitting and robust uncertainty.

    If complete dates span enough months, nuisance models train only on earlier
    months and predict later months. Rows in the initial warm-up period are not
    used in the final effect estimate. Without usable dates, shuffled K-folds
    are used and the result is labeled accordingly.
    """
    if folds < 2 or folds >= len(df):
        raise ValueError("folds must be at least 2 and fewer than the number of rows")
    required = {"price", "units_sold"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if df[["price", "units_sold"]].isna().any().any():
        raise ValueError("price and units_sold must not contain missing values")
    if (df["price"] <= 0).any() or (df["units_sold"] < 0).any():
        raise ValueError("price must be positive and units_sold cannot be negative")

    work = df.copy().reset_index(drop=True)
    if "date" in work:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        if work["date"].notna().any():
            if "day_of_week" not in work:
                work["day_of_week"] = work["date"].dt.dayofweek
            if "month" not in work:
                work["month"] = work["date"].dt.month
            if "quarter" not in work:
                work["quarter"] = work["date"].dt.quarter
            if "weekend" not in work:
                work["weekend"] = (work["date"].dt.dayofweek >= 5).astype(float)

    features = [column for column in FEATURES if column in work.columns and not work[column].isna().all()]
    if not features:
        raise ValueError("At least one supported control column is required for DML estimation")
    if work[features].isna().all(axis=0).all():
        raise ValueError("All candidate control values are missing")

    y = np.log1p(work["units_sold"].to_numpy(dtype=float))
    d = np.log(work["price"].to_numpy(dtype=float))
    X = work[features]
    categorical = [column for column in features if column in CATEGORICAL]
    numeric = [column for column in features if column not in CATEGORICAL]

    splits, validation_strategy = _crossfit_splits(work, folds, seed)
    yhat = np.full(len(work), np.nan)
    dhat = np.full(len(work), np.nan)
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        y_model = _make_pipeline(categorical, numeric, seed + fold_idx)
        d_model = _make_pipeline(categorical, numeric, seed + folds + fold_idx)
        y_model.fit(X.iloc[train_idx], y[train_idx])
        d_model.fit(X.iloc[train_idx], d[train_idx])
        yhat[test_idx] = y_model.predict(X.iloc[test_idx])
        dhat[test_idx] = d_model.predict(X.iloc[test_idx])

    scored = np.isfinite(yhat) & np.isfinite(dhat)
    if scored.sum() < max(30, folds * 5):
        raise ValueError("Too few out-of-fold rows to estimate a stable effect; add more data or use fewer folds")
    y_scored, d_scored = y[scored], d[scored]
    ry, rd = y_scored - yhat[scored], d_scored - dhat[scored]
    denominator = float(np.dot(rd, rd))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        raise ValueError("Residualized price has no usable variation; elasticity cannot be estimated")

    elasticity = float(np.dot(rd, ry) / denominator)
    residual = ry - elasticity * rd
    n_scored = len(ry)
    # HC1 heteroskedasticity-robust standard error for the residualized slope.
    se = float(np.sqrt((n_scored / max(n_scored - 1, 1)) * np.dot(rd**2, residual**2) / denominator**2))

    rng = np.random.default_rng(seed)
    bootstrap = []
    row_indices = np.arange(n_scored)
    cluster_columns = [column for column in ("store_id", "product_id", "sku_id") if column in work]
    cluster_column = cluster_columns[0] if cluster_columns else None
    cluster_labels = work.loc[scored, cluster_column].astype("string").fillna("__missing__").to_numpy() if cluster_column else None
    clusters = np.unique(cluster_labels) if cluster_labels is not None else None
    use_cluster_bootstrap = clusters is not None and len(clusters) >= 30
    cluster_to_rows = {
        cluster: np.flatnonzero(cluster_labels == cluster)
        for cluster in clusters
    } if use_cluster_bootstrap else None
    for _ in range(300):
        if use_cluster_bootstrap:
            drawn_clusters = rng.choice(clusters, len(clusters), replace=True)
            sample = np.concatenate([cluster_to_rows[cluster] for cluster in drawn_clusters])
        else:
            sample = rng.choice(row_indices, n_scored, replace=True)
        rd_sample, ry_sample = rd[sample], ry[sample]
        boot_denominator = float(np.dot(rd_sample, rd_sample))
        if boot_denominator > np.finfo(float).eps:
            bootstrap.append(float(np.dot(rd_sample, ry_sample) / boot_denominator))
    if len(bootstrap) < 100:
        raise ValueError("Bootstrap uncertainty was unstable; check whether price varies enough in this dataset")
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

    return {
        "elasticity": elasticity,
        "standard_error": se,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "outcome_rmse": float(np.sqrt(mean_squared_error(y_scored, yhat[scored]))),
        "outcome_r2": float(r2_score(y_scored, yhat[scored])),
        "treatment_rmse": float(np.sqrt(mean_squared_error(d_scored, dhat[scored]))),
        "treatment_r2": float(r2_score(d_scored, dhat[scored])),
        "n_observations": int(n_scored),
        "n_rows_total": int(len(work)),
        "n_features": int(len(features)),
        "features": features,
        "validation_strategy": validation_strategy,
        "uncertainty_strategy": f"cluster bootstrap by {cluster_column}" if use_cluster_bootstrap else "row bootstrap",
        "n_clusters": int(len(clusters)) if clusters is not None else 0,
        "residualized_price_sd": float(np.std(rd, ddof=1)),
    }
