
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def estimate_dml(df, folds=5, seed=42):
    if folds < 2 or folds > len(df):
        raise ValueError("folds must be at least 2 and no greater than the number of rows")
    required = {"price", "units_sold"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if df[["price", "units_sold"]].isna().any().any() or (df["price"] <= 0).any() or (df["units_sold"] < 0).any():
        raise ValueError("price must be positive and price/units_sold must not contain missing values")
    work = df.copy()
    work["log_price"] = np.log(work["price"])
    work["log_units"] = np.log1p(work["units_sold"])
    y = work["log_units"].to_numpy()
    d = work["log_price"].to_numpy()

    features = [
        "category","region","customer_segment","day_of_week","month","quarter",
        "weekend","holiday","promotion","cost","competitor_price",
        "demand_index","inventory","ad_spend","rating","competitor_count"
    ]
    X = work[features]
    cat = ["category","region","customer_segment"]
    num = [c for c in features if c not in cat]
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
                             remainder="passthrough")
    Xv = prep.fit_transform(X)

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    yhat = np.zeros(len(df)); dhat = np.zeros(len(df))
    for tr, te in kf.split(Xv):
        # Keep fitting in-process so the project also runs in restricted Windows
        # environments where joblib worker pools cannot create named pipes.
        ym = ExtraTreesRegressor(n_estimators=80, min_samples_leaf=8, random_state=seed, n_jobs=1)
        dm = ExtraTreesRegressor(n_estimators=80, min_samples_leaf=8, random_state=seed+1, n_jobs=1)
        ym.fit(Xv[tr], y[tr]); dm.fit(Xv[tr], d[tr])
        yhat[te] = ym.predict(Xv[te]); dhat[te] = dm.predict(Xv[te])

    ry, rd = y-yhat, d-dhat
    denominator = float(np.dot(rd, rd))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        raise ValueError("Residualized price has no usable variation; the elasticity cannot be estimated")
    elasticity = float(np.dot(rd, ry)/denominator)
    residual = ry - elasticity * rd
    # HC1 heteroskedasticity-robust standard error for the residualized
    # one-regressor estimate.
    se = float(np.sqrt((len(df) / max(len(df) - 1, 1)) * np.dot(rd**2, residual**2) / denominator**2))
    rng = np.random.default_rng(seed)
    boots = []
    idx = np.arange(len(df))
    for _ in range(250):
        b = rng.choice(idx, len(idx), replace=True)
        rdb, ryb = rd[b], ry[b]
        boots.append(np.dot(rdb, ryb)/np.dot(rdb, rdb))
    ci = np.quantile(boots, [.025,.975])
    return {
        "elasticity": elasticity, "standard_error": se,
        "ci_low": float(ci[0]), "ci_high": float(ci[1]),
        "outcome_rmse": float(np.sqrt(mean_squared_error(y,yhat))),
        "outcome_r2": float(r2_score(y,yhat)),
        "treatment_rmse": float(np.sqrt(mean_squared_error(d,dhat))),
        "treatment_r2": float(r2_score(d,dhat)),
        "n_observations": int(len(df))
    }
