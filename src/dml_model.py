from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

@dataclass
class DMLResult:
    elasticity: float
    standard_error: float
    ci_low: float
    ci_high: float
    outcome_rmse: float
    outcome_r2: float
    treatment_rmse: float
    treatment_r2: float
    n_observations: int

class DoubleMLPricingModel:
    """Cross-fitted partially-linear DML estimator for price elasticity."""

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.result_: DMLResult | None = None
        self.fitted_ = False

    @staticmethod
    def _build_preprocessor(categorical, numerical):
        return ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
                ("num", "passthrough", numerical),
            ],
            remainder="drop",
        )

    def fit(self, df: pd.DataFrame) -> "DoubleMLPricingModel":
        required = {
            "category", "region", "day_of_week", "holiday", "promotion",
            "cost", "competitor_price", "demand_index", "inventory",
            "price", "units_sold"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        work = df.copy()
        work["log_price"] = np.log(work["price"].clip(lower=1e-6))
        work["log_demand"] = np.log1p(work["units_sold"].clip(lower=0))

        x_cols = [
            "category", "region", "day_of_week", "holiday", "promotion",
            "cost", "competitor_price", "demand_index", "inventory"
        ]
        categorical = ["category", "region"]
        numerical = [c for c in x_cols if c not in categorical]

        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        y = work["log_demand"].to_numpy(float)
        d = work["log_price"].to_numpy(float)
        x = work[x_cols]

        y_res = np.zeros(len(work))
        d_res = np.zeros(len(work))
        y_oof = np.zeros(len(work))
        d_oof = np.zeros(len(work))

        for train_idx, test_idx in kf.split(x):
            y_model = Pipeline([
                ("prep", self._build_preprocessor(categorical, numerical)),
                ("model", RandomForestRegressor(
                    n_estimators=180, min_samples_leaf=8,
                    random_state=self.random_state, n_jobs=-1
                )),
            ])
            d_model = Pipeline([
                ("prep", self._build_preprocessor(categorical, numerical)),
                ("model", RandomForestRegressor(
                    n_estimators=180, min_samples_leaf=8,
                    random_state=self.random_state + 1, n_jobs=-1
                )),
            ])
            y_model.fit(x.iloc[train_idx], y[train_idx])
            d_model.fit(x.iloc[train_idx], d[train_idx])
            y_pred = y_model.predict(x.iloc[test_idx])
            d_pred = d_model.predict(x.iloc[test_idx])
            y_oof[test_idx] = y_pred
            d_oof[test_idx] = d_pred
            y_res[test_idx] = y[test_idx] - y_pred
            d_res[test_idx] = d[test_idx] - d_pred

        denom = float(np.sum(d_res ** 2))
        if denom <= 1e-12:
            raise RuntimeError("Residualized treatment has insufficient variation.")
        theta = float(np.sum(d_res * y_res) / denom)

        score = d_res * (y_res - theta * d_res)
        var = float(np.mean(score ** 2) / (np.mean(d_res ** 2) ** 2) / len(df))
        se = float(np.sqrt(max(var, 0)))

        rng = np.random.default_rng(self.random_state)
        bootstrap = []
        n = len(df)
        for _ in range(120):
            idx = rng.integers(0, n, n)
            db = d_res[idx]
            yb = y_res[idx]
            bootstrap.append(float(np.sum(db * yb) / max(np.sum(db ** 2), 1e-12)))
        ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

        self.result_ = DMLResult(
            elasticity=theta,
            standard_error=se,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            outcome_rmse=float(mean_squared_error(y, y_oof) ** 0.5),
            outcome_r2=float(r2_score(y, y_oof)),
            treatment_rmse=float(mean_squared_error(d, d_oof) ** 0.5),
            treatment_r2=float(r2_score(d, d_oof)),
            n_observations=n,
        )
        self.fitted_ = True
        self.oof_predictions_ = pd.DataFrame({
            "actual_log_demand": y,
            "predicted_log_demand": y_oof,
            "actual_log_price": d,
            "predicted_log_price": d_oof,
            "outcome_residual": y_res,
            "treatment_residual": d_res,
        })
        return self

    def save(self, path: str | Path):
        if not self.fitted_:
            raise RuntimeError("Fit the model before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "DoubleMLPricingModel":
        return joblib.load(path)

    def metrics_dict(self):
        if self.result_ is None:
            raise RuntimeError("Model is not fitted.")
        return asdict(self.result_)

def save_metrics(model: DoubleMLPricingModel, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(model.metrics_dict(), f, indent=2)
