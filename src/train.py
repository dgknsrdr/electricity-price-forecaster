"""
Walk-forward backtest + final model training.

Backtest: for every month in the last 12 months, train LightGBM on all data
before that month and forecast that month. This mimics how the model would
have been used in real life (no peeking at the future, retrained monthly).

Outputs:
    reports/backtest_predictions.parquet   actual, model and baseline forecasts per hour
    reports/metrics.csv                    overall MAE / RMSE per method
    reports/metrics_by_month.csv           monthly MAE per method
    reports/feature_importance.csv         average gain importance across folds
    models/lgbm.txt                        final model trained on all data

Usage:
    python src/train.py                    # full run (~3-5 min)
    python src/train.py --no-weather       # check how much weather features help
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

TARGET = "price"
WEATHER_PREFIXES = ("temp_", "wind100", "radiation", "cloud", "wind_daily")

PARAMS = dict(
    objective="l1",            # optimise MAE directly; robust to price spikes
    learning_rate=0.03,
    n_estimators=1500,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    random_state=42,
    verbose=-1,
)


def load(no_weather: bool):
    X = pd.read_parquet(PROCESSED / "features.parquet")
    y = X.pop(TARGET)
    if no_weather:
        X = X.drop(columns=[c for c in X.columns if c.startswith(WEATHER_PREFIXES)])
    return X, y


def fit_predict(X_train, y_train, X_test):
    model = lgb.LGBMRegressor(**PARAMS).fit(X_train, y_train)
    pred = model.predict(X_test)
    # prices can't go below 0 or above the regulatory cap (proxy: 30-day max)
    pred = np.clip(pred, 0, X_test["price_max_30d"].values)
    return model, pred


def walk_forward(X, y, days=365):
    end = X.index.max()
    start = end.normalize() - pd.Timedelta(days=days - 1)
    months = pd.date_range(start.replace(day=1), end, freq="MS")

    preds, importances = [], []
    for m in months:
        fold_start = max(m, start)
        fold_end = m + pd.offsets.MonthBegin(1)
        train = X.index < fold_start
        test = (X.index >= fold_start) & (X.index < fold_end)
        model, pred = fit_predict(X[train], y[train], X[test])
        preds.append(pd.Series(pred, index=X.index[test]))
        importances.append(pd.Series(model.booster_.feature_importance("gain"), index=X.columns))
        mae = np.abs(pred - y[test]).mean()
        print(f"  {fold_start.date()} -> {(fold_end - pd.Timedelta(days=1)).date()}: "
              f"train {train.sum():,} h, test {test.sum():,} h, MAE {mae:,.0f}")

    pred = pd.concat(preds)
    importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return pred, importance / importance.sum()


def score(pred, actual):
    err = pred - actual
    return {
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "MAE_pct_of_mean_price": err.abs().mean() / actual.mean() * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-weather", action="store_true")
    args = parser.parse_args()

    X, y = load(args.no_weather)
    print(f"{len(X):,} rows, {X.shape[1]} features{' (no weather)' if args.no_weather else ''}")
    print("Walk-forward backtest (monthly retraining, last 12 months):")
    pred, importance = walk_forward(X, y)

    bt = pd.DataFrame({
        "actual": y.loc[pred.index],
        "lightgbm": pred,
        "naive_d1": X.loc[pred.index, "price_lag_24"],
        "naive_d7": X.loc[pred.index, "price_lag_168"],
    })
    bt["naive_d1_d7_avg"] = (bt["naive_d1"] + bt["naive_d7"]) / 2

    methods = ["lightgbm", "naive_d1_d7_avg", "naive_d1", "naive_d7"]
    metrics = pd.DataFrame({m: score(bt[m], bt["actual"]) for m in methods}).T
    by_month = pd.DataFrame({
        m: (bt[m] - bt["actual"]).abs().resample("ME").mean() for m in methods
    })

    best_naive = metrics.loc["naive_d1_d7_avg", "MAE"]
    improvement = (1 - metrics.loc["lightgbm", "MAE"] / best_naive) * 100
    print("\nResults:")
    print(metrics.round(1).to_string())
    print(f"\nLightGBM improves MAE by {improvement:.1f}% over the best naive baseline.")

    if args.no_weather:
        return  # experiment only; don't overwrite the main reports

    REPORTS.mkdir(exist_ok=True)
    bt.to_parquet(REPORTS / "backtest_predictions.parquet")
    metrics.round(2).to_csv(REPORTS / "metrics.csv")
    by_month.round(1).to_csv(REPORTS / "metrics_by_month.csv")
    importance.round(4).to_csv(REPORTS / "feature_importance.csv", header=["share_of_gain"])

    print("\nTraining final model on all data...")
    MODELS.mkdir(exist_ok=True)
    final = lgb.LGBMRegressor(**PARAMS).fit(X, y)
    final.booster_.save_model(str(MODELS / "lgbm.txt"))
    (MODELS / "features.json").write_text(json.dumps(list(X.columns), indent=2))
    print(f"saved reports to {REPORTS} and model to {MODELS}")


if __name__ == "__main__":
    main()
