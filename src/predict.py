"""
Daily forecast job: run on the morning of day D-1 to forecast all 24 hours of day D.

What it does:
  1. Downloads the last ~60 days of EPİAŞ data (prices up to today, realised
     consumption / generation up to yesterday) and recent weather + tomorrow's
     weather forecast (Open-Meteo forecast API).
  2. Builds exactly the same features as in training (src/features.py).
  3. Forecasts tomorrow with the saved model (models/lgbm.txt).
  4. Appends the forecast to forecasts/forecasts.csv and fills in actual prices
     for earlier forecasts, so the file doubles as a live track record.

Usage:
    python src/predict.py                     # forecast tomorrow
    python src/predict.py --date 2026-11-05   # forecast a specific day (needs D-1 prices)
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import merge_raw          # noqa: E402
from features import build_features          # noqa: E402
from fetch_data import fetch_eptr, fetch_weather_forecast, make_client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FORECASTS = ROOT / "forecasts" / "forecasts.csv"
TZ = ZoneInfo("Europe/Istanbul")
HISTORY_DAYS = 60          # enough for the 30-day rolling features and 14-day lags
COLUMNS = ["target_ts", "forecast", "naive", "actual", "created_at"]


def get_recent_data(target: date):
    """Download what is known on the morning of target-1."""
    today = target - timedelta(days=1)
    start = target - timedelta(days=HISTORY_DAYS)
    client = make_client()
    print(f"Downloading EPİAŞ data {start} -> {today}")
    ptf = fetch_eptr(client, "mcp", start, today)
    cons = fetch_eptr(client, "rt-cons", start, today - timedelta(days=1))
    gen = fetch_eptr(client, "rt-gen", start, today - timedelta(days=1))
    days_back = (date.today() - start).days + 1
    print(f"Downloading weather (past {days_back} days + forecast)")
    weather = fetch_weather_forecast(past_days=min(days_back, 92), forecast_days=3)
    return ptf, cons, gen, weather


def make_forecast(ptf, cons, gen, weather, target: date) -> pd.DataFrame:
    start = pd.Timestamp(target - timedelta(days=HISTORY_DAYS))
    end = pd.Timestamp(target) + pd.Timedelta(hours=23)
    index = pd.date_range(start, end, freq="h")
    hourly = merge_raw(ptf, cons, gen, weather, index=index)

    X = build_features(hourly)
    rows = X.loc[str(target)].drop(columns="price")

    feature_names = json.loads((MODELS / "features.json").read_text())
    rows = rows[feature_names]
    missing = rows.columns[rows.isna().any()].tolist()
    if missing:
        raise RuntimeError(
            f"Missing inputs for {target}: {missing}. "
            "Usually this means yesterday's EPİAŞ data is not published yet; try again later."
        )

    model = lgb.Booster(model_file=str(MODELS / "lgbm.txt"))
    pred = np.clip(model.predict(rows), 0, rows["price_max_30d"].values)
    naive = (rows["price_lag_24"] + rows["price_lag_168"]) / 2

    return pd.DataFrame({
        "target_ts": rows.index,
        "forecast": pred.round(2),
        "naive": naive.values.round(2),
        "actual": np.nan,
        "created_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),
    })


def actual_prices(ptf: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(ptf["date"].str[:19])
    return pd.Series(ptf["price"].values, index=ts).groupby(level=0).last()


def update_history(new: pd.DataFrame, actuals: pd.Series) -> pd.DataFrame:
    if FORECASTS.exists():
        hist = pd.read_csv(FORECASTS, parse_dates=["target_ts"])
        hist = pd.concat([hist, new], ignore_index=True)
    else:
        hist = new
    # one forecast per hour: keep the latest run
    hist = hist.drop_duplicates("target_ts", keep="last").sort_values("target_ts")
    known = hist["target_ts"].map(actuals)
    hist["actual"] = known.combine_first(hist["actual"])
    FORECASTS.parent.mkdir(exist_ok=True)
    hist[COLUMNS].to_csv(FORECASTS, index=False)
    return hist


def report(hist: pd.DataFrame, target: date):
    done = hist.dropna(subset=["actual"])
    if len(done):
        last_day = done["target_ts"].dt.date.max()
        d = done[done["target_ts"].dt.date == last_day]
        print(f"Latest scored day {last_day}: model MAE {np.abs(d.forecast - d.actual).mean():.0f}, "
              f"naive MAE {np.abs(d.naive - d.actual).mean():.0f} TL/MWh")
        print(f"Live track record ({done['target_ts'].dt.date.nunique()} days): "
              f"model MAE {np.abs(done.forecast - done.actual).mean():.0f}, "
              f"naive MAE {np.abs(done.naive - done.actual).mean():.0f} TL/MWh")
    t = hist[hist["target_ts"].dt.date == target]
    print(f"\nForecast for {target} (TL/MWh): mean {t.forecast.mean():.0f}, "
          f"min {t.forecast.min():.0f} at {t.loc[t.forecast.idxmin(), 'target_ts']:%H:%M}, "
          f"max {t.forecast.max():.0f} at {t.loc[t.forecast.idxmax(), 'target_ts']:%H:%M}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="day to forecast (default: tomorrow, Istanbul time)")
    args = parser.parse_args()
    target = (date.fromisoformat(args.date) if args.date
              else datetime.now(TZ).date() + timedelta(days=1))

    ptf, cons, gen, weather = get_recent_data(target)
    new = make_forecast(ptf, cons, gen, weather, target)
    hist = update_history(new, actual_prices(ptf))
    report(hist, target)
    print(f"saved to {FORECASTS}")


if __name__ == "__main__":
    main()
