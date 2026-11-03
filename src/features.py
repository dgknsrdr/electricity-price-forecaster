"""
Feature engineering for the day-ahead price forecast.

Forecast setting (see notebooks/01_eda.ipynb, section 5):
    We forecast all 24 hours of day D on the morning of D-1 (~10:00).
    Known at that time:
      - PTF prices up to the end of D-1 (published on D-2 afternoon)
      - realised consumption / generation up to the morning of D-1
        -> we only use them up to the end of D-2 (lag >= 48h for the same hour)
      - a weather forecast for D
        -> in the backtest we use realised weather for D as a stand-in
           ("perfect weather forecast"); this is optimistic and documented.

Every feature below is built so that, for a row at hour t on day D,
it only uses information available on the morning of D-1.

Usage:
    python src/features.py      # writes data/processed/features.parquet
"""

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

CITIES = ["istanbul", "ankara", "izmir", "canakkale", "konya"]
TARGET = "price"


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    years = range(idx.year.min() - 1, idx.year.max() + 2)
    tr = holidays.Turkey(years=years)
    day = idx.normalize()
    is_hol = pd.Series(day.map(lambda d: d in tr), index=idx).astype(int)
    prev_hol = pd.Series((day - pd.Timedelta(days=1)).map(lambda d: d in tr), index=idx).astype(int)
    next_hol = pd.Series((day + pd.Timedelta(days=1)).map(lambda d: d in tr), index=idx).astype(int)
    return pd.DataFrame({
        "hour": idx.hour,
        "dayofweek": idx.dayofweek,
        "month": idx.month,
        "dayofyear": idx.dayofyear,
        "is_weekend": (idx.dayofweek >= 5).astype(int),
        "is_holiday": is_hol.values,
        "day_after_holiday": prev_hol.values,
        "day_before_holiday": next_hol.values,
    }, index=idx)


def daily_stat(s: pd.Series, how: str, lag_days: int) -> pd.Series:
    """Statistic of a full past day (D - lag_days), broadcast to every hour of D."""
    daily = s.resample("D").agg(how)
    daily.index = daily.index + pd.Timedelta(days=lag_days)
    return daily.reindex(s.index.normalize()).set_axis(s.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.asfreq("h")
    p = df[TARGET]
    X = calendar_features(df.index)

    # --- price history (known up to end of D-1) ---
    X["price_lag_24"] = p.shift(24)       # same hour, D-1
    X["price_lag_48"] = p.shift(48)       # same hour, D-2
    X["price_lag_168"] = p.shift(168)     # same hour, D-7
    X["price_lag_336"] = p.shift(336)     # same hour, D-14
    X["price_d1_mean"] = daily_stat(p, "mean", 1)
    X["price_d1_max"] = daily_stat(p, "max", 1)
    X["price_d1_min"] = daily_stat(p, "min", 1)
    X["price_d1_std"] = daily_stat(p, "std", 1)
    X["price_d1_last"] = daily_stat(p, "last", 1)          # 23:00 of D-1
    X["price_d1_zero_hours"] = daily_stat((p <= 0.01).astype(float), "sum", 1)
    # same hour, average of last 7 days (D-1 .. D-7)
    X["price_sameh_mean_7d"] = sum(p.shift(24 * k) for k in range(1, 8)) / 7
    # price level over the last 7 / 30 days (handles TL inflation drift)
    X["price_mean_7d"] = daily_stat(p, "mean", 1).rolling(24 * 7, min_periods=24).mean()
    X["price_mean_30d"] = daily_stat(p, "mean", 1).rolling(24 * 30, min_periods=24).mean()
    # proxy for the current regulatory cap: max price over the last 30 days
    X["price_max_30d"] = daily_stat(p, "max", 1).rolling(24 * 30, min_periods=24).max()
    X["lag24_vs_cap"] = X["price_lag_24"] / X["price_max_30d"]

    # --- fundamentals (realised data, only up to end of D-2) ---
    for col in ["consumption", "renewable_share", "wind_solar_share", "naturalGas",
                "dammedHydro", "wind", "sun"]:
        X[f"{col}_lag_48"] = df[col].shift(48)
        X[f"{col}_d2_mean"] = daily_stat(df[col], "mean", 2)
    X["consumption_lag_168"] = df["consumption"].shift(168)

    # --- weather for day D (stand-in for a weather forecast) ---
    for c in CITIES:
        X[f"temp_{c}"] = df[f"temperature_2m_{c}"]
    X["temp_mean"] = X[[f"temp_{c}" for c in CITIES]].mean(axis=1)
    X["temp_dev_18"] = (X["temp_mean"] - 18).abs()        # heating / cooling demand
    X["wind100_canakkale"] = df["wind_speed_100m_canakkale"]
    X["wind100_izmir"] = df["wind_speed_100m_izmir"]
    X["wind100_mean"] = df[[f"wind_speed_100m_{c}" for c in CITIES]].mean(axis=1)
    X["radiation_konya"] = df["shortwave_radiation_konya"]
    X["radiation_mean"] = df[[f"shortwave_radiation_{c}" for c in CITIES]].mean(axis=1)
    X["cloud_mean"] = df[[f"cloud_cover_{c}" for c in CITIES]].mean(axis=1)
    X["radiation_daily_sum"] = X["radiation_mean"].groupby(df.index.normalize()).transform("sum")
    X["wind_daily_mean"] = X["wind100_mean"].groupby(df.index.normalize()).transform("mean")

    X[TARGET] = p
    return X


def main():
    df = pd.read_parquet(PROCESSED / "hourly.parquet")
    X = build_features(df)
    X = X.iloc[336 + 24:]           # drop warm-up period with incomplete lags
    out = PROCESSED / "features.parquet"
    X.to_parquet(out)
    print(f"{X.shape[0]:,} rows, {X.shape[1] - 1} features -> {out}")
    print("NaNs:", int(X.isna().sum().sum()))


if __name__ == "__main__":
    main()
