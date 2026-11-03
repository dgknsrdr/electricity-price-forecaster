"""
Download raw data for the Turkish day-ahead electricity price forecaster.

- EPİAŞ Transparency Platform 2.0 (via the eptr2 package): hourly PTF (MCP),
  real-time consumption and real-time generation by source.
- Open-Meteo: hourly weather for a few demand / renewable-heavy cities.

Usage:
    python src/fetch_data.py --start 2022-01-01            # up to yesterday
    python src/fetch_data.py --start 2024-01-01 --end 2024-03-31
    python src/fetch_data.py --only weather                 # just the weather step

Needs EPTR_USERNAME and EPTR_PASSWORD in a .env file.
"""

import argparse
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from eptr2 import EPTR2

# Paths relative to the project root, so the script works from any working directory
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# eptr2 call keys. If one fails, run EPTR2(...).get_available_calls() to see current names.
EPTR_DATASETS = {
    "ptf": "mcp",               # day-ahead market clearing price (PTF)
    "consumption": "rt-cons",   # real-time consumption
    "generation": "rt-gen",     # real-time generation by source
}

# Demand centres + regions with lots of wind / solar
CITIES = {
    "istanbul": (41.01, 28.97),
    "ankara": (39.93, 32.86),
    "izmir": (38.42, 27.14),
    "canakkale": (40.15, 26.41),   # wind
    "konya": (37.87, 32.48),       # solar
}
WEATHER_VARS = "temperature_2m,wind_speed_100m,shortwave_radiation,cloud_cover"


def date_chunks(start: date, end: date, days: int):
    """Split [start, end] into consecutive chunks so API calls stay small."""
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def fetch_eptr(client: EPTR2, key: str, start: date, end: date) -> pd.DataFrame:
    frames = []
    for s, e in date_chunks(start, end, days=30):
        for attempt in range(3):
            try:
                df = client.call(key, start_date=s.isoformat(), end_date=e.isoformat())
                frames.append(pd.DataFrame(df))
                break
            except Exception as exc:  # network hiccups, token refresh, rate limits
                print(f"  {key} {s}->{e} failed ({exc}), retry {attempt + 1}/3")
                time.sleep(5 * (attempt + 1))
        time.sleep(1)  # be polite to the API
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_client() -> EPTR2:
    """EPİAŞ client; credentials come from .env locally or env vars in GitHub Actions."""
    load_dotenv(ROOT / ".env")
    return EPTR2(
        username=os.environ["EPTR_USERNAME"],
        password=os.environ["EPTR_PASSWORD"],
        use_dotenv=False,          # we load .env ourselves (avoids the "not found" warning)
        tgt_path=str(ROOT),        # keep the login token in the project root (gitignored)
    )


def to_wide(weather: pd.DataFrame) -> pd.DataFrame:
    """Long (time, city, vars) -> wide: one row per hour, columns like temperature_2m_istanbul."""
    weather = weather.copy()
    weather["time"] = pd.to_datetime(weather["time"])
    weather = weather.drop_duplicates(subset=["time", "city"])
    return weather.pivot(index="time", columns="city").pipe(
        lambda w: w.set_axis([f"{v}_{c}" for v, c in w.columns], axis=1)
    ).reset_index()


def fetch_weather_forecast(past_days: int = 60, forecast_days: int = 2) -> pd.DataFrame:
    """Recent weather + forecast for the next days (Open-Meteo forecast API, max 92 past days)."""
    frames = []
    for city, (lat, lon) in CITIES.items():
        for attempt in range(4):
            try:
                r = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "hourly": WEATHER_VARS,
                        "past_days": past_days,
                        "forecast_days": forecast_days,
                        "timezone": "Europe/Istanbul",
                    },
                    timeout=120,
                )
                r.raise_for_status()
                break
            except requests.RequestException as exc:
                print(f"  weather forecast {city} failed ({exc}), retry {attempt + 1}/4")
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"Weather forecast download failed for {city}")
        df = pd.DataFrame(r.json()["hourly"])
        df["city"] = city
        frames.append(df)
    return to_wide(pd.concat(frames, ignore_index=True))


def fetch_weather(start: date, end: date) -> pd.DataFrame:
    """Open-Meteo archive, one city and one year per request (big ranges time out)."""
    frames = []
    for city, (lat, lon) in CITIES.items():
        for s, e in date_chunks(start, end, days=365):
            for attempt in range(4):
                try:
                    r = requests.get(
                        "https://archive-api.open-meteo.com/v1/archive",
                        params={
                            "latitude": lat,
                            "longitude": lon,
                            "start_date": s.isoformat(),
                            "end_date": e.isoformat(),
                            "hourly": WEATHER_VARS,
                            "timezone": "Europe/Istanbul",
                        },
                        timeout=180,
                    )
                    r.raise_for_status()
                    break
                except requests.RequestException as exc:
                    print(f"  weather {city} {s}->{e} failed ({exc}), retry {attempt + 1}/4")
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(f"Weather download failed for {city} {s}->{e}")
            df = pd.DataFrame(r.json()["hourly"])
            df["city"] = city
            frames.append(df)
            print(f"  weather {city} {s}->{e}: {len(df):,} rows")
            time.sleep(1)
    return to_wide(pd.concat(frames, ignore_index=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    parser.add_argument(
        "--only", choices=["epias", "weather"],
        help="download just one source (e.g. --only weather to retry the weather step)",
    )
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "epias"):
        client = make_client()
        for name, key in EPTR_DATASETS.items():
            print(f"Fetching {name} ({key}) {start} -> {end}")
            df = fetch_eptr(client, key, start, end)
            out = RAW_DIR / f"{name}.parquet"
            df.to_parquet(out, index=False)
            print(f"  saved {len(df):,} rows to {out}")

    if args.only in (None, "weather"):
        print(f"Fetching weather {start} -> {end}")
        weather = fetch_weather(start, end)
        out = RAW_DIR / "weather.parquet"
        weather.to_parquet(out, index=False)
        print(f"  saved {len(weather):,} rows to {out}")


if __name__ == "__main__":
    main()
