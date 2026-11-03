"""
Merge the raw downloads into one hourly table: data/processed/hourly.parquet

One row per hour (Europe/Istanbul local time, Turkey has no DST since 2016):
    price, price_eur, consumption, generation by source, renewable share, weather.

Usage:
    python src/build_dataset.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

GEN_COLS = [
    "naturalGas", "dammedHydro", "lignite", "river", "importCoal", "wind", "sun",
    "fueloil", "geothermal", "asphaltiteCoal", "blackCoal", "biomass",
    "importExport", "wasteheat",
]
RENEWABLE = ["dammedHydro", "river", "wind", "sun", "geothermal", "biomass"]


def to_local_ts(s: pd.Series) -> pd.Series:
    """'2022-01-01T00:00:00+03:00' -> naive local timestamp 2022-01-01 00:00."""
    return pd.to_datetime(s.str[:19])


def load() -> pd.DataFrame:
    ptf = pd.read_parquet(RAW / "ptf.parquet")
    ptf = pd.DataFrame({
        "ts": to_local_ts(ptf["date"]),
        "price": ptf["price"],
        "price_eur": ptf["priceEur"],
    })

    cons = pd.read_parquet(RAW / "consumption.parquet")
    cons = pd.DataFrame({"ts": to_local_ts(cons["date"]), "consumption": cons["consumption"]})

    gen = pd.read_parquet(RAW / "generation.parquet")
    gen = gen.assign(ts=to_local_ts(gen["date"]))[["ts", "total", *GEN_COLS]]
    gen = gen.rename(columns={"total": "generation_total"})

    weather = pd.read_parquet(RAW / "weather.parquet").rename(columns={"time": "ts"})

    df = (
        ptf.merge(cons, on="ts", how="left")
        .merge(gen, on="ts", how="left")
        .merge(weather, on="ts", how="left")
        .sort_values("ts")
        .set_index("ts")
    )

    df["renewable"] = df[RENEWABLE].sum(axis=1)
    df["renewable_share"] = df["renewable"] / df["generation_total"]
    df["wind_solar_share"] = (df["wind"] + df["sun"]) / df["generation_total"]
    return df


def main():
    df = load()
    assert df.index.is_unique, "duplicate hours"
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing = full.difference(df.index)
    print(f"{len(df):,} hours, {df.index.min()} -> {df.index.max()}, missing hours: {len(missing)}")
    print("NaNs per column (top 5):")
    print(df.isna().sum().sort_values(ascending=False).head())

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "hourly.parquet"
    df.to_parquet(out)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
