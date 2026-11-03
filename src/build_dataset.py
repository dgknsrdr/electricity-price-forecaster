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
    return merge_raw(
        pd.read_parquet(RAW / "ptf.parquet"),
        pd.read_parquet(RAW / "consumption.parquet"),
        pd.read_parquet(RAW / "generation.parquet"),
        pd.read_parquet(RAW / "weather.parquet"),
    )


def merge_raw(ptf, cons, gen, weather, index=None) -> pd.DataFrame:
    """Merge raw EPİAŞ + weather frames into one hourly table.

    index: optional full hourly index (e.g. extended to tomorrow for forecasting);
           by default the hours present in the price data.
    """
    ptf = pd.DataFrame({
        "ts": to_local_ts(ptf["date"]),
        "price": ptf["price"],
        "price_eur": ptf["priceEur"],
    })

    cons = pd.DataFrame({"ts": to_local_ts(cons["date"]), "consumption": cons["consumption"]})

    gen = gen.assign(ts=to_local_ts(gen["date"]))[["ts", "total", *GEN_COLS]]
    gen = gen.rename(columns={"total": "generation_total"})

    weather = weather.rename(columns={"time": "ts"})

    if index is None:
        index = pd.DatetimeIndex(ptf["ts"].sort_values())
    df = pd.DataFrame(index=pd.DatetimeIndex(index, name="ts"))
    for part in (ptf, cons, gen, weather):
        df = df.join(part.drop_duplicates("ts").set_index("ts"), how="left")

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
