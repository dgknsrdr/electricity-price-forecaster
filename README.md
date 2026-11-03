# ⚡ Turkish Day-Ahead Electricity Price Forecaster

Forecasting tomorrow's hourly electricity price (PTF / Market Clearing Price) in Turkey — with a **live pipeline** that publishes a new forecast every day and tracks its own accuracy against what actually happened.

> 🔗 **Live dashboard:** _link here once deployed_
> 📊 **Current model vs. naive baseline:** _MAE improvement % here_

## Why this matters
Electricity suppliers, large industrial consumers and renewable plants decide how much to buy, sell or store based on tomorrow's price. A better forecast means cheaper energy and less risk. In Turkey, the day-ahead market clears around midday for the following day, so the forecast has to be made **in the morning, using only information available at that time**.

## What it does
1. **Collects data daily** — prices, consumption and generation by source from EPİAŞ Transparency Platform; weather from Open-Meteo.
2. **Builds features** — calendar (hour, weekday, holidays), lagged prices, demand, renewable share, temperature, wind, solar radiation.
3. **Forecasts** the 24 hourly prices for the next day (LightGBM, compared against naive baselines).
4. **Evaluates itself** — every day yesterday's forecast is compared with the real price and the error history is stored.
5. **Shows it** — a Streamlit dashboard with tomorrow's forecast, forecast vs. actual, and rolling error.

## Project structure
```
├── data/
│   ├── raw/            # untouched API pulls (parquet)
│   └── processed/      # feature tables
├── notebooks/          # EDA and experiments
├── src/
│   ├── fetch_data.py   # EPİAŞ + weather download
│   ├── features.py     # feature engineering
│   ├── train.py        # model training + time-based backtest
│   └── predict.py      # daily forecast job
├── app/
│   └── dashboard.py    # Streamlit app
├── .github/workflows/  # daily scheduled run
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
cp .env..example .env.          # add your EPİAŞ username/password
python src/fetch_data.py --start 2022-01-01
```
EPİAŞ requires a free account for API access (register on the EPİAŞ registration platform).

## Methodology notes
- **No leakage:** features only use data that would be available at forecast time (morning of D-1).
- **Time-based validation:** walk-forward backtest, never random splits.
- **Baselines first:** "same hour yesterday" and "same hour last week". The model is only useful if it beats them.
- **Regime changes:** price caps and inflation shift the price level over time — handled explicitly (see notebooks).

## Results
_Backtest table, error by hour of day, and the biggest misses with explanations go here._

## Data sources
- EPİAŞ Transparency Platform 2.0 (via `eptr2`)
- Open-Meteo historical & forecast weather API
