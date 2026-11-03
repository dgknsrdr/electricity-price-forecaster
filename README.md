# ⚡ Turkish Day-Ahead Electricity Price Forecaster

Forecasting tomorrow's hourly electricity price (PTF / Market Clearing Price) in Turkey — with a **live pipeline** that publishes a new forecast every day and tracks its own accuracy against what actually happened.

> 🔗 **Live dashboard:** _link here once deployed_
> 📊 **Backtest (last 12 months):** MAE **388 TL/MWh** vs 478 for the best naive baseline: **19% better**

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
│   ├── build_dataset.py # merge raw files into one hourly table
│   ├── features.py     # feature engineering
│   ├── train.py        # model training + time-based backtest
│   └── predict.py      # daily forecast job
├── reports/            # backtest predictions and metrics
├── app/
│   └── dashboard.py    # Streamlit app
├── .github/workflows/  # daily scheduled run
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env          # add your EPİAŞ username/password
python src/fetch_data.py --start 2022-01-01   # download raw data
python src/build_dataset.py                    # merge into one hourly table
python src/features.py                         # leakage-safe features
python src/train.py                            # walk-forward backtest + final model
```
EPİAŞ requires a free account for API access (register on the EPİAŞ registration platform).

## Methodology notes
- **No leakage:** features only use data that would be available at forecast time (morning of D-1).
- **Time-based validation:** walk-forward backtest, never random splits.
- **Baselines first:** "same hour yesterday" and "same hour last week". The model is only useful if it beats them.
- **Regime changes:** price caps and inflation shift the price level over time — handled explicitly (see notebooks).

## Results
Walk-forward backtest, Nov 2025 – Nov 2026, retrained monthly (8,760 hourly forecasts):

| Method | MAE (TL/MWh) | RMSE | MAE % of mean price |
|---|---|---|---|
| **LightGBM** | **388** | **559** | **17.6%** |
| Naive: avg of same hour D-1 and D-7 | 478 | 697 | 21.7% |
| Naive: same hour yesterday | 520 | 807 | 23.6% |
| Naive: same hour last week | 606 | 908 | 27.5% |

- The model beats the best baseline in 12 of 13 months and at every hour of the day.
- **Failure case:** in April 2026 the spring price collapse (many near-zero hours from hydro + solar) was outside anything seen in training, and the model stayed anchored to old price levels (MAE 667 vs 523). Details and next steps are in `notebooks/02_model.ipynb`.
- **Weather caveat:** the backtest uses realised weather as a stand-in for forecasts. Without weather features the MAE is 398, so real-world performance should lie between 388 and 398.

## Data sources
- EPİAŞ Transparency Platform 2.0 (via `eptr2`)
- Open-Meteo historical & forecast weather API
