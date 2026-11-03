"""
Streamlit dashboard: tomorrow's forecast, live track record and backtest results.

Run locally:
    streamlit run app/dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
GITHUB_USER = "dgknsrdr"
REPO = "electricity-price-forecaster"
REPO_URL = f"https://github.com/{GITHUB_USER}/{REPO}"
# Read live forecasts straight from GitHub so the app updates as soon as the daily job commits
LIVE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/main/forecasts/forecasts.csv"

C_ACTUAL, C_MODEL, C_NAIVE = "#1f2937", "#2563eb", "#f59e0b"

st.set_page_config(page_title="Turkish Electricity Price Forecast", page_icon="⚡", layout="wide")


@st.cache_data(ttl=900)
def load_live() -> pd.DataFrame:
    try:
        df = pd.read_csv(LIVE_URL, parse_dates=["target_ts"])
    except Exception:
        df = pd.read_csv(ROOT / "forecasts" / "forecasts.csv", parse_dates=["target_ts"])
    return df.sort_values("target_ts")


@st.cache_data
def load_backtest():
    bt = pd.read_parquet(ROOT / "reports" / "backtest_predictions.parquet")
    metrics = pd.read_csv(ROOT / "reports" / "metrics.csv", index_col=0)
    by_month = pd.read_csv(ROOT / "reports" / "metrics_by_month.csv", index_col=0, parse_dates=True)
    by_month.index = by_month.index.to_period("M").to_timestamp()   # label months by their first day
    importance = pd.read_csv(ROOT / "reports" / "feature_importance.csv", index_col=0)["share_of_gain"]
    return bt, metrics, by_month, importance


def mae(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).mean())


def line_chart(df, x, series, height=380, yaxis="TL/MWh"):
    fig = go.Figure()
    for col, name, color, dash in series:
        if col in df and df[col].notna().any():
            fig.add_trace(go.Scatter(x=df[x], y=df[col], name=name, mode="lines",
                                     line=dict(color=color, width=2.2 if col != "naive" else 1.4, dash=dash)))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis_title=yaxis, legend=dict(orientation="h", y=1.08, x=0))
    return fig


live = load_live()
bt, metrics, by_month, importance = load_backtest()

# ---------------- header ----------------
st.title("⚡ Turkish Day-Ahead Electricity Price Forecast")
st.markdown(
    "Hourly forecast of tomorrow's **PTF (market clearing price)** in Turkey, produced automatically every "
    f"morning from EPİAŞ and weather data. [Code and methodology on GitHub]({REPO_URL})."
)

target_day = live["target_ts"].dt.date.max()
tomorrow = live[live["target_ts"].dt.date == target_day]
scored = live.dropna(subset=["actual"])
bt_mae, naive_mae = metrics.loc["lightgbm", "MAE"], metrics.loc["naive_d1_d7_avg", "MAE"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Forecast for {target_day:%d %b %Y}", f"{tomorrow['forecast'].mean():,.0f} TL/MWh", help="Daily average")
peak = tomorrow.loc[tomorrow["forecast"].idxmax()]
c2.metric("Most expensive hour", f"{peak['target_ts']:%H:%M}", f"{peak['forecast']:,.0f} TL/MWh", delta_color="off")
c3.metric("Backtest MAE (12 months)", f"{bt_mae:,.0f} TL/MWh",
          f"{(1 - bt_mae / naive_mae):.0%} better than naive", delta_color="normal")
if len(scored):
    live_mae, live_naive = mae(scored["forecast"], scored["actual"]), mae(scored["naive"], scored["actual"])
    c4.metric(f"Live MAE ({scored['target_ts'].dt.date.nunique()} days)", f"{live_mae:,.0f} TL/MWh",
              f"naive: {live_naive:,.0f}", delta_color="off")
else:
    c4.metric("Live track record", "starting", "first actuals arrive tomorrow", delta_color="off")

# ---------------- tomorrow ----------------
st.subheader(f"Hourly forecast for {target_day:%A, %d %B %Y}")
st.plotly_chart(line_chart(tomorrow, "target_ts", [
    ("forecast", "Model forecast", C_MODEL, None),
    ("naive", "Naive baseline (avg of D-1 and D-7)", C_NAIVE, "dash"),
    ("actual", "Actual", C_ACTUAL, None),
]), width="stretch")

# ---------------- live track record ----------------
st.subheader("Live track record: forecast vs actual")
if len(scored):
    last = scored[scored["target_ts"] >= scored["target_ts"].max() - pd.Timedelta(days=30)]
    st.plotly_chart(line_chart(last, "target_ts", [
        ("actual", "Actual", C_ACTUAL, None),
        ("forecast", "Model forecast", C_MODEL, None),
        ("naive", "Naive baseline", C_NAIVE, "dash"),
    ]), width="stretch")
    daily = scored.assign(day=scored["target_ts"].dt.date).groupby("day").apply(
        lambda d: pd.Series({"model MAE": mae(d.forecast, d.actual), "naive MAE": mae(d.naive, d.actual)}),
        include_groups=False,
    ).round(0)
    st.dataframe(daily.sort_index(ascending=False), width="stretch")
else:
    st.info("The live forecasts started recently. Each morning the previous forecast is scored against "
            "the published price, and this section fills up day by day.")

# ---------------- backtest ----------------
st.subheader("Backtest: last 12 months, retrained monthly")
left, right = st.columns([3, 2])
with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=by_month.index, y=by_month["lightgbm"], name="LightGBM",
                             mode="lines+markers", line=dict(color=C_MODEL, width=2.5)))
    fig.add_trace(go.Scatter(x=by_month.index, y=by_month["naive_d1_d7_avg"], name="Naive baseline",
                             mode="lines+markers", line=dict(color=C_NAIVE, width=2, dash="dash")))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Monthly MAE (TL/MWh)",
                      xaxis=dict(dtick="M2", tickformat="%b %Y"), legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(fig, width="stretch")
with right:
    table = metrics.rename(index={
        "lightgbm": "LightGBM", "naive_d1_d7_avg": "Naive: avg D-1 & D-7",
        "naive_d1": "Naive: yesterday", "naive_d7": "Naive: last week"})
    table.columns = ["MAE", "RMSE", "MAE % of mean"]
    st.dataframe(table.round(1), width="stretch")
    st.caption("April 2026 is the one month where the model loses to the baseline: the spring price "
               "collapse (hydro + solar, many zero-price hours) was outside anything seen in training.")

st.markdown("**Explore any week of the backtest** (try 13 April 2026 to see the spring failure case)")
week_start = st.date_input("Week starting", value=pd.Timestamp("2026-10-12").date(),
                           min_value=bt.index.min().date(),
                           max_value=(bt.index.max() - pd.Timedelta(days=7)).date())
week = bt.loc[str(week_start): str(pd.Timestamp(week_start) + pd.Timedelta(days=6))].reset_index()
st.plotly_chart(line_chart(week, "ts", [
    ("actual", "Actual", C_ACTUAL, None),
    ("lightgbm", "Model forecast", C_MODEL, None),
    ("naive_d1_d7_avg", "Naive baseline", C_NAIVE, "dash"),
]), width="stretch")
st.caption(f"Week MAE: model {mae(week['lightgbm'], week['actual']):,.0f}, "
           f"naive {mae(week['naive_d1_d7_avg'], week['actual']):,.0f} TL/MWh")

# ---------------- method ----------------
with st.expander("How it works"):
    top = (importance.head(10) * 100).round(1)
    st.markdown(f"""
- **Data:** hourly prices, consumption and generation by source from the EPİAŞ Transparency Platform;
  temperature, wind and solar radiation for five cities from Open-Meteo. History since January 2022.
- **No leakage:** the forecast for day D is made on the morning of D-1 using only data published by then
  (prices up to D-1, consumption and generation up to D-2, a weather forecast for D).
- **Model:** LightGBM trained on an MAE objective with 53 features (price lags, demand, renewable share,
  weather, calendar and holidays). Forecasts are clipped to [0, regulatory price cap].
- **Validation:** walk-forward backtest over the last 12 months, compared against naive baselines.
- **Most important features:** {", ".join(f"`{k}` ({v}%)" for k, v in top.items())}.
""")
st.caption(f"Data: EPİAŞ Şeffaflık Platformu, Open-Meteo · Built by Doğukan Serdar · [GitHub]({REPO_URL})")
