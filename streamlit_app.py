import os
import streamlit as st
import pandas as pd
from datetime import date
from src.model import predict_tickers, forecast_next_day
from src.core.settings import settings
from src.core.logging import logger
import httpx

st.set_page_config(page_title="Stock Predictor — FAANG Upgrade", layout="wide")
st.title("Stock Predictor")

cfg = settings.as_config_dict()
horizon_days = int(cfg["training"].get("horizon_days", 1))
API_URL = os.getenv("SP_API_URL") or settings.api_url

colx, coly, colz = st.columns([3, 1, 1])
tickers = colx.text_input("Stock tickers (comma-separated, e.g., AAPL, MSFT)", "AAPL,MSFT")
auto_train = coly.toggle("Auto-train if needed", True)
force_retrain = colz.toggle("Force retrain now", False)

col1, col2 = st.columns(2)
start = col1.date_input("Start date (training + prediction window begins)", value=date(2020, 1, 1))
end = col2.date_input("End date (latest data to use)", value=date.today())

today = date.today()
if end > today:
    st.warning("End date is in the future; using today instead.")
    end = today

col3, col4 = st.columns(2)
p_buy = float(col3.slider("BUY threshold (min probability price will rise)", 0.70, 0.99, 0.70))
p_sell = float(col4.slider("SELL threshold (max probability price will rise)", 0.01, 0.30, 0.30))

min_conf = st.slider("Only show rows/forecasts with model confidence ≥ this", 0.70, 0.99, 0.70)

c1, c2 = st.columns(2)
do_predict = c1.button("Predict & show signals")
do_forecast = c2.button("Forecast next trading day")

with st.expander("How this works"):
    st.markdown(f"""
- **BUY/SELL rules**: BUY if probability ≥ {p_buy:.2f}. SELL if probability ≤ {p_sell:.2f}.
- **Confidence filter**: We only show rows with `max(P(up), 1−P(up)) ≥ {int(min_conf*100)}%`.
- **Next-day forecast**: Separate regressor predicts log-return over {horizon_days} day(s), then converts to a price.
""")

tlist = [t.strip().upper() for t in tickers.split(",") if t.strip()]
cfg["training"]["prob_buy"] = p_buy
cfg["training"]["prob_sell"] = p_sell

def call_predict(tickers, start, end, cfg):
    if API_URL:
        payload = {"tickers": tickers, "start": str(start), "end": str(end),
                   "prob_buy": cfg["training"]["prob_buy"], "prob_sell": cfg["training"]["prob_sell"]}
        with httpx.Client(timeout=60) as client:
            r = client.post(f"{API_URL}/predict", json=payload)
            r.raise_for_status()
            return r.json()
    else:
        logger.info("predict.request", extra={"tickers": tickers, "start": str(start), "end": str(end), "prob_buy": cfg["training"]["prob_buy"]})
        return predict_tickers(tickers, start, end, cfg, auto_train=auto_train, force_retrain=force_retrain)

def call_forecast(tickers, start, end, cfg):
    if API_URL:
        payload = {"tickers": tickers, "start": str(start), "end": str(end),
                   "prob_buy": cfg["training"]["prob_buy"], "prob_sell": cfg["training"]["prob_sell"]}
        with httpx.Client(timeout=60) as client:
            r = client.post(f"{API_URL}/forecast", json=payload)
            r.raise_for_status()
            return r.json()
    else:
        logger.info("forecast.request", extra={"tickers": tickers, "start": str(start), "end": str(end)})
        return forecast_next_day(tickers, start, end, cfg, auto_train=auto_train, force_retrain=force_retrain)

csv_rows = []
if do_predict:
    with st.spinner("Predicting..."):
        obj = call_predict(tlist, start, end, cfg)

    for r in obj["results"]:
        if r.get("status") != "ok":
            st.warning(f'{r.get("ticker")}: {r.get("status")}')
            continue

        st.subheader(f'{r["ticker"]}')
        df = pd.DataFrame(r["signals"])
        if df.empty:
            st.info("No signals returned.")
            continue

        df["confidence"] = df["prob_up"].apply(lambda p: max(p, 1 - p))
        df_filtered = df[df["confidence"] >= min_conf].copy().sort_values("date")

        if df_filtered.empty:
            st.info(f"No rows meet the confidence ≥ {min_conf:.2f} requirement.")
            continue

        m1, m2 = st.columns(2)
        m1.metric("Training samples used", r.get("meta", {}).get("n", "-"))
        m2.metric(f"Signals shown (≥ {int(min_conf*100)}% conf)", len(df_filtered))

        t1, t2 = st.columns(2)
        with t1:
            st.caption(f"Closing price over time (confidence ≥ {int(min_conf*100)}%)")
            st.line_chart(df_filtered.set_index("date")[["close"]])
        with t2:
            st.caption(f"Predicted probability of price going up (confidence ≥ {int(min_conf*100)}%)")
            st.line_chart(df_filtered.set_index("date")[["prob_up"]])

        display_df = df_filtered.rename(columns={
            "date": "Date", "open": "Open Price", "close": "Close Price",
            "prob_up": "Probability (Up)", "signal": "Signal", "confidence": "Model Confidence",
        })
        st.dataframe(display_df.tail(200)[["Date","Open Price","Close Price","Probability (Up)","Model Confidence","Signal"]], hide_index=True)

        for s in df_filtered.to_dict(orient="records"):
            csv_rows.append([r["ticker"], s["date"], s.get("open"), s["close"], s["prob_up"], s["confidence"], s["signal"]])

    if csv_rows:
        out = pd.DataFrame(csv_rows, columns=["Ticker","Date","Open Price","Close Price","Probability (Up)","Model Confidence","Signal"])
        st.download_button("Download signals CSV (filtered)", out.to_csv(index=False).encode(), "signals_filtered.csv", "text/csv")

if do_forecast:
    with st.spinner("Forecasting next trading day..."):
        res = call_forecast(tlist, start, end, cfg)

    forecasts = []
    for r in res["results"]:
        if r.get("status") != "ok":
            st.warning(f'{r.get("ticker")}: {r.get("status")}')
            continue
        conf = max(r["prob_up"], 1 - r["prob_up"])
        if conf < min_conf:
            st.info(f'{r["ticker"]}: model confidence {conf:.2f} < {min_conf:.2f} — not showing forecast.')
            continue
        forecasts.append({
            "Ticker": r["ticker"],
            "Last Date": r["last_date"],
            "Next Trading Day": r["next_date"],
            "Last Close": r["last_close"],
            "Predicted Next Close": r["pred_close"],
            "Expected % Change": 100.0 * r["expected_change_pct"],
            "Probability (Up)": r["prob_up"],
        })

    if len(forecasts) == 0:
        st.info("No forecasts met the confidence requirement.")
    else:
        dfF = pd.DataFrame(forecasts)
        st.subheader("Next-Day Price Forecasts (filtered by confidence)")
        st.dataframe(dfF, hide_index=True)
        st.download_button("Download forecasts CSV", dfF.to_csv(index=False).encode(), "next_day_forecasts.csv", "text/csv")
