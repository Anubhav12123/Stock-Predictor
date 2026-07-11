# Stock Predictor

Streamlit stock predictor with calibrated probabilities, next-day forecasting, and a FastAPI service backend.

## What it does

- Pulls historical price data per ticker and trains a classifier to estimate the probability of a price rise over a configurable horizon.
- Calibrates prediction probabilities and exposes BUY/SELL thresholds and a minimum-confidence filter in the UI.
- Forecasts the next trading day and backtests signals against historical data.
- Serves predictions through a FastAPI service (`src/api`) in addition to the Streamlit UI, with Prometheus metrics.

## Stack

Python 3.11, Streamlit, FastAPI + Uvicorn, scikit-learn, XGBoost, LightGBM, Optuna, pandas/NumPy, DuckDB, Pandera, Poetry.

## Project layout

- `streamlit_app.py` — interactive UI for predictions and forecasts
- `src/api/` — FastAPI service (`main.py`, `service.py`)
- `src/model.py` — training, prediction, and forecasting logic
- `src/features.py` — feature engineering
- `src/backtest.py`, `src/eval.py` — backtesting and evaluation
- `src/tune.py` — Optuna hyperparameter tuning
- `src/core/` — settings and logging
- `src/data/` — data ingestion and schemas
- `test/` — unit tests
- `artifacts/` — trained model artifacts (`.joblib`)

## Development

```bash
poetry install

# Streamlit UI
poetry run streamlit run streamlit_app.py

# FastAPI service
poetry run uvicorn src.api.main:app --reload

# Tests
poetry run pytest
```

## Docker

```bash
docker build -t stock-predictor .
docker run -p 8000:8000 stock-predictor
```
