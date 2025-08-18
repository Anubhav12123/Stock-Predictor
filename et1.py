from __future__ import annotations
import duckdb, pathlib
import pandas as pd
import yfinance as yf
from datetime import date
from .schemas import OHLCVSchema

DATA = pathlib.Path("data")
DB = DATA / "market.duckdb"


def ensure_db():
    DATA.mkdir(exist_ok=True)
    con = duckdb.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS ohlcv (Date TIMESTAMP, Open DOUBLE, High DOUBLE, Low DOUBLE, Close DOUBLE, AdjClose DOUBLE, Volume BIGINT, Ticker VARCHAR)")
    con.close()


def ingest_yf_daily(ticker: str, start: date, end: date) -> int:
    ensure_db()
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return 0
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Adj Close": "AdjClose"})
    df["Ticker"] = ticker
    df = OHLCVSchema.validate(df)  # schema validation
    con = duckdb.connect(DB)
    con.execute("INSERT INTO ohlcv BY NAME SELECT * FROM df")
    con.close()
    return len(df)


def load_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame:
    ensure_db()
    con = duckdb.connect(DB)
    df = con.execute(
        "SELECT * FROM ohlcv WHERE Ticker = ? AND Date >= ? AND Date < ? ORDER BY Date ASC",
        [ticker, pd.Timestamp(start), pd.Timestamp(end)],
    ).df()
    con.close()
    return df
