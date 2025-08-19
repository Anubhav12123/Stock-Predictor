import numpy as np
import pandas as pd


def _to_series(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return pd.to_numeric(x, errors="coerce")


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    series = _to_series(series)
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1 / window, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    series = _to_series(series)
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    m = ema_f - ema_s
    s = m.ewm(span=signal, adjust=False).mean()
    return m, s


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    hl = (df["High"] - df["Low"]).abs()
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def bollinger_pct(series: pd.Series, window: int = 20) -> pd.Series:
    series = _to_series(series)
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    return (series - lower) / (upper - lower + 1e-9)


def roc(series: pd.Series, window: int = 10) -> pd.Series:
    series = _to_series(series)
    return series.pct_change(window)


def build_features(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = out["Close"].pct_change()
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_10"] = out["Close"].pct_change(10)

    out["vol_5"] = out["ret_1"].rolling(5).std()
    out["vol_10"] = out["ret_1"].rolling(10).std()

    out["sma_5"] = out["Close"].rolling(5).mean() / out["Close"]
    out["sma_10"] = out["Close"].rolling(10).mean() / out["Close"]
    out["sma_20"] = out["Close"].rolling(20).mean() / out["Close"]

    out["rsi_14"] = rsi(out["Close"], 14)

    m, s = macd(out["Close"])
    out["macd"] = m
    out["macd_signal"] = s

    out["atr_14"] = atr(out, 14)
    out["bb_pct"] = bollinger_pct(out["Close"], 20)
    out["roc_10"] = roc(out["Close"], 10)

    future = out["Close"].shift(-horizon_days)
    out["Target"] = (future > out["Close"]).astype(int)

    out = out.dropna().reset_index(drop=True)
    return out


def build_features_inference(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = out["Close"].pct_change()
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_10"] = out["Close"].pct_change(10)

    out["vol_5"] = out["ret_1"].rolling(5).std()
    out["vol_10"] = out["ret_1"].rolling(10).std()

    out["sma_5"] = out["Close"].rolling(5).mean() / out["Close"]
    out["sma_10"] = out["Close"].rolling(10).mean() / out["Close"]
    out["sma_20"] = out["Close"].rolling(20).mean() / out["Close"]

    out["rsi_14"] = rsi(out["Close"], 14)

    m, s = macd(out["Close"])
    out["macd"] = m
    out["macd_signal"] = s

    out["atr_14"] = atr(out, 14)
    out["bb_pct"] = bollinger_pct(out["Close"], 20)
    out["roc_10"] = roc(out["Close"], 10)

    out = out.dropna().reset_index(drop=True)
    return out
