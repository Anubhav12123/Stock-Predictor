import pandas as pd

def pnl(signals: pd.Series, close: pd.Series, fee_bps: float = 2.0):
    pos = (signals == "BUY").astype(int) - (signals == "SELL").astype(int)
    ret = close.pct_change().fillna(0.0)
    gross = (pos.shift().fillna(0) * ret)
    cost = (pos.diff().abs().fillna(0)) * (fee_bps / 1e4)
    equity = (1 + gross - cost).cumprod()
    return equity, float(gross.sum()), float(equity.iloc[-1] - 1.0)
