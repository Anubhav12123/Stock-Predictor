import pandas as pd
from src.backtest import pnl

def test_pnl_runs():
    s = pd.Series(["HOLD"]*10)
    c = pd.Series([100,101,102,103,104,103,102,103,104,105])
    equity, gross, final = pnl(s, c, fee_bps=2)
    assert len(equity) == 10
    assert isinstance(gross, float)
