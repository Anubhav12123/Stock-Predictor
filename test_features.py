import pandas as pd
from src.features import build_features

def test_build_features_basic():
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=40),
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": [100+i*0.1 for i in range(40)],
        "AdjClose": [100+i*0.1 for i in range(40)], "Volume": 1000, "Ticker": "TEST"
    })
    out = build_features(df, horizon_days=1)
    assert {"ret_1","rsi_14","macd","macd_signal","atr_14","bb_pct","roc_10","Target"} <= set(out.columns)
