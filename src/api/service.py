from __future__ import annotations
from typing import List, Dict, Any
from datetime import date
from ..model import predict_tickers, forecast_next_day

def predict_batch(tickers: List[str], start: str, end: str, prob_buy: float, prob_sell: float) -> Dict[str, Any]:
    cfg = {"training": {"prob_buy": float(prob_buy), "prob_sell": float(prob_sell)}}
    res = predict_tickers(tickers, date.fromisoformat(start), date.fromisoformat(end), cfg, auto_train=True)
    return res

def forecast_next(tickers: List[str], start: str, end: str, prob_buy: float, prob_sell: float) -> Dict[str, Any]:
    cfg = {"training": {"prob_buy": float(prob_buy), "prob_sell": float(prob_sell)}}
    res = forecast_next_day(tickers, date.fromisoformat(start), date.fromisoformat(end), cfg, auto_train=True)
    return res
