from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from ..core.logging import logger
from .service import predict_batch, forecast_next

app = FastAPI(title="Stock Predictor API")

PREDICT_COUNTER = Counter("predict_requests_total", "Total predict requests")
FORECAST_COUNTER = Counter("forecast_requests_total", "Total forecast requests")
LATENCY = Histogram("request_latency_seconds", "Request latency", buckets=(0.05,0.1,0.2,0.5,1,2,5,10))

class PredictRequest(BaseModel):
    tickers: List[str] = Field(default_factory=list)
    start: str
    end: str
    prob_buy: float = 0.70
    prob_sell: float = 0.30

@app.post("/predict")
def predict(req: PredictRequest):
    with LATENCY.time():
        PREDICT_COUNTER.inc()
        logger.info("api.predict", extra=req.model_dump())
        return predict_batch(**req.model_dump())

@app.post("/forecast")
def forecast(req: PredictRequest):
    with LATENCY.time():
        FORECAST_COUNTER.inc()
        logger.info("api.forecast", extra=req.model_dump())
        return forecast_next(**req.model_dump())

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
