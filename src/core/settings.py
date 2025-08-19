from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingSettings(BaseModel):
    lookback_years: int = Field(default=7, ge=1)
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = 42

    prob_buy: float = Field(default=0.70, ge=0.5, le=0.99)
    prob_sell: float = Field(default=0.30, ge=0.01, le=0.5)

    min_rows: int = Field(default=200, ge=50)
    horizon_days: int = Field(default=1, ge=1)
    tscv_splits: int = Field(default=5, ge=2)
    tune_iter: int = Field(default=25, ge=5)

    calibrate: bool = True
    calibration_cv_splits: int = Field(default=3, ge=2)
    calibration_method: str = "isotonic"

    features: List[str] = [
        "ret_1", "ret_5", "ret_10",
        "vol_5", "vol_10",
        "sma_5", "sma_10", "sma_20",
        "rsi_14",
        "macd", "macd_signal",
        "atr_14", "bb_pct", "roc_10",
    ]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SP_", env_file=".env", extra="ignore")
    training: TrainingSettings = TrainingSettings()
    api_url: str | None = None
    sentry_dsn: str | None = None

    def as_config_dict(self) -> dict:
        return {"training": self.training.model_dump()}


settings = Settings()
