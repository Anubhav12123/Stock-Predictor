from __future__ import annotations
import os, json, hashlib, joblib, numpy as np, pandas as pd
from math import exp
from datetime import date, datetime
from pandas.tseries.offsets import BDay
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, accuracy_score, balanced_accuracy_score, mean_absolute_error

from .features import build_features, build_features_inference
import yfinance as yf

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")


def default_start_end(lookback_years: int):
    from datetime import datetime, timedelta
    e = datetime.utcnow().date()
    s = e - timedelta(days=365 * lookback_years)
    return s, e


def _cfg_sig(cfg: dict) -> str:
    s = json.dumps(cfg.get("training", {}), sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


def _best_threshold(y_true: np.ndarray, proba: np.ndarray):
    grid = np.linspace(0.35, 0.65, 31)
    best_t, best_mcc = 0.5, -1.0
    for t in grid:
        yhat = (proba >= t).astype(int)
        m = matthews_corrcoef(y_true, yhat)
        if m > best_mcc:
            best_mcc, best_t = m, t
    return float(best_t), float(best_mcc)


def fetch_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Adj Close": "AdjClose"}).reset_index().rename(columns={"Date": "Date"})
    df["Ticker"] = ticker
    return df[["Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"]]


def model_path(ticker: str):
    return os.path.join(ARTIFACTS, f"{ticker.upper()}.joblib")


def _needs_retrain(bundle: dict, end_req: date, cfg: dict) -> bool:
    try:
        meta = bundle.get("meta", {})
        old_end = date.fromisoformat(meta.get("end", "1900-01-01"))
        old_sig = meta.get("cfg_sig")
        regress_missing = "regressor" not in bundle
        return (old_end < end_req) or (old_sig != _cfg_sig(cfg)) or regress_missing
    except Exception:
        return True


def ensure_model(ticker: str, start: date, end: date, cfg: dict, force_retrain: bool = False):
    path = model_path(ticker)
    if not os.path.exists(path) or force_retrain:
        res = train_one(ticker, start, end, cfg)
        return res.get("status") == "trained"
    try:
        bundle = joblib.load(path)
        if _needs_retrain(bundle, end, cfg):
            res = train_one(ticker, start, end, cfg)
            return res.get("status") == "trained"
    except Exception:
        res = train_one(ticker, start, end, cfg)
        return res.get("status") == "trained"
    return True


def train_one(ticker: str, start: date, end: date, cfg: dict):
    from sklearn.calibration import CalibratedClassifierCV

    raw = fetch_ohlcv(ticker, start, end)
    if raw.empty:
        return {"ticker": ticker, "status": "no_data"}

    horizon = int(cfg["training"].get("horizon_days", 1))
    df = build_features(raw, horizon_days=horizon)
    if len(df) < cfg["training"]["min_rows"]:
        return {"ticker": ticker, "status": "not_enough_rows", "rows": int(len(df))}

    X = df[cfg["training"]["features"]]
    y_cls = df["Target"].values
    tscv = TimeSeriesSplit(n_splits=cfg["training"].get("tscv_splits", 5))

    # Classifier
    base_cls = HistGradientBoostingClassifier(random_state=cfg["training"]["random_state"])
    space_cls = {
        "max_depth": [None, 3, 5, 7],
        "learning_rate": np.linspace(0.03, 0.2, 8),
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [10, 20, 30, 50],
        "l2_regularization": np.linspace(0.0, 1.0, 6),
    }
    search_cls = RandomizedSearchCV(
        base_cls, param_distributions=space_cls, n_iter=cfg["training"].get("tune_iter", 25),
        scoring="balanced_accuracy", cv=tscv, n_jobs=-1, verbose=0, refit=True,
        random_state=cfg["training"]["random_state"],
    )
    search_cls.fit(X, y_cls)
    best_cls = search_cls.best_estimator_

    model_cls = best_cls
    if cfg["training"].get("calibrate", True):
        cal_cv = TimeSeriesSplit(n_splits=cfg["training"].get("calibration_cv_splits", 3))
        method = cfg["training"].get("calibration_method", "isotonic")
        calibrator = CalibratedClassifierCV(best_cls, method=method, cv=cal_cv)
        calibrator.fit(X, y_cls)
        model_cls = calibrator

    tr_idx, te_idx = list(tscv.split(X))[-1]
    proba_val = model_cls.predict_proba(X.iloc[te_idx])[:, 1]
    t_star, mcc_star = _best_threshold(y_cls[te_idx], proba_val)
    yhat_val = (proba_val >= t_star).astype(int)
    acc = float(accuracy_score(y_cls[te_idx], yhat_val))
    bacc = float(balanced_accuracy_score(y_cls[te_idx], yhat_val))

    # Regressor (next-day log-return)
    y_reg = np.log(df["Close"].shift(-horizon) / df["Close"])
    df_reg = df.copy()
    df_reg["y_reg"] = y_reg
    df_reg = df_reg.dropna().reset_index(drop=True)

    Xr = df_reg[cfg["training"]["features"]]
    yr = df_reg["y_reg"].values

    base_reg = HistGradientBoostingRegressor(random_state=cfg["training"]["random_state"])
    space_reg = {
        "max_depth": [None, 3, 5, 7],
        "learning_rate": np.linspace(0.03, 0.2, 8),
        "max_leaf_nodes": [31, 63, 127, 255],
        "min_samples_leaf": [10, 20, 30, 50],
        "l2_regularization": np.linspace(0.0, 1.0, 6),
    }
    search_reg = RandomizedSearchCV(
        base_reg, param_distributions=space_reg, n_iter=max(15, int(cfg["training"].get("tune_iter", 25) * 0.6)),
        scoring="neg_mean_absolute_error", cv=tscv, n_jobs=-1, verbose=0, refit=True,
        random_state=cfg["training"]["random_state"],
    )
    search_reg.fit(Xr, yr)
    best_reg = search_reg.best_estimator_

    # Simple validation MAE on overlapping indices
    valid_mask = te_idx[:-horizon] if horizon < len(te_idx) else te_idx[:0]
    if len(valid_mask) > 0:
        mae_val = float(mean_absolute_error(yr[: len(yr)][valid_mask[: len(yr)]], best_reg.predict(Xr.iloc[: len(yr)].iloc[valid_mask[: len(yr)]])))
    else:
        mae_val = float("nan")

    meta = {
        "ticker": ticker.upper(),
        "start": str(start),
        "end": str(end),
        "n": int(len(df)),
        "cv_best_score_bal_acc": float(search_cls.best_score_),
        "val_mcc_at_t": float(mcc_star),
        "val_acc_at_t": acc,
        "val_bal_acc_at_t": bacc,
        "reg_val_mae_logret": mae_val,
        "threshold": float(t_star),
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cfg_sig": _cfg_sig(cfg),
    }

    os.makedirs(ARTIFACTS, exist_ok=True)
    joblib.dump(
        {"model": model_cls, "regressor": best_reg, "features": cfg["training"]["features"], "horizon": horizon, "meta": meta, "cfg": cfg},
        model_path(ticker),
    )
    return {"ticker": ticker, "status": "trained", "rows": int(len(df)), "val_bal_acc": bacc}


def train_models(tickers, start, end, cfg):
    if start is None or end is None:
        s, e = default_start_end(cfg["training"]["lookback_years"])
    else:
        s, e = start, end
    out = []
    for t in tickers:
        out.append(train_one(t, s, e, cfg))
    return out


def predict_tickers(tickers, start, end, cfg, auto_train: bool = True, force_retrain: bool = False):
    if start is None or end is None:
        s, e = default_start_end(cfg["training"]["lookback_years"])
    else:
        s, e = start, end

    results = []
    for t in tickers:
        if auto_train:
            ensure_model(t, s, e, cfg, force_retrain=force_retrain)

        path = model_path(t)
        if not os.path.exists(path):
            results.append({"ticker": t, "status": "model_missing"})
            continue

        bundle = joblib.load(path)
        feats = bundle["features"]
        thresh = float(bundle.get("meta", {}).get("threshold", 0.5))
        horizon = int(bundle.get("horizon", cfg["training"].get("horizon_days", 1)))

        raw = fetch_ohlcv(t, s, e)
        if raw.empty:
            results.append({"ticker": t, "status": "no_data"})
            continue

        df = build_features(raw, horizon_days=horizon)
        if df.empty:
            results.append({"ticker": t, "status": "no_features"})
            continue

        X = df[feats]
        proba = bundle["model"].predict_proba(X)[:, 1]

        p_buy_cfg = float(cfg["training"]["prob_buy"])
        p_sell_cfg = float(cfg["training"]["prob_sell"])
        effective_buy = max(thresh, p_buy_cfg)
        effective_sell = min(1.0 - thresh, p_sell_cfg)

        signals = []
        for i, row in df.iterrows():
            p = float(proba[i])
            if p >= effective_buy:
                sig = "BUY"
            elif p <= effective_sell:
                sig = "SELL"
            else:
                sig = "HOLD"
            signals.append({
                "date": str(row["Date"]) if "Date" in df.columns else None,
                "open": float(row["Open"]) if "Open" in df.columns else None,
                "close": float(row["Close"]),
                "prob_up": p,
                "signal": sig
            })

        meta = bundle.get("meta", {})
        results.append({"ticker": t.upper(), "status": "ok", "meta": meta, "signals": signals})

    return {"results": results}


def forecast_next_day(tickers, start, end, cfg, auto_train: bool = True, force_retrain: bool = False):
    if start is None or end is None:
        s, e = default_start_end(cfg["training"]["lookback_years"])
    else:
        s, e = start, end

    out = []
    for t in tickers:
        if auto_train:
            ensure_model(t, s, e, cfg, force_retrain=force_retrain)

        path = model_path(t)
        if not os.path.exists(path):
            out.append({"ticker": t, "status": "model_missing"})
            continue

        bundle = joblib.load(path)
        feats = bundle["features"]
        horizon = int(bundle.get("horizon", cfg["training"].get("horizon_days", 1)))

        raw = fetch_ohlcv(t, s, e)
        if raw.empty:
            out.append({"ticker": t, "status": "no_data"})
            continue

        finf = build_features_inference(raw)
        if finf.empty:
            out.append({"ticker": t, "status": "no_features"})
            continue

        x_last = finf.iloc[-1]
        X_last = finf[feats].iloc[[-1]]

        p_up = float(bundle["model"].predict_proba(X_last)[:, 1][0])

        last_close = float(x_last["Close"])
        yhat_log = float(bundle["regressor"].predict(X_last)[0])
        pred_close = last_close * exp(yhat_log)

        last_date = pd.to_datetime(x_last["Date"]).date() if "Date" in finf.columns else None
        next_trading_day = (pd.Timestamp(last_date) + BDay(horizon)).date() if last_date else None

        out.append({
            "ticker": t.upper(),
            "status": "ok",
            "last_date": str(last_date) if last_date else None,
            "next_date": str(next_trading_day) if next_trading_day else None,
            "last_close": last_close,
            "pred_close": float(pred_close),
            "expected_change_pct": float(np.exp(yhat_log) - 1.0),
            "prob_up": p_up,
            "horizon_days": horizon,
            "meta": bundle.get("meta", {}),
        })

    return {"results": out}
