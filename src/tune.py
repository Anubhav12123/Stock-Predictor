from __future__ import annotations
import optuna, numpy as np
from typing import List, Tuple
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef

def _mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)

def tune_xgb(X, y, splits: List[Tuple], decision_floor: float = 0.7, n_trials: int = 50):
    def obj(trial: optuna.Trial):
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            eval_metric="logloss", tree_method="hist", n_jobs=-1
        )
        scores = []
        for tr, te in splits:
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[te])[:, 1]
            yhat = (p >= decision_floor).astype(int)  # enforce floor
            scores.append(_mcc(y[te], yhat))
        return -float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=n_trials)
    return study.best_params
