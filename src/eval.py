from __future__ import annotations
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_splits(n: int, k: int = 5):
    tscv = TimeSeriesSplit(n_splits=k)
    return list(tscv.split(np.arange(n)))
