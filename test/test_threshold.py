import numpy as np
from src.model import _best_threshold

def test_best_threshold_range():
    y = np.array([0,1,0,1,0,1,0,1])
    p = np.array([0.1,0.9,0.2,0.8,0.3,0.7,0.4,0.6])
    t, m = _best_threshold(y, p)
    assert 0.35 <= t <= 0.65
