from __future__ import annotations

import numpy as np


def apply_delta_percent(x: np.ndarray, delta: float) -> np.ndarray:
    """Apply counterfactual multiplicative intervention: x' = x * (1 + delta)."""
    x = np.asarray(x)
    return x * (1.0 + float(delta))
