from __future__ import annotations

import numpy as np


def collapse_time(y: np.ndarray, threshold: float) -> float:
    """First time index where y falls below threshold.

    Parameters
    ----------
    y:
        1D array of non-negative proxy abundance.
    threshold:
        Collapse threshold.

    Returns
    -------
    float
        Index (0-based) of collapse time, or np.inf if no collapse.
    """
    y = np.asarray(y)
    idx = np.where(y < threshold)[0]
    return float(idx[0]) if idx.size else float(np.inf)


def survival_probability(
    trajectories: np.ndarray, threshold: float, T: int | None = None
) -> float:
    """Fraction of trajectories that stay >= threshold up to time T."""
    trajectories = np.asarray(trajectories)
    if trajectories.ndim != 2:
        raise ValueError("trajectories must be 2D: (n_trajectories, n_time)")

    if T is None:
        T = trajectories.shape[1]
    T = int(T)

    alive = np.all(trajectories[:, :T] >= threshold, axis=1)
    return float(np.mean(alive))


def auc_delta(y_factual: np.ndarray, y_counterfactual: np.ndarray) -> float:
    """Difference in area under curve (counterfactual - factual)."""
    y_factual = np.asarray(y_factual)
    y_counterfactual = np.asarray(y_counterfactual)
    # np.trapezoid é o novo nome no NumPy 2.0+ (antes era np.trapz)
    return float(np.trapezoid(y_counterfactual) - np.trapezoid(y_factual))
