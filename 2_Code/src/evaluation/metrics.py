"""Evaluation metrics for imbalanced regression.

The primary metric is **SERA** (Squared Error–Relevance Area), which
integrates the squared error over a sweep of relevance thresholds so
that rare‐region accuracy is weighted more heavily.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..rareboost.relevance import RelevanceFunction


def sera(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    relevance_fn: RelevanceFunction,
    n_thresholds: int = 101,
) -> float:
    """Squared Error–Relevance Area.

    SERA = ∫₀¹ Σ_{i : φ(yᵢ)≥t} (yᵢ − ŷᵢ)² dt

    Approximated by the trapezoidal rule over *n_thresholds* steps.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    phi = relevance_fn.transform(y_true)
    sq_err = (y_true - y_pred) ** 2

    thresholds = np.linspace(0, 1, n_thresholds)
    dt = thresholds[1] - thresholds[0]

    sorted_idx = np.argsort(-phi)
    phi_sorted = phi[sorted_idx]
    sq_sorted = sq_err[sorted_idx]
    cum_sq = np.cumsum(sq_sorted)

    values = np.empty(n_thresholds)
    for i, t in enumerate(thresholds):
        count = np.searchsorted(-phi_sorted, -t, side="right")
        values[i] = cum_sq[count - 1] if count > 0 else 0.0

    try:
        integral = np.trapezoid(values, thresholds)
    except AttributeError:
        integral = np.trapz(values, thresholds)
    return float(integral)


def mse_rare(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    relevance_fn: RelevanceFunction,
    threshold: float = 0.5,
) -> float:
    """MSE restricted to samples with φ(y) > threshold."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = relevance_fn.get_rare_mask(y_true, threshold)
    if not mask.any():
        return float("nan")
    return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def mae_rare(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    relevance_fn: RelevanceFunction,
    threshold: float = 0.5,
) -> float:
    """MAE restricted to samples with φ(y) > threshold."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = relevance_fn.get_rare_mask(y_true, threshold)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def mse_overall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def mae_overall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman_rare(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    relevance_fn: RelevanceFunction,
    threshold: float = 0.5,
) -> float:
    """Spearman rank correlation on the rare region."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = relevance_fn.get_rare_mask(y_true, threshold)
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(rho)


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    relevance_fn: RelevanceFunction,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute every metric and return as a dict."""
    return {
        "sera": sera(y_true, y_pred, relevance_fn),
        "mse_rare": mse_rare(y_true, y_pred, relevance_fn, threshold),
        "mae_rare": mae_rare(y_true, y_pred, relevance_fn, threshold),
        "mse_overall": mse_overall(y_true, y_pred),
        "mae_overall": mae_overall(y_true, y_pred),
        "spearman_rare": spearman_rare(y_true, y_pred, relevance_fn, threshold),
    }
