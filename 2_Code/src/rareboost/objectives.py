"""Custom XGBoost / LightGBM objective functions weighted by relevance.

Every ``make_*`` factory returns a closure that captures the fitted
:class:`RelevanceFunction`.  Gradient / Hessian formulas follow directly
from differentiating the relevance‐weighted loss.

XGBoost API
-----------
``obj(y_pred, dtrain) -> (grad, hess)``
where ``dtrain`` is a :class:`xgboost.DMatrix`.

LightGBM API
-------------
``obj(y_true, y_pred) -> (grad, hess)``
"""

from __future__ import annotations

import numpy as np

from .relevance import RelevanceFunction


_EPS = 1e-7
_GRAD_CLIP = 1e4
_PHI_FLOOR = 0.05


def _floor_phi(phi: np.ndarray) -> np.ndarray:
    """Ensure every sample has at least _PHI_FLOOR relevance weight.

    Without this floor, common (high-density) samples receive phi≈0 and
    become invisible to the model, causing catastrophic SERA on datasets
    with large target range.
    """
    return np.maximum(phi, _PHI_FLOOR)


def make_relevance_weighted_mse_xgb(relevance_fn: RelevanceFunction):
    """Factory for XGBoost relevance‐weighted MSE objective.

    L_i = φ(y_i) · (y_i − ŷ_i)²
    grad = −2 φ(y_i)(y_i − ŷ_i)
    hess =  2 φ(y_i)
    """

    def _obj(y_pred: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        y_true = dtrain.get_label()
        phi = _floor_phi(relevance_fn.transform(y_true))
        residual = y_true - y_pred
        grad = -2.0 * phi * residual
        hess = 2.0 * phi
        grad = np.clip(grad, -_GRAD_CLIP, _GRAD_CLIP)
        hess = np.maximum(hess, _EPS)
        return grad, hess

    return _obj


def make_relevance_weighted_mse_lgb(relevance_fn: RelevanceFunction):
    """Factory for LightGBM relevance-weighted MSE objective.

    LightGBM 4.x passes custom objectives as ``fobj(y_pred, dataset)``
    when set via ``params["objective"]``.
    """

    def _obj(y_pred: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
        y_true = np.asarray(dataset.get_label(), dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        phi = _floor_phi(relevance_fn.transform(y_true))
        residual = y_true - y_pred
        grad = -2.0 * phi * residual
        hess = 2.0 * phi
        grad = np.clip(grad, -_GRAD_CLIP, _GRAD_CLIP)
        hess = np.maximum(hess, _EPS)
        return grad, hess

    return _obj


def make_relevance_weighted_mae_xgb(relevance_fn: RelevanceFunction):
    """Factory for XGBoost relevance‐weighted smooth MAE objective.

    Uses a Huber-like smooth approximation with a small adaptive δ so
    that the Hessian scales correctly with residual magnitude. Without
    this, the constant-Hessian MAE approximation causes leaf weights
    bounded to ±1 per tree, preventing convergence on large-scale targets.

    Quadratic region (|r| ≤ δ):  grad = −φ·r/δ,   hess = φ/δ
    Linear region   (|r| > δ):  grad = −φ·sign(r), hess = φ·δ/|r|
    """

    def _obj(y_pred: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        y_true = dtrain.get_label()
        phi = _floor_phi(relevance_fn.transform(y_true))
        residual = y_true - y_pred
        abs_r = np.abs(residual)

        delta = max(1.0, float(np.std(y_true)) * 0.1)
        quad_mask = abs_r <= delta

        grad = np.where(
            quad_mask,
            -phi * residual / delta,
            -phi * np.sign(residual),
        )

        hess = np.where(
            quad_mask,
            phi / delta,
            phi * (delta / np.maximum(abs_r, delta)),
        )

        grad = np.clip(grad, -_GRAD_CLIP, _GRAD_CLIP)
        hess = np.maximum(hess, _EPS)
        return grad, hess

    return _obj


def make_relevance_weighted_huber_xgb(
    relevance_fn: RelevanceFunction, delta: float = 1.0
):
    """Factory for XGBoost relevance‐weighted Huber objective.

    L_i = φ(y_i) · H_δ(r) where r = y_i − ŷ_i and:
        H_δ(r) = r²           if |r| ≤ δ
        H_δ(r) = 2δ|r| − δ²  if |r| > δ

    Both gradient and (approximate) Hessian are continuous at |r| = δ.
    The Hessian in the linear region uses a δ/|r| decay to prevent the
    leaf-weight explosion that occurs when Hessian ≈ 0.
    """

    def _obj(y_pred: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        y_true = dtrain.get_label()
        phi = _floor_phi(relevance_fn.transform(y_true))
        residual = y_true - y_pred
        abs_r = np.abs(residual)

        quad_mask = abs_r <= delta

        grad = np.where(
            quad_mask,
            -2.0 * phi * residual,
            -2.0 * phi * delta * np.sign(residual),
        )

        hess = np.where(
            quad_mask,
            2.0 * phi,
            2.0 * phi * (delta / np.maximum(abs_r, delta)),
        )

        grad = np.clip(grad, -_GRAD_CLIP, _GRAD_CLIP)
        hess = np.maximum(hess, _EPS)
        return grad, hess

    return _obj


def make_focal_regression_xgb(
    relevance_fn: RelevanceFunction, gamma: float = 2.0
):
    """Focal‐loss adaptation for regression.

    High‐error *and* high‐relevance samples receive the largest weight,
    but residuals are normalized by target scale to prevent the focal
    exponent from exploding on large-range datasets.

        w_i = φ(y_i) · (|r_i| / σ_y)^γ     (capped at 10)

    The underlying loss is MSE, so:

        grad = −2 w_i (y_i − ŷ_i)
        hess =  2 w_i
    """

    def _obj(y_pred: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        y_true = dtrain.get_label()
        phi = _floor_phi(relevance_fn.transform(y_true))
        residual = y_true - y_pred
        abs_r = np.abs(residual)

        y_scale = max(1.0, float(np.std(y_true)))
        abs_r_norm = abs_r / y_scale
        focal_weight = phi * (abs_r_norm + _EPS) ** gamma
        focal_weight = np.minimum(focal_weight, 10.0)

        grad = -2.0 * focal_weight * residual
        hess = 2.0 * focal_weight
        grad = np.clip(grad, -_GRAD_CLIP, _GRAD_CLIP)
        hess = np.maximum(hess, _EPS)
        return grad, hess

    return _obj


def make_relevance_weighted_eval_xgb(relevance_fn: RelevanceFunction):
    """Custom XGBoost evaluation metric based on SERA.

    Returns ``('sera', value)`` where lower is better.
    """

    def _eval(y_pred: np.ndarray, dtrain) -> tuple[str, float]:
        y_true = dtrain.get_label()
        phi = relevance_fn.transform(y_true)
        sq_err = (y_true - y_pred) ** 2

        thresholds = np.linspace(0, 1, 51)
        sera_val = 0.0
        dt = thresholds[1] - thresholds[0]
        for t in thresholds:
            mask = phi >= t
            if mask.any():
                sera_val += sq_err[mask].sum() * dt

        return "sera", float(sera_val)

    return _eval
