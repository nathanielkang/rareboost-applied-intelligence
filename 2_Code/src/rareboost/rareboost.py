"""Main RareBoost wrapper — gradient boosting with relevance‐weighted losses.

Supports XGBoost and LightGBM backends.  The wrapper:

1. Fits a :class:`RelevanceFunction` on the training targets.
2. Constructs a custom objective that up‐weights rare samples.
3. Trains the booster with optional early stopping on SERA.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .relevance import RelevanceFunction
from .objectives import (
    make_relevance_weighted_mse_xgb,
    make_relevance_weighted_mse_lgb,
    make_relevance_weighted_mae_xgb,
    make_relevance_weighted_huber_xgb,
    make_focal_regression_xgb,
    make_relevance_weighted_eval_xgb,
)

_LOSS_FACTORIES_XGB = {
    "relevance_mse": make_relevance_weighted_mse_xgb,
    "relevance_mae": make_relevance_weighted_mae_xgb,
}

_LOSS_FACTORIES_LGB = {
    "relevance_mse": make_relevance_weighted_mse_lgb,
}


class RareBoost:
    """Gradient boosting with relevance‐weighted custom losses.

    Parameters
    ----------
    booster : str
        ``'xgboost'`` or ``'lightgbm'``.
    loss : str
        One of ``'relevance_mse'``, ``'relevance_mae'``, ``'relevance_huber'``,
        ``'focal_regression'``.
    relevance_method : str
        Passed to :class:`RelevanceFunction`.
    relevance_threshold : float
        Default threshold for rare / common split.
    n_bins : int
        Number of bins for adaptive binning (unused in loss but stored for
        downstream analysis).
    adaptive_bandwidth : bool
        Use skewness‐adjusted KDE bandwidth.
    smoothing : bool
        Apply Gaussian smoothing to the relevance curve.
    huber_delta : float
        Delta for the Huber loss variant.
    focal_gamma : float
        Gamma for the focal regression variant.
    **booster_params
        Extra keyword arguments forwarded to the underlying booster
        (e.g. ``n_estimators``, ``max_depth``, ``learning_rate``).
    """

    def __init__(
        self,
        booster: str = "xgboost",
        loss: str = "relevance_mse",
        relevance_method: str = "kde",
        relevance_threshold: float = 0.5,
        n_bins: int = 20,
        adaptive_bandwidth: bool = True,
        smoothing: bool = True,
        huber_delta: float = 1.0,
        focal_gamma: float = 2.0,
        **booster_params: Any,
    ) -> None:
        self.booster = booster.lower()
        self.loss = loss
        self.relevance_method = relevance_method
        self.relevance_threshold = relevance_threshold
        self.n_bins = n_bins
        self.adaptive_bandwidth = adaptive_bandwidth
        self.smoothing = smoothing
        self.huber_delta = huber_delta
        self.focal_gamma = focal_gamma
        self.booster_params = booster_params

        self._model = None
        self._relevance_fn: RelevanceFunction | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RareBoost":
        y_train = np.asarray(y_train, dtype=np.float64).ravel()

        if self.loss == "relevance_huber":
            target_scale = max(1.0, float(np.std(y_train)))
            self._effective_huber_delta = max(self.huber_delta, target_scale)
        else:
            self._effective_huber_delta = self.huber_delta

        self._relevance_fn = RelevanceFunction(
            method=self.relevance_method,
            bandwidth="adaptive" if self.adaptive_bandwidth else "silverman",
            threshold=self.relevance_threshold,
            smoothing=self.smoothing,
        )
        self._relevance_fn.fit(y_train)

        if self.booster == "xgboost":
            self._fit_xgb(X_train, y_train, X_val, y_val)
        elif self.booster == "lightgbm":
            self._fit_lgb(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown booster: {self.booster!r}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        if self.booster == "xgboost":
            import xgboost as xgb
            if isinstance(X, xgb.DMatrix):
                dmat = X
            else:
                dmat = xgb.DMatrix(np.asarray(X))
            return self._model.predict(dmat)
        else:
            return self._model.predict(np.asarray(X))

    def get_relevance_function(self) -> RelevanceFunction:
        if self._relevance_fn is None:
            raise RuntimeError("Call fit() first.")
        return self._relevance_fn

    def get_feature_importance(self, importance_type: str = "gain") -> dict:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        if self.booster == "xgboost":
            return self._model.get_score(importance_type=importance_type)
        else:
            import lightgbm as lgb
            return dict(
                zip(
                    self._model.feature_name(),
                    self._model.feature_importance(importance_type=importance_type.replace("gain", "split")),
                )
            )

    def _fit_xgb(self, X_train, y_train, X_val, y_val):
        import xgboost as xgb

        obj_fn = self._make_xgb_objective()
        eval_fn = make_relevance_weighted_eval_xgb(self._relevance_fn)

        params = self._xgb_params()
        n_estimators = self.booster_params.get("n_estimators", 500)
        early_stopping_rounds = self.booster_params.get("early_stopping_rounds", None)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(np.asarray(X_val), label=np.asarray(y_val, dtype=np.float64).ravel())
            evals.append((dval, "val"))

        train_kwargs: dict[str, Any] = dict(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            obj=obj_fn,
            custom_metric=eval_fn,
            evals=evals,
            verbose_eval=False,
        )
        if early_stopping_rounds and X_val is not None:
            train_kwargs["early_stopping_rounds"] = early_stopping_rounds

        self._model = xgb.train(**train_kwargs)

    def _fit_lgb(self, X_train, y_train, X_val, y_val):
        import lightgbm as lgb

        obj_fn = self._make_lgb_objective()
        params = self._lgb_params()
        params["objective"] = obj_fn
        n_estimators = self.booster_params.get("n_estimators", 500)
        early_stopping_rounds = self.booster_params.get("early_stopping_rounds", None)

        train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        valid_sets = [train_set]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(
                np.asarray(X_val),
                label=np.asarray(y_val, dtype=np.float64).ravel(),
                free_raw_data=False,
            )
            valid_sets.append(val_set)
            valid_names.append("val")

        callbacks = []
        if early_stopping_rounds and X_val is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(period=0))

        self._model = lgb.train(
            params=params,
            train_set=train_set,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def _make_xgb_objective(self):
        rf = self._relevance_fn
        if self.loss == "relevance_mse":
            return make_relevance_weighted_mse_xgb(rf)
        elif self.loss == "relevance_mae":
            return make_relevance_weighted_mae_xgb(rf)
        elif self.loss == "relevance_huber":
            return make_relevance_weighted_huber_xgb(rf, delta=self._effective_huber_delta)
        elif self.loss == "focal_regression":
            return make_focal_regression_xgb(rf, gamma=self.focal_gamma)
        else:
            raise ValueError(f"Unknown loss: {self.loss!r}")

    def _make_lgb_objective(self):
        rf = self._relevance_fn
        if self.loss == "relevance_mse":
            return make_relevance_weighted_mse_lgb(rf)
        else:
            raise ValueError(
                f"LightGBM backend only supports 'relevance_mse'; got {self.loss!r}"
            )

    def _xgb_params(self) -> dict:
        defaults = {
            "max_depth": 6,
            "eta": 0.1,
            "verbosity": 0,
            "disable_default_eval_metric": True,
        }
        for k in ("max_depth", "learning_rate", "eta", "subsample",
                   "colsample_bytree", "min_child_weight", "reg_alpha",
                   "reg_lambda", "gamma", "seed"):
            if k in self.booster_params:
                defaults[k] = self.booster_params[k]
        if "learning_rate" in defaults and "eta" not in self.booster_params:
            defaults["eta"] = defaults.pop("learning_rate")
        return defaults

    def _lgb_params(self) -> dict:
        defaults = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "verbosity": -1,
            "num_threads": 1,
        }
        for k in ("max_depth", "learning_rate", "num_leaves", "subsample",
                   "colsample_bytree", "min_child_samples", "reg_alpha",
                   "reg_lambda", "seed"):
            if k in self.booster_params:
                defaults[k] = self.booster_params[k]
        return defaults


class RareBoostCV:
    """Cross‐validation helper for joint tuning of booster and relevance params.

    Parameters
    ----------
    param_grid : dict
        Keys are RareBoost constructor argument names; values are lists
        of candidate values.
    n_folds : int
        Number of CV folds.
    metric : str
        Metric to optimize (``'sera'``, ``'mse_rare'``, ``'mae_rare'``).
    random_state : int or None
        Random seed for fold splitting.
    """

    def __init__(
        self,
        param_grid: dict[str, list],
        n_folds: int = 5,
        metric: str = "sera",
        random_state: int | None = None,
    ) -> None:
        self.param_grid = param_grid
        self.n_folds = n_folds
        self.metric = metric
        self.random_state = random_state

        self.best_params_: dict | None = None
        self.best_score_: float | None = None
        self.cv_results_: list[dict] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RareBoostCV":
        from sklearn.model_selection import KFold
        from itertools import product
        from ..evaluation.metrics import sera, mse_rare, mae_rare

        metric_fn_map = {
            "sera": lambda yt, yp, rf: sera(yt, yp, rf),
            "mse_rare": lambda yt, yp, rf: mse_rare(yt, yp, rf),
            "mae_rare": lambda yt, yp, rf: mae_rare(yt, yp, rf),
        }
        metric_fn = metric_fn_map[self.metric]

        keys = sorted(self.param_grid.keys())
        combos = list(product(*(self.param_grid[k] for k in keys)))

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        results: list[dict] = []

        for combo in combos:
            params = dict(zip(keys, combo))
            fold_scores = []
            for train_idx, val_idx in kf.split(X):
                X_tr, X_va = X[train_idx], X[val_idx]
                y_tr, y_va = y[train_idx], y[val_idx]
                model = RareBoost(**params)
                model.fit(X_tr, y_tr, X_va, y_va)
                y_pred = model.predict(X_va)
                score = metric_fn(y_va, y_pred, model.get_relevance_function())
                fold_scores.append(score)
            mean_score = float(np.mean(fold_scores))
            results.append({"params": params, "mean_score": mean_score, "fold_scores": fold_scores})

        self.cv_results_ = results
        best = min(results, key=lambda r: r["mean_score"])
        self.best_params_ = best["params"]
        self.best_score_ = best["mean_score"]
        return self
