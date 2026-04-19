#!/usr/bin/env python
"""Full experiment runner for RareBoost.

Usage
-----
    python scripts/run_experiments.py --datasets abalone california_housing --seeds 5

Trains RareBoost + baselines on each dataset with multiple seeds and
writes results to CSV and JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.datasets import load_dataset, get_dataset_list
from src.rareboost.rareboost import RareBoost
from src.rareboost.relevance import RelevanceFunction
from src.evaluation.metrics import evaluate_all


def _vanilla_xgb(X_train, y_train, X_test, seed):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    params = {"max_depth": 6, "eta": 0.1, "verbosity": 0, "seed": seed}
    bst = xgb.train(params, dtrain, num_boost_round=500)
    return bst.predict(dtest)


def _vanilla_xgb_mae(X_train, y_train, X_test, seed):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    params = {"max_depth": 6, "eta": 0.1, "verbosity": 0, "seed": seed,
              "objective": "reg:absoluteerror"}
    bst = xgb.train(params, dtrain, num_boost_round=500)
    return bst.predict(dtest)


def _vanilla_xgb_huber(X_train, y_train, X_test, seed):
    """Custom Huber objective with smooth δ/|r| hessian decay.

    XGBoost's built-in reg:pseudohubererror has a known hessian collapse
    for large residuals that causes leaf weights to explode.  This manual
    implementation uses the same δ/|r| hessian approximation as
    RareBoost-Huber (without relevance weighting).
    """
    import xgboost as xgb

    delta = max(1.0, float(np.std(y_train)))

    def _huber_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        residual = y_true - y_pred
        abs_r = np.abs(residual)
        quad_mask = abs_r <= delta

        grad = np.where(quad_mask, -2.0 * residual, -2.0 * delta * np.sign(residual))
        hess = np.where(quad_mask, 2.0 * np.ones_like(residual),
                        2.0 * (delta / np.maximum(abs_r, delta)))
        grad = np.clip(grad, -1e4, 1e4)
        hess = np.maximum(hess, 1e-7)
        return grad, hess

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    params = {"max_depth": 6, "eta": 0.1, "verbosity": 0, "seed": seed,
              "disable_default_eval_metric": True}
    bst = xgb.train(params, dtrain, num_boost_round=500, obj=_huber_obj)
    return bst.predict(dtest)


def _vanilla_lgb(X_train, y_train, X_test, seed):
    import lightgbm as lgb
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    params = {"max_depth": 6, "learning_rate": 0.1, "verbosity": -1,
              "seed": seed, "num_threads": 1}
    bst = lgb.train(params, train_set, num_boost_round=500,
                     callbacks=[lgb.log_evaluation(period=0)])
    return bst.predict(X_test)


def _random_forest(X_train, y_train, X_test, seed):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf.predict(X_test)


def _smogn_xgb(X_train, y_train, X_test, seed):
    """SMOGN-style oversampling + XGBoost.

    Uses inline relevance-based SMOTE to oversample rare-region targets.
    Falls back to vanilla XGB if too few rare samples exist.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)

    rf = RelevanceFunction(smoothing=True)
    rf.fit(y_train)
    phi = rf.transform(y_train)
    rare_mask = phi > 0.3
    n_rare = int(rare_mask.sum())

    if n_rare >= 5 and n_rare < len(y_train) - 5:
        X_rare = X_train[rare_mask]
        y_rare = y_train[rare_mask]
        k = min(5, n_rare - 1)

        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_rare)
        _, indices = nn.kneighbors(X_rare)

        n_synthetic = min(n_rare * 2, len(y_train))
        X_syn = np.empty((n_synthetic, X_train.shape[1]), dtype=np.float32)
        y_syn = np.empty(n_synthetic, dtype=np.float64)

        for i in range(n_synthetic):
            base_idx = rng.integers(0, n_rare)
            neighbor_idx = indices[base_idx, rng.integers(1, k + 1)]
            lam = rng.uniform(0, 1)
            X_syn[i] = X_rare[base_idx] + lam * (X_rare[neighbor_idx] - X_rare[base_idx])
            y_syn[i] = y_rare[base_idx] + lam * (y_rare[neighbor_idx] - y_rare[base_idx])

        X_aug = np.vstack([X_train, X_syn])
        y_aug = np.concatenate([y_train, y_syn])
    else:
        X_aug, y_aug = X_train, y_train

    import xgboost as xgb
    dtrain = xgb.DMatrix(X_aug, label=y_aug)
    dtest = xgb.DMatrix(X_test)
    params = {"max_depth": 6, "eta": 0.1, "verbosity": 0, "seed": seed}
    bst = xgb.train(params, dtrain, num_boost_round=500)
    return bst.predict(dtest)


def _rebagg(X_train, y_train, X_test, seed):
    """Simple relevance-based bagging: bootstrap with relevance weights."""
    rng = np.random.default_rng(seed)
    rf = RelevanceFunction(smoothing=True)
    rf.fit(y_train)
    phi = rf.transform(y_train)
    weights = phi / phi.sum()

    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor

    idx = rng.choice(len(y_train), size=len(y_train), replace=True, p=weights)
    X_boot, y_boot = X_train[idx], y_train[idx]
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_boot, label=y_boot)
    dtest = xgb.DMatrix(X_test)
    params = {"max_depth": 6, "eta": 0.1, "verbosity": 0, "seed": seed}
    bst = xgb.train(params, dtrain, num_boost_round=500)
    return bst.predict(dtest)


BASELINES = {
    "XGB-MSE": _vanilla_xgb,
    "XGB-MAE": _vanilla_xgb_mae,
    "XGB-Huber": _vanilla_xgb_huber,
    "LGB-MSE": _vanilla_lgb,
    "RF": _random_forest,
    "SMOGN+XGB": _smogn_xgb,
    "REBAGG": _rebagg,
}

RAREBOOST_VARIANTS = {
    "RareBoost-MSE": {"loss": "relevance_mse"},
    "RareBoost-MAE": {"loss": "relevance_mae"},
    "RareBoost-Huber": {"loss": "relevance_huber"},
    "RareBoost-Focal": {"loss": "focal_regression"},
    "RareBoost-LGB": {"booster": "lightgbm", "loss": "relevance_mse"},
}


def run_single(
    dataset_name: str, seed: int, output_rows: list[dict]
) -> None:
    data = load_dataset(dataset_name, random_state=seed)
    X_tr, X_te = data["X_train"], data["X_test"]
    y_tr, y_te = data["y_train"], data["y_test"]

    rf = RelevanceFunction(smoothing=True)
    rf.fit(y_te)

    for method, fn in BASELINES.items():
        try:
            y_pred = fn(X_tr, y_tr, X_te, seed)
            metrics = evaluate_all(y_te, y_pred, rf)
        except Exception as e:
            metrics = {k: float("nan") for k in
                       ["sera", "mse_rare", "mae_rare", "mse_overall", "mae_overall", "spearman_rare"]}
            print(f"  WARN: {method} on {dataset_name} seed={seed} failed: {e}")
        output_rows.append({"dataset": dataset_name, "method": method, "seed": seed, **metrics})

    for method, extra_params in RAREBOOST_VARIANTS.items():
        try:
            kwargs = dict(
                n_estimators=500, max_depth=6, learning_rate=0.1, seed=seed
            )
            kwargs.update(extra_params)
            model = RareBoost(**kwargs)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            metrics = evaluate_all(y_te, y_pred, rf)
        except Exception as e:
            metrics = {k: float("nan") for k in
                       ["sera", "mse_rare", "mae_rare", "mse_overall", "mae_overall", "spearman_rare"]}
            print(f"  WARN: {method} on {dataset_name} seed={seed} failed: {e}")
        output_rows.append({"dataset": dataset_name, "method": method, "seed": seed, **metrics})


def main():
    parser = argparse.ArgumentParser(description="RareBoost experiments")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names (default: all)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds (default: 5)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    datasets = args.datasets or get_dataset_list()
    seeds = list(range(args.seeds))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets : {datasets}")
    print(f"Seeds    : {seeds}")
    print(f"Output   : {output_dir}")
    print()

    rows: list[dict] = []
    total = len(datasets) * len(seeds)
    pbar = tqdm(total=total, desc="Experiments")
    csv_path = output_dir / "experiment_results.csv"

    for ds in datasets:
        for s in seeds:
            t0 = time.time()
            run_single(ds, s, rows)
            elapsed = time.time() - t0
            pbar.set_postfix(dataset=ds, seed=s, time=f"{elapsed:.1f}s")
            pbar.update(1)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    pbar.close()
    df = pd.DataFrame(rows)

    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV -> {csv_path}")

    summary = (
        df.groupby(["dataset", "method"])
        .agg(["mean", "std"])
        .round(4)
    )
    summary_path = output_dir / "experiment_summary.csv"
    summary.to_csv(summary_path)
    print(f"Saved summary -> {summary_path}")

    json_path = output_dir / "experiment_results.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"Saved JSON -> {json_path}")

    print("\n--- Per-dataset SERA (mean +/- std) ---")
    sera_summary = df.groupby(["dataset", "method"])["sera"].agg(["mean", "std"]).round(4)
    print(sera_summary.to_string())


if __name__ == "__main__":
    main()
