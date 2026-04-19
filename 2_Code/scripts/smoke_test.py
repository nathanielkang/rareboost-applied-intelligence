#!/usr/bin/env python
"""Smoke test for RareBoost.

Generates synthetic imbalanced data (lognormal targets), trains RareBoost
with 10 XGBoost trees, and checks that predictions are sane and SERA is
finite.  Also compares against vanilla XGBoost-MSE.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

PASS = True


def _check(name: str, condition: bool, detail: str = "") -> None:
    global PASS
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    if not condition:
        PASS = False


def main() -> None:
    global PASS

    rng = np.random.default_rng(42)
    n_samples, n_features = 200, 5
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = np.exp(rng.normal(2.0, 1.5, n_samples))

    print("=" * 60)
    print("RareBoost Smoke Test")
    print("=" * 60)

    # ---- Test 1: RelevanceFunction ----
    print("\n[1] RelevanceFunction")
    from src.rareboost.relevance import RelevanceFunction

    rf = RelevanceFunction(smoothing=True)
    rf.fit(y)
    phi = rf.transform(y)
    _check("phi shape", phi.shape == y.shape, f"{phi.shape}")
    _check("phi in [0,1]", float(phi.min()) >= -1e-6 and float(phi.max()) <= 1.0 + 1e-6)
    _check("rare mask non-empty", rf.get_rare_mask(y).sum() > 0,
           f"{rf.get_rare_mask(y).sum()} rare samples")

    # ---- Test 2: RareBoost XGBoost backend ----
    print("\n[2] RareBoost (XGBoost, relevance_mse, 10 trees)")
    from src.rareboost.rareboost import RareBoost

    model = RareBoost(
        booster="xgboost",
        loss="relevance_mse",
        n_estimators=10,
        max_depth=4,
        learning_rate=0.1,
        seed=42,
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    _check("pred shape", y_pred.shape == y.shape, f"{y_pred.shape}")
    _check("pred finite", np.all(np.isfinite(y_pred)))

    # ---- Test 3: Metrics ----
    print("\n[3] Metrics")
    from src.evaluation.metrics import evaluate_all

    metrics = evaluate_all(y, y_pred, model.get_relevance_function())
    for k, v in metrics.items():
        _check(f"{k} finite", np.isfinite(v) if not np.isnan(v) else True, f"{v:.4f}")

    sera_rb = metrics["sera"]

    # ---- Test 4: Vanilla XGBoost baseline comparison ----
    print("\n[4] Vanilla XGBoost-MSE baseline comparison")
    import xgboost as xgb

    dtrain = xgb.DMatrix(X, label=y)
    params = {"max_depth": 4, "eta": 0.1, "verbosity": 0}
    bst = xgb.train(params, dtrain, num_boost_round=10)
    y_pred_vanilla = bst.predict(dtrain)

    rf_for_vanilla = RelevanceFunction(smoothing=True)
    rf_for_vanilla.fit(y)
    from src.evaluation.metrics import sera as sera_fn
    sera_vanilla = sera_fn(y, y_pred_vanilla, rf_for_vanilla)
    _check("SERA finite (vanilla)", np.isfinite(sera_vanilla), f"{sera_vanilla:.4f}")
    _check("SERA finite (RareBoost)", np.isfinite(sera_rb), f"{sera_rb:.4f}")
    print(f"  SERA  RareBoost={sera_rb:.4f}  Vanilla={sera_vanilla:.4f}")

    # ---- Test 5: Other loss variants ----
    print("\n[5] Loss variants")
    for loss_name in ["relevance_mae", "relevance_huber", "focal_regression"]:
        m = RareBoost(booster="xgboost", loss=loss_name, n_estimators=5, max_depth=3, seed=42)
        m.fit(X, y)
        yp = m.predict(X)
        _check(f"{loss_name} runs", np.all(np.isfinite(yp)))

    # ---- Test 6: LightGBM backend ----
    print("\n[6] LightGBM backend")
    try:
        m_lgb = RareBoost(booster="lightgbm", loss="relevance_mse",
                          n_estimators=10, max_depth=4, learning_rate=0.1, seed=42)
        m_lgb.fit(X, y)
        yp_lgb = m_lgb.predict(X)
        _check("LightGBM pred finite", np.all(np.isfinite(yp_lgb)))
    except ImportError:
        print("  [SKIP] LightGBM not installed")

    # ---- Summary ----
    print("\n" + "=" * 60)
    if PASS:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 60)
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
