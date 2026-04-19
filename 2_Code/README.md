# RareBoost — reference implementation

Implementation for the manuscript *Making Gradient Boosting Care About Rare Targets* (RareBoost): relevance-weighted objectives for gradient boosting in **imbalanced regression**, with optional adaptive KDE relevance weights for **XGBoost** and **LightGBM**.

This folder is the **code supplement** for journal submission: **no precomputed result tables** are included; run the scripts below to reproduce metrics on your machine.

## Setup

```bash
pip install -r requirements.txt
```

## Smoke test (quick)

```bash
python scripts/smoke_test.py
```

Uses a small synthetic dataset, short training, and checks that objectives and metrics run without error.

## Full benchmark

```bash
python scripts/run_experiments.py --output_dir results
```

Optional filters:

```bash
python scripts/run_experiments.py --datasets abalone california_housing --seeds 3 --output_dir results
```

Outputs under `--output_dir` (default `results/`): `experiment_results.csv`, `experiment_summary.csv`, and `experiment_results.json`.

## Layout

```
├── configs/default.yaml       # Default hyperparameters
├── scripts/
│   ├── smoke_test.py
│   └── run_experiments.py
├── src/
│   ├── rareboost/             # Core method
│   ├── data/datasets.py       # Public benchmark loaders
│   └── evaluation/metrics.py
└── requirements.txt
```

## Method (short)

A kernel density estimate on the target defines a relevance weight for each sample; custom boosting objectives reweight gradients toward low-density (rare) regions. Supported variants include relevance-weighted MSE, MAE, Huber, and focal-style regression; both XGBoost and LightGBM backends are supported.

## Metrics

| Metric | Role |
|--------|------|
| SERA | Squared error–relevance area (primary ranking metric in the paper) |
| MSE-rare / MAE-rare | Error on rare-region samples |
| MSE-overall / MAE-overall | Standard full-sample error |
| Spearman-rare | Rank correlation on the rare region |
