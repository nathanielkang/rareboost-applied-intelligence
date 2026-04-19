# RareBoost

Relevance-weighted gradient boosting for **imbalanced regression**: custom XGBoost / LightGBM objectives that up-weight low-density (rare) regions of the target, with optional adaptive KDE relevance weights.

## Install

```bash
cd 2_Code
pip install -r requirements.txt
```

## Run

Smoke test (synthetic data, short run):

```bash
python scripts/smoke_test.py
```

Full benchmark (writes `results/` locally):

```bash
python scripts/run_experiments.py --output_dir results
```

Optional subset:

```bash
python scripts/run_experiments.py --datasets abalone california_housing --seeds 3 --output_dir results
```

## Layout

This repository tracks **`2_Code/`** only (implementation). The paper lives in the private submission pack, not on GitHub.

See `2_Code/README.md` for metrics, method summary, and package tree.

## Citation

Cite the published article when available (journal landing page and DOI). This repo does not host the manuscript or BibTeX sources.
