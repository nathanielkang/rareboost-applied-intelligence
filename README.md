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

| Path | Purpose |
|------|---------|
| `2_Code/` | Python package, configs, and scripts (this is what you run) |
| `1_Manuscript/` | LaTeX paper sources and bibliography (optional if you only need the code) |

See `2_Code/README.md` for metrics, method summary, and package tree.

## Citation

Cite the paper that matches the version you use. Draft bibliography entries are in `1_Manuscript/references.bib`; update from the publisher landing page once the article is official.
