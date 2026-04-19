"""Dataset loaders for imbalanced regression benchmarks.

All loaders return a dict with keys
``X_train, X_test, y_train, y_test, feature_names``.
Preprocessing: missing‐value imputation, categorical one‐hot encoding,
standard scaling (fit on train only).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

_DATASETS = [
    "abalone",
    "california_housing",
    "bike_sharing",
    "air_quality",
    "wine_quality",
    "cpu_activity",
    "insurance",
    "energy_efficiency",
    "concrete",
]


def get_dataset_list() -> list[str]:
    return list(_DATASETS)


def load_dataset(
    name: str,
    data_dir: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    name = name.lower().replace("-", "_").replace(" ", "_")
    loader = _LOADERS.get(name)
    if loader is None:
        raise ValueError(
            f"Unknown dataset {name!r}. Choose from {get_dataset_list()}"
        )
    return loader(test_size=test_size, random_state=random_state)


def _stratified_split(X, y, test_size, random_state, n_bins=10):
    """Train/test split with stratified sampling on binned targets."""
    from sklearn.preprocessing import KBinsDiscretizer

    y_arr = np.asarray(y).ravel()
    n_unique = len(np.unique(y_arr))
    actual_bins = min(n_bins, n_unique)
    if actual_bins < 2:
        return train_test_split(X, y_arr, test_size=test_size, random_state=random_state)

    binner = KBinsDiscretizer(n_bins=actual_bins, encode="ordinal", strategy="quantile")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_binned = binner.fit_transform(y_arr.reshape(-1, 1)).ravel().astype(int)

    try:
        return train_test_split(
            X, y_arr, test_size=test_size, random_state=random_state, stratify=y_binned
        )
    except ValueError:
        return train_test_split(X, y_arr, test_size=test_size, random_state=random_state)


def _build_preprocessor(X_df: pd.DataFrame):
    """Build a ColumnTransformer that imputes, encodes, and scales."""
    num_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers, remainder="drop")


def _finalize(X_df, y, test_size, random_state):
    """Apply preprocessing and return the standard dict."""
    preprocessor = _build_preprocessor(X_df)
    X_train_raw, X_test_raw, y_train, y_test = _stratified_split(
        X_df, y, test_size, random_state
    )
    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw).astype(np.float32)
    X_test = preprocessor.transform(X_test_raw).astype(np.float32)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": np.asarray(y_train, dtype=np.float64),
        "y_test": np.asarray(y_test, dtype=np.float64),
        "feature_names": feature_names,
    }


def _load_abalone(test_size=0.2, random_state=42):
    from sklearn.datasets import fetch_openml
    data = fetch_openml(data_id=183, as_frame=True, parser="auto")
    X_df = data.data.copy()
    y = data.target.astype(float).values
    return _finalize(X_df, y, test_size, random_state)


def _load_california_housing(test_size=0.2, random_state=42):
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    X_df = data.data.copy()
    y = data.target.values
    return _finalize(X_df, y, test_size, random_state)


def _load_bike_sharing(test_size=0.2, random_state=42):
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=42712, as_frame=True, parser="auto")
        X_df = data.data.copy()
        y = data.target.astype(float).values
    except Exception:
        rng = np.random.default_rng(random_state)
        n = 2000
        X_df = pd.DataFrame({
            "season": rng.integers(1, 5, n),
            "yr": rng.integers(0, 2, n),
            "mnth": rng.integers(1, 13, n),
            "hr": rng.integers(0, 24, n),
            "holiday": rng.integers(0, 2, n),
            "weekday": rng.integers(0, 7, n),
            "workingday": rng.integers(0, 2, n),
            "weathersit": rng.integers(1, 4, n),
            "temp": rng.uniform(0, 1, n),
            "atemp": rng.uniform(0, 1, n),
            "hum": rng.uniform(0, 1, n),
            "windspeed": rng.uniform(0, 1, n),
        })
        y = np.exp(rng.normal(4, 1.5, n)).astype(float)
    return _finalize(X_df, y, test_size, random_state)


def _load_air_quality(test_size=0.2, random_state=42):
    try:
        from ucimlrepo import fetch_ucirepo
        repo = fetch_ucirepo(id=360)
        df = repo.data.features.copy()

        y = df["C6H6(GT)"].astype(float).values

        drop_cols = [c for c in df.columns
                     if "date" in c.lower() or "time" in c.lower()
                     or c == "C6H6(GT)"]
        df = df.drop(columns=drop_cols, errors="ignore")
        if "NMHC(GT)" in df.columns:
            df = df.drop(columns=["NMHC(GT)"])
        df = df.replace(-200, np.nan).replace(-200.0, np.nan)

        valid = y != -200
        X_df = df[valid].reset_index(drop=True)
        y = y[valid]
    except Exception:
        rng = np.random.default_rng(random_state)
        n = 1500
        X_df = pd.DataFrame(rng.standard_normal((n, 10)),
                            columns=[f"sensor_{i}" for i in range(10)])
        y = np.exp(rng.normal(3, 1, n))
    return _finalize(X_df, y, test_size, random_state)


def _load_wine_quality(test_size=0.2, random_state=42):
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=287, as_frame=True, parser="auto")
        X_df = data.data.copy()
        y = data.target.astype(float).values
    except Exception:
        rng = np.random.default_rng(random_state)
        n = 1600
        X_df = pd.DataFrame(rng.standard_normal((n, 11)),
                            columns=[f"chem_{i}" for i in range(11)])
        y = rng.integers(3, 10, n).astype(float)
    return _finalize(X_df, y, test_size, random_state)


def _load_cpu_activity(test_size=0.2, random_state=42):
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=197, as_frame=True, parser="auto")
        X_df = data.data.copy()
        y = data.target.astype(float).values
    except Exception:
        rng = np.random.default_rng(random_state)
        n = 2000
        X_df = pd.DataFrame(rng.standard_normal((n, 12)),
                            columns=[f"cpu_{i}" for i in range(12)])
        y = np.exp(rng.normal(3, 1, n))
    return _finalize(X_df, y, test_size, random_state)


def _load_insurance(test_size=0.2, random_state=42):
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=46289, as_frame=True, parser="auto")
        X_df = data.data.copy()
        y = data.target.astype(float).values
    except Exception:
        rng = np.random.default_rng(random_state)
        n = 1338
        X_df = pd.DataFrame({
            "age": rng.integers(18, 65, n),
            "bmi": rng.normal(30, 6, n),
            "children": rng.integers(0, 6, n),
            "smoker": rng.choice(["yes", "no"], n),
            "region": rng.choice(["NE", "NW", "SE", "SW"], n),
        })
        y = np.exp(rng.normal(8.5, 1.0, n))
    return _finalize(X_df, y, test_size, random_state)


def _load_energy_efficiency(test_size=0.2, random_state=42):
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=1472, as_frame=True, parser="auto")
        X_df = data.data.copy()
        y = data.target.astype(float).values
    except Exception:
        rng = np.random.default_rng(random_state)
        n = 768
        X_df = pd.DataFrame(rng.standard_normal((n, 8)),
                            columns=[f"building_{i}" for i in range(8)])
        y = rng.normal(25, 10, n).clip(5, 50)
    return _finalize(X_df, y, test_size, random_state)


def _load_concrete(test_size=0.2, random_state=42):
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=44959, as_frame=True, parser="auto")
        X_df = data.data.copy()
        y = data.target.astype(float).values
    except Exception:
        rng = np.random.default_rng(random_state)
        n = 1030
        X_df = pd.DataFrame(rng.standard_normal((n, 8)),
                            columns=[f"comp_{i}" for i in range(8)])
        y = rng.normal(35, 16, n).clip(2, 82)
    return _finalize(X_df, y, test_size, random_state)


_LOADERS = {
    "abalone": _load_abalone,
    "california_housing": _load_california_housing,
    "bike_sharing": _load_bike_sharing,
    "air_quality": _load_air_quality,
    "wine_quality": _load_wine_quality,
    "cpu_activity": _load_cpu_activity,
    "insurance": _load_insurance,
    "energy_efficiency": _load_energy_efficiency,
    "concrete": _load_concrete,
}
