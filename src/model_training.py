"""Model training, evaluation and artifact saving for CarWise.

Leakage rules followed here:
* The data is split ONCE into train / validation / test (70/15/15, fixed seed).
* Imputation, scaling, one-hot categories and target encodings are fitted on
  training rows only (``FeatureBuilder``); training rows get out-of-fold encodings.
* The validation set is used for XGBoost early stopping and to choose the model.
* The test set is used only for the final reported metrics.
* The top-1% price cap is computed from TRAINING prices only.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from .data_preprocessing import (
    CATEGORICAL_COLS,
    FEATURE_COLUMNS,
    NUMERIC_COLS,
    PROJECT_ROOT,
    TARGET,
    FeatureBuilder,
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "carwise_model.joblib"
REPORT_PATH = ARTIFACT_DIR / "training_report.joblib"

SEED = 42
TRIM_QUANTILE = 0.99
INTERVAL_LEVEL = 0.90
INTERVAL_FLOOR = 50_000.0  # rupees; stops tiny predictions giving unrealistically tight ranges


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """R², MAE and RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"r2": r2, "mae": float(np.mean(np.abs(resid))), "rmse": float(np.sqrt(np.mean(resid ** 2)))}


def split_data(df: pd.DataFrame, seed: int = SEED, val_size: float = 0.15,
               test_size: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproducible train / validation / test split."""
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1.0 - test_size),
                                  random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


def build_boosted_model(seed: int = SEED, quick: bool = False) -> tuple[Any, str]:
    """XGBoost if it imports on this machine, otherwise a scikit-learn fallback."""
    n_trees = 300 if quick else 1000
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=n_trees, learning_rate=0.03, max_depth=6, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
            early_stopping_rounds=50, eval_metric="rmse", n_jobs=-1, random_state=seed,
        )
        return model, "XGBoost"
    except Exception:  # ImportError, or a missing DLL on Windows
        model = HistGradientBoostingRegressor(
            max_iter=n_trees, learning_rate=0.05, max_depth=8, min_samples_leaf=10,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=30, random_state=seed,
        )
        return model, "HistGradientBoosting"


def _fit_regressor(model: Any, boosted: bool, X_fit: pd.DataFrame, y_fit: np.ndarray,
                   X_stop: pd.DataFrame, y_stop: np.ndarray) -> None:
    if boosted and type(model).__name__ == "XGBRegressor":
        model.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], verbose=False)
    else:
        model.fit(X_fit, y_fit)


def _group_name(feature: str) -> str:
    if feature in NUMERIC_COLS or feature.endswith("_price_enc"):
        return feature
    for col in CATEGORICAL_COLS:
        if feature.startswith(col + "_"):
            return col
    return feature


def compute_importance(entry: dict[str, Any], val_df: pd.DataFrame, seed: int) -> pd.DataFrame | None:
    """Feature importance, with one-hot columns summed back to their original feature."""
    model, names = entry["model"], entry["feature_names"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        X_val = entry["builder"].transform(val_df[FEATURE_COLUMNS])[names]
        result = permutation_importance(model, X_val, val_df[TARGET].values, n_repeats=3,
                                        random_state=seed, scoring="r2")
        values = np.clip(result.importances_mean, 0, None)
    frame = pd.DataFrame({"feature": [_group_name(n) for n in names], "importance": values})
    frame = frame.groupby("feature", as_index=False)["importance"].sum()
    total = frame["importance"].sum()
    if total > 0:
        frame["importance"] = frame["importance"] / total
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def _run_experiment(name: str, builder: FeatureBuilder, model: Any, *, boosted: bool,
                    train_df: pd.DataFrame, stop_df: pd.DataFrame, val_df: pd.DataFrame,
                    test_df: pd.DataFrame, cap_mask: np.ndarray, trimmed: bool,
                    columns: list[str] | None = None) -> dict[str, Any]:
    X_tr = builder.fit_transform(train_df[FEATURE_COLUMNS], train_df[TARGET])
    X_val = builder.transform(val_df[FEATURE_COLUMNS])
    X_te = builder.transform(test_df[FEATURE_COLUMNS])
    X_stop = X_val if stop_df is val_df else builder.transform(stop_df[FEATURE_COLUMNS])
    if columns:
        X_tr, X_val, X_te, X_stop = X_tr[columns], X_val[columns], X_te[columns], X_stop[columns]

    y_tr = train_df[TARGET].values
    _fit_regressor(model, boosted, X_tr, y_tr, X_stop, stop_df[TARGET].values)

    pred_val = np.asarray(model.predict(X_val), dtype=float)
    pred_te = np.asarray(model.predict(X_te), dtype=float)
    y_val, y_te = val_df[TARGET].values, test_df[TARGET].values
    return {
        "name": name, "builder": builder, "model": model, "trimmed": trimmed,
        "feature_names": list(X_tr.columns), "train_df": train_df,
        "pred_val": pred_val, "pred_test": pred_te,
        "val": regression_metrics(y_val, pred_val),
        "test": regression_metrics(y_te, pred_te),
        "test_capped": regression_metrics(y_te[cap_mask], pred_te[cap_mask]),
    }


def conformal_interval(y_val: np.ndarray, pred_val: np.ndarray, y_test: np.ndarray,
                       pred_test: np.ndarray, level: float = INTERVAL_LEVEL) -> dict[str, float]:
    """Split-conformal prediction interval using relative errors on the validation set.

    interval = prediction +/- q * max(prediction, floor), where q is the
    ceil((n+1)*level)/n quantile of |error| / max(prediction, floor).
    Coverage is re-measured on the untouched test set and stored.
    """
    def scores(y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.abs(y - p) / np.maximum(p, INTERVAL_FLOOR)

    s = np.sort(scores(y_val, pred_val))
    n = len(s)
    k = min(n, math.ceil((n + 1) * level))
    q = float(s[k - 1])
    coverage = float(np.mean(scores(y_test, pred_test) <= q))
    return {"level": level, "relative_halfwidth": q, "floor": INTERVAL_FLOOR,
            "test_coverage": coverage}


def build_metadata(entry: dict[str, Any], price_cap: float, interval: dict[str, float],
                   engine_label: str, selection_rule: str, seed: int) -> dict[str, Any]:
    """Everything the web app needs to build the form and validate inputs."""
    train = entry["train_df"]
    ranges, medians = {}, {}
    for col in NUMERIC_COLS:
        s = pd.to_numeric(train[col], errors="coerce").dropna()
        if col in ("seats", "engine", "max_power", "mileage"):
            s = s[s > 0]
        ranges[col] = [float(s.min()), float(s.max())]
        medians[col] = float(s.median())

    def mode(series: pd.Series) -> str:
        return str(series.mode().iloc[0])

    brand_models = {b: sorted(g["model"].astype(str).unique().tolist())
                    for b, g in train.groupby("brand")}
    defaults: dict[str, dict[str, Any]] = {}
    for model_name, g in train.groupby("model"):
        seats = pd.to_numeric(g["seats"], errors="coerce")
        seats = seats[seats > 0]
        defaults[str(model_name)] = {
            "brand": mode(g["brand"]),
            "engine": float(pd.to_numeric(g["engine"], errors="coerce").median()),
            "max_power": float(pd.to_numeric(g["max_power"], errors="coerce").median()),
            "mileage": float(pd.to_numeric(g["mileage"], errors="coerce").median()),
            "seats": float(seats.median()) if len(seats) else medians["seats"],
            "fuel_type": mode(g["fuel_type"]),
            "transmission_type": mode(g["transmission_type"]),
            "seller_type": mode(g["seller_type"]),
        }
    return {
        "model_name": entry["name"], "engine_label": engine_label, "seed": seed,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "selection_rule": selection_rule,
        "trimmed": bool(entry["trimmed"]),
        "price_cap": float(price_cap) if entry["trimmed"] else None,
        "train_price_range": [float(train[TARGET].min()), float(train[TARGET].max())],
        "numeric_ranges": ranges, "numeric_medians": medians,
        "brand_models": brand_models, "model_defaults": defaults,
        "categories": {c: sorted(train[c].dropna().astype(str).unique().tolist())
                       for c in ("fuel_type", "transmission_type", "seller_type")},
        "mode_categories": {c: mode(train[c])
                            for c in ("fuel_type", "transmission_type", "seller_type")},
        "feature_columns": FEATURE_COLUMNS, "interval": interval,
    }


def run_experiments(df: pd.DataFrame, seed: int = SEED, quick: bool = False,
                    progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Train and compare the seven models. Returns everything needed to save artifacts."""
    train, val, test = split_data(df, seed)
    price_cap = float(train[TARGET].quantile(TRIM_QUANTILE))  # training prices only
    cap_mask = (test[TARGET] <= price_cap).values
    progress(f"Split -> train {len(train):,} | validation {len(val):,} | test {len(test):,}")
    progress(f"Top-1% price cap (from training prices only): Rs {price_cap:,.0f}")
    progress(f"Test rows at or below the cap: {int(cap_mask.sum()):,} of {len(test):,}")

    _, boost_label = build_boosted_model(seed, quick)
    progress(f"Boosting engine: {boost_label}" + (" (quick mode)" if quick else ""))

    trim_train = train[train[TARGET] <= price_cap].reset_index(drop=True)
    trim_val = val[val[TARGET] <= price_cap].reset_index(drop=True)

    def boosted() -> Any:
        return build_boosted_model(seed, quick)[0]

    common = dict(val_df=val, test_df=test, cap_mask=cap_mask)
    specs: list[dict[str, Any]] = [
        dict(name="1. Simple Linear Regression (max_power)", builder=FeatureBuilder(False, seed=seed),
             model=LinearRegression(), boosted=False, columns=["max_power"],
             train_df=train, stop_df=val, trimmed=False),
        dict(name="2. Multiple Linear Regression", builder=FeatureBuilder(False, seed=seed),
             model=LinearRegression(), boosted=False, columns=None,
             train_df=train, stop_df=val, trimmed=False),
        dict(name="3. Decision Tree", builder=FeatureBuilder(False, seed=seed),
             model=DecisionTreeRegressor(max_depth=8, random_state=seed), boosted=False,
             columns=None, train_df=train, stop_df=val, trimmed=False),
        dict(name="4. Random Forest", builder=FeatureBuilder(False, seed=seed),
             model=RandomForestRegressor(n_estimators=100 if quick else 300, max_features=0.33,
                                         min_samples_leaf=1, n_jobs=-1, random_state=seed),
             boosted=False, columns=None, train_df=train, stop_df=val, trimmed=False),
        dict(name=f"5. {boost_label}", builder=FeatureBuilder(False, seed=seed),
             model=boosted(), boosted=True, columns=None, train_df=train, stop_df=val,
             trimmed=False),
        dict(name=f"6. {boost_label} + Target Encoding", builder=FeatureBuilder(True, seed=seed),
             model=boosted(), boosted=True, columns=None, train_df=train, stop_df=val,
             trimmed=False),
        dict(name=f"7. {boost_label} + Target Enc. + Top-1% Trim",
             builder=FeatureBuilder(True, seed=seed), model=boosted(), boosted=True,
             columns=None, train_df=trim_train, stop_df=trim_val, trimmed=True),
    ]

    entries: dict[str, dict[str, Any]] = {}
    for spec in specs:
        start = time.time()
        entry = _run_experiment(spec["name"], spec["builder"], spec["model"],
                                boosted=spec["boosted"], train_df=spec["train_df"],
                                stop_df=spec["stop_df"], trimmed=spec["trimmed"],
                                columns=spec["columns"], **common)
        entries[spec["name"]] = entry
        progress(f"  trained {spec['name']:<52} test R2 = {entry['test']['r2']:.4f} "
                 f"({time.time() - start:.1f}s)")

    rows = []
    for name, e in entries.items():
        rows.append({
            "Model": name,
            "R² (test)": e["test"]["r2"], "MAE (test)": e["test"]["mae"],
            "RMSE (test)": e["test"]["rmse"],
            "R² (test ≤ cap)": e["test_capped"]["r2"], "MAE (test ≤ cap)": e["test_capped"]["mae"],
            "RMSE (test ≤ cap)": e["test_capped"]["rmse"],
            "R² (val)": e["val"]["r2"], "RMSE (val)": e["val"]["rmse"],
            "Train rows": len(e["train_df"]),
        })
    comparison = pd.DataFrame(rows)

    best_name = min(entries, key=lambda n: entries[n]["val"]["rmse"])
    best = entries[best_name]
    selection_rule = ("Lowest RMSE on the validation set (full price range). "
                      "The test set was not used to choose the model.")
    progress(f"Selected model: {best_name}")

    interval = conformal_interval(val[TARGET].values, best["pred_val"], test[TARGET].values,
                                  best["pred_test"])
    progress(f"Prediction range: +/-{interval['relative_halfwidth']:.1%} (relative), "
             f"{interval['level']:.0%} target, {interval['test_coverage']:.1%} measured on test")

    metadata = build_metadata(best, price_cap, interval, boost_label, selection_rule, seed)
    importance = compute_importance(best, val, seed)
    return {
        "comparison": comparison, "best_name": best_name, "entries": entries,
        "metadata": metadata, "importance": importance, "price_cap": price_cap,
        "splits": {"train": len(train), "validation": len(val), "test": len(test),
                   "test_at_or_below_cap": int(cap_mask.sum())},
        "y_test": test[TARGET].values, "cap_mask": cap_mask,
        "engine_label": boost_label, "quick": quick, "seed": seed,
    }


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_artifacts(results: dict[str, Any], cleaning_report: dict[str, Any],
                   inspection: dict[str, Any]) -> None:
    """Write the deployable model bundle and the report used by the web app."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    best = results["entries"][results["best_name"]]
    bundle = {"builder": best["builder"], "model": best["model"], "metadata": results["metadata"]}
    joblib.dump(bundle, MODEL_PATH, compress=3)

    report = {
        "comparison": results["comparison"], "best_name": results["best_name"],
        "best_metrics": {"val": best["val"], "test": best["test"],
                         "test_capped": best["test_capped"]},
        "test_predictions": {n: e["pred_test"].astype("float32")
                             for n, e in results["entries"].items()},
        "y_test": results["y_test"].astype("float32"), "cap_mask": results["cap_mask"],
        "importance": results["importance"], "price_cap": results["price_cap"],
        "splits": results["splits"], "metadata": results["metadata"],
        "cleaning_report": cleaning_report, "inspection": inspection,
        "engine_label": results["engine_label"], "quick": results["quick"],
        "seed": results["seed"], "trained_at": results["metadata"]["trained_at"],
    }
    joblib.dump(report, REPORT_PATH, compress=3)
