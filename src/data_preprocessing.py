"""Data loading, validation, cleaning and leakage-safe feature building for CarWise.

Two kinds of preparation live here:

1. Deterministic row-level cleaning (``clean_dataset``). These rules do not learn
   anything from the data distribution, so they are safe to apply before the
   train/validation/test split.
2. Learned preprocessing (``FeatureBuilder``). Imputation statistics, category
   lists, target encodings and the scaler are fitted on TRAINING rows only and
   then re-used unchanged for validation, test and live predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "cardekho_dataset.csv"

TARGET = "selling_price"
NUMERIC_COLS = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
HIGH_CARD_CATS = ["brand", "model"]
LOW_CARD_CATS = ["seller_type", "fuel_type", "transmission_type"]
CATEGORICAL_COLS = HIGH_CARD_CATS + LOW_CARD_CATS
FEATURE_COLUMNS = CATEGORICAL_COLS + NUMERIC_COLS
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET]

KM_THRESHOLD = 500_000


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def validate_columns(df: pd.DataFrame) -> None:
    """Raise a readable error if required columns are missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required column(s): "
            + ", ".join(missing)
            + ". Found columns: "
            + ", ".join(map(str, df.columns))
        )


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Read the CarDekho CSV, drop index leftovers and check the schema."""
    csv_path = Path(path) if path else DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: '{csv_path}'. Place 'cardekho_dataset.csv' inside the "
            "'data' folder of the CarWise project and run again."
        )
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    leftovers = [c for c in df.columns if c.lower().startswith("unnamed")]
    df = df.drop(columns=leftovers)
    validate_columns(df)
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Force numeric columns to numbers and tidy text columns."""
    out = df.copy()
    for col in NUMERIC_COLS + [TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_COLS:
        s = out[col].astype(object)
        out[col] = s.where(s.isna(), s.astype(str).str.strip())
    return out


def iqr_outlier_count(series: pd.Series) -> int:
    """Number of values outside 1.5 x IQR whiskers."""
    s = series.dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def inspect_data(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise shape, types, statistics, missing values and suspicious rows."""
    d = coerce_types(df)
    invalid = {
        "seats = 0": int((d["seats"] == 0).sum()),
        f"km_driven > {KM_THRESHOLD:,}": int((d["km_driven"] > KM_THRESHOLD).sum()),
        "selling_price missing or <= 0": int((d[TARGET].isna() | (d[TARGET] <= 0)).sum()),
        "engine / max_power / mileage <= 0": int(
            (d[["engine", "max_power", "mileage"]] <= 0).any(axis=1).sum()
        ),
        "vehicle_age < 0": int((d["vehicle_age"] < 0).sum()),
        "fuel_type = 'Electric'": int((d["fuel_type"] == "Electric").sum()),
    }
    outliers = {col: iqr_outlier_count(d[col]) for col in NUMERIC_COLS + [TARGET]}
    return {
        "shape": (int(df.shape[0]), int(df.shape[1])),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "describe": d[NUMERIC_COLS + [TARGET]].describe().T,
        "missing": d[REQUIRED_COLUMNS].isna().sum(),
        "duplicates": int(df.duplicated().sum()),
        "invalid": invalid,
        "outliers_iqr": outliers,
    }


# --------------------------------------------------------------------------- #
# Cleaning (deterministic rules only)
# --------------------------------------------------------------------------- #
def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the notebook's cleaning rules where the real data supports them.

    Returns the cleaned frame and a report listing how many rows each step touched.
    Seats = 0 is NOT fixed here: the replacement value is a statistic (per-model
    median) and must be learned from training rows only. See ``FeatureBuilder``.
    """
    steps: list[dict[str, Any]] = []

    def log(step: str, rows: int, action: str, detail: str = "") -> None:
        steps.append({"step": step, "rows_affected": int(rows), "action": action, "detail": detail})

    d = coerce_types(df)
    rows_initial = len(d)

    # 1. Exact duplicates (done before splitting so a listing cannot sit in train AND test)
    before = len(d)
    d = d.drop_duplicates().reset_index(drop=True)
    log("Duplicate records", before - len(d), "removed",
        "Exact duplicate rows would be over-weighted and could appear in both train and test.")

    # 2. Rows without a usable target
    bad_target = d[TARGET].isna() | (d[TARGET] <= 0)
    log("Missing or invalid selling price", bad_target.sum(), "removed",
        "A row without a valid target cannot be used for training or evaluation.")
    d = d[~bad_target].reset_index(drop=True)

    # 3. E-Class seat typo: only if the data clearly supports it
    typo = (d["model"] == "E-Class") & (d["seats"] == 2)
    others = d.loc[(d["model"] == "E-Class") & (d["seats"] != 2), "seats"].dropna()
    supported = (
        typo.any() and len(others) >= 10 and (others == 5).mean() >= 0.9
    )
    if supported:
        d.loc[typo, "seats"] = 5
        log("E-Class with 2 seats", typo.sum(), "corrected to 5",
            f"{len(others)} other E-Class rows, {(others == 5).mean():.0%} of them have 5 seats.")
    else:
        log("E-Class with 2 seats", 0, "no change",
            "No row matched, or the other E-Class rows do not clearly show 5 seats.")

    # 4. Extreme km_driven
    extreme = d["km_driven"] > KM_THRESHOLD
    sample = ""
    if extreme.any():
        s = d.loc[extreme, ["brand", "model", "vehicle_age", "km_driven"]].head(3)
        sample = "Examples: " + "; ".join(
            f"{r.brand} {r.model}, {int(r.vehicle_age) if pd.notna(r.vehicle_age) else '?'} yrs, "
            f"{int(r.km_driven):,} km" for r in s.itertuples()
        )
    log(f"km_driven above {KM_THRESHOLD:,}", extreme.sum(), "removed",
        sample or "No rows above the threshold.")
    d = d[~extreme].reset_index(drop=True)

    # 5. 'Electric' fuel label: investigate before correcting
    ev = d["fuel_type"] == "Electric"
    if ev.any():
        ev_df = d[ev]
        combos = ", ".join(
            f"{b} {m} x{n}" for (b, m), n in ev_df.groupby(["brand", "model"]).size().items()
        )
        has_engine = bool((ev_df["engine"].fillna(0) > 0).all())
        if has_engine and ev.sum() <= 20 and (d["fuel_type"] == "Petrol").any():
            d.loc[ev, "fuel_type"] = "Petrol"
            log("'Electric' fuel label", ev.sum(), "relabelled to Petrol",
                f"Rows: {combos}. Every one has a combustion engine size (engine > 0 cc), "
                "so they look like petrol hybrids. Please confirm for your CSV.")
        else:
            log("'Electric' fuel label", 0, "kept as is",
                f"Rows: {combos}. Not clearly mislabelled, so no change was made.")
    else:
        log("'Electric' fuel label", 0, "no change", "No 'Electric' rows found.")

    # 6. Non-positive technical specs are impossible -> treat as missing
    bad_spec_rows = 0
    for col in ["engine", "max_power", "mileage"]:
        bad = d[col] <= 0
        bad_spec_rows += int(bad.sum())
        d.loc[bad, col] = np.nan
    bad_age = d["vehicle_age"] < 0
    bad_spec_rows += int(bad_age.sum())
    d.loc[bad_age, "vehicle_age"] = np.nan
    log("Non-positive engine / power / mileage / age", bad_spec_rows, "set to missing",
        "Imputed later using training-set statistics only.")

    # 7. Report values that will be imputed later (not changed here)
    log("seats = 0", int((d["seats"] == 0).sum()), "deferred",
        "Replaced by the per-model median seat count computed on training rows only.")
    remaining_missing = int(d[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    log("Rows with other missing values", remaining_missing, "deferred",
        "Filled with training-set per-model medians (numbers) or 'Unknown' (text).")

    d = d.reset_index(drop=True)
    report = {
        "rows_initial": int(rows_initial),
        "rows_final": int(len(d)),
        "rows_removed": int(rows_initial - len(d)),
        "steps": steps,
    }
    return d, report


def cleaning_report_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Convert the cleaning report to a display-friendly table."""
    return pd.DataFrame(report["steps"]).rename(columns={
        "step": "Step", "rows_affected": "Rows affected", "action": "Action", "detail": "Details"})


# --------------------------------------------------------------------------- #
# Target encoding (training-only, out-of-fold for training rows)
# --------------------------------------------------------------------------- #
class TargetEncoder:
    """Smoothed median-price encoding for high-cardinality columns."""

    def __init__(self, columns: list[str], smoothing: float = 10.0):
        self.columns = columns
        self.smoothing = smoothing
        self.global_median_: float = 0.0
        self.maps_: dict[str, dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        y = pd.Series(np.asarray(y, dtype=float))
        X = X.reset_index(drop=True)
        self.global_median_ = float(y.median())
        self.maps_ = {}
        for col in self.columns:
            g = pd.DataFrame({"k": X[col].astype(str).values, "y": y.values})
            stats = g.groupby("k")["y"].agg(["median", "count"])
            enc = (stats["count"] * stats["median"] + self.smoothing * self.global_median_) / (
                stats["count"] + self.smoothing)
            self.maps_[col] = enc.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self.columns:
            out[f"{col}_price_enc"] = (
                X[col].astype(str).map(self.maps_[col]).fillna(self.global_median_).astype(float)
            )
        return out

    def fit_transform_oof(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                          seed: int = 42) -> pd.DataFrame:
        """Encode training rows with maps built from the OTHER folds (no self-leakage),
        then fit the final maps on all training rows for use on new data."""
        X = X.reset_index(drop=True)
        y = pd.Series(np.asarray(y, dtype=float))
        out = pd.DataFrame(index=X.index, columns=[f"{c}_price_enc" for c in self.columns],
                           dtype=float)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fit_idx, enc_idx in kf.split(X):
            fold = TargetEncoder(self.columns, self.smoothing).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            out.iloc[enc_idx] = fold.transform(X.iloc[enc_idx]).values
        self.fit(X, y)
        return out


# --------------------------------------------------------------------------- #
# Feature builder
# --------------------------------------------------------------------------- #
class FeatureBuilder:
    """Imputation + encoding + scaling, all learned from training rows only.

    use_target_encoding=False : brand/model are one-hot encoded (notebook models 1-5)
    use_target_encoding=True  : brand/model become smoothed median prices (models 6-7);
                                only the small categoricals are one-hot encoded.
    """

    def __init__(self, use_target_encoding: bool = False, n_splits: int = 5, seed: int = 42,
                 smoothing: float = 10.0):
        self.use_target_encoding = use_target_encoding
        self.n_splits = n_splits
        self.seed = seed
        self.smoothing = smoothing
        self.model_medians_: dict[str, dict[str, float]] = {}
        self.global_medians_: dict[str, float] = {}
        self.categories_: dict[str, list[str]] = {}
        self.target_encoder_: TargetEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []

    # -- imputation -------------------------------------------------------- #
    @staticmethod
    def _prepare_text(X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy().reset_index(drop=True)
        for col in CATEGORICAL_COLS:
            s = Xc[col].astype(object)
            Xc[col] = s.where(s.notna(), "Unknown").astype(str).str.strip()
        return Xc

    @staticmethod
    def _numeric_view(Xc: pd.DataFrame, col: str) -> pd.Series:
        s = pd.to_numeric(Xc[col], errors="coerce")
        if col in ("seats", "engine", "max_power", "mileage"):
            s = s.where(s > 0)
        return s

    def _fit_imputer(self, Xc: pd.DataFrame) -> None:
        self.model_medians_, self.global_medians_ = {}, {}
        for col in NUMERIC_COLS:
            s = self._numeric_view(Xc, col)
            self.global_medians_[col] = float(s.median())
            self.model_medians_[col] = s.groupby(Xc["model"]).median().dropna().to_dict()

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = self._prepare_text(X)
        for col in NUMERIC_COLS:
            s = self._numeric_view(Xc, col)
            s = s.fillna(Xc["model"].map(self.model_medians_[col]))
            s = s.fillna(self.global_medians_[col])
            Xc[col] = s.round() if col == "seats" else s
        return Xc

    # -- assembling -------------------------------------------------------- #
    def _onehot_columns(self) -> list[str]:
        return LOW_CARD_CATS if self.use_target_encoding else CATEGORICAL_COLS

    def _assemble(self, Xc: pd.DataFrame, encoded: pd.DataFrame | None) -> pd.DataFrame:
        parts = [Xc[NUMERIC_COLS].astype(float)]
        if encoded is not None:
            parts.append(encoded.astype(float))
        for col in self._onehot_columns():
            cat = pd.Series(pd.Categorical(Xc[col], categories=self.categories_[col]),
                            index=Xc.index)
            parts.append(pd.get_dummies(cat, prefix=col, dtype=float))
        return pd.concat(parts, axis=1)

    # -- public API -------------------------------------------------------- #
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit on training rows and return the scaled training matrix."""
        X = X.reset_index(drop=True)
        y = pd.Series(np.asarray(y, dtype=float))
        Xc = self._prepare_text(X)
        self._fit_imputer(Xc)
        Xc = self._impute(X)
        self.categories_ = {col: sorted(Xc[col].unique().tolist()) for col in self._onehot_columns()}
        encoded = None
        if self.use_target_encoding:
            self.target_encoder_ = TargetEncoder(HIGH_CARD_CATS, self.smoothing)
            encoded = self.target_encoder_.fit_transform_oof(Xc, y, self.n_splits, self.seed)
        matrix = self._assemble(Xc, encoded)
        self.feature_names_ = list(matrix.columns)
        self.scaler_ = StandardScaler().fit(matrix.values)
        return pd.DataFrame(self.scaler_.transform(matrix.values), columns=self.feature_names_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the already-fitted preprocessing to new rows (validation/test/live)."""
        if self.scaler_ is None:
            raise RuntimeError("FeatureBuilder must be fitted before transform().")
        Xc = self._impute(X)
        encoded = self.target_encoder_.transform(Xc) if self.use_target_encoding else None
        matrix = self._assemble(Xc, encoded).reindex(columns=self.feature_names_, fill_value=0.0)
        return pd.DataFrame(self.scaler_.transform(matrix.values), columns=self.feature_names_)
