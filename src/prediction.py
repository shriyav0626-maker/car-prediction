"""Load the trained CarWise model and turn form inputs into a price estimate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data_preprocessing import CATEGORICAL_COLS, FEATURE_COLUMNS, NUMERIC_COLS
from .model_training import MODEL_PATH

# Physically impossible values are rejected; merely unusual values only trigger warnings.
HARD_LIMITS: dict[str, tuple[float, float, str]] = {
    "vehicle_age": (0, 40, "Vehicle age (years)"),
    "km_driven": (0, 1_000_000, "Kilometres driven"),
    "mileage": (1, 60, "Mileage (km per litre / kg)"),
    "engine": (200, 8000, "Engine capacity (cc)"),
    "max_power": (20, 1000, "Maximum power (bhp)"),
    "seats": (2, 10, "Number of seats"),
}
LABELS = {
    "vehicle_age": "vehicle age", "km_driven": "kilometres driven", "mileage": "mileage",
    "engine": "engine capacity", "max_power": "maximum power", "seats": "seat count",
}


def format_inr(value: float) -> str:
    """Format a number with Indian digit grouping, e.g. 575000 -> ₹5,75,000."""
    n = int(round(float(value)))
    sign = "-" if n < 0 else ""
    digits = str(abs(n))
    if len(digits) <= 3:
        return f"{sign}₹{digits}"
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return f"{sign}₹{','.join(groups + [tail])}"


@dataclass
class PredictionResult:
    price: float
    low: float | None = None
    high: float | None = None
    warnings: list[str] = field(default_factory=list)


class CarPricePredictor:
    """Wraps the saved bundle (fitted preprocessing + model + metadata)."""

    def __init__(self, bundle: dict[str, Any]):
        self.builder = bundle["builder"]
        self.model = bundle["model"]
        self.meta = bundle["metadata"]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "CarPricePredictor":
        model_path = Path(path) if path else MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Run 'python train_model.py' first to create it.")
        return cls(joblib.load(model_path))

    # -- metadata helpers used by the form --------------------------------- #
    @property
    def brands(self) -> list[str]:
        return sorted(self.meta["brand_models"])

    def models_for(self, brand: str) -> list[str]:
        return self.meta["brand_models"].get(brand, [])

    def categories(self, column: str) -> list[str]:
        return self.meta["categories"][column]

    def defaults_for(self, model: str) -> dict[str, Any]:
        """Typical specs for a model, used to pre-fill the form."""
        fallback = dict(self.meta["numeric_medians"])
        fallback.update(self.meta["mode_categories"])
        return {**fallback, **self.meta["model_defaults"].get(model, {})}

    # -- validation -------------------------------------------------------- #
    def validate(self, inputs: dict[str, Any]) -> list[str]:
        """Return human-readable error messages (empty list means valid)."""
        errors: list[str] = []
        for col in CATEGORICAL_COLS:
            value = inputs.get(col)
            if value is None or str(value).strip() == "":
                errors.append(f"Please choose a value for {col.replace('_', ' ')}.")
        for col, (low, high, label) in HARD_LIMITS.items():
            value = inputs.get(col)
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"{label} must be a number.")
                continue
            if not np.isfinite(number) or number < low or number > high:
                errors.append(f"{label} must be between {low:,.0f} and {high:,.0f}.")
        return errors

    # -- prediction -------------------------------------------------------- #
    def predict(self, inputs: dict[str, Any]) -> PredictionResult:
        errors = self.validate(inputs)
        if errors:
            raise ValueError(" ".join(errors))

        row = pd.DataFrame([{c: inputs[c] for c in FEATURE_COLUMNS}])
        for col in NUMERIC_COLS:
            row[col] = pd.to_numeric(row[col])
        features = self.builder.transform(row)
        price = max(0.0, float(self.model.predict(features)[0]))

        warnings = self._warnings(inputs, price)
        low = high = None
        interval = self.meta.get("interval")
        if interval:
            half = interval["relative_halfwidth"] * max(price, interval["floor"])
            low, high = max(0.0, price - half), price + half
        return PredictionResult(price=price, low=low, high=high, warnings=warnings)

    def _warnings(self, inputs: dict[str, Any], price: float) -> list[str]:
        warnings: list[str] = []
        brand, model = str(inputs["brand"]), str(inputs["model"])
        if brand not in self.meta["brand_models"]:
            warnings.append(f"'{brand}' was not in the training data, so its brand effect is "
                            "replaced by the dataset-wide average. Treat the estimate with caution.")
        elif model not in self.meta["brand_models"][brand]:
            known = self.meta["model_defaults"].get(model)
            if known is None:
                warnings.append(f"'{model}' was not seen for {brand} in the training data, so its "
                                "model effect is replaced by the dataset-wide average.")
            else:
                warnings.append(f"'{model}' is normally listed under {known['brand']}, not {brand}.")

        for col, (low, high) in self.meta["numeric_ranges"].items():
            value = float(inputs[col])
            if value < low or value > high:
                warnings.append(
                    f"The {LABELS[col]} ({value:,.0f}) is outside the range seen in training "
                    f"({low:,.0f} to {high:,.0f}); the estimate may be unreliable.")

        cap = self.meta.get("price_cap")
        p_low, p_high = self.meta["train_price_range"]
        if cap is not None and price > cap:
            warnings.append(
                f"This estimate is above {format_inr(cap)}, the upper price limit this model was "
                "trained on (the top 1% of prices was excluded). Luxury cars are not supported.")
        elif price > p_high:
            warnings.append("This estimate is above the highest price in the training data.")
        elif price < p_low:
            warnings.append("This estimate is below the lowest price in the training data.")
        return warnings
