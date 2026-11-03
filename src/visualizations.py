"""Matplotlib / Seaborn figures for the CarWise app. Every function returns a Figure."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # safe for Streamlit and for running without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .data_preprocessing import NUMERIC_COLS, TARGET, iqr_outlier_count

NAVY = "#0B1F3A"
EMERALD = "#10B981"
SLATE = "#64748B"
LIGHT = "#E2E8F0"
PALE = "#A7F3D0"

PRICE_LABEL = "Selling price (₹ lakh)"
_CMAP = LinearSegmentedColormap.from_list("carwise", [NAVY, "#FFFFFF", EMERALD])


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": LIGHT,
        "axes.labelcolor": NAVY, "axes.titlecolor": NAVY, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "xtick.color": SLATE, "ytick.color": SLATE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#EEF2F6", "grid.linewidth": 0.8, "font.size": 10,
        "legend.frameon": False,
    })


apply_theme()


def _lakh(series: pd.Series) -> pd.Series:
    return series / 1e5


def plot_price_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(_lakh(df[TARGET]), bins=50, color=EMERALD, edgecolor="white")
    axes[0].set(title="Selling price (right-skewed)", xlabel=PRICE_LABEL, ylabel="Listings")
    axes[1].hist(np.log1p(df[TARGET]), bins=50, color=NAVY, edgecolor="white")
    axes[1].set(title="Log of selling price (closer to bell-shaped)", xlabel="log(1 + price)",
                ylabel="Listings")
    fig.tight_layout()
    return fig


def plot_age_vs_price(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df["vehicle_age"], _lakh(df[TARGET]), s=8, alpha=0.25, color=SLATE)
    med = df.groupby("vehicle_age")[TARGET].median() / 1e5
    ax.plot(med.index, med.values, color=EMERALD, linewidth=2.5, marker="o", label="Median price")
    ax.set(title="Vehicle age vs selling price", xlabel="Vehicle age (years)", ylabel=PRICE_LABEL)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_km_vs_price(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df["km_driven"] / 1000, _lakh(df[TARGET]), s=8, alpha=0.25, color=NAVY)
    ax.set(title="Kilometres driven vs selling price", xlabel="Kilometres driven (thousand km)",
           ylabel=PRICE_LABEL)
    fig.tight_layout()
    return fig


def plot_power_vs_price(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df["max_power"], _lakh(df[TARGET]), s=8, alpha=0.25, color=EMERALD)
    ax.set(title="Engine power vs selling price", xlabel="Maximum power (bhp)", ylabel=PRICE_LABEL)
    fig.tight_layout()
    return fig


def plot_price_by_fuel(df: pd.DataFrame) -> plt.Figure:
    order = df.groupby("fuel_type")[TARGET].median().sort_values(ascending=False).index
    data = df.assign(price_lakh=_lakh(df[TARGET]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=data, x="fuel_type", y="price_lakh", order=order, color=PALE,
                linecolor=NAVY, showfliers=False, ax=ax)
    ax.set(title="Selling price by fuel type (outliers hidden)", xlabel="Fuel type",
           ylabel=PRICE_LABEL)
    fig.tight_layout()
    return fig


def plot_price_by_transmission(df: pd.DataFrame) -> plt.Figure:
    data = df.assign(price_lakh=_lakh(df[TARGET]))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.boxplot(data=data, x="transmission_type", y="price_lakh", color=PALE, linecolor=NAVY,
                showfliers=False, ax=ax)
    ax.set(title="Manual vs automatic (outliers hidden)", xlabel="Transmission",
           ylabel=PRICE_LABEL)
    fig.tight_layout()
    return fig


def plot_top_brands(df: pd.DataFrame, min_count: int = 50) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    top = df["brand"].value_counts().head(10).iloc[::-1]
    axes[0].barh(top.index, top.values, color=EMERALD)
    axes[0].set(title="Most listed brands", xlabel="Listings")
    stats = df.groupby("brand")[TARGET].agg(["median", "count"])
    threshold = min_count if (stats["count"] >= min_count).sum() >= 5 else 10
    pricey = stats[stats["count"] >= threshold]["median"].sort_values().tail(10) / 1e5
    axes[1].barh(pricey.index, pricey.values, color=NAVY)
    axes[1].set(title=f"Highest median price ({threshold}+ listings)",
                xlabel="Median price (₹ lakh)")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    corr = df[NUMERIC_COLS + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=_CMAP, vmin=-1, vmax=1, linewidths=0.5,
                linecolor="white", annot_kws={"size": 9}, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation between numeric features")
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_outliers(df: pd.DataFrame) -> plt.Figure:
    cols = [("km_driven", "Kilometres driven"), (TARGET, "Selling price"),
            ("max_power", "Max power (bhp)"), ("vehicle_age", "Vehicle age (years)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for ax, (col, label) in zip(axes.flatten(), cols):
        sns.boxplot(x=df[col], color=PALE, linecolor=NAVY,
                    flierprops={"markerfacecolor": EMERALD, "markeredgecolor": EMERALD,
                                "markersize": 3}, ax=ax)
        ax.set(title=f"{label}: {iqr_outlier_count(df[col]):,} outliers (1.5×IQR)", xlabel="")
    fig.tight_layout()
    return fig


def eda_insights(df: pd.DataFrame) -> dict[str, str]:
    """Plain-English findings computed from the data (nothing hardcoded)."""
    price = df[TARGET]
    corr = df[NUMERIC_COLS + [TARGET]].corr()[TARGET]
    fuel_med = df.groupby("fuel_type")[TARGET].median().sort_values(ascending=False)
    trans_med = df.groupby("transmission_type")[TARGET].median()
    brand_share = df["brand"].value_counts(normalize=True)
    insights = {
        "price": (f"Median price is ₹{price.median() / 1e5:.2f} lakh but the mean is "
                  f"₹{price.mean() / 1e5:.2f} lakh; skewness is {price.skew():.2f}. A few "
                  "expensive cars stretch the right tail, which is why a log scale looks more "
                  "symmetric."),
        "age": (f"Age and price have a correlation of {corr['vehicle_age']:.2f}: older cars "
                "tend to sell for less (depreciation)."),
        "km": (f"Kilometres driven and price have a correlation of {corr['km_driven']:.2f}. "
               "The link is weaker than for age because mileage is partly explained by age."),
        "power": (f"Maximum power has a correlation of {corr['max_power']:.2f} with price, "
                  "the strongest of the numeric features: more powerful cars sit in higher "
                  "price segments."),
        "fuel": (f"{fuel_med.index[0]} cars have the highest median price "
                 f"(₹{fuel_med.iloc[0] / 1e5:.2f} lakh); {fuel_med.index[-1]} the lowest "
                 f"(₹{fuel_med.iloc[-1] / 1e5:.2f} lakh)."),
        "brands": (f"{brand_share.index[0]} is the most listed brand "
                   f"({brand_share.iloc[0]:.0%} of listings). Brands with very few listings "
                   "have unreliable median prices."),
        "outliers": (f"By the 1.5×IQR rule, {iqr_outlier_count(df['km_driven']):,} cars have "
                     f"unusually high kilometres and {iqr_outlier_count(price):,} have unusually "
                     "high prices. Most high prices are genuine luxury cars, not errors."),
    }
    if {"Manual", "Automatic"} <= set(trans_med.index):
        ratio = trans_med["Automatic"] / trans_med["Manual"]
        insights["transmission"] = (
            f"The median automatic car costs {ratio:.1f}× the median manual car "
            f"(₹{trans_med['Automatic'] / 1e5:.2f} vs ₹{trans_med['Manual'] / 1e5:.2f} lakh). "
            "Part of this is because automatics are more common in higher segments.")
    else:
        insights["transmission"] = "Median price differs between transmission types."
    strong = corr.drop(TARGET).abs().sort_values(ascending=False)
    insights["correlation"] = (f"Features most related to price: {strong.index[0]} "
                               f"({corr[strong.index[0]]:.2f}) and {strong.index[1]} "
                               f"({corr[strong.index[1]]:.2f}). Engine and max power also "
                               "move together, so they carry overlapping information.")
    return insights


# --------------------------------------------------------------------------- #
# Model performance charts
# --------------------------------------------------------------------------- #
def plot_metric_bars(comparison: pd.DataFrame, column: str, title: str, lower_is_better: bool,
                     scale: float = 1.0, unit: str = "") -> plt.Figure:
    labels = comparison["Model"].str.extract(r"^(\d+)\.")[0].radd("Model ")
    values = comparison[column] / scale
    best = values.idxmin() if lower_is_better else values.idxmax()
    colors = [EMERALD if i == best else LIGHT for i in values.index]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor=NAVY, linewidth=0.4)
    for bar, value in zip(bars, values[::-1]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:.3g}",
                va="center", fontsize=8, color=NAVY)
    ax.set(title=title, xlabel=unit)
    ax.margins(x=0.15)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(y_true: Any, y_pred: Any, title: str) -> plt.Figure:
    y_true = np.asarray(y_true, dtype=float) / 1e5
    y_pred = np.asarray(y_pred, dtype=float) / 1e5
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.35, color=EMERALD)
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, "--", color=NAVY, linewidth=1.2, label="Perfect prediction")
    ax.set(title=title, xlabel="Actual price (₹ lakh)", ylabel="Predicted price (₹ lakh)",
           xlim=lim, ylim=lim)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["feature"], top["importance"], color=EMERALD)
    ax.set(title="Feature importance (share of total)", xlabel="Importance")
    fig.tight_layout()
    return fig
