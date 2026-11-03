"""CarWise — Used Car Price Prediction (Streamlit web app).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import visualizations as viz
from src.data_preprocessing import (
    DATA_PATH,
    cleaning_report_frame,
    clean_dataset,
    inspect_data,
    load_dataset,
)
from src.model_training import MODEL_PATH, REPORT_PATH
from src.prediction import CarPricePredictor, format_inr

PAGES = ["Home", "Predict Price", "Data Analysis", "Model Performance", "About"]

st.set_page_config(page_title="CarWise — Used Car Price Prediction", page_icon="🚗",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", system-ui, sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1150px; }
.cw-brand { font-size: 1.6rem; font-weight: 800; margin-bottom: 0; }
.cw-brand span { color: #10B981; }
.cw-tag { opacity: 0.65; font-size: 0.85rem; margin-bottom: 1.2rem; }
.cw-hero { background: rgba(128,128,128,0.07); border: 1px solid rgba(128,128,128,0.25);
  border-radius: 18px; padding: 2.2rem 2.4rem; margin-bottom: 1.4rem; }
.cw-hero h1 { font-size: 2.6rem; margin: 0 0 0.3rem 0; }
.cw-hero h1 span { color: #10B981; }
.cw-hero .tag { font-size: 1.15rem; color: #10B981; font-weight: 600; margin-bottom: 0.8rem; }
.cw-hero p { opacity: 0.85; font-size: 1.02rem; max-width: 760px; margin: 0; }
.cw-card { background: rgba(128,128,128,0.07); border: 1px solid rgba(128,128,128,0.25);
  border-radius: 14px; padding: 1.1rem 1.2rem; height: 100%; }
.cw-label { opacity: 0.65; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; }
.cw-value { font-size: 1.55rem; font-weight: 700; margin: 0.25rem 0; }
.cw-note { opacity: 0.65; font-size: 0.8rem; }
.cw-step { background: rgba(128,128,128,0.07); border: 1px solid rgba(128,128,128,0.25);
  border-left: 4px solid #10B981; border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.7rem; }
.cw-result { background: #0B1F3A; color: #FFFFFF; border-radius: 16px; padding: 1.6rem 1.8rem;
  box-shadow: 0 4px 14px rgba(11,31,58,0.18); margin: 0.6rem 0 1rem 0; }
.cw-result .lab { color: #A7F3D0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .08em; }
.cw-result .price { font-size: 2.8rem; font-weight: 800; margin: 0.2rem 0; }
.cw-result .range { color: #CBD5E1; font-size: 0.95rem; }
.cw-section { font-weight: 700; font-size: 1.05rem; margin: 1.2rem 0 0.4rem 0;
  padding-bottom: 0.3rem; border-bottom: 2px solid #10B981; display: inline-block; }
.stButton > button[kind="primary"] { background: #10B981; border: none; color: #FFFFFF;
  font-weight: 600; border-radius: 10px; padding: 0.55rem 1.6rem; }
.stButton > button[kind="primary"]:hover { background: #059669; color: #FFFFFF; }
footer { visibility: hidden; }
</style>
"""


# --------------------------------------------------------------------------- #
# Cached loaders (the model is loaded once, never retrained)
# --------------------------------------------------------------------------- #
def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner=False)
def load_predictor(stamp: float) -> tuple[CarPricePredictor | None, str | None]:
    try:
        return CarPricePredictor.load(), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as error:  # corrupt file or library version mismatch
        return None, f"{type(error).__name__}: {error}"


@st.cache_data(show_spinner=False)
def load_report(stamp: float) -> dict[str, Any] | None:
    if not REPORT_PATH.exists():
        return None
    try:
        return joblib.load(REPORT_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner="Loading dataset...")
def load_data(stamp: float):
    if not DATA_PATH.exists():
        return None
    raw = load_dataset()
    cleaned, report = clean_dataset(raw)
    return raw, cleaned, report, inspect_data(raw)


def show_figure(fig: plt.Figure) -> None:
    st.pyplot(fig)
    plt.close(fig)


def card(label: str, value: str, note: str = "") -> str:
    return (f'<div class="cw-card"><div class="cw-label">{html.escape(label)}</div>'
            f'<div class="cw-value">{html.escape(value)}</div>'
            f'<div class="cw-note">{html.escape(note)}</div></div>')


def go_to(page: str) -> None:
    st.session_state["nav"] = page


def similar_listings(df: pd.DataFrame, brand: str, model: str, age: int, fuel: str,
                     transmission: str) -> tuple[pd.DataFrame | None, str]:
    """Find real listings that resemble the car, widening the match until enough are found."""
    base = df[(df["brand"] == brand) & (df["model"] == model)]
    if base.empty:
        return None, ""
    near_age = (base["vehicle_age"] - age).abs() <= 1
    tiers = [
        ("same model, age within 1 year, same fuel and transmission",
         near_age & (base["fuel_type"] == fuel) & (base["transmission_type"] == transmission)),
        ("same model, age within 1 year", near_age),
        ("same model, any age", pd.Series(True, index=base.index)),
    ]
    for label, mask in tiers:
        subset = base[mask]
        if len(subset) >= 5:
            return subset, label
    return base, f"same model, any age (only {len(base)} listings in the dataset)"


def render_actual_prices(df: pd.DataFrame, inputs: dict[str, Any]) -> float | None:
    """Show real listed prices for similar cars. Returns their median price."""
    subset, label = similar_listings(df, inputs["brand"], inputs["model"], inputs["vehicle_age"],
                                     inputs["fuel_type"], inputs["transmission_type"])
    if subset is None:
        st.info("This model has no listings in the dataset, so no actual prices can be shown.")
        return None
    prices = subset["selling_price"]
    median = float(prices.median())
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Median actual price", format_inr(round(median, -3)),
                     f"{len(subset):,} similar listings"), unsafe_allow_html=True)
    c2.markdown(card("Typical range (10th to 90th percentile)",
                     f"{format_inr(round(prices.quantile(0.10), -3))} to "
                     f"{format_inr(round(prices.quantile(0.90), -3))}",
                     f"lowest {format_inr(prices.min())}, highest {format_inr(prices.max())}"),
                unsafe_allow_html=True)
    c3.markdown(card("Matched on", "Similar cars", label), unsafe_allow_html=True)

    closest = subset.assign(_age=(subset["vehicle_age"] - inputs["vehicle_age"]).abs(),
                            _km=(subset["km_driven"] - inputs["km_driven"]).abs())
    closest = closest.sort_values(["_age", "_km"]).head(5)
    table = pd.DataFrame({
        "Age (years)": closest["vehicle_age"].astype(int),
        "Km driven": closest["km_driven"].map("{:,.0f}".format),
        "Fuel": closest["fuel_type"], "Transmission": closest["transmission_type"],
        "Actual listed price": closest["selling_price"].map(format_inr),
    })
    st.markdown("**Closest real listings**")
    st.dataframe(table, hide_index=True)
    st.caption("These are real used-car selling prices from the dataset, not the original showroom "
               "price of a new car, which the dataset does not contain.")
    return median


def training_needed() -> None:
    st.warning("The model has not been trained yet. Place `cardekho_dataset.csv` in the `data` "
               "folder, then run `python train_model.py` in the terminal and refresh this page.")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_home(report: dict[str, Any] | None) -> None:
    st.markdown(
        '<div class="cw-hero"><h1>Car<span>Wise</span></h1>'
        '<div class="tag">Know the Worth Before You Buy.</div>'
        "<p>CarWise estimates the fair selling price of a second-hand car in India from its "
        "specifications. It learns from thousands of real CarDekho listings using machine "
        "learning, and runs entirely on your own computer.</p></div>",
        unsafe_allow_html=True)

    cols = st.columns(4)
    if report:
        metrics = report["best_metrics"]["test"]
        name = report["best_name"].split(". ", 1)[-1]
        cleaned_rows = report["cleaning_report"]["rows_final"]
        initial_rows = report["cleaning_report"]["rows_initial"]
        values = [
            ("Dataset size", f"{cleaned_rows:,}", f"listings after cleaning ({initial_rows:,} raw)"),
            ("Selected model", name, "chosen on validation RMSE"),
            ("R² (test set)", f"{metrics['r2']:.3f}", "share of price variation explained"),
            ("MAE (test set)", format_inr(metrics["mae"]), "average error per prediction"),
        ]
    else:
        values = [("Dataset size", "—", "train the model to see this"),
                  ("Selected model", "—", "not trained yet"),
                  ("R² (test set)", "—", "not trained yet"),
                  ("MAE (test set)", "—", "not trained yet")]
    for col, (label, value, note) in zip(cols, values):
        col.markdown(card(label, value, note), unsafe_allow_html=True)

    st.write("")
    if report:
        st.button("Predict a car price", type="primary", on_click=go_to, args=("Predict Price",))
    else:
        training_needed()

    st.markdown('<div class="cw-section">How it works</div>', unsafe_allow_html=True)
    steps = [
        ("1. Learn from data", "The system studies past used-car listings: brand, model, age, "
         "kilometres, fuel, transmission, power and the price each car sold for."),
        ("2. Prepare safely", "Cleaning rules and encodings are learned from training data only, "
         "so the test results are not artificially inflated."),
        ("3. Compare models", "Seven models are trained, from a simple straight line to gradient "
         "boosting, and scored on cars the model has never seen."),
        ("4. Predict", "Enter a car's details. The saved model applies the same preparation and "
         "returns an estimated price with an honest warning if the car is unusual."),
    ]
    for title, text in steps:
        st.markdown(f'<div class="cw-step"><b>{title}</b><br>{text}</div>', unsafe_allow_html=True)


def page_predict(predictor: CarPricePredictor | None, error: str | None) -> None:
    st.markdown("## Predict Car Price")
    if predictor is None:
        if error == "missing":
            training_needed()
        else:
            st.error(f"The saved model could not be loaded ({error}). Run `python train_model.py` "
                     "again with the same Python environment you use for the app.")
        return

    st.caption("Fill in the car's details. Typical specifications are pre-filled when you pick a model.")

    st.markdown('<div class="cw-section">Car</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    brand = c1.selectbox("Brand", predictor.brands, key="brand")
    model = c2.selectbox("Model", predictor.models_for(brand), key=f"model_{brand}")
    d = predictor.defaults_for(model)
    ranges = predictor.meta["numeric_ranges"]
    tag = f"{brand}_{model}"

    st.markdown('<div class="cw-section">Usage</div>', unsafe_allow_html=True)
    u1, u2 = st.columns(2)
    max_age = max(int(ranges["vehicle_age"][1]), 10)
    age = u1.slider("Vehicle age (years)", 0, max_age,
                    int(min(max(round(predictor.meta["numeric_medians"]["vehicle_age"]), 0), max_age)),
                    key=f"age_{tag}")
    u1.caption("Vehicle age = current year minus manufacturing year.")
    km = u2.number_input("Kilometres driven", min_value=0, max_value=1_000_000,
                         value=int(round(predictor.meta["numeric_medians"]["km_driven"], -3)),
                         step=1000, key=f"km_{tag}")

    st.markdown('<div class="cw-section">Specifications</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

    def index_of(options: list[str], wanted: str) -> int:
        return options.index(wanted) if wanted in options else 0

    fuels, trans, sellers = (predictor.categories(c) for c in
                             ("fuel_type", "transmission_type", "seller_type"))
    fuel = s1.selectbox("Fuel type", fuels, index=index_of(fuels, d["fuel_type"]), key=f"fuel_{tag}")
    transmission = s2.selectbox("Transmission", trans,
                                index=index_of(trans, d["transmission_type"]), key=f"trans_{tag}")
    seller = s3.selectbox("Seller type", sellers, index=index_of(sellers, d["seller_type"]),
                          key=f"seller_{tag}")

    m1, m2, m3, m4 = st.columns(4)
    mileage = m1.number_input("Mileage (km/l)", min_value=1.0, max_value=60.0,
                              value=float(min(max(round(d["mileage"], 1), 1.0), 60.0)),
                              step=0.1, key=f"mileage_{tag}")
    engine = m2.number_input("Engine (cc)", min_value=200, max_value=8000,
                             value=int(min(max(round(d["engine"]), 200), 8000)), step=50,
                             key=f"engine_{tag}")
    power = m3.number_input("Max power (bhp)", min_value=20.0, max_value=1000.0,
                            value=float(min(max(round(d["max_power"], 1), 20.0), 1000.0)),
                            step=1.0, key=f"power_{tag}")
    seat_options = list(range(2, 11))
    seats_default = int(round(d["seats"]))
    seats = m4.selectbox("Seats", seat_options,
                         index=seat_options.index(seats_default) if seats_default in seat_options else 3,
                         key=f"seats_{tag}")

    st.write("")
    if not st.button("Predict Price", type="primary"):
        return

    inputs = {"brand": brand, "model": model, "vehicle_age": age, "km_driven": km,
              "seller_type": seller, "fuel_type": fuel, "transmission_type": transmission,
              "mileage": mileage, "engine": engine, "max_power": power, "seats": seats}
    errors = predictor.validate(inputs)
    if errors:
        for message in errors:
            st.error(message)
        return

    st.markdown('<div class="cw-section">Actual listed prices in the dataset</div>',
                unsafe_allow_html=True)
    loaded = load_data(_mtime(DATA_PATH))
    median_actual = None
    if loaded is None:
        st.info("The dataset file was not found, so actual listing prices cannot be shown. Place "
                "`cardekho_dataset.csv` in the `data` folder to enable this section.")
    else:
        median_actual = render_actual_prices(loaded[1], inputs)

    result = predictor.predict(inputs)
    range_html = ""
    if result.low is not None and result.high is not None:
        level = predictor.meta["interval"]["level"]
        range_html = (f'<div class="range">Likely range ({level:.0%} target): '
                      f"{format_inr(round(result.low, -3))} to {format_inr(round(result.high, -3))}</div>")
    st.markdown(
        f'<div class="cw-result"><div class="lab">Estimated fair price</div>'
        f'<div class="price">{format_inr(round(result.price, -3))}</div>{range_html}</div>',
        unsafe_allow_html=True)
    if median_actual:
        diff = (result.price - median_actual) / median_actual
        direction = "above" if diff >= 0 else "below"
        st.write(f"The predicted price is **{abs(diff):.0%} {direction}** the median actual price "
                 f"of similar listings ({format_inr(round(median_actual, -3))}).")
    for message in result.warnings:
        st.warning(message)
    cov = predictor.meta["interval"]["test_coverage"] if predictor.meta.get("interval") else None
    extra = (f" The range was calibrated on validation data and covered {cov:.0%} of unseen test "
             "cars." if cov is not None else "")
    st.caption("This is a statistical estimate, not a guaranteed market or resale price. Real "
               "prices also depend on condition, location, service history and negotiation." + extra)


def page_data(report: dict[str, Any] | None) -> None:
    st.markdown("## Data Analysis")
    loaded = load_data(_mtime(DATA_PATH))
    if loaded is None:
        st.warning("The dataset file `data/cardekho_dataset.csv` was not found, so charts cannot be "
                   "drawn. Place the file there and refresh.")
        if report:
            st.markdown('<div class="cw-section">Cleaning summary (from last training run)</div>',
                        unsafe_allow_html=True)
            st.dataframe(cleaning_report_frame(report["cleaning_report"]), hide_index=True)
        return

    raw, df, cleaning, inspection = loaded
    st.markdown('<div class="cw-section">Dataset overview</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (label, value, note) in zip(cols, [
        ("Raw rows", f"{inspection['shape'][0]:,}", f"{inspection['shape'][1]} columns"),
        ("Duplicates", f"{inspection['duplicates']:,}", "exact duplicate rows"),
        ("Missing cells", f"{int(inspection['missing'].sum()):,}", "in required columns"),
        ("Rows after cleaning", f"{cleaning['rows_final']:,}",
         f"{cleaning['rows_removed']:,} removed")]):
        col.markdown(card(label, value, note), unsafe_allow_html=True)

    with st.expander("Column names and data types"):
        st.dataframe(pd.DataFrame({"Column": list(inspection["dtypes"]),
                                   "Type": list(inspection["dtypes"].values())}), hide_index=True)
    with st.expander("Descriptive statistics (before cleaning)"):
        st.dataframe(inspection["describe"].round(2))
    with st.expander("Suspicious values found before cleaning"):
        st.dataframe(pd.DataFrame({"Check": list(inspection["invalid"]),
                                   "Rows": list(inspection["invalid"].values())}), hide_index=True)

    st.markdown('<div class="cw-section">Cleaning summary</div>', unsafe_allow_html=True)
    st.caption("Counts come from your actual CSV. Corrections are applied only when the data supports them.")
    st.dataframe(cleaning_report_frame(cleaning), hide_index=True)

    notes = viz.eda_insights(df)
    charts = [
        ("Selling price distribution", viz.plot_price_distribution, notes["price"]),
        ("Vehicle age vs selling price", viz.plot_age_vs_price, notes["age"]),
        ("Kilometres driven vs selling price", viz.plot_km_vs_price, notes["km"]),
        ("Engine power vs selling price", viz.plot_power_vs_price, notes["power"]),
        ("Selling price by fuel type", viz.plot_price_by_fuel, notes["fuel"]),
        ("Manual vs automatic transmission", viz.plot_price_by_transmission, notes["transmission"]),
        ("Popular car brands", viz.plot_top_brands, notes["brands"]),
        ("Correlation heatmap", viz.plot_correlation_heatmap, notes["correlation"]),
        ("Outlier detection", viz.plot_outliers, notes["outliers"]),
    ]
    st.markdown('<div class="cw-section">Exploratory analysis (cleaned data)</div>',
                unsafe_allow_html=True)
    for title, builder, note in charts:
        st.markdown(f"**{title}**")
        show_figure(builder(df))
        st.info(note)


def page_performance(report: dict[str, Any] | None) -> None:
    st.markdown("## Model Performance")
    if report is None:
        training_needed()
        return

    comparison: pd.DataFrame = report["comparison"]
    best = report["best_name"]
    test = report["best_metrics"]["test"]
    st.success(f"Selected model: **{best}**, chosen by the lowest validation RMSE. "
               f"Test results: R² {test['r2']:.3f}, MAE {format_inr(test['mae'])}, "
               f"RMSE {format_inr(test['rmse'])}.")

    split = report["splits"]
    st.caption(f"Split: {split['train']:,} train / {split['validation']:,} validation / "
               f"{split['test']:,} test rows (seed {report['seed']}). Metrics are measured on the "
               "test set, which was never used for training, tuning or model selection."
               + (" Quick mode was used, so models are smaller than normal." if report.get("quick") else ""))

    display = pd.DataFrame({
        "Model": comparison["Model"],
        "R² (test)": comparison["R² (test)"].map("{:.4f}".format),
        "MAE (test)": comparison["MAE (test)"].map(format_inr),
        "RMSE (test)": comparison["RMSE (test)"].map(format_inr),
        "R² (test ≤ cap)": comparison["R² (test ≤ cap)"].map("{:.4f}".format),
        "MAE (test ≤ cap)": comparison["MAE (test ≤ cap)"].map(format_inr),
        "Train rows": comparison["Train rows"].map("{:,}".format),
    })
    st.markdown('<div class="cw-section">Model comparison</div>', unsafe_allow_html=True)
    st.dataframe(display, hide_index=True)
    st.info(
        f"Columns marked “≤ cap” score only test cars priced at or below {format_inr(report['price_cap'])}, "
        "the 99th percentile of TRAINING prices. This lets the top-1% trimming experiment (model 7) be "
        "compared fairly with the others. The plain test columns use every test car, including "
        "luxury ones, so they are the honest numbers for the whole market. The notebook's "
        "trimmed-model score was measured on a trimmed test set only.")

    st.markdown('<div class="cw-section">R², MAE and RMSE</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        show_figure(viz.plot_metric_bars(comparison, "R² (test)", "R² (higher is better)", False))
    with c2:
        show_figure(viz.plot_metric_bars(comparison, "MAE (test)", "MAE (lower is better)", True,
                                         1e5, "₹ lakh"))
    with c3:
        show_figure(viz.plot_metric_bars(comparison, "RMSE (test)", "RMSE (lower is better)", True,
                                         1e5, "₹ lakh"))
    st.caption("The green bar marks the best model on each metric. It can differ between metrics "
               "and from the selected model, which is chosen on validation data, not test data.")

    st.markdown('<div class="cw-section">Actual vs predicted (test set)</div>', unsafe_allow_html=True)
    names = list(report["test_predictions"])
    choice = st.selectbox("Model", names, index=names.index(best))
    show_figure(viz.plot_actual_vs_predicted(report["y_test"], report["test_predictions"][choice],
                                             choice))
    st.caption("Points close to the dashed line are accurate predictions.")

    if report.get("importance") is not None:
        st.markdown('<div class="cw-section">Feature importance (selected model)</div>',
                    unsafe_allow_html=True)
        show_figure(viz.plot_feature_importance(report["importance"]))
        st.caption("Importance of one-hot columns is added back together per original feature.")

    interval = report["metadata"].get("interval")
    if interval:
        st.markdown('<div class="cw-section">Prediction range</div>', unsafe_allow_html=True)
        st.write(f"The price range uses split-conformal calibration on the validation set "
                 f"(about ±{interval['relative_halfwidth']:.0%} of the prediction, "
                 f"{interval['level']:.0%} target). On the untouched test set it covered "
                 f"{interval['test_coverage']:.1%} of cars.")


def page_about(report: dict[str, Any] | None) -> None:
    st.markdown("## About the Project")
    sections = {
        "Problem statement": "Buyers and sellers of used cars in India often cannot tell whether an "
        "asking price is fair. Prices depend on many factors at once, so a simple rule of thumb is "
        "unreliable.",
        "Objectives": "Build a machine-learning model that estimates a used car's selling price from "
        "its specifications, compare several algorithms fairly, and present the result in an easy "
        "web application.",
        "Dataset": "The CarDekho used-car dataset: one row per listing, with brand, model, age, "
        "kilometres driven, seller, fuel, transmission, mileage, engine size, power, seats and the "
        "selling price (the target).",
        "Methodology": "1) Load and inspect the data. 2) Clean it using rules supported by the data. "
        "3) Split into training, validation and test sets. 4) Learn all preprocessing from training "
        "data only. 5) Train seven models. 6) Choose the model on validation data. 7) Report final "
        "scores once on the test set. 8) Save the model and preprocessing together.",
        "Algorithms": "Simple and multiple linear regression (straight-line baselines), decision tree "
        "(if-then rules), random forest (many trees averaged), and XGBoost (trees built one after "
        "another, each fixing the previous errors). Two XGBoost variants add target encoding (brand and "
        "model replaced by typical prices) and removal of the most expensive 1% of training cars.",
        "Avoiding data leakage": "Scalers, imputation values, category lists and target encodings are "
        "learned from training rows only. Training rows receive out-of-fold encodings so a car's own "
        "price never leaks into its features. The 99th-percentile price cap uses training prices only.",
        "Evaluation metrics": "R² is the share of price variation explained (higher is better). MAE is "
        "the average rupee error (lower is better). RMSE is similar but punishes large errors more "
        "(lower is better).",
        "Limitations": "The model does not know a car's condition, accident history, colour, city or "
        "service record. It performs poorly on very rare or luxury cars. It reflects the time period "
        "of the dataset, so prices may be outdated. The price range is approximate.",
        "Future improvements": "Add location and owner-count features, train separate models for "
        "budget and premium cars, use cross-validation and hyperparameter search, refresh the "
        "dataset regularly, and model price on a log scale.",
    }
    for title, text in sections.items():
        st.markdown(f"**{title}**")
        st.write(text)
    if report:
        st.caption(f"Last trained: {report['trained_at']} | Boosting engine: {report['engine_label']}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if "nav" not in st.session_state:
        st.session_state["nav"] = "Home"

    with st.sidebar:
        st.markdown('<div class="cw-brand">Car<span>Wise</span></div>'
                    '<div class="cw-tag">Know the Worth Before You Buy.</div>',
                    unsafe_allow_html=True)
        st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")

    predictor, error = load_predictor(_mtime(MODEL_PATH))
    report = load_report(_mtime(REPORT_PATH))
    page = st.session_state["nav"]

    if page == "Home":
        page_home(report)
    elif page == "Predict Price":
        page_predict(predictor, error)
    elif page == "Data Analysis":
        page_data(report)
    elif page == "Model Performance":
        page_performance(report)
    else:
        page_about(report)


main()
