"""Train the CarWise models and save the artifacts.

Usage (from the CarWise folder, with the virtual environment active):
    python train_model.py
    python train_model.py --data data/cardekho_dataset.csv --seed 42
    python train_model.py --quick        # faster, smaller models (for slow PCs / testing)
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.data_preprocessing import DATA_PATH, clean_dataset, inspect_data, load_dataset
from src.model_training import MODEL_PATH, REPORT_PATH, SEED, run_experiments, save_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the CarWise price prediction models.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to cardekho_dataset.csv")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default 42)")
    parser.add_argument("--quick", action="store_true", help="Use smaller models for a faster run")
    args = parser.parse_args()

    print("=" * 72)
    print("CarWise training")
    print("=" * 72)

    try:
        raw = load_dataset(args.data)
    except (FileNotFoundError, ValueError) as error:
        print(f"\nERROR: {error}")
        return 1

    inspection = inspect_data(raw)
    print(f"\nLoaded {inspection['shape'][0]:,} rows x {inspection['shape'][1]} columns")
    print(f"Exact duplicate rows: {inspection['duplicates']:,}")
    print("Suspicious values found before cleaning:")
    for label, count in inspection["invalid"].items():
        print(f"  - {label}: {count:,}")

    cleaned, report = clean_dataset(raw)
    print("\nCleaning summary:")
    for step in report["steps"]:
        print(f"  - {step['step']}: {step['rows_affected']:,} row(s) -> {step['action']}")
    print(f"Rows: {report['rows_initial']:,} -> {report['rows_final']:,} "
          f"({report['rows_removed']:,} removed)\n")

    print("Training and comparing seven models (this can take a few minutes)...")
    results = run_experiments(cleaned, seed=args.seed, quick=args.quick)
    save_artifacts(results, report, inspection)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    table = results["comparison"][["Model", "R² (test)", "MAE (test)", "RMSE (test)",
                                   "R² (test ≤ cap)"]].copy()
    table["MAE (test)"] = table["MAE (test)"].round(0)
    table["RMSE (test)"] = table["RMSE (test)"].round(0)
    table["R² (test)"] = table["R² (test)"].round(4)
    table["R² (test ≤ cap)"] = table["R² (test ≤ cap)"].round(4)
    print("\nMeasured results (held-out test set):")
    print(table.to_string(index=False))
    print(f"\nSelected model : {results['best_name']}")
    print(f"Saved model    : {MODEL_PATH}")
    print(f"Saved report   : {REPORT_PATH}")
    print("\nNext step: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
