# CarWise — Used Car Price Prediction

*Know the Worth Before You Buy.*

## 1. Project Overview

CarWise estimates the fair selling price of a second-hand car in India from its specifications (brand, model, age, kilometres driven, fuel, transmission, engine, power and so on). It is trained on the CarDekho used-car dataset and presented through a Streamlit web app. It runs fully offline on your PC, with no paid services or API keys.

The project is based on a reference notebook, *Second Hand Car Price Prediction*, with technical corrections. The main one is that all preprocessing and target encoding are fitted on training data only, which avoids data leakage.

## 2. Features

- Data validation, cleaning and an exploratory data analysis dashboard.
- Seven models compared with R², MAE and RMSE, from Simple Linear Regression to XGBoost with target encoding.
- Leakage-free pipeline: preprocessing is fitted on training data only, and the test set is used once.
- Saved model artifact (Joblib), so the app never retrains on page refresh.
- Price predictor with input validation, unknown-brand/model handling and out-of-range warnings.
- Five-page Streamlit app: Home, Predict Price, Data Analysis, Model Performance, About.

## 3. Technology Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit, Joblib.

## 4. Folder Structure

```
CarWise/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml   <- light theme (navy text, emerald accent)
├── data/            <- place cardekho_dataset.csv here
├── artifacts/       <- trained model and report are saved here
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── model_training.py
    ├── prediction.py
    └── visualizations.py
```

## 5. Install Dependencies (Windows 10/11)

Install **Python 3.10, 3.11 or 3.12** from python.org and tick "Add Python to PATH". Then open a terminal in VS Code (`Terminal > New Terminal`) inside the `CarWise` folder.

**Command Prompt (cmd):**
```bat
py -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**PowerShell:**
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

When the environment is active, your prompt starts with `(venv)`.

## 6. Place the Dataset

The dataset is **not** included. Download the CarDekho used-car dataset (the ~15,411-row version with `selling_price`, `brand`, `model`, `vehicle_age`, `km_driven`, `seller_type`, `fuel_type`, `transmission_type`, `mileage`, `engine`, `max_power`, `seats`) and save it exactly as:

```
CarWise\data\cardekho_dataset.csv
```

## 7. Train the Models

```bat
python train_model.py
```

This loads and checks the CSV, prints a cleaning summary, trains and compares the seven models, selects the best one on the validation set and saves two files into `artifacts/`:

- `carwise_model.joblib` — the chosen model plus the exact preprocessing fitted during training
- `training_report.joblib` — metrics, charts data and cleaning summary shown in the app

Training takes a few minutes. On a slow PC you can use smaller models with `python train_model.py --quick`. Other options: `--data path\to\file.csv` and `--seed 42`. Run training again whenever you change the dataset or the code, then restart Streamlit.

The scores are whatever your data produces. Nothing is hardcoded, and you should not expect the reference notebook's numbers: CarWise fixes the notebook's data leakage, so honest scores are usually lower.

## 8. Launch the App

```bat
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. To stop the app, press `Ctrl + C` in the terminal.

## 9. Using the Price Predictor

1. Open **Predict Price** from the sidebar.
2. Choose brand, then model (the model list updates for the selected brand).
3. Enter age, kilometres driven, fuel, transmission, seller type, mileage, engine, power and seats.
4. Click **Predict Price**.
5. Read the estimate and any warning about inputs outside the training range.

The result is an estimate, not a guaranteed market or resale price.

## 10. Interpreting the Metrics

| Metric | Meaning | Better is |
|---|---|---|
| R² | Share of price variation explained. 1.0 is perfect, 0 is no better than guessing the average. | Higher |
| MAE | Average error in rupees per prediction. | Lower |
| RMSE | Like MAE but punishes big errors more, so expensive cars influence it strongly. | Lower |

All values shown in the app are measured on a held-out test set when you run training. Nothing is hardcoded.

Two sets of test scores are shown. **Test** uses every held-out car. **Test ≤ cap** uses only cars priced at or below the 99th-percentile price of the training data, so the top-1% trimming experiment (model 7) can be compared fairly. The model is chosen on a separate validation set, never on the test set.

The price range on the Predict page is a split-conformal range (a standard method that uses the model's errors on validation data). Its measured coverage on the test set is displayed in the app.

## 11. Common Errors and Solutions

| Problem | Solution |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick "Add Python to PATH", or use `py` instead of `python`. |
| PowerShell: "running scripts is disabled" | Run the `Set-ExecutionPolicy` command in section 5. |
| `FileNotFoundError: data\cardekho_dataset.csv` | Put the CSV in the `data` folder with that exact name. |
| "Missing required columns" at training | The CSV schema differs. The error lists the missing columns. |
| `No module named 'xxx'` | Activate the venv, then run `pip install -r requirements.txt`. |
| App says the model is not trained | Run `python train_model.py` first, then restart Streamlit. |
| XGBoost fails to install or import | Training falls back to a scikit-learn gradient-boosting model automatically. |
| `--quick` mode scores look different | Quick mode trains smaller models. Run without `--quick` for the final results. |
| App shows an old model after retraining | Refresh the browser page; the app reloads when the model file changes. |
| "Model could not be loaded" | Retrain with the same virtual environment you use for `streamlit run`. |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502`. |
