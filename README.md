# Crop Yield Prediction

This project implements the crop yield regression workflow shown in `ml.ppt.pptx`.

## Workflow

1. Load `crop_yield.csv` with pandas.
2. Strip leading and trailing whitespace from column names.
3. Remove rows with a missing `Yield` target.
4. Separate features (`X`) from the continuous `Yield` target (`y`).
5. Detect categorical and numerical feature columns from their dtypes.
6. Impute numerical values with the median.
7. Impute categorical values with the most frequent value and one-hot encode them.
8. Split the data 80/20 with `random_state=42`.
9. Train `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`.
10. Evaluate using MAE, MSE, RMSE, and R2.
11. Plot actual versus predicted test-set yields when requested.
12. Predict the yield for a new JSON sample with the same feature names as the CSV.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the dataset next to the script as `crop_yield.csv`. It must contain a numeric `Yield` column. The remaining columns may be numeric or categorical; the script detects their types automatically.

This workspace includes a reproducible sample `crop_yield.csv` with 1,000 synthetic agricultural records. It contains 10 missing `Yield` values and additional missing feature values so the target removal and preprocessing steps can be tested. The generated data is for workflow testing, not real agricultural decision-making.

## Run evaluation

```powershell
python crop_yield_model.py crop_yield.csv
```

Save the PPT's actual-versus-predicted evaluation chart:

```powershell
python crop_yield_model.py crop_yield.csv --plot actual_vs_predicted.png
```

## Predict a new sample

Pass a JSON object containing the feature columns used by the dataset:

```powershell
python crop_yield_model.py crop_yield.csv --values '{"N":95,"P":45,"K":40,"temperature":28.5,"humidity":72,"ph":6.8,"rainfall":210,"crop":"rice","soil":"loamy"}'
```

On Windows PowerShell, you can place the sample in a JSON file and use `--values-file` to avoid shell quoting issues:

```powershell
python crop_yield_model.py crop_yield.csv --values-file unseen_sample.json
```

The preprocessing is fitted only on the training data and is reused for the test set and new sample, so missing-value handling and one-hot encoding stay consistent with the PPT workflow.

## Run the web app

Start the local demonstration app with:

```powershell
.\.venv\Scripts\python.exe .\app.py
```

Open http://127.0.0.1:8000 in a browser. The form loads the feature schema from the trained model, shows the held-out metrics, and sends each manually entered unseen sample through the fitted pipeline.
