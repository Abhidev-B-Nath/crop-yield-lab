"""Crop yield prediction following the workflow in ml.ppt.pptx."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Yield"
RANDOM_STATE = 42


def load_and_clean_data(dataset_path: str | Path) -> pd.DataFrame:
    """Load the CSV, normalize column names, and remove missing targets."""
    data = pd.read_csv(dataset_path)
    data.columns = data.columns.str.strip()

    if TARGET not in data.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    return data.dropna(subset=[TARGET]).copy()


def build_pipeline(
    categorical_features: list[str], numerical_features: list[str]
) -> Pipeline:
    """Build the PPT's preprocessing and random-forest regression pipeline."""
    numerical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_model(
    dataset_path: str | Path,
) -> tuple[Pipeline, dict[str, float], pd.DataFrame, int]:
    """Train on 80% of the data and evaluate on the held-out 20%."""
    data = load_and_clean_data(dataset_path)
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

    if not numerical_features and not categorical_features:
        raise ValueError("No usable feature columns were found in the dataset.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline(categorical_features, numerical_features)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
    }
    evaluation = pd.DataFrame(
        {"Actual": y_test.to_numpy(), "Predicted": y_pred},
        index=y_test.index,
    )
    return pipeline, metrics, evaluation, len(X_train)


def parse_prediction_values(values: str) -> dict[str, Any]:
    """Read a JSON object used as one new input sample."""
    try:
        sample = json.loads(values)
    except json.JSONDecodeError as error:
        raise ValueError("--values must be a valid JSON object.") from error
    if not isinstance(sample, dict) or not sample:
        raise ValueError("--values must contain a non-empty JSON object.")
    return sample


def parse_prediction_file(values_path: str | Path) -> dict[str, Any]:
    """Read one new input sample from a JSON file."""
    return parse_prediction_values(Path(values_path).read_text(encoding="utf-8"))


def save_evaluation_plot(evaluation: pd.DataFrame, output_path: str | Path) -> None:
    """Save the PPT's actual-versus-predicted scatter plot."""
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(evaluation["Actual"], evaluation["Predicted"], alpha=0.75)
    minimum = min(evaluation["Actual"].min(), evaluation["Predicted"].min())
    maximum = max(evaluation["Actual"].max(), evaluation["Predicted"].max())
    axis.plot([minimum, maximum], [minimum, maximum], "r--", label="Ideal Fit (y = x)")
    axis.set_title("Actual vs Predicted Crop Yield")
    axis.set_xlabel("Actual Yield (tonnes per hectare)")
    axis.set_ylabel("Predicted Yield (tonnes per hectare)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict crop yield in tonnes per hectare.")
    parser.add_argument("dataset", type=Path, help="Path to crop_yield.csv")
    parser.add_argument(
        "--values",
        help="JSON object for one new sample; omit it to run evaluation only.",
    )
    parser.add_argument(
        "--values-file",
        type=Path,
        help="JSON file containing one new sample; use instead of --values.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Save an actual-versus-predicted evaluation plot to this path.",
    )
    args = parser.parse_args()

    pipeline, metrics, evaluation, training_samples = train_model(args.dataset)
    print(f"Training samples: {training_samples}")
    print(f"Testing samples: {len(evaluation)}")
    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    if args.plot:
        save_evaluation_plot(evaluation, args.plot)
        print(f"Evaluation plot saved to: {args.plot}")

    if args.values and args.values_file:
        parser.error("use only one of --values or --values-file")
    if args.values or args.values_file:
        sample = (
            parse_prediction_values(args.values)
            if args.values
            else parse_prediction_file(args.values_file)
        )
        new_data = pd.DataFrame([sample])
        predicted_yield = float(pipeline.predict(new_data)[0])
        print(f"\nPredicted Yield: {predicted_yield:.2f} tonnes per hectare")


if __name__ == "__main__":
    main()
