"""Load saved fraud detection artifacts and make predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from tensorflow import keras

from .data import prepare_dataframe


def load_artifacts(model_dir: str = "artifacts") -> dict:
    """Load saved models from disk."""
    model_path = Path(model_dir)
    return {
        "random_forest": joblib.load(model_path / "random_forest_model.pkl"),
        "logistic_regression": joblib.load(model_path / "logistic_regression_model.pkl"),
        "xgboost": joblib.load(model_path / "xgboost_model.pkl"),
        "dnn": keras.models.load_model(model_path / "fraud_detection_dnn.keras"),
    }


def prepare_inference_frame(csv_path: str) -> pd.DataFrame:
    """Prepare a CSV of transaction rows for inference."""
    frame = pd.read_csv(csv_path)
    if "Class" in frame.columns:
        frame = frame.drop(columns=["Class"])
    if {"Time", "Amount"}.issubset(frame.columns):
        prepared, _ = prepare_dataframe(frame.assign(Class=0))
        frame = prepared.drop(columns=["Class"])
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fraud predictions with saved models.")
    parser.add_argument("csv_path", help="CSV file containing one or more transactions.")
    parser.add_argument("--model-dir", default="artifacts", help="Directory containing saved model files.")
    args = parser.parse_args()

    artifacts = load_artifacts(args.model_dir)
    frame = prepare_inference_frame(args.csv_path)

    rf_pred = artifacts["random_forest"].predict(frame)
    xgb_pred = artifacts["xgboost"].predict(frame)
    lr_pred = artifacts["logistic_regression"].predict(frame)
    dnn_pred = (artifacts["dnn"].predict(frame, verbose=0).flatten() > 0.5).astype(int)

    output = frame.copy()
    output["rf_prediction"] = rf_pred
    output["xgb_prediction"] = xgb_pred
    output["lr_prediction"] = lr_pred
    output["dnn_prediction"] = dnn_pred

    print(output.head().to_string(index=False))


if __name__ == "__main__":
    main()
