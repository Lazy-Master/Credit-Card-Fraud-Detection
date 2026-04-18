"""Train fraud detection models and save artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import load_dataset, prepare_dataframe, split_and_resample
from .modeling import build_classical_models, build_dnn_model, evaluate_predictions, results_to_frame


def train_pipeline(data_path: str | None = None, output_dir: str = "artifacts") -> pd.DataFrame:
    """Run the full training pipeline and save models plus summary artifacts."""
    df = load_dataset(data_path=data_path)
    df_processed, scaler = prepare_dataframe(df)
    prepared = split_and_resample(df_processed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trained_models = {}
    results = []

    for name, model in build_classical_models().items():
        model.fit(prepared.X_train_resampled, prepared.y_train_resampled)
        trained_models[name] = model
        y_pred = model.predict(prepared.X_test)
        y_prob = model.predict_proba(prepared.X_test)[:, 1]
        results.append(evaluate_predictions(prepared.y_test, y_pred, y_prob, name))

    dnn_model = build_dnn_model(prepared.X_train_resampled.shape[1])
    dnn_model.fit(
        prepared.X_train_resampled,
        prepared.y_train_resampled,
        epochs=20,
        batch_size=256,
        validation_split=0.2,
        verbose=1,
    )
    dnn_prob = dnn_model.predict(prepared.X_test).flatten()
    dnn_pred = (dnn_prob > 0.5).astype(int)
    results.append(evaluate_predictions(prepared.y_test, dnn_pred, dnn_prob, "Deep Neural Network"))

    results_df = results_to_frame(results)
    results_df.to_csv(output_path / "model_performance_summary.csv", index=False)

    random_forest = trained_models["Random Forest"]
    feature_importance = pd.DataFrame(
        {
            "feature": prepared.X_train.columns,
            "importance": random_forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(output_path / "random_forest_feature_importance.csv", index=False)

    joblib.dump(trained_models["Random Forest"], output_path / "random_forest_model.pkl")
    joblib.dump(trained_models["XGBoost"], output_path / "xgboost_model.pkl")
    joblib.dump(trained_models["Logistic Regression"], output_path / "logistic_regression_model.pkl")
    joblib.dump(scaler, output_path / "robust_scaler.pkl")
    dnn_model.save(output_path / "fraud_detection_dnn.keras")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--data-path", default=None, help="Optional path to a local creditcard.csv file.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where trained models and reports will be saved.")
    args = parser.parse_args()

    results_df = train_pipeline(data_path=args.data_path, output_dir=args.output_dir)
    print("Training complete. Model summary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
