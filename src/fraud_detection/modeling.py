"""Model definitions and evaluation helpers for fraud detection."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras


def build_classical_models(random_state: int = 42) -> Dict[str, object]:
    """Return the classical ML models used in the notebook."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
    }


def build_dnn_model(input_dim: int) -> keras.Model:
    """Create the deep neural network used in the notebook."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate_predictions(y_true, y_pred, y_prob, model_name: str) -> dict:
    """Compute the evaluation metrics used in the notebook."""
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_prob),
    }


def results_to_frame(results: list[dict]) -> pd.DataFrame:
    """Convert model results into a sorted dataframe."""
    frame = pd.DataFrame(results)
    return frame.sort_values("F1-Score", ascending=False).reset_index(drop=True)
