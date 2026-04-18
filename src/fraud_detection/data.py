"""Data loading and preprocessing helpers for credit card fraud detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/nsethi31/"
    "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
)


@dataclass
class PreparedData:
    """Container for train-test splits and preprocessing artifacts."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_resampled: pd.DataFrame
    y_train_resampled: pd.Series
    scaler: RobustScaler


def load_dataset(data_path: Optional[str] = None, url: str = DEFAULT_DATA_URL) -> pd.DataFrame:
    """Load the fraud detection dataset from a local CSV path or the default URL."""
    if data_path:
        return pd.read_csv(Path(data_path))
    return pd.read_csv(url)


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, RobustScaler]:
    """Scale the `Amount` and `Time` columns and return the transformed dataframe."""
    required_columns = {"Time", "Amount", "Class"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    scaler = RobustScaler()
    prepared = df.copy()
    prepared["Scaled_Amount"] = scaler.fit_transform(prepared[["Amount"]])
    prepared["Scaled_Time"] = scaler.fit_transform(prepared[["Time"]])
    prepared = prepared.drop(["Time", "Amount"], axis=1)
    return prepared, scaler


def split_and_resample(
    df_processed: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    """Split the dataset and apply SMOTE only on the training partition."""
    X = df_processed.drop("Class", axis=1)
    y = df_processed["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_resampled=X_train_resampled,
        y_train_resampled=y_train_resampled,
        scaler=RobustScaler(),
    )
