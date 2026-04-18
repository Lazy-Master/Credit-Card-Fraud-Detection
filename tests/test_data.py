import pandas as pd

from src.fraud_detection.data import prepare_dataframe, split_and_resample


def test_prepare_dataframe_adds_scaled_columns():
    frame = pd.DataFrame(
        {
            "Time": [1.0, 2.0, 3.0, 4.0],
            "Amount": [10.0, 15.0, 5.0, 7.5],
            "V1": [0.1, 0.2, -0.3, 0.4],
            "Class": [0, 0, 1, 0],
        }
    )

    prepared, _ = prepare_dataframe(frame)

    assert "Scaled_Amount" in prepared.columns
    assert "Scaled_Time" in prepared.columns
    assert "Amount" not in prepared.columns
    assert "Time" not in prepared.columns


def test_split_and_resample_returns_expected_shapes():
    rows = []
    for idx in range(40):
        rows.append(
            {
                "V1": float(idx),
                "V2": float(idx % 3),
                "Scaled_Amount": float(idx) / 10,
                "Scaled_Time": float(idx) / 5,
                "Class": 1 if idx in (5, 11, 17, 23, 29) else 0,
            }
        )

    frame = pd.DataFrame(rows)
    prepared = split_and_resample(frame, test_size=0.25, random_state=42)

    assert len(prepared.X_train) > 0
    assert len(prepared.X_test) > 0
    assert len(prepared.X_train_resampled) >= len(prepared.X_train)
