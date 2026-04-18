from src.fraud_detection.modeling import build_classical_models, evaluate_predictions


def test_build_classical_models_contains_expected_models():
    models = build_classical_models()
    assert set(models.keys()) == {"Logistic Regression", "Random Forest", "XGBoost"}


def test_evaluate_predictions_returns_expected_metrics():
    metrics = evaluate_predictions(
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 0],
        y_prob=[0.1, 0.9, 0.2, 0.4],
        model_name="Demo",
    )

    assert metrics["Model"] == "Demo"
    assert 0 <= metrics["Accuracy"] <= 1
    assert 0 <= metrics["F1-Score"] <= 1
    assert 0 <= metrics["AUC-ROC"] <= 1
