import typing as T

import pytest


@pytest.fixture
def logistic_model_config() -> dict[str, T.Any]:
    metrics = ["accuracy_score", "classification_report", "roc_auc_score"]
    return {
        "sklearn_model": "sklearn.linear_model.LogisticRegression",
        "input_path": "data/gemini_sample_data.csv",
        "target": "Purchased",
        "metrics": metrics,
    }
