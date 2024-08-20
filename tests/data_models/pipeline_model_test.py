from pathlib import Path

import pytest
import yaml
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from maxML.data_models.pipeline_model import PipelineModel


@pytest.mark.parametrize(
    "model_config_path, model_estimator",
    [
        pytest.param(
            "tests/test_configs/logistic_model.yaml",
            LogisticRegression,
            id="logistic_model",
        ),
        pytest.param(
            "tests/test_configs/linear_model.yaml", LinearRegression, id="linear_model"
        ),
    ],
)
def test_retrieve_model_type(model_config_path: str, model_estimator: BaseEstimator):
    pipeline_model = PipelineModel(
        **yaml.safe_load(open(Path.cwd() / model_config_path))
    )
    assert isinstance(pipeline_model.sklearn_model, BaseEstimator)
    assert isinstance(pipeline_model.sklearn_model, model_estimator)
