import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from maxML.config_schemas import load_config
from maxML.config_schemas import PipelineConfig


# TODO: Add test case defined by no preprocessors.
@pytest.mark.parametrize(
    "pipeline_config_path, model_estimator",
    [
        pytest.param(
            "tests/test_configs/logistic_model.yaml",
            LogisticRegression,
            id="logistic_model",
        ),
        # NOTE: Currently tests with no preprocessors but that should be its own case.
        pytest.param(
            "tests/test_configs/linear_model.yaml", LinearRegression, id="linear_model"
        ),
    ],
)
def test_retrieve_model_type(pipeline_config_path: str, model_estimator: BaseEstimator):
    pipeline_config = load_config(PipelineConfig, pipeline_config_path)
    assert isinstance(pipeline_config.sklearn_model, BaseEstimator)
    assert isinstance(pipeline_config.sklearn_model, model_estimator)
