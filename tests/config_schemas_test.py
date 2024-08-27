import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from maxML.config_schemas import load_config
from maxML.config_schemas import PipelineConfig


@pytest.mark.parametrize(
    "pipeline_config_path, model_estimator",
    [
        pytest.param(
            "tests/test_configs/columntransformer_logistic.yaml",
            LogisticRegression,
            id="logistic_model",
        ),
        pytest.param(
            "tests/test_configs/no_preprocessors.yaml",
            LogisticRegression,
            id="no_preprocessors",
        ),
    ],
)
def test_retrieve_model_type(pipeline_config_path: str, model_estimator: BaseEstimator):
    """
    Assert that the PipelineConfig is able to retrieve the defined Model Estimator.
    """
    pipeline_config = load_config(PipelineConfig, pipeline_config_path)
    assert isinstance(pipeline_config.sklearn_model, BaseEstimator)
    assert isinstance(pipeline_config.sklearn_model, model_estimator)
