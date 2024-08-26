import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from maxML.config_schemas import load_config
from maxML.config_schemas import PipelineConfig


@pytest.mark.parametrize(
    "pipeline_config_path, model_estimator",
    [
        pytest.param(
            "tests/test_configs/logistic_model.yaml",
            LogisticRegression,
            id="logistic_model",
        ),
        pytest.param(
            "tests/test_configs/linear_model.yaml", LinearRegression, id="linear_model"
        ),
        pytest.param(
            "tests/test_configs/no_preprocessing.yaml",
            LogisticRegression,
            id="no_preprocessing",
        ),
    ],
)
def test_retrieve_model_type(pipeline_config_path: str, model_estimator: BaseEstimator):
    """Assert that the PipelineConfig is able to retrieve the defined Model Estimator."""
    pipeline_config = load_config(PipelineConfig, pipeline_config_path)
    assert isinstance(pipeline_config.sklearn_model, BaseEstimator)
    assert isinstance(pipeline_config.sklearn_model, model_estimator)


@pytest.mark.parametrize(
    "pipeline_config_path, error_msg",
    [
        pytest.param(
            "tests/test_configs/no_preprocessor.yaml",
            "preprocessing dict must contain a preprocessor key.",
            id="no_preprocessor",
        ),
        pytest.param(
            "tests/test_configs/invalid_preprocessor.yaml",
            "preprocessor must be in ['ColumnTransformerPreprocessor'].",
            id="invalid_preprocessor",
        ),
    ],
)
def test_validate_preprocessor_invalid(pipeline_config_path: str, error_msg: str):
    """Test that when the PipelineConfig has a preprocessing section but is missing or has an invalid preprocessor key, that it raises a KeyError with the appropriate error message."""
    with pytest.raises(KeyError) as e:
        load_config(PipelineConfig, pipeline_config_path)
    assert e.value.args[0] == error_msg
