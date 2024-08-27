import pytest
from sklearn.compose import ColumnTransformer

from maxML.config_schemas import load_config
from maxML.config_schemas import PipelineConfig
from maxML.preprocessors import ColumnTransformerPreprocessor
from maxML.preprocessors import get_preprocessor
from maxML.preprocessors import Preprocessor
from tests.conftest import logistic_preprocessor_fixture


@pytest.mark.parametrize(
    "pipeline_config_path, expected_preprocessor",
    [
        pytest.param(
            "tests/test_configs/logistic_model.yaml",
            ColumnTransformerPreprocessor,
            id="column_transformer_preprocessor",
        ),
    ],
)
def test_get_preprocessor(
    pipeline_config_path: str, expected_preprocessor: Preprocessor
):
    """
    Test that the get_preprocessor function can load the correct Preprocessor given the
    PipelineConfig params.
    """
    pipeline_config = load_config(PipelineConfig, pipeline_config_path)
    preprocessor = get_preprocessor(pipeline_config)
    assert preprocessor == expected_preprocessor


@pytest.mark.parametrize(
    "pipeline_config_path, preprocessor_fixture",
    [
        pytest.param(
            "tests/test_configs/logistic_model.yaml",
            logistic_preprocessor_fixture(),
            id="logistic_model",
        ),
    ],
)
def test_ColumnTransformerPreprocessor_compose(
    pipeline_config_path: str, preprocessor_fixture: ColumnTransformer
):
    """
    Test that the ColumnTransformerPreprocessor can compose the PipelineConfig into a
    ColumnTransformer.
    """
    pipeline_config = load_config(PipelineConfig, pipeline_config_path)
    preprocessor = ColumnTransformerPreprocessor.compose(pipeline_config)
    # TODO: Figure out a way to compare these two objects effectively.
    assert preprocessor
    assert preprocessor_fixture
