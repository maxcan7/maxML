import pytest
from sklearn.compose import ColumnTransformer

from maxML.config_schemas import load_config
from maxML.config_schemas import PipelineConfig
from maxML.preprocessors import ColumnTransformerPreprocessor
from tests.conftest import logistic_preprocessor_fixture


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
def test_ColumnTransformerPreprocessor(
    pipeline_config_path: str, preprocessor_fixture: ColumnTransformer
):
    pipeline_config = load_config(PipelineConfig, pipeline_config_path)
    preprocessor = ColumnTransformerPreprocessor.compose(pipeline_config)
    # TODO: Figure out a way to compare these two objects effectively.
    assert preprocessor
    assert preprocessor_fixture
