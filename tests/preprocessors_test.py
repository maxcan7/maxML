import pytest
from sklearn.compose import ColumnTransformer

from maxML.config_schemas import load_config
from maxML.config_schemas import PipelineConfig
from maxML.preprocessors import ColumnTransformerPreprocessor
from maxML.preprocessors import do_preprocessing
from maxML.preprocessors import get_preprocessor
from maxML.preprocessors import Preprocessor
from tests.conftest import columntransformer_preprocessor_fixture


@pytest.mark.parametrize(
    "pipeline_config_path, expected_preprocessor",
    [
        pytest.param(
            "tests/test_configs/columntransformer_logistic.yaml",
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
    preprocessor = get_preprocessor(pipeline_config.preprocessors[0])
    assert preprocessor == expected_preprocessor


@pytest.mark.parametrize(
    "pipeline_config_path",
    [
        pytest.param(
            "tests/test_configs/no_preprocessors.yaml",
            id="no_preprocessors",
        ),
    ],
)
def test_do_preprocessing(pipeline_config_path: str):
    """
    Test that if both the PreprocessorConfig preprocessor and pipelines fields are None,
    do_preprocessing returns False.
    """
    pipeline_config = load_config(PipelineConfig, pipeline_config_path)
    assert not do_preprocessing(pipeline_config.preprocessors[0])


@pytest.mark.parametrize(
    "pipeline_config_path",
    [
        pytest.param(
            "tests/test_configs/no_preprocessors.yaml",
            id="no_preprocessors",
        ),
    ],
)
def test_get_preprocessor_invalid(pipeline_config_path: str):
    """
    Test that the get_preprocessor function will return a KeyError if the preprocessor
    or pipelines fields are None.
    """
    pipeline_config = load_config(PipelineConfig, pipeline_config_path)
    with pytest.raises(KeyError):
        get_preprocessor(pipeline_config.preprocessors[0])


@pytest.mark.parametrize(
    "pipeline_config_path, preprocessor_fixture",
    [
        pytest.param(
            "tests/test_configs/columntransformer_logistic.yaml",
            columntransformer_preprocessor_fixture(),
            id="column_transformer_preprocessor",
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
    preprocessor = ColumnTransformerPreprocessor.compose(
        pipeline_config.preprocessors[0]
    )
    # TODO: Figure out a way to compare these two objects effectively.
    assert preprocessor
    assert preprocessor_fixture
