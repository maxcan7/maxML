import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion

from maxML.config_schemas import load_config
from maxML.config_schemas import PipelineConfig
from maxML.preprocessors import ColumnTransformerPreprocessor
from maxML.preprocessors import compose_preprocessor
from maxML.preprocessors import do_preprocessing
from maxML.preprocessors import FeatureUnionPreprocessor
from maxML.preprocessors import get_preprocessor_fn
from maxML.preprocessors import Preprocessor
from tests.conftest import columntransformer_preprocessor_fixture
from tests.conftest import featureunion_preprocessor_fixture


@pytest.mark.parametrize(
    "pipeline_config_path, expected_preprocessors",
    [
        pytest.param(
            "tests/test_configs/columntransformer_logistic.yaml",
            [ColumnTransformerPreprocessor],
            id="columntransformer_preprocessor",
        ),
        pytest.param(
            "tests/test_configs/two_preprocessors_logistic.yaml",
            [ColumnTransformerPreprocessor, FeatureUnionPreprocessor],
            id="two_preprocessors",
        ),
    ],
)
def test_get_preprocessor_fn(
    pipeline_config_path: str, expected_preprocessors: list[Preprocessor]
):
    """
    Test that the get_preprocessor function can load the correct Preprocessors given the
    PipelineConfig params.
    """
    pipeline_config: PipelineConfig = load_config(pipeline_config_path)
    for i in range(len(pipeline_config.preprocessors)):
        preprocessor = get_preprocessor_fn(pipeline_config.preprocessors[i])
        assert preprocessor == expected_preprocessors[i]


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
    pipeline_config: PipelineConfig = load_config(pipeline_config_path)
    assert not do_preprocessing(pipeline_config.preprocessors)


@pytest.mark.parametrize(
    "pipeline_config_path",
    [
        pytest.param(
            "tests/test_configs/no_preprocessors.yaml",
            id="no_preprocessors",
        ),
    ],
)
def test_get_preprocessor_fn_invalid(pipeline_config_path: str):
    """
    Test that the get_preprocessor function will return a KeyError if the preprocessor
    or pipelines fields are None.
    """
    pipeline_config: PipelineConfig = load_config(pipeline_config_path)
    with pytest.raises(KeyError):
        get_preprocessor_fn(pipeline_config.preprocessors[0])


@pytest.mark.parametrize(
    "pipeline_config_path, preprocessor_fixture",
    [
        pytest.param(
            "tests/test_configs/columntransformer_logistic.yaml",
            columntransformer_preprocessor_fixture(),
            id="columntransformer_preprocessor",
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
    pipeline_config: PipelineConfig = load_config(pipeline_config_path)
    preprocessor = ColumnTransformerPreprocessor.compose(
        pipeline_config.preprocessors[0]
    )
    # TODO: Figure out a way to compare these two objects effectively.
    assert preprocessor
    assert preprocessor_fixture


@pytest.mark.parametrize(
    "pipeline_config_path, preprocessor_fixture",
    [
        pytest.param(
            "tests/test_configs/two_preprocessors_logistic.yaml",
            featureunion_preprocessor_fixture(),
            id="featureunion_preprocessor",
        ),
    ],
)
def test_FeatureUnionPreprocessor_compose(
    pipeline_config_path: str, preprocessor_fixture: FeatureUnion
):
    """
    Test that the FeatureUnionPreprocessor can compose the PipelineConfig into a
    FeatureUnion.
    """
    pipeline_config: PipelineConfig = load_config(pipeline_config_path)
    preprocessor = FeatureUnionPreprocessor.compose(pipeline_config.preprocessors[1])
    # TODO: Figure out a way to compare these two objects effectively.
    assert preprocessor
    assert preprocessor_fixture


@pytest.mark.parametrize(
    "pipeline_config_path, preprocessor_fixture",
    [
        pytest.param(
            "tests/test_configs/two_preprocessors_logistic.yaml",
            columntransformer_preprocessor_fixture(),
            id="columntransformer_preprocessor",
        ),
        # TODO: Test both composing ColumnTransformer into FeatureUnion and reverse.
        # pytest.param(
        #     "tests/test_configs/two_preprocessors_logistic.yaml",
        #     featureunion_preprocessor_fixture(),
        #     id="featureunion_preprocessor",
        # ),
    ],
)
def test_compose_preprocessor(
    pipeline_config_path: str, preprocessor_fixture: FeatureUnion
):
    """
    Test that the compose_preprocessor function can properly compose multiple
    preprocessors into one.
    """
    pipeline_configs: PipelineConfig = load_config(pipeline_config_path)
    preprocessor = compose_preprocessor(pipeline_configs.preprocessors)
    # TODO: Figure out a way to compare these two objects effectively.
    assert preprocessor
    assert preprocessor_fixture
