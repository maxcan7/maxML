import pytest
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from maxML.config_schemas import PipelineConfig
from maxML.config_schemas import load_config
from maxML.preprocessors import ColumnTransformerPreprocessor
from maxML.preprocessors import Preprocessor
from maxML.preprocessors import compose_preprocessor
from maxML.preprocessors import do_preprocessing
from maxML.preprocessors import get_preprocessor_fn
from tests.conftest import columntransformer_fixture
from tests.conftest import columntransformer_poly_fixture


@pytest.mark.parametrize(
    "pipeline_config_path, expected_preprocessors",
    [
        pytest.param(
            "tests/test_configs/columntransformer_logistic.yaml",
            [ColumnTransformerPreprocessor],
            id="columntransformer_preprocessor",
        ),
        pytest.param(
            "tests/test_configs/columntransformer_poly_logistic.yaml",
            [ColumnTransformerPreprocessor],
            id="columntransformer_poly_preprocessor",
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


def compare_estimators(
    estimator1: BaseEstimator,
    estimator2: BaseEstimator,
) -> bool:
    """
    Compare two estimators, including their parameters.
    """
    if not isinstance(estimator1, type(estimator2)):
        return False
    if not hasattr(estimator1, "get_params") or not hasattr(estimator2, "get_params"):
        # If estimators don't have get_params, just compare them directly
        return estimator1 == estimator2
    params1 = estimator1.get_params()
    params2 = estimator2.get_params()
    if params1 != params2:
        return False
    return True


def compare_pipelines(pipeline1: Pipeline, pipeline2: Pipeline) -> bool:
    """
    Compare two pipelines, including their steps and the parameters
    of the estimators within each step.
    """
    if not isinstance(pipeline1, Pipeline) or not isinstance(pipeline2, Pipeline):
        return False
    if len(pipeline1.steps) != len(pipeline2.steps):
        return False
    for (name1, step1), (name2, step2) in zip(pipeline1.steps, pipeline2.steps):
        if name1 != name2:
            return False
        if not isinstance(step1, type(step2)):
            return False
        if not compare_estimators(step1, step2):
            return False
    return True


def compare_transformers(
    transformer1: ColumnTransformer,
    transformer2: ColumnTransformer,
) -> bool:
    """
    Compare two ColumnTransformers, including their transformers and the
    pipelines within each transformer.
    """
    if not isinstance(transformer1, ColumnTransformer) or not isinstance(
        transformer2, ColumnTransformer
    ):
        return False

    if len(transformer1.transformers) != len(transformer2.transformers):
        return False

    for t1, t2 in zip(transformer1.transformers, transformer2.transformers):
        if not all(
            [
                t1[0] == t2[0],  # name
                compare_pipelines(t1[1], t2[1]),  # pipeline
                t1[2] == t2[2],  # columns
            ]
        ):
            return False
    return True


@pytest.mark.parametrize(
    "pipeline_config_path, transformer_fixture",
    [
        pytest.param(
            "tests/test_configs/columntransformer_logistic.yaml",
            columntransformer_fixture(),
            id="columntransformer_preprocessor",
        ),
        pytest.param(
            "tests/test_configs/columntransformer_poly_logistic.yaml",
            columntransformer_poly_fixture(),
            id="columntransformer_poly_preprocessor",
        ),
    ],
)
def test_ColumnTransformerPreprocessor_compose(
    pipeline_config_path: str, transformer_fixture: ColumnTransformer
):
    """
    Test that the ColumnTransformerPreprocessor can compose the PipelineConfig into a
    ColumnTransformer.
    """
    pipeline_config: PipelineConfig = load_config(pipeline_config_path)
    preprocessor = ColumnTransformerPreprocessor.compose(
        pipeline_config.preprocessors[0]
    )
    assert isinstance(preprocessor, ColumnTransformer)
    assert compare_transformers(preprocessor, transformer_fixture)


@pytest.mark.parametrize(
    "pipeline_config_path, transformer_fixture",
    [
        pytest.param(
            "tests/test_configs/columntransformer_logistic.yaml",
            columntransformer_fixture(),
            id="columntransformer_preprocessor",
        ),
    ],
)
def test_compose_preprocessor(
    pipeline_config_path: str, transformer_fixture: ColumnTransformer
):
    """
    Test that the compose_preprocessor function can properly compose a
    ColumnTransformerPreprocessor from config.
    """
    pipeline_configs: PipelineConfig = load_config(pipeline_config_path)
    preprocessor = compose_preprocessor(pipeline_configs.preprocessors)
    compare_transformers(preprocessor, transformer_fixture)
