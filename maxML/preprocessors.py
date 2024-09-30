from functools import partial
from typing import Optional
from typing import Protocol

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from maxML.config_schemas import PreprocessorConfig
from maxML.utils import get_estimator_fn


class Preprocessor(Protocol):
    @staticmethod
    def compose(
        preprocessor_config: PreprocessorConfig,
        preprocessor: Optional[ColumnTransformer | FeatureUnion] = None,
    ) -> ColumnTransformer | FeatureUnion:
        """
        Protocol defines the interface for all Preprocessors.
        Inputs:
        preprocessor_config: Config for composing a Preprocessor.
        preprocessor: Another ColumnTransformer or FeatureUnion that can be added as a
                      transformer to the next preprocessor such that all maxML
                      Pipelines can expect a single preprocessor as the composition of
                      all preprocessors.
        """
        ...


class ColumnTransformerPreprocessor:
    @staticmethod
    def compose(
        preprocessor_config: PreprocessorConfig,
        preprocessor: Optional[Preprocessor] = None,
    ) -> ColumnTransformer:
        """
        Parses the pipelines dicts into their Estimators and composes a
        ColumnTransformer.
        """
        transformers = _compose(preprocessor_config)
        if preprocessor:
            transformers.insert(0, ("composed_preprocessor", preprocessor))
        return ColumnTransformer(transformers=transformers)


class FeatureUnionPreprocessor:
    @staticmethod
    def compose(
        preprocessor_config: PreprocessorConfig,
        preprocessor: Optional[Preprocessor] = None,
    ) -> FeatureUnion:
        """
        Parses the pipelines dicts into their Estimators and composes a
        FeatureUnion.
        """
        transformers = _compose(preprocessor_config)
        if preprocessor:
            transformers.insert(0, ("composed_preprocessor", preprocessor))
        return FeatureUnion(transformer_list=transformers)


PREPROCESSORS = {
    "ColumnTransformerPreprocessor": ColumnTransformerPreprocessor,
    "FeatureUnionPreprocessor": FeatureUnionPreprocessor,
}


def _compose(
    preprocessor_config: PreprocessorConfig,
) -> list[tuple]:
    """
    Parses the pipelines dicts into their Estimators and returns the list of
    transformer tuples.
    """
    transformers = []
    for pipeline in preprocessor_config.pipelines:  # type: ignore
        steps_buffer = []
        for pipe_step in pipeline["steps"]:
            estimator_fn = get_estimator_fn(pipe_step["sklearn_module"])
            if "args" in pipe_step.keys():
                estimator_fn = partial(estimator_fn, **pipe_step["args"])
            estimator = (pipe_step["name"], estimator_fn())
            steps_buffer.append(estimator)
        transformer = (
            pipeline["name"],
            Pipeline(steps_buffer),
            pipeline["columns"],
        )
        transformers.append(transformer)
    return transformers


def get_preprocessor_fn(preprocessor_config: PreprocessorConfig) -> Preprocessor:
    """
    Return Preprocessor module as defined in preprocessor field in config.
    """
    if not preprocessor_config.preprocessor:
        raise KeyError("preprocessor_config is missing preprocessor field.")
    if not preprocessor_config.pipelines:
        raise KeyError("preprocessor_config is missing pipelines field.")
    preprocessor_type = preprocessor_config.preprocessor
    return PREPROCESSORS[preprocessor_type]  # type: ignore


def compose_preprocessor(
    preprocessor_configs: list[PreprocessorConfig],
) -> ColumnTransformer | FeatureUnion:
    """
    Loops over the list of preprocessor_configs, retrieves the appriate Preprocessor,
    and composes the preprocessor instance i.e. ColumnTransformer or FeatureUnion.

    If there are multiple preprocessors, each preceding preprocessor is composed into
    the next, such that a single preprocessor as the composition of all preprocessors
    is returned.
    """
    preprocessors: list = []
    for i, preprocessor_config in enumerate(preprocessor_configs):
        preprocessor_fn = get_preprocessor_fn(preprocessor_config)
        compose_method = preprocessor_fn.compose
        if preprocessors:
            compose_method = partial(compose_method, preprocessor=preprocessors[i - 1])
        preprocessors.append(compose_method(preprocessor_config=preprocessor_config))
    return preprocessors[-1]


def do_preprocessing(
    preprocessor_configs: list[PreprocessorConfig],
) -> bool:
    """
    If both preprocessor and pipelines fields for the first PreprocessorConfig are None,
    then returns False.
    This is used to determine whether or not to do preprocessing.
    If only one is empty, it will raise a KeyError on get_preprocessor.
    """
    return any(
        [preprocessor_configs[0].preprocessor, preprocessor_configs[0].pipelines]
    )
