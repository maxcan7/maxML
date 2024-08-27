from functools import partial
from typing import Protocol

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from maxML.config_schemas import PreprocessorConfig
from maxML.utils import get_estimator_fn


def _compose(
    preprocessor_config: PreprocessorConfig,
) -> list[tuple[str, BaseEstimator, list[str]]]:
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


class Preprocessor(Protocol):
    @staticmethod
    def compose(
        preprocessor_config: PreprocessorConfig,
    ) -> ColumnTransformer | FeatureUnion: ...


class ColumnTransformerPreprocessor:
    @staticmethod
    def compose(preprocessor_config: PreprocessorConfig) -> ColumnTransformer:
        """
        Parses the pipelines dicts into their Estimators and composes a
        ColumnTransformer.
        """
        transformers = _compose(preprocessor_config)
        return ColumnTransformer(transformers=transformers)


class FeatureUnionPreprocessor:
    @staticmethod
    def compose(preprocessor_config: PreprocessorConfig) -> FeatureUnion:
        """
        Parses the pipelines dicts into their Estimators and composes a
        FeatureUnion.
        """
        transformers = _compose(preprocessor_config)
        return FeatureUnion(transformer_list=transformers)


PREPROCESSORS = {
    "ColumnTransformerPreprocessor": ColumnTransformerPreprocessor,
    "FeatureUnionPreprocessor": FeatureUnionPreprocessor,
}


def do_preprocessing(preprocessor_config: PreprocessorConfig) -> bool:
    """
    If both preprocessor and pipelines fields are None, then returns False.
    This is used to determine whether or not to do preprocessing.
    If only one is empty, it will raise a KeyError on get_preprocessor.
    """
    return any([preprocessor_config.preprocessor, preprocessor_config.pipelines])


def get_preprocessor(preprocessor_config: PreprocessorConfig) -> Preprocessor:
    """
    Return Preprocessor module as defined in preprocessor field in config.
    """
    if not preprocessor_config.preprocessor:
        raise KeyError("preprocessor_config is missing preprocessor field.")
    if not preprocessor_config.pipelines:
        raise KeyError("preprocessor_config is missing pipelines field.")
    preprocessor_type = preprocessor_config.preprocessor
    return PREPROCESSORS[preprocessor_type]  # type: ignore
