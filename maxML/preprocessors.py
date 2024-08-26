import importlib
from functools import partial
from typing import Protocol

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from maxML.config_schemas import PipelineConfig


# TODO: Add FeatureUnionPreprocessor
# NOTE: May be able to consolidate them instead?


class Preprocessor(Protocol):
    @staticmethod
    def compose(
        pipeline_config: PipelineConfig,
    ) -> ColumnTransformer | FeatureUnion: ...


class ColumnTransformerPreprocessor:
    @staticmethod
    def compose(pipeline_config: PipelineConfig) -> ColumnTransformer:
        """
        Parses the pipelines dicts into their Estimators and composes a
        ColumnTransformer.
        """
        transformers = []
        # TODO: Resolve mypy error relating to config_schema preprocessing field type.
        for pipeline in pipeline_config.preprocessing.pipelines:  # type: ignore
            steps_buffer = []
            for pipe_step in pipeline["steps"]:
                module_name = ".".join(pipe_step["sklearn_module"].split(".")[:-1])
                module_obj = importlib.import_module(module_name)
                function_name = pipe_step["sklearn_module"].split(".")[-1]
                estimator_fn = getattr(module_obj, function_name)
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
        return ColumnTransformer(transformers=transformers)


PREPROCESSORS = {"ColumnTransformerPreprocessor": ColumnTransformerPreprocessor}


def get_preprocessor(pipeline_config: PipelineConfig) -> Preprocessor | None:
    """
    Return Preprocessor module as defined in preprocessor field in config.
    Validation of preprocessor handled in PipelineConfig.
    """
    if pipeline_config.preprocessing:
        preprocessor_type = pipeline_config.preprocessing.preprocessor
        return PREPROCESSORS[preprocessor_type]
    else:
        # TODO: Implement as logging, monad, or Exception.
        print("No preprocessing field found in pipeline config.")
        return None
