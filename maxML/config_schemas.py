from pathlib import Path
from typing import Any
from typing import Optional
from typing import TypeVar

import yaml
from pydantic import BaseModel
from pydantic import field_validator
from sklearn.base import BaseEstimator

from maxML.utils import get_estimator_fn


ModelType = TypeVar("ModelType", bound=BaseModel)

PREPROCESSORS = ["ColumnTransformerPreprocessor", "FeatureUnionPreprocessor"]
PIPELINES_DICT_TYPE = list[dict[Any, Any]]
PREPROCESSORS_LIST_TYPE = list[dict[str, str | PIPELINES_DICT_TYPE]]

EVALUATORS = ["LogisticEvaluator", "LinearEvaluator"]


class EvaluatorConfig(BaseModel):
    evaluator: Optional[str] = None
    metrics: Optional[list[str]] = None


class PreprocessorConfig(BaseModel):
    preprocessor: Optional[str] = None
    pipelines: Optional[PIPELINES_DICT_TYPE] = None


class PipelineConfig(BaseModel):
    """
    Pydantic schema for parsing and validating a maxML pipeline config.
    """

    sklearn_model: str
    input_path: str
    target: str
    preprocessors: list[PreprocessorConfig]
    train_test_split: dict[str, Any]  # TODO: Turn into Pydantic Schema.
    evaluators: list[EvaluatorConfig]

    @field_validator("sklearn_model", mode="after")
    @staticmethod
    def retrieve_model_type(sklearn_model: str) -> BaseEstimator:
        """
        Given that BaseEstimator is not supported by pydantic-core, convert
        the sklearn_model str to an sklearn model Estimator through pydantic
        field validation after parsing.
        """
        return get_estimator_fn(sklearn_model)()

    # TODO: Consolidate validate_preprocessor and validate_evaluator?

    @field_validator("preprocessors", mode="before")
    @staticmethod
    def validate_preprocessor(
        preprocessors: PREPROCESSORS_LIST_TYPE,
    ) -> list[PreprocessorConfig]:
        preprocessor_configs = []
        if not preprocessors:
            return [PreprocessorConfig()]
        for preprocessor in preprocessors:
            if "preprocessor" not in preprocessor.keys():
                raise KeyError("preprocessors dict must contain a preprocessor key.")
            if preprocessor["preprocessor"] not in PREPROCESSORS:
                raise KeyError(f"preprocessor must be in {PREPROCESSORS}.")
            preprocessor_configs.append(PreprocessorConfig(**preprocessor))
        return preprocessor_configs

    @field_validator("evaluators", mode="before")
    @staticmethod
    def validate_evaluator(
        evaluators: list[dict[str, str | list[str]]]
    ) -> list[EvaluatorConfig]:
        evaluator_configs = []
        if not evaluators:
            return [EvaluatorConfig()]
        for evaluator in evaluators:
            if "evaluator" not in evaluator.keys():
                raise KeyError("evaluators dict must contain an evaluator key.")
            if evaluator["evaluator"] not in EVALUATORS:
                raise KeyError(f"evaluator must be in {EVALUATORS}.")
            evaluator_configs.append(EvaluatorConfig(**evaluator))
        return evaluator_configs


def load_config(config_class: type[ModelType], model_config_path: str) -> ModelType:
    """
    Load yaml config, parse and validate with config_class schema.
    """
    # NOTE: Using FullLoader to automatically parse python object pyyaml tags e.g.
    # !!python/name:numpy.nan
    # TODO: Replace the use of FullLoader with a custom serialization layer.
    return config_class(
        **yaml.load(open(Path.cwd() / model_config_path), Loader=yaml.FullLoader)
    )
