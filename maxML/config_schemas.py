from pathlib import Path
from typing import Any
from typing import Optional
from typing import TypeVar

import yaml
from pydantic import BaseModel
from pydantic import field_validator
from pydantic.fields import FieldInfo
from sklearn.base import BaseEstimator

from maxML.utils import get_estimator_fn


ModelType = TypeVar("ModelType", bound=BaseModel)

PREPROCESSORS = ["ColumnTransformerPreprocessor", "FeatureUnionPreprocessor"]
PIPELINES_DICT_TYPE = list[dict[Any, Any]]
PREPROCESSORS_LIST_TYPE = list[dict[str, str | PIPELINES_DICT_TYPE]]

EVALUATORS = ["LogisticEvaluator", "LinearEvaluator"]


class EvaluatorConfig(BaseModel):
    type: Optional[str] = None
    metrics: Optional[list[str]] = None


class PreprocessorConfig(BaseModel):
    type: Optional[str] = None
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

    @field_validator("preprocessors", "evaluators", mode="before")
    @staticmethod
    def validate_config_list(
        config_list: list[dict[str, Any]], field: FieldInfo
    ) -> list[BaseModel]:
        """
        Generic validator for lists of configuration dictionaries
        (preprocessors or evaluators).
        """
        valid_options = (
            PREPROCESSORS
            if field.field_name == "preprocessors"  # type: ignore
            else EVALUATORS
        )
        config_class = (
            PreprocessorConfig
            if field.field_name == "preprocessors"  # type: ignore
            else EvaluatorConfig
        )

        validated_configs = []
        if not config_list:
            return [config_class()]

        for config in config_list:
            if "type" not in config:
                raise KeyError(
                    f"{field.field_name} dict must contain a type key."  # type: ignore
                )
            if config["type"] not in valid_options:
                raise KeyError(
                    f"Invalid type: {field.field_name}"  # type: ignore
                    f"Must be one of {valid_options}"
                )
            validated_configs.append(config_class(**config))
        return validated_configs


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
