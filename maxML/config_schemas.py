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
    metrics: list[str]

    @field_validator("sklearn_model", mode="after")
    @staticmethod
    def retrieve_model_type(sklearn_model: str) -> BaseEstimator:
        """
        Given that BaseEstimator is not supported by pydantic-core, convert
        the sklearn_model str to an sklearn model Estimator through pydantic
        field validation after parsing.
        """
        return get_estimator_fn(sklearn_model)()

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


def load_config(config_class: type[ModelType], model_config_path: str) -> ModelType:
    """
    Load yaml config, parse and validate with config_class schema.
    """
    # NOTE: Using FullLoader to automatically parse python object pyyaml tags e.g.
    # !!python/name:numpy.nan
    # TODO: Is there a safer yet equally flexible way to do this?
    return config_class(
        **yaml.load(open(Path.cwd() / model_config_path), Loader=yaml.FullLoader)
    )
