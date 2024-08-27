from pathlib import Path
from typing import Any
from typing import TypeVar

import yaml
from pydantic import BaseModel
from pydantic import field_validator
from sklearn.base import BaseEstimator

from maxML.utils import get_estimator_fn


ModelType = TypeVar("ModelType", bound=BaseModel)
PREPROCESSORS = ["ColumnTransformerPreprocessor"]


class PreprocessingConfig(BaseModel):
    preprocessor: str
    pipelines: list[dict[Any, Any]]


class PipelineConfig(BaseModel):
    """
    Pydantic schema for parsing and validating a maxML pipeline config.
    """

    sklearn_model: str
    input_path: str
    target: str
    preprocessing: PreprocessingConfig | None = None
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

    @field_validator("preprocessing", mode="before")
    @staticmethod
    def validate_preprocessor(
        preprocessing: dict[str, str | list[dict[Any, Any]]] | None,
    ) -> PreprocessingConfig | None:
        if not preprocessing:
            return None
        if "preprocessor" not in preprocessing.keys():
            raise KeyError("preprocessing dict must contain a preprocessor key.")
        if preprocessing["preprocessor"] not in PREPROCESSORS:
            raise KeyError(f"preprocessor must be in {PREPROCESSORS}.")
        return PreprocessingConfig(**preprocessing)


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
