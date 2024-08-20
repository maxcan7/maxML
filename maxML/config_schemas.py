import importlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic import field_validator
from sklearn.base import BaseEstimator


class PipelineConfig(BaseModel):
    """
    Pydantic schema for parsing and validating a maxML pipeline config.
    """

    sklearn_model: str
    input_path: str
    target: str
    # TODO: Break out preprocessing into its own pydantic schema.
    preprocessing: dict[str, list[dict[Any, Any]]] | None = None
    metrics: list[str]

    @field_validator("sklearn_model", mode="after")
    def retrieve_model_type(sklearn_model: str) -> BaseEstimator:
        """
        Given that BaseEstimator is not supported by pydantic-core, convert
        the sklearn_model str to an sklearn model Estimator through pydantic
        field validation after parsing.
        """
        module_name = ".".join(sklearn_model.split(".")[:-1])
        module_obj = importlib.import_module(module_name)
        function_name = sklearn_model.split(".")[-1]
        return getattr(module_obj, function_name)()


def load_config(config_class: BaseModel, model_config_path: str) -> BaseModel:
    """
    Load yaml config, parse and validate with config_class schema.
    """
    return config_class(**yaml.safe_load(open(Path.cwd() / model_config_path)))
